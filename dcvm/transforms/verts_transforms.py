import numpy as np
import torch

from dcvm.transforms.utils_transforms import DimensionConverter

def interpolate_rescale_field_torch(displacement_field_tuple, verts_list, img_size=[64,64,64]):
    '''
    displacement_field_tuple: tuple of torch.tensors (n_batch, n_dim, h, w, d)
    verts_list: list of numpy.ndarray (n_verts, n_dim) or list of torch.tensor (n_batch, n_verts, n_dim)
    '''
    new_field_list = []
        
    for idx, verts in enumerate(verts_list):
        if len(verts_list) == len(displacement_field_tuple):
            field = displacement_field_tuple[idx].clone()
        elif len(displacement_field_tuple) == 1:
            field = displacement_field_tuple[0].clone()
        else:
            raise ValueError('displacement_field_tuple should have same length as verts_list or have length 1')
        
        # to convert from [-1,1] to [0,img.shape[dim_idx]-1] ---- don't need to do this for field, b/c [-1,1] is for affine_grid
        # comment this later b/c we wanna be using consistent field magnitude from model prediction --- having second thoughts about this
        for dim_idx in range(field.shape[1]):
            field[:,dim_idx,:,:,:] = field[:,dim_idx,:,:,:]*(img_size[dim_idx]-1)/2

        n_batch = field.shape[0]

        if not torch.is_tensor(verts):
            verts = torch.tensor(verts, dtype=torch.get_default_dtype(), device=field.device).unsqueeze(0)
        verts_dim_converted_dim_idx = []
        for dim_idx, img_size_indiv in enumerate(img_size):
            dim_convert = DimensionConverter(img_size_indiv)
            verts_dim_converted_dim_idx.append(dim_convert.from_dim_size(verts[:,:,dim_idx]).unsqueeze(2))
        verts_dim_converted = torch.cat(verts_dim_converted_dim_idx, dim=2) # (n_batch, n_pts, 3)

        verts_dim_converted = verts_dim_converted.unsqueeze(1).unsqueeze(1) # (n_batch, 1, 1, n_pts, 3) (required for input to grid_sample)
        # verts_dim_converted = torch.flip(verts_dim_converted, [4])
        verts_dim_converted_rearrange = torch.cat([verts_dim_converted[:,:,:,:,2].unsqueeze(4),
                                                   verts_dim_converted[:,:,:,:,1].unsqueeze(4),
                                                   verts_dim_converted[:,:,:,:,0].unsqueeze(4)], dim=4)

        # field = torch.flip(field, [1])
        field_rearrange = torch.cat([field[:,2,:,:,:].unsqueeze(1),
                                     field[:,1,:,:,:].unsqueeze(1),
                                     field[:,0,:,:,:].unsqueeze(1)], dim=1)
        new_field = torch.nn.functional.grid_sample(field_rearrange, verts_dim_converted_rearrange, align_corners=True).permute(0,2,3,4,1).squeeze(1).squeeze(1) # (n_batch, 3, 1, 1, n_pts) to (n_batch, n_pts, 3)
        
        new_field_list.append(new_field)
    
    return new_field_list

def move_verts_with_field(verts_list, interp_field_list, convert_to_np=False):
    new_verts_list = []
    for idx, verts in enumerate(verts_list):
        if len(verts_list) == len(interp_field_list):
            interp_field = interp_field_list[idx]
        elif len(interp_field_list) == 1:
            interp_field = interp_field_list[0]
        
        if convert_to_np:
            new_verts_list.append(verts.cpu().numpy() + interp_field.cpu().numpy())
        else:
            new_verts_list.append(verts + interp_field)
        
    return new_verts_list

def apply_linear_transform_on_verts(verts, src_to_tgt_transformation):
    '''
    verts: np.array, n_pts x 3
    '''
    verts_homo_coords = np.concatenate([verts.T, np.ones([1, verts.shape[0]])], axis=0)
    new_verts = np.dot(src_to_tgt_transformation, verts_homo_coords)[:3].T
    return new_verts

def apply_linear_transform_on_verts_torch(verts, src_to_tgt_transformation):
    '''
    verts: torch.tensor, n_batch x n_pts x 3
    '''
    verts_homo_coords = torch.cat([verts, torch.ones([1, verts.shape[1], 1], device=verts.device)], axis=2)
    new_verts = torch.matmul(src_to_tgt_transformation, verts_homo_coords.permute(0,2,1)).permute(0,2,1)[:,:,:3]
    return new_verts

def interpolate_voxelgrid_at_verts_torch(voxelgrid, verts, dims=None, spacing=[1,1,1], origin=[0,0,0], grid_sample_mode='bilinear'):
    '''
    voxelgrid: torch.tensor (n_batch, n_channels, h, w, d)
    verts: torch.tensor (n_batch, n_verts, 3)
    '''
    if dims is None:
        dims = voxelgrid.shape[-3:]
        
    verts_dim_converted_dim_idx = []
    for idx, (dims_entry, spacing_entry, origin_entry) in enumerate(zip(dims, spacing, origin)):
        ''' manually adjust offset if align_corners=False.. but seems like no offset is fine for align_corners=True '''
        # if np.allclose(np.array(origin), np.zeros(3)) and np.allclose(np.array(spacing), np.ones(3)):
        #     offset = -0.5 + 0.5*spacing_entry # align center of bottom-left corner pixel at (0,0), and assume original spacing=[1,1,1] and origin is [0,0,0]
        # else:
        #     offset = 0
        # dim_convert = DimensionConverter(dims_entry, offset=offset)
        dim_convert = DimensionConverter(dims_entry)
        verts_dim_converted_dim_idx.append(dim_convert.from_dim_size(verts[:,:,idx]/spacing_entry - origin_entry/spacing_entry).unsqueeze(2))
    verts_dim_converted = torch.cat(verts_dim_converted_dim_idx, dim=2) # (n_batch, n_pts, 3)
    
    verts_dim_converted = verts_dim_converted[:,None,None,:,:] # (n_batch, 1, 1, n_pts, 3) (required for input to grid_sample)
    interp_vals = torch.nn.functional.grid_sample(voxelgrid.permute(0,1,4,3,2), verts_dim_converted, align_corners=True, padding_mode="border", mode=grid_sample_mode).permute(0,2,3,4,1).squeeze(1).squeeze(1) # (n_batch, 3, 1, 1, n_pts) to (n_batch, n_pts, 3)
    return interp_vals

def get_verts_transformed_single_displacement_field(displacement_field_tuple, verts_template, img_size):
    interp_field_list = interpolate_rescale_field_torch(displacement_field_tuple, [verts_template], img_size=img_size)
    transformed_verts_list = move_verts_with_field([verts_template], interp_field_list)
    return transformed_verts_list[0]