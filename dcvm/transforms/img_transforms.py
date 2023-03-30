import numpy as np
import torch

from dcvm.transforms.utils_transforms import DimensionConverterDP

def ct_normalize(pixel_array, min_bound=-158.0, max_bound=864.0):
    pix_normalized = (pixel_array - min_bound) / (max_bound - min_bound)
    pix_normalized[pix_normalized>1] = 1.
    pix_normalized[pix_normalized<0] = 0.
    return pix_normalized

def apply_linear_transform_on_img_torch(src_data, src_to_tgt_transformation, tgt_shape, grid_sample_mode='bilinear'):
    '''
    decide which source voxels to sample, starting from target voxel coordinates
    src_data = img_cuda[None,None,:,:,:]
    src_to_tgt_transformation: homogeneous coordinate frame in 3D space, shape (4,4)
    tgt_shape = [128,128,128]
    '''
    x, y, z = torch.arange(tgt_shape[0]), torch.arange(tgt_shape[1]), torch.arange(tgt_shape[2])
    xmesh, ymesh, zmesh = torch.meshgrid(x, y, z, indexing='ij') # (x.shape, y.shape, z.shape)
    coords_stack = torch.stack([xmesh, ymesh, zmesh], dim=0).reshape(3,-1) # (3, np.prod(tgt_shape))
    homo_coords = torch.cat([coords_stack, torch.ones([1,coords_stack.shape[1]])], dim=0).to(dtype=torch.get_default_dtype(), device=src_data.device) # (4, np.prod(tgt_shape))

    dim_conv = DimensionConverterDP(src_data.shape[-3:])
    with torch.no_grad():
        sample_coords = torch.matmul(torch.linalg.inv(src_to_tgt_transformation), homo_coords)[:3].T # (np.prod(tgt_shape), 3)
        grid = dim_conv.from_orig_img_dim(sample_coords).reshape(*tgt_shape,-1)[None,:,:,:,:].flip([-1]) # (1, *tgt_shape, 3)
        new_img = torch.nn.functional.grid_sample(src_data, grid, align_corners=True, mode=grid_sample_mode) # (1, 1, *tgt_shape)
    return new_img