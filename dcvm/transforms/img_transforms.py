import numpy as np
import torch

from dcvm.transforms.utils_transforms import DimensionConverterDP

def ct_normalize(pixel_array, min_bound=-158.0, max_bound=864.0):
    pix_normalized = (pixel_array - min_bound) / (max_bound - min_bound)
    pix_normalized[pix_normalized>1] = 1.
    pix_normalized[pix_normalized<0] = 0.
    
    return pix_normalized

def apply_linear_transform_on_img_torch(src_data, src_to_tgt_transformation, tgt_shape, interpn_method='linear'):
    # decide which source voxels to sample, starting from target voxel coordinates
    # src_data = img_cuda[None,None,:,:,:]
    # src_to_tgt_transformation = torch.linalg.inv(transformation_cuda)
    # tgt_shape = [128,128,128]

    x, y, z = np.arange(tgt_shape[0]), np.arange(tgt_shape[1]), np.arange(tgt_shape[2])
    xmesh, ymesh, zmesh = np.meshgrid(x, y, z, indexing='ij')
    coords_stack = np.stack([xmesh, ymesh, zmesh], axis=0).reshape(3,-1)
    homo_coords = np.concatenate([coords_stack, np.ones([1,coords_stack.shape[1]])], axis=0)
    homo_coords_torch = torch.tensor(homo_coords, dtype=torch.get_default_dtype(), device=src_data.device)

    x_src, y_src, z_src = np.arange(src_data.shape[-3]), np.arange(src_data.shape[-2]), np.arange(src_data.shape[-1])
    dim_conv = DimensionConverterDP(src_data.shape[-3:])
    with torch.no_grad():
        sample_coords = torch.matmul(torch.linalg.inv(src_to_tgt_transformation), homo_coords_torch)[:3].T
        grid = dim_conv.from_orig_img_dim(sample_coords).reshape(*tgt_shape,-1)[None,:,:,:,:].flip([-1])
        new_img = torch.nn.functional.grid_sample(src_data, grid, align_corners=True)
    return new_img

