"""
    Copyright 2024 Daniel H. Pak, Yale University

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import interpn

from dcvm.transforms.utils_transforms import DimensionConverterDP

def ct_normalize(pixel_array, min_bound=-158.0, max_bound=864.0):
    pix_normalized = (pixel_array - min_bound) / (max_bound - min_bound)
    pix_normalized[pix_normalized>1] = 1.
    pix_normalized[pix_normalized<0] = 0.
    return pix_normalized

def apply_linear_transform_on_img_torch(src_data, src_to_tgt_transformation, tgt_shape, grid_sample_mode='bilinear', grid_padding_mode='zeros', require_grad=False):
    '''
    decide which source voxels to sample, starting from target voxel coordinates
    src_data = img_cuda[n_transformations,n_channels,:,:,:]
    src_to_tgt_transformation: homogeneous coordinate frame in 3D space, shape (n_transformations,4,4) or (4,4)
    tgt_shape = [128,128,128]
    '''
    if len(src_to_tgt_transformation.shape) == 2:
        src_to_tgt_transformation = src_to_tgt_transformation.clone()[None]
    
    x, y, z = torch.arange(tgt_shape[0]), torch.arange(tgt_shape[1]), torch.arange(tgt_shape[2])
    xmesh, ymesh, zmesh = torch.meshgrid(x, y, z, indexing='ij') # (x.shape, y.shape, z.shape)
    coords_stack = torch.stack([xmesh, ymesh, zmesh], dim=0).reshape(3,-1) # (3, np.prod(tgt_shape))
    homo_coords = torch.cat([coords_stack, torch.ones([1,coords_stack.shape[1]])], dim=0).to(dtype=torch.get_default_dtype(), device=src_data.device) # (4, np.prod(tgt_shape))

    dim_conv = DimensionConverterDP(src_data.shape[-3:])
    if not require_grad:
        with torch.no_grad():
            sample_coords = torch.matmul(torch.linalg.inv(src_to_tgt_transformation), homo_coords)[:,:3].permute(0,2,1) # (n_tramsforations, np.prod(tgt_shape), 3)
            grid = dim_conv.from_orig_img_dim(sample_coords).reshape(src_to_tgt_transformation.shape[0],*tgt_shape,3).flip([-1]) # (n_transformations, *tgt_shape, 3)
            new_img = torch.nn.functional.grid_sample(src_data, grid, align_corners=True, mode=grid_sample_mode, padding_mode=grid_padding_mode) # (n_transformations, n_channels, *tgt_shape)
    else:
        sample_coords = torch.matmul(torch.linalg.inv(src_to_tgt_transformation), homo_coords)[:,:3].permute(0,2,1) # (n_tramsforations, np.prod(tgt_shape), 3)
        grid = dim_conv.from_orig_img_dim(sample_coords).reshape(src_to_tgt_transformation.shape[0],*tgt_shape,3).flip([-1]) # (n_transformations, *tgt_shape, 3)
        new_img = torch.nn.functional.grid_sample(src_data, grid, align_corners=True, mode=grid_sample_mode, padding_mode=grid_padding_mode) # (n_transformations, n_channels, *tgt_shape)
    return new_img

def apply_linear_transform_on_img(src_data, src_to_tgt_transformation, tgt_shape, interpn_method='linear'):
    # decide which source voxels to sample, starting from target voxel coordinates
    x, y, z = np.arange(tgt_shape[0]), np.arange(tgt_shape[1]), np.arange(tgt_shape[2])
    xmesh, ymesh, zmesh = np.meshgrid(x, y, z, indexing='ij')
    coords = np.stack([xmesh, ymesh, zmesh], axis=0)
    coords_stack = coords.reshape(3,-1)
    homo_coords = np.concatenate([coords_stack, np.ones([1,coords_stack.shape[1]])], axis=0)
    sample_coords = np.dot(np.linalg.inv(src_to_tgt_transformation), homo_coords)[:3].T
    x_src, y_src, z_src = np.arange(src_data.shape[0]), np.arange(src_data.shape[1]), np.arange(src_data.shape[2])
    new_img = interpn([x_src, y_src, z_src], src_data, sample_coords, method=interpn_method, bounds_error=False, fill_value=0).reshape(tgt_shape)
    return new_img

class LinearTransformOnImgTorch:
    # mostly to pre-define the grid and not waste time re-calculating the same things
    def __init__(self, tgt_shape, device='cuda'):
        x, y, z = np.arange(tgt_shape[0]), np.arange(tgt_shape[1]), np.arange(tgt_shape[2])
        xmesh, ymesh, zmesh = np.meshgrid(x, y, z, indexing='ij')
        coords_stack = np.stack([xmesh, ymesh, zmesh], axis=0).reshape(3,-1)
        homo_coords = np.concatenate([coords_stack, np.ones([1,coords_stack.shape[1]])], axis=0)
        self.homo_coords_torch = torch.tensor(homo_coords, dtype=torch.get_default_dtype(), device=device)
        self.tgt_shape = tgt_shape

    def __call__(self, src_data, src_to_tgt_transformation, mode='bilinear', padding_mode='zeros'):
        with torch.no_grad():
            sample_coords = torch.matmul(torch.linalg.inv(src_to_tgt_transformation), self.homo_coords_torch)[:3].T
            dim_conv = DimensionConverterDP(src_data.shape[-3:])
            grid = dim_conv.from_orig_img_dim(sample_coords).reshape(*self.tgt_shape,-1)[None,:,:,:,:].flip([-1])
            new_img = F.grid_sample(src_data, grid, align_corners=True, mode=mode, padding_mode=padding_mode)
        return new_img
    
class ImgMinMaxNormalize:
    def __init__(self, min_bound=-200, max_bound=1000):
        # originally, min_bound=-158.0, max_bound=864.0
        self.min_bound = min_bound
        self.max_bound = max_bound
    
    def __call__(self, data_dict):
        img = data_dict['img']
        img_normalized = (img - self.min_bound) / (self.max_bound - self.min_bound)
        img_normalized[img_normalized>1] = 1.
        img_normalized[img_normalized<0] = 0.
        output_dict = {key: val for key, val in data_dict.items() if key != 'img'}
        output_dict['img'] = img_normalized
        return output_dict
    
class ImgZscoreNormalize:
    def __init__(self, eps=1e-8):
        self.eps = eps
    
    def __call__(self, data_dict):
        img = data_dict['img']
        mean = img.mean(dim=[-3,-2,-1], keepdim=True)
        stdev = img.std(dim=[-3,-2,-1], keepdim=True)
        img_normalized = (img - mean) / (stdev + self.eps)
        output_dict = {key: val for key, val in data_dict.items() if key != 'img'}
        output_dict['img'] = img_normalized
        return output_dict