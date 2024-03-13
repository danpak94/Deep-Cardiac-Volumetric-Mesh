"""
    Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation

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
    
    Modifications:
    - Mostly to keep the older version of Airlab for compatibility with the rest of the dcvm library
    - Removed unused methods
"""

##

import torch as torch
import torch.nn.functional as F
import numpy as np
from pdb import set_trace

##

"""
    Base class for a transformation
"""
class _Transformation(torch.nn.Module):
    def __init__(self, image_size, diffeomorphic=False, dtype=torch.float32, device='cpu'):
        super(_Transformation, self).__init__()
        
        self._dtype = dtype
        self._device = device
        self._dim = len(image_size)
        self._image_size = np.array(image_size)
        self._diffeomorphic = diffeomorphic
        self._constant_flow = None
        
        self._compute_flow = None
        
        if self._diffeomorphic:
            self._diffeomorphic_calculater = Diffeomorphic(image_size, dtype=dtype, device=device)
        else:
            self._diffeomorphic_calculater = None
    
    def get_inverse_displacement(self, trans_parameters):
        flow = self._compute_flow(trans_parameters)
        
        if self._diffeomorphic:
                inv_displacement = self._diffeomorphic_calculater.calculate(flow * -1)
        else:
            print("error displacement ")
            inv_displacement = None
            
        return inv_displacement
    
    def _compute_diffeomorphic_displacement(self, flow):
        return self._diffeomorphic_calculater.calculate(flow)

##

class _KernelTransformation_DP(_Transformation):
    def __init__(self, image_size, diffeomorphic=False, dtype=torch.float32, device='cpu'):
        super(_KernelTransformation_DP, self).__init__(image_size, diffeomorphic, dtype, device)
        
        self._kernel = None
        self._stride = 1
        self._padding = 0
        
        assert self._dim == 2 or self._dim == 3
        
        if self._dim == 2:
            self._compute_flow = self._compute_flow_2d
        else:
            self._compute_flow = self._compute_flow_3d
            
    def _initialize(self):
        cp_grid_shape = np.ceil(np.divide(self._image_size, self._stride)).astype(dtype=int)
        
        # new image size after convolution
        inner_image_size = np.multiply(self._stride, cp_grid_shape) - (self._stride - 1)
        
        # add one control point at each side - not knots..?
        cp_grid_shape = cp_grid_shape + 2
        
        # center image between control points
        image_size_diff = inner_image_size - self._image_size
        image_size_diff_floor = np.floor((np.abs(image_size_diff)/2))*np.sign(image_size_diff)
        
        self._crop_start = image_size_diff_floor + np.remainder(image_size_diff, 2)*np.sign(image_size_diff)
        self._crop_end = image_size_diff_floor
        
        cp_grid_shape = [1, self._dim] + cp_grid_shape.tolist() # n_batch, n_dim, h, w, d
        self.cp_grid_shape = cp_grid_shape
        
        # copy to gpu if needed
        self.to(dtype=self._dtype, device=self._device)
        
        # convert to integer
        self._padding = self._padding.astype(dtype=int).tolist()
        self._stride = self._stride.astype(dtype=int).tolist()
        
        self._crop_start = self._crop_start.astype(dtype=int)
        self._crop_end = self._crop_end.astype(dtype=int)
        
    def _compute_flow_2d(self, trans_parameters):
        displacement = F.conv_transpose2d(trans_parameters, self._kernel,
                                          padding=self._padding, stride=self._stride, groups=2)
        
        # crop displacement
        return displacement[:, :,
                            self._stride[0] + self._crop_start[0]:-self._stride[0] - self._crop_end[0],
                            self._stride[1] + self._crop_start[1]:-self._stride[1] - self._crop_end[1]]
    
    def _compute_flow_3d(self, trans_parameters):
        # compute dense displacement
        displacement = F.conv_transpose3d(trans_parameters, self._kernel,
                                          padding=self._padding, stride=self._stride, groups=3)
        
        # crop displacement
        return displacement[:, :, 
                            self._stride[0] + self._crop_start[0]:-self._stride[0] - self._crop_end[0],
                            self._stride[1] + self._crop_start[1]:-self._stride[1] - self._crop_end[1],
                            self._stride[2] + self._crop_start[2]:-self._stride[2] - self._crop_end[2]]
    
    def forward(self, trans_parameters):
        flow = self._compute_flow(trans_parameters)
        
        if self._diffeomorphic:
            displacement = self._compute_diffeomorphic_displacement(flow)
        else:
            displacement = flow
            
        return displacement

##

"""
    bspline kernel transformation
"""
class BsplineTransformation(_KernelTransformation_DP):
    def __init__(self, image_size, sigma, diffeomorphic=False, order=2, dtype=torch.float32, device='cpu'):
        super(BsplineTransformation, self).__init__(image_size, diffeomorphic, dtype, device)
        
        self._stride = np.array(sigma)
        
        # compute bspline kernel
        self._kernel = bspline_kernel(sigma, dim=self._dim, order=order, asTensor=True, dtype=dtype)
        
        self._padding = (np.array(self._kernel.size()) - 1) / 2
        
        self._kernel.unsqueeze_(0).unsqueeze_(0)
        self._kernel = self._kernel.expand(self._dim, *((np.ones(self._dim + 1, dtype=int)*-1).tolist()))
        self._kernel = self._kernel.to(dtype=dtype, device=self._device)
        
        self._initialize()
        
##
"""
    Create a 1d bspline kernel matrix
"""
def bspline_kernel_1d(sigma, order=2, asTensor=False, dtype=torch.float32, device='cpu'):
    kernel_ones = torch.ones(1, 1, sigma)
    kernel = kernel_ones
    
    padding = sigma - 1
    
    for i in range(1, order + 1):
        kernel = F.conv1d(kernel, kernel_ones, padding=padding)/sigma
	
    if asTensor:
        return kernel[0, 0, ...].to(dtype=dtype, device=device)
    else:
        return kernel[0, 0, ...].numpy()

"""
    Create a 2d bspline kernel matrix
"""
def bspline_kernel_2d(sigma=[1, 1], order=2, asTensor=False, dtype=torch.float32, device='cpu'):
    kernel_ones = torch.ones(1, 1, *sigma)
    kernel = kernel_ones
    padding = np.array(sigma) - 1
    
    for i in range(1, order + 1):
        kernel = F.conv2d(kernel, kernel_ones, padding=(padding).tolist())/(sigma[0]*sigma[1])
        
    if asTensor:
        return kernel[0, 0, ...].to(dtype=dtype, device=device)
    else:
        return kernel[0, 0, ...].numpy()
    
"""
    Create a 3d bspline kernel matrix
"""
def bspline_kernel_3d(sigma=[1, 1, 1], order=2, asTensor=False, dtype=torch.float32, device='cpu'):
    kernel_ones = torch.ones(1, 1, *sigma)
    kernel = kernel_ones
    padding = np.array(sigma) - 1
    
    for i in range(1, order + 1):
        kernel = F.conv3d(kernel, kernel_ones, padding=(padding).tolist())/(sigma[0]*sigma[1]*sigma[2])
        
    if asTensor:
        return kernel[0, 0, ...].to(dtype=dtype, device=device)
    else:
        return kernel[0, 0, ...].numpy()

"""
    Create a bspline kernel matrix for a given dim
"""
def bspline_kernel(sigma, order=2, dim=1, asTensor=False, dtype=torch.float32, device='cpu'):
    assert dim > 0 and dim <=3
    
    if dim == 1:
        return bspline_kernel_1d(sigma, order=order, asTensor=asTensor, dtype=dtype, device=device)
    elif dim == 2:
        return bspline_kernel_2d(sigma, order=order, asTensor=asTensor, dtype=dtype, device=device)
    else:
        return bspline_kernel_3d(sigma, order=order, asTensor=asTensor, dtype=dtype, device=device)

##

class Diffeomorphic():
    r"""
    Diffeomorphic transformation. This class computes the matrix exponential of a given flow field using the scaling
    and squaring algorithm according to:
              Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
              Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
              MICCAI 2018
              and
              Diffeomorphic Demons: Efficient Non-parametric Image Registration
              Tom Vercauterena et al., 2008

    """
    def __init__(self, image_size=None, scaling=10, dtype=torch.float32, device='cpu'):
        self._dtype = dtype
        self._device = device
        self._dim = len(image_size)
        self._image_size = image_size
        self._scaling = scaling
        self._init_scaling = 8
        
        if image_size is not None:
            self._image_grid = compute_grid(image_size, dtype=dtype, device=device)
        else:
            self._image_grid = None
            
    def set_image_size(self, image_szie):
        self._image_size = image_szie
        self._image_grid = compute_grid(self._image_size, dtype=self._dtype, device=self._device)
        
    def calculate(self, displacement):
        if self._dim == 2:
            return Diffeomorphic.diffeomorphic_2D(displacement, self._image_grid, self._scaling)
        else:
            return Diffeomorphic.diffeomorphic_3D(displacement, self._image_grid, self._scaling)
        
    @staticmethod
    def _compute_scaling_value(displacement):
        with torch.no_grad():
            scaling = 8
            norm = torch.norm(displacement / (2 ** scaling))
            
            while norm > 0.5:
                scaling += 1
                norm = torch.norm(displacement / (2 ** scaling))
                
        return scaling
    
    @staticmethod
    def diffeomorphic_2D(displacement, grid, scaling=-1):
        if scaling < 0:
            scaling = Diffeomorphic._compute_scaling_value(displacement)
            
        displacement = displacement / (2 ** scaling)
        
        for i in range(scaling):
            displacement_trans = displacement.permute(0,2,3,1)
            displacement = displacement + F.grid_sample(displacement, displacement_trans + grid, align_corners=True)
            
        return displacement
    
    @staticmethod
    def diffeomorphic_3D(displacement, grid, scaling=-1):
        displacement = displacement / (2 ** scaling)
        
        for i in range(scaling):
            displacement_trans = displacement.permute(0,2,3,4,1)
            displacement = displacement + F.grid_sample(displacement, displacement_trans + grid, align_corners=True)
            
        return displacement
    
##
        
def compute_grid(image_size, dtype=torch.float32, device='cpu'):
    dim = len(image_size)
    
    if dim == 2:
        nx = image_size[0]
        ny = image_size[1]
        
        x = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)
        y = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)
        
        x = x.expand(nx, -1)
        y = y.expand(ny, -1).transpose(0, 1)
        
        x.unsqueeze_(0).unsqueeze_(3)
        y.unsqueeze_(0).unsqueeze_(3)
        
        return torch.cat((x, y), 3).to(dtype=dtype, device=device)
    
    elif dim == 3:
        nz = image_size[0]
        ny = image_size[1]
        nx = image_size[2]
        
        x = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)
        y = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)
        z = torch.linspace(-1, 1, steps=nz).to(dtype=dtype)
        
        x = x.expand(ny, -1).expand(nz, -1, -1)
        y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2)
        z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1)
        
        x.unsqueeze_(0).unsqueeze_(4)
        y.unsqueeze_(0).unsqueeze_(4)
        z.unsqueeze_(0).unsqueeze_(4)
        
        return torch.cat((x, y, z), 4).to(dtype=dtype, device=device)
    else:
        print("Error " + dim + "is not a valid grid type")
