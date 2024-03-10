import numpy as np
import torch

class DimensionConverter():
    '''
    originally going to and from [0, dim_size-1] and [-1, 1]
    now going to and from [offset, dim_size-1+offset] and [-1, 1]
    '''
    def __init__(self, dim_size, offset=0):
        self.dim_size = dim_size - 1 # minus 1 here b/c if img has size 64, we want [0,63] range
        self.offset = offset
    
    def to_dim_size(self, x):
        m = self.dim_size/2
        b = self.offset + self.dim_size/2
        
        y = m*x + b
        
        return y
    
    def from_dim_size(self, x):
        m = 2/self.dim_size
        b = -1 - 2*self.offset/self.dim_size
        
        y = m*x + b
        
        return y

class DimensionConverterDP():
    '''
    going to and from [0, dim_size-1] and [-1, 1]
    '''
    def __init__(self, shape):
        self.shape = np.array(shape) - 1 # minus 1 here b/c if img has size 64, we want [0,63] range
    
    def to_orig_img_dim(self, x):
        m = torch.tensor(self.shape/2, dtype=torch.get_default_dtype(), device=x.device)
        b = torch.tensor(self.shape/2, dtype=torch.get_default_dtype(), device=x.device)
        
        y = m*x + b
        
        return y
    
    def from_orig_img_dim(self, x):
        m = torch.tensor(2/self.shape, dtype=torch.get_default_dtype(), device=x.device)
        b = -1
        
        y = m*x + b
        
        return y