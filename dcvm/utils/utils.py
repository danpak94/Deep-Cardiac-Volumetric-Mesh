import os
import torch
import json
import importlib
import types

import re
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

class Config():
    '''
    Use example:
        config = Config(config_dict)

    Note: need to load config_dict to make it recursive-able
    1. enable dot notation (e.g. config.model.architecture, etc.)
    2. enable nice prints (e.g. print(config) or just typing config in notebook)
    '''
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    @classmethod
    def from_file(self, file_path):
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return self(config_dict)

    def keys(self):
        return self.__dict__.keys()
    
    def __str__(self, indent=4, level=0):
        out = []
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                out.append(f"{' ' * indent * level}{key}:")
                out.append(value.__str__(indent=indent, level=level + 1))
            else:
                out.append(f"{' ' * indent * level}{key}: {value}")
        return "\n".join(out)

def setup_torch(cuda_available):
    torch.manual_seed(1234)
    if cuda_available: torch.cuda.manual_seed(230)
    torch.set_default_dtype(torch.float32)

