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

import sys
import os
import json
import importlib

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
    def from_json(cls, file_path):
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    @classmethod
    def from_py(cls, filepath):
        module_name = os.path.splitext(os.path.basename(filepath))[0] # file_name = 'config.py'
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def keys(self):
        return self.__dict__.keys()
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self, indent=4, level=0):
        out = []
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                out.append(f"{' ' * indent * level}{key}:")
                out.append(value.__str__(indent=indent, level=level + 1))
            else:
                out.append(f"{' ' * indent * level}{key}: {value}")
        return "\n".join(out)

def flatten_list_of_lists(lst_of_lsts, return_len_each_sublist=False):
    flat_lst = [x for sublist in lst_of_lsts for x in sublist]
    len_each_sublist = [len(sublist) for sublist in lst_of_lsts]
    if return_len_each_sublist:
        return flat_lst, len_each_sublist
    else:
        return flat_lst

def unflatten_list_of_lists(flat_lst, len_each_sublist):
    lst_of_lsts = []
    start = 0
    for n in len_each_sublist:
        lst_of_lsts.append(flat_lst[start:start+n])
        start += n
    return lst_of_lsts

def pv_get_point_id_callback(mesh_pv, point_id):
    if 'RegionId' in mesh_pv.point_data.keys():
        print('RegionId: {}'.format(mesh_pv.point_data['RegionId'][0]))
    print('Point idx: {}'.format(point_id))

def pv_get_point_id_callback_point_picker(point, point_picker):
    print('Point idx: {}'.format(point_picker.GetPointId()))

def pv_get_cell_id_callback(cell_id):
    print('Cell idx: {}'.format(cell_id.cell_data['original_cell_ids']))