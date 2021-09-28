# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:39:56 2018

@author: Daniel
"""

##

import os
import pickle
import numpy as np
import pandas as pd
import nrrd
import torch
from torch.utils.data import Dataset
from scipy.stats import multivariate_normal
from pdb import set_trace

from data_transforms import ct_normalize
import utils_sp

import re
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

##

'''
I have different Dataset classes and fetch_dataloader functions for CT-related
stuff b/c I have different sets of images and labels that I use for each
task.

In comparison, MNIST is a fixed dataset so we only need one Dataset class and
fetch_dataloader (fixed and paired set of image & label)
'''

##

class CT_seg_all_Dataset(Dataset):
    def __init__(self, data_dir, label_dir, landmarks_filepath, transform):
        super().__init__()
        """
        We load images and labels as we go - Args: data_dir, label_dir
        Also, we apply various transforms during runtime
        Unique about CTlandmark_seg_Dataset is that it doesn't care whether or not
        there's an associated valve or aortic root label. If there are none,
        the output tuple will also contain None in their place. If there exist 
        such labels, then they will be loaded
        """
        
        # there may be more ct_filenames than label_filenames or vice versa
        ct_filenames = os.listdir(data_dir)
        aortic_root_label_filenames = os.listdir(os.path.join(label_dir, 'aortic_root'))
        valve_1_label_filenames = os.listdir(os.path.join(label_dir, 'valve_1'))
        valve_2_label_filenames = os.listdir(os.path.join(label_dir, 'valve_2'))
        valve_3_label_filenames = os.listdir(os.path.join(label_dir, 'valve_3'))
        
        self.ct_filepaths = sorted([os.path.join(data_dir, f) for f in ct_filenames], key=natural_key)
        self.aortic_root_label_filepaths = sorted([os.path.join(os.path.join(label_dir, 'aortic_root'), f) for f in aortic_root_label_filenames], key=natural_key)
        self.valve_1_label_filepaths = sorted([os.path.join(os.path.join(label_dir, 'valve_1'), f) for f in valve_1_label_filenames], key=natural_key)
        self.valve_2_label_filepaths = sorted([os.path.join(os.path.join(label_dir, 'valve_2'), f) for f in valve_2_label_filenames], key=natural_key)
        self.valve_3_label_filepaths = sorted([os.path.join(os.path.join(label_dir, 'valve_3'), f) for f in valve_3_label_filenames], key=natural_key)
        
        with open(landmarks_filepath, 'rb') as f:
            self.landmarks = pickle.load(f)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.ct_filepaths)
    
    def calc_landmark_heatmap(self, ct, mu, stdev):
        # isotropic gaussian
        covariance = np.diag((np.ones(3)*stdev)**2)
        
        floor = np.floor(mu-stdev*3).astype(int)
        ceil = np.floor(mu+stdev*3).astype(int)
        
        # to make sure we don't index outside of allowed size later
        floor[0] = np.clip(floor[0], 0, ct.shape[0])
        floor[1] = np.clip(floor[1], 0, ct.shape[1])
        floor[2] = np.clip(floor[2], 0, ct.shape[2])
        
        ceil[0] = np.clip(ceil[0], 0, ct.shape[0])
        ceil[1] = np.clip(ceil[1], 0, ct.shape[1])
        ceil[2] = np.clip(ceil[2], 0, ct.shape[2])
        
        x = np.arange(floor[0], ceil[0])
        y = np.arange(floor[1], ceil[1])
        z = np.arange(floor[2], ceil[2])
    
        xmesh, ymesh, zmesh = np.meshgrid(x, y, z)
        pos = np.stack((xmesh, ymesh, zmesh), axis=3)
        
        rv = multivariate_normal(mu, covariance)
        
        lm = np.zeros(ct.shape)
        
        lm_rv = rv.pdf(pos)
        
        lm[floor[0]:ceil[0],
           floor[1]:ceil[1],
           floor[2]:ceil[2]] = lm_rv
        
        return lm
    
    def __getitem__(self, idx):
        # all of this to get label_filepath
        ct_filename = os.path.basename(self.ct_filepaths[idx])
        split_strs = os.path.splitext(ct_filename)[0].split('_')
        P_phase = '{}_{}'.format(split_strs[0], split_strs[1])
        
        aortic_root_label_filepath = [f for f in self.aortic_root_label_filepaths if P_phase in f]
        valve_1_label_filepath = [f for f in self.valve_1_label_filepaths if P_phase in f]
        valve_2_label_filepath = [f for f in self.valve_2_label_filepaths if P_phase in f]
        valve_3_label_filepath = [f for f in self.valve_3_label_filepaths if P_phase in f]
        
        print(P_phase)
        
        ct = np.load(self.ct_filepaths[idx])
        
        try:
            aortic_root_label = nrrd.read(aortic_root_label_filepath[0])[0].astype(float)
            valve_1_label = nrrd.read(valve_1_label_filepath[0])[0].astype(float)
            valve_2_label = nrrd.read(valve_2_label_filepath[0])[0].astype(float)
            valve_3_label = nrrd.read(valve_3_label_filepath[0])[0].astype(float)
        except:
            aortic_root_label = None
            valve_1_label = None
            valve_2_label = None
            valve_3_label = None
        
        mu1 = self.landmarks[P_phase]['landmark1']
        mu2 = self.landmarks[P_phase]['landmark2']
        mu3 = self.landmarks[P_phase]['landmark3']
        
        # need to data and label in dictionary to apply identical transforms to both
        # (if we do one at a time, transforms are randomized so they wouldn't necessarily correspond)        
        data_dict = {'ct': ct,
                     'landmark1': mu1,
                     'landmark2': mu2,
                     'landmark3': mu3,
                     'aortic_root_label': aortic_root_label,
                     'valve_1_label': valve_1_label,
                     'valve_2_label': valve_2_label,
                     'valve_3_label': valve_3_label}
        
        data_dict = self.transform(data_dict)
        
        ct = data_dict['ct']
        aortic_root_label = data_dict['aortic_root_label']
        valve_1_label = data_dict['valve_1_label']
        valve_2_label = data_dict['valve_2_label']
        valve_3_label = data_dict['valve_3_label']
        
        ct = ct_normalize(ct, min_bound=-158.0, max_bound=864.0)
        ct = torch.Tensor(np.ascontiguousarray(ct))
        ct = ct.unsqueeze(0)
        
        try:
            aortic_root_label = torch.Tensor(np.ascontiguousarray(aortic_root_label))
            valve_1_label = torch.Tensor(np.ascontiguousarray(valve_1_label))
            valve_2_label = torch.Tensor(np.ascontiguousarray(valve_2_label))
            valve_3_label = torch.Tensor(np.ascontiguousarray(valve_3_label))
            all_labels = torch.cat((aortic_root_label.unsqueeze(0), valve_1_label.unsqueeze(0), valve_2_label.unsqueeze(0), valve_3_label.unsqueeze(0)), dim=0)
        except:
            all_labels = None
        
        return (ct, all_labels)

##

class CT_just_load_Dataset(Dataset):
    def __init__(self, data_dir, label_dir, transform):
        super().__init__()
        """
        We load images and labels as we go - Args: data_dir, label_dir
        Also, we apply various transforms during runtime
        Unique about CTlandmark_seg_Dataset is that it doesn't care whether or not
        there's an associated valve or aortic root label. If there are none,
        the output tuple will also contain None in their place. If there exist 
        such labels, then they will be loaded
        """
        
        # there may be more ct_filenames than label_filenames or vice versa
        ct_filenames = os.listdir(data_dir)
        aortic_root_label_filenames = os.listdir(os.path.join(label_dir, 'aortic_root'))
        valve_1_label_filenames = os.listdir(os.path.join(label_dir, 'valve_1'))
        valve_2_label_filenames = os.listdir(os.path.join(label_dir, 'valve_2'))
        valve_3_label_filenames = os.listdir(os.path.join(label_dir, 'valve_3'))
        
        self.ct_filepaths = sorted([os.path.join(data_dir, f) for f in ct_filenames], key=natural_key)
        self.aortic_root_label_filepaths = sorted([os.path.join(os.path.join(label_dir, 'aortic_root'), f) for f in aortic_root_label_filenames], key=natural_key)
        self.valve_1_label_filepaths = sorted([os.path.join(os.path.join(label_dir, 'valve_1'), f) for f in valve_1_label_filenames], key=natural_key)
        self.valve_2_label_filepaths = sorted([os.path.join(os.path.join(label_dir, 'valve_2'), f) for f in valve_2_label_filenames], key=natural_key)
        self.valve_3_label_filepaths = sorted([os.path.join(os.path.join(label_dir, 'valve_3'), f) for f in valve_3_label_filenames], key=natural_key)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.ct_filepaths)
    
    def __getitem__(self, idx):
        # all of this to get label_filepath
        ct_filename = os.path.basename(self.ct_filepaths[idx])
        split_strs = os.path.splitext(ct_filename)[0].split('_')
        P_phase = '{}_{}'.format(split_strs[0], split_strs[1])
        
        aortic_root_label_filepath = [f for f in self.aortic_root_label_filepaths if P_phase in f]
        valve_1_label_filepath = [f for f in self.valve_1_label_filepaths if P_phase in f]
        valve_2_label_filepath = [f for f in self.valve_2_label_filepaths if P_phase in f]
        valve_3_label_filepath = [f for f in self.valve_3_label_filepaths if P_phase in f]
        
        print(P_phase)
        
        ct = np.load(self.ct_filepaths[idx])
        
        try:
            aortic_root_label = nrrd.read(aortic_root_label_filepath[0])[0].astype(float)
            valve_1_label = nrrd.read(valve_1_label_filepath[0])[0].astype(float)
            valve_2_label = nrrd.read(valve_2_label_filepath[0])[0].astype(float)
            valve_3_label = nrrd.read(valve_3_label_filepath[0])[0].astype(float)
        except:
            aortic_root_label = None
            valve_1_label = None
            valve_2_label = None
            valve_3_label = None
        
        # need to data and label in dictionary to apply identical transforms to both
        # (if we do one at a time, transforms are randomized so they wouldn't necessarily correspond)        
        data_dict = {'ct': ct,
                     'aortic_root_label': aortic_root_label,
                     'valve_1_label': valve_1_label,
                     'valve_2_label': valve_2_label,
                     'valve_3_label': valve_3_label}
        
        data_dict = self.transform(data_dict)
        
        ct = data_dict['ct']
        aortic_root_label = data_dict['aortic_root_label']
        valve_1_label = data_dict['valve_1_label']
        valve_2_label = data_dict['valve_2_label']
        valve_3_label = data_dict['valve_3_label']
        
#        ct = ct_normalize(ct, min_bound=-158.0, max_bound=864.0)
        ct = torch.Tensor(np.ascontiguousarray(ct))
        ct = ct.unsqueeze(0)
        
        try:
            aortic_root_label = torch.Tensor(np.ascontiguousarray(aortic_root_label))
            valve_1_label = torch.Tensor(np.ascontiguousarray(valve_1_label))
            valve_2_label = torch.Tensor(np.ascontiguousarray(valve_2_label))
            valve_3_label = torch.Tensor(np.ascontiguousarray(valve_3_label))
            all_labels = torch.cat((aortic_root_label.unsqueeze(0), valve_1_label.unsqueeze(0), valve_2_label.unsqueeze(0), valve_3_label.unsqueeze(0)), dim=0)
        except:
            all_labels = None
        
        return (ct, all_labels)
    
class CT_just_load_gt_pcl_Dataset(Dataset):
    def __init__(self, data_dir, gt_pcl_dir, transform):
        super().__init__()
        """
        We load images and labels as we go - Args: data_dir, label_dir
        Also, we apply various transforms during runtime
        Unique about CTlandmark_seg_Dataset is that it doesn't care whether or not
        there's an associated valve or aortic root label. If there are none,
        the output tuple will also contain None in their place. If there exist 
        such labels, then they will be loaded
        """
        
        # there may be more ct_filenames than label_filenames or vice versa
        self.P_phase_list = sorted([os.path.splitext(f)[0] for f in os.listdir(data_dir)], key=natural_key)
        
        self.ct_filepaths = sorted([os.path.join(data_dir, '{}.npy'.format(f)) for f in self.P_phase_list], key=natural_key)
        self.gt_pcl_filepaths = sorted([os.path.join(gt_pcl_dir, '{}.pt'.format(f)) for f in self.P_phase_list], key=natural_key)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.ct_filepaths)
    
    def __getitem__(self, idx):
        P_phase = self.P_phase_list[idx]
        print(P_phase)
        
        ct = np.load(self.ct_filepaths[idx])
        gt_pcl_list = torch.load(self.gt_pcl_filepaths[idx])
        
        # need to data and label in dictionary to apply identical transforms to both
        # (if we do one at a time, transforms are randomized so they wouldn't necessarily correspond)        
        data_dict = {'ct': ct,
                     'gt_pcl_list': gt_pcl_list}
        
        data_dict = self.transform(data_dict)
        
        ct = data_dict['ct']
        gt_pcl_list = data_dict['gt_pcl_list']
        
        ct = torch.Tensor(np.ascontiguousarray(ct))
        ct = ct.unsqueeze(0)
        
        return (ct, gt_pcl_list)

##

class CT_just_load_seg_gt_pcl_Dataset(Dataset):
    def __init__(self, data_dir, seg_label_dir, gt_pcl_dir, transform):
        super().__init__()
        """
        We load images and labels as we go - Args: data_dir, label_dir
        Also, we apply various transforms during runtime
        Unique about CTlandmark_seg_Dataset is that it doesn't care whether or not
        there's an associated valve or aortic root label. If there are none,
        the output tuple will also contain None in their place. If there exist 
        such labels, then they will be loaded
        """

        # there may be more ct_filenames than label_filenames or vice versa
        self.P_phase_list = sorted([os.path.splitext(f)[0] for f in os.listdir(data_dir)], key=natural_key)

        self.ct_filepaths = sorted([os.path.join(data_dir, '{}.npy'.format(f)) for f in self.P_phase_list], key=natural_key)
        self.gt_pcl_filepaths = sorted([os.path.join(gt_pcl_dir, '{}.pt'.format(f)) for f in self.P_phase_list], key=natural_key)

        aortic_root_label_filenames = os.listdir(os.path.join(seg_label_dir, 'aortic_root'))
        valve_1_label_filenames = os.listdir(os.path.join(seg_label_dir, 'valve_1'))
        valve_2_label_filenames = os.listdir(os.path.join(seg_label_dir, 'valve_2'))
        valve_3_label_filenames = os.listdir(os.path.join(seg_label_dir, 'valve_3'))

        self.aortic_root_label_filepaths = sorted([os.path.join(os.path.join(seg_label_dir, 'aortic_root'), f) for f in aortic_root_label_filenames], key=natural_key)
        self.valve_1_label_filepaths = sorted([os.path.join(os.path.join(seg_label_dir, 'valve_1'), f) for f in valve_1_label_filenames], key=natural_key)
        self.valve_2_label_filepaths = sorted([os.path.join(os.path.join(seg_label_dir, 'valve_2'), f) for f in valve_2_label_filenames], key=natural_key)
        self.valve_3_label_filepaths = sorted([os.path.join(os.path.join(seg_label_dir, 'valve_3'), f) for f in valve_3_label_filenames], key=natural_key)

        self.transform = transform

    def __len__(self):
        return len(self.ct_filepaths)

    def __getitem__(self, idx):
        P_phase = self.P_phase_list[idx]
        print(P_phase)

        ct = np.load(self.ct_filepaths[idx])
        gt_pcl_list = torch.load(self.gt_pcl_filepaths[idx])

        aortic_root_label_filepath = [f for f in self.aortic_root_label_filepaths if P_phase in f]
        valve_1_label_filepath = [f for f in self.valve_1_label_filepaths if P_phase in f]
        valve_2_label_filepath = [f for f in self.valve_2_label_filepaths if P_phase in f]
        valve_3_label_filepath = [f for f in self.valve_3_label_filepaths if P_phase in f]

        aortic_root_label = nrrd.read(aortic_root_label_filepath[0])[0].astype(float)
        valve_1_label = nrrd.read(valve_1_label_filepath[0])[0].astype(float)
        valve_2_label = nrrd.read(valve_2_label_filepath[0])[0].astype(float)
        valve_3_label = nrrd.read(valve_3_label_filepath[0])[0].astype(float)

        # need to data and label in dictionary to apply identical transforms to both
        # (if we do one at a time, transforms are randomized so they wouldn't necessarily correspond)
        data_dict = {'ct': ct,
                     'aortic_root_label': aortic_root_label,
                     'valve_1_label': valve_1_label,
                     'valve_2_label': valve_2_label,
                     'valve_3_label': valve_3_label,
                     'gt_pcl_list': gt_pcl_list}

        data_dict = self.transform[0](data_dict)
        data_dict = self.transform[1](data_dict)

        ct = data_dict['ct']
        aortic_root_label = data_dict['aortic_root_label']
        valve_1_label = data_dict['valve_1_label']
        valve_2_label = data_dict['valve_2_label']
        valve_3_label = data_dict['valve_3_label']
        gt_pcl_list = data_dict['gt_pcl_list']

        ct = torch.Tensor(np.ascontiguousarray(ct))
        ct = ct.unsqueeze(0)

        aortic_root_label = torch.Tensor(np.ascontiguousarray(aortic_root_label))
        valve_1_label = torch.Tensor(np.ascontiguousarray(valve_1_label))
        valve_2_label = torch.Tensor(np.ascontiguousarray(valve_2_label))
        valve_3_label = torch.Tensor(np.ascontiguousarray(valve_3_label))
        all_labels = torch.cat((aortic_root_label.unsqueeze(0), valve_1_label.unsqueeze(0), valve_2_label.unsqueeze(0), valve_3_label.unsqueeze(0)), dim=0)

        return ct, (all_labels, gt_pcl_list)
    
##

class CT_img_mesh_variable_template_Dataset(Dataset):
    def __init__(self, data_dir, template_dir, label_dir, transform):
        super().__init__()
        """
        We load images and labels as we go - Args: data_dir, label_dir
        Also, we apply various transforms during runtime
        Unique about CTlandmark_seg_Dataset is that it doesn't care whether or not
        there's an associated valve or aortic root label. If there are none,
        the output tuple will also contain None in their place. If there exist 
        such labels, then they will be loaded
        """
        
        # there may be more ct_filenames than label_filenames or vice versa
        ct_filenames = os.listdir(data_dir)
        aortic_root_label_filenames = os.listdir(os.path.join(label_dir, 'aortic_root'))
        valve_1_label_filenames = os.listdir(os.path.join(label_dir, 'valve_1'))
        valve_2_label_filenames = os.listdir(os.path.join(label_dir, 'valve_2'))
        valve_3_label_filenames = os.listdir(os.path.join(label_dir, 'valve_3'))
        
        self.ct_filepaths = sorted([os.path.join(data_dir, f) for f in ct_filenames], key=natural_key)
        self.aortic_root_label_filepaths = sorted([os.path.join(os.path.join(label_dir, 'aortic_root'), f) for f in aortic_root_label_filenames], key=natural_key)
        self.valve_1_label_filepaths = sorted([os.path.join(os.path.join(label_dir, 'valve_1'), f) for f in valve_1_label_filenames], key=natural_key)
        self.valve_2_label_filepaths = sorted([os.path.join(os.path.join(label_dir, 'valve_2'), f) for f in valve_2_label_filenames], key=natural_key)
        self.valve_3_label_filepaths = sorted([os.path.join(os.path.join(label_dir, 'valve_3'), f) for f in valve_3_label_filenames], key=natural_key)
        
        self.seg_filepaths = sorted([os.path.join(template_dir, f) for f in os.listdir(template_dir) if '_seg' in f], key=utils_sp.natural_key)
        self.hypermesh_filepaths = [os.path.join(template_dir, f) for f in os.listdir(template_dir) if '_hypermesh' in f]
        
        self.transform = transform
        
    def __len__(self):
        return len(self.ct_filepaths)
    
    def __getitem__(self, idx):
        # all of this to get label_filepath
        ct_filename = os.path.basename(self.ct_filepaths[idx])
        split_strs = os.path.splitext(ct_filename)[0].split('_')
        P_phase = '{}_{}'.format(split_strs[0], split_strs[1])
        
        aortic_root_label_filepath = [f for f in self.aortic_root_label_filepaths if P_phase in f]
        valve_1_label_filepath = [f for f in self.valve_1_label_filepaths if P_phase in f]
        valve_2_label_filepath = [f for f in self.valve_2_label_filepaths if P_phase in f]
        valve_3_label_filepath = [f for f in self.valve_3_label_filepaths if P_phase in f]
        
        print(P_phase)
        
        ct = np.load(self.ct_filepaths[idx])
        
        # randomly choose a template from the list of templates
        rand_idx = np.random.randint(len(self.seg_filepaths))
        seg_filepath = self.seg_filepaths[rand_idx]
        seg_template = np.load(seg_filepath)
        
        template_P_phase = os.path.splitext(os.path.basename(seg_filepath))[0].split('_seg')[0]
        
        hm_file_exists = False
        for hf in self.hypermesh_filepaths:
            hypermesh_P_phase = os.path.splitext(os.path.basename(hf))[0].split('_hypermesh')[0]
            if template_P_phase == hypermesh_P_phase:
                hm_file_exists = True
                hm_template_filepath = hf
        
        if hm_file_exists:
            print('Template: {}_hypermesh.npy'.format(template_P_phase))
            load_hm = np.load(hm_template_filepath, allow_pickle=True)
            verts_list_template = list(load_hm[0,:])
            faces_list_template = list(load_hm[1,:])
        else:
            print('Template: {}_seg marching cubes'.format(template_P_phase))
            verts_list_template, faces_list_template = utils_sp.seg_to_mesh(seg_template)
        
        try:
            aortic_root_label = nrrd.read(aortic_root_label_filepath[0])[0].astype(float)
            valve_1_label = nrrd.read(valve_1_label_filepath[0])[0].astype(float)
            valve_2_label = nrrd.read(valve_2_label_filepath[0])[0].astype(float)
            valve_3_label = nrrd.read(valve_3_label_filepath[0])[0].astype(float)
        except:
            aortic_root_label = None
            valve_1_label = None
            valve_2_label = None
            valve_3_label = None
        
        # need to data and label in dictionary to apply identical transforms to both
        # (if we do one at a time, transforms are randomized so they wouldn't necessarily correspond)        
        data_dict = {'ct': ct,
                     'aortic_root_label': aortic_root_label,
                     'valve_1_label': valve_1_label,
                     'valve_2_label': valve_2_label,
                     'valve_3_label': valve_3_label}
        
        data_dict = self.transform(data_dict)
        
        ct = data_dict['ct']
        aortic_root_label = data_dict['aortic_root_label']
        valve_1_label = data_dict['valve_1_label']
        valve_2_label = data_dict['valve_2_label']
        valve_3_label = data_dict['valve_3_label']
        
        ct = ct_normalize(ct, min_bound=-158.0, max_bound=864.0)
        ct = torch.Tensor(np.ascontiguousarray(ct))
        ct = ct.unsqueeze(0)
        
        try:
            aortic_root_label = torch.Tensor(np.ascontiguousarray(aortic_root_label))
            valve_1_label = torch.Tensor(np.ascontiguousarray(valve_1_label))
            valve_2_label = torch.Tensor(np.ascontiguousarray(valve_2_label))
            valve_3_label = torch.Tensor(np.ascontiguousarray(valve_3_label))
            all_labels = torch.cat((aortic_root_label.unsqueeze(0), valve_1_label.unsqueeze(0), valve_2_label.unsqueeze(0), valve_3_label.unsqueeze(0)), dim=0)
        except:
            all_labels = None
        
        return (ct, seg_template, verts_list_template, faces_list_template), all_labels

##

import pyvista as pv
import glob
import utils_cgal_related as utils_cgal
from pytorch3d.ops import sample_points_from_meshes

class Mesh_GtPcl_Dataset(Dataset):
    def __init__(self, data_dir, label_dir):
        super().__init__()

        input_mesh_path = glob.glob(os.path.join(data_dir, '*.vtk'))[0]
        input_extra_info_path = glob.glob(os.path.join(data_dir, '*.pkl'))[0]
        template_verts, template_faces = utils_cgal.get_verts_faces_from_pyvista(pv.read(input_mesh_path))
        with open(input_extra_info_path, 'rb') as f:
            self.extra_info_template = pickle.load(f)
        template_faces_tri = utils_cgal.split_quad_to_2_tri_mesh(template_faces)
        self.verts_list_template_torch = utils_sp.np_list_to_torch_list([template_verts], n_batch=None)
        self.faces_list_template_torch = utils_sp.np_list_to_torch_list([template_faces_tri], n_batch=None)

        label_mesh_path = glob.glob(os.path.join(label_dir, '*.vtk'))[0]
        label_extra_info_path = glob.glob(os.path.join(label_dir, '*.pkl'))[0]
        label_verts, label_faces = utils_cgal.get_verts_faces_from_pyvista(pv.read(label_mesh_path))
        with open(label_extra_info_path, 'rb') as f:
            faces_list = pickle.load(f)
        mesh_verts_list = [label_verts, label_verts, label_verts, label_verts]
        mesh_tri_faces_list = []
        for faces in faces_list:
            mesh_pv = utils_sp.mesh_to_pyvista_PolyData(label_verts, faces)
            mesh_pv.triangulate(inplace=True)
            mesh_tri_faces_list.append(utils_cgal.get_verts_faces_from_pyvista(mesh_pv)[1])
        mesh = utils_sp.mesh_to_pytorch3d_Mesh(mesh_verts_list, mesh_tri_faces_list)
        self.gt_pcl = sample_points_from_meshes(mesh).cpu()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return (self.verts_list_template_torch, self.faces_list_template_torch, self.extra_info_template), self.gt_pcl

class ExtractedVolSurface_GtPcl_Dataset(Dataset):
    def __init__(self, template_filename, target_filename):
        mesh_hex_pv_template = pv.read(os.path.join('../template_for_deform', template_filename))
        surf_template = mesh_hex_pv_template.extract_geometry()
        surf_template = surf_template.triangulate()
        verts, faces = utils_cgal.get_verts_faces_from_pyvista(surf_template)

        self.verts_list_template_torch = utils_sp.np_list_to_torch_list([verts], n_batch=None)
        self.faces_list_template_torch = utils_sp.np_list_to_torch_list([faces], n_batch=None)
        self.extra_info_template = []

        mesh_hex_pv_target = pv.read(os.path.join('../template_for_deform', target_filename))
        surf_target = mesh_hex_pv_target.extract_geometry()
        surf_target = surf_target.triangulate()
        verts, faces = utils_cgal.get_verts_faces_from_pyvista(surf_target)
        mesh = utils_sp.mesh_to_pytorch3d_Mesh([verts], [faces])

        self.gt_pcl = sample_points_from_meshes(mesh).cpu()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return (self.verts_list_template_torch, self.faces_list_template_torch, self.extra_info_template), self.gt_pcl
