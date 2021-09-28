# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 17:44:31 2019

@author: Daniel
"""

import torchvision.transforms as transforms
import eval_helper
import custom_dataset_classes as cdc
import data_transforms as dt

## in data_loader.py

'''
Note: empty transforms are placeholders (and they are required to be there)
'''

def get_transforms(params):
    task = params.task
    if task == 'mtm_smooth' or task == 'mtm_geo' or task == 'mtm_arap' or task == 'mtm_weighted_arap':
        train_transformer = transforms.Compose([dt.BsplineDeformGridSample_64x64x64(sigma=params.bspline_deform_sigma,
                                                                                    order=params.bspline_deform_order,
                                                                                    deform_chance=params.bspline_deform_chance)])
        eval_transformer = transforms.Compose([])
        
    return train_transformer, eval_transformer

## in data_loader.py

def get_full_ds(data_dir, label_dir, eval_transformer, params):
    task = params.task
    if task == 'mtm_smooth' or task == 'mtm_geo' or task == 'mtm_arap' or task == 'mtm_weighted_arap':
        full_ds = cdc.CT_just_load_gt_pcl_Dataset(data_dir, label_dir, eval_transformer)
    
    return full_ds

## in main.py

def get_loss_fn(params, task):
    if task == 'mtm_smooth':
        loss_fn = eval_helper.ChamferSmoothnessGtPcl(lambdas=params.loss_lambdas)
    elif task == 'mtm_geo':
        loss_fn = eval_helper.ChamferGeoGtPcl(lambdas=params.loss_lambdas, edge_loss_which=params.edge_loss_which)
    elif task == 'mtm_arap':
        loss_fn = eval_helper.ChamferARAPGtPcl(params.arap_template_filename, lambdas=params.loss_lambdas)
    elif task == 'mtm_weighted_arap':
        loss_fn = eval_helper.ChamferWeightedARAPGtPcl(params.arap_template_filename1, params.arap_template_filename2,
                                                       lambdas=params.loss_lambdas, softmax_base_exp=params.arap_softmax_base_exp,
                                                       deform_gradient_method=params.deform_gradient_method, distortion_type=params.distortion_type)

    return loss_fn
