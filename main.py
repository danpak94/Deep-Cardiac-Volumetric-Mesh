# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:55:38 2018

@author: Daniel
"""

##

import os
import torch
import logging
import importlib
import numpy as np
import warnings
import argparse

import utils_sp as utils
import eval_helper
from train import train
from evaluate import evaluate

from things_to_modify_for_tasks import get_loss_fn
from data_loader import fetch_dataloader
from custom_dataset_classes import natural_key

from pdb import set_trace

## function that runs training + validation steps

def train_and_val(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, exp_dir, restore_file=None):
    
    if restore_file is not None:
        restore_path = os.path.join(exp_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        checkpoint = utils.load_checkpoint(restore_path, model, optimizer)
        
        epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        train_metrics_list = checkpoint['train_metrics_list']
        val_metrics_list = checkpoint['val_metrics_list']
    else:
        epoch = 0
        best_val_loss = np.inf
        train_metrics_list = []
        val_metrics_list = []
    
    seg_net_loaded = False
    
    while epoch < params.num_epochs:
        
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        
        # set model to training mode
        model.train()
        
        if 'preload_seg_net' in params.dict.keys():
            if params.preload_seg_net == "True":                
                if not seg_net_loaded:
                    print('preload seg_net')
                    unet_dir = '../experiments/71_multilabel_cropped_aligned_unet'
                    utils.load_checkpoint(os.path.join(unet_dir, 'best.pth.tar'), model=model.seg_net_instance)
                    # doing this boolean check b/c this loading needs to happen after model.train() and then set requires_grad = False for just the seg_net
                    seg_net_loaded = True
                
                for segnet_params in model.seg_net_instance.parameters():
                    segnet_params.requires_grad = False
                    
                model.seg_net_instance.eval()
        
        train_metrics = train(model, optimizer, loss_fn, train_dl, metrics, params)
        torch.cuda.empty_cache()
        
        # set model to evaluation mode
        model.eval()
        val_metrics = evaluate(model, loss_fn, val_dl, metrics, params)
        torch.cuda.empty_cache()
        
        train_metrics_list.append(train_metrics)
        val_metrics_list.append(val_metrics)
        
        is_best = val_metrics['loss']<=best_val_loss
        
        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'best_val_loss': best_val_loss,
                               'train_metrics_list': train_metrics_list,
                               'val_metrics_list': val_metrics_list,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=exp_dir)
        
        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new lowest loss")
            best_val_loss = val_metrics['loss']
            
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(exp_dir, "metrics_val_best.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(exp_dir, "metrics_val_last.json")
        utils.save_dict_to_json(val_metrics, last_json_path)
        
        epoch += 1
    
    print('epoch == params.num_epochs')
    print('Train and Evaluate ALL DONE')

## Functions to remove clutter in main execution

def setup_torch(cuda_available):
    torch.manual_seed(1234)
    if cuda_available: torch.cuda.manual_seed(230)
    torch.set_default_dtype(torch.float32)

def save_stuff_from_server(train_dl, val_dl, test_dl, exp_dir):
    train_idx = train_dl.dataset.indices
    val_idx = val_dl.dataset.indices
    test_idx = test_dl.dataset.indices
    
    all_ct_filepaths = train_dl.dataset.dataset.ct_filepaths
    train_ct_filepaths = sorted([all_ct_filepaths[idx] for idx in train_idx], key=natural_key)
    val_ct_filepaths = sorted([all_ct_filepaths[idx] for idx in val_idx], key=natural_key)
    test_ct_filepaths = sorted([all_ct_filepaths[idx] for idx in test_idx], key=natural_key)
    
    stuff_from_server = {'train_ct_mean': train_dl.dataset.dataset.train_ct_mean,
                         'train_ct_filepaths': train_ct_filepaths,
                         'val_ct_filepaths': val_ct_filepaths,
                         'test_ct_filepaths': test_ct_filepaths}
    
    utils.save_dict_to_json(stuff_from_server, os.path.join(exp_dir, "stuff_from_server.json"))

def argparse_stuff():
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings('ignore', message='.*From scipy 0.13.0.*')
    warnings.filterwarnings('ignore', message='.*nn.Upsampling is deprecated.*')
    
    parser = argparse.ArgumentParser()
    
    '''
    parser arguments
    
    data_dir: directory where images are stored (with exception to MNIST)
    label_dir: directory where label (e.g. segmentation map) is stored
    exp_dir: directory where params.json and other job-specific things are/should be stored
    restore_file: restore file name within exp_dir to load into model and start training from
    '''
    
    parser.add_argument('--data_dir', default='../../data/ct_npy/npy_combined_full_train_64x64x64')
    # parser.add_argument('--label_dir', default='../../data/valve_seg/npy_sep_valves_64x64x64')
    parser.add_argument('--label_dir', default='../../data/gt_pcl/liangmesh_train_64x64x64')
    # parser.add_argument('--label_dir', default='../../data/gt_pcl/combined_full_train_64x64x64_one_component')
    # parser.add_argument('--label_dir', default='../../data/gt_pcl/combined_full_train_64x64x64')
    # parser.add_argument('--label_dir', default='../../data/gt_pcl/skel_surf_train_64x64x64_one_component')
    def string_delimited_process(string_delimited):
        split = string_delimited.split(',')
        if len(split) == 1:
            return(split[0])
        else:
            return split
    # parser.add_argument('--label_dir', type=string_delimited_process, default='../../data/valve_seg/npy_sep_valves_64x64x64,../../data/gt_pcl/combined_full_train_64x64x64_one_component')
    # parser.add_argument('--data_dir', default='../template_for_deform/closed_base_surf_combined')
    # parser.add_argument('--label_dir', default='../template_for_deform/open_base_surf_combined')
    parser.add_argument('--exp_dir', default='../experiments/debugging')
    parser.add_argument('--restore_file', default=None)
    
    args = parser.parse_args()
    
    return args

##

if __name__ == '__main__':
    args = argparse_stuff()
    params = utils.load_params(args.exp_dir)
    
    print('This is the task being run: {}'.format(params.task))
    
    setup_torch(params.cuda)
    
    train_dl, val_dl, test_dl = fetch_dataloader(params, args.data_dir, label_dir=args.label_dir)
    # save_stuff_from_server(train_dl, val_dl, test_dl, args.exp_dir)

    net = importlib.import_module('model.{}'.format(params.model_used))
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    # optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.learning_rate)

    loss_fn = get_loss_fn(params, params.task)
    metrics = eval_helper.metrics
    
    logger = utils.Logger()
    logger.set_logger(os.path.join(args.exp_dir, 'train.log'))
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    
    train_and_val(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, args.exp_dir, \
        restore_file=args.restore_file)
