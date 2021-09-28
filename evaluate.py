# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 00:05:28 2018

@author: Daniel
"""

##

import logging
import numpy as np
import torch
import numbers

##

def evaluate(model, loss_fn, val_test_dl, metrics, params):
    # summary for current eval loop
    summary_batch_list = []
    
    # compute metrics over the dataset
    for data_batch, labels_batch in val_test_dl:
        # move to GPU if available
        if type(data_batch) is list:
            for data_batch_idx, x in enumerate(data_batch):
                if torch.is_tensor(x):
                    if params.cuda:
                        data_batch[data_batch_idx] = data_batch[data_batch_idx].contiguous().cuda()
                    data_batch[data_batch_idx] = torch.autograd.Variable(data_batch[data_batch_idx])
        else:
            if params.cuda:
                data_batch = data_batch.contiguous().cuda()
            data_batch = torch.autograd.Variable(data_batch)
            
        if type(labels_batch) is list:
            for labels_list_idx, x in enumerate(labels_batch):
                if torch.is_tensor(x):
                    if params.cuda:
                        labels_batch[labels_list_idx] = labels_batch[labels_list_idx].contiguous().cuda()
                    labels_batch[labels_list_idx] = torch.autograd.Variable(labels_batch[labels_list_idx])    
        else:
            if params.cuda:
                labels_batch = labels_batch.contiguous().cuda()
            labels_batch = torch.autograd.Variable(labels_batch)

        with torch.no_grad():
            output_batch = model(data_batch)
            loss_items = loss_fn(output_batch, labels_batch)
        
        loss_terms = None
        
        if isinstance(loss_items, tuple):
            loss = loss_items[0]
            loss_terms = loss_items[1]
        else:
            loss = loss_items
        
        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        
        summary_batch['loss'] = loss.item()
        
        if loss_terms is not None:
            summary_batch['loss_terms'] = loss_terms
            
        summary_batch_list.append(summary_batch)
    
    # compute mean of all metrics in summary
    metrics_mean = {}
    for metric in summary_batch_list[0]:
        if isinstance(summary_batch_list[0][metric], numbers.Number):
            metrics_mean[metric] = np.mean([summary_batch[metric] for summary_batch in summary_batch_list])
        elif isinstance(summary_batch_list[0][metric], list):
            if isinstance(summary_batch_list[0][metric][0], numbers.Number):
                metrics_mean[metric] = np.mean([summary_batch[metric] for summary_batch in summary_batch_list], axis=0)
            elif isinstance(summary_batch_list[0][metric][0], str):
                metrics_mean[metric] = summary_batch_list[0][metric]

    metrics_string = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items() if isinstance(v, numbers.Number))
    
    logging.info("- Eval metrics : " + metrics_string)
    
    return metrics_mean

##