# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 18:47:00 2018

@author: Daniel
"""

##

import logging
import numpy as np
import torch
from tqdm import tqdm
import numbers

import utils_sp as utils

def train(model, optimizer, loss_fn, train_dl, metrics, params):
    # summary for current training loop and a running average object for loss
    summary_batch_list = []
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(train_dl)) as t:
        for i, (train_batch, labels_batch) in enumerate(train_dl):
            # move to GPU if available
            if type(train_batch) is list:
                for train_batch_idx, x in enumerate(train_batch):
                    if torch.is_tensor(x):
                        if params.cuda:
                            train_batch[train_batch_idx] = train_batch[train_batch_idx].contiguous().cuda()
                        train_batch[train_batch_idx] = torch.autograd.Variable(train_batch[train_batch_idx])
            else:
                if params.cuda:
                    train_batch = train_batch.contiguous().cuda()
                train_batch = torch.autograd.Variable(train_batch)
            
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

            output_batch = model(train_batch)

            loss_items = loss_fn(output_batch, labels_batch)
            loss_terms = None
            
            if isinstance(loss_items, tuple):
                loss = loss_items[0]
                loss_terms = loss_items[1]
            else:
                loss = loss_items

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()
            
            # performs updates using calculated gradients
            optimizer.step()
            
            # Evaluate summaries only once in a while
            if i % min(len(train_dl), params.save_summary_steps) == 0:                
                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                
                summary_batch['loss'] = loss.item()
                
                if loss_terms is not None:
                    summary_batch['loss_terms'] = loss_terms
                    
                summary_batch_list.append(summary_batch)
            
            loss_avg.update(loss.item())
            
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
            torch.cuda.empty_cache()
    
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
    
    logging.info("- Train metrics: " + metrics_string)
    
    return metrics_mean

##
