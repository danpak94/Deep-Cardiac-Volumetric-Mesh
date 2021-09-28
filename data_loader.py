# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:37:11 2019

@author: Daniel
"""

##

import numpy as np
from copy import copy
from torch.utils.data import DataLoader, random_split

from things_to_modify_for_tasks import get_transforms, get_full_ds

##

def fetch_dataloader(params, data_dir, label_dir=None, return_ds=False):
    # The only custom parts in this function (specific to task)
    train_transformer, eval_transformer = get_transforms(params)
    full_ds = get_full_ds(data_dir, label_dir, eval_transformer, params)
    
    n_total = full_ds.__len__()
    
    n_train = int(np.ceil(n_total*params.train_fraction))
    n_val = max(int(n_total*params.val_fraction), 1)
    n_test = n_total - n_train - n_val
    
    n_train_val_test = [n_train, n_val, n_test]
    
    train_ds, val_ds, test_ds = random_split(full_ds, n_train_val_test)
    
    # train_ct_filepaths = [train_ds.dataset.ct_filepaths[idx] for idx in train_ds.indices]
    # train_ct_mean = get_total_mean(train_ct_filepaths)
    #
    # full_ds.train_ct_mean = train_ct_mean # in case we want to zero_center testing set

    train_ds.dataset = copy(full_ds)
    train_ds.dataset.transform = train_transformer # different transforms for training vs. val & test sets
    
    train_dl = DataLoader(train_ds, batch_size=params.batch_size, shuffle=True,
                          num_workers=params.num_workers,
                          pin_memory=params.cuda)

    if len(val_ds) > 0:
        val_dl = DataLoader(val_ds, batch_size=params.batch_size, shuffle=False,
                            num_workers=params.num_workers,
                            pin_memory=params.cuda)
    else:
        val_dl = train_dl

    if len(test_ds) > 0:
        test_dl = DataLoader(test_ds, batch_size=params.batch_size, shuffle=False,
                            num_workers=params.num_workers,
                            pin_memory=params.cuda)
    else:
        test_dl = train_dl
    
    if return_ds:
        return train_dl, val_dl, test_dl, train_ds, val_ds, test_ds
    else:
        return train_dl, val_dl, test_dl

##

def fetch_dataloader_test(params, data_dir, label_dir=None, return_ds=False):
    # The only custom parts in this function (specific to task)
    _, eval_transformer = get_transforms(params)
    test_ds = get_full_ds(data_dir, label_dir, eval_transformer, params)
    
    test_dl = DataLoader(test_ds, batch_size=params.batch_size, shuffle=False,
                        num_workers=params.num_workers,
                        pin_memory=params.cuda)
    
    if return_ds:
        return test_dl, test_ds
    else:
        return test_dl
