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

import os
import torch
import json
import importlib

from dcvm.utils import Config

def load_config(exp_dir):
    json_path = os.path.join(exp_dir, 'config.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    config = Config.from_json(json_path)
    config.cuda = torch.cuda.is_available()
    return config

def load_checkpoint(filepath, model=None, optimizer=None, scheduler=None, map_location=torch.device('cuda')):
    if not os.path.isfile(filepath):
        raise("File doesn't exist {}".format(filepath))
    checkpoint = torch.load(filepath, map_location=map_location)
    if model:
        model.load_state_dict(checkpoint['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint

def save_checkpoint(state, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)

def load_model(exp_dir, checkpoint_name='best', map_location=torch.device('cuda')):
    config = load_config(exp_dir)
    net = importlib.import_module('dcvm.models.{}'.format(config.model_used))
    model = net.Net(config).cuda() if config.cuda else net.Net(config)
    # map_location = torch.device('cuda') if params.cuda else torch.device('cpu')
    load_checkpoint(os.path.join(exp_dir, 'checkpoint_{}.pt'.format(checkpoint_name)), model=model, map_location=map_location)
    model.eval()
    return model