import os
import torch
import json
import importlib

from dcvm.utils import Config

def load_config(exp_dir):
    json_path = os.path.join(exp_dir, 'config.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    config = Config.from_file(json_path)
    config.cuda = torch.cuda.is_available()
    return config

def load_checkpoint(restore_path, model=None, optimizer=None, scheduler=None, map_location=torch.device('cuda')):
    if not os.path.isfile(restore_path):
        raise("File doesn't exist {}".format(restore_path))
    checkpoint = torch.load(restore_path, map_location=map_location)
    if model:
        model.load_state_dict(checkpoint['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler'])
    return checkpoint

def load_model(exp_dir, checkpoint_name='best', map_location=torch.device('cuda')):
    config = load_config(exp_dir)
    net = importlib.import_module('dcvm.models.{}'.format(config.model_used))
    model = net.Net(config).cuda() if config.cuda else net.Net(config)
    # map_location = torch.device('cuda') if params.cuda else torch.device('cpu')
    load_checkpoint(os.path.join(exp_dir, 'checkpoint_{}.pt'.format(checkpoint_name)), model=model, map_location=map_location)
    model.eval()
    return model