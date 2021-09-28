# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 17:31:11 2018

@author: Daniel
"""

##

import sys
import os
import shutil
import json
import logging
import numpy as np
import pandas as pd
import torch
import numbers
import nrrd
import pickle
import pyvista as pv
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import torch.nn.functional as F

from pytorch3d.structures import Meshes
import pytorch3d
from pytorch3d.ops import knn_points

import importlib

##

import re
def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def load_params(exp_dir):
    json_path = os.path.join(exp_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    
    params = Params(json_path)
    params.cuda = torch.cuda.is_available()
    
    return params

def load_model(exp_dir):
    params = load_params(exp_dir)

    net = importlib.import_module('model.{}'.format(params.model_used))
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    map_location = torch.device('cuda') if params.cuda else torch.device('cpu')
    load_checkpoint(os.path.join(exp_dir, 'best.pth.tar'), model=model, map_location=map_location)
    model.eval()
    
    return model

def check_convert_cuda(train_batch, labels_batch, params):
    if type(train_batch) is list:
        for train_batch_idx, _ in enumerate(train_batch):
            if params.cuda:
                train_batch[train_batch_idx] = train_batch[train_batch_idx].contiguous().cuda()
            train_batch[train_batch_idx] = torch.autograd.Variable(train_batch[train_batch_idx])
    else:
        if params.cuda:
            train_batch = train_batch.contiguous().cuda()
        train_batch = torch.autograd.Variable(train_batch)
    
    if type(labels_batch) is list:
        for labels_list_idx, _ in enumerate(labels_batch):
            if params.cuda:
                labels_batch[labels_list_idx] = labels_batch[labels_list_idx].contiguous().cuda()
            labels_batch[labels_list_idx] = torch.autograd.Variable(labels_batch[labels_list_idx])
    else:
        if params.cuda:
            labels_batch = labels_batch.contiguous().cuda()
        labels_batch = torch.autograd.Variable(labels_batch)
    
    return train_batch, labels_batch

##

class Params():
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
    
class Logger():
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
    
    def set_logger(self, log_path):
        if not self.logger.handlers:
            # Logging to a file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            self.logger.addHandler(file_handler)
    
            # Logging to console
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(stream_handler)

    def reset_logger(self):
        self.logger.handlers = []
        
class RunningAverage():
    """A simple class that maintains the running average of a quantity"""
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
    

def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        new_d = {}
        for k,v in d.items():
            if isinstance(v, numbers.Number):
                #TODO: round to 10 sigfigs.. this rounds to 10th decimal place (important for when value is small)
                new_d[k] = round(float(v), 10)
            elif isinstance(v, list):
                if len(v)>0:
                    if not isinstance(v[0], torch.Tensor):
                        new_d[k] = v
            elif isinstance(v, np.ndarray):
                new_d[k] = list(np.around(v.astype(float), 10))
        json.dump(new_d, f, indent=4)

def modify_json(json_path, param, param_val):
    with open(json_path, 'r') as rf:
        data = json.load(rf)
    
    try:
        data[param] = int(param_val)
    except ValueError:
        data[param] = float(param_val)
    except:
        data[param] = param_val
    
    with open(json_path, 'w') as wf:
        json.dump(data, wf, indent=4)

def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath_last = os.path.join(checkpoint, 'last.pth.tar')
    filepath_best = os.path.join(checkpoint, 'best.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath_last)
    if is_best:
        if os.path.isfile(filepath_best):
            os.remove(filepath_best)
        shutil.copyfile(filepath_last, filepath_best)
        
def load_checkpoint(restore_path, model=None, optimizer=None, map_location=torch.device('cuda')):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(restore_path):
        raise("File doesn't exist {}".format(restore_path))
    
    checkpoint = torch.load(restore_path, map_location=map_location)
    
    if model:
        model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    
    return checkpoint

def change_ct_filepaths_of_dataset(dataset, server_ct_filepaths):
    orig_dataset_ct_filepaths = dataset.ct_filepaths[:]
    dataset.ct_filepaths = []
    
    for f in server_ct_filepaths:
        P_phase = os.path.splitext(os.path.basename(f))[0]
        
        filepath_list = [f for f in orig_dataset_ct_filepaths if P_phase in f]
        
        if len(filepath_list)>0:
            dataset.ct_filepaths.append(filepath_list[0])
            
##

##

def get_data_shape_array(data_dir_list):
    n_files = 0
    for data_dir in data_dir_list:
        filename_list = os.listdir(data_dir)
        for filename in filename_list:
            n_files += 1
    
    data_shape_array = np.zeros([n_files, 3])
    
    file_idx = 0
    
    for data_dir in data_dir_list:
        filename_list = os.listdir(data_dir)
        for filename in filename_list:
            filepath = os.path.join(data_dir, filename)
            print(filepath)
            data = np.load(filepath)
            data_shape_array[file_idx, :] = np.array(data.shape)
            
            file_idx += 1
            
    return data_shape_array

##

def organize_df(exp_dir_list, re_index_dict, re_column_dict):
    
    train_mean_store = []
    train_var_store = []
    test_mean_store = []
    test_var_store = []
    
    for exp_dir in exp_dir_list:
        
        exp_idx = os.path.basename(exp_dir).split('_')[0]
        try:
            exp_idx = int(exp_idx)
        except:
            exp_idx = int(os.path.basename(os.path.dirname(exp_dir)).split('_')[0])
        
        train_dice = pd.read_csv(os.path.join(exp_dir, 'dice_df.csv'), index_col=0)
        
        test_dice1 = pd.read_csv(os.path.join(exp_dir, 'dice_df2.csv'), index_col=0)
        test_dice2 = pd.read_csv(os.path.join(exp_dir, 'dice_df3.csv'), index_col=0)
        test_dice = pd.concat([test_dice1, test_dice2])
        
        train_mean_store.append(train_dice.mean().to_frame(name=exp_idx))
        train_var_store.append(train_dice.var().to_frame(name=exp_idx))
        test_mean_store.append(test_dice.mean().to_frame(name=exp_idx))
        test_var_store.append(test_dice.var().to_frame(name=exp_idx))
    
    train_mean_df = pd.concat(train_mean_store, axis=1, sort=True).transpose()
    train_var_df = pd.concat(train_mean_store, axis=1, sort=True).transpose()
    test_mean_df = pd.concat(test_mean_store, axis=1, sort=True).transpose()
    test_var_df = pd.concat(test_var_store, axis=1, sort=True).transpose()
    
    train_mean_df = train_mean_df.rename(index=re_index_dict, columns=re_column_dict)
    train_var_df = train_var_df.rename(index=re_index_dict, columns=re_column_dict)
    test_mean_df = test_mean_df.rename(index=re_index_dict, columns=re_column_dict)
    test_var_df = test_var_df.rename(index=re_index_dict, columns=re_column_dict)
    
    return train_mean_df, train_var_df, test_mean_df, test_var_df

##

def unit_vector_torch(data, device='cpu'):
    """
    copy of function from gui_DP/transformations.py, but with autograd.np instead
    """
    data = torch.tensor(data, dtype=torch.get_default_dtype(), device=device)
    data /= torch.sqrt(torch.matmul(data, data))
        
    return data

def rotation_matrix_torch(batch_angle, direction, point=None):
    """
    copy of function from gui_DP/transformations.py, but with autograd.np instead
    """
    n_batch = batch_angle.shape[0]
    
    sina = torch.sin(batch_angle).unsqueeze(1)
    cosa = torch.cos(batch_angle).unsqueeze(1)
    unit_direction = unit_vector_torch(direction[:3], batch_angle.device).repeat(n_batch, 1)
    sina_unit_direction = unit_direction * sina
    
    # rotation matrix around unit vector
    component1 = torch.mul(torch.eye(3, dtype=torch.get_default_dtype(), device=batch_angle.device).repeat(n_batch, 1, 1), cosa.unsqueeze(2))
    
    component2 = torch.mul(torch.einsum('bp,bq->bpq', unit_direction, unit_direction), (1.0 - cosa).unsqueeze(2))
    
    dir1_mat = torch.tensor([[ 0,  0,  0],
                             [ 0,  0, -1],
                             [ 0,  1,  0]], dtype=torch.get_default_dtype(), device=batch_angle.device).repeat(n_batch, 1, 1)
    
    dir2_mat = torch.tensor([[ 0,  0,  1],
                             [ 0,  0,  0],
                             [-1,  0,  0]], dtype=torch.get_default_dtype(), device=batch_angle.device).repeat(n_batch, 1, 1)
    
    dir3_mat = torch.tensor([[ 0, -1,  0],
                             [ 1,  0,  0],
                             [ 0,  0,  0]], dtype=torch.get_default_dtype(), device=batch_angle.device).repeat(n_batch, 1, 1)
    
    component3 = torch.mul(dir1_mat, sina_unit_direction[:,0].unsqueeze(1).unsqueeze(2)) \
               + torch.mul(dir2_mat, sina_unit_direction[:,1].unsqueeze(1).unsqueeze(2)) \
               + torch.mul(dir3_mat, sina_unit_direction[:,2].unsqueeze(1).unsqueeze(2))
    
    R = component1 + component2 + component3
    
    if point is not None:
        # rotation not around origin
        point = torch.tensor(point[:3], dtype=torch.get_default_dtype(), device=batch_angle.device).unsqueeze(1).repeat(n_batch, 1, 1)
        M_intermediate = torch.cat((R, point - torch.matmul(R, point)), dim=2)
    else:
        M_intermediate = torch.cat((R, torch.tensor([0,0,0], dtype=torch.get_default_dtype(), device=batch_angle.device).unsqueeze(1).repeat(n_batch, 1, 1)), dim=2)
    
    M = torch.cat((M_intermediate, torch.tensor([0,0,0,1], dtype=torch.get_default_dtype(), device=batch_angle.device).unsqueeze(0).repeat(n_batch, 1, 1)), dim=1)
    
    return M

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

##

#def get_mesh_DP(w, verts, faces, colors_input=np.array([1,1,1,0.1]), reduce_factor=1, translate_vector=None, draw_edges=True, draw_faces=True):
#    # pix_resampled 3d data
#    # isolevel = constant where we want surface
#    # colors = 4d np array, [r,g,b,opacity]
#    
##    mesh_data = gl.MeshData(vertexes=verts, faces=faces[0:554969,:])
#    mesh_data = gl.MeshData(vertexes=verts, faces=faces)
#    colors = np.ones((mesh_data.faceCount(), 4))
#    colors[:,0:3] = np.tile(np.linspace(0.1, 1, mesh_data.faceCount()).reshape(-1,1), (1,3))
#    colors = colors*colors_input
##    colors = np.tile(colors_input, (mesh_data.faceCount(), 1))
#    mesh_data.setFaceColors(colors)
#    
#    mesh = gl.GLMeshItem(meshdata=mesh_data, smooth=False, edgeColor=colors_input, drawEdges=draw_edges, drawFaces=draw_faces)
#        
##    mesh.setGLOptions('additive') # What was this for...????!!!????
##    d1, d2, d3 = pix_resampled.shape
#    if translate_vector is None:
#        d1, d2, d3 = verts.mean(axis=0)
#    else:
#        d1, d2, d3 = translate_vector
#    mesh.translate(-d1, -d2, -d3)
##    mesh.translate(-d1/2*reduce_factor, -d2/2*reduce_factor, -d3/3*reduce_factor)
##    mesh.translate(-123.112, -92.985, +151.829)
##    mesh2.rotate(-90, 0,1,0)
##    mesh.rotate(90, 0,0,1)
#    
#    return mesh, mesh_data
#
#def change_vertex_color(mesh, mesh_data, colors_input):
#    '''
#    colors_input: list of length 4 (r,g,b,alpha) or numpy.ndarray (n_vertexes, 4)
#    '''
#    if isinstance(colors_input, list):
#        if len(colors_input) == 4:
#            colors = np.tile(colors_input, (mesh_data.faceCount(),1))
#        else:
#            raise ValueError('colors_input is a list, must be length 4')
#    elif isinstance(colors_input, np.ndarray):
#        if colors_input.shape[0] == mesh_data.vertexes().shape[0] and colors_input.shape[1] == 4:
#            colors = colors_input
#        else:
#            raise ValueError('colors_input is np.ndarray, must be of shape (n_vertexes, 4)')
#    else:
#        raise ValueError('invalid colors_input (check function for specification)')
#        
#    mesh_data.setVertexColors(colors)
#    mesh.setMeshData(meshdata=mesh_data)
#
#def change_face_color(mesh, mesh_data, colors_input):
#    '''
#    colors_input: list of length 4 (r,g,b,alpha) or numpy.ndarray (n_faces, 4)
#    '''
#    if isinstance(colors_input, list):
#        if len(colors_input) == 4:
#            colors = np.tile(colors_input, (mesh_data.faceCount(),1))
#        else:
#            raise ValueError('colors_input is a list, must be length 4')
#    elif isinstance(colors_input, np.ndarray):
#        if colors_input.shape[0] == mesh_data.faceCount() and colors_input.shape[1] == 4:
#            colors = colors_input
#        else:
#            raise ValueError('colors_input is np.ndarray, must be of shape (n_faces, 4)')
#    else:
#        raise ValueError('invalid colors_input (check function for specification)')
#        
#    mesh_data.setFaceColors(colors)
#    mesh.setMeshData(meshdata=mesh_data)
#
#def display_colorbar(cmap, plot_vals, return_cb=False):
#    '''
#    
#    '''
#    fig = plt.figure(figsize=(8, 1), num='colorbar')
#    ax1 = fig.add_axes([0.05, 0.5, 0.9, 0.15])
#    
#    norm = matplotlib.colors.Normalize(vmin=plot_vals.min(), vmax=plot_vals.max())
#    
#    cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap,
#                                           norm=norm,
#                                           orientation='horizontal')
#    
#    if return_cb:
#        return cb1
#
#def plot3d_mesh(verts_list, faces_list, translate_vector=None, window_title=None, draw_edges=True, draw_faces=True, choose_plot=[1,1,1,1]):
#    pg.setConfigOption('background', 'w')
#    
#    w = gl.GLViewWidget()
#    w.show()
#    
#    if window_title:
#        w.setWindowTitle(window_title)
#    else:
#        w.setWindowTitle('pyqtgraph example: GLIsosurface')
#        
#    w.setCameraPosition(distance=100)
#    
#    #g = gl.GLGridItem()
#    #g.scale(20,20,1)
#    #w.addItem(g)
#    
#    axes = gl.GLAxisItem()
#    axes.setSize(100,100,100)
#    w.addItem(axes)
#    
##    colors_input_list = [np.array([0,1,0,0.2]),
##                         np.array([1,0,0,0.2]),
##                         np.array([0,0,1,0.2]),
##                         np.array([1,1,0,0.2])]
#    
#    colors_input_list = [np.array([0,1,0,1]),
#                         np.array([1,0,0,1]),
#                         np.array([0,0,1,1]),
#                         np.array([1,1,0,1])]
#    
#    if translate_vector is None:
#        translate_vector = verts_list[0].mean(axis=0)
#    
#    mesh_list = []
#    mesh_data_list = []
#    
#    for idx in range(len(verts_list)):
#        if choose_plot[idx] == 1:
#            verts = verts_list[idx]
#            faces = faces_list[idx]
#            colors_input = colors_input_list[idx]
#            
#            mesh, mesh_data = get_mesh_DP(w, verts, faces, colors_input=colors_input, translate_vector=translate_vector, draw_edges=draw_edges, draw_faces=draw_faces)
#            
#            mesh_list.append(mesh)
#            mesh_data_list.append(mesh_data)
#        
#            w.addItem(mesh)
#        else:
#            mesh_list.append([])
#            mesh_data_list.append([])
#        
#    return mesh_list, mesh_data_list

##

def get_verts_faces_from_pytorch3d_Mesh(mesh):
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()
    
    return verts, faces

def point_cloud_to_voxel(pcl, voxel_shape=[64,64,64]):
    '''
    pcl: np.array [n_points, n_dim]
    '''
    n_dim = pcl.shape[1]
    
    if n_dim != len(voxel_shape):
        raise ValueError('pcl.shape[1] must equal len(voxel_shape)')
    
    voxel = np.zeros(voxel_shape)
    for point_idx in range(pcl.shape[0]):
        point = pcl[point_idx, :]
        idx = np.round(point).astype(int)
        voxel[idx[0], idx[1], idx[2]] += 1
        
    voxel = (voxel > 0)    
    
    return voxel

def mesh_to_pyvista_PolyData(verts, faces):
    '''
    verts: np.ndarray
    faces: np.ndarray [n_faces, n_verts_per_face]
    OR
    faces: list of lists [ [0,1,2], [2,3,4,5], [2,3,4], ... ]
    '''
    if isinstance(faces, np.ndarray):
        faces_pv = np.hstack(np.concatenate([faces.shape[1]*np.ones([faces.shape[0],1]), faces], axis=1)).astype(int)
    elif isinstance(faces, list):
        """ convert for pyvista, also for vtk """
        faces_pv = []
        for face in faces:
            faces_pv += ([len(face)] + face)
        faces_pv = np.array(faces_pv).astype(int)

    if faces_pv[0] == 2:
        mesh_pv = pv.PolyData(verts)
        mesh_pv.lines = faces_pv
        return mesh_pv
    else:
        return pv.PolyData(verts, faces_pv)

def pyvista_PolyData_to_mesh(custom_pv):
    verts = custom_pv.points.copy()
    faces = custom_pv.faces.reshape(-1, custom_pv.faces[0]+1)[:,1:]
    return verts, faces

##

from skimage import measure
from scipy.interpolate import interpn
# import trimesh

# def choose_one_mesh_with_most_verts(verts, faces):
#     mesh = trimesh.Trimesh(vertices=verts, faces=faces)
#     split_meshes = mesh.split()
#     if len(split_meshes) > 1:
#         verts_counts = []
#         for m in split_meshes:
#             verts_counts.append(m.vertices.shape[0])
# 
#         idx = np.array(verts_counts).argmax()
#         
#         return np.array(split_meshes[idx].vertices), np.array(split_meshes[idx].faces)
#     else :
#         return verts, faces

def seg_to_mesh(seg, step_size=1):
    '''
    seg: np array, 4xhxwxd
    step_size: positive floats work, but will have same effect as rounded integer, larger step_size = coarser mesh
    '''
    verts_list = []
    faces_list = []
    for seg_each_comp in seg:
        verts, faces, _, _ = measure.marching_cubes_lewiner(seg_each_comp, step_size=step_size)
        verts_list.append(np.ascontiguousarray(verts))
        faces_list.append(np.ascontiguousarray(faces))

    # aortic_verts, aortic_faces, _, _ = measure.marching_cubes_lewiner(seg[0,:,:,:], step_size=step_size)
    # valve_1_verts, valve_1_faces, _, _ = measure.marching_cubes_lewiner(seg[1,:,:,:], step_size=step_size)
    # valve_2_verts, valve_2_faces, _, _ = measure.marching_cubes_lewiner(seg[2,:,:,:], step_size=step_size)
    # valve_3_verts, valve_3_faces, _, _ = measure.marching_cubes_lewiner(seg[3,:,:,:], step_size=step_size)
    
    # aortic_verts, aortic_faces = choose_one_mesh_with_most_verts(aortic_verts, aortic_faces)
    # valve_1_verts, valve_1_faces = choose_one_mesh_with_most_verts(valve_1_verts, valve_1_faces)
    # valve_2_verts, valve_2_faces = choose_one_mesh_with_most_verts(valve_2_verts, valve_2_faces)
    # valve_3_verts, valve_3_faces = choose_one_mesh_with_most_verts(valve_3_verts, valve_3_faces)
    
    # verts_list = [np.ascontiguousarray(aortic_verts),
    #               np.ascontiguousarray(valve_1_verts),
    #               np.ascontiguousarray(valve_2_verts),
    #               np.ascontiguousarray(valve_3_verts)]
    #
    # faces_list = [np.ascontiguousarray(aortic_faces),
    #               np.ascontiguousarray(valve_1_faces),
    #               np.ascontiguousarray(valve_2_faces),
    #               np.ascontiguousarray(valve_3_faces)]
    
    return verts_list, faces_list

def mesh_to_pytorch3d_Mesh(verts_list, faces_list):
    if isinstance(verts_list[0], np.ndarray):
        verts_list_torch = np_list_to_torch_list(verts_list, n_batch=None, device='cuda') # (n_verts, n_dim)
    elif isinstance(verts_list[0], torch.Tensor):
        verts_list_torch = [verts.squeeze().cuda() for verts in verts_list] # (n_verts, n_dim)

    if isinstance(faces_list[0], np.ndarray):
        faces_list_torch = np_list_to_torch_list(faces_list, n_batch=None, device='cuda') # (n_faces, n_dim)
    elif isinstance(faces_list[0], torch.Tensor):
        faces_list_torch = [faces.squeeze().cuda() for faces in faces_list] # (n_faces, n_dim)
    
    meshes = Meshes(verts=verts_list_torch, faces=faces_list_torch)
    
    return meshes

def interpolate_field(displacement_field_tuple, verts_list, img_size=[64,64,64]):
    X = np.arange(0,img_size[0])
    Y = np.arange(0,img_size[1])
    Z = np.arange(0,img_size[2])
    
    new_field_list = []
    
    for idx, verts in enumerate(verts_list):
        if len(verts_list) == len(displacement_field_tuple):    
            field = displacement_field_tuple[idx].squeeze().cpu().numpy()
        elif len(displacement_field_tuple) == 1:
            field = displacement_field_tuple[0].squeeze().cpu().numpy()
        else:
            raise ValueError('displacement_field_tuple should have same length as verts_list or have length 1')
        
        new_field = np.zeros([verts.shape[0], 3])
        
        new_field[:,0] = interpn((X, Y, Z), field[0,:,:,:]*img_size[0], verts)
        new_field[:,1] = interpn((X, Y, Z), field[1,:,:,:]*img_size[1], verts)
        new_field[:,2] = interpn((X, Y, Z), field[2,:,:,:]*img_size[2], verts)
        
        new_field_list.append(new_field)
    
    return new_field_list

# def calc_curvature(verts, faces):
#     verts = verts.copy()
#     faces = faces.copy()
#     verts = torch.tensor(verts, dtype=torch.get_default_dtype(), device='cpu').unsqueeze(0)
#     faces = torch.tensor(faces, dtype=torch.get_default_dtype(), device='cpu').unsqueeze(0)
#     
#     laplacian1 = LaplacianLoss(faces)
#     Lx1 = laplacian1(verts).squeeze()
#     curvature = torch.norm(Lx1, p=2, dim=1).numpy()
#     
#     return curvature

def batched_index_select(input, dim, index):
    views = [input.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    
    return torch.gather(input, dim, index)

def interpolate_rescale_field_torch(displacement_field_tuple, verts_list, img_size=[64,64,64], reversed_field=True):
    '''
    displacement_field_tuple: tuple of torch.tensors (n_batch, n_dim, h, w, d)
    verts_list: list of numpy.ndarray (n_verts, n_dim) or list of torch.tensor (n_batch, n_verts, n_dim)
    '''
    new_field_list = []
    
    for idx, verts in enumerate(verts_list):
        if len(verts_list) == len(displacement_field_tuple):
            field = displacement_field_tuple[idx].clone()
        elif len(displacement_field_tuple) == 1:
            field = displacement_field_tuple[0].clone()
        else:
            raise ValueError('displacement_field_tuple should have same length as verts_list or have length 1')
        
        n_batch = field.shape[0]
        
        # to convert from [-1,1] to [0,img.shape[dim_idx]-1] ---- don't need to do this for field, b/c [-1,1] is for affine_grid
        # comment this later b/c we wanna be using consistent field magnitude from model prediction --- having second thoughts about this
        for dim_idx in range(field.shape[1]):
            field[:,dim_idx,:,:,:] = field[:,dim_idx,:,:,:]*(img_size[dim_idx]-1)/2
        
        if reversed_field:
            field_rearrange = torch.cat([field[:,2,:,:,:].unsqueeze(1),
                                         field[:,1,:,:,:].unsqueeze(1),
                                         field[:,0,:,:,:].unsqueeze(1)], dim=1)

            x_mesh, y_mesh, z_mesh = torch.meshgrid(torch.arange(0, img_size[0], dtype=torch.get_default_dtype(), device=field.device), \
                                                    torch.arange(0, img_size[1], dtype=torch.get_default_dtype(), device=field.device), \
                                                    torch.arange(0, img_size[2], dtype=torch.get_default_dtype(), device=field.device))
            
            meshes = torch.cat([x_mesh.unsqueeze(0),
                                y_mesh.unsqueeze(0),
                                z_mesh.unsqueeze(0)]).repeat(n_batch, 1, 1, 1, 1)
            
            src_points = meshes + field_rearrange # (n_batch, 3, d, h, w)
            src_points = src_points.reshape(n_batch, 3, -1).permute([0,2,1])
            
            # TODO: verts should be (n_batch, n_points, n_dim), not (n_points, n_dim)
            # TODO: also should not squeeze src_points and verts for torch operation
            # _, nearest_neighbor_idxes, _ = knn_points(verts, src_points)
            # nearest_neighbor_idxes = nearest_neighbor_idxes[:,:,0]
            # new_field = -batched_index_select(field_rearrange.reshape(n_batch, 3, -1).permute([0,2,1]), 1, nearest_neighbor_idxes)

            new_field_list_temp = []
            _, nearest_neighbor_idxes_all, _ = knn_points(verts, src_points, K=1)
            for temp_idx in range(nearest_neighbor_idxes_all.shape[2]):
                nearest_neighbor_idxes = nearest_neighbor_idxes_all[:,:,temp_idx]
        #        new_field = field_rearrange.reshape(n_batch, 3, -1).permute([0,2,1])[:,nearest_neighbor_idxes,:]
                new_field_temp = -batched_index_select(field_rearrange.reshape(n_batch, 3, -1).permute([0,2,1]), 1, nearest_neighbor_idxes)
                new_field_list_temp.append(new_field_temp)
            new_field = torch.stack(new_field_list_temp).mean(dim=0)
        else:
            dim_convert = DimensionConverter(64)
            if not torch.is_tensor(verts):
                verts = torch.tensor(verts, dtype=torch.get_default_dtype(), device=field.device).unsqueeze(0)
            verts_dim_converted = dim_convert.from_dim_size(verts) # (n_batch, n_pts, 3)
            verts_dim_converted = verts_dim_converted.unsqueeze(1).unsqueeze(1) # (n_batch, 1, 1, n_pts, 3) (required for input to grid_sample)
            verts_dim_converted_rearrange = torch.cat([verts_dim_converted[:,:,:,:,2].unsqueeze(4),
                                                       verts_dim_converted[:,:,:,:,1].unsqueeze(4),
                                                       verts_dim_converted[:,:,:,:,0].unsqueeze(4)], dim=4)

            field_rearrange = torch.cat([field[:,2,:,:,:].unsqueeze(1),
                                         field[:,1,:,:,:].unsqueeze(1),
                                         field[:,0,:,:,:].unsqueeze(1)], dim=1)
            new_field = F.grid_sample(field_rearrange, verts_dim_converted_rearrange, align_corners=True).permute(0,2,3,4,1).squeeze(1).squeeze(1) # (n_batch, 3, 1, 1, n_pts) to (n_batch, n_pts, 3)
        
#        print('nearest_neighbor location')
#        print(src_points[:,nearest_neighbor_idxes,:])
        
        # TODO: should not squeeze new_field for torch operation
#        new_field_list.append(-new_field.squeeze().cpu().numpy())
        new_field_list.append(new_field)
    
    return new_field_list

def move_verts_with_field(verts_list, interp_field_list, convert_to_np=False):
    new_verts_list = []
    for idx, verts in enumerate(verts_list):
        if len(verts_list) == len(interp_field_list):
            interp_field = interp_field_list[idx]
        elif len(interp_field_list) == 1:
            interp_field = interp_field_list[0]
        
        if convert_to_np:
            new_verts_list.append(verts.cpu().numpy() + interp_field.cpu().numpy())
        else:
            new_verts_list.append(verts + interp_field)
        
    return new_verts_list

##

# import file_conversions as file_conv

def pointCloud_to_occupancy_grid(pointCloud, grid_dim):
    pc_int = pointCloud.astype(int)
    occupancy_grid = np.zeros(grid_dim)

    occupancy_grid[pc_int[:,0], pc_int[:,1], pc_int[:,2]] = 1
    occupancy_grid = occupancy_grid.astype(int)

    return occupancy_grid

def verts_list_to_occ_grid(verts_list, grid_dim=[64,64,64]):
    '''
    verts_list: list of verts, np.ndarray [n_pts, n_dim]
    '''
    occ_grid_array = np.zeros([len(verts_list), *grid_dim])
    for idx, verts in enumerate(verts_list):
        if  verts is not None:
            occ_grid = pointCloud_to_occupancy_grid(verts, grid_dim)
            occ_grid_array[idx,:,:,:] = occ_grid

    return occ_grid_array

def np_list_to_torch_list(np_list, dtype=torch.get_default_dtype(), n_batch=1, device='cpu'):
    torch_list = []
    
    for np_entry in np_list:
        np_entry = np_entry.copy()
        if n_batch is not None:
            torch_list.append(torch.tensor(np_entry, dtype=dtype, device=device).repeat(n_batch,1,1))
        else:
            torch_list.append(torch.tensor(np_entry, dtype=dtype, device=device))
    
    return torch_list

def torch_list_to_np_list(torch_list):
    np_list = []
    
    for torch_entry in torch_list:
        torch_entry = torch_entry.clone()
        np_list.append(torch_entry.squeeze().cpu().numpy())
    
    return np_list

def torch_list_cpu_cuda_convert(input_list, device='cuda'):
    output_list = []
    
    for entry in input_list:
        entry = entry.clone()
        output_list.append(entry.to(device=device))
    
    return output_list

##

from scipy import spatial
def get_nearest_neighbor_scipy(src_points, interp_points):
    '''
    src_points, interp_points: torch.tensor (n_batch x n_points x n_dim)
    '''
    n_batch = src_points.shape[0]
    
    src_points_np = src_points.detach().cpu().numpy()
    if torch.is_tensor(interp_points):
        interp_points_np = interp_points.detach().cpu().numpy()
    else:
        interp_points_np = interp_points[np.newaxis,:,:]
    
    indices = torch.zeros([n_batch, interp_points_np.shape[1]], dtype=torch.long, device=src_points.device)
    
    for batch_idx in range(n_batch):
        indices_np = spatial.KDTree(src_points_np[batch_idx,:,:]).query(interp_points_np[batch_idx,:,:])[1]
        indices[batch_idx, :] = torch.tensor(indices_np, dtype=torch.long, device=src_points.device)
            
    return indices

##

def create_template_seg(P_phase, save=False):
    from data_loader import fetch_dataloader_test
    
    exp_dir = '../experiment_results/10_bspline/order_2'
    params = load_params(exp_dir)
    data_dir = '../../data/ct_npy/npy_combined_full_train'
    label_dir = '../../data/valve_seg/npy_sep_valves'
    test_dl, test_ds = fetch_dataloader_test(params, data_dir, label_dir=label_dir, return_ds=True)
    
    P_phase_list = sorted([os.path.splitext(f)[0] for f in os.listdir(data_dir)], key=natural_key)
    idx = P_phase_list.index(P_phase)
    
    _, labels = test_ds[idx]
    
    if save:
        np.save(os.path.join('../template_for_deform', '{}_labels.npy'.format(P_phase)), labels)
    else:
        return labels

def crop_img_seg_64x64x64(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir, save=False):
    from data_loader import fetch_dataloader_test
    
    if not os.path.exists(dst_img_dir):
        os.makedirs(dst_img_dir)
        
    if not os.path.exists(dst_label_dir):
        os.makedirs(dst_label_dir)
    
    aortic_root_dir = os.path.join(dst_label_dir, 'aortic_root')
    valve_1_dir = os.path.join(dst_label_dir, 'valve_1')
    valve_2_dir = os.path.join(dst_label_dir, 'valve_2')
    valve_3_dir = os.path.join(dst_label_dir, 'valve_3')
            
    if not os.path.exists(aortic_root_dir):
        os.makedirs(aortic_root_dir)
    if not os.path.exists(valve_1_dir):
        os.makedirs(valve_1_dir)
    if not os.path.exists(valve_2_dir):
        os.makedirs(valve_2_dir)    
    if not os.path.exists(valve_3_dir):
        os.makedirs(valve_3_dir)
    
    exp_dir = '../experiment_results/10_bspline/order_2' # important to get AlignRoateCrop_all transform
    params = load_params(exp_dir)
    test_dl, test_ds = fetch_dataloader_test(params, src_img_dir, label_dir=src_label_dir, return_ds=True)
    
    P_phase_list = sorted([os.path.splitext(f)[0] for f in os.listdir(src_img_dir)], key=natural_key)
    
    img_list = []
    labels_list = []
    
    for idx, P_phase in enumerate(P_phase_list):
        img, labels = test_ds[idx]
        
        img = img.squeeze().cpu().numpy()
        labels = labels.squeeze().cpu().numpy()
        
        print('{} check'.format(P_phase))
        
        if save:
            np.save(os.path.join(dst_img_dir, '{}.npy'.format(P_phase)), img)
            
            nrrd.write(os.path.join(aortic_root_dir, '{}_aortic_root_seg.nrrd'.format(P_phase)), labels[0,:,:,:])
            nrrd.write(os.path.join(valve_1_dir, '{}_valve_1_label.nrrd'.format(P_phase)), labels[1,:,:,:])
            nrrd.write(os.path.join(valve_2_dir, '{}_valve_2_label.nrrd'.format(P_phase)), labels[2,:,:,:])
            nrrd.write(os.path.join(valve_3_dir, '{}_valve_3_label.nrrd'.format(P_phase)), labels[3,:,:,:])
        else:
            img_list.append(img)
            labels_list.append(labels)
    
    if not save:
        return img_list, labels_list
    
##

def load_skel_file(skel_path, only_point_coordinates=True):
    '''
    output: 4 np arrays
    '''
    
    '''
    From looking at source code from Deep Points Consolidation paper
    
    ON: original (!)
    SN: samples (!)
    GropuID: ?
    Confidence_Sigma samples.vert[i].eigen_confidence
    Sample_isVirtual: ?
    Samples_isBranch: ?
    DSN: inner_points (!)
    dual_corresponding_index: samples.vert[i].inner_index
    skel_radius: samples.vert[i].skel_radius
    SkelPN = skel_points (!)
    
    key_list = ['ON', 'SN', 'DSN', 'SkelPN'] # just for memo, not for use
    '''
    
    ON_entry_list = []
    SN_entry_list = []
    DSN_entry_list = []
    SkelPN_entry_list = []
    
    store_ON = False
    store_SN = False
    store_DSN = False
    store_SkelPN = False
    
    with open(skel_path, 'r') as fstream:
        for line_idx, line in enumerate(fstream):
            
            line = line.rstrip('\n')
            line_parse = line.split(' ')
            
            if line_parse[0] == 'ON':
                n_pts = int(line_parse[1])
                idx = 0
                store_ON = True
                
            if line_parse[0] == 'SN':
                n_pts = int(line_parse[1])
                idx = 0
                store_SN = True
                
            if line_parse[0] == 'DSN':
                n_pts = int(line_parse[1])
                idx = 0
                store_DSN = True
                
            if line_parse[0] == 'SkelPN':
                n_pts = int(line_parse[1])
                idx = 0
                store_SkelPN = True
                
            if store_ON:
                if idx > 0:
                    ON_entry_list.append([float(num) for num in line.split('\t')[:-1]])
                idx += 1
                if idx > n_pts:
                    store_ON = False
                    idx -= 1
                    
            if store_SN:
                if idx > 0:
                    SN_entry_list.append([float(num) for num in line.split('\t')[:-1]])
                idx += 1
                if idx > n_pts:
                    store_SN = False
                    idx -= 1
                
            if store_DSN:
                if idx > 0:
                    DSN_entry_list.append([float(num) for num in line.split('\t')[:-1]])
                idx += 1
                if idx > n_pts:
                    store_DSN = False
                    idx -= 1
                    
            if store_SkelPN:
                if idx > 0:
                    SkelPN_entry_list.append([float(num) for num in line.split('\t')[:-1]])
                idx += 1
                if idx > n_pts:
                    store_SkelPN = False
                    idx -= 1
    
    ''' Checking if loaded properly '''
#    entry_lists = [ON_entry_list, ON_entry_list, DSN_entry_list, SkelPN_entry_list]
#   
#    for entry_list in entry_lists:
#        print(len(entry_list))
#        
#        for idx, e in enumerate(entry_list):
#            if len(e) != 6:
#                print(idx)
#                print('NOOO')
    
    if only_point_coordinates:
        return np.array(ON_entry_list)[:,0:3], np.array(SN_entry_list)[:,0:3], np.array(DSN_entry_list)[:,0:3], np.array(SkelPN_entry_list)[:,0:3]
    else:
        return np.array(ON_entry_list), np.array(SN_entry_list), np.array(DSN_entry_list), np.array(SkelPN_entry_list)
    
def convert_verts_array_for_ply(verts, normals=None):
    '''
    verts: np.array [n_verts, n_dim]
    '''
    if normals is not None:
        dtype = [('x', 'f4'),
                 ('y', 'f4'),
                 ('z', 'f4'),
                 ('nx', 'f4'),
                 ('ny', 'f4'),
                 ('nz', 'f4'),
                 ('red', 'u1'),
                 ('green', 'u1'),
                 ('blue', 'u1'),
                 ('alpha', 'u1')]
        
        list_of_tuples = []
        for vert, normal in zip(verts.tolist(), normals.tolist()):
            list_of_tuples.append(tuple(vert + normal + [255, 255, 255, 255]))
        
        verts_ply = np.array(list_of_tuples, dtype=dtype)
        
    else:   
        dtype = [('x', 'f4'),
                 ('y', 'f4'),
                 ('z', 'f4')]
    
        list_of_tuples = []
        for vert in verts.tolist():
            list_of_tuples.append(tuple(vert))
        
        verts_ply = np.array(list_of_tuples, dtype=dtype)
    
    return verts_ply

def convert_faces_array_for_ply(faces):
    '''
    faces: np.array [n_faces, n_dim]
    '''
    dtype = [('vertex_indices', 'i4', 3)]
    
    list_of_tuples = []
    for face in faces.tolist():
        list_of_tuples.append((face,))
    
    faces_ply = np.array(list_of_tuples, dtype=dtype)
    
    return faces_ply    

def convert_to_ply(verts, normals=None, faces=None, save_filepath=None):
    '''
    verts_normals: np.array [n_points, 3] or [n_points, 6] (if there are normals)
    faces: np.array [n_faces, n_connections] or None
    '''
    from plyfile import PlyData, PlyElement
    verts_ply = convert_verts_array_for_ply(verts, normals=normals)
    
    if faces is not None:
        faces_ply = convert_faces_array_for_ply(faces)
    else:
        faces_ply = np.array(([]), dtype=[('vertex_indices', 'i4', (3,))])
        
    el1 = PlyElement.describe(verts_ply, 'vertex')
    el2 = PlyElement.describe(faces_ply, 'face')
    
    plydata = PlyData([el1, el2], text=True, byte_order='=')
    
    if save_filepath is None:
        return plydata
    else:
        if os.path.splitext(save_filepath)[1] != '.ply':
            save_filepath = '{}{}'.format(save_filepath, '.ply')
            
        plydata.write(save_filepath)

def ply_flip_normals(dpoints_dir):
    from plyfile import PlyData
    
    ply_filepath_list = sorted([os.path.join(dpoints_dir, f) for f in os.listdir(dpoints_dir) if os.path.splitext(f)[1]=='.ply'], key=natural_key)
    for ply_filepath in ply_filepath_list:
        print(ply_filepath)
        
        plydata = PlyData.read(ply_filepath)
        
        plydata['vertex']['nx'] = -plydata['vertex']['nx']
        plydata['vertex']['ny'] = -plydata['vertex']['ny']
        plydata['vertex']['nz'] = -plydata['vertex']['nz']
        
        plydata.write(ply_filepath)

def load_hypermesh_abaqus_inp_file(filepath):
    store_verts = False
    store_faces_2d = False
    store_faces_3d = False
    
    with open(filepath, 'r') as fstream:
        verts_key_dict = {}
        verts_list = []
        faces_2d_list = []
        faces_3d_list = []
        for line_idx, line in enumerate(fstream):
            
            line = line.rstrip('\n')
            line_parse = line.split('*')
            
            if len(line_parse) > 1:
                if line_parse[1] == 'NODE':
                    store_verts = True
                    count = 0
                    continue
                
                if 'ELEMENT' in line_parse[1]:
                    if line.split(',')[1] == 'TYPE=S3':
                        store_faces_2d = True
                        n_elem = 3
#                        dict_key = line.split(',')[2].split('=')[1]
                        continue
                
                    elif line.split(',')[1] == 'TYPE=S4R':
                        store_faces_2d = True
                        n_elem = 4
#                        dict_key = line.split(',')[2].split('=')[1]
                        continue

                    elif line.split(',')[1] == 'TYPE=C3D8R':
                        store_faces_3d = True
#                        dict_key = line.split(',')[2].split('=')[1]
                        continue
            
            if store_verts:
                if len(line_parse) == 1:
                    line_parse2 = line.split(',  ')
                    verts_key_dict[int(line_parse2[0])] = count
                    verts_list.append(line_parse2[1:4])
                    count += 1
                else:
                    store_verts = False
            
            if store_faces_2d:
                if len(line_parse) == 1:
                    line_parse2 = line.split(',     ')
                    faces_2d_list.append(line_parse2[1:n_elem+1])
                else:
                    store_faces_2d = False
            
            if store_faces_3d:
                if len(line_parse) == 1:
                    line_parse2 = line.split(',     ')
                    if line_parse2[-1][-1] == ',':
                        line_save = line_parse2[1:] # part of combining this line with next line
                        line_save[-1] = line_save[-1][:-1] # getting rid of comma in last entry
                    if len(line_parse2) == 1:
                        faces_3d_list.append(line_save + [line.split('     ')[1]])
                else:
                    store_faces_3d = False
   
    verts = np.array(verts_list).astype(float)
    faces_2d = np.array(faces_2d_list).astype(int)
    faces_3d = np.array(faces_3d_list).astype(int)
    
    def replace_with_dict(ar, dic):
        # Extract out keys and values
        k = np.array(list(dic.keys()))
        v = np.array(list(dic.values()))
    
        # Get argsort indices
        sidx = k.argsort()
    
        # Drop the magic bomb with searchsorted to get the corresponding
        # places for a in keys (using sorter since a is not necessarily sorted).
        # Then trace it back to original order with indexing into sidx
        # Finally index into values for desired output.
        return v[sidx[np.searchsorted(k,ar,sorter=sidx)]]
    
    faces_2d = replace_with_dict(faces_2d, verts_key_dict)
    faces_3d = replace_with_dict(faces_3d, verts_key_dict)
    
    return verts, faces_2d, faces_3d


def load_liang_meshes(P_phase_dir):
    aortic_wall_dir = os.path.join(P_phase_dir, 'aortic_wall_mesh')
    leaflets_dir = os.path.join(P_phase_dir, 'leaflets_mesh')
    
    for f in os.listdir(aortic_wall_dir):
        if os.path.splitext(f)[1] == '.vtk' and os.path.splitext(os.path.splitext(f)[0])[1] != '.json':
            if 'OutputWall' in f:
                aortic_wall_filepath = os.path.join(aortic_wall_dir, f)
    
    for f in os.listdir(leaflets_dir):
        if os.path.splitext(f)[1] == '.vtk' and os.path.splitext(os.path.splitext(f)[0])[1] != '.json':
            if 'OutputLeaflet_1' in f:
                leaflet1_filepath = os.path.join(leaflets_dir, f)
            if 'OutputLeaflet_2' in f:
                leaflet2_filepath = os.path.join(leaflets_dir, f)
            if 'OutputLeaflet_3' in f:
                leaflet3_filepath = os.path.join(leaflets_dir, f)
    
    m0 = pv.read(aortic_wall_filepath)
    m1 = pv.read(leaflet1_filepath)
    m2 = pv.read(leaflet2_filepath)
    m3 = pv.read(leaflet3_filepath)
    
    verts_list = [m0.points, m1.points, m2.points, m3.points]
    faces_list = [m0.faces.reshape(-1, 5)[:, 1:], \
                  m1.faces.reshape(-1, 5)[:, 1:], \
                  m2.faces.reshape(-1, 5)[:, 1:], \
                  m3.faces.reshape(-1, 5)[:, 1:]]
    
    return verts_list, faces_list

def load_stl_one_component(filepath):
    mesh = pv.read(filepath)
    
    verts_list = [mesh.points]
    faces_list = [mesh.faces.reshape(-1, 4)[:, 1:]]
    
    return verts_list, faces_list

##

# import igl
import utils_cgal_related as utils_cgal

def get_template_verts_faces_list(template_load_fn, template_P_phase):
    extra_info = None

    if template_load_fn is None or template_load_fn == 'marching_cubes':
        template_filepath = os.path.join('../template_for_deform', '{}_seg.npy'.format(template_P_phase))
        seg_template = np.load(template_filepath)
        verts_list_template, faces_list_template = seg_to_mesh(seg_template)
        print('Template: marching cubes | {}'.format(template_P_phase))
    elif template_load_fn == 'liang':
        load_liang = np.load(os.path.join('../template_for_deform', '{}_liang_mesh.npy'.format(template_P_phase)), allow_pickle=True)
        verts_list_template = list(load_liang[0, :])
        faces_list_template = list(load_liang[1, :])
        print("Template: hand-labeled using Liang's module | {}".format(template_P_phase))
    elif template_load_fn == 'hypermesh':
        load_hypermesh = np.load(os.path.join('../template_for_deform', '{}_hypermesh.npy'.format(template_P_phase)), allow_pickle=True)
        verts_list_template = list(load_hypermesh[0, :])
        faces_list_template = list(load_hypermesh[1, :])
        print('Template: hand-labeled using hypermesh | {}'.format(template_P_phase))
    elif template_load_fn == 'one_component':
        verts_list_template, faces_list_template = load_stl_one_component(os.path.join('../template_for_deform', '{}_one_component.stl'.format(template_P_phase)))
        print('Template: one_component stl | {}'.format(template_P_phase))
    elif template_load_fn == 'stitched_with_mesh_corr':
        with open(os.path.join('../template_for_deform', '{}_gt_mesh_combined_64x64x64.pkl'.format(template_P_phase)), 'rb') as f:
            all_verts_64x64x64, all_faces, idx_tracks, verts_len_list, faces_len_list = pickle.load(f)

        verts_list_template = [all_verts_64x64x64]

        len_faces = np.array([len(face) for face in all_faces])
        if (len_faces != 3).sum() > 0:
            # convert all_faces to triangular
            all_tri_faces = utils_cgal.split_quad_to_2_tri_mesh(all_faces)
            # update faces_len_list to match converted triangular faces
            _, faces_sep_list = utils_cgal.get_sep_components_from_all_verts_all_faces(all_verts_64x64x64, all_faces, idx_tracks, verts_len_list, faces_len_list)
            faces_len_list = [len(utils_cgal.split_quad_to_2_tri_mesh(faces_sep)) for faces_sep in faces_sep_list]
        else:
            all_tri_faces = all_faces

        faces_list_template = [all_tri_faces]
        extra_info = (idx_tracks, verts_len_list, faces_len_list)

        # verts_new_list, faces_new_list = utils_cgal.get_sep_components_from_all_verts_all_faces_torch(all_verts_64x64x64, all_faces, idx_tracks, verts_len_list, faces_len_list)
        # faces_new_tri_list = [utils_cgal.split_quad_to_2_tri_mesh(faces) for faces in faces_new_list]
        # faces_new_tri_list.append(faces_new_tri_list.pop(1)) # need change order of leaflets to match gt_pcl ordering

        # verts_list_template = verts_new_list
        # faces_list_template = faces_new_tri_list
        print('Template: stitched_with_mesh_corr | {}'.format(template_P_phase))
    elif template_load_fn == 'cad_open_valve_base_surf':
        # matters for mtm vs gcn (mtm shouldn't be able to handle closed_valve template)
        mesh_3d_pv = pv.read('../template_for_deform/CAD_open_valve_vol_mesh_64x64x64.vtk')
        with open('../template_for_deform/CAD_open_valve_base_surf_faces_tri.pkl', 'rb') as f:
            base_surf_faces_tri = pickle.load(f)

        verts, elems = utils_cgal.get_verts_faces_from_pyvista(mesh_3d_pv)
        verts_list_template = [verts]
        faces_list_template = [base_surf_faces_tri]
        extra_info = elems

    elif template_load_fn == 'cad_open_valve_base_surf_sep':
        # matters for gcn (trains much better for closed valve if separate target gt_pcl)
        mesh_3d_pv = pv.read('../template_for_deform/CAD_open_valve_vol_mesh_64x64x64.vtk')
        with open('../template_for_deform/CAD_open_valve_sep_faces_list.pkl', 'rb') as f:
            faces_sep_list = pickle.load(f)

        verts, elems = utils_cgal.get_verts_faces_from_pyvista(mesh_3d_pv)
        verts_list_template = [verts]
        faces_list_template = [utils_cgal.get_verts_faces_from_pyvista(mesh_to_pyvista_PolyData(verts, faces).triangulate())[1] for faces in faces_sep_list[:4]]
        
        extra_info = elems

    elif template_load_fn == 'cad_closed_valve_base_surf':
        # matters for mtm vs gcn (mtm shouldn't be able to handle closed_valve template)
        mesh_3d_pv = pv.read('../template_for_deform/CAD_closed_valve_vol_mesh_64x64x64.vtk')
        with open('../template_for_deform/CAD_closed_valve_base_surf_faces_tri.pkl', 'rb') as f:
            base_surf_faces_tri = pickle.load(f)

        verts, elems = utils_cgal.get_verts_faces_from_pyvista(mesh_3d_pv)
        verts_list_template = [verts]
        faces_list_template = [base_surf_faces_tri]
        extra_info = elems
    elif template_load_fn == 'cad_closed_valve_base_surf_sep':
        # matters for gcn (trains much better for closed valve if separate target gt_pcl)
        mesh_3d_pv = pv.read('../template_for_deform/CAD_closed_valve_vol_mesh_64x64x64.vtk')
        with open('../template_for_deform/CAD_closed_valve_sep_faces_list.pkl', 'rb') as f:
            faces_sep_list = pickle.load(f)

        verts, elems = utils_cgal.get_verts_faces_from_pyvista(mesh_3d_pv)
        verts_list_template = [verts]
        faces_list_template = [utils_cgal.get_verts_faces_from_pyvista(mesh_to_pyvista_PolyData(verts, faces).triangulate())[1] for faces in faces_sep_list[:4]]
        
        extra_info = elems

    else:
        raise ValueError('wrong params.template_load_fn')

    return verts_list_template, faces_list_template, extra_info

##

class DimensionConverter():
    '''
    going to and from [0, dim_size] and [-1, 1]
    '''
    def __init__(self, dim_size):
        self.dim_size = dim_size - 1 # minus 1 here b/c if img has size 64, we want [0,63] range
    
    def to_dim_size(self, x):
        m = self.dim_size/2
        b = self.dim_size/2
        
        y = m*x + b
        
        return y
    
    def from_dim_size(self, x):
        m = 2/self.dim_size
        b = -1
        
        y = m*x + b
        
        return y

##

def get_aligned_verts(verts, P_phase):
    # try:
    #     src_shape = np.load('../../data/ct_npy/npy_combined_full_train_64x64x64/{}.npy'.format(P_phase)).shape
    # except:
    #     src_shape = np.load('../../data/ct_npy/npy_combined_full_test_64x64x64/{}.npy'.format(P_phase)).shape
    
    #DPDPDP get rid of this later and uncomment above
    src_shape = [64,64,64]
    
    landmarks_filepath = '../../code_general/gui_DP/save_landmarks/all_landmark_info.pkl'
    with open(landmarks_filepath, 'rb') as f:
        landmarks_all = pickle.load(f)
    
    landmarks = np.vstack(tuple(landmarks_all[P_phase].values()))
    
    return align_mesh_to_cropped(verts, landmarks, src_shape)

def align_mesh_to_cropped(verts, landmarks, src_shape, dst_shape=[64,64,64]):
    from data_transforms import AlignRotateTranslateCrop3D_all
    import transformations as tfm

    align_fn = AlignRotateTranslateCrop3D_all(net_input_size=dst_shape)
    
    align_fn.original_image_size = src_shape
    align_fn.initialize_interp_points()
    
    lm1 = landmarks[0,:]
    lm2 = landmarks[1,:]
    lm3 = landmarks[2,:]
    
    circumcenter = align_fn.calc_cicrumcenter(lm1, lm2, lm3)
    (x,y,z,d) = align_fn.calc_plane_params(lm1, lm2, lm3)
    
    rotx = np.arctan2( y, z );
    if z >= 0:
       roty = -np.arctan2( x * np.cos(rotx), z );
    else:
       roty = np.arctan2( x * np.cos(rotx), -z );
    
    rotz = 0
    
    rotation_center = np.asarray(src_shape)/2
    rot_mat1 = tfm.rotation_matrix(rotx, [-1,0,0], point=rotation_center)
    rot_mat2 = tfm.rotation_matrix(roty, [0,-1,0], point=rotation_center)
    rot_mat3 = tfm.rotation_matrix(rotz, [x,y,z], point=circumcenter)
    
    rot_mat123 = np.dot(rot_mat3, np.dot(rot_mat2, rot_mat1))
    
    # calculating angle_to_align_vertical_circumcenter_lm1
    circumcenter_proj = utils_gui.calc_2d_coordinate_on_plane(circumcenter, align_fn.plane_origin, align_fn.ortho1, align_fn.ortho2, rot_mat123)
    lm1_proj = utils_gui.calc_2d_coordinate_on_plane(lm1, align_fn.plane_origin, align_fn.ortho1, align_fn.ortho2, rot_mat123)
    vec_cc_lm1 = lm1_proj - circumcenter_proj
    angle_cc_lm1 = np.arctan2(vec_cc_lm1[0], vec_cc_lm1[1])
    rotz = angle_cc_lm1
    
    rot_mat4 = tfm.rotation_matrix(rotz, [x,y,z], point=circumcenter)
    rot_mat = np.dot(rot_mat4, rot_mat123) # only rotation for getting the plane, still need to translate to match circumcenter
    
    verts_transformed = np.dot(np.array([[-1,0,0],
                                          [0,-1,0],
                                          [0,0,1]]), verts.T).T
    
    verts_rotated = np.dot(np.linalg.inv(rot_mat[0:3, 0:3]), verts_transformed.T).T
    translate = np.dot(np.linalg.inv(rot_mat[0:3,0:3]), circumcenter.T).T - (np.array([dst_shape])-1)/2
    verts_transformed = verts_rotated - translate
    
    # set_trace()
    
    return verts_transformed

##

def get_node_degree(verts, faces):
    meshes = Meshes(verts, faces)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
    V = verts_packed.shape[0]  # sum(V_n)

    e0, e1 = edges_packed.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (sum(E_n), 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (sum(E_n), 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*sum(E_n))

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=edges_packed.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    deg = torch.sparse.sum(A, dim=1).to_dense()

    return deg

def get_edges_pytorch3d(verts, faces):
    """
    verts, faces in what format?
    """
    meshes = mesh_to_pytorch3d_Mesh([verts], [faces])
    edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
    return edges_packed

def get_edges_pyvista(mesh_pv):
    edges = mesh_pv.extract_all_edges().lines
    return edges.reshape(-1,3)[:,1:]

# face_to_edge_DP is inaccurate (gets edges duplicated inconsistent amount of times)
def face_to_edge_DP(faces):
    '''
    faces: torch.tensor [n_faces, n_verts_per_face]
    '''
    if not torch.is_tensor(faces):
        faces = torch.tensor(faces)

    faces = faces.squeeze()

    if faces.shape[1] == 3:
        edges = torch.cat([faces[:,[0,1]],
                           faces[:,[1,2]],
                           faces[:,[2,0]]], dim=0)
    elif faces.shape[1] == 4:
        edges = torch.cat([faces[:,[0,1]],
                           faces[:,[1,2]],
                           faces[:,[2,3]],
                           faces[:,[3,0]]], dim=0)

    edges = edges.type(torch.LongTensor)

    if faces.is_cuda:
        edges = edges.cuda()
    else:
        edges = edges.cpu()

    return edges

##

from scipy.spatial import cKDTree

def get_close_enough_verts(verts, cp_grid, dist_threshold=1, img_dim=64, return_all=False):
    '''
    verts: torch.tensor [V, Dim]
    cp_grid: torch.tensor [Dim, H, W, D], displacement of control points
    '''
    if torch.is_tensor(verts):
        verts = verts.squeeze().detach().cpu().numpy()
    if torch.is_tensor(cp_grid):
        cp_grid = cp_grid.detach().cpu().numpy()
    
    # using np.meshgrid here because utils.seg_to_mesh is also in numpy format
    mesh_vals = np.linspace(0, img_dim, cp_grid.shape[2])
    mesh_grid = np.stack(np.meshgrid(mesh_vals, mesh_vals, mesh_vals))
    
    cp_grid_pts = cp_grid.squeeze().reshape(3,-1).transpose(1,0)
    mesh_grid_pts = mesh_grid.reshape(3,-1).transpose(1,0)
    
    point_tree = cKDTree(verts)
    
    all_verts_idxes_list = point_tree.query_ball_point(cp_grid_pts + mesh_grid_pts, dist_threshold)
    
    close_verts_idxes_list = []
    linear_idx_mesh_cp_grid_list = []
    close_verts_list = []
    for linear_idx_mesh_cp_grid, verts_idxes in enumerate(all_verts_idxes_list):
        if len(verts_idxes)>0:
            # if len > 0, then vert(s) are close            
            close_verts_idxes_list.append(verts_idxes)
            linear_idx_mesh_cp_grid_list.append(linear_idx_mesh_cp_grid)
            close_verts_list.append(point_tree.data[verts_idxes])
            
    if return_all:
        return close_verts_idxes_list, linear_idx_mesh_cp_grid_list, close_verts_list
    else:
        return close_verts_idxes_list, linear_idx_mesh_cp_grid_list

##

def get_one_entry(seg_output, transformed_verts_list, displacement_field_tuple):
    if isinstance(seg_output, list):
        seg_output = seg_output[-1]
    if not torch.is_tensor(transformed_verts_list[0]):
        transformed_verts_list = transformed_verts_list[-1]
    if isinstance(displacement_field_tuple, list):
        displacement_field_tuple = displacement_field_tuple[-1]
    
    return seg_output, transformed_verts_list, displacement_field_tuple

##

def get_img_label_64x64x64(P_phase, data_dir):
    import torchvision.transforms as transforms
    import custom_dataset_classes as cdc
    
    label_dir = '../../data/valve_seg/npy_sep_valves_64x64x64'
    transform = transforms.Compose([])
    ds = cdc.CT_just_load_Dataset(data_dir, label_dir, transform)
    
    P_phase_list = [os.path.splitext(os.path.basename(f))[0] for f in ds.ct_filepaths]
    idx = P_phase_list.index(P_phase)
    
    img, label = ds[idx]
    
    return img.squeeze().numpy(), label.squeeze().numpy()

##

def check_overlap_between_patient_files(dir1, dir2):
    d1 = [f for f in os.listdir(dir1) if 'phase' in f and 'take_out' not in f]
    d1 = sorted(d1, key=natural_key)
    
    P1 = [d.split('_')[0] for d in d1]
    
    d2 = [f for f in os.listdir(dir2) if 'phase' in f and 'take_out' not in f]
    d2 = sorted(d2, key=natural_key)
    
    P2 = [d.split('_')[0] for d in d2]
    
    test = [i for i in P1 if i in P2]
    print(test)
    
    print('{} P_phase files in dir1'.format(len(d1)))
    print('{} P_phase files in dir2'.format(len(d2)))
    
def choose_10000_pts_random(pcl_list, n_desired_pts = 10000):
    pcl_chosen_list = []
    for pcl in pcl_list:
        n_pts = pcl.shape[0]
        
        rand_idx = np.random.choice(np.arange(n_pts), size=n_desired_pts, replace=False)
        
        pcl_chosen = pcl[rand_idx,:]
        pcl_chosen_list.append(pcl_chosen)
    
    return pcl_chosen_list

def load_vtk_mesh(vtk_filepath):
    mesh_pv = pv.read(vtk_filepath)
    
    verts = mesh_pv.points
    faces = mesh_pv.faces.reshape(-1, 4)[:, 1:]
    
    return verts, faces

##

import transformations as tfm
import utils_gui

def calc_cicrumcenter(lm1, lm2, lm3):
    '''
    calculate the circumcenter (point equidistance from a, b, and c)
    lm1, lm2, lm3 are the three 3D coordinates of landmark points
    '''
    ac = lm3-lm1
    ab = lm2-lm1
    abXac = np.cross(ab, ac)
    
    to_circumsphere_center = (np.cross(abXac, ab)*np.linalg.norm(ac)**2 + np.cross(ac, abXac)*np.linalg.norm(ac)**2)/(2*np.linalg.norm(abXac)**2)
    
    circumcenter = lm1 + to_circumsphere_center
    
    return circumcenter

def calc_plane_params(lm1, lm2, lm3):
    x1 = lm1[0]
    y1 = lm1[1]
    z1 = lm1[2]
    
    x2 = lm2[0]
    y2 = lm2[1]
    z2 = lm2[2]
    
    x3 = lm3[0]
    y3 = lm3[1]
    z3 = lm3[2]
    
    vector1 = [x2 - x1, y2 - y1, z2 - z1]
    vector2 = [x3 - x1, y3 - y1, z3 - z1]
    
    cross_product = [vector1[1] * vector2[2] - vector1[2] * vector2[1], -1 * (vector1[0] * vector2[2] - vector1[2] * vector2[0]), vector1[0] * vector2[1] - vector1[1] * vector2[0]]
    
    a = cross_product[0]
    b = cross_product[1]
    c = cross_product[2]
    d = - (cross_product[0] * x1 + cross_product[1] * y1 + cross_product[2] * z1)
    
    return a, b, c, d

def get_rot_mat_match_liang_mesh_to_64x64x64(P_phase):
    try:
        ct_data = np.load('../../data/ct_npy/npy_combined_full_test_combined/{}.npy'.format(P_phase))
    except:
        ct_data = np.load('../../data/ct_npy/npy_combined_full_train/{}.npy'.format(P_phase))
        
    landmarks_filepath = '../../code_general/gui_DP/save_landmarks/all_landmark_info.pkl'
    
    with open(landmarks_filepath, 'rb') as f:
        landmarks = pickle.load(f)
    
    lm1 = landmarks[P_phase]['landmark1']
    lm2 = landmarks[P_phase]['landmark2']
    lm3 = landmarks[P_phase]['landmark3']
    circumcenter = calc_cicrumcenter(lm1, lm2, lm3)
    (x,y,z,d) = calc_plane_params(lm1, lm2, lm3)
    
    rotx = np.arctan2( y, z );
    if z >= 0:
       roty = -np.arctan2( x * np.cos(rotx), z );
    else:
       roty = np.arctan2( x * np.cos(rotx), -z );
    
    rotz = 0
    
    rotation_center = np.asarray(ct_data.shape)/2
    rot_mat1 = tfm.rotation_matrix(rotx, [-1,0,0], point=rotation_center)
    rot_mat2 = tfm.rotation_matrix(roty, [0,-1,0], point=rotation_center)
    rot_mat3 = tfm.rotation_matrix(rotz, [x,y,z], point=circumcenter)
    
    rot_mat123 = np.dot(rot_mat3, np.dot(rot_mat2, rot_mat1))
    
    plane_origin = [ct_data.shape[0], 0, ct_data.shape[2]/2]
    ortho1 = [-1,0,0]
    ortho2 = [0,1,0]
    
    # calculating angle_to_align_vertical_circumcenter_lm1
    circumcenter_proj = utils_gui.calc_2d_coordinate_on_plane(circumcenter, plane_origin, ortho1, ortho2, rot_mat123)
    lm1_proj = utils_gui.calc_2d_coordinate_on_plane(lm1, plane_origin, ortho1, ortho2, rot_mat123)
    vec_cc_lm1 = lm1_proj - circumcenter_proj
    angle_cc_lm1 = np.arctan2(vec_cc_lm1[0], vec_cc_lm1[1])
    rotz = angle_cc_lm1
    
    rot_mat4 = tfm.rotation_matrix(rotz, [x,y,z], point=circumcenter)
    rot_mat = np.dot(rot_mat4, rot_mat123)
    
    normal = np.dot(rot_mat[0:3, 0:3], np.array([0,0,1]))
    normal = normal/np.linalg.norm(normal)
    
    x_overshoot = np.arange(round(np.sqrt(ct_data.shape[2]**2 + ct_data.shape[0]**2))+1)
    y_overshoot = np.arange(round(np.sqrt(ct_data.shape[2]**2 + ct_data.shape[1]**2))+1)
    z_initial = ct_data.shape[2]/2
    
    x_mesh, y_mesh = np.meshgrid(x_overshoot, y_overshoot)
    x_mesh = x_mesh.T
    y_mesh = y_mesh.T
    z_mesh = z_initial*np.ones(x_mesh.shape)
        
    xyz_orig = np.concatenate((np.expand_dims(x_mesh, 2),
                               np.expand_dims(y_mesh, 2),
                               np.expand_dims(z_mesh, 2)), axis=2)
    
    xyz_orig_homogeneous_coord = np.concatenate((x_mesh.reshape(1,-1),
                                                 y_mesh.reshape(1,-1),
                                                 z_mesh.reshape(1,-1),
                                                 np.ones(x_mesh.reshape(1,-1).shape)), axis=0)
    
    xyz_rotated_homogeneous_coord = np.dot(rot_mat, xyz_orig_homogeneous_coord)
    xyz_rotated = utils_gui.convert_to_orig_shape(xyz_rotated_homogeneous_coord, xyz_orig[:,:,0]) # size (298,298,3), but actually a 2D plane (3 xyz values for each position of intensity)
    
    min_dist_idx = np.unravel_index(np.argmin(np.linalg.norm(xyz_rotated-circumcenter, axis=2)), xyz_rotated.shape[0:2])
    xyz_min_dist = xyz_rotated[min_dist_idx[0], min_dist_idx[1],:]
    
    xyz_rotated_orig = xyz_rotated + (circumcenter - xyz_min_dist)
    
    net_input_size = [64,64,64]
    dists = np.arange(-net_input_size[2]/2,net_input_size[2]/2)+0.5
    
    idxes = [0,63]
    for idx in idxes:
        xyz_rotated = xyz_rotated_orig + normal*dists[idx]
        linear_idx = np.argmin(np.linalg.norm(xyz_rotated-circumcenter, axis=2))
        sub_idx = np.unravel_index(linear_idx, xyz_rotated.shape[0:2])
        
        mid_x = sub_idx[0]
        mid_y = sub_idx[1]
        inc_x = int(net_input_size[0]/2)
        inc_y = int(net_input_size[1]/2)
        
        if idx == 0:
            rgi_idx_bot = xyz_rotated[mid_x-inc_x:mid_x+inc_x,
                                      mid_y-inc_y:mid_y+inc_y,
                                      :]
        elif idx == 63:
            rgi_idx_top = xyz_rotated[mid_x-inc_x:mid_x+inc_x,
                                      mid_y-inc_y:mid_y+inc_y,
                                      :]
    
    mat1 = np.array([[0,64,0,0],
                     [0,0,64,0],
                     [0,0,0,64],
                     [1,1,1,1]])
    
    mat2 = np.ones([4,4])
    mat2[0:3,0] = rgi_idx_bot[0,0,:]
    mat2[0:3,1] = rgi_idx_bot[63,0,:]
    mat2[0:3,2] = rgi_idx_bot[0,63,:]
    mat2[0:3,3] = rgi_idx_top[0,0,:]
    
    rot_mat = np.dot(mat2, np.linalg.inv(mat1))

    # for checking P3_phase4 rot_mat
    # rot_mat = np.array([[0.741422, 0.149688, 0.629984, 71.845],
    #                     [-0.1625, 0.970078, -0.0392344, 112.025],
    #                     [-0.626797, -0.0744531, 0.755359, 63.326],
    #                     [0, 0, 0, 1]])
    
    return rot_mat

def convert_liang_mesh_verts_to_64x64x64(verts_list, P_phase):
    template_rot_mat_64x64x64 = get_rot_mat_match_liang_mesh_to_64x64x64(P_phase)
    transformed_verts_list = []
    for v in verts_list:
        flip_v = np.dot(np.array([[-1, 0, 0, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]]), np.concatenate((v, np.ones([v.shape[0], 1])), axis=1).T)
        transformed_v = np.dot(np.linalg.inv(template_rot_mat_64x64x64), flip_v)
        transformed_verts_list.append(transformed_v.T[:, 0:3])

    return transformed_verts_list

def convert_64x64x64_to_liang_mesh_dim(verts_list, P_phase):
    template_rot_mat_64x64x64 = get_rot_mat_match_liang_mesh_to_64x64x64(P_phase)
    transformed_verts_list = []
    for v in verts_list:
        transformed_v = np.dot(template_rot_mat_64x64x64, np.concatenate((v, np.ones([v.shape[0], 1])), axis=1).T)
        flip_v = np.dot(np.array([[-1, 0, 0, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]]), transformed_v)
        transformed_verts_list.append(flip_v.T[:, 0:3])

    return transformed_verts_list

def reverse_displacement_field_tuple(displacement_field_tuple):
    displacement_field_np = displacement_field_tuple[0].squeeze().permute(1,2,3,0).cpu().numpy()
    inv_displacements_np = get_inverse_displacements_np(displacement_field_np)
    return (torch.tensor(inv_displacements_np, dtype=torch.get_default_dtype(), device=displacement_field_tuple[0].device).permute(3,0,1,2).unsqueeze(0),)

def get_inverse_displacements_np(displacements_np):
    """
    :param displacements_np: np.ndarray of shape [h,w,d,3]
    :return: inv_displacements_np: np.ndarray of shape [h,w,d,3]
    """
    displacements = vtk.vtkImageData()
    displacements.SetDimensions(64,64,64)
    displacements.AllocateScalars(vtk.VTK_FLOAT, 3)
    displacements_pointer = vtk.util.numpy_support.vtk_to_numpy(displacements.GetPointData().GetScalars()).reshape([64,64,64,3])
    displacements_pointer[:] = displacements_np

    grid_t = vtk.vtkGridTransform()
    grid_t.SetDisplacementGridData(displacements)
    grid_t.SetInterpolationModeToLinear()
    inv_grid_t = grid_t.GetInverse()

    tform_to_grid = vtk.vtkTransformToGrid()
    tform_to_grid.SetInput(inv_grid_t)
    tform_to_grid.SetGridOrigin(0.0, 0.0, 0.0)
    tform_to_grid.SetGridExtent(0, 63, 0, 63, 0, 63)
    tform_to_grid.SetGridSpacing(1.0, 1.0, 1.0)
    tform_to_grid.Update()

    inv_displacements_np = vtk_to_numpy(tform_to_grid.GetOutput().GetPointData().GetScalars()).reshape([64,64,64,3])
    return inv_displacements_np
