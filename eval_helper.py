# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 23:40:10 2018

@author: Daniel
"""

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import pickle
# from sklearn.metrics import confusion_matrix
from scipy import sparse
from pdb import set_trace

import utils_sp
import utils_cgal_related as utils_cgal

import sys
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss
from pytorch3d.ops import sample_points_from_meshes
# from chamfer_DP import chamfer_distance_asymmetric, chamfer_distance_eval
import pytorch3d
from chamfer_DP2 import chamfer_distance_eval

from pytorch3d.loss import chamfer_distance

import pyvista as pv

class DiceAndSmoothness():
    def __init__(self, lambdas=[0, 1e-6, 1e-4], squared_cardinality_den=False, eps=1e-6):
        self.lambdas = lambdas
        self.squared_cardinality_den = squared_cardinality_den
        self.eps = eps
    
    def __call__(self, pred, target):
        ''' 
        pred = (torch.Tensor, tuple of torch tensors)
        target = torch.Tensor (just seg)
        
        seg shape = [n_batch, n_channels, H, W, D]
        '''
        
        seg_output = pred[0]
        displacement_field_tuple = pred[1]
        
        assert seg_output.shape[1] == target.shape[1], 'n_channels of network output should be same as n_channels of labels'
        
        dice_store = torch.zeros(target.shape[1], device=seg_output.device)
        
        for i in range(target.shape[1]):
            dice_store[i] = torch.neg(negative_soft_dice_binary_torch(seg_output[:,i,:,:,:], target[:,i,:,:,:], squared_cardinality_den=self.squared_cardinality_den, eps=self.eps))
        
        dice_mean = dice_store.mean()
        neg_dice = torch.neg(dice_mean)
        
        deriv1_mag, grad_output = get_gradient_displacement_field(displacement_field_tuple)
        deriv2_mag, deriv_output = get_squared_2nd_deriv_displacement_field(displacement_field_tuple)
        mag_displacement_field = get_mag_displacement_field(displacement_field_tuple)
        
        print('list of dices for each component')
        print(torch.neg(dice_store).tolist())
        print('summed dice')
        print(neg_dice.item())
        print('deriv2_mag')
        print(deriv2_mag.item())
        print('mag_displacement_field')
        print(mag_displacement_field.item())
        
        '''
        lambda for grad: 1e-5
        lambda for 2nd_order_deriv: 1e-6
        lambda for mag: 1e-4
        '''
        return neg_dice + self.lambdas[0] * deriv1_mag + self.lambdas[1] * deriv2_mag + self.lambdas[2] * mag_displacement_field
    
class DiceAndDiffeomorphic():
    def __init__(self, lambdas=[1e-4], squared_cardinality_den=False, eps=1e-6):
        self.lambdas = lambdas
        self.squared_cardinality_den = squared_cardinality_den
        self.eps = eps
    
    def __call__(self, pred, target):
        ''' 
        pred = (torch.Tensor, tuple of torch tensors)
        target = torch.Tensor (just seg)
        
        seg shape = [n_batch, n_channels, H, W, D]
        '''
        
        seg_output = pred[0]
        displacement_field_tuple = pred[1]
        
        assert seg_output.shape[1] == target.shape[1], 'n_channels of network output should be same as n_channels of labels'
        
        dice_store = torch.zeros(target.shape[1], device=seg_output.device)
        
        for i in range(target.shape[1]):
            dice_store[i] = torch.neg(negative_soft_dice_binary_torch(seg_output[:,i,:,:,:], target[:,i,:,:,:], squared_cardinality_den=self.squared_cardinality_den, eps=self.eps))
        
        dice_mean = dice_store.mean()
        neg_dice = torch.neg(dice_mean)
        
        jacobian_penalty = get_jacobian_penalty(displacement_field_tuple)
        
        print('list of dices for each component')
        print(torch.neg(dice_store))
        print('summed dice')
        print(neg_dice.item())
        print('jacobian_penalty')
        print(jacobian_penalty.item())
        
        '''
        lambda for jacobian_penalty: 1e-4
        '''
        return neg_dice + self.lambdas[0] * jacobian_penalty
    
class DiceDiffeoSmoothRegistration():
    def __init__(self, lambdas=[2.5e-5, 1e-4, 1e-2], squared_cardinality_den=False, eps=1e-6):
        self.lambdas = lambdas
        self.squared_cardinality_den = squared_cardinality_den
        self.eps = eps
    
    def __call__(self, pred, target):
        ''' 
        pred = ([torch.Tensor, torch.Tensor, tuple of torch tensors], torch.Tensor)
        target = torch.Tensor (just seg)
        
        seg shape = [n_batch, n_channels, H, W, D]
        '''
        
        seg_output = pred[0][0]
        img_output = pred[0][1]
        displacement_field_tuple = pred[0][2]
        img_input = pred[1]
        
        assert seg_output.shape[1] == target.shape[1], 'n_channels of network output should be same as n_channels of labels'
        
        device = seg_output.device
        dice_store = torch.zeros(target.shape[1], device=device)
        
        for i in range(target.shape[1]):
            dice_store[i] = torch.neg(negative_soft_dice_binary_torch(seg_output[:,i,:,:,:], target[:,i,:,:,:], squared_cardinality_den=self.squared_cardinality_den, eps=self.eps))
        
        dice_mean = dice_store.mean()
        neg_dice = torch.neg(dice_mean)
        
        grad_mag, grad_output = get_gradient_displacement_field(displacement_field_tuple)
        deriv_mag, deriv_output = get_squared_2nd_deriv_displacement_field(displacement_field_tuple)
        mag_displacement_field = get_mag_displacement_field(displacement_field_tuple)
        jacobian_penalty = get_jacobian_penalty(displacement_field_tuple)
        
#        registration_loss = get_registration_loss(img_output*seg_output[:,0,:,:,:].unsqueeze(1), img_input*target[:,0,:,:,:].unsqueeze(1))
        registration_loss = get_registration_loss(img_output*target[:,0,:,:,:].unsqueeze(1), img_input*target[:,0,:,:,:].unsqueeze(1))
        
        print('list of dices for each component')
        print(torch.neg(dice_store))
        print('jacobian_penalty')
        print(jacobian_penalty.item())
        print('mag_displacement_field')
        print(mag_displacement_field.item())
        print('registration loss')
        print(registration_loss)
        
        '''
        lambda for jacobian_penalty: 1e-4
        '''
        return neg_dice + self.lambdas[0] * jacobian_penalty + self.lambdas[1] * mag_displacement_field + self.lambdas[2] * registration_loss
    
class DiceDiffeoSmoothLaplacian():
    def __init__(self, lambdas=[2.5e-5, 1e-4, 1], squared_cardinality_den=False, eps=1e-6):
        self.lambdas = lambdas
        self.squared_cardinality_den = squared_cardinality_den
        self.eps = eps
    
    def __call__(self, pred, target):
        ''' 
        pred = ([torch.Tensor, list of torch.Tensor, tuple of torch tensors], [list of torch tensors, list of torch tensors])
        target = torch.Tensor (just seg)
        
        seg shape = [n_batch, n_channels, H, W, D]
        '''
        
        seg_output = pred[0][0]
        transformed_verts_list = pred[0][1]
        displacement_field_tuple = pred[0][2]
        
        template_verts_list = pred[1][0]
        template_faces_list = pred[1][1]
        
        assert seg_output.shape[1] == target.shape[1], 'n_channels of network output should be same as n_channels of labels'
        
        device = seg_output.device
        dice_store = torch.zeros(target.shape[1], device=device)
        
        for i in range(target.shape[1]):
            dice_store[i] = torch.neg(negative_soft_dice_binary_torch(seg_output[:,i,:,:,:], target[:,i,:,:,:], squared_cardinality_den=self.squared_cardinality_den, eps=self.eps))
        
        dice_mean = dice_store.mean()
        neg_dice = torch.neg(dice_mean)
        
#        grad_mag, grad_output = get_gradient_displacement_field(displacement_field_tuple)
#        deriv_mag, deriv_output = get_squared_2nd_deriv_displacement_field(displacement_field_tuple)
        mag_displacement_field = get_mag_displacement_field(displacement_field_tuple)
        jacobian_penalty = get_jacobian_penalty(displacement_field_tuple)
        
        laplacian_loss = get_laplacian_loss_total(transformed_verts_list, template_verts_list, template_faces_list)
        
        print('list of dices for each component')
        print(torch.neg(dice_store))
        print('jacobian_penalty')
        print(jacobian_penalty.item())
        print('mag_displacement_field')
        print(mag_displacement_field.item())
        print('laplacian')
        print(laplacian_loss.item())
        
        '''
        lambda for jacobian_penalty: 1e-4
        '''
        return neg_dice + self.lambdas[0] * jacobian_penalty + self.lambdas[1] * mag_displacement_field + self.lambdas[2] * laplacian_loss

class DiceEdge():
    def __init__(self, lambdas=[3], squared_cardinality_den=False, eps=1e-6):
        self.lambdas = lambdas
        self.squared_cardinality_den = squared_cardinality_den
        self.eps = eps
    
    def __call__(self, pred, target):
        ''' 
        pred = ([torch.Tensor, list of torch.Tensor, tuple of torch tensors], [list of torch tensors, list of torch tensors])
        target = torch.Tensor (just seg)
        
        seg shape = [n_batch, n_channels, H, W, D]
        '''
        
        seg_output = pred[0][0]
        transformed_verts_list = pred[0][1]
        displacement_field_tuple = pred[0][2]
        
        template_verts_list = pred[1][0]
        template_faces_list = pred[1][1]
        
        assert seg_output.shape[1] == target.shape[1], 'n_channels of network output should be same as n_channels of labels'
        
        device = seg_output.device
        dice_store = torch.zeros(target.shape[1], device=device)
        
        for i in range(target.shape[1]):
            dice_store[i] = torch.neg(negative_soft_dice_binary_torch(seg_output[:,i,:,:,:], target[:,i,:,:,:], squared_cardinality_den=self.squared_cardinality_den, eps=self.eps))
        
        dice_mean = dice_store.mean()
        neg_dice = torch.neg(dice_mean)
        
        mag_displacement_field = get_mag_displacement_field(displacement_field_tuple)
        jacobian_penalty = get_jacobian_penalty(displacement_field_tuple)
        laplacian_loss = get_laplacian_loss_total(transformed_verts_list, template_verts_list, template_faces_list)
        
        # TODO: this only works for batch size 1.. consider using padded inputs
        template_verts_list_squeezed = []
        transformed_verts_list_squeezed = []
        template_faces_list_squeezed = []
        for v_temp, v_trans, f in zip(template_verts_list, transformed_verts_list, template_faces_list):
            template_verts_list_squeezed.append(v_temp.squeeze())
            transformed_verts_list_squeezed.append(v_trans.squeeze())
            template_faces_list_squeezed.append(f.squeeze())
            
        template_mesh = Meshes(verts=template_verts_list_squeezed, faces=template_faces_list_squeezed)
        transformed_mesh = Meshes(verts=transformed_verts_list_squeezed, faces=template_faces_list_squeezed)
        edge_loss = get_mesh_correspondence_edge_loss(template_mesh, transformed_mesh)
        
        print('list of dices for each component')
        print(torch.neg(dice_store))
        print('jacobian_penalty')
        print(jacobian_penalty.item())
        print('mag_displacement_field')
        print(mag_displacement_field.item())
        print('laplacian')
        print(laplacian_loss.item())
        print('edge loss')
        print(edge_loss.item())
        print(' ')
        
        '''
        lambda for jacobian_penalty: 3 (magnitude around 0.1 even without explicitly putting in loss function)
        '''
        return neg_dice + self.lambdas[0] * edge_loss

##

class AwChamferLeafletDiceSmoothness():
    def __init__(self, lambdas=[10, 0, 1e-3, 0]):
        self.lambdas = lambdas

    def __call__(self, pred, target):
        '''
        pred = ([torch.Tensor, list of torch.Tensor, tuple of torch tensors], [list of torch tensors, list of torch tensors])
        target = torch.Tensor (just seg)

        seg shape = [n_batch, n_channels, H, W, D]
        '''

        seg_output = pred[0][0]
        transformed_verts_list = pred[0][1]
        displacement_field_tuple = pred[0][2]
        template_faces_list = pred[1][1]

        seg_output, transformed_verts_list, displacement_field_tuple = utils_sp.get_one_entry(seg_output, transformed_verts_list, displacement_field_tuple)

        dice_store = torch.zeros(target.shape[1], device=seg_output.device)
        for i in range(1, target.shape[1]):
            dice_store[i] = torch.neg(negative_soft_dice_binary_torch(seg_output[:,i,:,:,:], target[:,i,:,:,:], squared_cardinality_den=False, eps=1e-6))
        dice_mean = dice_store.mean()
        neg_dice = torch.neg(dice_mean)

        target_verts_list, target_faces_list = utils_sp.seg_to_mesh(target.squeeze(0).cpu().numpy()[0,:,:,:][np.newaxis,:,:,:])
        target_meshes = utils_sp.mesh_to_pytorch3d_Mesh(target_verts_list, target_faces_list)
        transformed_meshes = utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list[0], template_faces_list[0])
        target_pcl = sample_points_from_meshes(target_meshes)
        transformed_pcl = sample_points_from_meshes(transformed_meshes)

        chamfer_loss, _ = chamfer_distance(target_pcl, transformed_pcl)

        deriv1_mag, grad_output = get_gradient_displacement_field(displacement_field_tuple)
        deriv2_mag, deriv_output = get_squared_2nd_deriv_displacement_field(displacement_field_tuple)
        mag_displacement_field = get_mag_displacement_field(displacement_field_tuple)

        print('chamfer loss')
        print(chamfer_loss.item())
        print('neg dice')
        print(neg_dice.item())
        print('deriv1_mag')
        print(deriv1_mag.item())
        print('deriv2_mag')
        print(deriv2_mag.item())
        print('mag_displacement_field')
        print(mag_displacement_field.item())
        print(' ')

        return chamfer_loss + self.lambdas[0]*neg_dice + self.lambdas[1]*deriv1_mag + self.lambdas[2]*deriv2_mag + self.lambdas[3]*mag_displacement_field, \
               [chamfer_loss.item(), neg_dice.item(), deriv2_mag.item(), deriv1_mag.item(), mag_displacement_field.item()], \
               ['chamfer', 'neg_dice', 'deriv2_mag', 'deriv1_mag', 'mag_field']

##

class ChamferAndSmoothness():
    def __init__(self, lambdas=[0, 1e-4, 0]):
        self.lambdas = lambdas

    def __call__(self, pred, target):
        '''
        pred = ([torch.Tensor, list of torch.Tensor, tuple of torch tensors], [list of torch tensors, list of torch tensors])
        target = torch.Tensor (just seg)

        seg shape = [n_batch, n_channels, H, W, D]
        '''

        # seg_output = pred[0][0]
        seg_output = None # it's actually not used here
        transformed_verts_list = pred[0][1]
        displacement_field_tuple = pred[0][2]
        template_faces_list = pred[1][1]

        _, transformed_verts_list, displacement_field_tuple = utils_sp.get_one_entry(seg_output, transformed_verts_list, displacement_field_tuple)

        target_verts_list, target_faces_list = utils_sp.seg_to_mesh(target.squeeze(0).cpu().numpy())
        target_meshes = utils_sp.mesh_to_pytorch3d_Mesh(target_verts_list, target_faces_list)
        transformed_meshes = utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list, template_faces_list)
        target_pcl = sample_points_from_meshes(target_meshes)
        transformed_pcl = sample_points_from_meshes(transformed_meshes)

        chamfer_loss, _ = chamfer_distance(target_pcl, transformed_pcl)
        deriv1_mag, grad_output = get_gradient_displacement_field(displacement_field_tuple)
        deriv2_mag, deriv_output = get_squared_2nd_deriv_displacement_field(displacement_field_tuple)
        mag_displacement_field = get_mag_displacement_field(displacement_field_tuple)

        print('chamfer loss')
        print(chamfer_loss.item())
        print('deriv1_mag')
        print(deriv1_mag.item())
        print('deriv2_mag')
        print(deriv2_mag.item())
        print('mag_displacement_field')
        print(mag_displacement_field.item())
        print(' ')

        return chamfer_loss + self.lambdas[0]*deriv1_mag + self.lambdas[1]*deriv2_mag + self.lambdas[2]*mag_displacement_field, \
               [chamfer_loss.item(), deriv2_mag.item(), deriv1_mag.item(), mag_displacement_field.item()], \
               ['chamfer', 'deriv2_mag', 'deriv1_mag', 'mag_field']

class MarchingCubesChamferGeo():
    def __init__(self, lambdas=[10, 1, 10], edge_loss_which='correspondence'):
        self.lambdas = lambdas
        self.edge_loss_which = edge_loss_which

        real_template_verts_list, real_template_faces_list, _ = utils_sp.get_template_verts_faces_list('stitched_with_mesh_corr', 'P16_phase1')
        self.real_template_verts_list_torch = utils_sp.np_list_to_torch_list(real_template_verts_list, n_batch=1, device='cuda')
        self.real_template_faces_list_torch = utils_sp.np_list_to_torch_list(real_template_faces_list, n_batch=1, device='cuda')
        self.real_template_meshes = utils_sp.mesh_to_pytorch3d_Mesh(self.real_template_verts_list_torch, self.real_template_faces_list_torch)

    def __call__(self, pred, target):
        '''
        pred = ([torch.Tensor, list of torch.Tensor, tuple of torch tensors], [list of torch tensors, list of torch tensors])
        target = torch.Tensor (just seg)

        seg shape = [n_batch, n_channels, H, W, D]
        '''

        #!! everything here should use deformed marching cubes template mesh
        seg_output = None # it's actually not used here
        transformed_verts_list = pred[0][1]
        displacement_field_tuple = pred[0][2] # forward field
        template_faces_list = pred[1][1]

        _, transformed_verts_list, displacement_field_tuple = utils_sp.get_one_entry(seg_output, transformed_verts_list, displacement_field_tuple)

        target_verts_list, target_faces_list = utils_sp.seg_to_mesh(target.squeeze(0).cpu().numpy())
        target_meshes = utils_sp.mesh_to_pytorch3d_Mesh(target_verts_list, target_faces_list)
        transformed_meshes = utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list, template_faces_list)
        target_pcl = sample_points_from_meshes(target_meshes)
        transformed_pcl = sample_points_from_meshes(transformed_meshes)

        chamfer_loss, _ = chamfer_distance(target_pcl, transformed_pcl)

        #!! grab the real template mesh and calculate geo losses
        interp_field_list = utils_sp.interpolate_rescale_field_torch(displacement_field_tuple, self.real_template_verts_list_torch, img_size=[64,64,64], reversed_field=False)
        real_transformed_verts_list_torch = utils_sp.move_verts_with_field(self.real_template_verts_list_torch, interp_field_list)
        real_transformed_meshes = utils_sp.mesh_to_pytorch3d_Mesh(real_transformed_verts_list_torch, self.real_template_faces_list_torch)

        if self.edge_loss_which == 'correspondence':
            edge_loss = get_mesh_correspondence_edge_loss(self.real_template_meshes, real_transformed_meshes)
        elif self.edge_loss_which == 'uniformity':
            edge_loss = get_mesh_uniformity_loss(real_transformed_meshes)
        elif self.edge_loss_which == 'norm':
            edge_loss = mesh_edge_loss(real_transformed_meshes)

        laplacian_loss = mesh_laplacian_smoothing(real_transformed_meshes)
        normal_consistency_loss = mesh_normal_consistency(real_transformed_meshes)

        print('chamfer loss')
        print(chamfer_loss.item())
        print('edge loss')
        print(edge_loss.item())
        print('laplacian_loss')
        print(laplacian_loss.item())
        print('normal_consistency_loss')
        print(normal_consistency_loss.item())
        print(' ')

        return chamfer_loss + self.lambdas[0] * edge_loss + self.lambdas[1] * laplacian_loss + self.lambdas[2] * normal_consistency_loss, \
               [chamfer_loss.item(), edge_loss.item(), laplacian_loss.item(), normal_consistency_loss.item()], \
               ['chamfer', 'edge', 'laplacian', 'normal']

class MarchingCubesChamferGeoStitchedSep():
    def __init__(self, lambdas=[10, 1, 10], edge_loss_which='correspondence'):
        self.lambdas = lambdas
        self.edge_loss_which = edge_loss_which

        with torch.no_grad():
            real_template_verts_list, real_template_faces_list, self.real_template_extra_info = utils_sp.get_template_verts_faces_list('stitched_with_mesh_corr', 'P16_phase1')
            self.real_template_verts_list_torch = utils_sp.np_list_to_torch_list(real_template_verts_list, n_batch=1, device='cuda')
            self.real_template_faces_list_torch = utils_sp.np_list_to_torch_list(real_template_faces_list, n_batch=1, device='cuda')

            idx_tracks, verts_len_list, faces_len_list = self.real_template_extra_info
            real_template_verts_sep_list = [self.real_template_verts_list_torch[0], self.real_template_verts_list_torch[0], self.real_template_verts_list_torch[0], self.real_template_verts_list_torch[0]]
            real_template_faces_sep_list = torch.split(self.real_template_faces_list_torch[0], faces_len_list, dim=1)
            self.real_template_sep_meshes = utils_sp.mesh_to_pytorch3d_Mesh(real_template_verts_sep_list, real_template_faces_sep_list)

    def __call__(self, pred, target):
        '''
        pred = ([torch.Tensor, list of torch.Tensor, tuple of torch tensors], [list of torch tensors, list of torch tensors])
        target = torch.Tensor (just seg)

        seg shape = [n_batch, n_channels, H, W, D]
        '''

        #!! everything here should use deformed marching cubes template mesh
        seg_output = None # it's actually not used here
        transformed_verts_list = pred[0][1]
        displacement_field_tuple = pred[0][2] # forward field
        template_faces_list = pred[1][1]

        _, transformed_verts_list, displacement_field_tuple = utils_sp.get_one_entry(seg_output, transformed_verts_list, displacement_field_tuple)

        target_verts_list, target_faces_list = utils_sp.seg_to_mesh(target.squeeze(0).cpu().numpy())
        target_meshes = utils_sp.mesh_to_pytorch3d_Mesh(target_verts_list, target_faces_list)
        transformed_meshes = utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list, template_faces_list)
        target_pcl = sample_points_from_meshes(target_meshes)
        transformed_pcl = sample_points_from_meshes(transformed_meshes)

        chamfer_loss, _ = chamfer_distance(target_pcl, transformed_pcl)

        #!! grab the real template mesh and calculate geo losses
        interp_field_list = utils_sp.interpolate_rescale_field_torch(displacement_field_tuple, self.real_template_verts_list_torch, img_size=[64,64,64], reversed_field=False)
        real_transformed_verts_list_torch = utils_sp.move_verts_with_field(self.real_template_verts_list_torch, interp_field_list)

        # # insert dividing stitched mesh into separate parts here
        idx_tracks, verts_len_list, faces_len_list = self.real_template_extra_info
        real_transformed_verts_sep_list = [real_transformed_verts_list_torch[0], real_transformed_verts_list_torch[0], real_transformed_verts_list_torch[0], real_transformed_verts_list_torch[0]]
        real_template_faces_sep_list = torch.split(self.real_template_faces_list_torch[0], faces_len_list, dim=1)
        real_transformed_sep_meshes = utils_sp.mesh_to_pytorch3d_Mesh(real_transformed_verts_sep_list, real_template_faces_sep_list)
        # # done separating

        if self.edge_loss_which == 'correspondence':
            edge_loss = get_mesh_correspondence_edge_loss(self.real_template_sep_meshes, real_transformed_sep_meshes)
        elif self.edge_loss_which == 'uniformity':
            edge_loss = get_mesh_uniformity_loss(real_transformed_sep_meshes)
        elif self.edge_loss_which == 'norm':
            edge_loss = mesh_edge_loss(real_transformed_sep_meshes)

        laplacian_loss = mesh_laplacian_smoothing_DP(real_transformed_sep_meshes)
        normal_consistency_loss = mesh_normal_consistency(real_transformed_sep_meshes)

        print('chamfer loss')
        print(chamfer_loss.item())
        print('edge loss')
        print(edge_loss.item())
        print('laplacian_loss')
        print(laplacian_loss.item())
        print('normal_consistency_loss')
        print(normal_consistency_loss.item())
        print(' ')

        return chamfer_loss + self.lambdas[0] * edge_loss + self.lambdas[1] * laplacian_loss + self.lambdas[2] * normal_consistency_loss, \
               [chamfer_loss.item(), edge_loss.item(), laplacian_loss.item(), normal_consistency_loss.item()], \
               ['chamfer', 'edge', 'laplacian', 'normal']

## ChamferEdgeLaplacianNormal

class ChamferEdgeLaplacianNormal():
    def __init__(self, lambdas=[10, 1, 1], asymmetric_chamfer=False, edge_loss_which='norm'):
        self.lambdas = lambdas
        self.asymmetric_chamfer = asymmetric_chamfer
        self.edge_loss_which = edge_loss_which

    def __call__(self, pred, target):
        '''
        pred = ([torch.Tensor, list of torch.Tensor, tuple of torch tensors], [list of torch tensors, list of torch tensors])
        target = torch.Tensor (just seg)

        seg shape = [n_batch, n_channels, H, W, D]
        '''

        seg_output = pred[0][0]
        transformed_verts_list = pred[0][1]
        displacement_field_tuple = pred[0][2]
        
        seg_output, transformed_verts_list, displacement_field_tuple = utils_sp.get_one_entry(seg_output, transformed_verts_list, displacement_field_tuple)
            
        template_verts_list = pred[1][0]
        template_faces_list = pred[1][1]
        
        target_verts_list, target_faces_list = utils_sp.seg_to_mesh(target.squeeze(0).cpu().numpy())

        template_meshes = utils_sp.mesh_to_pytorch3d_Mesh(template_verts_list, template_faces_list)
        target_meshes = utils_sp.mesh_to_pytorch3d_Mesh(target_verts_list, target_faces_list)
        transformed_meshes = utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list, template_faces_list)
        
        target_pcl = sample_points_from_meshes(target_meshes)
        transformed_pcl = sample_points_from_meshes(transformed_meshes)

        if self.asymmetric_chamfer:
            raise Exception('asymmetric_chamfer not implemented for pytorch3d 0.2.0 yet')
            # chamfer_loss, _ = chamfer_distance_asymmetric(target_pcl, transformed_pcl) # make sure chamfer_distance_asymmetric(target, moving template) -- refer to zzz_chamfer_distance_test.py
        else:
            chamfer_loss, _ = chamfer_distance(target_pcl, transformed_pcl)
        
        if self.edge_loss_which == 'correspondence':
            edge_loss = get_mesh_correspondence_edge_loss(template_meshes, transformed_meshes)
        elif self.edge_loss_which == 'uniformity':
            edge_loss = get_mesh_uniformity_loss(transformed_meshes)
        elif self.edge_loss_which == 'norm':
            edge_loss = mesh_edge_loss(transformed_meshes)
        
        laplacian_loss = mesh_laplacian_smoothing(transformed_meshes)
        normal_consistency_loss = mesh_normal_consistency(transformed_meshes)
        
        print('chamfer loss')
        print(chamfer_loss.item())
        print('edge loss')
        print(edge_loss.item())
        print('laplacian_loss')
        print(laplacian_loss.item())
        print('normal_consistency_loss')
        print(normal_consistency_loss.item())
        print(' ')

        '''
        lambda for jacobian_penalty: 3 (magnitude around 0.1 even without explicitly putting in loss function)
        '''
        return chamfer_loss + self.lambdas[0] * edge_loss + self.lambdas[1] * laplacian_loss + self.lambdas[2] * normal_consistency_loss, \
               [chamfer_loss.item(), edge_loss.item(), laplacian_loss.item(), normal_consistency_loss.item()], \
               ['chamfer', 'edge', 'laplacian', 'normal']

## miccai2021

class ChamferSmoothnessGtPcl_old():
    def __init__(self, lambdas=[1e-3]):
        self.lambdas = lambdas

    def __call__(self, pred, target):
        '''
        pred = ([torch.Tensor, list of torch.Tensor, tuple of torch tensors], [list of torch tensors, list of torch tensors])
        target = torch.Tensor (gt_pcl) [n_batch, n_components(4), H, W, D]
        '''
        transformed_verts_list = pred[0][1]
        displacement_field_tuple = pred[0][2]
        template_faces_list = pred[1][1]
        _, transformed_verts_list, displacement_field_tuple = utils_sp.get_one_entry(None, transformed_verts_list, displacement_field_tuple)

        transformed_meshes = utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list, template_faces_list)
        target_pcl = torch.cat([entry.squeeze(1) for entry in target], dim=1) # combining into one pcl
        transformed_pcl = sample_points_from_meshes(transformed_meshes, num_samples=target_pcl.shape[1])

        chamfer_loss, _ = chamfer_distance(target_pcl, transformed_pcl)
        deriv2_mag, deriv_output = get_squared_2nd_deriv_displacement_field(displacement_field_tuple)

        print('chamfer loss')
        print(chamfer_loss.item())
        print('deriv2_mag')
        print(deriv2_mag.item())
        print(' ')

        return chamfer_loss + self.lambdas[0]*deriv2_mag, \
               [chamfer_loss.item(), deriv2_mag.item()], \
               ['chamfer', 'deriv2_mag']

class ChamferSmoothnessGtPcl():
    def __init__(self, lambdas=[1e-3]):
        self.lambdas = lambdas

    def __call__(self, pred, target):
        '''
        pred = ([torch.Tensor, list of torch.Tensor, tuple of torch tensors], [list of torch tensors, list of torch tensors])
        target = torch.Tensor (gt_pcl) [n_batch, n_components(4), H, W, D]
        '''
        transformed_verts_list = pred[0][1]
        displacement_field_tuple = pred[0][2]
        template_faces_list = pred[1][1]
        _, transformed_verts_list, displacement_field_tuple = utils_sp.get_one_entry(None, transformed_verts_list, displacement_field_tuple)

        from pytorch3d.structures import Pointclouds
        target_pcl = Pointclouds([entry.squeeze(0) for entry in target])
        num_samples_list = [10000, 3000, 3000, 3000]
        transformed_gt_pcl_list = []
        for faces, num_samples in zip(template_faces_list, num_samples_list):
            transformed_gt_pcl_list.append(sample_points_from_meshes(utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list[0], faces), num_samples=num_samples))
        transformed_pcl = Pointclouds([entry.squeeze(0) for entry in transformed_gt_pcl_list])

        chamfer_loss, _ = chamfer_distance(target_pcl, transformed_pcl)

        # transformed_meshes = utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list, template_faces_list)
        # target_pcl = torch.cat([entry.squeeze(1) for entry in target], dim=1) # combining into one pcl
        # transformed_pcl = sample_points_from_meshes(transformed_meshes, num_samples=target_pcl.shape[1])
        # chamfer_loss, _ = chamfer_distance(target_pcl, transformed_pcl)

        deriv2_mag, deriv_output = get_squared_2nd_deriv_displacement_field(displacement_field_tuple)

        print('chamfer loss')
        print(chamfer_loss.item())
        print('deriv2_mag')
        print(deriv2_mag.item())
        print(' ')

        return chamfer_loss + self.lambdas[0]*deriv2_mag, \
               [chamfer_loss.item(), deriv2_mag.item()], \
               ['chamfer', 'deriv2_mag']


class ChamferGeoGtPcl_old():
    def __init__(self, lambdas=[10, 1, 10], edge_loss_which="correspondence"):
        self.lambdas = lambdas
        self.edge_loss_which = edge_loss_which

    def __call__(self, pred, target):
        '''
        pred = ([torch.Tensor, list of torch.Tensor, tuple of torch tensors], [list of torch tensors, list of torch tensors])
        target = torch.Tensor (gt_pcl) [n_batch, n_components(4), H, W, D]
        '''
        transformed_verts_list = pred[0][1]
        template_verts_list = pred[1][0]
        template_faces_list = pred[1][1]
        _, transformed_verts_list, _ = utils_sp.get_one_entry(None, transformed_verts_list, None)

        template_meshes = utils_sp.mesh_to_pytorch3d_Mesh(template_verts_list, template_faces_list)
        transformed_meshes = utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list, template_faces_list)
        target_pcl = torch.cat([entry.squeeze(1) for entry in target], dim=1) # combining into one pcl
        transformed_pcl = sample_points_from_meshes(transformed_meshes, num_samples=target_pcl.shape[1])

        chamfer_loss, _ = chamfer_distance(target_pcl, transformed_pcl)

        if self.edge_loss_which == 'correspondence':
            edge_loss = get_mesh_correspondence_edge_loss(template_meshes, transformed_meshes)
        elif self.edge_loss_which == 'length':
            edge_loss = mesh_edge_loss(transformed_meshes)

        laplacian_loss = mesh_laplacian_smoothing_DP(transformed_meshes)
        normal_consistency_loss = mesh_normal_consistency(transformed_meshes)
        set_trace()
        print('chamfer loss')
        print(chamfer_loss.item())
        print('edge loss')
        print(edge_loss.item())
        print('laplacian_loss')
        print(laplacian_loss.item())
        print('normal_consistency_loss')
        print(normal_consistency_loss.item())
        print(' ')

        return chamfer_loss + self.lambdas[0] * edge_loss + self.lambdas[1] * laplacian_loss + self.lambdas[2] * normal_consistency_loss, \
               [chamfer_loss.item(), edge_loss.item(), laplacian_loss.item(), normal_consistency_loss.item()], \
               ['chamfer', 'edge', 'laplacian', 'normal']

class ChamferGeoGtPcl():
    def __init__(self, lambdas=[10, 1, 10], edge_loss_which="correspondence"):
        self.lambdas = lambdas
        self.edge_loss_which = edge_loss_which

    def __call__(self, pred, target):
        '''
        pred = ([torch.Tensor, list of torch.Tensor, tuple of torch tensors], [list of torch tensors, list of torch tensors])
        target = torch.Tensor (gt_pcl) [n_batch, n_components(4), H, W, D]
        '''
        transformed_verts_list = pred[0][1]
        template_verts_list = pred[1][0]
        template_faces_list = pred[1][1]
        template_elems = pred[1][2]
        _, transformed_verts_list, _ = utils_sp.get_one_entry(None, transformed_verts_list, None)

        from pytorch3d.structures import Pointclouds
        target_pcl = Pointclouds([entry.squeeze(0) for entry in target])
        num_samples_list = [10000, 3000, 3000, 3000]
        template_mesh_list = []
        transformed_mesh_list = []
        transformed_gt_pcl_list = []
        for faces, num_samples in zip(template_faces_list, num_samples_list):
            template_mesh_list.append(utils_sp.mesh_to_pytorch3d_Mesh(template_verts_list[0], faces))
            transformed_mesh = utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list[0], faces)
            transformed_mesh_list.append(transformed_mesh)
            transformed_gt_pcl_list.append(sample_points_from_meshes(transformed_mesh, num_samples=num_samples))
        transformed_pcl = Pointclouds([entry.squeeze(0) for entry in transformed_gt_pcl_list])

        chamfer_loss, _ = chamfer_distance(target_pcl, transformed_pcl)

        # template_meshes = utils_sp.mesh_to_pytorch3d_Mesh(template_verts_list, template_faces_list)
        # transformed_meshes = utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list, template_faces_list)
        # target_pcl = torch.cat([entry.squeeze(1) for entry in target], dim=1) # combining into one pcl
        # transformed_pcl = sample_points_from_meshes(transformed_meshes, num_samples=target_pcl.shape[1])

        # chamfer_loss, _ = chamfer_distance(target_pcl, transformed_pcl)

        from pytorch3d.structures import join_meshes_as_batch
        template_meshes = join_meshes_as_batch(template_mesh_list)
        transformed_meshes = join_meshes_as_batch(transformed_mesh_list)

        all_edges = utils_sp.get_edges_pyvista(utils_cgal.mesh_to_pyvista_UnstructuredGrid(template_verts_list[0].squeeze().cpu().numpy(), template_elems))
        if self.edge_loss_which == 'correspondence':
            # edge_loss = get_mesh_correspondence_edge_loss(template_meshes, transformed_meshes)
            edge_loss = get_mesh_correspondence_edge_loss(template_meshes, transformed_meshes, edges_packed=all_edges)
        elif self.edge_loss_which == 'length':
            # edge_loss = mesh_edge_loss(transformed_meshes)
            edge_loss = mesh_edge_loss_DP(transformed_meshes, edges_packed=all_edges)

        laplacian_loss = mesh_laplacian_smoothing_DP(transformed_meshes)
        normal_consistency_loss = mesh_normal_consistency(transformed_meshes)
        print('chamfer loss')
        print(chamfer_loss.item())
        print('edge loss')
        print(edge_loss.item())
        print('laplacian_loss')
        print(laplacian_loss.item())
        print('normal_consistency_loss')
        print(normal_consistency_loss.item())
        print(' ')

        return chamfer_loss + self.lambdas[0] * edge_loss + self.lambdas[1] * laplacian_loss + self.lambdas[2] * normal_consistency_loss, \
               [chamfer_loss.item(), edge_loss.item(), laplacian_loss.item(), normal_consistency_loss.item()], \
               ['chamfer', 'edge', 'laplacian', 'normal']

class ChamferARAPGtPcl():
    def __init__(self, template_filename, lambdas=[1]):
        self.lambdas = lambdas
        mesh_hex_pv_template = pv.read(os.path.join('../template_for_deform', template_filename))
        verts, elems_tet = utils_cgal.get_verts_faces_from_pyvista(mesh_hex_pv_template.triangulate())
        verts_template_torch = torch.tensor(verts, dtype=torch.get_default_dtype(), device='cuda')
        elems_tet_template_torch = torch.tensor(elems_tet, dtype=int, device='cuda')

        self.calc_deformation_gradient = CalcDeformationGradient(verts_template_torch, elems_tet_template_torch)

    def __call__(self, pred, target):
        transformed_verts_list = pred[0][1]
        template_faces_list = pred[1][1] # must be tri, only used for chamfer distance calculation

        _, transformed_verts_list, _ = utils_sp.get_one_entry(None, transformed_verts_list, None)
        # transformed_meshes = utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list, template_faces_list)
        # target_pcl = torch.cat([entry.squeeze(1) for entry in target], dim=1) # combining into one pcl
        # transformed_pcl = sample_points_from_meshes(transformed_meshes, num_samples=target_pcl.shape[1])

        from pytorch3d.structures import Pointclouds
        target_pcl = Pointclouds([entry.squeeze(0) for entry in target])
        num_samples_list = [10000, 3000, 3000, 3000]
        transformed_gt_pcl_list = []
        for faces, num_samples in zip(template_faces_list, num_samples_list):
            transformed_gt_pcl_list.append(sample_points_from_meshes(utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list[0], faces), num_samples=num_samples))
        transformed_pcl = Pointclouds([entry.squeeze(0) for entry in transformed_gt_pcl_list])

        chamfer_loss, _ = chamfer_distance(target_pcl, transformed_pcl)

        deform_gradient = self.calc_deformation_gradient(transformed_verts_list[0])
        arap_energy = get_arap_energy(deform_gradient).mean()

        print('chamfer loss')
        print(chamfer_loss.item())
        print('arap_energy')
        print(arap_energy.item())

        return chamfer_loss + self.lambdas[0]*arap_energy, \
               [chamfer_loss.item(), arap_energy.item()], \
               ['chamfer', 'arap']

class ChamferWeightedARAPGtPcl():
    def __init__(self, template_filenames1, template_filenames2, lambdas=[1], softmax_base_exp='e', deform_gradient_method='tet', distortion_type='ARAP'):
        self.lambdas = lambdas
        self.softmax_base_exp = softmax_base_exp
        self.distortion_type = distortion_type
        mesh_hex_pv_template1 = pv.read(os.path.join('../template_for_deform', template_filenames1[0]))
        mesh_hex_pv_template2 = pv.read(os.path.join('../template_for_deform', template_filenames2[0]))

        if deform_gradient_method == 'tet':
            verts1, elems_tet1 = utils_cgal.get_verts_faces_from_pyvista(mesh_hex_pv_template1.triangulate())
            verts2, elems_tet2 = utils_cgal.get_verts_faces_from_pyvista(mesh_hex_pv_template2.triangulate())
            verts_template_torch1 = torch.tensor(verts1, dtype=torch.get_default_dtype(), device='cuda')
            verts_template_torch2 = torch.tensor(verts2, dtype=torch.get_default_dtype(), device='cuda')
            elems_tet_template_torch1 = torch.tensor(elems_tet1, dtype=int, device='cuda')
            elems_tet_template_torch2 = torch.tensor(elems_tet2, dtype=int, device='cuda')
            self.calc_deformation_gradient1 = CalcDeformationGradient(verts_template_torch1, elems_tet_template_torch1)
            self.calc_deformation_gradient2 = CalcDeformationGradient(verts_template_torch2, elems_tet_template_torch2)
        elif deform_gradient_method == 'hex_FEM':
            verts1, elems_hex1 = utils_cgal.get_verts_faces_from_pyvista(mesh_hex_pv_template1)
            verts2, elems_hex2 = utils_cgal.get_verts_faces_from_pyvista(mesh_hex_pv_template2)
            verts_template_torch1 = torch.tensor(verts1, dtype=torch.get_default_dtype(), device='cuda')
            verts_template_torch2 = torch.tensor(verts2, dtype=torch.get_default_dtype(), device='cuda')
            elems_hex_template_torch1 = torch.tensor(elems_hex1, dtype=int, device='cuda')
            elems_hex_template_torch2 = torch.tensor(elems_hex2, dtype=int, device='cuda')
            self.calc_deformation_gradient1 = CalcDeformationGradientFEM(verts_template_torch1, elems_hex_template_torch1)
            self.calc_deformation_gradient2 = CalcDeformationGradientFEM(verts_template_torch2, elems_hex_template_torch2)

        with open(os.path.join('../template_for_deform', template_filenames1[1]), 'rb') as f:
            base_surf_faces_tri1 = pickle.load(f)
        with open(os.path.join('../template_for_deform', template_filenames2[1]), 'rb') as f:
            base_surf_faces_tri2 = pickle.load(f)
        self.template_pcl1 = sample_points_from_meshes(utils_sp.mesh_to_pytorch3d_Mesh([verts1], [base_surf_faces_tri1]), num_samples=19000)
        self.template_pcl2 = sample_points_from_meshes(utils_sp.mesh_to_pytorch3d_Mesh([verts2], [base_surf_faces_tri2]), num_samples=19000)

    def __call__(self, pred, target):
        transformed_verts_list = pred[0][1]
        template_faces_list = pred[1][1] # must be tri, only used for chamfer distance calculation

        _, transformed_verts_list, _ = utils_sp.get_one_entry(None, transformed_verts_list, None)
        # transformed_meshes = utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list*len(template_faces_list), template_faces_list)
        # transformed_pcl = sample_points_from_meshes(transformed_meshes, num_samples=target_pcl.shape[1])
        # target_pcl = torch.cat([entry.squeeze(1) for entry in target], dim=1) # combining into one pcl

        from pytorch3d.structures import Pointclouds
        target_pcl = Pointclouds([entry.squeeze(0) for entry in target])
        num_samples_list = [10000, 3000, 3000, 3000]
        transformed_gt_pcl_list = []
        for faces, num_samples in zip(template_faces_list, num_samples_list):
            transformed_gt_pcl_list.append(sample_points_from_meshes(utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list[0], faces), num_samples=num_samples))
        transformed_pcl = Pointclouds([entry.squeeze(0) for entry in transformed_gt_pcl_list])

        chamfer_loss, _ = chamfer_distance(target_pcl, transformed_pcl)

        deform_gradient1 = self.calc_deformation_gradient1(transformed_verts_list[0])
        deform_gradient2 = self.calc_deformation_gradient2(transformed_verts_list[0])
        if self.distortion_type == 'ARAP':
            arap_energy1 = get_arap_energy(deform_gradient1).mean()
            arap_energy2 = get_arap_energy(deform_gradient2).mean()
        elif self.distortion_type == 'ARAP_hex_FEM':
            arap_energy1 = get_arap_energy_hex(deform_gradient1).mean()
            arap_energy2 = get_arap_energy_hex(deform_gradient2).mean()
        elif self.distortion_type == 'Symmetric ARAP':
            raise NotImplementedError('symmetric ARAP')
        elif self.distortion_type == 'Co-rotational':
            raise NotImplementedError('Co-rotational')
        elif self.distortion_type == 'Dirichlet':
            raise NotImplementedError('Dirichlet')
        elif self.distortion_type == 'Symmetric Dirichlet':
            raise NotImplementedError('Symmetric Dirichlet')
        elif self.distortion_type == 'MIPS':
            raise NotImplementedError('MIPS')

        # chamfer_for_weight1 = chamfer_distance(transformed_pcl, self.template_pcl1)[0]
        # chamfer_for_weight2 = chamfer_distance(transformed_pcl, self.template_pcl2)[0]
        chamfer_for_weight1 = chamfer_distance(torch.cat(transformed_pcl.points_list()).unsqueeze(0), self.template_pcl1)[0]
        chamfer_for_weight2 = chamfer_distance(torch.cat(transformed_pcl.points_list()).unsqueeze(0), self.template_pcl2)[0]
        weights = 1 - softmax_DP(torch.stack([chamfer_for_weight1, chamfer_for_weight2]), base_exp=self.softmax_base_exp)
        arap_energy = weights[0]*arap_energy1 + weights[1]*arap_energy2

        print('chamfer loss')
        print(chamfer_loss.item())
        print('arap_energy')
        print(arap_energy.item())

        return chamfer_loss + self.lambdas[0]*arap_energy, \
               [chamfer_loss.item(), arap_energy.item()], \
               ['chamfer', 'arap']

class ChamferWeightedARAPGtPclShapeCode():
    def __init__(self, template_filenames1, template_filenames2, lambdas=[1, 1e-4], softmax_base_exp='e'):
        self.lambdas = lambdas
        self.softmax_base_exp = softmax_base_exp
        mesh_hex_pv_template1 = pv.read(os.path.join('../template_for_deform', template_filenames1[0]))
        mesh_hex_pv_template2 = pv.read(os.path.join('../template_for_deform', template_filenames2[0]))
        verts1, elems_tet1 = utils_cgal.get_verts_faces_from_pyvista(mesh_hex_pv_template1.triangulate())
        verts2, elems_tet2 = utils_cgal.get_verts_faces_from_pyvista(mesh_hex_pv_template2.triangulate())
        verts_template_torch1 = torch.tensor(verts1, dtype=torch.get_default_dtype(), device='cuda')
        verts_template_torch2 = torch.tensor(verts2, dtype=torch.get_default_dtype(), device='cuda')
        elems_tet_template_torch1 = torch.tensor(elems_tet1, dtype=int, device='cuda')
        elems_tet_template_torch2 = torch.tensor(elems_tet2, dtype=int, device='cuda')

        self.calc_deformation_gradient1 = CalcDeformationGradient(verts_template_torch1, elems_tet_template_torch1)
        self.calc_deformation_gradient2 = CalcDeformationGradient(verts_template_torch2, elems_tet_template_torch2)

        with open(os.path.join('../template_for_deform', template_filenames1[1]), 'rb') as f:
            base_surf_faces_tri1 = pickle.load(f)
        with open(os.path.join('../template_for_deform', template_filenames2[1]), 'rb') as f:
            base_surf_faces_tri2 = pickle.load(f)
        self.template_pcl1 = sample_points_from_meshes(utils_sp.mesh_to_pytorch3d_Mesh([verts1], [base_surf_faces_tri1]), num_samples=10000)
        self.template_pcl2 = sample_points_from_meshes(utils_sp.mesh_to_pytorch3d_Mesh([verts2], [base_surf_faces_tri2]), num_samples=10000)

    def __call__(self, pred, target):
        transformed_verts_list = pred[0][1]
        shape_code_list_torch = pred[0][3]
        template_faces_list = pred[1][1] # must be tri, only used for chamfer distance calculation

        _, transformed_verts_list, _ = utils_sp.get_one_entry(None, transformed_verts_list, None)

        from pytorch3d.structures import Pointclouds
        target_pcl = Pointclouds([entry.squeeze(0) for entry in target])
        num_samples_list = [10000, 3000, 3000, 3000]
        transformed_gt_pcl_list = []
        for faces, num_samples in zip(template_faces_list, num_samples_list):
            transformed_gt_pcl_list.append(sample_points_from_meshes(utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list[0], faces), num_samples=num_samples))
        transformed_pcl = Pointclouds([entry.squeeze(0) for entry in transformed_gt_pcl_list])

        chamfer_loss, _ = chamfer_distance(target_pcl, transformed_pcl)

        deform_gradient1 = self.calc_deformation_gradient1(transformed_verts_list[0])
        deform_gradient2 = self.calc_deformation_gradient2(transformed_verts_list[0])
        arap_energy1 = get_arap_energy(deform_gradient1).mean()
        arap_energy2 = get_arap_energy(deform_gradient2).mean()

        chamfer_for_weight1 = chamfer_distance(torch.cat(transformed_pcl.points_list()).unsqueeze(0), self.template_pcl1)[0]
        chamfer_for_weight2 = chamfer_distance(torch.cat(transformed_pcl.points_list()).unsqueeze(0), self.template_pcl2)[0]
        weights = 1 - softmax_DP(torch.stack([chamfer_for_weight1, chamfer_for_weight2]), base_exp=self.softmax_base_exp)
        arap_energy = weights[0]*arap_energy1 + weights[1]*arap_energy2

        shape_code_norm = torch.sum(torch.norm(shape_code_list_torch[0], dim=1))

        print('chamfer loss')
        print(chamfer_loss.item())
        print('arap_energy')
        print(arap_energy.item())
        print('shape_code_norm')
        print(shape_code_norm.item())

        return chamfer_loss + self.lambdas[0]*arap_energy + self.lambdas[1]*shape_code_norm, \
               [chamfer_loss.item(), arap_energy.item(), shape_code_norm.item()], \
               ['chamfer', 'arap', 'shape_code_norm']

## ChamferEdgeLaplacianNormalGtPcl

class ChamferEdgeLaplacianNormalGtPcl():
    def __init__(self, lambdas=[10, 1, 1], asymmetric_chamfer=False, edge_loss_which='norm'):
        self.lambdas = lambdas
        self.asymmetric_chamfer = asymmetric_chamfer
        self.edge_loss_which = edge_loss_which

    def __call__(self, pred, target):
        '''
        pred = ([torch.Tensor, list of torch.Tensor, tuple of torch tensors], [list of torch tensors, list of torch tensors])
        target = torch.Tensor (gt_pcl) [n_batch, n_components(4), H, W, D]
        '''
        seg_output = pred[0][0]
        transformed_verts_list = pred[0][1]
        displacement_field_tuple = pred[0][2]
        # transformed_seg_template_marching_cubes_pcl_list = pred[0][3]
        
        _, transformed_verts_list, _ = utils_sp.get_one_entry(seg_output, transformed_verts_list, displacement_field_tuple)
        
        template_verts_list = pred[1][0]
        template_faces_list = pred[1][1]
        # template_extra_info = pred[1][2]
        template_extra_info = None # turning off the splitting of verts_list and faces_list into separate components (slows it down a lot)

        # # insert dividing here
        if template_extra_info is not None:
            idx_tracks, verts_len_list, faces_len_list = template_extra_info

            template_verts_new_list, template_faces_new_list = \
                utils_cgal.get_sep_components_from_all_verts_all_faces_torch(template_verts_list[0],
                                                                             template_faces_list[0],
                                                                             idx_tracks, verts_len_list, faces_len_list)

            # don't need transformed_faces_new_list here b/c only verts are transformed
            transformed_verts_new_list, _ = \
                utils_cgal.get_sep_components_from_all_verts_all_faces_torch(transformed_verts_list[0],
                                                                             template_faces_list[0],
                                                                             idx_tracks, verts_len_list, faces_len_list)

            # TODO: need to fix ordering of leaflets in the saved file... need some system to organize which leaflet is which
            # template_faces_list should already have been converted to triangular mesh, and faces_len_list should already have been adjusted accordingly
            template_verts_new_list.append(template_verts_new_list.pop(1)) # need change order of leaflets to match gt_pcl ordering
            template_faces_new_list.append(template_faces_new_list.pop(1)) # need change order of leaflets to match gt_pcl ordering
            transformed_verts_new_list.append(transformed_verts_new_list.pop(1)) # need change order of leaflets to match gt_pcl ordering

            template_verts_list = template_verts_new_list[:]
            template_faces_list = template_faces_new_list[:]
            transformed_verts_list = transformed_verts_new_list[:]

        template_meshes = utils_sp.mesh_to_pytorch3d_Mesh(template_verts_list, template_faces_list)
        transformed_meshes = utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list, template_faces_list)

        # template_meshes = utils_sp.mesh_to_pytorch3d_Mesh(template_verts_list, template_faces_list)
        # transformed_meshes = utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list, template_faces_list)

        target_pcl = torch.stack([entry.squeeze() for entry in target], dim=0)
        transformed_pcl = sample_points_from_meshes(transformed_meshes, num_samples=target_pcl.shape[1])

        # if transformed_seg_template_marching_cubes_pcl_list is not None:
        #     transformed_seg_template_marching_cubes_pcl = transformed_seg_template_marching_cubes_pcl_list[0]
        #     transformed_pcl = sample_points_from_meshes(transformed_meshes, num_samples=target_pcl.shape[1]-transformed_seg_template_marching_cubes_pcl.shape[1])
        #     transformed_pcl = torch.cat([transformed_pcl, transformed_seg_template_marching_cubes_pcl], axis=1)
        # else:
        #     transformed_pcl = sample_points_from_meshes(transformed_meshes, num_samples=target_pcl.shape[1])

        if self.asymmetric_chamfer:
            raise Exception('asymmetric_chamfer not implemented for pytorch3d 0.2.0 yet')
            # chamfer_loss, _ = chamfer_distance_asymmetric(target_pcl, transformed_pcl) # make sure chamfer_distance_asymmetric(target, moving template) -- refer to zzz_chamfer_distance_test.py
        else:
            chamfer_loss, _ = chamfer_distance(target_pcl, transformed_pcl)
        
        if self.edge_loss_which == 'correspondence':
            edge_loss = get_mesh_correspondence_edge_loss(template_meshes, transformed_meshes)
        elif self.edge_loss_which == 'uniformity':
            edge_loss = get_mesh_uniformity_loss(transformed_meshes)
        elif self.edge_loss_which == 'norm':
            edge_loss = mesh_edge_loss(transformed_meshes)
        
        laplacian_loss = mesh_laplacian_smoothing(transformed_meshes)
        normal_consistency_loss = mesh_normal_consistency(transformed_meshes)
        
        print('chamfer loss')
        print(chamfer_loss.item())
        print('edge loss')
        print(edge_loss.item())
        print('laplacian_loss')
        print(laplacian_loss.item())
        print('normal_consistency_loss')
        print(normal_consistency_loss.item())
        print(' ')

        '''
        lambda for jacobian_penalty: 3 (magnitude around 0.1 even without explicitly putting in loss function)
        '''
        return chamfer_loss + self.lambdas[0] * edge_loss + self.lambdas[1] * laplacian_loss + self.lambdas[2] * normal_consistency_loss, \
               [chamfer_loss.item(), edge_loss.item(), laplacian_loss.item(), normal_consistency_loss.item()], \
               ['chamfer', 'edge', 'laplacian', 'normal']

##

class ChamferEdgeLaplacianNormalGtPclStitchedSep():
    def __init__(self, lambdas=[10, 1, 1], asymmetric_chamfer=False, edge_loss_which='norm'):
        self.lambdas = lambdas
        self.asymmetric_chamfer = asymmetric_chamfer
        self.edge_loss_which = edge_loss_which

    def __call__(self, pred, target):
        '''
        pred = ([torch.Tensor, list of torch.Tensor, tuple of torch tensors], [list of torch tensors, list of torch tensors])
        target = torch.Tensor (gt_pcl) [n_batch, n_components(4), H, W, D]
        '''
        seg_output = pred[0][0]
        transformed_verts_list = pred[0][1]
        displacement_field_tuple = pred[0][2]

        _, transformed_verts_list, _ = utils_sp.get_one_entry(seg_output, transformed_verts_list, displacement_field_tuple)

        template_verts_list = pred[1][0]
        template_faces_list = pred[1][1]
        template_extra_info = pred[1][2]

        # # insert dividing stitched mesh into separate parts here
        template_verts_sep_list = [template_verts_list[0], template_verts_list[0], template_verts_list[0], template_verts_list[0]]
        transformed_verts_sep_list = [transformed_verts_list[0], transformed_verts_list[0], transformed_verts_list[0], transformed_verts_list[0]]
        # idx_tracks, verts_len_list, faces_len_list = template_extra_info
        # template_faces_sep_list = torch.split(template_faces_list[0], faces_len_list, dim=1)
        template_faces_sep_list = [utils_cgal.split_quad_to_2_tri_mesh(faces) for faces in template_extra_info]
        # # done separating

        template_sep_meshes = utils_sp.mesh_to_pytorch3d_Mesh(template_verts_sep_list, template_faces_sep_list)
        transformed_sep_meshes = utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_sep_list, template_faces_sep_list)

        # target_pcl = torch.stack([entry.squeeze() for entry in target], dim=0)
        # transformed_pcl = sample_points_from_meshes(transformed_sep_meshes, num_samples=int(target_pcl.shape[1]/len(transformed_sep_meshes))).reshape(-1,3).unsqueeze(0)

        target_pcl = target.squeeze(0)
        transformed_pcl = sample_points_from_meshes(transformed_sep_meshes, num_samples=10000)

        chamfer_loss, _ = chamfer_distance(target_pcl, transformed_pcl)
        if self.edge_loss_which == 'correspondence':
            edge_loss = get_mesh_correspondence_edge_loss(template_sep_meshes, transformed_sep_meshes)
        elif self.edge_loss_which == 'uniformity':
            edge_loss = get_mesh_uniformity_loss(transformed_sep_meshes)
        elif self.edge_loss_which == 'norm':
            edge_loss = mesh_edge_loss(transformed_sep_meshes)

        laplacian_loss = mesh_laplacian_smoothing_DP(transformed_sep_meshes)
        normal_consistency_loss = mesh_normal_consistency(transformed_sep_meshes)

        print('chamfer loss')
        print(chamfer_loss.item())
        print('edge loss')
        print(edge_loss.item())
        print('laplacian_loss')
        print(laplacian_loss.item())
        print('normal_consistency_loss')
        print(normal_consistency_loss.item())
        print(' ')

        '''
        lambda for jacobian_penalty: 3 (magnitude around 0.1 even without explicitly putting in loss function)
        '''
        return chamfer_loss + self.lambdas[0] * edge_loss + self.lambdas[1] * laplacian_loss + self.lambdas[2] * normal_consistency_loss, \
               [chamfer_loss.item(), edge_loss.item(), laplacian_loss.item(), normal_consistency_loss.item()], \
               ['chamfer', 'edge', 'laplacian', 'normal']

##

class DiceAndGtPclMeshLosses():
    def __init__(self, lambdas=[1, 10, 1, 10], asymmetric_chamfer=False, edge_loss_which='norm'):
        self.lambdas = lambdas
        self.asymmetric_chamfer = asymmetric_chamfer
        self.edge_loss_which = edge_loss_which

    def __call__(self, pred, target):
        '''
        pred = ([torch.Tensor, list of torch.Tensor, tuple of torch tensors], [list of torch tensors, list of torch tensors])
        target = torch.Tensor (gt_pcl) [n_batch, n_components(4), H, W, D]
        '''
        seg_output = pred[0][0]
        transformed_verts_list = pred[0][1]
        displacement_field_tuple = pred[0][2]

        target_seg = target[0]
        target_mesh = target[1]

        seg_output, transformed_verts_list, _ = utils_sp.get_one_entry(seg_output, transformed_verts_list, displacement_field_tuple)

        # segmentation loss here
        assert seg_output.shape[1] == target_seg.shape[1], 'n_channels of network output should be same as n_channels of labels'
        dice_store = torch.zeros(target_seg.shape[1], device=seg_output.device)
        for i in range(target_seg.shape[1]):
            dice_store[i] = torch.neg(negative_soft_dice_binary_torch(seg_output[:,i,:,:,:], target_seg[:,i,:,:,:], squared_cardinality_den=False, eps=1e-6))
        dice_mean = dice_store.mean()
        neg_dice = torch.neg(dice_mean)

        # mesh losses here
        template_verts_list = pred[1][0]
        template_faces_list = pred[1][1]
        # template_extra_info = pred[1][2]
        template_extra_info = None # turning off the splitting of verts_list and faces_list into separate components (slows it down a lot)

        # insert dividing here
        if template_extra_info is not None:
            idx_tracks, verts_len_list, faces_len_list = template_extra_info

            template_verts_new_list, template_faces_new_list = \
                utils_cgal.get_sep_components_from_all_verts_all_faces_torch(template_verts_list[0],
                                                                             template_faces_list[0],
                                                                             idx_tracks, verts_len_list, faces_len_list)

            # don't need transformed_faces_new_list here b/c only verts are transformed
            transformed_verts_new_list, _ = \
                utils_cgal.get_sep_components_from_all_verts_all_faces_torch(transformed_verts_list[0],
                                                                             template_faces_list[0],
                                                                             idx_tracks, verts_len_list, faces_len_list)

            # TODO: need to fix ordering of leaflets in the saved file... need some system to organize which leaflet is which
            # template_faces_list should already have been converted to triangular mesh, and faces_len_list should already have been adjusted accordingly
            template_verts_new_list.append(template_verts_new_list.pop(1)) # need change order of leaflets to match gt_pcl ordering
            template_faces_new_list.append(template_faces_new_list.pop(1)) # need change order of leaflets to match gt_pcl ordering
            transformed_verts_new_list.append(transformed_verts_new_list.pop(1)) # need change order of leaflets to match gt_pcl ordering

            template_verts_list = template_verts_new_list[:]
            template_faces_list = template_faces_new_list[:]
            transformed_verts_list = transformed_verts_new_list[:]

        template_meshes = utils_sp.mesh_to_pytorch3d_Mesh(template_verts_list, template_faces_list)
        transformed_meshes = utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list, template_faces_list)

        # template_meshes = utils_sp.mesh_to_pytorch3d_Mesh(template_verts_list, template_faces_list)
        # transformed_meshes = utils_sp.mesh_to_pytorch3d_Mesh(transformed_verts_list, template_faces_list)

        target_pcl = torch.stack([entry.squeeze() for entry in target_mesh], dim=0).to(transformed_meshes.device)
        transformed_pcl = sample_points_from_meshes(transformed_meshes, num_samples=target_pcl.shape[1])

        if self.asymmetric_chamfer:
            raise Exception('asymmetric_chamfer not implemented for pytorch3d 0.2.0 yet')
            # chamfer_loss, _ = chamfer_distance_asymmetric(target_pcl, transformed_pcl) # make sure chamfer_distance_asymmetric(target, moving template) -- refer to zzz_chamfer_distance_test.py
        else:
            chamfer_loss, _ = chamfer_distance(target_pcl, transformed_pcl)

        if self.edge_loss_which == 'correspondence':
            edge_loss = get_mesh_correspondence_edge_loss(template_meshes, transformed_meshes)
        elif self.edge_loss_which == 'uniformity':
            edge_loss = get_mesh_uniformity_loss(transformed_meshes)
        elif self.edge_loss_which == 'norm':
            edge_loss = mesh_edge_loss(transformed_meshes)

        laplacian_loss = mesh_laplacian_smoothing(transformed_meshes)
        normal_consistency_loss = mesh_normal_consistency(transformed_meshes)

        print('chamfer loss')
        print(chamfer_loss.item())
        print('neg dice')
        print(neg_dice.item())
        print('edge loss')
        print(edge_loss.item())
        print('laplacian_loss')
        print(laplacian_loss.item())
        print('normal_consistency_loss')
        print(normal_consistency_loss.item())
        print('testing DPDP')
        print(chamfer_loss + self.lambdas[0] * neg_dice + self.lambdas[1] * edge_loss + self.lambdas[2] * laplacian_loss + self.lambdas[3] * normal_consistency_loss)
        print(' ')

        return chamfer_loss + self.lambdas[0] * neg_dice + self.lambdas[1] * edge_loss + self.lambdas[2] * laplacian_loss + self.lambdas[3] * normal_consistency_loss, \
               [chamfer_loss.item(), neg_dice.item(), edge_loss.item(), laplacian_loss.item(), normal_consistency_loss.item()], \
               ['chamfer', 'neg_dice', 'edge', 'laplacian', 'normal']

##

class EvaluateDeformModelMesh():
    def __init__(self, template_verts_list, template_faces_list, reversed_field=True, gt_pcl_list=None, num_sample_pts=10000, img_size=[64,64,64]):    
        '''
        template_verts_list: list of np.ndarray [n_verts, n_dim]
        template_verts_list: list of np.ndarray [n_faces, n_dim]
        field: torch.tensor [n_batch, n_channel, h, w, d]
        '''
        self.template_verts_list = template_verts_list
        self.template_faces_list = []
        for verts, faces in zip(template_verts_list, template_faces_list):
            mesh_pv_tri = utils_sp.mesh_to_pyvista_PolyData(verts, faces).triangulate()
            self.template_faces_list.append(utils_cgal.get_verts_faces_from_pyvista(mesh_pv_tri)[1])
        self.gt_pcl_list = gt_pcl_list
        self.img_size = img_size
        self.num_sample_pts = num_sample_pts
        self.reversed_field = reversed_field
    
    def replace_with_dict2(self, ar, dic):
        # Extract out keys and values
        k = np.array(list(dic.keys()))
        v = np.array(list(dic.values()))
    
        # Get argsort indices
        sidx = k.argsort()
    
        ks = k[sidx]
        vs = v[sidx]
        return vs[np.searchsorted(ks,ar)]

    def clip_off_transformed_mesh_within_pcl(self, pcl_list):
        z_min = np.vstack(pcl_list)[:,2].min()
        z_max = np.vstack(pcl_list)[:,2].max()
        
        new_verts_list = []
        new_faces_list = []
        for verts, faces in zip(self.transformed_verts_list, self.template_faces_list):
            idx_key = np.arange(verts.shape[0])
            
            new_verts = verts[np.logical_and(verts[:,2]<z_max, verts[:,2]>z_min)]
            new_idx_key = idx_key[np.logical_and(verts[:,2]<z_max, verts[:,2]>z_min)]
            idx_dict = dict(zip(new_idx_key, np.arange(len(new_idx_key))))
            
            verts_remove_idxes = np.argwhere(np.logical_or(verts[:,2]>z_max, verts[:,2]<z_min))
            
            new_faces = faces.copy()
            for idx in verts_remove_idxes:
                new_faces = new_faces[(new_faces != idx).all(axis=1),:]
            
            new_faces = self.replace_with_dict2(new_faces, idx_dict)
            
            new_verts_list.append(new_verts)
            new_faces_list.append(new_faces)
        
        self.transformed_verts_list = new_verts_list
        self.template_faces_list = new_faces_list
    
    def clip_off_pcl_within_transformed_mesh(self):
        z_min = np.vstack(self.transformed_verts_list)[:,2].min()
        z_max = np.vstack(self.transformed_verts_list)[:,2].max()
        
        new_gt_pcl_list = []
        for gt_pcl in self.gt_pcl_list:
            new_gt_pcl = gt_pcl.clone()
            new_gt_pcl = new_gt_pcl[(new_gt_pcl[:,2]>z_min) & (new_gt_pcl[:,2]<z_max)]
            
            new_gt_pcl_list.append(new_gt_pcl)
        
        self.gt_pcl_list = new_gt_pcl_list
    
    def choose_n_pts_random(self, pcl_list, n_desired_pts=10000):
        pcl_chosen_list = []
        for pcl in pcl_list:
            n_pts = pcl.shape[0]
            
            rand_idx = np.random.choice(np.arange(n_pts), size=n_desired_pts, replace=False)
            
            pcl_chosen = pcl[rand_idx,:]
            pcl_chosen_list.append(pcl_chosen)
        
        return pcl_chosen_list

    def input_transformed_verts_list(self, verts_list):
        self.transformed_verts_list = verts_list
    
    def calc_transformed_verts_list(self, field):
        '''
        output: self.transformed_verts_list (list of np.array [n_verts, n_dim])
        '''
        if not isinstance(field, tuple):
            field = (field,)

        template_verts_list_torch = utils_sp.np_list_to_torch_list(self.template_verts_list, n_batch=1, device=field[0].device)
        template_verts_list_interp = template_verts_list_torch

        interp_field_list = utils_sp.interpolate_rescale_field_torch(field, template_verts_list_interp, img_size=self.img_size, reversed_field=self.reversed_field)

        # all_verts, all_faces, template_extra_info = utils_sp.get_template_verts_faces_list('stitched_with_mesh_corr', 'P16_phase1')
        # _, faces_sep_list_template = utils_cgal.get_sep_components_from_all_verts_all_faces(all_verts[0], all_faces[0], *template_extra_info)
        # faces_sep_list_template_torch = utils_sp.np_list_to_torch_list(faces_sep_list_template, n_batch=1, device='cuda')
        # from pytorch3d.ops import GraphConv
        # gconv = GraphConv(3, 3, init='zero').cuda()
        # gconv.w0.weight.data = torch.eye(3, dtype=gconv.w0.weight.data.dtype, device=gconv.w0.weight.data.device)
        # gconv.w1.weight.data = torch.eye(3, dtype=gconv.w1.weight.data.dtype, device=gconv.w1.weight.data.device)
        # gconv.w0.bias.data.zero_()
        # gconv.w1.bias.data.zero_()
        # interp_field_list_smooth = []
        # for interp_field, faces_torch in zip(interp_field_list, faces_sep_list_template_torch):
        #     verts = interp_field
        #     faces = faces_torch
        #     edges = utils_sp.get_edges(verts, faces)
        #     deg = utils_sp.get_node_degree(verts, faces)
        #     with torch.no_grad():
        #         interp_field_smooth = gconv(verts.squeeze(), edges)/(deg+1).unsqueeze(1)
        #         interp_field_smooth = gconv(interp_field_smooth.squeeze(), edges)/(deg+1).unsqueeze(1)
        #         interp_field_list_smooth.append(interp_field_smooth.unsqueeze(0))
        #         # interp_field_smooth_list = [interp_field_smooth.unsqueeze(0)]
        #         # interp_field_list = interp_field_smooth_list
        # interp_field_list = interp_field_list_smooth

        transformed_verts_list_torch = utils_sp.move_verts_with_field(template_verts_list_torch, interp_field_list)
        
        self.transformed_verts_list = utils_sp.torch_list_to_np_list(transformed_verts_list_torch)
    
    def calc_sampled_pts(self):
        '''
        output: self.sample_pts (torch.tensor, [len(self.transformed_verts_list), n_pts (10000 default for sampling), n_dim])
        '''
        if not hasattr(self, 'transformed_verts_list'):
            raise ValueError('run self.calc_transformed_verts_list(field) first')

        # sample pts from transformed mesh
        mesh = utils_sp.mesh_to_pytorch3d_Mesh(self.transformed_verts_list, self.template_faces_list)
        self.sampled_pts = sample_points_from_meshes(mesh, num_samples=self.num_sample_pts)
        
        # getting rid of self.sampled_pts that's above/below max/min of gt_pcl_list?
#        if self.gt_pcl_list is not None:
#            z_upper = self.gt_pcl_list[0][:,2].max()
#            z_lower = self.gt_pcl_list[0][:,2].min()
#            
#            n_pts_min = self.num_sample_pts
#            processed_pts = []
#            for idx in range(self.sampled_pts.shape[0]):
#                pts = self.sampled_pts[idx,:,:].cpu()
#                pts = pts[pts[:,2] < z_upper]
#                pts = pts[pts[:,2] > z_lower]
#                processed_pts.append(pts)
#                if pts.shape[0] < n_pts_min:
#                    n_pts_min = pts.shape[0]
#            
#            for idx, pts in enumerate(processed_pts):
#                rand_idx = np.random.choice(np.arange(pts.shape[0]), n_pts_min, replace=False)
#                processed_pts[idx] = pts[rand_idx,:]
#            
#            self.sampled_pts = torch.stack(processed_pts).cuda()
    
    def get_two_pcls_to_match_n_pts(self, verts1, verts2):
        n_pts_min = min(verts1.shape[1], verts2.shape[1])
        rand_idx1 = np.random.choice(np.arange(verts1.shape[1]), n_pts_min, replace=False)
        rand_idx2 = np.random.choice(np.arange(verts2.shape[1]), n_pts_min, replace=False)
        
        verts1_processed = verts1[:,rand_idx1,:]
        verts2_processed = verts2[:,rand_idx2,:]
        
        return verts1_processed, verts2_processed

    def get_sampled_pts_to_trimeshes_dist(self, verts_list, faces_list):
        '''
        verts_list, faces_list: list of np.array or torch.tensor [n_verts or n_faces, n_dim]

        Uses AABB tree to compute - doesn't sound like the sampling + distance method that most others use
        '''
        import igl

        splits = np.split(self.sampled_pts.cpu().numpy(), self.sampled_pts.shape[0], axis=0)
        sampled_pts_np_list = [s.squeeze() for s in splits]

        pts_to_meshes_dist_list = []

        for sample_pts_np, verts, faces in zip(sampled_pts_np_list, verts_list, faces_list):
            if torch.is_tensor(verts):
                verts = verts.squeeze().cpu().numpy()

            if torch.is_tensor(faces):
                faces = faces.squeeze().cpu().numpy()

            # #DPDP TEST delete later # didn't make that much of a difference
            # sample_pts_np = sample_pts_np[0:1000]

            dist_for_each_pt, idx_to_closest, three_closest_pts = igl.point_mesh_squared_distance(sample_pts_np, verts,
                                                                                                  faces)

            pts_to_meshes_dist_list.append(np.sqrt(np.array(dist_for_each_pt)))

        pts_to_mesh_dists = np.vstack(pts_to_meshes_dist_list).T  # pts_to_mesh_dists range > 0

        mean_dist_each_comp = pts_to_mesh_dists.mean(axis=0)
        std_dist_each_comp = pts_to_mesh_dists.std(axis=0)

        return mean_dist_each_comp, std_dist_each_comp

#    def get_sampled_pts_to_target_mesh_chamfer(self, target_verts_list, target_faces_list):
#        '''
#        chamfer_distance from transformed_mesh pcl to target_mesh pcl
#        '''
#        target_mesh = utils_sp.mesh_to_pytorch3d_Mesh(target_verts_list, target_faces_list)
#        target_pcl = sample_points_from_meshes(target_mesh, num_samples=self.num_sample_pts)
#        
#        chamfer_dist = chamfer_distance(self.sampled_pts[0,:,:].unsqueeze(0), target_pcl[0,:,:].unsqueeze(0), point_reduction='none')
#        
#        chamfer_dists, _ = chamfer_distance_eval(target_pcl, self.sampled_pts) # chamfer_dists > 0
#        
#        mean_chamfer = chamfer_dists.mean(axis=1).cpu().numpy()
#        std_chamfer = chamfer_dists.std(axis=1).cpu().numpy()
#        
#        return mean_chamfer, std_chamfer
    
    def get_sampled_pts_to_gt_pcl_chamfer(self):
        '''
        target_pcl_list = torch.tensor (n_classes, n_pts, n_dim)
        
        use for dpoints comparison
        '''
        n_min = np.inf
        for gt_pcl in self.gt_pcl_list:
            if gt_pcl.shape[0] < n_min:
                n_min = gt_pcl.shape[0]
                
        gt_pcl_processed = torch.stack(self.choose_n_pts_random(self.gt_pcl_list, n_desired_pts=n_min)).cuda()
        
        sampled_pts_matched, target_pcl_matched = self.get_two_pcls_to_match_n_pts(self.sampled_pts, gt_pcl_processed)
        
#        target_pcl_matched = target_pcl
#        sampled_pts_matched = self.sampled_pts

        chamfer_dists, _ = chamfer_distance_eval(sampled_pts_matched, target_pcl_matched)
        
        mean_chamfer = chamfer_dists.mean(axis=1).cpu().numpy()
        std_chamfer = chamfer_dists.std(axis=1).cpu().numpy()
        
        return mean_chamfer, std_chamfer
    
    def get_averged_hausdorff_distance(self, target_verts_list, target_faces_list):
        import igl
        
        hd_each_comp_list = []
        for va, fa, vb, fb in zip(self.transformed_verts_list, self.template_faces_list, target_verts_list, target_faces_list):
            if torch.is_tensor(va):
                va = va.squeeze().cpu().numpy()
            va = np.asfortranarray(va)
                
            if torch.is_tensor(vb):
                vb = vb.squeeze().cpu().numpy()
            vb = np.asfortranarray(vb)

            if torch.is_tensor(fa):
                fa = fa.squeeze().cpu().numpy()
            
            if torch.is_tensor(fb):
                fb = fb.squeeze().cpu().numpy()
            
            hd_val = igl.hausdorff(va.astype(np.float32), fa.astype(np.int32), vb.astype(np.float32), fb.astype(np.int32))
            
            hd_each_comp_list.append(hd_val)
            
        hd_all = np.vstack(hd_each_comp_list).T # pts_to_mesh_dists range > 0
        
        mean_hd_each_comp = hd_all.mean(axis=0)
        std_hd_each_comp = hd_all.std(axis=0)
        
        return mean_hd_each_comp, std_hd_each_comp

##

class CalcDeformationGradient():
    def __init__(self, verts_template, elems_template):
        """ verts and elems should be torch.Tensors, elems_template should be tet """
        self.elems_template = elems_template

        verts_template_tet = verts_template[self.elems_template].permute(0,2,1) # (n_elems, n_dim (3), n_verts_per_elem)
        verts_template_tet_centered = verts_template_tet - verts_template_tet[:,:,0].unsqueeze(2)
        verts_template_tet_centered = verts_template_tet_centered[:,:,1:]

        with torch.no_grad():
            self.D_m_inv = torch.inverse(verts_template_tet_centered)

    def __call__(self, verts_transformed):
        verts_transformed_tet = verts_transformed.squeeze()[self.elems_template].permute(0,2,1) # (n_elems, n_dim (3), n_verts_per_elem)
        verts_transformed_tet_centered = verts_transformed_tet - verts_transformed_tet[:,:,0].unsqueeze(2)
        verts_transformed_tet_centered_three = verts_transformed_tet_centered[:,:,1:].clone()

        deform_gradient = torch.matmul(verts_transformed_tet_centered_three, self.D_m_inv)

        return deform_gradient

def svd_rv(deform_gradient):
    # deform_gradient.retain_grad()
    deform_gradient_cpu = deform_gradient.cpu()
    U_cpu, Sigma_cpu, V_cpu = torch.svd(deform_gradient_cpu)
    U, Sigma, V = U_cpu.cuda(), Sigma_cpu.cuda(), V_cpu.cuda()

    # V.retain_grad()
    # deform_gradient.retain_grad()
    # test = torch.norm(V, dim=(1,2), p=2).mean()
    # set_trace()

    L = torch.eye(3).repeat(U.shape[0], 1, 1).cuda()
    L[:, 2,2] = torch.det(torch.matmul(U, V.permute(0,2,1)))

    # see where to pull the reflection out of
    detU = torch.det(U)
    detV = torch.det(V)
    logical_and1 = ((detU<0) & (detV>0))
    logical_and2 = ((detU>0) & (detV<0))

    # push the reflection to the diagonal
    U_new = U.clone()
    V_new = V.clone()
    U_new[logical_and1] = torch.matmul(U[logical_and1], L[logical_and1])
    V_new[logical_and2] = torch.matmul(V[logical_and2], L[logical_and2])

    Sigma = torch.matmul(torch.diag_embed(Sigma), L)

    return U_new, Sigma, V_new

def get_arap_energy(deform_gradient):
    U, Sigma, V = svd_rv(deform_gradient)
    S = torch.matmul(torch.matmul(V, Sigma), V.permute(0,2,1))

    arap_energy_tet = torch.pow(torch.norm(S - torch.eye(3).repeat(S.shape[0], 1, 1).cuda(), dim=(1,2), p=2), 2)
    arap_energy_hex = arap_energy_tet.reshape(-1,6).mean(dim=1)

    return arap_energy_hex

def softmax_DP(input_tensor1d_orig, base_exp='e'):
    input_tensor1d = input_tensor1d_orig - input_tensor1d_orig.max()
    if base_exp == 'e':
        output = torch.exp(input_tensor1d)/torch.sum(torch.exp(input_tensor1d))
    else:
        output = torch.pow(base_exp, input_tensor1d)/torch.sum(torch.pow(base_exp, input_tensor1d))

    return output

class CalcDeformationGradientFEM():
    def __init__(self, verts_template, elems_template):
        """ verts and elems should be torch.Tensors, elems_template should be tet """
        self.elems_template = elems_template

        n_elems = self.elems_template.shape[0]
        verts_template_hex = verts_template[self.elems_template].permute(0,2,1)

        quadrature_pt_list = [
            [0.5, 0.5, 0.5],
            [0.25, 0.5, 0.5],
            [0.75, 0.5, 0.5],
            [0.5, 0.25, 0.5],
            [0.5, 0.75, 0.5],
            [0.5, 0.5, 0.25],
            [0.5, 0.5, 0.75]
        ]

        H_Dm_inv_list = []

        with torch.no_grad():
            for quadrature_pt in quadrature_pt_list:
                H = self.get_H(quadrature_pt).repeat(n_elems, 1, 1)
                H_Dm_inv_list.append(torch.matmul(H, torch.inverse(torch.matmul(verts_template_hex, H))))

            self.H_Dm_inv = torch.cat(H_Dm_inv_list, dim=0)

    def get_H(self, quadrature_pt):
        u,v,w = quadrature_pt
        H = torch.tensor([
            [-(1-v)*(1-w), -(1-u)*(1-w), -(1-u)*(1-v)],
            [+(1-v)*(1-w), -(1+u)*(1-w), -(1+u)*(1-v)],
            [-(1+v)*(1-w), +(1-u)*(1-w), -(1-u)*(1+v)],
            [+(1+v)*(1-w), +(1+u)*(1-w), -(1+u)*(1+v)],
            [-(1-v)*(1+w), -(1-u)*(1+w), +(1-u)*(1-v)],
            [+(1-v)*(1+w), -(1+u)*(1+w), +(1+u)*(1-v)],
            [-(1+v)*(1+w), +(1-u)*(1+w), +(1-u)*(1+v)],
            [+(1+v)*(1+w), +(1+u)*(1+w), +(1+u)*(1+v)],
        ], dtype=torch.get_default_dtype(), device='cuda')

        return H

    def __call__(self, verts_transformed):
        verts_transformed_hex = verts_transformed.squeeze()[self.elems_template].permute(0,2,1).repeat(7,1,1) # (n_elems*n_quadrature_pts, n_dim (3), n_verts_per_elem)

        deform_gradient = torch.matmul(verts_transformed_hex, self.H_Dm_inv)

        return deform_gradient

def get_arap_energy_hex(deform_gradient):
    U, Sigma, V = svd_rv(deform_gradient)
    S = torch.matmul(torch.matmul(V, Sigma), V.permute(0,2,1))

    arap_energy_quadrature_pts = torch.pow(torch.norm(S - torch.eye(3).repeat(S.shape[0], 1, 1).cuda(), dim=(1,2), p=2), 2)
    arap_energy_hex = arap_energy_quadrature_pts.reshape(7,-1).mean(dim=0)

    return arap_energy_hex

class CalcGraphLaplacian():
    def __init__(self, verts_template, edges_template):
        with torch.no_grad():
            # all for calculating L matrix
            verts_packed = verts_template  # torch.tensor (sum(V_n), 3)
            edges_packed = edges_template # torch.tensor (sum(E_n), 3)
            V = verts_packed.shape[0]  # sum(V_n)

            e0, e1 = edges_packed.unbind(1)

            idx01 = torch.stack([e0, e1], dim=1)  # (sum(E_n), 2)
            idx10 = torch.stack([e1, e0], dim=1)  # (sum(E_n), 2)
            idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*sum(E_n))

            # First, we construct the adjacency matrix,
            # i.e. A[i, j] = 1 if (i,j) is an edge, or
            # A[e0, e1] = 1 &  A[e1, e0] = 1
            ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts_template.device)
            A = torch.sparse.FloatTensor(idx, ones, (V, V))

            # the sum of i-th row of A gives the degree of the i-th vertex
            deg = torch.sparse.sum(A, dim=1).to_dense()

            # We construct the Laplacian matrix by adding the non diagonal values
            # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
            deg0 = deg[e0]
            deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
            deg1 = deg[e1]
            deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
            val = torch.cat([deg0, deg1])
            L = torch.sparse.FloatTensor(idx, val, (V, V))

            # Then we add the diagonal values L[i, i] = -1.
            # idx = torch.arange(V, device=self.device)
            # DP: add -1 to diagonal values that only have connections to at least one other vertex
            idx = edges_packed.unique()
            idx = torch.stack([idx, idx], dim=0)
            ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts_template.device)
            L -= torch.sparse.FloatTensor(idx, ones, (V, V))

            self.L = L
            self.laplacian_template = self.L.mm(verts_packed)

    def __call__(self, verts_transformed):
        laplacian_transformed = self.L.mm(verts_transformed)
        laplacian_diff = torch.pow(torch.norm(self.laplacian_template - laplacian_transformed, dim=1), 2)

        return laplacian_diff

def mesh_laplacian_smoothing_DP(meshes):
    """
    adjusting pytorch3d mesh_laplacian_smoothing to only consider connected vertices
    """

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    # num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    edges_packed = meshes.edges_packed()
    edges_packed_to_mesh_idx = meshes.edges_packed_to_mesh_idx()
    num_connected_verts_per_mesh = []
    for idx in range(N):
        num_connected_verts_per_mesh.append(edges_packed[edges_packed_to_mesh_idx == idx].unique().shape[0])
    num_connected_verts_per_mesh = torch.tensor(num_connected_verts_per_mesh, dtype=torch.int32, device=meshes.device)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_connected_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
    weights = 1.0 / weights.float()

    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        L = laplacian_packed_DP(meshes)

    loss = L.mm(verts_packed)
    loss = loss.norm(dim=1)

    loss = loss * weights
    return loss.sum() / N

def laplacian_packed_DP(meshes):
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
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=meshes.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    # idx = torch.arange(V, device=self.device)
    # DP: add -1 to diagonal values that only have connections to at least one other vertex
    idx = edges_packed.unique()
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=meshes.device)
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L

def get_mesh_correspondence_edge_loss(mesh1, mesh2, edges_packed=None):
    """
    Computes mesh edge length regularization loss averaged across all meshes
    in a batch. Each mesh contributes equally to the final loss, regardless of
    the number of edges per mesh in the batch by weighting each mesh with the
    inverse number of edges. For example, if mesh 3 (out of N) has only E=4
    edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
    contribute to the final loss.

    Args:
        meshes: Meshes object with a batch of meshes.
        target_length: Resting value for the edge length.

    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """    
    if mesh1.edges_packed().shape == mesh2.edges_packed().shape:
        if not torch.all(torch.eq(mesh1.edges_packed(), mesh2.edges_packed())):
            raise ValueError('mesh1 and mesh2 must have mesh correspondence')
    else:
        raise ValueError('mesh1 and mesh2 must have mesh correspondence')
    
    N = len(mesh1)

    if edges_packed is None:
        edges_packed = mesh1.edges_packed()  # (sum(E_n), 2)
    verts_packed1 = mesh1.verts_packed()  # (sum(V_n), 3)
    verts_packed2 = mesh2.verts_packed()  # (sum(V_n), 3)
    
    # Determine the weight for each edge based on the number of edges in the
    # mesh it corresponds to.
    # TODo (nikhilar) Find a faster way of computing the weights for each edge
    # as this is currently a bottleneck for meshes with a large number of faces.
    # edge_to_mesh_idx = mesh1.edges_packed_to_mesh_idx()  # (sum(E_n), )
    # num_edges_per_mesh = mesh1.num_edges_per_mesh()  # N
    # weights = num_edges_per_mesh.gather(0, edge_to_mesh_idx)
    # weights = 1.0 / weights.float()
    weights = 1.0 / edges_packed.shape[0] # DP: assume one mesh

    verts_edges1 = verts_packed1[edges_packed] # (n_edges, 2, 3)
    verts_edges2 = verts_packed2[edges_packed] # (n_edges, 2, 3)
    mesh1_v0, mesh1_v1 = verts_edges1.unbind(1) # (n_edges, 3)
    mesh2_v0, mesh2_v1 = verts_edges2.unbind(1) # (n_edges, 3)
    
    mesh1_edge_lengths = (mesh1_v0 - mesh1_v1).norm(dim=1, p=2)
    mesh2_edge_lengths = (mesh2_v0 - mesh2_v1).norm(dim=1, p=2)
    
    loss = (mesh1_edge_lengths/mesh1_edge_lengths.max() - mesh2_edge_lengths/mesh2_edge_lengths.max())**2
    loss_normalized = loss * weights # weights is 1/|E| for each mesh, so this effectively makes summing loss_normalized the average
    
    return loss_normalized.sum() / N # still need the N to do average over multiple different meshes

def mesh_edge_loss_DP(meshes, target_length: float = 0.0, edges_packed=None):
    """
    Args:
        meshes: Meshes object with a batch of meshes.
        target_length: Resting value for the edge length.

    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    if edges_packed is None:
        edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)

    # Determine the weight for each edge based on the number of edges in the
    # mesh it corresponds to.
    # TODO (nikhilar) Find a faster way of computing the weights for each edge
    # as this is currently a bottleneck for meshes with a large number of faces.
    # edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(E_n), )
    # num_edges_per_mesh = meshes.num_edges_per_mesh()  # N
    # weights = num_edges_per_mesh.gather(0, edge_to_mesh_idx)
    # weights = 1.0 / weights.float()
    weights = 1.0 / edges_packed.shape[0] # DP: assume one mesh

    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)
    loss = ((v0 - v1).norm(dim=1, p=2) - target_length) ** 2.0
    loss = loss * weights

    return loss.sum() / N


def get_mesh_uniformity_loss(mesh):
    N = len(mesh)
    
    edges_packed = mesh.edges_packed()  # (sum(E_n), 3)
    verts_packed = mesh.verts_packed()  # (sum(V_n), 3)
    edge_to_mesh_idx = mesh.edges_packed_to_mesh_idx()  # (sum(E_n), )
    num_edges_per_mesh = mesh.num_edges_per_mesh()  # N
    
    # Determine the weight for each edge based on the number of edges in the
    # mesh it corresponds to.
    # TODo (nikhilar) Find a faster way of computing the weights for each edge
    # as this is currently a bottleneck for meshes with a large number of faces.
    weights = num_edges_per_mesh.gather(0, edge_to_mesh_idx)
    weights = 1.0 / weights.float()
    
    verts_edges = verts_packed[edges_packed] # (n_edges, 2, 3)
    mesh_v0, mesh_v1 = verts_edges.unbind(1) # (n_edges, 3)
    
    mesh_edge_lengths = (mesh_v0 - mesh_v1).norm(dim=1, p=2)
    
    edge_length_variance = ((mesh_edge_lengths - mesh_edge_lengths.mean()) ** 2.0)*weights # weights is 1/|E| for each mesh, so summing this is averaging variance for each mesh
    
    return edge_length_variance.sum() / N

def get_laplacian_loss_total(verts_output_list, template_verts_list, template_faces_list):
    dtype = template_verts_list[0].dtype
    device = template_verts_list[0].device
    curvature_store = torch.zeros([len(verts_output_list)], dtype=dtype, device=device)
    
    for list_idx, (verts_output, template_verts, template_faces) in enumerate(zip(verts_output_list, template_verts_list, template_faces_list)):
        laplacian1 = LaplacianLoss(template_faces)
        Lx_template = laplacian1(template_verts)
        
        Lx_output = laplacian1(verts_output)
        
        curvature_store[list_idx] = torch.norm(Lx_output - Lx_template, p=2, dim=2).mean()
        
    return curvature_store.mean()

def get_registration_loss(pred_img, target_img):
    return torch.norm(pred_img - target_img)

def get_mag_displacement_field(displacement_field_tuple):
    mag = 0
    
    for displacement_field in displacement_field_tuple:
        mag += torch.norm(displacement_field)
        
    return mag

def get_gradient_displacement_field(displacement_field_tuple, eps=1e-6):
    device = displacement_field_tuple[0].device
    n_batch = displacement_field_tuple[0].shape[0]

    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.get_default_dtype(), device=device).repeat(3,1,1).repeat(n_batch,1,1,1,1)

    sobel_y = torch.tensor([[-1,-2,-1],
                            [ 0, 0, 0],
                            [ 1, 2, 1]], dtype=torch.get_default_dtype(), device=device).repeat(3,1,1).repeat(n_batch,1,1,1,1)

    sobel_z = torch.tensor([[[-1,-1,-1],
                             [-2,-2,-2],
                             [-1,-1,-1]],
                            [[ 0, 0, 0],
                             [ 0, 0, 0],
                             [ 0, 0, 0]],
                            [[ 1, 1, 1],
                             [ 2, 2, 2],
                             [ 1, 1, 1]]], dtype=torch.get_default_dtype(), device=device).repeat(n_batch,1,1,1,1)

    grad_mag_squared = 0
    grad_output = []

    for displacement_field in displacement_field_tuple:
        for dim in range(3):
            displacement_field_1dim = displacement_field[:,dim,:,:,:].unsqueeze(1)
            dx = F.conv3d(displacement_field_1dim, sobel_x, padding=1)
            dy = F.conv3d(displacement_field_1dim, sobel_y, padding=1)
            dz = F.conv3d(displacement_field_1dim, sobel_z, padding=1)

    #        grad_output.append({'dx': dx,
    #                            'dy': dy,
    #                            'dz': dz})

            grad_mag_squared += (dx.pow(2).sum() + dy.pow(2).sum() + dz.pow(2).sum())

    if grad_mag_squared == 0:
        grad_mag = (grad_mag_squared + eps).sqrt()
    else:
        grad_mag = grad_mag_squared.sqrt()

    return grad_mag, grad_output

def get_squared_2nd_deriv_displacement_field(displacement_field_tuple, eps=1e-6):
    device = displacement_field_tuple[0].device
    n_batch = displacement_field_tuple[0].shape[0]
    
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.get_default_dtype(), device=device).repeat(3,1,1).repeat(n_batch,1,1,1,1)
    
    sobel_y = torch.tensor([[-1,-2,-1],
                            [ 0, 0, 0],
                            [ 1, 2, 1]], dtype=torch.get_default_dtype(), device=device).repeat(3,1,1).repeat(n_batch,1,1,1,1)
    
    sobel_z = torch.tensor([[[-1,-1,-1],
                             [-2,-2,-2],
                             [-1,-1,-1]],
                            [[ 0, 0, 0],
                             [ 0, 0, 0],
                             [ 0, 0, 0]],
                            [[ 1, 1, 1],
                             [ 2, 2, 2],
                             [ 1, 1, 1]]], dtype=torch.get_default_dtype(), device=device).repeat(n_batch,1,1,1,1)
    
    deriv_mag_squared = 0
    deriv_output = []
    
    for displacement_field in displacement_field_tuple:
        for dim in range(3):
            displacement_field_1dim = displacement_field[:,dim,:,:,:].unsqueeze(1)
            dxdx = F.conv3d(F.conv3d(displacement_field_1dim, sobel_x, padding=1), sobel_x, padding=1)
            dydy = F.conv3d(F.conv3d(displacement_field_1dim, sobel_y, padding=1), sobel_y, padding=1)
            dzdz = F.conv3d(F.conv3d(displacement_field_1dim, sobel_z, padding=1), sobel_z, padding=1)
            dxdy = F.conv3d(F.conv3d(displacement_field_1dim, sobel_x, padding=1), sobel_y, padding=1)
            dxdz = F.conv3d(F.conv3d(displacement_field_1dim, sobel_x, padding=1), sobel_z, padding=1)
            dydz = F.conv3d(F.conv3d(displacement_field_1dim, sobel_y, padding=1), sobel_z, padding=1)
        
    #        deriv_output.append({'dxdx': dxdx,
    #                             'dydy': dydy,
    #                             'dzdz': dzdz,
    #                             'dxdy': dxdy,
    #                             'dxdz': dxdz,
    #                             'dydz': dydz})
        
            deriv_mag_squared += (dxdx.pow(2).sum() + dydy.pow(2).sum() + dzdz.pow(2).sum() + 
                                  2*dxdy.pow(2).sum() + 2*dxdz.pow(2).sum() + 2*dydz.pow(2).sum())
            
    if deriv_mag_squared == 0:
        deriv_mag = (deriv_mag_squared + eps).sqrt()
    else:
        deriv_mag = deriv_mag_squared.sqrt()
            
    return deriv_mag, deriv_output

def get_jacobian_penalty(displacement_field_tuple, threshold=0.3, eps=1e-6):
    '''
    displacement_field_tuple = length 4, each element is torch.tensor of shape (n_batch, 3, h, w, d)
    '''
    
    device = displacement_field_tuple[0].device
    n_batch = displacement_field_tuple[0].shape[0]
    
    sobel_x = 1/24*torch.tensor([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=torch.get_default_dtype(), device=device).repeat(3,1,1).repeat(n_batch,1,1,1,1)
    
    sobel_y = 1/24*torch.tensor([[-1,-2,-1],
                                 [ 0, 0, 0],
                                 [ 1, 2, 1]], dtype=torch.get_default_dtype(), device=device).repeat(3,1,1).repeat(n_batch,1,1,1,1)
    
    sobel_z = 1/24*torch.tensor([[[-1,-1,-1],
                                  [-2,-2,-2],
                                  [-1,-1,-1]],
                                 [[ 0, 0, 0],
                                  [ 0, 0, 0],
                                  [ 0, 0, 0]],
                                 [[ 1, 1, 1],
                                  [ 2, 2, 2],
                                  [ 1, 1, 1]]], dtype=torch.get_default_dtype(), device=device).repeat(n_batch,1,1,1,1)
    
    y_mesh, z_mesh, x_mesh = np.meshgrid(np.arange(displacement_field_tuple[0].shape[2]),
                                         np.arange(displacement_field_tuple[0].shape[3]),
                                         np.arange(displacement_field_tuple[0].shape[4]))
    
    x_mesh = torch.tensor(x_mesh, dtype=torch.get_default_dtype(), device=device)
    y_mesh = torch.tensor(y_mesh, dtype=torch.get_default_dtype(), device=device)
    z_mesh = torch.tensor(z_mesh, dtype=torch.get_default_dtype(), device=device)
    
    ''' maybe just need to change mesh order '''
    
    x_mesh = x_mesh.repeat(n_batch,1,1,1,1)
    y_mesh = y_mesh.repeat(n_batch,1,1,1,1)
    z_mesh = z_mesh.repeat(n_batch,1,1,1,1)
    
    h, w, d = displacement_field_tuple[0].shape[2:]
#    penalty_matrix = torch.zeros([n_batch, 1, h-2, w-2, d-2], dtype=torch.get_default_dtype(), device=device)
    
    penalty_sum = 0
    
    for displacement_field in displacement_field_tuple:
#        a = F.conv3d(displacement_field[:,0,:,:,:].unsqueeze(1), sobel_x, padding=1) # dFdx
#        b = F.conv3d(displacement_field[:,0,:,:,:].unsqueeze(1), sobel_y, padding=1) # dFdy
#        c = F.conv3d(displacement_field[:,0,:,:,:].unsqueeze(1), sobel_z, padding=1) # dFdz
#        
#        d = F.conv3d(displacement_field[:,1,:,:,:].unsqueeze(1), sobel_x, padding=1) # dGdx
#        e = F.conv3d(displacement_field[:,1,:,:,:].unsqueeze(1), sobel_y, padding=1) # dGdy
#        f = F.conv3d(displacement_field[:,1,:,:,:].unsqueeze(1), sobel_z, padding=1) # dGdz
#        
#        g = F.conv3d(displacement_field[:,2,:,:,:].unsqueeze(1), sobel_x, padding=1) # dHdx
#        h = F.conv3d(displacement_field[:,2,:,:,:].unsqueeze(1), sobel_y, padding=1) # dHdy
#        i = F.conv3d(displacement_field[:,2,:,:,:].unsqueeze(1), sobel_z, padding=1) # dHdz
#        
#        det_jacobian = a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)
#        
#        penalty_matrix[det_jacobian < threshold] = threshold**2/(det_jacobian.pow(2)[det_jacobian < threshold] + eps) - 2
#        
#        penalty_sum = penalty_matrix.sum()
        
        displacement_field_x = (displacement_field[:,0,:,:,:]*displacement_field.shape[2]).unsqueeze(1)
        displacement_field_y = (displacement_field[:,1,:,:,:]*displacement_field.shape[3]).unsqueeze(1)
        displacement_field_z = (displacement_field[:,2,:,:,:]*displacement_field.shape[4]).unsqueeze(1)
        
        x_new = x_mesh + displacement_field_x
        y_new = y_mesh + displacement_field_y
        z_new = z_mesh + displacement_field_z
        
        a_jac = F.conv3d(x_new, sobel_x) # dFdx
        b_jac = F.conv3d(x_new, sobel_y) # dFdy
        c_jac = F.conv3d(x_new, sobel_z) # dFdz
        
        d_jac = F.conv3d(y_new, sobel_x) # dGdx
        e_jac = F.conv3d(y_new, sobel_y) # dGdy
        f_jac = F.conv3d(y_new, sobel_z) # dGdz
        
        g_jac = F.conv3d(z_new, sobel_x) # dHdx
        h_jac = F.conv3d(z_new, sobel_y) # dHdy
        i_jac = F.conv3d(z_new, sobel_z) # dHdz
        
        det_jacobian = a_jac*(e_jac*i_jac-f_jac*h_jac) - b_jac*(d_jac*i_jac-f_jac*g_jac) + c_jac*(d_jac*h_jac-e_jac*g_jac)
        
        penalty_sum += torch.norm(det_jacobian - 1)
#        penalty_sum = (-torch.log(det_jacobian)).sum()
        
#        penalty_matrix[det_jacobian < threshold] = threshold**2/(det_jacobian.pow(2)[det_jacobian < threshold] + eps) - 2
#        
#        penalty_sum = penalty_matrix.sum()
        
#        set_trace()
        
    return penalty_sum

class LaplacianLoss(torch.autograd.Function):
    def __init__(self, faces):
        """
        Faces is B x F x 3, cuda torch Variabe.
        Reuse faces.
        """
        self.F_np = faces.data.cpu().numpy()
        self.F = faces.data
        self.L = None

    def convert_as(self, src, trg):
        src = src.type_as(trg)
        if src.is_cuda:
            src = src.cuda(device=trg.get_device())
        return src

    def cotangent(self, V, F):
        """
        Input:
          V: B x N x 3
          F: B x F  x3p
        Outputs:
          C: B x F x 3 list of cotangents corresponding
            angles for triangles, columns correspond to edges 23,31,12
            
        B x F x 3 x 3
        """
        indices_repeat = torch.stack([F, F, F], dim=2)
        
        #v1 is the list of first triangles B*F*3, v2 second and v3 third
        v1 = torch.gather(V, 1, indices_repeat[:, :, :, 0].long())
        v2 = torch.gather(V, 1, indices_repeat[:, :, :, 1].long())
        v3 = torch.gather(V, 1, indices_repeat[:, :, :, 2].long())
        
        l1 = torch.sqrt(((v2 - v3)**2).sum(2)) #distance of edge 2-3 for every face B*F
        l2 = torch.sqrt(((v3 - v1)**2).sum(2))
        l3 = torch.sqrt(((v1 - v2)**2).sum(2))
        
        # semiperimieters
        sp = (l1 + l2 + l3) * 0.5
        
        # Heron's formula for area #FIXME why the *2 ? Heron formula is without *2 It's the 0.5 than appears in the (0.5(cotalphaij + cotbetaij))
        A = 2*torch.sqrt( sp * (sp-l1)*(sp-l2)*(sp-l3))
        
        # Theoreme d Al Kashi : c2 = a2 + b2 - 2ab cos(angle(ab))
        cot23 = (l2**2 + l3**2 - l1**2)
        cot31 = (l1**2 + l3**2 - l2**2)
        cot12 = (l1**2 + l2**2 - l3**2)
        
        # 2 in batch #proof page 98 http://www.cs.toronto.edu/~jacobson/images/alec-jacobson-thesis-2013-compressed.pdf
        C = torch.stack([cot23, cot31, cot12], 2) / torch.unsqueeze(A, 2) / 4
        
        return C
    
    def forward(self, V):
        """
        If forward is explicitly called, V is still a Parameter or Variable
        But if called through __call__ it's a tensor.
        This assumes __call__ was used.
        
        Input:
           V: B x N x 3
           F: B x F x 3
        Outputs: Lx B x N x 3
        
         Numpy also doesnt support sparse tensor, so stack along the batch
        """

        V_np = V.cpu().numpy()
        batchV = V_np.reshape(-1, 3)

        if self.L is None:
            # print('Computing the Laplacian!')
            # Compute cotangents
            C = self.cotangent(V, self.F)
            C_np = C.cpu().numpy()
            batchC = C_np.reshape(-1, 3)
            # Adjust face indices to stack:
            offset = np.arange(0, V.size(0)).reshape(-1, 1, 1) * V.size(1)
            F_np = self.F_np + offset
            batchF = F_np.reshape(-1, 3)

            rows = batchF[:, [1, 2, 0]].reshape(-1) #1,2,0 i.e to vertex 2-3 associate cot(23)
            cols = batchF[:, [2, 0, 1]].reshape(-1) #2,0,1 This works because triangles are oriented ! (otherwise 23 could be associated to more than 1 cot))

            # Final size is BN x BN
            BN = batchV.shape[0]
            L = sparse.csr_matrix((batchC.reshape(-1), (rows, cols)), shape=(BN,BN))
            L = L + L.T
            # np.sum on sparse is type 'matrix', so convert to np.array
            M = sparse.diags(np.array(np.sum(L, 1)).reshape(-1), format='csr')
            L = L - M
            # remember this
            self.L = L
            # TODO The normalization by the size of the voronoi cell is missing.
            # import matplotlib.pylab as plt
            # plt.ion()
            # plt.clf()
            # plt.spy(L)
            # plt.show()
            # import ipdb; ipdb.set_trace()

        Lx = self.L.dot(batchV).reshape(V_np.shape)

        return self.convert_as(torch.Tensor(Lx), V)

    def backward(self, grad_out):
        """
        Just L'g = Lg
        Args:
           grad_out: B x N x 3
        Returns:
           grad_vertices: B x N x 3
        """
        g_o = grad_out.cpu().numpy()
        # Stack
        g_o = g_o.reshape(-1, 3)
        Lg = self.L.dot(g_o).reshape(grad_out.shape)

        return self.convert_as(torch.Tensor(Lg), grad_out)

class NegativeSoftDiceBinary():
    def __init__(self, squared_cardinality_den=False, eps=1e-6):
        self.squared_cardinality_den = squared_cardinality_den
        self.eps = eps
    
    def __call__(self, pred, target):
        return negative_soft_dice_binary_torch(pred, target, squared_cardinality_den=self.squared_cardinality_den, eps=self.eps)

class NegativeSoftDiceMultilabel():
    def __init__(self, squared_cardinality_den=False, eps=1e-6):
        self.squared_cardinality_den = squared_cardinality_den
        self.eps = eps
    
    def __call__(self, pred, target):
        ''' 
        pred = torch.Tensor
        target = list of torch.Tensors, each torch.Tensor has shape [n_batch, H, W, D]
        
        pred.shape = [n_batch, n_channels, H, W, D]
        target = [n_batch, n_channels, H, W, D]
        '''
        
        assert pred.shape[1] == target.shape[1], 'n_channels of network output should be same as n_channels of labels'
        
        dice_store = torch.zeros(target.shape[1], device=pred.device)
        
        for i in range(target.shape[1]):
            dice_store[i] = torch.neg(negative_soft_dice_binary_torch(pred[:,i,:,:,:], target[:,i,:,:,:], squared_cardinality_den=self.squared_cardinality_den, eps=self.eps))
        
        output = dice_store.mean()
#        output = torch.prod(dice_store)
        
        return torch.neg(output)
    
class STN_seg_2_losses():
    def __init__(self):
        self.seg_loss = NegativeSoftDiceMultilabel()
        self.stn_loss = NegativeSoftDiceMultilabel()
        # self.stn_loss = HausdorffDice4LabelsLoss()
    
    def __call__(self, ct0_seg_output, ct0_seg_label, ct0_all_labels_cropped, ct1_all_labels_cropped):
        ''' 
        pred = torch.Tensor
        target = list of torch.Tensors, each torch.Tensor has shape [n_batch, H, W, D]
        
        pred.shape = [n_batch, n_channels, H, W, D]
        target = [n_batch, n_channels, H, W, D]
        '''
        
        seg_loss = self.seg_loss(ct0_seg_output, ct0_seg_label)
        
        stn_loss = self.stn_loss(ct0_all_labels_cropped, ct1_all_labels_cropped)
        
        # print(' ')
        # print('seg_loss')
        # print(seg_loss)
        # print(' ')
        # print('stn_loss')
        # print(stn_loss)
        # print(' ')
        
        output = seg_loss + stn_loss

        # output = stn_loss
        
        return output

class HausdorffDice4LabelsLoss():
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        
        self.hausdorff_loss_fn = AveragedHausdorffLoss()
        self.dice_loss_fn = NegativeSoftDiceBinary(squared_cardinality_den=False)
        self.aortic_root_channel = 0
        self.valve_channels = [1,2,3]
        
    def __call__(self, ct0_all_labels_cropped, ct1_all_labels_cropped):
        n_batch = ct0_all_labels_cropped.shape[0]
        
        dice = self.dice_loss_fn(ct0_all_labels_cropped[:,self.aortic_root_channel,:,:,:], ct1_all_labels_cropped[:,self.aortic_root_channel,:,:,:])
        
        hausdorff_store = torch.zeros(n_batch, len(self.valve_channels), device=ct0_all_labels_cropped.device)
        for batch_idx in range(n_batch):
            for channel_list_idx, channel_idx in enumerate(self.valve_channels):
                ct0_valve_label_idx = ct0_all_labels_cropped[batch_idx,channel_idx,:,:,:].nonzero().float()
                ct1_valve_label_idx = ct1_all_labels_cropped[batch_idx,channel_idx,:,:,:].nonzero().float()
                
                hausdorff_store[batch_idx, channel_list_idx] = self.hausdorff_loss_fn(ct0_valve_label_idx, ct1_valve_label_idx)
        
        hausdorff = hausdorff_store.pow(2).mean(dim=1).sqrt()
        
        print('hausdorff_aortic_root_loss: {}'.format(dice.cpu().item()))
        print('hausdorff_valves_loss: {}'.format(hausdorff.cpu().item()))
        
        loss = dice + hausdorff*self.alpha
        
        return loss


class AveragedHausdorffLoss(nn.Module):
    def __init__(self):
        super(nn.Module, self).__init__()

    def cdist(self, x, y):
        '''
        Input: x is a Nxd Tensor
               y is a Mxd Tensor
        Output: dist is a NxM matrix where dist[i,j] is the norm
               between x[i,:] and y[j,:]
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||
        '''
        differences = x.unsqueeze(1) - y.unsqueeze(0)
        distances = torch.sum(differences**2, -1).sqrt()
        return distances

    def __call__(self, set1, set2):
        """
        Compute the Averaged Hausdorff Distance function
         between two unordered sets of points (the function is symmetric).
         Batches are not supported, so squeeze your inputs first!
        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Averaged Hausdorff Distance between set1 and set2.
        """
        
        assert set1.ndimension() == 2, 'got %s' % set1.ndimension()
        assert set2.ndimension() == 2, 'got %s' % set2.ndimension()

        assert set1.size()[1] == set2.size()[1], \
            'The points in both sets must have the same number of dimensions, got %s and %s.'\
            % (set2.size()[1], set2.size()[1])

        d2_matrix = self.cdist(set1, set2)

        # Modified Chamfer Loss
        term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
        term_2 = torch.mean(torch.min(d2_matrix, 0)[0])

        res = term_1 + term_2

        return res


class WeightedBinaryCrossEntropy():
    def __init__(self, weights=None):
        self.weights = weights
    
    def __call__(self, pred, target):
        if self.weights is not None:
            assert len(self.weights) == 2
            
            loss = self.weights[0] * (target * torch.log(pred)) + \
                   self.weights[1] * ((1 - target) * torch.log(1 - pred))
        else:
            loss = target * torch.log(pred) + (1 - target) * torch.log(1 - pred)
    
        return torch.neg(torch.mean(loss))

class MultimodalElboLoss():
    def __init__(self):
        pass
    
    def __call__(self, 
                 ct_pred, ct_target,
                 aortic_root_pred, aortic_root_target,
                 valve_1_pred, valve_1_target,
                 valve_2_pred, valve_2_target,
                 valve_3_pred, valve_3_target,
                 mu, logvar):
        
        BCE = 0
        n_modalities = 0
        
        if ct_pred is not None and ct_target is not None:
            ct_BCE = F.binary_cross_entropy(ct_pred, ct_target)
            BCE += ct_BCE
            n_modalities += 1
            
        if aortic_root_pred is not None and aortic_root_target is not None:
            aortic_root_BCE = F.binary_cross_entropy(aortic_root_pred, aortic_root_target)
            BCE += aortic_root_BCE
            n_modalities += 1
            
        if valve_1_pred is not None and valve_1_target is not None:
            valve_1_BCE = F.binary_cross_entropy(valve_1_pred, valve_1_target)
            BCE += valve_1_BCE
            n_modalities += 1
            
        if valve_2_pred is not None and valve_2_target is not None:
            valve_2_BCE = F.binary_cross_entropy(valve_2_pred, valve_2_target)
            BCE += valve_2_BCE
            n_modalities += 1
            
        if valve_3_pred is not None and valve_3_target is not None:
            valve_3_BCE = F.binary_cross_entropy(valve_3_pred, valve_3_target)
            BCE += valve_3_BCE
            n_modalities += 1

        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # NOTE: we use lambda_i = 1 for all i since each modality is roughly equal
        ELBO = torch.mean(BCE / float(n_modalities) + KLD)
        
        return ELBO
    


def dice_binary_thresholded_np(y_pred, y_true, threshold=0.5, eps=1e-6):
    '''
    2*TP/((TP+FP) + (TP+FN))
    y_pred is softmax output
    y_true is one hot encoding of target
    y_pred and y_true size: b x X x Y x Z
    '''
    
#    sum_axes = tuple(range(0, len(y_pred.shape))) # y_pred and y_true size: X x Y x Z
    sum_axes = tuple(range(1, len(y_pred.shape)))
    
    y_pred_thresholded = (y_pred>threshold)
    y_true_thresholded = (y_true>threshold)
    
    numerator = 2*np.sum(y_pred_thresholded * y_true_thresholded, axis=sum_axes)
    denominator = np.sum(y_pred_thresholded, sum_axes) + np.sum(y_true_thresholded, sum_axes)
    dice_score = numerator / (denominator + eps)
    
    return dice_score[0]

def dice_binary_thresholded(y_pred, y_true, threshold=0.5, eps=1e-6):
    '''
    2*TP/((TP+FP) + (TP+FN))
    y_pred is softmax output
    y_true is one hot encoding of target
    y_pred and y_true size: b x X x Y x Z
    '''
    sum_axes = tuple(range(1, len(y_pred.shape)))
    
    y_pred_thresholded = (y_pred>threshold)
    y_true_thresholded = (y_true>threshold)
    
    numerator = 2*torch.sum(y_pred_thresholded * y_true_thresholded, sum_axes)
    denominator = torch.sum(y_pred_thresholded, sum_axes) + torch.sum(y_true_thresholded, sum_axes)
    dice_score = numerator.float() / (denominator.float() + eps)
    
    dice_score_sum = torch.sum(dice_score)
    
    return dice_score_sum

def dice_binary_thresholded_multilabel(y_pred, y_true, threshold=0.5, eps=1e-6):
    '''
    2*TP/((TP+FP) + (TP+FN))
    y_pred is softmax output
    y_true is one hot encoding of target
    y_pred and y_true size: b x c x X x Y x Z
    '''
    n_channel = y_pred.shape[1]
    
    binary_dice_store = torch.zeros([n_channel])
    for channel_idx in range(n_channel):        
        binary_dice = dice_binary_thresholded(y_pred[:,channel_idx,:,:,:], y_true[:,channel_idx,:,:,:], threshold=threshold, eps=eps)
        binary_dice_store[channel_idx] = binary_dice.item()
    
    dice_score = torch.mean(binary_dice_store)
    
    return dice_score, binary_dice_store

def negative_soft_dice_binary_torch(y_pred_torch, y_true_torch, squared_cardinality_den=True, eps=1e-6):
    '''
    2*TP/((TP+FP) + (TP+FN))
    y_pred is softmax output of shape (num_samples, num_classes)
    y_true is one hot encoding of target (shape= (num_samples, num_classes))
    y_pred and y_true size: b x X x Y x Z
    '''
    sum_axes = tuple(range(1,len(y_pred_torch.shape)))
    
    numerator = 2*torch.sum(y_pred_torch * y_true_torch, sum_axes)
    
    if squared_cardinality_den:
        denominator = torch.sum(torch.pow(y_pred_torch, 2), sum_axes) + torch.sum(torch.pow(y_true_torch,2), sum_axes)
    else:
        denominator = torch.sum(y_pred_torch, sum_axes) + torch.sum(y_true_torch, sum_axes)
    
    dice_scores = numerator / (denominator + eps)
    
    dice_score_sum = torch.sum(dice_scores)
    
    return torch.neg(dice_score_sum)

def soft_dice_np(y_pred, y_true, squared_cardinality_den=True, eps=1e-6):
    sum_axes = tuple(range(2,len(y_pred.shape)))
    
    numerator = 2*np.sum(y_pred * y_true, axis=sum_axes)
    
    if squared_cardinality_den:
        denominator = np.sum(np.power(y_pred, 2), sum_axes) + np.sum(np.power(y_true, 2), sum_axes)
    else:
        denominator = np.sum(y_pred, sum_axes) + np.sum(y_true, sum_axes)
    
    dice_scores = numerator / (denominator + eps)
    
    dice_score_sum = np.sum(dice_scores)
    
    return dice_score_sum

def soft_intersection_over_union(y_pred, y_true, eps=1e-6):
    '''
    TP/(TP+FP+FN)
    '''
    sum_axes = tuple(range(2,len(y_pred.shape)))
    
    numerator = np.sum(y_pred * y_true, sum_axes)
    
    denominator = np.sum(y_pred, sum_axes) + np.sum(y_true, sum_axes) - numerator
    
    return np.sum(numerator / (denominator + eps))

# def confusion_matrix_DP(y_pred, y_true, classify_threshold=0.5):
#     if not np.array_equal(y_pred, y_pred.astype(bool)):
#         y_pred = (y_pred >= classify_threshold).astype(int)
#     
#     cm = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1))
#     cm = cm.reshape(-1)
#     
#     return cm

def vae_loss_function(output_batch, labels_batch):
    recon_x, mu, logvar = output_batch
    x = labels_batch
    
#    BCE = torch.sqrt((recon_x-x).pow(2).sum()) # debugging
    # BCE = F.binary_cross_entropy(recon_x.view(-1, 64*64*32), x.view(-1, 64*64*32), reduction='sum')
    BCE = F.binary_cross_entropy(recon_x, x)
    
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

def ct_seq_mt_loss_function(output_batch, labels_batch):
    class_pred, seg_pred = output_batch
    class_true = labels_batch
#    class_true = torch.zeros(class_pred.shape).cuda()
#    for b in range(class_pred.shape[0]):
#        class_true[b, labels_batch[b]] = 1
    
    class_loss = F.binary_cross_entropy(class_pred, class_true)
    
    return class_loss

# def loss_fn(y_pred, y_true):
#     loss = -soft_dice_torch(y_pred, y_true, squared_cardinality_den=False)
#     return loss

# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
#    'soft_dice': soft_dice_np,
#    'soft_intersection_over_union': soft_intersection_over_union,
#    'confusion_matrix': confusion_matrix_DP
    # could add more metrics such as accuracy for each token type
}
