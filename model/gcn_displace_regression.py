# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 15:54:56 2018

@author: Daniel
"""

import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import utils_sp
import airlab_stuff as al
import igl
from scipy.spatial import cKDTree
import utils_cgal_related as utils_cgal

from pytorch3d.ops import GraphConv

from pdb import set_trace

class GCN_block(nn.Module):
    def __init__(self, n_feat_init, n_feat_hidden, n_feat_final):
        super(GCN_block, self).__init__()
        
        self.gconvs = nn.Sequential(GraphConv(n_feat_init, n_feat_hidden),
                                    nn.ReLU(),
                                    GraphConv(n_feat_hidden, n_feat_hidden),
                                    nn.ReLU(),
                                    GraphConv(n_feat_hidden, n_feat_hidden),
                                    nn.ReLU(),
                                    GraphConv(n_feat_hidden, n_feat_final))
    
    def forward(self, feats_sampled_verts, edges_packed):
        input_t = feats_sampled_verts
        
        for layer in self.gconvs:
            if isinstance(layer, GraphConv):
                input_t = layer(input_t, edges_packed)
            else:
                input_t = layer(input_t)
                
        return input_t

class Net(nn.Module):
    def __init__(self, params):
        super(Net, self).__init__()
        
        self.n_dim = 3
        self.n_channels_in = params.seg_net_n_channels_in
        self.n_channels_out = params.seg_net_n_channels_out
        self.base_n_filter = params.seg_net_base_n_filter
        self.input_shape = params.seg_net_input_size
        self.n_batch = params.batch_size
        device = 'cuda' if params.cuda else 'cpu'
        self.device = device

        self.verts_list_template, self.faces_list_template, self.extra_info_template = utils_sp.get_template_verts_faces_list(params.template_load_fn, params.template_P_phase)
        self.verts_list_template_torch = utils_sp.np_list_to_torch_list(self.verts_list_template, n_batch=self.n_batch, device=device)
        self.faces_list_template_torch = utils_sp.np_list_to_torch_list(self.faces_list_template, n_batch=self.n_batch, device=device, dtype=int)
        self.template_edges = utils_sp.get_edges_pyvista(utils_cgal.mesh_to_pyvista_UnstructuredGrid(self.verts_list_template[0], self.extra_info_template)) # self.extra_info_template is hex_elems
        self.template_edges = torch.tensor(self.template_edges, dtype=int, device=device)
        hex_mesh_pv = utils_cgal.mesh_to_pyvista_UnstructuredGrid(self.verts_list_template[0], self.extra_info_template)
        ext_surf = hex_mesh_pv.copy().extract_geometry().triangulate()
        ext_surf.compute_normals(inplace=True, point_normals=True, cell_normals=False)
        self.template_point_normals_torch = torch.tensor(ext_surf['Normals'], dtype=torch.get_default_dtype(), device=device).repeat(self.n_batch, 1, 1)

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=params.seg_net_dropout_rate)
        self.upscale = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(self.n_channels_in, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_c1_2 = nn.Conv3d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(self.base_n_filter)
        
        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2)
        self.inorm3d_c2 = nn.InstanceNorm3d(self.base_n_filter*2)
        
        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4)
        self.inorm3d_c3 = nn.InstanceNorm3d(self.base_n_filter*4)
        
        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
        self.inorm3d_c4 = nn.InstanceNorm3d(self.base_n_filter*8)
        
        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*16, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter*16, self.base_n_filter*16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*8)
        
        self.conv3d_l0 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*8, kernel_size = 1, stride=1, padding=0, bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(self.base_n_filter*8)
        
        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*16)
        self.conv3d_l1 = nn.Conv3d(self.base_n_filter*16, self.base_n_filter*8, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*4)
        
        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*8)
        self.conv3d_l2 = nn.Conv3d(self.base_n_filter*8, self.base_n_filter*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*2)
        
        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*4)
        self.conv3d_l3 = nn.Conv3d(self.base_n_filter*4, self.base_n_filter*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter)
        
        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter*2)
        self.conv3d_l4 = nn.Conv3d(self.base_n_filter*2, self.n_channels_out*self.n_dim, kernel_size=1, stride=1, padding=0, bias=False)

#        self.conv3d_l4 = nn.Conv3d(self.base_n_filter*2, self.base_n_filter, kernel_size=1, stride=1, padding=0, bias=False)
#        # ignoring deep supervision b/c want straight features for gcn feature input
        self.ds2_1x1_conv3d = nn.Conv3d(self.base_n_filter*8, self.n_channels_out*self.n_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(self.base_n_filter*4, self.n_channels_out*self.n_dim, kernel_size=1, stride=1, padding=0, bias=False)       
        self.conv3d_l4.weight.data.zero_()  # zero-ing out the weights (bias is None) to make displacement_field_tuple all zeros as initial values
        self.ds2_1x1_conv3d.weight.data.zero_() # zero-ing out the weights (bias is None) to make displacement_field_tuple all zeros as initial values 
        self.ds3_1x1_conv3d.weight.data.zero_() # zero-ing out the weights (bias is None) to make displacement_field_tuple all zeros as initial values
        
        # n_gcn_feat_init = self.base_n_filter*(2+4+8) + 6
        n_gcn_feat_init = self.base_n_filter*(2+4+8) + 3
        n_gcn_feat_hidden = 128
        n_gcn_feat_final = 3
        n_components = len(self.verts_list_template_torch)
        
        self.gconvs_components_list = nn.ModuleList([])
        for j in range(n_components):
            self.gconvs_components_list.append(GCN_block(n_gcn_feat_init, n_gcn_feat_hidden, n_gcn_feat_final))
    
    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm3d(feat_in),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'), # DP: nn.functional.interpolate instead of nn.UpSample
            # should be feat_in*2 or feat_in
            nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(feat_out),
            nn.LeakyReLU())
    
    def vert_align_3d_DP(self, feats_list, verts, orig_img_dim=64):
        '''
        feats_list: list of torch.tensor [n_batch, n_channel, h, w, d] where h,w,d can be different for each entry of list
        verts: torch.tensor [n_batch, n_verts, n_dim (e.g. 3 for 3d)]
        '''
        # verts in x,y,z physical coordinates (e.g. [0,63]). need to change to [-1,1] coordinate for grid_sample
        dim_conv = utils_sp.DimensionConverter(orig_img_dim)
        verts_grid = dim_conv.from_dim_size(verts)
        grid = verts_grid[:, None, None, :, :] # expanding from [n_batch, n_verts, n_dim] to [n_batch, 1, 1, n_verts, n_dim]
        
        # iterate through feats, which can have different h,w,d. but grid stays the same b/c coordinate system is [-1,1], normalized to image shape
        feats_sampled_list = []
        for feats in feats_list:
            feat_sampled = F.grid_sample(feats, grid, align_corners=True) # (n_batch, n_channel, 1, 1, n_verts)
            feat_sampled_reshape = feat_sampled.squeeze(2).squeeze(2).permute(0,2,1) # (n_batch, n_verts, n_channel)
            feats_sampled_list.append(feat_sampled_reshape)
        feats_sampled = torch.cat(feats_sampled_list, dim=2) # (n_batch, n_verts, sum of n_channels for each feat in list)
        
        return feats_sampled
    
    def forward(self, x):
        #  Level 1 context pathway
        conv3d_c1_1_out = self.conv3d_c1_1(x)
        residual_1 = conv3d_c1_1_out
        lrelu_conv3d_c1_1_out = self.lrelu(conv3d_c1_1_out)
        conv3d_c1_2_out = self.conv3d_c1_2(lrelu_conv3d_c1_1_out)
        conv3d_c1_2_out = self.dropout3d(conv3d_c1_2_out)
        lrelu_conv_c1_out = self.lrelu_conv_c1(conv3d_c1_2_out)
        # Element Wise Summation
        residual_lrelu_conv_c1_out = lrelu_conv_c1_out + residual_1
        context_1 = self.lrelu(residual_lrelu_conv_c1_out)
        inorm3d_c1_out = self.inorm3d_c1(residual_lrelu_conv_c1_out)
        lrelu_inorm3d_c1_out = self.lrelu(inorm3d_c1_out) # [1, 16, 64, 64, 64]
        
        # Level 2 context pathway
        conv3d_c2 = self.conv3d_c2(lrelu_inorm3d_c1_out)
        residual_2 = conv3d_c2
        norm_lrelu_conv_c2_out = self.norm_lrelu_conv_c2(conv3d_c2)
        norm_lrelu_conv_c2_out = self.dropout3d(norm_lrelu_conv_c2_out)
        norm_lrelu_conv_c2x2_out = self.norm_lrelu_conv_c2(norm_lrelu_conv_c2_out)
        residual_norm_lrelu_conv_c2x2_out = norm_lrelu_conv_c2x2_out + residual_2
        inorm3d_c2_out = self.inorm3d_c2(residual_norm_lrelu_conv_c2x2_out)
        lrelu_inorm3d_c2_out = self.lrelu(inorm3d_c2_out) # [1, 32, 32, 32, 32]
        context_2 = lrelu_inorm3d_c2_out
        
        # Level 3 context pathway
        conv3d_c3_out = self.conv3d_c3(lrelu_inorm3d_c2_out)
        residual_3 = conv3d_c3_out
        norm_lrelu_conv_c3_out = self.norm_lrelu_conv_c3(conv3d_c3_out)
        norm_lrelu_conv_c3_out = self.dropout3d(norm_lrelu_conv_c3_out)
        norm_lrelu_conv_c3x2_out = self.norm_lrelu_conv_c3(norm_lrelu_conv_c3_out)
        residual_norm_lrelu_conv_c3x2_out = norm_lrelu_conv_c3x2_out + residual_3
        inorm3d_c3_out = self.inorm3d_c3(residual_norm_lrelu_conv_c3x2_out)
        lrelu_inorm3d_c3_out = self.lrelu(inorm3d_c3_out) # [1, 64, 16, 16, 16]
        context_3 = lrelu_inorm3d_c3_out
        
        # Level 4 context pathway
        conv3d_c4_out = self.conv3d_c4(lrelu_inorm3d_c3_out)
        residual_4 = conv3d_c4_out
        norm_lrelu_conv_c4_out = self.norm_lrelu_conv_c4(conv3d_c4_out)
        norm_lrelu_conv_c4_out = self.dropout3d(norm_lrelu_conv_c4_out)
        norm_lrelu_conv_c4x2_out = self.norm_lrelu_conv_c4(norm_lrelu_conv_c4_out)
        residual_norm_lrelu_conv_c4x2_out = norm_lrelu_conv_c4x2_out + residual_4
        inorm3d_c4_out = self.inorm3d_c4(residual_norm_lrelu_conv_c4x2_out)
        lrelu_inorm3d_c4_out = self.lrelu(inorm3d_c4_out) # [1, 128, 8, 8, 8]
        context_4 = lrelu_inorm3d_c4_out
        
        # Level 5
        conv3d_c5_out = self.conv3d_c5(lrelu_inorm3d_c4_out)
        residual_5 = conv3d_c5_out
        norm_lrelu_conv_c5_out = self.norm_lrelu_conv_c5(conv3d_c5_out)
        norm_lrelu_conv_c5_out = self.dropout3d(norm_lrelu_conv_c5_out)
        norm_lrelu_conv_c5x2_out = self.norm_lrelu_conv_c5(norm_lrelu_conv_c5_out)
        residual_norm_lrelu_conv_c5x2_out = norm_lrelu_conv_c5x2_out + residual_5 # [1, 256, 4, 4, 4]
        upscale_l0_output = self.norm_lrelu_upscale_conv_norm_lrelu_l0(residual_norm_lrelu_conv_c5x2_out) # [1, 128, 8, 8, 8]
        
        conv3d_l0_out = self.conv3d_l0(upscale_l0_output)
        inorm3d_l0_out = self.inorm3d_l0(conv3d_l0_out)
        lrelu_inorm3d_l0_out = self.lrelu(inorm3d_l0_out) # [1, 128, 8, 8, 8]
        
        # Level 1 localization pathway
        context_4_lrelu_inorm3d_l0_out = torch.cat([lrelu_inorm3d_l0_out, context_4], dim=1) # [1, 256, 8, 8, 8]
        conv_norm_lrelu_l1_out = self.conv_norm_lrelu_l1(context_4_lrelu_inorm3d_l0_out)
        conv3d_l1_out = self.conv3d_l1(conv_norm_lrelu_l1_out) # [1, 128, 8, 8, 8]
        upscale_l1_out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(conv3d_l1_out) # [1, 64, 16, 16, 16]
        
        # Level 2 localization pathway
        context_3_upscale_l1_out = torch.cat([upscale_l1_out, context_3], dim=1) # [1, 128, 16, 16, 16]
        conv_norm_lrelu_l2_out = self.conv_norm_lrelu_l2(context_3_upscale_l1_out)
        ds2 = conv_norm_lrelu_l2_out
        conv3d_l2_out = self.conv3d_l2(conv_norm_lrelu_l2_out) # [1, 64, 16, 16, 16]
        upscale_l2_out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(conv3d_l2_out) # [1, 32, 32, 32, 32]
        
        # Level 3 localization pathway
        context_2_upscale_l2_out = torch.cat([upscale_l2_out, context_2], dim=1) # [1, 64, 32, 32, 32]
        conv_norm_lrelu_l3_out = self.conv_norm_lrelu_l3(context_2_upscale_l2_out)
        ds3 = conv_norm_lrelu_l3_out
        conv3d_l3_out = self.conv3d_l3(conv_norm_lrelu_l3_out) # [1, 32, 32, 32, 32]
        upscale_l3_out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(conv3d_l3_out) # [1, 16, 64, 64, 64]
        
        # Level 4 localization pathway
        context_1_upscale_l3_out = torch.cat([upscale_l3_out, context_1], dim=1) # [1, 32, 64, 64, 64]
        conv_norm_lrelu_l4_out = self.conv_norm_lrelu_l4(context_1_upscale_l3_out)
        ds4 = conv_norm_lrelu_l4_out # added by DP

#        # Do the next 3 lines if trying to upscale images and feed as one img instead of 3 separate ones with different h,w,d
#        ds2_upscaled = self.upscale(self.upscale(ds2))
#        ds3_upscaled = self.upscale(ds3)
#        img_feats = torch.cat([ds2_upscaled, ds3_upscaled, ds4], dim=1)
        
        # For single refinement step
        # run a separate gcn for each component, store feats_out for each component to do vertex feature averaging later (reaction on 02/11/2021: ?????)
        verts_list_transformed_torch = []
        displacement_field_tuple = []
        
        for verts, faces, gconvs_each_component in zip(self.verts_list_template_torch, self.faces_list_template_torch, self.gconvs_components_list):
            feats_sampled = self.vert_align_3d_DP([ds2, ds3, ds4], verts) # [N, V, F]
            # feats_sampled_verts = torch.cat([feats_sampled, verts, self.template_point_normals_torch], dim=2)
            feats_sampled_verts = torch.cat([feats_sampled, verts], dim=2)
            xyz_shift = gconvs_each_component(feats_sampled_verts.squeeze(), self.template_edges) # gcn_output, [N, 3?]
            xyz_shift = xyz_shift.unsqueeze(0)
            
            print('xyz_shift min and max')
            print([xyz_shift.min().item(), xyz_shift.max().item()])
            print(' ')
            
            # move verts
            verts_transformed = verts + xyz_shift
            verts_list_transformed_torch.append(verts_transformed)
            displacement_field_tuple.append(xyz_shift)
            
        displacement_field_tuple = tuple(displacement_field_tuple)
        seg_output_dummy = torch.zeros([1,4,64,64,64], device='cpu')
        
        return [seg_output_dummy, verts_list_transformed_torch, displacement_field_tuple], [self.verts_list_template_torch, self.faces_list_template_torch, self.extra_info_template]
