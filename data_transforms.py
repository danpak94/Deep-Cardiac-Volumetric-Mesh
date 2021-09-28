# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 12:28:43 2018

@author: Daniel
"""

import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter
from skimage import measure
import torch
import torch.nn.functional as F

import utils_gui
import transformations as tfm
import utils_sp

##

def ct_normalize(pixel_array, min_bound=-1000.0, max_bound=400.0):
    pix_normalized = (pixel_array - min_bound) / (max_bound - min_bound)
    pix_normalized[pix_normalized>1] = 1.
    pix_normalized[pix_normalized<0] = 0.
    
    return pix_normalized

def get_total_mean(ct_filepaths):
    mean_list = []
    num_pixels_list = []
    
    for f in ct_filepaths:
        pix_resampled = np.load(f)
        pix_normalized = ct_normalize(pix_resampled)
        
        mean_list.append(pix_normalized.mean())
        num_pixels_list.append(pix_normalized.size)

    mean_array = np.asarray(mean_list)
    num_pixels_array = np.asarray(num_pixels_list)
    
    total_mean = ((mean_array*num_pixels_array).sum())/(num_pixels_array.sum())
        
    return total_mean

def zero_center(ct_data, pixel_mean):
    ct_centered = ct_data - pixel_mean
    
    return ct_centered

##

class RandomAugmentSTN(object):
    '''
    Translate 3D image randomly, padding with 0's where image is not available
    Not manually implemented, using torch.nn.functional functions (F.grid_sample, etc.)
    '''
    def __init__(self, \
                 max_translate_ratio = 0.4, \
                 max_rotate_angles = [20,20,20], \
                 elastic_deform_sigma = [2,4], \
                 elastic_deform_alpha = [20,40], \
                 elastic_deform_chance = 0.9, \
                 crop_min_shape = [150,150,100], \
                 translate=True, rotate=True, elastic_deform=True, crop=True):
        '''
        e.g. if max_translate_ratio=0.2, we're saying we can move in each direction 20% of total pixels
        '''
        self.max_translate_ratio = max_translate_ratio
        self.max_rotate_angles = max_rotate_angles
        self.elastic_deform_sigma = elastic_deform_sigma
        self.elastic_deform_alpha = elastic_deform_alpha
        self.elastic_deform_chance = elastic_deform_chance
        self.crop_min_shape = crop_min_shape
        self.translate = translate
        self.rotate = rotate
        self.elastic_deform = elastic_deform
        self.crop = crop
        
        if self.crop:
            self.max_translate_ratio = 1.0
    
    def calc_cicrumcenter(self, lm1, lm2, lm3):
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
    
    def __call__(self, data_dict):
        src_dim1, src_dim2, src_dim3 = data_dict['ct'].shape
        
        if self.rotate:
            angle1 = ((torch.rand(1)-0.5)*2)*self.max_rotate_angles[0]*np.pi/180
            angle2 = ((torch.rand(1)-0.5)*2)*self.max_rotate_angles[1]*np.pi/180
            angle3 = ((torch.rand(1)-0.5)*2)*self.max_rotate_angles[2]*np.pi/180
            
            rot_mat1 = utils_sp.rotation_matrix_torch(angle1, [1,0,0])
            rot_mat2 = utils_sp.rotation_matrix_torch(angle2, [0,1,0])
            rot_mat3 = utils_sp.rotation_matrix_torch(angle3, [0,0,1])
            
            theta_rotation = torch.matmul(rot_mat3, torch.matmul(rot_mat2, rot_mat1))
        else:
            theta_rotation = torch.eye(4).unsqueeze(0)
        
        lm1 = data_dict['landmark1']
        lm2 = data_dict['landmark2']
        lm3 = data_dict['landmark3']
        
        circumcenter = np.dot(theta_rotation.numpy().squeeze()[0:3,0:3], self.calc_cicrumcenter(lm1, lm2, lm3)-np.array([src_dim1, src_dim2, src_dim3])/2)+np.array([src_dim1, src_dim2, src_dim3])/2
        
        if self.crop:            
            dst_dim1 = np.round(src_dim1 - (src_dim1 - self.crop_min_shape[0])*np.random.rand()).astype(int)
            dst_dim2 = np.round(src_dim2 - (src_dim2 - self.crop_min_shape[1])*np.random.rand()).astype(int)
            dst_dim3 = np.round(src_dim3 - (src_dim3 - self.crop_min_shape[2])*np.random.rand()).astype(int)
        else:
            dst_dim1, dst_dim2, dst_dim3 = src_dim1, src_dim2, src_dim3
        
        valve_upper_boundary = np.array([dst_dim1, dst_dim2, dst_dim3])/2 + np.sqrt(64**2*2)/2
        valve_lower_boundary = np.array([dst_dim1, dst_dim2, dst_dim3])/2 - np.sqrt(64**2*2)/2
        
        upper_limit = np.array([dst_dim1, dst_dim2, dst_dim3]) - valve_upper_boundary
        lower_limit = np.zeros(3) - valve_lower_boundary
        
        translate_center_valve = circumcenter - np.array([src_dim1, src_dim2, src_dim3])/2
        translations = translate_center_valve + np.random.rand(3)*(upper_limit-lower_limit) + lower_limit # uniform sampling from lower_limit to upper_limit
        
        if self.translate:
            theta_translation = torch.Tensor([[1,0,0,translations[2]],\
                                              [0,1,0,translations[1]],\
                                              [0,0,1,translations[0]],\
                                              [0,0,0,1]]).unsqueeze(0)
        else:
            theta_translation = torch.eye(4).unsqueeze(0)
        
        grid_shape = [1, dst_dim1, dst_dim2, dst_dim3, 3]
        
        if self.elastic_deform:
            if np.random.uniform() <= self.elastic_deform_chance:
                random_noise = np.random.uniform(low=-1.0, high=1.0, size=grid_shape)
                
                if isinstance(self.elastic_deform_sigma, list):
                    sigma = np.random.uniform(low=self.elastic_deform_sigma[0], high=self.elastic_deform_sigma[1])
                else:
                    sigma = self.elastic_deform_sigma
                    
                random_noise = np.random.uniform(low=-1.0, high=1.0, size=grid_shape)
                
                filtered_random_noise_dim1 = gaussian_filter(random_noise[:,:,:,:,0], sigma=sigma)/(src_dim1/2)
                filtered_random_noise_dim2 = gaussian_filter(random_noise[:,:,:,:,1], sigma=sigma)/(src_dim2/2)
                filtered_random_noise_dim3 = gaussian_filter(random_noise[:,:,:,:,2], sigma=sigma)/(src_dim3/2)
                filtered_random_noise = np.stack([filtered_random_noise_dim1, filtered_random_noise_dim2, filtered_random_noise_dim3], axis=4)
                filtered_random_noise = torch.Tensor(filtered_random_noise)
            else:
                filtered_random_noise = torch.zeros(grid_shape)
            
            if isinstance(self.elastic_deform_alpha, list):
                alpha = np.random.uniform(low=self.elastic_deform_alpha[0], high=self.elastic_deform_alpha[1])
            else:
                alpha = self.elastic_deform_alpha
            
            alpha_filtered_random_noise = alpha*filtered_random_noise
        else:
            alpha_filtered_random_noise = torch.zeros(grid_shape)
        
        theta_resolution_correction_src = torch.tensor([[1/(src_dim3/2),0,0,0], \
                                                        [0,1/(src_dim2/2),0,0], \
                                                        [0,0,1/(src_dim1/2),0], \
                                                        [0,0,0,1]], dtype=torch.get_default_dtype()).unsqueeze(0)
        
        theta_resolution_correction_dst = torch.tensor([[dst_dim3/2,0,0,0],\
                                                        [0,dst_dim2/2,0,0],\
                                                        [0,0,dst_dim1/2,0],\
                                                        [0,0,0,1]], dtype=torch.get_default_dtype()).unsqueeze(0)
        
        theta_final = torch.matmul(theta_resolution_correction_src, torch.matmul(theta_translation, torch.matmul(theta_rotation, theta_resolution_correction_dst)))[:,:3]
        
        output_dict = {}
        
        for key, val in data_dict.items():
            '''
            val in numpy, shape (x,y,z)
            all val's have the same shape
            '''
            if not 'landmark' in key:
                with torch.no_grad():
                    input_shape = [1, 1, dst_dim1, dst_dim2, dst_dim3]
                    grid = F.affine_grid(theta_final, input_shape, align_corners=True) # should be shape (N_batch,64,64,32,3)
                    
                    grid_elastic_deform = grid + alpha_filtered_random_noise
                    
                    val_transformed = F.grid_sample(torch.Tensor(val).unsqueeze(0).unsqueeze(1), grid_elastic_deform, align_corners=True)
                    output_dict[key] = val_transformed.numpy().squeeze()
            else:
                output_dict[key] = data_dict[key]
        return output_dict

##

class AlignRotateTranslateCrop3D_all(object):
    '''
    Crop in 3D from 250x250x160 to net_input_size (e.g. 64x64x32)
    Align so that triangle formed by 3 landmark points match for different patients (here, matching means aligned circumcenter and
    line from circumcenter to one point in triangle)
    
    Args:
        net_input_size = list of length 3 (H x W x D)
    '''
    def __init__(self, net_input_size=[64, 64, 64]):
        self.net_input_size = net_input_size
    
    def initialize_interp_points(self):
        '''
        Outputs self.xyz_orig and self.xyz_orig_homogeneous_coord
        axial slice
        '''
        x_overshoot = np.arange(round(np.sqrt(self.original_image_size[2]**2 + self.original_image_size[0]**2))+1)
        y_overshoot = np.arange(round(np.sqrt(self.original_image_size[2]**2 + self.original_image_size[1]**2))+1)
        z_initial = self.original_image_size[2]/2
        
        x_mesh, y_mesh = np.meshgrid(x_overshoot, y_overshoot)
        x_mesh = x_mesh.T
        y_mesh = y_mesh.T
        z_mesh = z_initial*np.ones(x_mesh.shape)
            
        self.xyz_orig = np.concatenate((np.expand_dims(x_mesh, 2),
                                        np.expand_dims(y_mesh, 2),
                                        np.expand_dims(z_mesh, 2)), axis=2)
        
        self.xyz_orig_homogeneous_coord = np.concatenate((x_mesh.reshape(1,-1),
                                                          y_mesh.reshape(1,-1),
                                                          z_mesh.reshape(1,-1),
                                                          np.ones(x_mesh.reshape(1,-1).shape)), axis=0)
        
        self.plane_origin = [self.original_image_size[0], 0, self.original_image_size[2]/2]
        self.ortho1 = [-1,0,0]
        self.ortho2 = [0,1,0]
        
    def calc_cicrumcenter(self, lm1, lm2, lm3):
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
    
    def calc_plane_params(self, lm1, lm2, lm3):
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
    
    def process_label(self, label, one_connected_component=False):
        if one_connected_component:
            label_thresholded = label>0.5
            all_labels, n_labels = measure.label(label_thresholded, return_num=True)
            
            mask = np.ones(all_labels.shape)
            
            if n_labels > 1:
                most_freq_non_zero_val = np.bincount(all_labels[all_labels!=0].reshape(-1)).argmax()
                mask[all_labels!=most_freq_non_zero_val] = 0
            
            output = label*mask
        else:
            output = label
        
        return output
    
    def __call__(self, data_dict):
        ct_data = data_dict['ct']
#        lm1_heatmap = data_dict['landmark1']
#        lm2_heatmap = data_dict['landmark2']
#        lm3_heatmap = data_dict['landmark3']
        aortic_root_label = data_dict['aortic_root_label']
        valve_1_label = data_dict['valve_1_label']
        valve_2_label = data_dict['valve_2_label']
        valve_3_label = data_dict['valve_3_label']
        
        self.original_image_size = ct_data.shape
        self.initialize_interp_points()
        
        x_orig = np.arange(self.original_image_size[0])
        y_orig = np.arange(self.original_image_size[1])
        z_orig = np.arange(self.original_image_size[2])
        rgi_linear_ct = RegularGridInterpolator((x_orig, y_orig, z_orig), ct_data, method='linear')
        try:
            rgi_linear_aortic_root_label = RegularGridInterpolator((x_orig, y_orig, z_orig), aortic_root_label, method='linear')
            rgi_linear_valve_1_label = RegularGridInterpolator((x_orig, y_orig, z_orig), valve_1_label, method='linear')
            rgi_linear_valve_2_label = RegularGridInterpolator((x_orig, y_orig, z_orig), valve_2_label, method='linear')
            rgi_linear_valve_3_label = RegularGridInterpolator((x_orig, y_orig, z_orig), valve_3_label, method='linear')
        except:
            pass
        
#        lm1 = np.array(np.unravel_index(lm1_heatmap.argmax(), lm1_heatmap.shape)).astype(float)
#        lm2 = np.array(np.unravel_index(lm2_heatmap.argmax(), lm2_heatmap.shape)).astype(float)
#        lm3 = np.array(np.unravel_index(lm3_heatmap.argmax(), lm3_heatmap.shape)).astype(float)
        lm1 = data_dict['landmark1']
        lm2 = data_dict['landmark2']
        lm3 = data_dict['landmark3']
        circumcenter = self.calc_cicrumcenter(lm1, lm2, lm3)
        (x,y,z,d) = self.calc_plane_params(lm1, lm2, lm3)
        
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
        
        # calculating angle_to_align_vertical_circumcenter_lm1
        circumcenter_proj = utils_gui.calc_2d_coordinate_on_plane(circumcenter, self.plane_origin, self.ortho1, self.ortho2, rot_mat123)
        lm1_proj = utils_gui.calc_2d_coordinate_on_plane(lm1, self.plane_origin, self.ortho1, self.ortho2, rot_mat123)
        vec_cc_lm1 = lm1_proj - circumcenter_proj
        angle_cc_lm1 = np.arctan2(vec_cc_lm1[0], vec_cc_lm1[1])
        rotz = angle_cc_lm1
        
        rot_mat4 = tfm.rotation_matrix(rotz, [x,y,z], point=circumcenter)
        rot_mat = np.dot(rot_mat4, rot_mat123)
        
        xyz_rotated_homogeneous_coord = np.dot(rot_mat, self.xyz_orig_homogeneous_coord)
        xyz_rotated = utils_gui.convert_to_orig_shape(xyz_rotated_homogeneous_coord, self.xyz_orig[:,:,0]) # size (298,298,3), but actually a 2D plane (3 xyz values for each position of intensity)
        
        normal = np.dot(rot_mat[0:3, 0:3], np.array([0,0,1]))
        normal = normal/np.linalg.norm(normal)
        
        min_dist_idx = np.unravel_index(np.argmin(np.linalg.norm(xyz_rotated-circumcenter, axis=2)), xyz_rotated.shape[0:2])
        xyz_min_dist = xyz_rotated[min_dist_idx[0], min_dist_idx[1],:]
        
        xyz_rotated_orig = xyz_rotated + (circumcenter - xyz_min_dist)
        
        ct_data_cropped = np.zeros(self.net_input_size)
        if aortic_root_label is not None:
            aortic_root_label_cropped = np.zeros(self.net_input_size)
            valve_1_label_cropped = np.zeros(self.net_input_size)
            valve_2_label_cropped = np.zeros(self.net_input_size)
            valve_3_label_cropped = np.zeros(self.net_input_size)
        else:
            aortic_root_label_cropped = None
            valve_1_label_cropped = None
            valve_2_label_cropped = None
            valve_3_label_cropped = None
        
        dists = np.arange(-self.net_input_size[2]/2,self.net_input_size[2]/2)+0.5
        
        bbox = np.array([[0, ct_data.shape[0]-1], [0, ct_data.shape[1]-1],[0, ct_data.shape[2]-1]])
        # using for loop b/c restrict_to_bbox function isn't compatible for more dimensions than (shape0, shape1, 3)
        for idx in range(self.net_input_size[2]):
            xyz_rotated = xyz_rotated_orig + normal*dists[idx]
            
            xyz_within_bbox, idx_bool = utils_gui.restrict_to_bbox(xyz_rotated, bbox)
        
            linear_idx = np.argmin(np.linalg.norm(xyz_within_bbox-circumcenter, axis=2))
            sub_idx = np.unravel_index(linear_idx, xyz_within_bbox.shape[0:2])
            
            mid_x = sub_idx[0]
            mid_y = sub_idx[1]
            inc_x = int(self.net_input_size[0]/2)
            inc_y = int(self.net_input_size[1]/2)
            
            rgi_idx = xyz_within_bbox[mid_x-inc_x:mid_x+inc_x,
                                      mid_y-inc_y:mid_y+inc_y,
                                      :]
            
            ct_data_cropped[:,:,idx] = rgi_linear_ct(rgi_idx)
            try:
                aortic_root_label_cropped[:,:,idx] = rgi_linear_aortic_root_label(rgi_idx)
                valve_1_label_cropped[:,:,idx] = rgi_linear_valve_1_label(rgi_idx)
                valve_2_label_cropped[:,:,idx] = rgi_linear_valve_2_label(rgi_idx)
                valve_3_label_cropped[:,:,idx] = rgi_linear_valve_3_label(rgi_idx)
            except:
                pass
        
        aortic_root_label_cropped_processed = self.process_label(aortic_root_label_cropped, one_connected_component=False)
        valve_1_label_cropped_processed = self.process_label(valve_1_label_cropped, one_connected_component=False)
        valve_2_label_cropped_processed = self.process_label(valve_2_label_cropped, one_connected_component=False)
        valve_3_label_cropped_processed = self.process_label(valve_3_label_cropped, one_connected_component=False)
        
        output_dict = {'ct': ct_data_cropped,
                       'aortic_root_label': aortic_root_label_cropped_processed,
                       'valve_1_label': valve_1_label_cropped_processed,
                       'valve_2_label': valve_2_label_cropped_processed,
                       'valve_3_label': valve_3_label_cropped_processed}
        
        return output_dict

##

class ElasticDeformationAlign_all(object):
    '''
    First align crop then perform elastic deformation.. probably should have it as a separate thing
    
    Args:
        same args as elasticdeform.deform_random_grid
    '''
    def __init__(self, net_input_size=[64, 64, 64], sigma=[2,4], alpha=[20,40], deform_chance=0.9):
        self.net_input_size = net_input_size
        self.sigma = sigma
        self.alpha = alpha
        self.deform_chance = deform_chance
    
    def initialize_interp_points(self):
        '''
        Outputs self.xyz_orig and self.xyz_orig_homogeneous_coord
        axial slice
        '''
        x_overshoot = np.arange(round(np.sqrt(self.original_image_size[2]**2 + self.original_image_size[0]**2))+1)
        y_overshoot = np.arange(round(np.sqrt(self.original_image_size[2]**2 + self.original_image_size[1]**2))+1)
        z_initial = self.original_image_size[2]/2
        
        x_mesh, y_mesh = np.meshgrid(x_overshoot, y_overshoot)
        x_mesh = x_mesh.T
        y_mesh = y_mesh.T
        z_mesh = z_initial*np.ones(x_mesh.shape)
        
        self.xyz_orig = np.concatenate((np.expand_dims(x_mesh, 2),
                                        np.expand_dims(y_mesh, 2),
                                        np.expand_dims(z_mesh, 2)), axis=2)
        
        self.xyz_orig_homogeneous_coord = np.concatenate((x_mesh.reshape(1,-1),
                                                          y_mesh.reshape(1,-1),
                                                          z_mesh.reshape(1,-1),
                                                          np.ones(x_mesh.reshape(1,-1).shape)), axis=0)
        
        self.plane_origin = [self.original_image_size[0], 0, self.original_image_size[2]/2]
        self.ortho1 = [-1,0,0]
        self.ortho2 = [0,1,0]
        
    def calc_cicrumcenter(self, lm1, lm2, lm3):
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
    
    def calc_plane_params(self, lm1, lm2, lm3):
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
    
    def get_deform_field(self):
        return self.rgi_idx_normal_arr, self.rgi_idx_distorted_arr
    
    def __call__(self, data_dict):
        ct_data = data_dict['ct']
        aortic_root_label = data_dict['aortic_root_label']
        valve_1_label = data_dict['valve_1_label']
        valve_2_label = data_dict['valve_2_label']
        valve_3_label = data_dict['valve_3_label']
        
        self.original_image_size = ct_data.shape
        self.initialize_interp_points()
        
        x_orig = np.arange(self.original_image_size[0])
        y_orig = np.arange(self.original_image_size[1])
        z_orig = np.arange(self.original_image_size[2])
        rgi_linear_ct = RegularGridInterpolator((x_orig, y_orig, z_orig), ct_data, method='linear')
        rgi_linear_aortic_root_label = RegularGridInterpolator((x_orig, y_orig, z_orig), aortic_root_label, method='linear')
        rgi_linear_valve_1_label = RegularGridInterpolator((x_orig, y_orig, z_orig), valve_1_label, method='linear')
        rgi_linear_valve_2_label = RegularGridInterpolator((x_orig, y_orig, z_orig), valve_2_label, method='linear')
        rgi_linear_valve_3_label = RegularGridInterpolator((x_orig, y_orig, z_orig), valve_3_label, method='linear')
        
        lm1 = data_dict['landmark1']
        lm2 = data_dict['landmark2']
        lm3 = data_dict['landmark3']
        circumcenter = self.calc_cicrumcenter(lm1, lm2, lm3)
        (x,y,z,d) = self.calc_plane_params(lm1, lm2, lm3)
        
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
        
        # calculating angle_to_align_vertical_circumcenter_lm1
        circumcenter_proj = utils_gui.calc_2d_coordinate_on_plane(circumcenter, self.plane_origin, self.ortho1, self.ortho2, rot_mat123)
        lm1_proj = utils_gui.calc_2d_coordinate_on_plane(lm1, self.plane_origin, self.ortho1, self.ortho2, rot_mat123)
        vec_cc_lm1 = lm1_proj - circumcenter_proj
        angle_cc_lm1 = np.arctan2(vec_cc_lm1[0], vec_cc_lm1[1])
        rotz = angle_cc_lm1
        
        rot_mat4 = tfm.rotation_matrix(rotz, [x,y,z], point=circumcenter)
        rot_mat = np.dot(rot_mat4, rot_mat123)
        
        xyz_rotated_homogeneous_coord = np.dot(rot_mat, self.xyz_orig_homogeneous_coord)
        xyz_rotated = utils_gui.convert_to_orig_shape(xyz_rotated_homogeneous_coord, self.xyz_orig[:,:,0]) # size (298,298,3), but actually a 2D plane (3 xyz values for each position of intensity)
        
#        normal = np.array([x,y,z])
        normal = np.dot(rot_mat[0:3, 0:3], np.array([0,0,1]))
        normal = normal/np.linalg.norm(normal)
        
#        dist = np.min(np.linalg.norm(xyz_rotated-circumcenter, axis=2))
        
        min_dist_idx = np.unravel_index(np.argmin(np.linalg.norm(xyz_rotated-circumcenter, axis=2)), xyz_rotated.shape[0:2])
        xyz_min_dist = xyz_rotated[min_dist_idx[0], min_dist_idx[1],:]
        
#        xyz_rotated_orig = xyz_rotated + normal*dist # translating xyz_rotated coordinates to be in plane formed by landmarks
        xyz_rotated_orig = xyz_rotated + (circumcenter - xyz_min_dist)
        
        ct_data_cropped = np.zeros(self.net_input_size)
        aortic_root_label_cropped = np.zeros(self.net_input_size)
        valve_1_label_cropped = np.zeros(self.net_input_size)
        valve_2_label_cropped = np.zeros(self.net_input_size)
        valve_3_label_cropped = np.zeros(self.net_input_size)
        
        dists = np.arange(-self.net_input_size[2]/2, self.net_input_size[2]/2)+0.5
        
        bbox = np.array([[0, ct_data.shape[0]-1], [0, ct_data.shape[1]-1],[0, ct_data.shape[2]-1]])
        
        self.rgi_idx_normal_arr = np.zeros([self.net_input_size[0], self.net_input_size[1], self.net_input_size[2], 3])
        self.rgi_idx_distorted_arr = np.zeros([self.net_input_size[0], self.net_input_size[1], self.net_input_size[2], 3])
        
        # if else condition so that sometimes it's the non-distorted image
        if np.random.uniform() <= self.deform_chance:
            random_noise = np.random.uniform(low=-1.0, high=1.0, size=[xyz_rotated_orig.shape[0], xyz_rotated_orig.shape[1], self.net_input_size[2], 3])
            
            if isinstance(self.sigma, list):
                sigma = np.random.uniform(low=self.sigma[0], high=self.sigma[1])
            else:
                sigma = self.sigma
            filtered_random_noise_x = gaussian_filter(random_noise[:,:,:,0], sigma=sigma)
            filtered_random_noise_y = gaussian_filter(random_noise[:,:,:,1], sigma=sigma)
            filtered_random_noise_z = gaussian_filter(random_noise[:,:,:,2], sigma=sigma)
            filtered_random_noise = np.stack([filtered_random_noise_x, filtered_random_noise_y, filtered_random_noise_z], axis=3)
        else:
            filtered_random_noise = np.zeros([xyz_rotated_orig.shape[0], xyz_rotated_orig.shape[1], self.net_input_size[2], 3])
            
        # using for loop b/c restrict_to_bbox function isn't compatible for more dimensions than (shape0, shape1, 3)
        for idx in range(self.net_input_size[2]):
            xyz_rotated = xyz_rotated_orig + normal*dists[idx]
            
            xyz_rotated_normal = xyz_rotated
            
            if isinstance(self.alpha, list):
                alpha = np.random.uniform(low=self.alpha[0], high=self.alpha[1])
            else:
                alpha = self.alpha
                
            xyz_rotated_distorted = xyz_rotated + alpha*filtered_random_noise[:,:,idx,:]
            
            xyz_within_bbox_normal, idx_bool = utils_gui.restrict_to_bbox(xyz_rotated_normal, bbox)
            xyz_within_bbox_distorted, idx_bool = utils_gui.restrict_to_bbox(xyz_rotated_distorted, bbox)
            
            linear_idx = np.argmin(np.linalg.norm(xyz_within_bbox_distorted-circumcenter, axis=2))
            sub_idx = np.unravel_index(linear_idx, xyz_within_bbox_distorted.shape[0:2])
            
            mid_x = sub_idx[0]
            mid_y = sub_idx[1]
            inc_x = int(self.net_input_size[0]/2)
            inc_y = int(self.net_input_size[1]/2)
            
            rgi_idx_normal = xyz_within_bbox_normal[mid_x-inc_x:mid_x+inc_x,
                                                    mid_y-inc_y:mid_y+inc_y,
                                                    :]
            
            rgi_idx_distorted = xyz_within_bbox_distorted[mid_x-inc_x:mid_x+inc_x,
                                                          mid_y-inc_y:mid_y+inc_y,
                                                          :]
            
            self.rgi_idx_normal_arr[:,:,idx,:] = rgi_idx_normal
            self.rgi_idx_distorted_arr[:,:,idx,:] = rgi_idx_distorted
            
            ct_data_cropped[:,:,idx] = rgi_linear_ct(rgi_idx_distorted)
            aortic_root_label_cropped[:,:,idx] = rgi_linear_aortic_root_label(rgi_idx_distorted)
            valve_1_label_cropped[:,:,idx] = rgi_linear_valve_1_label(rgi_idx_distorted)
            valve_2_label_cropped[:,:,idx] = rgi_linear_valve_2_label(rgi_idx_distorted)
            valve_3_label_cropped[:,:,idx] = rgi_linear_valve_3_label(rgi_idx_distorted)
        
        output_dict = {'ct': ct_data_cropped,
                       'aortic_root_label': aortic_root_label_cropped,
                       'valve_1_label': valve_1_label_cropped,
                       'valve_2_label': valve_2_label_cropped,
                       'valve_3_label': valve_3_label_cropped}
        
        return output_dict
    
## 1 64x64x64

class ElasticDeformGridSample_64x64x64(object):
    '''
    '''
    def __init__(self, \
                 sigma = [2,4], \
                 alpha = [20,40], \
                 deform_chance = 0.9):
        '''
        e.g. if max_translate_ratio=0.2, we're saying we can move in each direction 20% of total pixels
        '''
        self.sigma = sigma
        self.alpha = alpha
        self.deform_chance = deform_chance
        
    def __call__(self, data_dict):
        src_shape = data_dict['ct'].shape
        dst_shape = data_dict['ct'].shape
        
        grid_shape = [1, *dst_shape, 3]
        
        if np.random.uniform() <= self.deform_chance:
            random_noise = np.random.uniform(low=-1.0, high=1.0, size=grid_shape)
            
            if isinstance(self.sigma, list):
                sigma = np.random.uniform(low=self.sigma[0], high=self.sigma[1])
            else:
                sigma = self.sigma
                
            random_noise = np.random.uniform(low=-1.0, high=1.0, size=grid_shape)
            
            filtered_random_noise_dim1 = gaussian_filter(random_noise[:,:,:,:,0], sigma=sigma)/(src_shape[0]/2)
            filtered_random_noise_dim2 = gaussian_filter(random_noise[:,:,:,:,1], sigma=sigma)/(src_shape[1]/2)
            filtered_random_noise_dim3 = gaussian_filter(random_noise[:,:,:,:,2], sigma=sigma)/(src_shape[2]/2)
            filtered_random_noise = np.stack([filtered_random_noise_dim1, filtered_random_noise_dim2, filtered_random_noise_dim3], axis=4)
            filtered_random_noise = torch.Tensor(filtered_random_noise)
        else:
            filtered_random_noise = torch.zeros(grid_shape)
        
        if isinstance(self.alpha, list):
            alpha = np.random.uniform(low=self.alpha[0], high=self.alpha[1])
        else:
            alpha = self.alpha
        
        alpha_filtered_random_noise = alpha*filtered_random_noise
        
        output_dict = {}
        
        for key, val in data_dict.items():
            '''
            val in numpy, shape (x,y,z)
            all val's have the same shape
            '''
            if not 'landmark' in key and not 'gt_pcl_list' in key:
                with torch.no_grad():
                    output_shape = [1, 1, *dst_shape]
                    grid_identity = F.affine_grid(torch.eye(4).repeat([1, 1, 1])[:,0:3,:], output_shape, align_corners=True) # should be shape (N_batch,64,64,32,3)
                    
                    grid_elastic_deform = grid_identity + alpha_filtered_random_noise
                    
                    val_transformed = F.grid_sample(torch.Tensor(val).unsqueeze(0).unsqueeze(1), grid_elastic_deform, align_corners=True)
                    output_dict[key] = val_transformed.numpy().squeeze()
            else:
                output_dict[key] = data_dict[key]
        return output_dict

##

class ElasticDeformGtPclTrain_64x64x64(object):
    '''
    '''
    def __init__(self, \
                 sigma = [2,4], \
                 alpha = [20,40], \
                 deform_chance = 0.9):
        
        self.sigma = sigma
        self.alpha = alpha
        self.deform_chance = deform_chance
        
    def __call__(self, data_dict):
        src_shape = data_dict['ct'].shape
        dst_shape = data_dict['ct'].shape
        
        grid_shape = [1, *dst_shape, 3]
        
        if np.random.uniform() <= self.deform_chance:
            random_noise = np.random.uniform(low=-1.0, high=1.0, size=grid_shape)
            
            if isinstance(self.sigma, list):
                sigma = np.random.uniform(low=self.sigma[0], high=self.sigma[1])
            else:
                sigma = self.sigma
                
            random_noise = np.random.uniform(low=-1.0, high=1.0, size=grid_shape)
            
            filtered_random_noise_dim1 = gaussian_filter(random_noise[:,:,:,:,0], sigma=sigma)/(src_shape[0]/2)
            filtered_random_noise_dim2 = gaussian_filter(random_noise[:,:,:,:,1], sigma=sigma)/(src_shape[1]/2)
            filtered_random_noise_dim3 = gaussian_filter(random_noise[:,:,:,:,2], sigma=sigma)/(src_shape[2]/2)
            filtered_random_noise = np.stack([filtered_random_noise_dim1, filtered_random_noise_dim2, filtered_random_noise_dim3], axis=4)
            filtered_random_noise = torch.Tensor(filtered_random_noise)
        else:
            filtered_random_noise = torch.zeros(grid_shape)
        
        if isinstance(self.alpha, list):
            alpha = np.random.uniform(low=self.alpha[0], high=self.alpha[1])
        else:
            alpha = self.alpha
        
        alpha_filtered_random_noise = alpha*filtered_random_noise
        
        output_dict = {}
        
        for key, val in data_dict.items():
            if key == 'ct':
                with torch.no_grad():
                    output_shape = [1, 1, *dst_shape]
                    grid_identity = F.affine_grid(torch.eye(4).repeat([1, 1, 1])[:,0:3,:], output_shape, align_corners=True) # should be shape (N_batch,64,64,32,3)
                    
                    grid_elastic_deform = grid_identity + alpha_filtered_random_noise
                    
                    val_transformed = F.grid_sample(torch.Tensor(val).unsqueeze(0).unsqueeze(1), grid_elastic_deform, align_corners=True)
                    output_dict[key] = val_transformed.numpy().squeeze()
                    
            elif key == 'gt_pcl_list':
                # verts_list = [verts.cpu() for verts in torch.split(val, 1, dim=0)]
                verts_list = [gt_pcl.unsqueeze(0).cuda() for gt_pcl in val]
                displacement_field_tuple = (alpha_filtered_random_noise.cuda().permute(0,4,1,2,3),)*len(verts_list)
                interp_field_list = utils_sp.interpolate_rescale_field_torch(displacement_field_tuple, verts_list, img_size=src_shape, reversed_field=True)
                transformed_verts_list = utils_sp.move_verts_with_field(verts_list, interp_field_list)
                
                output_dict[key] = [verts.cpu() for verts in transformed_verts_list]
            else:
                output_dict[key] = data_dict[key]
                
        return output_dict

##

import airlab_stuff as al

class BsplineDeformGridSample_64x64x64(object):
    '''
    '''
    def __init__(self,
                 sigma = [12,12,12],
                 order = 3,
                 deform_chance = 0.9):

        self.sigma = sigma
        self.order = order
        self.deform_chance = deform_chance
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __call__(self, data_dict):
        src_shape = data_dict['ct'].shape
        dst_shape = data_dict['ct'].shape
        output_shape = [1, 1, *dst_shape]

        transformation = al.BsplineTransformation(src_shape,
                                                  sigma=self.sigma,
                                                  order=self.order,
                                                  dtype=torch.get_default_dtype(),
                                                  device=self.device,
                                                  diffeomorphic=True)

        with torch.no_grad():
            grid_identity = F.affine_grid(torch.eye(4, dtype=torch.get_default_dtype(), device=self.device).repeat([1, 1, 1])[:,0:3,:], output_shape, align_corners=True) # should be shape (N_batch,64,64,64,3)
            dist_btw_cp = output_shape[2]/transformation.cp_grid_shape[2]
            max_cp_shift = dist_btw_cp/2 * 0.1
            if np.random.uniform() <= self.deform_chance:
                cp_grid = torch.empty(transformation.cp_grid_shape, dtype=torch.get_default_dtype(), device=self.device).uniform_(-max_cp_shift, max_cp_shift)
            else:
                cp_grid = torch.zeros(transformation.cp_grid_shape, dtype=torch.get_default_dtype(), device=self.device)
            displacement_field = transformation(cp_grid)
            grid_displacement = displacement_field.permute(0,2,3,4,1)
            grid_transform = grid_identity + grid_displacement

            displacement_field_inverse = transformation.get_inverse_displacement(cp_grid)
            # displacement_field_inverse = utils_sp.reverse_displacement_field_tuple((displacement_field,))[0]

        output_dict = {}

        for key, val in data_dict.items():
            '''
            val in numpy, shape (x,y,z)
            all val's have the same shape
            '''
            with torch.no_grad():
                if 'ct' == key or '_label' in key:
                    val_transformed = F.grid_sample(torch.tensor(val, dtype=torch.get_default_dtype(), device=self.device).unsqueeze(0).unsqueeze(1), grid_transform, align_corners=True)
                    output_dict[key] = val_transformed.cpu().numpy().squeeze()
                elif key == 'gt_pcl_list':
                    # !!!! what I used to do..
                    # verts_list = [gt_pcl.unsqueeze(0).cuda() for gt_pcl in val]
                    # # displacement_field_tuple = (grid_displacement.permute(0,4,1,2,3),)*len(verts_list)
                    # displacement_field_tuple = (displacement_field,)*len(verts_list)
                    # interp_field_list = utils_sp.interpolate_rescale_field_torch(displacement_field_tuple, verts_list, img_size=src_shape, reversed_field=True)
                    # transformed_verts_list = utils_sp.move_verts_with_field(verts_list, interp_field_list)
                    # output_dict[key] = [verts.cpu() for verts in transformed_verts_list]

                    # !!!! what I should do especially if I have diffeomorphic..
                    verts_list = [gt_pcl.unsqueeze(0).cuda() for gt_pcl in val]
                    displacement_field_tuple = (displacement_field_inverse,)*len(verts_list)
                    interp_field_list = utils_sp.interpolate_rescale_field_torch(displacement_field_tuple, verts_list, img_size=src_shape, reversed_field=False)
                    transformed_verts_list = utils_sp.move_verts_with_field(verts_list, interp_field_list)
                    transformed_verts_list = [verts.squeeze(0) for verts in transformed_verts_list]
                    output_dict[key] = [verts.cpu() for verts in transformed_verts_list]
                else:
                    output_dict[key] = data_dict[key]
        return output_dict

## 2 64x64x64

#class ElasticDeform_64x64x64(object):
#    '''
#    Args:
#        same args as elasticdeform.deform_random_grid
#    '''
#    def __init__(self, net_input_size=[64, 64, 64], sigma=[2,4], alpha=[20,40], deform_chance=0.9):
#        self.net_input_size = net_input_size
#        self.sigma = sigma
#        self.alpha = alpha
#        self.deform_chance = deform_chance
#    
#    def initialize_interp_points(self):
#        '''
#        Outputs self.xyz_orig and self.xyz_orig_homogeneous_coord
#        axial slice
#        '''
#        x_overshoot = np.arange(round(np.sqrt(self.original_image_size[2]**2 + self.original_image_size[0]**2))+1)
#        y_overshoot = np.arange(round(np.sqrt(self.original_image_size[2]**2 + self.original_image_size[1]**2))+1)
#        z_initial = self.original_image_size[2]/2
#        
#        x_mesh, y_mesh = np.meshgrid(x_overshoot, y_overshoot)
#        x_mesh = x_mesh.T
#        y_mesh = y_mesh.T
#        z_mesh = z_initial*np.ones(x_mesh.shape)
#        
#        self.xyz_orig = np.concatenate((np.expand_dims(x_mesh, 2),
#                                        np.expand_dims(y_mesh, 2),
#                                        np.expand_dims(z_mesh, 2)), axis=2)
#        
#        self.xyz_orig_homogeneous_coord = np.concatenate((x_mesh.reshape(1,-1),
#                                                          y_mesh.reshape(1,-1),
#                                                          z_mesh.reshape(1,-1),
#                                                          np.ones(x_mesh.reshape(1,-1).shape)), axis=0)
#        
#        self.plane_origin = [self.original_image_size[0], 0, self.original_image_size[2]/2]
#        self.ortho1 = [-1,0,0]
#        self.ortho2 = [0,1,0]
#        
#    def calc_cicrumcenter(self, lm1, lm2, lm3):
#        '''
#        calculate the circumcenter (point equidistance from a, b, and c)
#        lm1, lm2, lm3 are the three 3D coordinates of landmark points
#        '''
#        ac = lm3-lm1
#        ab = lm2-lm1
#        abXac = np.cross(ab, ac)
#        
#        to_circumsphere_center = (np.cross(abXac, ab)*np.linalg.norm(ac)**2 + np.cross(ac, abXac)*np.linalg.norm(ac)**2)/(2*np.linalg.norm(abXac)**2)
#        
#        circumcenter = lm1 + to_circumsphere_center
#        
#        return circumcenter
#    
#    def calc_plane_params(self, lm1, lm2, lm3):
#        x1 = lm1[0]
#        y1 = lm1[1]
#        z1 = lm1[2]
#        
#        x2 = lm2[0]
#        y2 = lm2[1]
#        z2 = lm2[2]
#        
#        x3 = lm3[0]
#        y3 = lm3[1]
#        z3 = lm3[2]
#        
#        vector1 = [x2 - x1, y2 - y1, z2 - z1]
#        vector2 = [x3 - x1, y3 - y1, z3 - z1]
#        
#        cross_product = [vector1[1] * vector2[2] - vector1[2] * vector2[1], -1 * (vector1[0] * vector2[2] - vector1[2] * vector2[0]), vector1[0] * vector2[1] - vector1[1] * vector2[0]]
#        
#        a = cross_product[0]
#        b = cross_product[1]
#        c = cross_product[2]
#        d = - (cross_product[0] * x1 + cross_product[1] * y1 + cross_product[2] * z1)
#        
#        return a, b, c, d
#    
#    def get_deform_field(self):
#        return self.rgi_idx_normal_arr, self.rgi_idx_distorted_arr
#    
#    def __call__(self, data_dict):
#        ct_data = data_dict['ct']
#        aortic_root_label = data_dict['aortic_root_label']
#        valve_1_label = data_dict['valve_1_label']
#        valve_2_label = data_dict['valve_2_label']
#        valve_3_label = data_dict['valve_3_label']
#        
#        self.original_image_size = ct_data.shape
#        self.initialize_interp_points()
#        
#        x_orig = np.arange(self.original_image_size[0])
#        y_orig = np.arange(self.original_image_size[1])
#        z_orig = np.arange(self.original_image_size[2])
#        rgi_linear_ct = RegularGridInterpolator((x_orig, y_orig, z_orig), ct_data, method='linear')
#        rgi_linear_aortic_root_label = RegularGridInterpolator((x_orig, y_orig, z_orig), aortic_root_label, method='linear')
#        rgi_linear_valve_1_label = RegularGridInterpolator((x_orig, y_orig, z_orig), valve_1_label, method='linear')
#        rgi_linear_valve_2_label = RegularGridInterpolator((x_orig, y_orig, z_orig), valve_2_label, method='linear')
#        rgi_linear_valve_3_label = RegularGridInterpolator((x_orig, y_orig, z_orig), valve_3_label, method='linear')
#        
#        lm1 = data_dict['landmark1']
#        lm2 = data_dict['landmark2']
#        lm3 = data_dict['landmark3']
#        circumcenter = self.calc_cicrumcenter(lm1, lm2, lm3)
#        (x,y,z,d) = self.calc_plane_params(lm1, lm2, lm3)
#        
#        rotx = np.arctan2( y, z );
#        if z >= 0:
#           roty = -np.arctan2( x * np.cos(rotx), z );
#        else:
#           roty = np.arctan2( x * np.cos(rotx), -z );
#        
#        rotz = 0
#        
#        rotation_center = np.asarray(ct_data.shape)/2
#        rot_mat1 = tfm.rotation_matrix(rotx, [-1,0,0], point=rotation_center)
#        rot_mat2 = tfm.rotation_matrix(roty, [0,-1,0], point=rotation_center)
#        rot_mat3 = tfm.rotation_matrix(rotz, [x,y,z], point=circumcenter)
#        
#        rot_mat123 = np.dot(rot_mat3, np.dot(rot_mat2, rot_mat1))
#        
#        # calculating angle_to_align_vertical_circumcenter_lm1
#        circumcenter_proj = utils_gui.calc_2d_coordinate_on_plane(circumcenter, self.plane_origin, self.ortho1, self.ortho2, rot_mat123)
#        lm1_proj = utils_gui.calc_2d_coordinate_on_plane(lm1, self.plane_origin, self.ortho1, self.ortho2, rot_mat123)
#        vec_cc_lm1 = lm1_proj - circumcenter_proj
#        angle_cc_lm1 = np.arctan2(vec_cc_lm1[0], vec_cc_lm1[1])
#        rotz = angle_cc_lm1
#        
#        rot_mat4 = tfm.rotation_matrix(rotz, [x,y,z], point=circumcenter)
#        rot_mat = np.dot(rot_mat4, rot_mat123)
#        
#        xyz_rotated_homogeneous_coord = np.dot(rot_mat, self.xyz_orig_homogeneous_coord)
#        xyz_rotated = utils_gui.convert_to_orig_shape(xyz_rotated_homogeneous_coord, self.xyz_orig[:,:,0]) # size (298,298,3), but actually a 2D plane (3 xyz values for each position of intensity)
#        
##        normal = np.array([x,y,z])
#        normal = np.dot(rot_mat[0:3, 0:3], np.array([0,0,1]))
#        normal = normal/np.linalg.norm(normal)
#        
##        dist = np.min(np.linalg.norm(xyz_rotated-circumcenter, axis=2))
#        
#        min_dist_idx = np.unravel_index(np.argmin(np.linalg.norm(xyz_rotated-circumcenter, axis=2)), xyz_rotated.shape[0:2])
#        xyz_min_dist = xyz_rotated[min_dist_idx[0], min_dist_idx[1],:]
#        
##        xyz_rotated_orig = xyz_rotated + normal*dist # translating xyz_rotated coordinates to be in plane formed by landmarks
#        xyz_rotated_orig = xyz_rotated + (circumcenter - xyz_min_dist)
#        
#        ct_data_cropped = np.zeros(self.net_input_size)
#        aortic_root_label_cropped = np.zeros(self.net_input_size)
#        valve_1_label_cropped = np.zeros(self.net_input_size)
#        valve_2_label_cropped = np.zeros(self.net_input_size)
#        valve_3_label_cropped = np.zeros(self.net_input_size)
#        
#        dists = np.arange(-self.net_input_size[2]/2, self.net_input_size[2]/2)+0.5
#        
#        bbox = np.array([[0, ct_data.shape[0]-1], [0, ct_data.shape[1]-1],[0, ct_data.shape[2]-1]])
#        
#        self.rgi_idx_normal_arr = np.zeros([self.net_input_size[0], self.net_input_size[1], self.net_input_size[2], 3])
#        self.rgi_idx_distorted_arr = np.zeros([self.net_input_size[0], self.net_input_size[1], self.net_input_size[2], 3])
#        
#        # if else condition so that sometimes it's the non-distorted image
#        if np.random.uniform() <= self.deform_chance:
#            random_noise = np.random.uniform(low=-1.0, high=1.0, size=[xyz_rotated_orig.shape[0], xyz_rotated_orig.shape[1], self.net_input_size[2], 3])
#            
#            if isinstance(self.sigma, list):
#                sigma = np.random.uniform(low=self.sigma[0], high=self.sigma[1])
#            else:
#                sigma = self.sigma
#            filtered_random_noise_x = gaussian_filter(random_noise[:,:,:,0], sigma=sigma)
#            filtered_random_noise_y = gaussian_filter(random_noise[:,:,:,1], sigma=sigma)
#            filtered_random_noise_z = gaussian_filter(random_noise[:,:,:,2], sigma=sigma)
#            filtered_random_noise = np.stack([filtered_random_noise_x, filtered_random_noise_y, filtered_random_noise_z], axis=3)
#        else:
#            filtered_random_noise = np.zeros([xyz_rotated_orig.shape[0], xyz_rotated_orig.shape[1], self.net_input_size[2], 3])
#            
#        # using for loop b/c restrict_to_bbox function isn't compatible for more dimensions than (shape0, shape1, 3)
#        for idx in range(self.net_input_size[2]):
#            xyz_rotated = xyz_rotated_orig + normal*dists[idx]
#            
#            xyz_rotated_normal = xyz_rotated
#            
#            if isinstance(self.alpha, list):
#                alpha = np.random.uniform(low=self.alpha[0], high=self.alpha[1])
#            else:
#                alpha = self.alpha
#                
#            xyz_rotated_distorted = xyz_rotated + alpha*filtered_random_noise[:,:,idx,:]
#            
#            xyz_within_bbox_normal, idx_bool = utils_gui.restrict_to_bbox(xyz_rotated_normal, bbox)
#            xyz_within_bbox_distorted, idx_bool = utils_gui.restrict_to_bbox(xyz_rotated_distorted, bbox)
#            
#            linear_idx = np.argmin(np.linalg.norm(xyz_within_bbox_distorted-circumcenter, axis=2))
#            sub_idx = np.unravel_index(linear_idx, xyz_within_bbox_distorted.shape[0:2])
#            
#            mid_x = sub_idx[0]
#            mid_y = sub_idx[1]
#            inc_x = int(self.net_input_size[0]/2)
#            inc_y = int(self.net_input_size[1]/2)
#            
#            rgi_idx_normal = xyz_within_bbox_normal[mid_x-inc_x:mid_x+inc_x,
#                                                    mid_y-inc_y:mid_y+inc_y,
#                                                    :]
#            
#            rgi_idx_distorted = xyz_within_bbox_distorted[mid_x-inc_x:mid_x+inc_x,
#                                                          mid_y-inc_y:mid_y+inc_y,
#                                                          :]
#            
#            self.rgi_idx_normal_arr[:,:,idx,:] = rgi_idx_normal
#            self.rgi_idx_distorted_arr[:,:,idx,:] = rgi_idx_distorted
#            
#            ct_data_cropped[:,:,idx] = rgi_linear_ct(rgi_idx_distorted)
#            aortic_root_label_cropped[:,:,idx] = rgi_linear_aortic_root_label(rgi_idx_distorted)
#            valve_1_label_cropped[:,:,idx] = rgi_linear_valve_1_label(rgi_idx_distorted)
#            valve_2_label_cropped[:,:,idx] = rgi_linear_valve_2_label(rgi_idx_distorted)
#            valve_3_label_cropped[:,:,idx] = rgi_linear_valve_3_label(rgi_idx_distorted)
#        
#        output_dict = {'ct': ct_data_cropped,
#                       'aortic_root_label': aortic_root_label_cropped,
#                       'valve_1_label': valve_1_label_cropped,
#                       'valve_2_label': valve_2_label_cropped,
#                       'valve_3_label': valve_3_label_cropped}
#        
#        return output_dict
    
##
