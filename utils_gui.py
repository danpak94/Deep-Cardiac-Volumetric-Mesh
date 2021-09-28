# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 20:18:44 2019

@author: Daniel
"""

#%%

import numpy as np

def get_rgi_idx(rot_mat_prev, xyz_orig_homogeneous_coord, xyz_orig, bbox):
    xyz_rotated_homogeneous_coord = np.dot(rot_mat_prev, xyz_orig_homogeneous_coord)
    xyz_rotated = convert_to_orig_shape(xyz_rotated_homogeneous_coord, xyz_orig[:,:,0])
    xyz_within_bbox, idx_bool = restrict_to_bbox(xyz_rotated, bbox)
    rgi_idx = xyz_within_bbox
    
    return rgi_idx

def initialize_axial_interp_points(ct, plane):
        '''
        Uses self.ct and self.canvas.plane
        Outputs self.xyz_orig and self.xyz_orig_homogeneous_coord
        '''
        if plane == 'sagittal':
            x_initial = ct.shape[0]/2
            y_overshoot = np.arange(round(np.sqrt(ct.shape[1]**2 + ct.shape[0]**2))+1)
            z_overshoot = np.arange(round(np.sqrt(ct.shape[1]**2 + ct.shape[2]**2))+1)
            
            y_mesh, z_mesh = np.meshgrid(y_overshoot, z_overshoot)
            y_mesh = y_mesh.T
            z_mesh = z_mesh.T
            x_mesh = x_initial*np.ones(y_mesh.shape)
            
        elif plane == 'axial':
            x_overshoot = np.arange(round(np.sqrt(ct.shape[2]**2 + ct.shape[0]**2))+1)
            y_overshoot = np.arange(round(np.sqrt(ct.shape[2]**2 + ct.shape[1]**2))+1)
            z_initial = ct.shape[2]/2
            
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
        
        return xyz_orig_homogeneous_coord, xyz_orig

def calc_2d_coordinate_on_plane(point_intersection, plane_origin, ortho1, ortho2, rot_mat):
    '''
    Assume point (given by xyz coordinate) is directly on the plane
    '''
    plane_origin_rotated = apply_rot_mat_full(rot_mat, plane_origin)
    ortho1_rotated = apply_rot_mat_no_translation(rot_mat, ortho1)
    ortho2_rotated = apply_rot_mat_no_translation(rot_mat, ortho2)
    
    e1 = ortho1_rotated
    e2 = ortho2_rotated
    rp = point_intersection
    r0 = plane_origin_rotated
    
    t1 = np.dot(e1, rp-r0)
    t2 = np.dot(e2, rp-r0)
    
    return np.array((t1, t2))

#def calc_crosshair_center_xyz(x_center, y_center, plane_origin, ortho1, ortho2, rot_mat):
#    plane_origin_rotated = apply_rot_mat_full(rot_mat, plane_origin)
#    ortho1_rotated = apply_rot_mat_no_translation(rot_mat, ortho1)
#    ortho2_rotated = apply_rot_mat_no_translation(rot_mat, ortho2)
#    
#    t1 = x_center
#    t2 = y_center
#    e1 = ortho1_rotated
#    e2 = ortho2_rotated
#    r0 = plane_origin_rotated
#    
#    rp = r0 + t1*e1 + t2*e2
#    
#    return rp

def calc_line2D_endpoints(point1, point2, hyp):
    '''
    point1 and point2: 2d coordinates, np.array shape (2,)
    hyp: half of line length (10000)
    xlim_img: [0 250]
    ylim_img: [0 160]
    '''
    
    angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
    
    x_center = point1[0]
    y_center = point1[1]
    
    x1 = x_center + hyp*np.cos(angle)
    x2 = x_center + hyp*np.cos(angle + np.pi)
    y1 = y_center + hyp*np.sin(angle)
    y2 = y_center + hyp*np.sin(angle + np.pi)
    
    return np.array(((x1, x2), (y1, y2)))

def calc_angle_offset(line, press_angle):
    lx = line.get_xdata()
    ly = line.get_ydata()
    
    line_angle = np.arctan2(ly[1]-ly[0], lx[1]-lx[0])
    angle_offset = line_angle - press_angle
    
    return angle_offset

def get_line_crossing(line1, line2):
    x1 = line1.get_xdata()
    y1 = line1.get_ydata()
    x2 = line2.get_xdata()
    y2 = line2.get_ydata()
    
    m1 = (y1[1]-y1[0])/(x1[1]-x1[0]+1e-10)
    m2 = (y2[1]-y2[0])/(x2[1]-x2[0]+1e-10)
    
    b1 = y1[0] - m1*x1[0]
    b2 = y2[0] - m2*x2[0]
    
    A = np.array([[-m1, 1], [-m2, 1]])
    b = np.array([b1, b2])
    center = np.linalg.solve(A, b)
    
    return center

def apply_rot_mat_full(rot_mat, xyz_3d):
    xyz_homogeneous_coord = np.append(np.array(xyz_3d), 1)
    xyz_3d_modified = np.dot(rot_mat, xyz_homogeneous_coord)[:3]
    
    return xyz_3d_modified

def apply_rot_mat_no_translation(rot_mat, xyz_3d):
    rot_mat_no_translation = rot_mat.copy()
    rot_mat_no_translation[:3,3] = 0
    
    xyz_homogeneous_coord = np.append(np.array(xyz_3d), 1)
    xyz_3d_rotated = np.dot(rot_mat_no_translation, xyz_homogeneous_coord)[:3]
    
    return xyz_3d_rotated

def convert_to_orig_shape(xyz_homogeneous_coord, x_mesh):
    xyz_list = []
    for i in range(3):
        xyz = xyz_homogeneous_coord[i,:].reshape(x_mesh.shape[0], x_mesh.shape[1], -1)
        xyz_list.append(xyz)
    
    xyz_orig_shape = np.concatenate(tuple(xyz_list), axis=2)
    
    return xyz_orig_shape

def restrict_to_bbox(xyz_orig_shape, bbox):
    cond_list = []
    for i in range(3):
        cond = np.logical_and(xyz_orig_shape[:,:,i] >= bbox[i,0], xyz_orig_shape[:,:,i] <= bbox[i,1])
        cond_list.append(cond)
    
    idx_bool = np.logical_and(np.logical_and(cond_list[0], cond_list[1]), cond_list[2])
    
    xyz_within_bbox = xyz_orig_shape.copy()
    xyz_within_bbox[~idx_bool,:] = 0
    
    return xyz_within_bbox, idx_bool