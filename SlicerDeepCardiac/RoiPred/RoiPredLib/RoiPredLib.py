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
import shutil
import numpy as np
import torch
import dcvm
import pyvista as pv
import pyacvd
from scipy.ndimage import distance_transform_edt

try:
    import slicer
except:
    pass
import vtk

curr_file_dir_path = os.path.dirname(os.path.realpath(__file__))
dcvm_parent_dir = os.path.abspath(os.path.join(curr_file_dir_path, '../../..'))
# dcvm_parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))
temp_data_dir = os.path.abspath(os.path.join(dcvm_parent_dir, '../dcvm_data_files'))

def relocate_data():
    downloaded_exps_dir = os.path.join(temp_data_dir, 'experiments')
    downloaded_template_dir = os.path.join(temp_data_dir, 'template_for_deform')
    
    dcvm_exps_dir = os.path.join(dcvm_parent_dir, 'experiments')
    dcvm_template_dir = os.path.join(dcvm_parent_dir, 'template_for_deform')

    move_matching_files(downloaded_exps_dir, dcvm_exps_dir)
    move_matching_files(downloaded_template_dir, dcvm_template_dir)
    
    if os.path.isdir(temp_data_dir):
        shutil.rmtree(temp_data_dir, ignore_errors=True)
        print('Deleted: {}'.format(temp_data_dir))
        print(' ')

def move_matching_files(dirA, dirB):
    for root, dirs, files in os.walk(dirA):
        for dir_name in dirs:
            source_dir = os.path.join(root, dir_name)
            dest_dir = os.path.join(dirB, dir_name)
            os.makedirs(dest_dir, exist_ok=True)
            # if os.path.exists(dest_dir):
            for file_name in os.listdir(source_dir):
                source_file = os.path.join(source_dir, file_name)
                dest_file = os.path.join(dest_dir, file_name)
                if os.path.isfile(source_file):
                    shutil.move(source_file, dest_file)
                    print('src: {}'.format(source_file))
                    print('dst: {}'.format(dest_file))
                    print(' ')

def img_from_volumeNode(volumeNode):
    img = slicer.util.arrayFromVolume(volumeNode) # RAS
    img = img.transpose(2,1,0) # b/c vtk/Slicer flips when getting array from volume
    # mat = vtk.vtkMatrix4x4(); volumeNode.GetIJKToRASDirectionMatrix(mat)
    # if mat.GetElement(0,0) == -1: # flip based on IJK to RAS DirectionMatrix
    #     img = np.flip(img, axis=0)
    # if mat.GetElement(1,1) == -1:
    #     img = np.flip(img, axis=1)
    return img

def arrayFromVTKMatrix(vmatrix):
    """
    https://discourse.slicer.org/t/vtk-transform-matrix-as-python-list-tuple-array/11797
    Return vtkMatrix4x4 or vtkMatrix3x3 elements as numpy array.
    The returned array is just a copy and so any modification in the array will not affect the input matrix.
    To set VTK matrix from a numpy array, use :py:meth:`vtkMatrixFromArray` or
    :py:meth:`updateVTKMatrixFromArray`.
    """
    from vtk import vtkMatrix4x4
    from vtk import vtkMatrix3x3
    import numpy as np
    if isinstance(vmatrix, vtkMatrix4x4):
        matrixSize = 4
    elif isinstance(vmatrix, vtkMatrix3x3):
        matrixSize = 3
    else:
        raise RuntimeError("Input must be vtk.vtkMatrix3x3 or vtk.vtkMatrix4x4")
    narray = np.eye(matrixSize)
    vmatrix.DeepCopy(narray.ravel(), vmatrix)
    return narray

def get_IJKtoRAS(dataNode):
    t1 = vtk.vtkMatrix4x4()
    if isinstance(dataNode, slicer.vtkMRMLScalarVolumeNode):
        volumeNode = dataNode
    elif isinstance(dataNode, slicer.vtkMRMLSequenceNode):
        volumeNode = dataNode.GetNthDataNode(0)
    volumeNode.GetIJKToRASMatrix(t1)
    return arrayFromVTKMatrix(t1)

def get_RAStoIJK(dataNode):
    t1 = vtk.vtkMatrix4x4()
    if isinstance(dataNode, slicer.vtkMRMLScalarVolumeNode):
        volumeNode = dataNode
    elif isinstance(dataNode, slicer.vtkMRMLSequenceNode):
        volumeNode = dataNode.GetNthDataNode(0)
    volumeNode.GetRASToIJKMatrix(t1)
    return arrayFromVTKMatrix(t1)

def run_crop_volume_single(cropInputNode, roiNode, spacing=[1.25,1.25,1.25], device='cuda'):
    src_img = img_from_volumeNode(cropInputNode)
    src_img_origin = np.array(cropInputNode.GetOrigin())
    # src_img_RAStoIJK_torch = torch.tensor(src_img_RAStoIJK, dtype=torch.get_default_dtype(), device=device)

    src_img_torch = torch.Tensor(np.ascontiguousarray(src_img)).to(dtype=torch.get_default_dtype(), device=device)[None,None,:,:,:]
    
    # roiNode defined in RAS (bounds, center, etc.)
    tgt_shape = (np.array(roiNode.GetSize())/np.array(spacing)).astype(int)
    roi_bounds = np.array(roiNode.GetSize())
    roi_center = np.array(roiNode.GetCenter())
    roi_corner = roi_center - roi_bounds/2

    # src_spacing = np.array(cropInputNode.GetSpacing()) # already embedded in src_img_RAStoIJK
    dst_spacing = np.array(spacing)

    '''
    1. get sample coordinates in RAS
        - roi always stays in RAS
        - tgt_img_RAS is expected to stay orthogonal to canonical axes
        - tgt_ras: dst_spacing in RAS, origin at roi_corner
    2. convert RAS coordinates to src_img_IJK
    '''
    tgt_IJKtoRAS = np.array([
        [dst_spacing[0],0,0, roi_corner[0]],
        [0,dst_spacing[1],0, roi_corner[1]],
        [0,0,dst_spacing[2], roi_corner[2]],
        [0,0,0,1],
    ]) # voxel in IJK -> dst_spacing in RAS, [0,0,0] ijk origin -> roi_corner
    src_img_RAStoIJK = get_RAStoIJK(cropInputNode)
    tgt_to_src_transformation = np.dot(src_img_RAStoIJK, tgt_IJKtoRAS)

    src_to_tgt_transformation = torch.linalg.inv(torch.tensor(tgt_to_src_transformation, dtype=torch.get_default_dtype(), device=device))
    cropped_img_torch = dcvm.transforms.apply_linear_transform_on_img_torch(src_img_torch, src_to_tgt_transformation, tgt_shape, grid_sample_mode='bilinear')

    return cropped_img_torch

def run_crop_volume_sequence(cropInputSequenceNode, roiNode, spacing=[1,1,1], device='cuda'):
    cropped_img_torch_list = []
    for idx in range(cropInputSequenceNode.GetNumberOfDataNodes()):
        cropInputNode = cropInputSequenceNode.GetNthDataNode(idx)
        cropped_img_torch = run_crop_volume_single(cropInputNode, roiNode, spacing=spacing, device=device)
        cropped_img_torch_list.append(cropped_img_torch)
    return cropped_img_torch_list

def update_cropOutputNode(cropped_img_torch, cropOutputNodeName, roiNode, spacing=[1,1,1]):
    # create or update cropOutputNode (good for sanity checks)
    if not cropOutputNodeName in [volumeNode.GetName() for volumeNode in slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')]:
        cropOutputNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", cropOutputNodeName)
    else:
        cropOutputNode = [volumeNode for volumeNode in slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode') if volumeNode.GetName() == cropOutputNodeName][0]
    cropped_img_vtk = cropped_img_torch.squeeze().cpu().numpy()
    cropped_img_vtk = cropped_img_vtk.transpose([2,1,0])
    slicer.util.updateVolumeFromArray(cropOutputNode, cropped_img_vtk)
    roi_bounds = np.zeros(6)
    roiNode.GetBounds(roi_bounds)
    roi_origin = roi_bounds.reshape(-1,2)[:,0]
    cropOutputNode.SetOrigin(roi_origin)
    cropOutputNode.SetSpacing(spacing)

    return cropOutputNode

def update_cropOutputSequenceNode(cropped_img_torch_list, cropOutputSequenceNodeName, roiNode, spacing=[1,1,1]):
    if not cropOutputSequenceNodeName in [sequenceNode.GetName() for sequenceNode in slicer.util.getNodesByClass('vtkMRMLSequenceNode')]:
        cropOutputSequenceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", cropOutputSequenceNodeName)
    else:
        cropOutputSequenceNode = [sequenceNode for sequenceNode in slicer.util.getNodesByClass('vtkMRMLSequenceNode') if sequenceNode.GetName() == cropOutputSequenceNodeName][0]
        cropOutputSequenceNode.RemoveAllDataNodes()

    dummyCropOutputVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "dummy_crop_output")

    for idx, cropped_img_torch in enumerate(cropped_img_torch_list):
        dummyCropOutputVolumeNode = update_cropOutputNode(cropped_img_torch, dummyCropOutputVolumeNode.GetName(), roiNode, spacing=spacing)
        cropOutputSequenceNode.SetDataNodeAtValue(dummyCropOutputVolumeNode, str(idx))

    slicer.mrmlScene.RemoveNode(dummyCropOutputVolumeNode)

    return cropOutputSequenceNode

# def update_cropSequenceBrowserNode(cropSequenceBrowserNodeName, cropOutputSequenceNode):
#     if not cropSequenceBrowserNodeName in [node.GetName() for node in slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode')]:
#         sequenceBrowserNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", cropSequenceBrowserNodeName)
#         sequenceBrowserNode.AddSynchronizedSequenceNode(cropOutputSequenceNode)
#     else:
#         sequenceBrowserNode = [node for node in slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode') if node.GetName() == cropSequenceBrowserNodeName][0]
#         # sequenceBrowserNode.RemoveAllProxyNodes() # this must come before RemoveAllSequencesNodes to properly remove ProxyNodes (associated with SequenceNodes)
#         # sequenceBrowserNode.RemoveAllSequenceNodes()
#     # # For displaying crop output sequence.. probably don't want to do this b/c final output model sequence will be in original image coordinates
#     # slicer.modules.sequences.toolBar().setActiveBrowserNode(sequenceBrowserNode)
#     # mergedProxyNode = sequenceBrowserNode.GetProxyNode(cropOutputNode)
#     # slicer.util.setSliceViewerLayers(background=mergedProxyNode)

def preprocess_img(img, eps=10, min_bound=-158, max_bound=864):
    if (img.max() - img.min()) > 1 + eps:
        img = dcvm.transforms.ct_normalize(img, min_bound=min_bound, max_bound=max_bound)
    return img

def run_heart_single(pytorch_model_heart, cropped_img, heart_verts, heart_elems, heart_cell_types, heart_fiber_ori, cropOutputNode):
    img = cropped_img.clone()
    device = img.device
    img_size = list(img.squeeze().shape)
    tgt_ijk_to_ras = get_IJKtoRAS(cropOutputNode) # transformed_verts_np in target img IJK coordinates. Need to convert to RAS coordinates.

    with torch.no_grad():
        output = pytorch_model_heart(img)
        displacement_field_tuple = output[0]

        # deform template verts
        heart_verts_torch = torch.tensor(heart_verts, dtype=torch.get_default_dtype(), device=device)
        interp_field_list = dcvm.transforms.interpolate_rescale_field_torch(displacement_field_tuple, [heart_verts_torch.unsqueeze(0)], img_size=img_size)
        transformed_verts_np = dcvm.transforms.move_verts_with_field([heart_verts_torch.unsqueeze(0)], interp_field_list)[0].squeeze().cpu().numpy()
        transformed_verts_np = dcvm.transforms.apply_linear_transform_on_verts(transformed_verts_np, tgt_ijk_to_ras)

        if heart_fiber_ori is None:
            heart_fiber_ori_transformed = None
        else:
            # deform fiber_ori-defining points
            heart_fiber_ori_transformed = {}
            for key in heart_elems.keys():
                mesh_pv = dcvm.ops.mesh_to_UnstructuredGrid(heart_verts, heart_elems[key], heart_cell_types[key])
                centers = mesh_pv.cell_centers().points
                dirs1 = heart_fiber_ori[key][:,:3]
                dirs2 = heart_fiber_ori[key][:,3:]
                scale = 0.01
                dirs1_scaled = dirs1/np.linalg.norm(dirs1, axis=1)[:,None]*scale
                dirs2_scaled = dirs2/np.linalg.norm(dirs2, axis=1)[:,None]*scale
                dirs1_world = centers + dirs1_scaled
                dirs2_world = centers + dirs2_scaled

                centers_torch = torch.tensor(centers, dtype=torch.get_default_dtype(), device=device)
                dirs1_world_torch = torch.tensor(dirs1_world, dtype=torch.get_default_dtype(), device=device)
                dirs2_world_torch = torch.tensor(dirs2_world, dtype=torch.get_default_dtype(), device=device)

                interp_field_list = dcvm.transforms.interpolate_rescale_field_torch(displacement_field_tuple, [centers_torch.unsqueeze(0)], img_size=img_size)
                centers_transformed = dcvm.transforms.move_verts_with_field([centers_torch.unsqueeze(0)], interp_field_list)[0].squeeze().cpu().numpy()
                interp_field_list = dcvm.transforms.interpolate_rescale_field_torch(displacement_field_tuple, [dirs1_world_torch.unsqueeze(0)], img_size=img_size)
                dirs1_world_transformed = dcvm.transforms.move_verts_with_field([dirs1_world_torch.unsqueeze(0)], interp_field_list)[0].squeeze().cpu().numpy()
                interp_field_list = dcvm.transforms.interpolate_rescale_field_torch(displacement_field_tuple, [dirs2_world_torch.unsqueeze(0)], img_size=img_size)
                dirs2_world_transformed = dcvm.transforms.move_verts_with_field([dirs2_world_torch.unsqueeze(0)], interp_field_list)[0].squeeze().cpu().numpy()

                centers_transformed = dcvm.transforms.apply_linear_transform_on_verts(centers_transformed, tgt_ijk_to_ras)
                dirs1_world_transformed = dcvm.transforms.apply_linear_transform_on_verts(dirs1_world_transformed, tgt_ijk_to_ras)
                dirs2_world_transformed = dcvm.transforms.apply_linear_transform_on_verts(dirs2_world_transformed, tgt_ijk_to_ras)

                dirs1_local = dirs1_world_transformed - centers_transformed
                dirs2_local = dirs2_world_transformed - centers_transformed

                dirs1_local = dirs1_local/np.linalg.norm(dirs1_local, axis=1)[:,None]
                dirs2_local = dirs2_local/np.linalg.norm(dirs2_local, axis=1)[:,None]

                heart_fiber_ori_transformed[key] = np.concatenate([dirs1_local, dirs2_local], axis=1)    
    
    # transformed_verts_homo_np = np.concatenate([transformed_verts_np, np.ones([transformed_verts_np.shape[0], 1])], axis=1).T
    # transformed_verts_np = np.dot(tgt_ijk_to_ras, transformed_verts_homo_np)[:3,:].T

    mesh_pv_dict = {}
    for key in heart_elems.keys():
        mesh_pv_dict[key] = dcvm.ops.mesh_to_UnstructuredGrid(transformed_verts_np, heart_elems[key], heart_cell_types[key])
    
    return mesh_pv_dict, heart_fiber_ori_transformed

def run_ca2_single(pytorch_model_ca2, cropped_img, origInputNode, cropOutputNode):
    with torch.no_grad():
        output = pytorch_model_ca2(cropped_img) # output: [1,1,128,128,128]
    
    ca2_cropped_pv = dcvm.ops.seg_to_polydata(output.squeeze().cpu().numpy())
    ca2_pv = ca2_cropped_pv.copy()

    # ca2_pv.points in target img IJK coordinates. Need to convert to RAS coordinates, and then into source img IJK coordinates.
    ca2_pv_points = ca2_pv.points.copy()
    tgt_ijk_to_ras = get_IJKtoRAS(cropOutputNode)
    ras_to_src_ijk = get_RAStoIJK(origInputNode)
    ca2_pv_points_homo = np.concatenate([ca2_pv_points, np.ones([ca2_pv_points.shape[0], 1])], axis=1).T # should be replaced with dcvm.transforms.apply_linear_transform_on_verts
    ca2_pv.points = np.dot(ras_to_src_ijk, np.dot(tgt_ijk_to_ras, ca2_pv_points_homo))[:3,:].T

    if isinstance(origInputNode, slicer.vtkMRMLSequenceNode):
        inputVolumeNode = origInputNode.GetNthDataNode(0)
    elif isinstance(origInputNode, slicer.vtkMRMLScalarVolumeNode):
        inputVolumeNode = origInputNode
    # ca2_pv already in ijk coordinates, don't need to mess with spacing or origin
    ca2_seg = dcvm.ops.polydata_to_seg(ca2_pv, dims=inputVolumeNode.GetImageData().GetDimensions(), spacing=[1,1,1], origin=[0,0,0], tolerance=0.2)
    
    return ca2_pv, ca2_seg

def get_aorta_lv_stl(mesh_pv_dict, offset_dist=1.5, spacing_scaling=2, pyacvd_cluster_arg=20000):
    faces_filepath = 'C:/Users/danpa/OneDrive/Documents/research_code/Deep-Cardiac-Volumetric-Mesh-dev/template_for_deform/combined_v12/stl_aorta_lv_inner_surf_capped.pt'
    # faces_filepath = 'C:/Users/danpa/OneDrive/Documents/research_code/Deep-Cardiac-Volumetric-Mesh-dev/template_for_deform/combined_v12/stl_aw_inner_surf_capped.pt'
    
    # get verts and faces_capped
    verts, _ = dcvm.ops.get_verts_faces_from_pyvista(list(mesh_pv_dict.values())[0])
    faces_capped = torch.load(faces_filepath)

    # get seg_offset
    inner_surf_pv = dcvm.ops.mesh_to_PolyData(verts, faces_capped)
    seg, [dims, spacing, origin] = dcvm.ops.polydata_to_seg(inner_surf_pv, spacing=np.ones(3)/spacing_scaling, return_dims_spacing_origin=True)
    distances = distance_transform_edt(1-seg)
    seg_offset = np.logical_and(distances > 0, distances <= offset_dist*spacing_scaling).astype(float)

    ''' for regular '''
    # get seg from aorta_cap
    mitral_annulus_pv = dcvm.ops.mesh_to_PolyData(verts, [faces_capped[-1]]).clean()
    aorta_cap_pv = dcvm.ops.mesh_to_PolyData(verts, [faces_capped[-2]]).clean()
    def sample_points_vtk(mesh_pv, spacing_scaling=1):
        import vtk
        point_sampler = vtk.vtkPolyDataPointSampler()
        point_sampler.SetInputData(mesh_pv)
        point_sampler.SetDistance(0.1*spacing_scaling)
        point_sampler.SetPointGenerationMode(point_sampler.REGULAR_GENERATION)
        point_sampler.Update()
        points_sampled = pv.PolyData(point_sampler.GetOutput()).points
        return points_sampled

    aorta_cap_pts = sample_points_vtk(aorta_cap_pv)
    points_sampled = aorta_cap_pts

    # mitral_annulus_pts = sample_points_vtk(mitral_annulus_pv)
    # points_sampled = np.vstack([mitral_annulus_pts, aorta_cap_pts])

    pcl_occ_grid = np.zeros(dims)
    occ_idxes = (np.round(spacing_scaling*(points_sampled - origin))).astype(int)
    pcl_occ_grid[occ_idxes[:,0], occ_idxes[:,1], occ_idxes[:,2]] = 1
    dist_pcl_occ_grid = distance_transform_edt(1-pcl_occ_grid)
    pcl_occ_grid = (dist_pcl_occ_grid <= offset_dist*spacing_scaling).astype(float)

    seg_no_aorta_cap = np.clip(seg_offset - pcl_occ_grid, 0, 1)

    # convert back to polydata --> remesh --> smooth
    no_aorta_cap_pv = dcvm.ops.seg_to_polydata(seg_no_aorta_cap, isolevel=0.5, dims=dims, spacing=spacing, origin=origin)

    smooth_init_pv = no_aorta_cap_pv.smooth(n_iter=5, relaxation_factor=0.5)

    # ''' for truncated aw '''
    # def calc_plane_params(lm1, lm2, lm3):
    #     x1 = lm1[0]
    #     y1 = lm1[1]
    #     z1 = lm1[2]
        
    #     x2 = lm2[0]
    #     y2 = lm2[1]
    #     z2 = lm2[2]
        
    #     x3 = lm3[0]
    #     y3 = lm3[1]
    #     z3 = lm3[2]
        
    #     vector1 = [x2 - x1, y2 - y1, z2 - z1]
    #     vector2 = [x3 - x1, y3 - y1, z3 - z1]
        
    #     cross_product = [vector1[1] * vector2[2] - vector1[2] * vector2[1], -1 * (vector1[0] * vector2[2] - vector1[2] * vector2[0]), vector1[0] * vector2[1] - vector1[1] * vector2[0]]
        
    #     a = cross_product[0]
    #     b = cross_product[1]
    #     c = cross_product[2]
    #     d = - (cross_product[0] * x1 + cross_product[1] * y1 + cross_product[2] * z1)
        
    #     return a, b, c, d

    # plane1_pt_idxes = [5242, 5607, 5581]
    # plane2_pt_idxes = [20614, 22070, 12596]
    # plane1_pts = inner_surf_pv.points[plane1_pt_idxes]
    # plane2_pts = inner_surf_pv.points[plane2_pt_idxes]
    # x1,y1,z1,d1 = calc_plane_params(*plane1_pts)
    # x2,y2,z2,d2 = calc_plane_params(*plane1_pts)
    # plane1_normal_normalized = np.array([x1,y1,z1])/np.linalg.norm(np.array([x1,y1,z1]))
    # plane2_normal_normalized = np.array([x2,y2,z2])/np.linalg.norm(np.array([x2,y2,z2]))

    # xmesh, ymesh, zmesh = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), np.arange(dims[2]))
    # xyz_mesh = np.stack([xmesh, ymesh, zmesh], axis=-1)
    # max_dim = np.diff(np.array(inner_surf_pv.bounds).reshape(-1,2), axis=1).max()

    # plane1_pv = pv.Plane(plane1_pts[0], plane1_normal_normalized, i_size=max_dim, j_size=max_dim)
    # plane1_seg = np.zeros(dims)
    # dir_vec = xyz_mesh.reshape(-1,3) - ((plane1_pts[0]-origin)*2)
    # dir_vec = dir_vec/np.linalg.norm(dir_vec)
    # cell_select_bools = np.dot(dir_vec, -plane1_normal_normalized) > 0
    # selected_xyz = xyz_mesh.reshape(-1,3)[cell_select_bools]
    # plane1_seg[selected_xyz[:,0], selected_xyz[:,1], selected_xyz[:,2]] = 1

    # plane2_pv = pv.Plane(plane2_pts[0], plane2_normal_normalized, i_size=max_dim, j_size=max_dim)
    # plane2_seg = np.zeros(dims)
    # dir_vec = xyz_mesh.reshape(-1,3) - ((plane2_pts[0]-origin)*2)
    # dir_vec = dir_vec/np.linalg.norm(dir_vec)
    # cell_select_bools = np.dot(dir_vec, plane2_normal_normalized) > 0
    # selected_xyz = xyz_mesh.reshape(-1,3)[cell_select_bools]
    # plane2_seg[selected_xyz[:,0], selected_xyz[:,1], selected_xyz[:,2]] = 1

    # truncated_seg = np.clip(seg_offset - plane1_seg - plane2_seg, 0, 1)
    # truncated_pv = dcvm.ops.seg_to_polydata(truncated_seg, isolevel=0.5, dims=dims, spacing=spacing, origin=origin)

    # smooth_init_pv = truncated_pv.smooth(n_iter=5, relaxation_factor=0.5)

    clus = pyacvd.Clustering(smooth_init_pv)
    clus.cluster(pyacvd_cluster_arg)
    remeshed_pv = clus.create_mesh()

    smooth_pv = remeshed_pv.smooth(n_iter=2, relaxation_factor=0.5)
    
    return smooth_pv

def get_leaflets_stl(mesh_pv_dict, offset_dist=0.5, spacing_scaling=3, pyacvd_cluster_arg=5000, fuse_leaflets=False):
    # print(mesh_pv_dict.keys()) # ['lv', 'aorta', 'av_l1', 'av_l2', 'av_l3']

    leaflet_keys = ['av_l1', 'av_l2', 'av_l3']
    combined_leaflets_pv = pv.merge([mesh_pv_dict[key] for key in leaflet_keys]).extract_surface().clean()
    dims, spacing, origin = dcvm.ops.get_default_dims_spacing_origin(combined_leaflets_pv, spacing=np.ones(3)/spacing_scaling)

    aorta_orig_pv = mesh_pv_dict['aorta'].extract_surface().clean() # to make sure leaflets overlap with aorta at attachment curves
    aorta_seg = dcvm.ops.polydata_to_seg(aorta_orig_pv, dims=dims, spacing=spacing, origin=origin)

    leaflet_seg_list = []
    for key in leaflet_keys:
        leaflet_seg = dcvm.ops.polydata_to_seg(mesh_pv_dict[key].extract_surface().clean(), dims=dims, spacing=spacing, origin=origin)
        distances = distance_transform_edt(1-leaflet_seg)
        leaflet_seg_offset_added = (distances <= offset_dist*spacing_scaling).astype(float)
    
        neighbor_kernel_width = 7
        closing_kernel_width = 3
        device = 'cuda'
        neighbor_kernel = torch.tensor(dcvm.ops.get_binary_mask_sphere(neighbor_kernel_width, neighbor_kernel_width/2), dtype=torch.get_default_dtype(), device=device)[None,None]
        closing_kernel = torch.tensor(dcvm.ops.get_binary_mask_sphere(closing_kernel_width, closing_kernel_width/2), dtype=torch.get_default_dtype(), device=device)[None,None]

        aorta_seg_torch = torch.tensor(aorta_seg, dtype=torch.get_default_dtype(), device=device)[None,None]
        leaflet_seg_torch = torch.tensor(leaflet_seg_offset_added, dtype=torch.get_default_dtype(), device=device)[None,None]
        filtered_aorta = dcvm.ops.binary_filter_by_neighbor_torch(aorta_seg_torch, leaflet_seg_torch, neighbor_kernel)
        filtered_aorta_and_leaflet = torch.clamp(filtered_aorta + leaflet_seg_torch, 0, 1)
        filtered_aorta_and_leaflet_closed = dcvm.ops.binary_closing_torch(filtered_aorta_and_leaflet, closing_kernel, n_iter=1)
        leaflet_seg_list.append(filtered_aorta_and_leaflet_closed.squeeze().cpu().numpy())
    
    # convert back to polydata --> remesh --> smooth
    if fuse_leaflets:
        leaflet_seg_combined = np.clip(np.stack(leaflet_seg_list, axis=-1).sum(axis=-1), 0, 1)
        leaflet_combined_pv = dcvm.ops.seg_to_polydata(leaflet_seg_combined, isolevel=0.5, dims=dims, spacing=spacing, origin=origin)
    else:
        leaflet_combined_pv = pv.merge([dcvm.ops.seg_to_polydata(seg, isolevel=0.5, dims=dims, spacing=spacing, origin=origin) for seg in leaflet_seg_list])
    
    smooth_init_pv = leaflet_combined_pv.smooth(n_iter=5, relaxation_factor=0.5)

    clus = pyacvd.Clustering(smooth_init_pv)
    clus.cluster(pyacvd_cluster_arg)
    remeshed_pv = clus.create_mesh()

    smooth_pv = remeshed_pv.smooth(n_iter=2, relaxation_factor=0.5)
    
    return smooth_pv