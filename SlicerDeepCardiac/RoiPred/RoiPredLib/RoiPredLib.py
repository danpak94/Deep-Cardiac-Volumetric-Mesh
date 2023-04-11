import numpy as np
import torch
import dcvm

import slicer
import vtk

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
    2. conver RAS coordinates to src_img_IJK
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

def preprocess_img(img):
    if (img.max() - img.min()) > 1:
        img = dcvm.transforms.ct_normalize(img, min_bound=-158.0, max_bound=864.0)
    return img

def run_heart_single(pytorch_model_heart, cropped_img, verts_template_torch, heart_elems, heart_cell_types, heart_faces, cropOutputNode):
    img = cropped_img.clone()
    device = img.device
    img_size = list(img.squeeze().shape)

    with torch.no_grad():
        output = pytorch_model_heart(img)
        displacement_field_tuple = output[0]
        interp_field_list = dcvm.transforms.interpolate_rescale_field_torch(displacement_field_tuple, [verts_template_torch.to(device).unsqueeze(0)], img_size=img_size)
        transformed_verts_np = dcvm.transforms.move_verts_with_field([verts_template_torch.to(device).unsqueeze(0)], interp_field_list)[0].squeeze().cpu().numpy()

    # transformed_verts_np in target img IJK coordinates. Need to convert to RAS coordinates.
    tgt_ijk_to_ras = get_IJKtoRAS(cropOutputNode)
    transformed_verts_homo_np = np.concatenate([transformed_verts_np, np.ones([transformed_verts_np.shape[0], 1])], axis=1).T
    transformed_verts_np = np.dot(tgt_ijk_to_ras, transformed_verts_homo_np)[:3,:].T

    mesh_pv_dict = {}
    for key in heart_elems.keys():
        mesh_pv_dict[key] = dcvm.ops.mesh_to_UnstructuredGrid(transformed_verts_np, heart_elems[key], heart_cell_types[key])
    for key in heart_faces.keys():
        mesh_pv_dict[key] = dcvm.ops.mesh_to_PolyData(transformed_verts_np, heart_faces[key])
    
    return mesh_pv_dict

def run_ca2_single(pytorch_model_ca2, cropped_img, origInputNode, cropOutputNode):
    with torch.no_grad():
        output = pytorch_model_ca2(cropped_img) # output: [1,1,128,128,128]
    
    ca2_cropped_pv = dcvm.ops.seg_to_polydata(output.squeeze().cpu().numpy())
    ca2_pv = ca2_cropped_pv.copy()

    # ca2_pv.points in target img IJK coordinates. Need to convert to RAS coordinates, and then into source img IJK coordinates.
    ca2_pv_points = ca2_pv.points.copy()
    tgt_ijk_to_ras = get_IJKtoRAS(cropOutputNode)
    ras_to_src_ijk = get_RAStoIJK(origInputNode)
    ca2_pv_points_homo = np.concatenate([ca2_pv_points, np.ones([ca2_pv_points.shape[0], 1])], axis=1).T
    ca2_pv.points = np.dot(ras_to_src_ijk, np.dot(tgt_ijk_to_ras, ca2_pv_points_homo))[:3,:].T

    if isinstance(origInputNode, slicer.vtkMRMLSequenceNode):
        inputVolumeNode = origInputNode.GetNthDataNode(0)
    elif isinstance(origInputNode, slicer.vtkMRMLScalarVolumeNode):
        inputVolumeNode = origInputNode
    # ca2_pv already in ijk coordinates, don't need to mess with spacing or origin
    ca2_seg = dcvm.ops.polydata_to_seg(ca2_pv, dims=inputVolumeNode.GetImageData().GetDimensions(), spacing=[1,1,1], origin=[0,0,0])
    
    return ca2_pv, ca2_seg