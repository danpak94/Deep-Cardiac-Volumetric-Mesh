import importlib
import numpy as np
import torch
import pyvista as pv
import dcvm
import matplotlib
colors_models = matplotlib.colormaps['Set1'].colors
colors_segments = matplotlib.colormaps['Set2'].colors

import slicer
import qt
import vtk

class CustomQtSignalSender(qt.QObject):
    # https://discourse.slicer.org/t/how-to-use-signals-and-slots-in-slicer-3d/14013/5
    # https://discourse.slicer.org/t/use-of-qt-signal-leads-to-a-crash-on-exit/8321
    signal = qt.Signal(object) # we have to put this line here (not inside __init__) to prevent crashing.. idk why but it's important...
    def __init__(self):
        super(CustomQtSignalSender, self).__init__(None)

class ProgressBarAndRunTime():
    def __init__(self, progressBar):
        self.progressBar = progressBar
        self.timer = qt.QElapsedTimer()

    def start(self, value=0, maximum=1, text='Running ... %v / {}'):
        self.progressBar.value = value
        self.progressBar.maximum = maximum
        self.progressBar.setFormat(text.format(maximum))
        self.progressBar.show()
        slicer.app.processEvents() # https://github.com/Slicer/Slicer/blob/a5f75351073ef62fd6198d9480d86c0009d70f9b/Modules/Scripted/DICOMLib/DICOMSendDialog.py
        self.timer.start()

    def step(self, value):
        if value < self.progressBar.maximum:
            self.progressBar.value = value
            self.progressBar.setFormat('Running ... %v / {}'.format(self.progressBar.maximum))
        else:
            self.end()

    def end(self, text='Done: {} seconds'):
        time_elapsed = self.timer.elapsed() / 1000 # originally in milliseconds
        self.progressBar.setValue(self.progressBar.maximum)
        self.progressBar.setFormat(text.format(time_elapsed))

# def organize_subject_hierarchy(self):
#     '''
#     for each volume_node or volume_sequence_node:
#     1. create a subject item with the same name
#     2. place volume inside the subject
#     3. place all other outputs under the same subject
#     '''
#     shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
#     sceneId = shNode.GetSceneItemID()
#     childIds = vtk.vtkIdList() # dummy to save Ids
#     shNode.GetItemChildren(sceneId, childIds) # for all children
#     for itemIdIndex in range(childIds.GetNumberOfIds()):
#         shItemId = childIds.GetId(itemIdIndex)
#         if isinstance(shNode.GetItemDataNode(shItemId), slicer.vtkMRMLScalarVolumeNode):
#             subjectId = shNode.CreateSubjectItem(sceneId, shNode.GetItemName(shItemId))
#             shNode.SetItemParent(shItemId, subjectId)
#         if isinstance(shNode.GetItemDataNode(shItemId), slicer.vtkMRMLSequenceNode):
#             seqDataNode0 = shNode.GetItemDataNode(shItemId).GetNthDataNode(0)
#             if isinstance(seqDataNode0, slicer.vtkMRMLScalarVolumeNode):
#                 patient_num = seqDataNode0.GetName().split('_phase')[0]
#                 subjectId = shNode.CreateSubjectItem(sceneId, patient_num)
#                 shNode.SetItemParent(shItemId, subjectId)
#             pass

def img_from_volumeNode(volumeNode):
    img = slicer.util.arrayFromVolume(volumeNode) # RAS
    img = img.transpose(2,1,0) # b/c vtk/Slicer flips when getting array from volume
    mat = vtk.vtkMatrix4x4(); volumeNode.GetIJKToRASDirectionMatrix(mat)
    if mat.GetElement(0,0) == -1: # flip based on IJK to RAS DirectionMatrix
        img = np.flip(img, axis=0)
    if mat.GetElement(1,1) == -1:
        img = np.flip(img, axis=1)
    return img

# roiNode = slicer.util.getFirstNodeByName('RoiPred_ROI')

def run_crop_volume_single(cropInputNode, roiNode, spacing=[1,1,1], device='cuda'):
    src_img = img_from_volumeNode(cropInputNode)
    src_img_torch = torch.Tensor(np.ascontiguousarray(src_img)).to(dtype=torch.get_default_dtype(), device=device)[None,None,:,:,:]
    
    tgt_shape = (np.array(roiNode.GetSize())/np.array(spacing)).astype(int)
    crop_center = np.array(roiNode.GetCenter())

    src_spacing = np.array(cropInputNode.GetSpacing())
    dst_spacing = np.array(spacing)
    downsample_ratio = dst_spacing / src_spacing

    src_to_tgt_transformation = torch.linalg.inv(torch.tensor([
        [1*downsample_ratio[0],0,0,crop_center[0]-tgt_shape[0]*downsample_ratio[0]/2],
        [0,1*downsample_ratio[1],0,crop_center[1]-tgt_shape[1]*downsample_ratio[1]/2],
        [0,0,1*downsample_ratio[2],crop_center[2]-tgt_shape[2]*downsample_ratio[2]/2],
        [0,0,0,1],
    ], dtype=torch.get_default_dtype(), device=device))

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
    
    # # debug display cropOutputNode
    # appLogic = slicer.app.applicationLogic()
    # selectionNode = appLogic.GetSelectionNode()
    # selectionNode.SetActiveVolumeID(cropOutputNode.GetID())
    # appLogic.PropagateVolumeSelection()

    return cropOutputNode

def update_cropOutputSequenceNode(cropped_img_torch_list, cropOutputSequenceNodeName, roiNode, spacing=[1,1,1]):
    if not cropOutputSequenceNodeName in [sequenceNode.GetName() for sequenceNode in slicer.util.getNodesByClass('vtkMRMLSequenceNode')]:
        cropOutputSequenceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", cropOutputSequenceNodeName)
    else:
        cropOutputSequenceNode = [sequenceNode for sequenceNode in slicer.util.getNodesByClass('vtkMRMLSequenceNode') if sequenceNode.GetName() == cropOutputSequenceNodeName][0]
        cropOutputSequenceNode.RemoveAllDataNodes()

    dummyCropOutputVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "dummyCropOutputVolumeNode")
    dummyCropOutputVolumeNode.SetName('dummy_crop_output')

    for idx, cropped_img_torch in enumerate(cropped_img_torch_list):
        dummyCropOutputVolumeNode = update_cropOutputNode(cropped_img_torch, dummyCropOutputVolumeNode.GetName(), roiNode, spacing=spacing)
        cropOutputSequenceNode.SetDataNodeAtValue(dummyCropOutputVolumeNode, str(idx))

    slicer.mrmlScene.RemoveNode(dummyCropOutputVolumeNode)

    return cropOutputSequenceNode

def update_cropSequenceBrowserNode(cropSequenceBrowserNodeName, cropOutputSequenceNode):
    if not cropSequenceBrowserNodeName in [node.GetName() for node in slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode')]:
        sequenceBrowserNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", cropSequenceBrowserNodeName)
        sequenceBrowserNode.AddSynchronizedSequenceNode(cropOutputSequenceNode)
    else:
        sequenceBrowserNode = [node for node in slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode') if node.GetName() == cropSequenceBrowserNodeName][0]
        # sequenceBrowserNode.RemoveAllProxyNodes() # this must come before RemoveAllSequencesNodes to properly remove ProxyNodes (associated with SequenceNodes)
        # sequenceBrowserNode.RemoveAllSequenceNodes()
    
    # # For displaying crop output sequence.. probably don't want to do this b/c final output model sequence will be in original image coordinates
    # slicer.modules.sequences.toolBar().setActiveBrowserNode(sequenceBrowserNode)
    # mergedProxyNode = sequenceBrowserNode.GetProxyNode(cropOutputNode)
    # slicer.util.setSliceViewerLayers(background=mergedProxyNode)

def preprocess_img(img):
    if (img.max() - img.min()) > 1:
        img = dcvm.transforms.ct_normalize(img, min_bound=-158.0, max_bound=864.0)
    return img

def run_heart_single(pytorch_model_heart, cropped_img, verts_template_torch, heart_elems, heart_cell_types, heart_faces, origin_translate=[0,0,0], downsample_ratio=[1,1,1]):
    img = cropped_img.clone()
    device = img.device
    img_size = list(img.squeeze().shape)

    with torch.no_grad():
        output = pytorch_model_heart(img)
        displacement_field_tuple = output[0]
        interp_field_list = dcvm.transforms.interpolate_rescale_field_torch(displacement_field_tuple, [verts_template_torch.to(device).unsqueeze(0)], img_size=img_size)
        transformed_verts_np = dcvm.transforms.move_verts_with_field([verts_template_torch.to(device).unsqueeze(0)], interp_field_list)[0].squeeze().cpu().numpy()

    transformed_verts_np *= np.array(downsample_ratio) # assume 1mm/voxel --> 1*downsample_ratio mm/voxel
    transformed_verts_np += np.array(origin_translate)[None,:]

    mesh_pv_dict = {}
    for key in heart_elems.keys():
        mesh_pv_dict[key] = dcvm.ops.mesh_to_UnstructuredGrid(transformed_verts_np, heart_elems[key], heart_cell_types[key])
    for key in heart_faces.keys():
        mesh_pv_dict[key] = dcvm.ops.mesh_to_PolyData(transformed_verts_np, heart_faces[key])
    
    return mesh_pv_dict

def run_ca2_single(pytorch_model_ca2, cropped_img, origInputNode, origin_translate=[0,0,0], downsample_ratio=[1,1,1]):
    with torch.no_grad():
        output = pytorch_model_ca2(cropped_img) # output: [1,1,128,128,128]
    
    ca2_cropped_pv = dcvm.ops.seg_to_polydata(output.squeeze().cpu().numpy())
    ca2_pv = ca2_cropped_pv.copy()
    ca2_pv.points *= np.array(downsample_ratio)
    ca2_pv.points += np.array(origin_translate)[None,:]

    if isinstance(origInputNode, slicer.vtkMRMLSequenceNode):
        inputVolumeNode = origInputNode.GetNthDataNode(0)
    elif isinstance(origInputNode, slicer.vtkMRMLScalarVolumeNode):
        inputVolumeNode = origInputNode
    ca2_seg = dcvm.ops.polydata_to_seg(ca2_pv, dims=inputVolumeNode.GetImageData().GetDimensions()[::-1], spacing=inputVolumeNode.GetSpacing(), origin=inputVolumeNode.GetOrigin())
    
    return ca2_pv, ca2_seg

def update_model_nodes_from_pv_dict(mesh_pv_dict, modelNames_dict):
    if not set(modelNames_dict.values()) <= set([modelNode.GetName() for modelNode in slicer.util.getNodesByClass('vtkMRMLModelNode')]):
        # create separate model node for each mesh component here (lv, aorta, l1, l2, l3, etc.)
        for (key, mesh_pv), color in zip(mesh_pv_dict.items(), colors_models):
            modelNode = slicer.modules.models.logic().AddModel(pv.PolyData())
            modelNode.SetName(modelNames_dict[key])
            modelNode.SetAndObserveMesh(mesh_pv)
        modelNodes_dict = {key: [node for node in slicer.util.getNodesByClass('vtkMRMLModelNode') if node.GetName() == modelName][0] for key, modelName in modelNames_dict.items()}
    else:
        # update existing model nodes if they exist
        modelNodes_dict = {key: [node for node in slicer.util.getNodesByClass('vtkMRMLModelNode') if node.GetName() == modelName][0] for key, modelName in modelNames_dict.items()}
        for key, modelNode in modelNodes_dict.items():
            modelNode.SetAndObserveMesh(mesh_pv_dict[key])
    return modelNodes_dict

def update_model_sequence_nodes_from_pv_dict_list(mesh_pv_dict_list, modelNames_dict):
    tempModelNode = slicer.modules.models.logic().AddModel(pv.PolyData())

    if not set(modelNames_dict.values()) <= set([modelNode.GetName() for modelNode in slicer.util.getNodesByClass('vtkMRMLSequenceNode')]): # subset
        # create separate sequence node for each mesh component here (lv, aorta, l1, l2, l3, etc.)
        modelSequenceNodes_dict = {key: slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", modelName) for key, modelName in modelNames_dict.items()}
    else:
        # update old sequence nodes if they exist
        modelSequenceNodes_dict = {key: [node for node in slicer.util.getNodesByClass('vtkMRMLSequenceNode') if node.GetName() == modelName][0] for key, modelName in modelNames_dict.items()}
        for node in modelSequenceNodes_dict.values():
            node.RemoveAllDataNodes()
    
    for idx, mesh_pv_dict in enumerate(mesh_pv_dict_list):
        for key in mesh_pv_dict.keys():
            tempModelNode.SetName('dummy_{}_{}'.format(modelNames_dict[key], idx))
            tempModelNode.SetAndObserveMesh(mesh_pv_dict[key])
            modelSequenceNodes_dict[key].SetDataNodeAtValue(tempModelNode, str(idx))

    slicer.mrmlScene.RemoveNode(tempModelNode)

    return modelSequenceNodes_dict

def update_model_nodes_display(modelNodes):
    for modelNode, color in zip(modelNodes, colors_models):
        if modelNode is not None:
            modelNode.GetDisplayNode().SetColor(*color)
            modelNode.GetDisplayNode().SetEdgeVisibility(True)
            modelNode.GetDisplayNode().SetSliceIntersectionVisibility(True)
            modelNode.GetDisplayNode().SetSliceIntersectionOpacity(0.3)
            modelNode.GetDisplayNode().SetSliceIntersectionThickness(5)
            modelNode.GetDisplayNode().SetVisibility(True)

def update_seg_node_from_np(segmentArray, segmentationNodeName, segmentName, inputVolumeNode):
    segmentationNodes = [node for node in slicer.util.getNodesByClass('vtkMRMLSegmentationNode') if segmentationNodeName in node.GetName()]
    if len(segmentationNodes) == 0:
        # create new Segmentation node
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.SetName(segmentationNodeName)
    else:
        # update existing Segmentation node
        segmentationNode = segmentationNodes[0]
    segmentationNode.CreateDefaultDisplayNodes()

    # create new Segment only if segment_name doesn't exist in Segmentation already
    if segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName) == '':
        segmentationNode.GetSegmentation().AddEmptySegment(segmentName)

    # grab segmentNode to update
    segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)

    # update segment
    segmentArray = segmentArray.transpose([2,1,0])
    slicer.util.updateSegmentBinaryLabelmapFromArray(segmentArray, segmentationNode, segmentId, inputVolumeNode)

    return segmentationNode
    
def update_seg_node_display(segmentationNode, segmentNames):
    # segment color
    for segmentName, color in zip(segmentNames, colors_segments):
        segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
        # DPDP if segmentId is not None?
        if segmentId is not None:
            segmentNode = segmentationNode.GetSegmentation().GetSegment(segmentId)
            segmentNode.SetColor(color)

    # segmentation surface display
    # segmentationNode.GetSegmentation().SetConversionParameter("Surface smoothing", "False") # I wish I could do this, but doesn't work..
    segmentationNode.GetSegmentation().SetConversionParameter("Smoothing factor", "0.0")
    segmentationNode.CreateClosedSurfaceRepresentation()
    segmentationNode.GetDisplayNode().SetVisibility(True)

def update_seg_sequence_node_from_seg_list(ca2_seg_list, segmentationSequenceNodeName, segmentName, imgSequenceNode):
    tempSegmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")

    if not segmentationSequenceNodeName in [node.GetName() for node in slicer.util.getNodesByClass('vtkMRMLSequenceNode')]:
        # create separate sequence node for each mesh component here (lv, aorta, l1, l2, l3, etc.)
        segmentationSequenceNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", segmentationSequenceNodeName)
    else:
        # update old sequence nodes if they exist
        segmentationSequenceNode = [node for node in slicer.util.getNodesByClass('vtkMRMLSequenceNode') if node.GetName() == segmentationSequenceNodeName][0]
        segmentationSequenceNode.RemoveAllDataNodes()
    
    for idx, ca2_seg in enumerate(ca2_seg_list):
        segmentationNodeName = 'dummy_{}_{}'.format(segmentationSequenceNodeName, idx)
        tempSegmentationNode.SetName(segmentationNodeName)
        update_seg_node_from_np(ca2_seg, segmentationNodeName, segmentName, imgSequenceNode.GetNthDataNode(idx))
        update_seg_node_display(tempSegmentationNode, [segmentName])
        segmentationSequenceNode.SetDataNodeAtValue(tempSegmentationNode, str(idx))

    slicer.mrmlScene.RemoveNode(tempSegmentationNode)

    return segmentationSequenceNode

def update_outputSequenceBrowserNode(outputSequenceBrowserNodeName, cropInputNode=None, modelSequenceNodes_dict=None, segSequenceNode=None):
    # init browser node if it doesn't exist
    if not outputSequenceBrowserNodeName in [node.GetName() for node in slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode')]:
        sequenceBrowserNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", outputSequenceBrowserNodeName)
        sequenceBrowserNode.SetPlaybackRateFps(2.0) # slower b/c synced with model
        currentItemNumber = 0
    else: # grab browser node and delete proxy nodes to refresh display
        sequenceBrowserNode = [node for node in slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode') if node.GetName() == outputSequenceBrowserNodeName][0]
        currentItemNumber = sequenceBrowserNode.GetSelectedItemNumber()

    # # debug 
    # sequenceBrowserNode = [node for node in slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode') if 'output' in node.GetName()][0]
    # segProxyNode = [node for node in slicer.util.getNodesByClass('vtkMRMLSegmentationNode') if 'Segmentation' in node.GetName()][0]

    # img sequence
    if cropInputNode is not None: # original image
        if not (sequenceBrowserNode.IsSynchronizedSequenceNode(cropInputNode) or sequenceBrowserNode.GetMasterSequenceNode() == cropInputNode):
            sequenceBrowserNode.AddSynchronizedSequenceNode(cropInputNode)
        
        imgProxyNode = sequenceBrowserNode.GetProxyNode(cropInputNode)
        slicer.util.setSliceViewerLayers(background=imgProxyNode)

    # model sequence
    if modelSequenceNodes_dict is not None:
        for modelSequenceNode in modelSequenceNodes_dict.values(): # heart meshes
            if not (sequenceBrowserNode.IsSynchronizedSequenceNode(modelSequenceNode) or sequenceBrowserNode.GetMasterSequenceNode() == modelSequenceNode):
                sequenceBrowserNode.AddSynchronizedSequenceNode(modelSequenceNode)
        
        # need to set default display properties on proxyNodes
        # need to do it this way to make sure we get consistent coloring (zip(nodes, colors))
        proxyModelNodes = [sequenceBrowserNode.GetProxyNode(sequenceNode) for sequenceNode in modelSequenceNodes_dict.values()]
        update_model_nodes_display(proxyModelNodes)

    if segSequenceNode is not None: # ca2 seg
        if not (sequenceBrowserNode.IsSynchronizedSequenceNode(segSequenceNode) or sequenceBrowserNode.GetMasterSequenceNode() == segSequenceNode):
            sequenceBrowserNode.AddSynchronizedSequenceNode(segSequenceNode)

        # add default display properties for segProxyNode from segSequenceNode
        segProxyNode = sequenceBrowserNode.GetProxyNode(segSequenceNode)
        segmentation = segProxyNode.GetSegmentation()
        segmentNames = [segmentation.GetNthSegment(idx).GetName() for idx in range(segmentation.GetNumberOfSegments())]
        update_seg_node_display(segProxyNode, segmentNames)

    # For setting active sequence browser node to img + model sequence
    slicer.modules.sequences.toolBar().setActiveBrowserNode(sequenceBrowserNode)

    # do this to update proxy nodes (RemoveAllProxyNodes is not good b/c it ends up creating duplicates)
    sequenceBrowserNode.SelectNextItem()
    sequenceBrowserNode.SetSelectedItemNumber(currentItemNumber)

def put_models_in_folder(folderName, modelNames_dict):
    ''' we do this to allow for easy grouping of visualization '''
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()

    create_new_folder = True
    childIds = vtk.vtkIdList() # dummy to save Ids
    shNode.GetItemChildren(shNode.GetSceneItemID(), childIds)
    for itemIdIndex in range(childIds.GetNumberOfIds()): # for all children of the main Subject Hierarchy
        shItemId = childIds.GetId(itemIdIndex)
        if shNode.GetItemName(shItemId) == folderName:
            grandChildIds = vtk.vtkIdList()
            shNode.GetItemChildren(shItemId, grandChildIds)
            if grandChildIds.GetNumberOfIds() > 0:
                modelFolderItemId = shItemId
                create_new_folder = False
    if create_new_folder:
        modelFolderItemId = shNode.CreateFolderItem(shNode.GetSceneItemID(), folderName)
    
    childIds = vtk.vtkIdList() # dummy to save Ids
    shNode.GetItemChildren(shNode.GetSceneItemID(), childIds)
    for itemIdIndex in range(childIds.GetNumberOfIds()): # for all children of the main Subject Hierarchy
        shItemId = childIds.GetId(itemIdIndex)
        dataNode = shNode.GetItemDataNode(shItemId)
        if isinstance(dataNode, slicer.vtkMRMLModelNode): # check dataNode is modelNode
            if dataNode.GetName() in list(modelNames_dict.values()): # get dataNode's name is in the modelNames_dict
                shNode.SetItemParent(shItemId, modelFolderItemId)

    # folder display manipulation
    pluginHandler = slicer.qSlicerSubjectHierarchyPluginHandler().instance()
    folderPlugin = pluginHandler.pluginByName("Folder")
    folderPlugin.setDisplayVisibility(modelFolderItemId, 1)