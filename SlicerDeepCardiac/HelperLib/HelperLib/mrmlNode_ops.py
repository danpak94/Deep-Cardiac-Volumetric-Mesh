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

import matplotlib
colors_models = matplotlib.colormaps['Set1'].colors
colors_segments = matplotlib.colormaps['Set2'].colors[0::2]
colors_segmentPosts = matplotlib.colormaps['Set2'].colors[1::2]

import numpy as np
import torch
import slicer
import pyvista as pv
import HelperLib.HelperLib.vtk_ops as vtk_ops
import dcvm
import vtk

def update_model_nodes_from_pv_dict(mesh_pv_dict, modelNames_dict):
    for key, modelName in modelNames_dict.items():
        modelNode = slicer.util.getFirstNodeByClassByName('vtkMRMLModelNode', modelName) # None if modelNode with modelName doesn't exist
        if modelNode is None:
            modelNode = slicer.modules.models.logic().AddModel(pv.PolyData())
            modelNode.SetName(modelNames_dict[key])
            modelNode.SetAndObserveMesh(mesh_pv_dict[key])
        else:
            modelNode.SetAndObserveMesh(mesh_pv_dict[key])
    modelNodes_dict = {key: slicer.util.getFirstNodeByClassByName('vtkMRMLModelNode', modelName) for key, modelName in modelNames_dict.items()}

    return modelNodes_dict

def update_model_nodes_from_pv(mesh_pv_dict):
    '''
    mesh_pv_dict: {modelNodeName: pv.Polydata or pv.UnstructuredGrid}
    '''
    for modelName, mesh_pv in mesh_pv_dict.items():
        modelNode = slicer.util.getFirstNodeByClassByName('vtkMRMLModelNode', modelName) # None if modelNode with modelName doesn't exist
        if modelNode is None:
            modelNode = slicer.modules.models.logic().AddModel(pv.PolyData())
            modelNode.SetName(modelName)
            modelNode.SetAndObserveMesh(mesh_pv)
        else:
            modelNode.SetAndObserveMesh(mesh_pv)
    modelNodes_dict = {modelName: slicer.util.getFirstNodeByClassByName('vtkMRMLModelNode', modelName) for modelName in mesh_pv_dict.keys()}

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

def update_model_nodes_display(modelNodes, colors=colors_models, opacity2d=0.3):
    for modelNode, color in zip(modelNodes, colors):
        if modelNode is not None:
            modelNode.GetDisplayNode().SetColor(*color)
            modelNode.GetDisplayNode().SetEdgeVisibility(True)
            modelNode.GetDisplayNode().SetVisibility2D(True)
            modelNode.GetDisplayNode().SetSliceIntersectionOpacity(opacity2d)
            modelNode.GetDisplayNode().SetSliceIntersectionThickness(5)
            # modelNode.GetDisplayNode().SetVisibility(True)

def update_table_node_from_fiber_ori_dict(fiber_ori_dict):
    '''
    fiber_ori_dict: {tableName: np.ndarray}
    '''
    for tableName, fiber_ori in fiber_ori_dict.items():
        tableNode = slicer.util.getFirstNodeByClassByName('vtkMRMLTableNode', tableName) # None if modelNode with modelName doesn't exist
        if tableNode is None:
            tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
            tableNode.SetName(tableName)
        slicer.util.updateTableFromArray(tableNode, fiber_ori, columnNames=['x1', 'y1', 'z1', 'x2', 'y2', 'z2'])
    tableNodes_dict = {tableName: slicer.util.getFirstNodeByClassByName('vtkMRMLModelNode', tableName) for tableName in fiber_ori_dict.keys()}

    return tableNodes_dict

# def apply_oversampling_on_segmentationNode(segmentationNode, inputVolumeNode, oversampling_factor):
#     segmentationGeometryLogic = slicer.vtkSlicerSegmentationGeometryLogic()
#     segmentationGeometryLogic.SetInputSegmentationNode(segmentationNode)
#     segmentationGeometryLogic.SetSourceGeometryNode(inputVolumeNode)
#     segmentationGeometryLogic.SetOversamplingFactor(oversampling_factor)
#     segmentationGeometryLogic.CalculateOutputGeometry()
#     geometryImageData = segmentationGeometryLogic.GetOutputGeometryImageData()

#     # these three lines update the "Segmentation labelmap geometry" *display info* in 3D slicer GUI
#     geometryString = slicer.vtkSegmentationConverter.SerializeImageGeometry(geometryImageData)
#     segmentationNode.GetSegmentation().SetConversionParameter(slicer.vtkSegmentationConverter.GetReferenceImageGeometryParameterName(), geometryString)

#     # # this actually performs the resampling
#     # segmentationGeometryLogic.ResampleLabelmapsInSegmentationNode()

#     return geometryImageData

def get_oversampled_geometryImageData(segmentationNode, inputVolumeNode, oversampling_factor, apply_resample=False):
    segmentationGeometryLogic = slicer.vtkSlicerSegmentationGeometryLogic()
    segmentationGeometryLogic.SetInputSegmentationNode(segmentationNode)
    segmentationGeometryLogic.SetSourceGeometryNode(inputVolumeNode)
    if oversampling_factor != 1:
        segmentationGeometryLogic.SetOversamplingFactor(oversampling_factor)
    segmentationGeometryLogic.CalculateOutputGeometry()
    geometryImageData = segmentationGeometryLogic.GetOutputGeometryImageData()

    if apply_resample:
        # this actually performs the resampling
        segmentationGeometryLogic.ResampleLabelmapsInSegmentationNode()

    segmentationNode.Modified()

    return geometryImageData

def get_oversampled_seg_np_torch(segmentationNode, geometryImageData):
    '''
    assume segmentationNode is already in 1x (dims, spacing, origin) of original inputVolumeNode
    '''
    target_shape = geometryImageData.GetDimensions()
    spacing = geometryImageData.GetSpacing()
    seg_list = []
    for idx in range(segmentationNode.GetSegmentation().GetNumberOfSegments()):
        segmentId = segmentationNode.GetSegmentation().GetNthSegmentID(idx)
        seg = slicer.util.arrayFromSegmentInternalBinaryLabelmap(segmentationNode, segmentId)
        seg_list.append(seg.transpose(2,1,0))
    seg_orig = np.stack(seg_list, axis=0)
    seg_orig_torch = torch.tensor(seg_orig, dtype=torch.float, device='cuda')[None] # (1, c, h, w, d)
    transformation = torch.tensor([
        [1/spacing[0],0,0,0],
        [0,1/spacing[1],0,0],
        [0,0,1/spacing[2],0],
        [0,0,0,1],
    ], device='cuda')
    seg_resampled = dcvm.transforms.apply_linear_transform_on_img_torch(seg_orig_torch, transformation, target_shape)>0.5
    return seg_resampled

def apply_oversampling_on_segmentationNode(segmentationNode, inputVolumeNode, oversampling_factor):
    '''
    need to implement this using non-slicer operations if we're going to include it as part of algorithm
    https://discourse.slicer.org/t/change-segmentation-oversampling-factor/19025/5
    https://discourse.slicer.org/t/programatically-use-specify-geometry-python/18941
    vtkOrientedImageDataResample
    '''
    # save --> reload results in inconsistent seg cropping.. This returns segmentation geometry to the original input volume geometry.
    # geometryImageData_orig = get_oversampled_geometryImageData(segmentationNode, inputVolumeNode, oversampling_factor=1, apply_resample=True)
    geometryImageData = get_oversampled_geometryImageData(segmentationNode, inputVolumeNode, oversampling_factor=oversampling_factor, apply_resample=True)

    # if oversampling_factor != 1:
    #     # do the actual oversampling here
    #     geometryImageData_target = get_oversampled_geometryImageData(segmentationNode, inputVolumeNode, oversampling_factor=oversampling_factor, apply_resample=False)
    #     geometryImageData_target.SetOrigin([0,0,0])

        # oversampled_seg_np_all = get_oversampled_seg_np_torch(segmentationNode, geometryImageData_target)
        # oversampled_seg_np_all = oversampled_seg_np_all.squeeze(0).cpu().numpy()
        # for idx, oversampled_seg_np_each in enumerate(oversampled_seg_np_all):
        #     segment = segmentationNode.GetSegmentation().GetNthSegment(idx)
        #     update_seg_node_from_np(oversampled_seg_np_each, segmentationNode.GetName(), segment.GetName(), geometryImageData_target)

        # segmentName = 'test'
        # segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
        # segmentationNode.RemoveSegment(segmentId)
        # segmentationNode.AddSegmentFromBinaryLabelmapRepresentation(vtk_ops.get_vtkImageData_from_np(segmentArray, geometryImageData), segmentName)

    return geometryImageData

def update_seg_node_from_np(segmentArray, segmentationNodeName, segmentName, geometry_defining_obj):
    '''
    geometry_defining_obj: slicer.vtkMRMLVolumeNode or vtk.vtkImageData (faster if volumeNode doesn't exist already)
    '''
    segmentationNodes = [node for node in slicer.util.getNodesByClass('vtkMRMLSegmentationNode') if segmentationNodeName in node.GetName()]
    if len(segmentationNodes) == 0:
        # create new Segmentation node
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.SetName(segmentationNodeName)
    else:
        # update existing Segmentation node
        segmentationNode = segmentationNodes[0]
    segmentationNode.CreateDefaultDisplayNodes()

    if isinstance(geometry_defining_obj, slicer.vtkMRMLVolumeNode):
        inputVolumeNode = geometry_defining_obj
        if segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName) == '': # create new Segment only if segment_name doesn't exist in Segmentation already
            segmentationNode.GetSegmentation().AddEmptySegment(segmentName)
        segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
        slicer.util.updateSegmentBinaryLabelmapFromArray(segmentArray.transpose([2,1,0]), segmentationNode, segmentId, referenceVolumeNode=inputVolumeNode)
        segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(inputVolumeNode)
    # elif isinstance(geometry_defining_obj, vtk.vtkImageData):
    #     geometryImageData = geometry_defining_obj
    #     segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
    #     if segmentId:
    #         segmentationNode.RemoveSegment(segmentId)

    #     labelmap_vtkImageData = vtk_ops.get_vtkImageData_from_np(segmentArray, geometryImageData)
    #     slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmap_vtkImageData, segmentationNode, segmentName) # this has an annoying clip to extent built-in. Do a custom add

    #     # # converting labelmap from clipped extents to the original geometry..
    #     # segmentIDs = vtk.vtkStringArray()
    #     # segmentationNode.GetSegmentation().GetSegmentIDs(segmentIDs)
    #     # for index in range(segmentIDs.GetNumberOfValues()):
    #     #     currentSegmentID = segmentIDs.GetValue(index)
    #     #     currentSegment = segmentationNode.GetSegmentation().GetSegment(currentSegmentID)
    #     #     if segmentName in currentSegment.GetName():
    #     #         slicer.vtkOrientedImageDataResample.ResampleOrientedImageToReferenceOrientedImage()

    #     # setting it to the correct name
    #     segmentIDs = vtk.vtkStringArray()
    #     segmentationNode.GetSegmentation().GetSegmentIDs(segmentIDs)
    #     for index in range(segmentIDs.GetNumberOfValues()):
    #         currentSegmentID = segmentIDs.GetValue(index)
    #         currentSegment = segmentationNode.GetSegmentation().GetSegment(currentSegmentID)
    #         if segmentName in currentSegment.GetName():
    #             currentSegment.SetName(segmentName)
    #             break
        
        # segmentationNode.AddSegmentFromBinaryLabelmapRepresentation(vtkImageData, '{}'.format(segmentName)) # this seems ok, but it actually messes up geometry in a weird way.. I'm just not using it for now
    
    segmentationNode.Modified()

    return segmentationNode
    
def update_seg_node_display(segmentationNode, segmentNames, colors=colors_segments):
    # segment color
    for segmentName, color in zip(segmentNames, colors):
        segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
        if segmentId != '':
            segmentNode = segmentationNode.GetSegmentation().GetSegment(segmentId)
            segmentNode.SetColor(color)
            segmentationNode.GetDisplayNode().SetSegmentVisibility(segmentId, True)

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

def update_outputSequenceBrowserNode(outputSequenceBrowserNodeName, imgSequenceNode=None, modelSequenceNodes_dict=None, segSequenceNode=None):
    # delete existing browser nodes with 1. not desired browserName and 2. cropInputNode in it already
    existing_browser_nodes = []
    for browserNode in slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode'):
        if (browserNode.GetName() != outputSequenceBrowserNodeName) and (browserNode.IsSynchronizedSequenceNode(imgSequenceNode) or browserNode.GetMasterSequenceNode() == imgSequenceNode):
            slicer.mrmlScene.RemoveNode(browserNode)
            proxyVolumeNode = [node for node in slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode') if node.GetName() == imgSequenceNode.GetName()][0]
            slicer.mrmlScene.RemoveNode(proxyVolumeNode)

    # init browser node if it doesn't exist
    if not outputSequenceBrowserNodeName in [node.GetName() for node in slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode')]:
        sequenceBrowserNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", outputSequenceBrowserNodeName)
        sequenceBrowserNode.SetPlaybackRateFps(2.0) # slower b/c synced with model
        currentItemNumber = 0
    else: # grab browser node and delete proxy nodes to refresh display
        sequenceBrowserNode = [node for node in slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode') if node.GetName() == outputSequenceBrowserNodeName][0]
        currentItemNumber = sequenceBrowserNode.GetSelectedItemNumber()

    # img sequence
    if imgSequenceNode is not None: # original image
        if not (sequenceBrowserNode.IsSynchronizedSequenceNode(imgSequenceNode) or sequenceBrowserNode.GetMasterSequenceNode() == imgSequenceNode):
            sequenceBrowserNode.AddSynchronizedSequenceNode(imgSequenceNode)
        imgProxyNode = sequenceBrowserNode.GetProxyNode(imgSequenceNode)
        slicer.util.setSliceViewerLayers(background=imgProxyNode)

    # model sequence
    if modelSequenceNodes_dict is not None:
        # first, remove any existing model nodes (probably generated from proxy)
        modelNames = [node.GetName() for node in modelSequenceNodes_dict.values()]
        for modelNode in slicer.util.getNodesByClass('vtkMRMLModelNode'):
            if modelNode.GetName() in modelNames:
                slicer.mrmlScene.RemoveNode(modelNode)

        for modelSequenceNode in modelSequenceNodes_dict.values(): # heart meshes
            if not (sequenceBrowserNode.IsSynchronizedSequenceNode(modelSequenceNode) or sequenceBrowserNode.GetMasterSequenceNode() == modelSequenceNode):
                sequenceBrowserNode.AddSynchronizedSequenceNode(modelSequenceNode)
        
        # need to set default display properties on proxyNodes
        # need to do it this way to make sure we get consistent coloring (zip(nodes, colors))
        proxyModelNodes = [sequenceBrowserNode.GetProxyNode(sequenceNode) for sequenceNode in modelSequenceNodes_dict.values()]
        update_model_nodes_display(proxyModelNodes)

    if segSequenceNode is not None: # ca2 seg
        # first, remove any existing segmentation nodes (probably generated from proxy)
        for segmentationNode in slicer.util.getNodesByClass('vtkMRMLSegmentationNode'):
            if segmentationNode.GetName() in segSequenceNode.GetName():
                slicer.mrmlScene.RemoveNode(segmentationNode)

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

    return sequenceBrowserNode

def get_segment_np(segmentationNode, segmentName, referenceVolumeNode=None):
    segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
    segmentArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, referenceVolumeNode=referenceVolumeNode)
    # segmentArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId)
    seg_np = segmentArray.transpose([2,1,0]) # get this from slicer thresholded segment

    # segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
    # segmentArray = slicer.util.arrayFromSegmentInternalBinaryLabelmap(segmentationNode, segmentId)
    # seg_np = np.transpose(segmentArray, [2,1,0]) # same shape as geometryImageData.GetDimensions()
    return seg_np