import matplotlib
colors_models = matplotlib.colormaps['Set1'].colors
colors_segments = matplotlib.colormaps['Set2'].colors

import slicer
import pyvista as pv

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

def update_model_nodes_display(modelNodes, colors=colors_models):
    for modelNode, color in zip(modelNodes, colors):
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