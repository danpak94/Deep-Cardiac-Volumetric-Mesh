import slicer
import vtk

class UpdateCheckboxWithDataNodeVisibility():
    def __init__(self, checkbox):
        self.checkbox = checkbox
    def __call__(self, dataNode, event=None):
        all_parents_visibility = get_all_parents_visibility(dataNode.GetName())
        if isinstance(dataNode, slicer.vtkMRMLDisplayNode):
            displayNode = dataNode
        elif isinstance(dataNode, slicer.vtkMRMLDisplayableNode):
            displayNode = dataNode.GetDisplayNode()
        self.checkbox.checked = all(all_parents_visibility + [displayNode.GetVisibility()])

class UpdateCheckboxWithSegmentVisibility():
    def __init__(self, checkbox, segmentationNodeName_suffix, segmentName_suffix_list):
        self.checkbox = checkbox
        self.segmentationNodeName_suffix = segmentationNodeName_suffix
        self.segmentName_suffix_list = segmentName_suffix_list
    def __call__(self, segmentationNode, event=None):
        inputNodeName = segmentationNode.GetName().split(self.segmentationNodeName_suffix)[0]
        visibility_list = []
        for segmentName_suffix in self.segmentName_suffix_list:
            segmentName = '{}{}'.format(inputNodeName, segmentName_suffix)
            segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
            if segmentId == '':
                visibility_list.append(False)
            else:
                visibility_list.append(segmentationNode.GetDisplayNode().GetSegmentVisibility(segmentId))
        visibility_list.append(segmentationNode.GetDisplayNode().GetVisibility())
        all_parents_visibility = get_all_parents_visibility(segmentationNode.GetName())
        visibility_list += all_parents_visibility
        self.checkbox.checked = all(visibility_list)

def put_outputs_under_same_subject(subjectName, folderName_suffix, segmentationName_suffix):
    '''
    for each volume_node or volume_sequence_node:
    1. create a subject item with the same name
    3. place all other outputs under the same subject (same cropInputNode name prefix)
    '''
    # create new subject
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    _, subjectIds = get_all_subjects_with_matching_name(subjectName)
    if len(subjectIds) == 0:
        subjectId = shNode.CreateSubjectItem(shNode.GetSceneItemID(), subjectName)
        pluginHandler = slicer.qSlicerSubjectHierarchyPluginHandler().instance()
        folderPlugin = pluginHandler.pluginByName("Folder")
        folderPlugin.setDisplayVisibility(subjectId, True)
    else:
        subjectId = subjectIds[0]

    # traverse and relocate all models with subjectName prefix
    childIds = vtk.vtkIdList() # dummy to save Ids
    shNode.GetItemChildren(shNode.GetSceneItemID(), childIds, True) # recursively get all children in the scene
    for itemIdIndex in range(childIds.GetNumberOfIds()):
        shItemId = childIds.GetId(itemIdIndex)
        if subjectName == shNode.GetItemName(shItemId) and shNode.GetItemAttribute(shItemId, 'Level') != 'Patient': # volume
            shNode.SetItemParent(shItemId, subjectId)
        if '{}{}'.format(subjectName, folderName_suffix) in shNode.GetItemName(shItemId) and shNode.GetItemAttribute(shItemId, 'Level') == 'Folder': # model folder
            shNode.SetItemParent(shItemId, subjectId)
            shNode.SetItemExpanded(shItemId, False)
        if '{}{}'.format(subjectName, segmentationName_suffix) in shNode.GetItemName(shItemId): # segmentation
            shNode.SetItemParent(shItemId, subjectId)
            shNode.SetItemExpanded(shItemId, False)

def get_all_subjects_with_matching_name(name):
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    childIds = vtk.vtkIdList() # dummy to save Ids
    shNode.GetItemChildren(shNode.GetSceneItemID(), childIds, True) # recursively get all children in the scene
    subjectNames = []
    subjectIds = []
    for itemIdIndex in range(childIds.GetNumberOfIds()):
        shItemId = childIds.GetId(itemIdIndex)
        if name == shNode.GetItemName(shItemId) and shNode.GetItemAttribute(shItemId, 'Level') == 'Patient':
            subjectNames.append(shNode.GetItemName(shItemId))
            subjectIds.append(shItemId)
    return subjectNames, subjectIds

def get_all_folders_containing_suffix(suffix):
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    childIds = vtk.vtkIdList() # dummy to save Ids
    shNode.GetItemChildren(shNode.GetSceneItemID(), childIds, True) # recursively get all children in the scene
    folderNames = []
    foldreIds = []
    folderNodes = []
    for itemIdIndex in range(childIds.GetNumberOfIds()): # for all children of the main Subject Hierarchy
        shItemId = childIds.GetId(itemIdIndex)
        if shNode.GetItemName(shItemId).endswith(suffix) and shNode.GetItemAttribute(shItemId, 'Level') == 'Folder':
            folderNames.append(shNode.GetItemName(shItemId))
            foldreIds.append(shItemId)
            folderNodes.append(shNode.GetItemDataNode(shItemId))
    return folderNames, foldreIds, folderNodes

def get_all_segments_containing_suffix(suffix):
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    childIds = vtk.vtkIdList() # dummy to save Ids
    shNode.GetItemChildren(shNode.GetSceneItemID(), childIds, True) # recursively get all children in the scene
    segmentNames = []
    segmentIds = []
    segmentationNodes = []
    for itemIdIndex in range(childIds.GetNumberOfIds()): # for all children of the main Subject Hierarchy
        shItemId = childIds.GetId(itemIdIndex)
        if shNode.GetItemName(shItemId).endswith(suffix) and (shNode.GetItemOwnerPluginName(shItemId) == 'Segments'):
            segmentNames.append(shNode.GetItemName(shItemId))
            segmentIds.append(shItemId)
            segmentationNodes.append(shNode.GetItemDataNode(shNode.GetItemParent(shItemId)))
    return segmentNames, segmentIds, segmentationNodes

def get_all_models_containing_suffix(suffix):
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    childIds = vtk.vtkIdList() # dummy to save Ids
    shNode.GetItemChildren(shNode.GetSceneItemID(), childIds, True) # recursively get all children in the scene
    modelNames = []
    modelIds = []
    modelNodes = []
    for itemIdIndex in range(childIds.GetNumberOfIds()): # for all children of the main Subject Hierarchy
        shItemId = childIds.GetId(itemIdIndex)
        if shNode.GetItemName(shItemId).endswith(suffix) and (shNode.GetItemOwnerPluginName(shItemId) == 'Models'):
            modelNames.append(shNode.GetItemName(shItemId))
            modelIds.append(shItemId)
            modelNodes.append(shNode.GetItemDataNode(shItemId))
    return modelNames, modelIds, modelNodes

def put_models_in_folder(folderName, modelNames_dict):
    ''' we do this to allow for easy grouping of visualization '''
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()

    pluginHandler = slicer.qSlicerSubjectHierarchyPluginHandler().instance()
    folderPlugin = pluginHandler.pluginByName("Folder")

    _, folderItemIds, _ = get_all_folders_containing_suffix(folderName)
    if len(folderItemIds) == 0:
        # folderItemId = shNode.CreateFolderItem(shNode.GetSceneItemID(), folderName)
        folderItemId = folderPlugin.createFolderUnderItem(shNode.GetSceneItemID())
        shNode.SetItemName(folderItemId, folderName)
    else:
        folderItemId = folderItemIds[0]
    
    childIds = vtk.vtkIdList() # dummy to save Ids
    shNode.GetItemChildren(shNode.GetSceneItemID(), childIds)
    for itemIdIndex in range(childIds.GetNumberOfIds()): # for all children of the main Subject Hierarchy
        shItemId = childIds.GetId(itemIdIndex)
        dataNode = shNode.GetItemDataNode(shItemId)
        if isinstance(dataNode, slicer.vtkMRMLModelNode): # check dataNode is modelNode
            if dataNode.GetName() in list(modelNames_dict.values()): # get dataNode's name is in the modelNames_dict
                shNode.SetItemParent(shItemId, folderItemId)
    
    # need to do this to instantiate FolderDisplayNode (folderNode is None before this)
    folderPlugin.setDisplayVisibility(folderItemId, True)
    shNode.SetItemExpanded(folderItemId, False)

def set_folder_visibility(folderName, visibility, updateContentsVisibility=False):
    '''
    1. turn on all relevant model's visibility
    2. control model visibility with folder visibility
    3. Turn on all parents folder if they exist, only when visibility==True
    '''
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()

    folder_exists = False
    childIds = vtk.vtkIdList() # dummy to save Ids
    shNode.GetItemChildren(shNode.GetSceneItemID(), childIds, True) # for all children of the main Subject Hierarchy, recursively search
    for itemIdIndex in range(childIds.GetNumberOfIds()):
        shItemId = childIds.GetId(itemIdIndex)
        if (shNode.GetItemName(shItemId) == folderName) and (shNode.GetItemAttribute(shItemId, 'Level') == 'Folder'):
            modelFolderItemId = shItemId
            folder_exists = True
            break

    if folder_exists:
        if updateContentsVisibility:
            # turn on all data displays inside folder
            folderChildIds = vtk.vtkIdList()
            shNode.GetItemChildren(modelFolderItemId, folderChildIds)
            for dataItemIdIndex in range(folderChildIds.GetNumberOfIds()):
                dataItemId = folderChildIds.GetId(dataItemIdIndex)
                dataItemDataNode = shNode.GetItemDataNode(dataItemId)
                dataItemDataNode.GetDisplayNode().SetVisibility(True)

        # folder display manipulation
        pluginHandler = slicer.qSlicerSubjectHierarchyPluginHandler().instance()
        folderPlugin = pluginHandler.pluginByName("Folder")
        folderPlugin.setDisplayVisibility(modelFolderItemId, visibility)

        if visibility: # only turn on all parents up the hierarchy for visibility==True case
            set_all_parents_visibility(modelFolderItemId, visibility)

    return folder_exists

def set_segment_visibility(segmentName, visibility):
    '''
    1. Turn on all parent "folders" if they exist, only when visibility==True
    2. Set segment visibility
    '''
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()

    segment_exists = False
    childIds = vtk.vtkIdList() # dummy to save Ids
    shNode.GetItemChildren(shNode.GetSceneItemID(), childIds, True) # for all children of the main Subject Hierarchy, recursively search
    for itemIdIndex in range(childIds.GetNumberOfIds()):
        shItemId = childIds.GetId(itemIdIndex)
        if (shNode.GetItemName(shItemId) == segmentName) and (shNode.GetItemOwnerPluginName(shItemId) == 'Segments'):
            shSegmentId = shItemId
            segment_exists = True
            break

    if segment_exists:
        # set segment visibility
        segmentationNodeId = shNode.GetItemParent(shSegmentId)
        segmentationNode = shNode.GetItemDataNode(segmentationNodeId)
        segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segmentName)
        segmentationNode.GetDisplayNode().SetSegmentVisibility(segmentId, visibility)

        if visibility: # only turn on all parents up the hierarchy for visibility==True case
            segmentationNode.GetDisplayNode().SetVisibility(True)
            set_all_parents_visibility(segmentationNodeId, visibility)

    return segment_exists

def set_model_visibility(modelName, visibility):
    '''
    1. Turn on all parent "folders" if they exist, only when visibility==True
    2. Set segment visibility
    '''
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()

    model_exists = False
    childIds = vtk.vtkIdList() # dummy to save Ids
    shNode.GetItemChildren(shNode.GetSceneItemID(), childIds, True) # for all children of the main Subject Hierarchy, recursively search
    for itemIdIndex in range(childIds.GetNumberOfIds()):
        shItemId = childIds.GetId(itemIdIndex)
        if (shNode.GetItemName(shItemId) == modelName) and (shNode.GetItemOwnerPluginName(shItemId) == 'Models'):
            modelId = shItemId
            model_exists = True
            break

    if model_exists:
        # set segment visibility
        modelNode = shNode.GetItemDataNode(modelId)
        modelNode.GetDisplayNode().SetVisibility(visibility)

        if visibility: # only turn on all parents up the hierarchy for visibility==True case
            set_all_parents_visibility(modelId, visibility)

    return model_exists

def set_all_parents_visibility(shItemId, visibility):
    pluginHandler = slicer.qSlicerSubjectHierarchyPluginHandler().instance()
    folderPlugin = pluginHandler.pluginByName("Folder")

    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    parentIds = get_all_parents(shNode, shItemId)
    for parentId in parentIds:
        folderPlugin.setDisplayVisibility(parentId, visibility)

def get_all_parents(shNode, shItemId, parentIds=None):
    if parentIds is None:
        parentIds = []
        recursionEntry = True
    else:
        recursionEntry = False

    parentId = shNode.GetItemParent(shItemId)
    if (parentId != 0) and (not parentId == shNode.GetSceneItemID()): # default invalidId or sceneId
        parentIds.append(parentId)
        get_all_parents(shNode, parentId, parentIds)
    if recursionEntry:
        return parentIds
    
def get_children(folderName, recurse=False):
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    folderId = shNode.GetItemByName(folderName)

    childIds = vtk.vtkIdList() # dummy to save Ids
    shNode.GetItemChildren(folderId, childIds, recurse)
    childrenNames = []
    childrenIds = []
    childrenDataNodes = []
    for itemIdIndex in range(childIds.GetNumberOfIds()):
        shItemId = childIds.GetId(itemIdIndex)
        childrenNames.append(shNode.GetItemName(shItemId))
        childrenIds.append(shItemId)
        childrenDataNodes.append(shNode.GetItemDataNode(shItemId))
    return childrenNames, childrenIds, childrenDataNodes

def get_all_parents_visibility(itemName):
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    itemId = shNode.GetItemByName(itemName)
    parentIds = get_all_parents(shNode, itemId)
    visibility_list = [shNode.GetItemDataNode(parentId).GetVisibility() for parentId in parentIds]
    return visibility_list