import logging
import os
import sys
import time

import vtk
import qt
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import RoiPredSetup.RoiPredSetup
RoiPredSetup.RoiPredSetup.install_missing_pkgs_in_slicer()

import RoiPredLib.RoiPredLib as roi_pred_lib
import numpy as np
import torch
import pyvista as pv
import matplotlib
colors = matplotlib.colormaps['Set1'].colors

curr_file_dir_path = os.path.dirname(os.path.realpath(__file__))
dcvm_parent_dir = os.path.join(curr_file_dir_path, '../..')
if dcvm_parent_dir not in sys.path:
    sys.path.append(dcvm_parent_dir)
import dcvm

#
# RoiPred
#

class RoiPred(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "ROI pred"  # TODO: make this more human readable by adding spaces
        self.parent.categories = ["SlicerDeepCardiac"]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = ['CropVolume']  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Daniel Pak (Yale)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """get ROI image --> pytorch prediction --> save output to 3D Slicer model"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """NIH R01 and F31"""

        # Additional initialization step after application startup is complete
        # slicer.app.connect("startupCompleted()", registerSampleData)

#
# RoiPredWidget
#

class RoiPredWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cropOutputNodeName = "RoiPred_crop_output"
        self.roiNodeName = "RoiPred_ROI"
        self.outputSequenceBrowserNodeName = "RoiPred_model_sequence_browser"

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/RoiPred.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = RoiPredLogic()

        # Connections

        # enter() and exit() should just work without adding observers -- https://discourse.slicer.org/t/scripted-module-leak-addobserver/121/4

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.inputSequenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputSequenceSelector.setVisible(False) # need this to make sure we only see inputVolumeSelector by default
        self.cropInputSelector = self.ui.inputVolumeSelector
        self.ui.inputSequenceOrVolume.toggled.connect(self.updateParameterNodeFromGUI)

        self.ui.pytorchInputSpatialDim0.textChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.pytorchInputSpatialDim1.textChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.pytorchInputSpatialDim2.textChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.spacing.textChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.roiR.textChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.roiA.textChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.roiS.textChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.roiVisibility.toggled.connect(self.updateParameterNodeFromGUI)
        self.ui.useGPU.toggled.connect(self.updateParameterNodeFromGUI)
        self.ui.heartVisibility.toggled.connect(self.updateParameterNodeFromGUI)
        self.ui.ca2Visibility.toggled.connect(self.updateParameterNodeFromGUI)
        self.ui.templateFilenamePrefix.textChanged.connect(self.updateParameterNodeFromGUI)

        # self.ui.pytorchOutputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

        # Buttons
        self.ui.inputSequenceOrVolume.clicked.connect(self.onInputSequenceOrVolume)
        self.ui.initializeAndUpdateRoiNodeButton.clicked.connect(self.onInitializeAndUpdateRoiNode)
        self.ui.deleteRoiNodeButton.clicked.connect(self.onDeleteRoiNode)
        self.ui.heartModelLoadButton.clicked.connect(self.onHeartModelLoadButton)
        self.ui.templateLoadButton.clicked.connect(self.onTemplateLoadButton)
        self.ui.ca2ModelLoadButton.clicked.connect(self.onCa2ModelLoadButton)
        self.ui.crosshairPosButton.clicked.connect(self.onCrosshairPosButton)
        self.ui.modelPredCenterButton.clicked.connect(self.onModelPredCenterButton)
        self.ui.roiVisibility.clicked.connect(self.onRoiVisibilityButton)
        self.ui.downloadDataButton.clicked.connect(self.onDownloadDataButton)
        self.ui.heartExpDir.directoryChanged.connect(self.updateHeartExpDir)
        self.ui.ca2ExpDir.directoryChanged.connect(self.updateCa2ExpDir)
        self.ui.useGPU.clicked.connect(self.onUseGpuButton)
        self.ui.cropAndRunButton.clicked.connect(self.onCropAndRunButton)
        self.ui.heartVisibility.clicked.connect(self.onHeartVisibilityButton)
        self.ui.ca2Visibility.clicked.connect(self.onCa2VisibilityButton)
        self.ui.saveOutputsInpButton.clicked.connect(self.onSaveOutputsInpButton)
        self.ui.removeOutputNodesButton.clicked.connect(self.onRemoveOutputNodesButton)

        self.crosshair = slicer.util.getNode('Crosshair')

        moduleDir = os.path.dirname(slicer.util.modulePath(self.__module__))
        self.ui.roiVisibility.setIcon(qt.QIcon(os.path.join(moduleDir, 'Resources/Icons/VisibleOn.png')))
        self.ui.heartVisibility.setIcon(qt.QIcon(os.path.join(moduleDir, 'Resources/Icons/VisibleOn.png')))
        self.ui.ca2Visibility.setIcon(qt.QIcon(os.path.join(moduleDir, 'Resources/Icons/VisibleOn.png')))

        self.ui.resetCollapsibleButton.checked = False
        self.ui.saveCollapsibleButton.checked = False
        self.ui.volumeInfoCollapsibleButton.checked = False

        self.progress_bar_and_run_time = roi_pred_lib.ProgressBarAndRunTime(self.ui.progressBar)
        
        # Make sure parameter node is initialized (needed for module reload)
        # self.parameterNodeObserved = False
        self.initializeParameterNode()

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def enter(self):
        """
        Called each time the user opens this module.
        If we update this, need to restart 3D slicer to see the effects
        """
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        If we update this, need to restart 3D slicer to see the effects
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.GetNodeReference("cropInput"):
            if self._parameterNode.GetParameter("inputSequenceOrVolume") == "true": # input is Sequence
                firstSequenceNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSequenceNode")
                if firstSequenceNode:
                    self._parameterNode.SetNodeReferenceID("cropInput", firstSequenceNode.GetID())
            elif self._parameterNode.GetParameter("inputSequenceOrVolume") == "false": # input is Volume
                firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
                if firstVolumeNode:
                    self._parameterNode.SetNodeReferenceID("cropInput", firstVolumeNode.GetID())


    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        if inputParameterNode:
            self.logic.setDefaultParameters(inputParameterNode)

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        If we update this, need to restart 3D slicer to be not buggy with the changes
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # Update node selectors and sliders
        self.cropInputSelector.setCurrentNode(self._parameterNode.GetNodeReference("cropInput"))

        self.ui.pytorchInputSpatialDim0.text = self._parameterNode.GetParameter("pytorchInputSpatialDim0")
        self.ui.pytorchInputSpatialDim1.text = self._parameterNode.GetParameter("pytorchInputSpatialDim1")
        self.ui.pytorchInputSpatialDim2.text = self._parameterNode.GetParameter("pytorchInputSpatialDim2")
        self.ui.spacing.text = self._parameterNode.GetParameter("spacing")
        self.ui.roiR.text = self._parameterNode.GetParameter("roiR")
        self.ui.roiA.text = self._parameterNode.GetParameter("roiA")
        self.ui.roiS.text = self._parameterNode.GetParameter("roiS")

        self.ui.heartExpDir.directory = self._parameterNode.GetParameter("heartExpDir")
        self.ui.ca2ExpDir.directory = self._parameterNode.GetParameter("ca2ExpDir")
        self.ui.templateFilenamePrefix.text = self._parameterNode.GetParameter("templateFilenamePrefix")

        self.ui.originalVolumeSpacingDisplay0.text = self._parameterNode.GetParameter("originalVolumeSpacingDisplay0")
        self.ui.originalVolumeSpacingDisplay1.text = self._parameterNode.GetParameter("originalVolumeSpacingDisplay1")
        self.ui.originalVolumeSpacingDisplay2.text = self._parameterNode.GetParameter("originalVolumeSpacingDisplay2")
        self.ui.originalVolumeDimensionsDisplay0.text = self._parameterNode.GetParameter("originalVolumeDimensionsDisplay0")
        self.ui.originalVolumeDimensionsDisplay1.text = self._parameterNode.GetParameter("originalVolumeDimensionsDisplay1")
        self.ui.originalVolumeDimensionsDisplay2.text = self._parameterNode.GetParameter("originalVolumeDimensionsDisplay2")
        self.ui.croppedVolumeSpacingDisplay0.text = self._parameterNode.GetParameter("croppedVolumeSpacingDisplay0")
        self.ui.croppedVolumeSpacingDisplay1.text = self._parameterNode.GetParameter("croppedVolumeSpacingDisplay1")
        self.ui.croppedVolumeSpacingDisplay2.text = self._parameterNode.GetParameter("croppedVolumeSpacingDisplay2")
        self.ui.croppedVolumeDimensionsDisplay0.text = self._parameterNode.GetParameter("croppedVolumeDimensionsDisplay0")
        self.ui.croppedVolumeDimensionsDisplay1.text = self._parameterNode.GetParameter("croppedVolumeDimensionsDisplay1")
        self.ui.croppedVolumeDimensionsDisplay2.text = self._parameterNode.GetParameter("croppedVolumeDimensionsDisplay2")
        
        self.ui.inputSequenceOrVolume.checked = True if self._parameterNode.GetParameter("inputSequenceOrVolume") == "true" else False
        self.ui.roiVisibility.checked = True if self._parameterNode.GetParameter("roiVisibility") == "true" else False
        self.ui.useGPU.checked = True if self._parameterNode.GetParameter("useGPU") == "true" else False
        self.ui.heartVisibility.checked = True if self._parameterNode.GetParameter("heartVisibility") == "true" else False
        self.ui.ca2Visibility.checked = True if self._parameterNode.GetParameter("ca2Visibility") == "true" else False
        self.ui.heartModelLoadedCheck.checked = True if self._parameterNode.GetParameter("heartModelLoaded") == "true" else False
        self.ui.templateLoadedCheck.checked = True if self._parameterNode.GetParameter("templateLoaded") == "true" else False
        self.ui.ca2ModelLoadedCheck.checked = True if self._parameterNode.GetParameter("ca2ModelLoaded") == "true" else False
        
        ''' conditions for enabling/disabling buttons '''
        if self._parameterNode.GetNodeReference("roiNode") is None:
            self.ui.initializeAndUpdateRoiNodeButton.text = "Initialize RoiPred_ROI node"
        else:
            self.ui.initializeAndUpdateRoiNodeButton.text = "Update RoiPred_ROI node"

        if self._parameterNode.GetParameter("heartRunCompleted") == "true":
            self.ui.modelPredCenterButton.enabled = True
        else:
            self.ui.modelPredCenterButton.enabled = False

        if self._parameterNode.GetNodeReference("cropInput") and \
                self._parameterNode.GetNodeReference("roiNode") and \
                ((self._parameterNode.GetParameter("heartModelLoaded") == "true" and self._parameterNode.GetParameter("templateLoaded") == "true") or \
                  self._parameterNode.GetParameter("ca2ModelLoaded") == "true"):
            '''
            1. original volume selected
            2. RoiPred_ROI initialized
            3. heart model ready to run OR ca2 model ready to run
            '''
            self.ui.cropAndRunButton.enabled = True
        else:
            self.ui.cropAndRunButton.enabled = False

        self.ui.roiVisibility.enabled = True if self._parameterNode.GetNodeReference("roiNode") else False
        self.ui.heartVisibility.enabled = True if self._parameterNode.GetParameter("heartRunCompleted") == "true" else False
        self.ui.ca2Visibility.enabled = True if self._parameterNode.GetParameter("ca2RunCompleted") == "true" else False

        # if self.ui.followExpParams.checked: # this is not good enough.. this updateGUIfromParameterNode isn't called all the time
        #     self.ui.templateFilenamePrefix.enabled = True
        # else:
        #     self.ui.templateFilenamePrefix.enabled = False

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch
        
        if not self.ui.inputSequenceOrVolume.checked:
            self.cropInputSelector = self.ui.inputVolumeSelector
        else:
            self.cropInputSelector = self.ui.inputSequenceSelector
        self._parameterNode.SetNodeReferenceID("cropInput", self.cropInputSelector.currentNodeID)

        self.updateVolumeInfo()
        self._parameterNode.SetParameter("inputSequenceOrVolume", "true" if self.ui.inputSequenceOrVolume.checked else "false")
        self._parameterNode.SetParameter("pytorchInputSpatialDim0", self.ui.pytorchInputSpatialDim0.text)
        self._parameterNode.SetParameter("pytorchInputSpatialDim1", self.ui.pytorchInputSpatialDim1.text)
        self._parameterNode.SetParameter("pytorchInputSpatialDim2", self.ui.pytorchInputSpatialDim2.text)
        self._parameterNode.SetParameter("spacing", self.ui.spacing.text)
        self._parameterNode.SetParameter("roiR", self.ui.roiR.text)
        self._parameterNode.SetParameter("roiA", self.ui.roiA.text)
        self._parameterNode.SetParameter("roiS", self.ui.roiS.text)
        self._parameterNode.SetParameter("roiVisibility", "true" if self.ui.roiVisibility.checked else "false")
        self._parameterNode.SetParameter("useGPU", "true" if self.ui.useGPU.checked else "false")

        self._parameterNode.SetParameter("heartExpDir", self.ui.heartExpDir.directory)
        self._parameterNode.SetParameter("ca2ExpDir", self.ui.ca2ExpDir.directory)
        self._parameterNode.SetParameter("templateFilenamePrefix", self.ui.templateFilenamePrefix.text)

        self._parameterNode.EndModify(wasModified)

    def updateVolumeInfo(self):
        if self.cropInputSelector.currentNode():
            if isinstance(self.cropInputSelector.currentNode(), slicer.vtkMRMLSequenceNode):
                inputVolumeNode = self.cropInputSelector.currentNode().GetNthDataNode(0)
            elif isinstance(self.cropInputSelector.currentNode(), slicer.vtkMRMLScalarVolumeNode):
                inputVolumeNode = self.cropInputSelector.currentNode()
            spacing = inputVolumeNode.GetSpacing()
            if not inputVolumeNode.GetImageData() is None:
                dimensions = inputVolumeNode.GetImageData().GetDimensions()
            else:
                dimensions = [0,0,0]
            self.ui.originalVolumeSpacingDisplay0.text = str(spacing[0])
            self.ui.originalVolumeSpacingDisplay1.text = str(spacing[1])
            self.ui.originalVolumeSpacingDisplay2.text = str(spacing[2])
            self.ui.originalVolumeDimensionsDisplay0.text = str(dimensions[0])
            self.ui.originalVolumeDimensionsDisplay1.text = str(dimensions[1])
            self.ui.originalVolumeDimensionsDisplay2.text = str(dimensions[2])
        else:
            self.ui.originalVolumeSpacingDisplay0.text = "0"
            self.ui.originalVolumeSpacingDisplay1.text = "0"
            self.ui.originalVolumeSpacingDisplay2.text = "0"
            self.ui.originalVolumeDimensionsDisplay0.text = "0"
            self.ui.originalVolumeDimensionsDisplay1.text = "0"
            self.ui.originalVolumeDimensionsDisplay2.text = "0"
        
        if self._parameterNode.GetNodeReference("cropOutput"): # None by default, no way to choose. It's a temporary node specific to RoiPred
            if isinstance(self._parameterNode.GetNodeReference("cropOutput"), slicer.vtkMRMLSequenceNode):
                outputVolumeNode = self._parameterNode.GetNodeReference("cropOutput").GetNthDataNode(0)
            elif isinstance(self._parameterNode.GetNodeReference("cropOutput"), slicer.vtkMRMLScalarVolumeNode):
                outputVolumeNode = self._parameterNode.GetNodeReference("cropOutput")
            spacing = outputVolumeNode.GetSpacing()
            if not outputVolumeNode.GetImageData() is None:
                dimensions = outputVolumeNode.GetImageData().GetDimensions()
            else:
                dimensions = [0,0,0]
            self.ui.croppedVolumeSpacingDisplay0.text = str(spacing[0])
            self.ui.croppedVolumeSpacingDisplay1.text = str(spacing[1])
            self.ui.croppedVolumeSpacingDisplay2.text = str(spacing[2])
            self.ui.croppedVolumeDimensionsDisplay0.text = str(dimensions[0])
            self.ui.croppedVolumeDimensionsDisplay1.text = str(dimensions[1])
            self.ui.croppedVolumeDimensionsDisplay2.text = str(dimensions[2])
        else:
            self.ui.croppedVolumeSpacingDisplay0.text = "0"
            self.ui.croppedVolumeSpacingDisplay1.text = "0"
            self.ui.croppedVolumeSpacingDisplay2.text = "0"
            self.ui.croppedVolumeDimensionsDisplay0.text = "0"
            self.ui.croppedVolumeDimensionsDisplay1.text = "0"
            self.ui.croppedVolumeDimensionsDisplay2.text = "0"

    def onInputSequenceOrVolume(self, caller=None, event=None):
        self.initializeParameterNode()

    def onInitializeAndUpdateRoiNode(self, caller=None, event=None):
        roiNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLMarkupsROINode")

        spacing = float(self.ui.spacing.text)
        spatialDims = np.array([float(self.ui.pytorchInputSpatialDim0.text), float(self.ui.pytorchInputSpatialDim1.text), float(self.ui.pytorchInputSpatialDim2.text)])
        roiWindowSize = spatialDims*np.array([spacing, spacing, spacing]) # roi size defined in RAS coordinates, so it's agnostic to inputVolume spacing
        if self._parameterNode.GetNodeReference("roiNode") is None:

            roiNode = slicer.mrmlScene.CreateNodeByClass("vtkMRMLMarkupsROINode")
            slicer.mrmlScene.AddNode(roiNode)
            roiNode.SetName(self.roiNodeName)
            roiNode.GetDisplayNode().SetScaleHandleVisibility(False)
            roiNode.GetDisplayNode().SetFillVisibility(False)
            roiNode.GetDisplayNode().SetVisibility(True)
            roiNode.SetSize(roiWindowSize.tolist())
            roiNode.SetCenter([float(self.ui.roiR.text), float(self.ui.roiA.text), float(self.ui.roiS.text)])
            self.ui.roiVisibility.checked = True
            self.addObserver(roiNode, roiNode.PointModifiedEvent, self.updateRoiCenterGui)
            self._parameterNode.SetNodeReferenceID("roiNode", roiNode.GetID())
        else:
            roiNode = self._parameterNode.GetNodeReference("roiNode")
            roiNode.SetSize(roiWindowSize.tolist())
            roiNode.SetCenter([float(self.ui.roiR.text), float(self.ui.roiA.text), float(self.ui.roiS.text)])

    def onDeleteRoiNode(self, caller=None, event=None):
        # # delete roiNode in this instance
        # roiNode = self._parameterNode.GetNodeReference("roiNode")
        # if roiNode:
        #     self.removeObserver(roiNode, roiNode.PointModifiedEvent, self.updateRoiCenterGui)
        #     slicer.mrmlScene.RemoveNode(roiNode)

        # delete all roiNode's with the default name
        for roiNode in slicer.util.getNodesByClass('vtkMRMLMarkupsROINode'):
            if roiNode.GetName() == self.roiNodeName:
                self.removeObserver(roiNode, roiNode.PointModifiedEvent, self.updateRoiCenterGui)
                slicer.mrmlScene.RemoveNode(roiNode)

    def updateRoiCenterGui(self, observer, eventid):
        roiCenter = observer.GetCenter()
        self.ui.roiR.text = round(roiCenter[0], 2)
        self.ui.roiA.text = round(roiCenter[1], 2)
        self.ui.roiS.text = round(roiCenter[2], 2)

    def onCrosshairPosButton(self):
        roiRAS = self.crosshair.GetCrosshairRAS()
        self.ui.roiR.text = round(roiRAS[0], 2)
        self.ui.roiA.text = round(roiRAS[1], 2)
        self.ui.roiS.text = round(roiRAS[2], 2)
        if self._parameterNode.GetNodeReference("roiNode"):
            self._parameterNode.GetNodeReference("roiNode").SetCenter([float(self.ui.roiR.text), float(self.ui.roiA.text), float(self.ui.roiS.text)])

    def onModelPredCenterButton(self):
        inputNodeName = self.cropInputSelector.currentNode().GetName()
        modelNames_dict = {key: '{}_{}'.format(inputNodeName, key) for key in self.heart_mesh_keys}

        mesh_pv_list = [pv.UnstructuredGrid(modelNode.GetMesh()) for modelNode in slicer.util.getNodesByClass('vtkMRMLModelNode') if modelNode.GetName() in modelNames_dict.values()] # pv.UnstructuredGrid to make proxyNode compatible for pv.merge
        mesh_pv_all = pv.merge(mesh_pv_list)
        modelPredCenter = np.array(mesh_pv_all.bounds).reshape(-1,2).mean(axis=1)
        self.ui.roiR.text = round(modelPredCenter[0], 2)
        self.ui.roiA.text = round(modelPredCenter[1], 2)
        self.ui.roiS.text = round(modelPredCenter[2], 2)
        if self._parameterNode.GetNodeReference("roiNode"):
            self._parameterNode.GetNodeReference("roiNode").SetCenter([float(self._parameterNode.GetParameter('roiR')), float(self._parameterNode.GetParameter('roiA')), float(self._parameterNode.GetParameter('roiS'))])

    def onRoiVisibilityButton(self):
        if self.ui.roiVisibility.checked:
            self._parameterNode.GetNodeReference("roiNode").GetDisplayNode().SetVisibility(True)
        else:
            self._parameterNode.GetNodeReference("roiNode").GetDisplayNode().SetVisibility(False)
        # wasModified = self._parameterNode.StartModify()
        # if self._parameterNode.GetParameter("roiVisibility") == 'true':
        #     self._parameterNode.GetNodeReference("roiNode").GetDisplayNode().SetVisibility(True)
        # else:
        #     self._parameterNode.GetNodeReference("roiNode").GetDisplayNode().SetVisibility(False)
        # self._parameterNode.EndModify(wasModified)

    def onDownloadDataButton(self):
        self.progress_bar_and_run_time.start(maximum=1)
        dcvm.utils.download_data_and_relocate()
        self.progress_bar_and_run_time.end()

    def updateHeartExpDir(self):
        self._parameterNode.SetParameter("heartExpDir", self.ui.heartExpDir.directory)

    def updateCa2ExpDir(self):
        self._parameterNode.SetParameter("ca2ExpDir", self.ui.ca2ExpDir.directory)

    def onHeartVisibilityButton(self):
        '''
        1. turn on all relevant model's visibility
        2. control model visibility with folder visibility
        '''
        inputNodeName = self.cropInputSelector.currentNode().GetName()
        modelNames = ['{}_{}'.format(inputNodeName, key) for key in self.heart_mesh_keys]
        modelNodes = [[node for node in slicer.util.getNodesByClass('vtkMRMLModelNode') if node.GetName() == modelName][0] for modelName in modelNames]
        for modelNodes in modelNodes:
            modelNodes.GetDisplayNode().SetVisibility(True)

        # check if folder exists
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        folder_name = "{}_heart".format(inputNodeName)

        folder_exists = False
        childIds = vtk.vtkIdList() # dummy to save Ids
        shNode.GetItemChildren(shNode.GetSceneItemID(), childIds)
        for itemIdIndex in range(childIds.GetNumberOfIds()): # for all children of the main Subject Hierarchy
            shItemId = childIds.GetId(itemIdIndex)
            if shNode.GetItemName(shItemId) == folder_name:
                grandChildIds = vtk.vtkIdList()
                shNode.GetItemChildren(shItemId, grandChildIds)
                if grandChildIds.GetNumberOfIds() > 0:
                    modelFolderItemId = shItemId
                    folder_exists = True
        
        if folder_exists:
            # folder display manipulation
            pluginHandler = slicer.qSlicerSubjectHierarchyPluginHandler().instance()
            folderPlugin = pluginHandler.pluginByName("Folder")
            folderPlugin.setDisplayVisibility(modelFolderItemId, self.ui.heartVisibility.checked)
        else:
            self.ui.heartVisibility.checked = False

    def onCa2VisibilityButton(self):
        inputNodeName = self.cropInputSelector.currentNode().GetName()
        segmentationName = '{}_Segmentation'.format(inputNodeName)
        segmentationNode = [node for node in slicer.util.getNodesByClass('vtkMRMLSegmentationNode') if node.GetName() == segmentationName][0]
        segmentationNode.GetDisplayNode().SetVisibility(self.ui.ca2Visibility.checked)    

    def onHeartModelLoadButton(self):
        self.progress_bar_and_run_time.start(maximum=1)

        if self._parameterNode.GetParameter("useGPU") == "true":
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        self.pytorch_model_heart = dcvm.io.load_model(self._parameterNode.GetParameter("heartExpDir"), map_location=map_location)
        self._parameterNode.SetParameter("heartModelLoaded", "true")

        # print('Loading done: {}'.format(self._parameterNode.GetParameter("heartExpDir")))
        self.progress_bar_and_run_time.end()

    def onCa2ModelLoadButton(self):
        self.progress_bar_and_run_time.start(maximum=1)

        if self._parameterNode.GetParameter("useGPU") == "true":
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        self.model_ca2 = dcvm.io.load_model(self._parameterNode.GetParameter("ca2ExpDir"), map_location=map_location)
        self._parameterNode.SetParameter("ca2ModelLoaded", "true")

        self.progress_bar_and_run_time.end()

    def onTemplateLoadButton(self):
        self.progress_bar_and_run_time.start(maximum=1)

        template_dir = os.path.join(dcvm_parent_dir, 'template_for_deform/lv_av_aorta')
        template_filename_prefix = self._parameterNode.GetParameter("templateFilenamePrefix")
        self.verts_template_torch, self.laa_elems, self.laa_cell_types, self.laa_faces = dcvm.io.load_template_inference(template_dir, template_filename_prefix)
        self.heart_mesh_keys = list(self.laa_elems.keys()) + list(self.laa_faces.keys())
        self._parameterNode.SetParameter("templateLoaded", "true")
        
        self.progress_bar_and_run_time.end()

    def onUseGpuButton(self):
        # this may not be fully implemented yet
        if self._parameterNode.GetParameter("useGPU") == 'true':
            self.pytorch_model_heart = self.pytorch_model_heart.to(torch.device('cuda'))
            self.model_ca2 = self.model_ca2.to(torch.device('cuda'))
        else:
            self.pytorch_model_heart = self.pytorch_model_heart.to(torch.device('cpu'))
            self.model_ca2 = self.model_ca2.to(torch.device('cpu'))

    def run_crop_volume_sequence(self):
        # turn this into run_crop_volume_single and have a separate function to determine if we should run this multiple times for sequence
        # actually, probably better to just create a separate function for sequences b/c maybe we shouldn't keep creating and deleting the crop parameter node. Also may be inconsistent across image volumes?

        # choose this based on toggle status, not the cropInputNode type??
        cropInputNode = self._parameterNode.GetNodeReference("cropInput")

        if not '{}_seq'.format(self.cropOutputNodeName) in [sequenceNode.GetName() for sequenceNode in slicer.util.getNodesByClass('vtkMRMLSequenceNode')]:
            cropOutputNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", '{}_seq'.format(self.cropOutputNodeName))
        else:
            cropOutputNode = [sequenceNode for sequenceNode in slicer.util.getNodesByClass('vtkMRMLSequenceNode') if sequenceNode.GetName() == '{}_seq'.format(self.cropOutputNodeName)][0]
            cropOutputNode.RemoveAllDataNodes()

        cropVolumeParameters = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLCropVolumeParametersNode")
        cropVolumeParameters.SetROINodeID(self._parameterNode.GetNodeReference("roiNode").GetID())
        
        min_spacing = np.array(cropInputNode.GetNthDataNode(0).GetSpacing()).min()
        spacing = float(self._parameterNode.GetParameter("spacing"))
        zoomFactor = spacing/min_spacing
        cropVolumeParameters.SetSpacingScalingConst(zoomFactor)
        cropVolumeParameters.SetIsotropicResampling(True)

        dummyCropOutputVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "dummyCropOutputVolumeNode")
        
        for idx in range(cropInputNode.GetNumberOfDataNodes()):
            cropVolumeParameters.SetInputVolumeNodeID(cropInputNode.GetNthDataNode(idx).GetID())
            cropVolumeParameters.SetOutputVolumeNodeID(dummyCropOutputVolumeNode.GetID())
            slicer.modules.cropvolume.logic().Apply(cropVolumeParameters)
            cropOutputNode.SetDataNodeAtValue(dummyCropOutputVolumeNode, str(idx))

        sequenceBrowserNodeName = "RoiPred_crop_sequence_browser"
        if not sequenceBrowserNodeName in [node.GetName() for node in slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode')]:
            sequenceBrowserNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", sequenceBrowserNodeName)
            sequenceBrowserNode.AddSynchronizedSequenceNode(cropOutputNode)
        else:
            sequenceBrowserNode = [node for node in slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode') if node.GetName() == sequenceBrowserNodeName][0]
            # sequenceBrowserNode.RemoveAllProxyNodes() # this must come before RemoveAllSequencesNodes to properly remove ProxyNodes (associated with SequenceNodes)
            # sequenceBrowserNode.RemoveAllSequenceNodes()
        
        # For displaying crop output sequence.. probably don't want to do this b/c final output model sequence will be in original image coordinates
        # slicer.modules.sequences.toolBar().setActiveBrowserNode(sequenceBrowserNode)
        # mergedProxyNode = sequenceBrowserNode.GetProxyNode(cropOutputNode)
        # slicer.util.setSliceViewerLayers(background=mergedProxyNode)
        
        # CropVolumeSequence.CropVolumeSequenceLogic().run(cropInputNode, cropOutputNode, cropVolumeParameters)

        slicer.mrmlScene.RemoveNode(cropVolumeParameters)
        slicer.mrmlScene.RemoveNode(dummyCropOutputVolumeNode)
        self._parameterNode.SetNodeReferenceID("cropOutput", cropOutputNode.GetID())
        self.updateVolumeInfo()

    def run_crop_volume_single(self):
        cropInputNode = self._parameterNode.GetNodeReference("cropInput")

        if not self.cropOutputNodeName in [volumeNode.GetName() for volumeNode in slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode')]:
            cropOutputNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", self.cropOutputNodeName)
        else:
            cropOutputNode = [volumeNode for volumeNode in slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode') if volumeNode.GetName() == self.cropOutputNodeName][0]

        cropVolumeParameters = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLCropVolumeParametersNode")
        cropVolumeParameters.SetInputVolumeNodeID(cropInputNode.GetID())
        cropVolumeParameters.SetOutputVolumeNodeID(cropOutputNode.GetID())
        cropVolumeParameters.SetROINodeID(self._parameterNode.GetNodeReference("roiNode").GetID())
        
        min_spacing = np.array(cropInputNode.GetSpacing()).min()
        spacing = float(self._parameterNode.GetParameter("spacing"))
        zoomFactor = spacing/min_spacing
        cropVolumeParameters.SetSpacingScalingConst(zoomFactor)
        cropVolumeParameters.SetIsotropicResampling(True)

        slicer.modules.cropvolume.logic().Apply(cropVolumeParameters)

        slicer.mrmlScene.RemoveNode(cropVolumeParameters)
        self._parameterNode.SetNodeReferenceID("cropOutput", cropOutputNode.GetID())
        self.updateVolumeInfo()
        
        # slicer.modules.cropvolume.logic().FitROIToInputVolume(cropVolumeParameters)

    # if we were to ever use logic appropriately, we would probably only need to move run_*_single functions and possibly update_*_from_* functions (after some editing)
    def run_heart_single(self, inputVolumeNode):
        cropInputNode = self._parameterNode.GetNodeReference("cropInput")

        img = slicer.util.arrayFromVolume(inputVolumeNode) # RAS
        img = img.transpose(2,1,0) # b/c vtk/Slicer flips when getting array from volume
        mat = vtk.vtkMatrix4x4(); inputVolumeNode.GetIJKToRASDirectionMatrix(mat)
        if mat.GetElement(0,0) == -1: # flip based on IJK to RAS DirectionMatrix
            img = np.flip(img, axis=0)
        if mat.GetElement(1,1) == -1:
            img = np.flip(img, axis=1)
        if (img.max() - img.min()) > 1:
            img = dcvm.transforms.ct_normalize(img, min_bound=-158.0, max_bound=864.0)
        
        img = torch.Tensor(np.ascontiguousarray(img))
        img = img.to(next(self.pytorch_model_heart.parameters()).device)[None,None,:,:,:]
        img_size = list(img.squeeze().shape)

        with torch.no_grad():
            output = self.pytorch_model_heart(img)
            # print(img.device)
            displacement_field_tuple = output[0]
            interp_field_list = dcvm.transforms.interpolate_rescale_field_torch(displacement_field_tuple, [self.verts_template_torch.to(next(self.pytorch_model_heart.parameters()).device).unsqueeze(0)], img_size=img_size)
            transformed_verts_np = dcvm.transforms.move_verts_with_field([self.verts_template_torch.to(next(self.pytorch_model_heart.parameters()).device).unsqueeze(0)], interp_field_list)[0].squeeze().cpu().numpy()

        transformed_verts_np *= float(self._parameterNode.GetParameter('spacing')) # assume 1mm/voxel --> 1*spacing mm/voxel
        # transformed_verts_np += - np.array(cropInputNode.GetOrigin())[None,:] + np.array(inputVolumeNode.GetOrigin())[None,:] # fix this for sequence DPDP
        transformed_verts_np += np.array(inputVolumeNode.GetOrigin())[None,:]

        mesh_pv_dict = {}
        for key in self.heart_mesh_keys:
            if key in self.laa_elems.keys():
                mesh_pv_dict[key] = dcvm.ops.mesh_to_UnstructuredGrid(transformed_verts_np, self.laa_elems[key], self.laa_cell_types[key])
            elif key in self.laa_faces.keys():
                mesh_pv_dict[key] = dcvm.ops.mesh_to_PolyData(transformed_verts_np, self.laa_faces[key])
        
        return mesh_pv_dict
    
    def run_ca2_single(self, inputVolumeNode):
        img = slicer.util.arrayFromVolume(inputVolumeNode) # RAS
        img = img.transpose(2,1,0) # b/c vtk/Slicer flips when getting array from volume
        mat = vtk.vtkMatrix4x4(); inputVolumeNode.GetIJKToRASDirectionMatrix(mat)
        if mat.GetElement(0,0) == -1: # flip based on IJK to RAS DirectionMatrix
            img = np.flip(img, axis=0)
        if mat.GetElement(1,1) == -1:
            img = np.flip(img, axis=1)
        if img.max() - img.min() > 1:
            img = dcvm.transforms.ct_normalize(img, min_bound=-158.0, max_bound=864.0)
        
        img = torch.Tensor(np.ascontiguousarray(img))
        img = img.to(next(self.model_ca2.parameters()).device)[None,None,:,:,:]

        with torch.no_grad():
            output = self.model_ca2(img) # output: [1,1,128,128,128]
        
        ca2_cropped_pv = dcvm.ops.seg_to_polydata(output.squeeze().cpu().numpy())
        ca2_pv = ca2_cropped_pv.copy()
        ca2_pv.points *= float(self._parameterNode.GetParameter('spacing'))
        ca2_pv.points += np.array(inputVolumeNode.GetOrigin())[None,:]

        if isinstance(self.cropInputSelector.currentNode(), slicer.vtkMRMLSequenceNode):
            inputVolumeNode = self.cropInputSelector.currentNode().GetNthDataNode(0)
        elif isinstance(self.cropInputSelector.currentNode(), slicer.vtkMRMLScalarVolumeNode):
            inputVolumeNode = self.cropInputSelector.currentNode()
        dimensions = inputVolumeNode.GetImageData().GetDimensions()
        ca2_seg = dcvm.ops.polydata_to_seg(ca2_pv, dims=dimensions[::-1], spacing=[1,1,1], origin=[0,0,0])
        
        return ca2_pv, ca2_seg
    
    def update_model_nodes_from_pv(self, mesh_pv_dict, modelNames_dict):
        if not set(modelNames_dict.values()) <= set([modelNode.GetName() for modelNode in slicer.util.getNodesByClass('vtkMRMLModelNode')]):
            # create separate model node for each mesh component here (lv, aorta, l1, l2, l3, etc.)
            for (key, mesh_pv), color in zip(mesh_pv_dict.items(), colors):
                modelNode = slicer.modules.models.logic().AddModel(pv.PolyData())
                modelNode.SetName(modelNames_dict[key])
                modelNode.SetAndObserveMesh(mesh_pv)
                modelNode.GetDisplayNode().SetColor(*color)
                modelNode.GetDisplayNode().SetEdgeVisibility(True)
                modelNode.GetDisplayNode().SetSliceIntersectionVisibility(True)
                modelNode.GetDisplayNode().SetSliceIntersectionOpacity(0.3)
                modelNode.GetDisplayNode().SetSliceIntersectionThickness(5)
                modelNode.GetDisplayNode().SetVisibility(True)
        else:
            # update existing model nodes if they exist
            modelNodes_dict = {key: [node for node in slicer.util.getNodesByClass('vtkMRMLModelNode') if node.GetName() == modelName][0] for key, modelName in modelNames_dict.items()}
            for key, modelNode in modelNodes_dict.items():
                modelNode.SetAndObserveMesh(mesh_pv_dict[key])
                modelNode.GetDisplayNode().SetVisibility(True)
        self._parameterNode.SetParameter("heartVisibility", "true")

    def update_seg_node_from_np(self, segmentArray):
        inputNode = self.cropInputSelector.currentNode()
        inputNodeName = inputNode.GetName()

        segmentationNodes = [node for node in slicer.util.getNodesByClass('vtkMRMLSegmentationNode') if '{}_Segmentation'.format(inputNodeName) in node.GetName()]
        if len(segmentationNodes) == 0:
            # create new Segmentation node
            segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            segmentationNode.SetName('{}_Segmentation'.format(inputNodeName))
            segmentationNode.CreateDefaultDisplayNodes()
        else:
            # update existing Segmentation node
            segmentationNode = segmentationNodes[0]
        
        segment_name = "{}_ca2".format(inputNodeName)
        
        # create new Segment only if segment_name doesn't exist in Segmentation already
        if segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segment_name) == '':
            segmentationNode.GetSegmentation().AddEmptySegment(segment_name)

        # grab segmentNode to update
        segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segment_name)
        segmentNode = segmentationNode.GetSegmentation().GetSegment(segmentId)
        segmentNode.SetColor((177/256, 122/256, 101/256))

        # update segmentNode
        segmentArray = segmentArray.transpose([2,1,0])
        slicer.util.updateSegmentBinaryLabelmapFromArray(segmentArray, segmentationNode, segmentId, inputNode)

        # update plot info
        # segmentationNode.GetSegmentation().SetConversionParameter("Surface smoothing", "False") # I wish I could do this, but doesn't work..
        segmentationNode.GetSegmentation().SetConversionParameter("Smoothing factor", "0.0")
        segmentationNode.CreateClosedSurfaceRepresentation()
        segmentationNode.GetDisplayNode().SetVisibility(True)
        # segmentationNode.GetDisplayNode().SetOpacity(0.5)
        self._parameterNode.SetParameter("ca2Visibility", "true")

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

    def put_models_in_folder(self, inputNodeName, modelNames_dict):
        ''' we do this to allow for easy grouping of visualization '''
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()

        # check if folder exists
        folder_name = "{}_heart".format(inputNodeName)

        create_new_folder = True
        childIds = vtk.vtkIdList() # dummy to save Ids
        shNode.GetItemChildren(shNode.GetSceneItemID(), childIds)
        for itemIdIndex in range(childIds.GetNumberOfIds()): # for all children of the main Subject Hierarchy
            shItemId = childIds.GetId(itemIdIndex)
            if shNode.GetItemName(shItemId) == folder_name:
                grandChildIds = vtk.vtkIdList()
                shNode.GetItemChildren(shItemId, grandChildIds)
                if grandChildIds.GetNumberOfIds() > 0:
                    modelFolderItemId = shItemId
                    create_new_folder = False
        if create_new_folder:
            modelFolderItemId = shNode.CreateFolderItem(shNode.GetSceneItemID(), folder_name)
        
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

    def onCropAndRunButton(self):
        self.progress_bar_and_run_time.start(maximum=1)

        cropInputNode = self._parameterNode.GetNodeReference("cropInput")

        start = time.time()
        if self._parameterNode.GetParameter("inputSequenceOrVolume") == "true":
            self.run_crop_volume_sequence()
        else:
            self.run_crop_volume_single()
        cropOutputNode = self._parameterNode.GetNodeReference("cropOutput")
        end = time.time()
        print('Crop done: {} seconds'.format(end-start))
        
        inputNodeName = self.cropInputSelector.currentNode().GetName()
        if self._parameterNode.GetParameter("templateLoaded") == "true":
            modelNames_dict = {key: '{}_{}'.format(inputNodeName, key) for key in self.heart_mesh_keys}

        # turn off visibility for all existing model/segmentation nodes. Only turn on visibility for new model/segmentation nodes
        for modelNode in slicer.util.getNodesByClass('vtkMRMLModelNode'):
            modelNode.GetDisplayNode().SetVisibility(False)
        for segmentationNode in slicer.util.getNodesByClass('vtkMRMLSegmentationNode'):
            segmentationNode.GetDisplayNode().SetVisibility(False)

        start = time.time()

        # sequence
        if isinstance(cropOutputNode, slicer.vtkMRMLSequenceNode):
            tempModelNode = slicer.modules.models.logic().AddModel(pv.PolyData())

            if not set(modelNames_dict.values()) <= set([modelNode.GetName() for modelNode in slicer.util.getNodesByClass('vtkMRMLSequenceNode')]): # subset
                # create separate sequence node for each mesh component here (lv, aorta, l1, l2, l3, etc.)
                modelSequenceNodes_dict = {key: slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceNode", modelName) for key, modelName in modelNames_dict.items()}
            else:
                # update old sequence nodes if they exist
                modelSequenceNodes_dict = {key: [node for node in slicer.util.getNodesByClass('vtkMRMLSequenceNode') if node.GetName() == modelName][0] for key, modelName in modelNames_dict.items()}
                for node in modelSequenceNodes_dict.values():
                    node.RemoveAllDataNodes()
            
            # add / modify data in sequence
            for idx in range(cropOutputNode.GetNumberOfDataNodes()):
                mesh_pv_dict = self.run_heart_single(cropOutputNode.GetNthDataNode(idx))
                
                for key in self.heart_mesh_keys:
                    tempModelNode.SetName('dummy_{}_{}'.format(modelNames_dict[key], idx))
                    tempModelNode.SetAndObserveMesh(mesh_pv_dict[key])
                    modelSequenceNodes_dict[key].SetDataNodeAtValue(tempModelNode, str(idx))

            slicer.mrmlScene.RemoveNode(tempModelNode)
            
            # Add default SequenceBrowser behavior (add image and all meshes for synced display)
            if not self.outputSequenceBrowserNodeName in [node.GetName() for node in slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode')]:
                sequenceBrowserNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSequenceBrowserNode", self.outputSequenceBrowserNodeName)
                sequenceBrowserNode.SetPlaybackRateFps(2.0) # slower b/c synced with model
                for modelSequenceNode in modelSequenceNodes_dict.values():
                    sequenceBrowserNode.AddSynchronizedSequenceNode(modelSequenceNode)
                sequenceBrowserNode.AddSynchronizedSequenceNode(cropInputNode) # original image
            else:
                sequenceBrowserNode = [node for node in slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode') if node.GetName() == self.outputSequenceBrowserNodeName][0]
            
            # need to set default display properties on proxyNodes
            for sequenceNode, color in zip(modelSequenceNodes_dict.values(), roi_pred_lib.colors):
                proxyNode = sequenceBrowserNode.GetProxyNode(sequenceNode)
                proxyNode.GetDisplayNode().SetColor(*color)
                proxyNode.GetDisplayNode().SetEdgeVisibility(True)
                proxyNode.GetDisplayNode().SetSliceIntersectionVisibility(True)
                proxyNode.GetDisplayNode().SetSliceIntersectionOpacity(0.3)
                modelNode.GetDisplayNode().SetSliceIntersectionThickness(5)
                proxyNode.GetDisplayNode().SetVisibility(True)
            
            # For setting active sequence browser node to img + model sequence
            slicer.modules.sequences.toolBar().setActiveBrowserNode(sequenceBrowserNode)
            imgProxyNode = sequenceBrowserNode.GetProxyNode(cropInputNode)
            slicer.util.setSliceViewerLayers(background=imgProxyNode)

            self._parameterNode.SetParameter("heartRunCompleted", "true")
        
        # volume
        elif isinstance(cropOutputNode, slicer.vtkMRMLScalarVolumeNode):
            if (self._parameterNode.GetParameter("heartModelLoaded") == "true") and (self._parameterNode.GetParameter("templateLoaded") == "true"):
                mesh_pv_dict = self.run_heart_single(cropOutputNode)
                self.update_model_nodes_from_pv(mesh_pv_dict, modelNames_dict)
                self._parameterNode.SetParameter("heartRunCompleted", "true")

            if self._parameterNode.GetParameter("ca2ModelLoaded") == "true":
                ca2_pv, ca2_seg = self.run_ca2_single(cropOutputNode)
                # self.update_model_nodes_from_pv({'ca2': ca2_pv}, {'ca2': 'test_ca2'})
                self.update_seg_node_from_np(ca2_seg)
                self._parameterNode.SetParameter("ca2RunCompleted", "true")

            slicer.util.setSliceViewerLayers(background=cropInputNode)

        self.put_models_in_folder(inputNodeName, modelNames_dict)

        end = time.time()
        print('Model run(s) done: {} seconds'.format(end-start))

        self.progress_bar_and_run_time.end()

    def onSaveOutputsInpButton(self):
        inputNode = self.cropInputSelector.currentNode()
        inputNodeName = inputNode.GetName()
        modelNames_dict = {key: '{}_{}'.format(inputNodeName, key) for key in self.heart_mesh_keys}

        exp_dir = self._parameterNode.GetParameter("heartExpDir")
        save_dir = os.path.join(exp_dir, '3d_slicer_outputs')
        os.makedirs(save_dir, exist_ok=True)
        
        if isinstance(inputNode, slicer.vtkMRMLSequenceNode):
            # remove sequenceBrowserNode --> sequenceNodes --> proxyNodes
            modelSequenceNodes_dict = {key: node for key, modelName in modelNames_dict.items() for node in [node for node in slicer.util.getNodesByClass('vtkMRMLSequenceNode') if node.GetName() == modelName]}
            if len(modelSequenceNodes_dict) > 0: # to avoid error when model doesn't exist
                modelSequence0 = list(modelSequenceNodes_dict.values())[0]
                self.progress_bar_and_run_time.start(maximum=modelSequence0.GetNumberOfDataNodes())
                for itemIndex in range(modelSequence0.GetNumberOfDataNodes()): # all models should have the same number of data nodes
                    mesh_pv_dict_nth_timepoint = {key: pv.UnstructuredGrid(modelSequenceNode.GetNthDataNode(itemIndex).GetMesh()) for key, modelSequenceNode in modelSequenceNodes_dict.items()}
                    save_filepath = os.path.join(save_dir, '{}.inp'.format(inputNode.GetNthDataNode(itemIndex).GetName())) # define save_filepath
                    verts, _ = dcvm.ops.get_verts_faces_from_pyvista(list(mesh_pv_dict_nth_timepoint.values())[0])
                    elems_dict = {key: dcvm.ops.get_verts_faces_from_pyvista(mesh_pv)[1] for key, mesh_pv in mesh_pv_dict_nth_timepoint.items()} # list for 5 different comps at one time point
                    cell_types_dict = {key: mesh_pv.celltypes for key, mesh_pv in mesh_pv_dict_nth_timepoint.items()} # list for 5 different comps at one time point
                    dcvm.io.write_inp_file(save_filepath, verts, elems_dict, cell_types_dict)
                    # print('saved {}/{}'.format(itemIndex+1, modelSequence0.GetNumberOfDataNodes()))
                    self.progress_bar_and_run_time.step(itemIndex+1)
                
        elif isinstance(inputNode, slicer.vtkMRMLScalarVolumeNode):
            modelNodes_dict = {key: node for key, modelName in modelNames_dict.items() for node in [node for node in slicer.util.getNodesByClass('vtkMRMLModelNode') if node.GetName() == modelName]}
            if len(modelNodes_dict) > 0: # to avoid error when model doesn't exist
                self.progress_bar_and_run_time.start(maximum=1)
                mesh_pv_dict = {key: pv.UnstructuredGrid(modelNode.GetMesh()) for key, modelNode in modelNodes_dict.items()}
                save_filepath = os.path.join(save_dir, '{}.inp'.format(inputNodeName)) # define save_filepath
                verts, _ = dcvm.ops.get_verts_faces_from_pyvista(list(mesh_pv_dict.values())[0])
                elems_dict = {key: dcvm.ops.get_verts_faces_from_pyvista(mesh_pv)[1] for key, mesh_pv in mesh_pv_dict.items()}
                cell_types_dict = {key: mesh_pv.celltypes for key, mesh_pv in mesh_pv_dict.items()}
                dcvm.io.write_inp_file(save_filepath, verts, elems_dict, cell_types_dict)

        self.progress_bar_and_run_time.end()

    def onRemoveOutputNodesButton(self):
        inputNode = self.cropInputSelector.currentNode()
        inputNodeName = inputNode.GetName()
        modelNames_dict = {key: '{}_{}'.format(inputNodeName, key) for key in self.heart_mesh_keys}
        
        if isinstance(inputNode, slicer.vtkMRMLSequenceNode):
            # remove sequenceBrowserNode --> sequenceNodes --> proxyNodes
            outputSequenceBrowserNodes = [node for node in slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode') if node.GetName() == self.outputSequenceBrowserNodeName]
            modelSequenceNodes = [node for modelName in modelNames_dict.values() for node in [node for node in slicer.util.getNodesByClass('vtkMRMLSequenceNode') if node.GetName() == modelName]]
            modelSequenceProxyNodes = [node for modelName in modelNames_dict.values() for node in [node for node in slicer.util.getNodesByClass('vtkMRMLModelNode') if node.GetName() == modelName]]
            inputProxyNodes = [node for node in slicer.util.getNodesByClass('vtkMRMLScalarVolumeNode') if node.GetName() == inputNodeName]
            if len(outputSequenceBrowserNodes) > 0: # to avoid error when model doesn't exist
                for outputSequenceBrowserNode in outputSequenceBrowserNodes:
                    slicer.mrmlScene.RemoveNode(outputSequenceBrowserNode)
                for modelSequenceNode in modelSequenceNodes:
                    slicer.mrmlScene.RemoveNode(modelSequenceNode)
                for modelSequenceProxyNode in modelSequenceProxyNodes:
                    slicer.mrmlScene.RemoveNode(modelSequenceProxyNode)
                for inputProxyNode in inputProxyNodes:
                    slicer.mrmlScene.RemoveNode(inputProxyNode) # we can remove all input proxy nodes, even the original sequence ones.. it will generate new ones immediately if sequence browser is still alive
        elif isinstance(inputNode, slicer.vtkMRMLScalarVolumeNode):
            modelNodes = [node for modelName in modelNames_dict.values() for node in [node for node in slicer.util.getNodesByClass('vtkMRMLModelNode') if node.GetName() == modelName]]
            if len(modelNodes) > 0: # to avoid error when model doesn't exist
                for modelNode in modelNodes:
                    slicer.mrmlScene.RemoveNode(modelNode)
        
        segmentationNodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        if len(segmentationNodes) > 0:
            segmentationNode = [node for node in segmentationNodes if '{}_Segmentation'.format(inputNodeName) in node.GetName()][0]
            slicer.mrmlScene.RemoveNode(segmentationNode)

#
# RoiPredLogic
#

class RoiPredLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("inputSequenceOrVolume"):
            parameterNode.SetParameter("inputSequenceOrVolume", "false")
        if not parameterNode.GetParameter("pytorchInputSpatialDim0"):
            parameterNode.SetParameter("pytorchInputSpatialDim0", "128")
        if not parameterNode.GetParameter("pytorchInputSpatialDim1"):
            parameterNode.SetParameter("pytorchInputSpatialDim1", "128")
        if not parameterNode.GetParameter("pytorchInputSpatialDim2"):
            parameterNode.SetParameter("pytorchInputSpatialDim2", "128")
        if not parameterNode.GetParameter("spacing"):
            parameterNode.SetParameter("spacing", "1.25")
        if not parameterNode.GetParameter("roiR"):
            parameterNode.SetParameter("roiR", "0")
        if not parameterNode.GetParameter("roiA"):
            parameterNode.SetParameter("roiA", "0")
        if not parameterNode.GetParameter("roiS"):
            parameterNode.SetParameter("roiS", "0")
        if not parameterNode.GetParameter("roiVisibility"):
            parameterNode.SetParameter("roiVisibility", "false")
        if not parameterNode.GetParameter("heartExpDir") or parameterNode.GetParameter("heartExpDir") == '.':
            parameterNode.SetParameter("heartExpDir", os.path.join(dcvm_parent_dir, 'experiments/_newest_heart'))
        if not parameterNode.GetParameter("ca2ExpDir") or parameterNode.GetParameter("ca2ExpDir") == '.':
            parameterNode.SetParameter("ca2ExpDir", os.path.join(dcvm_parent_dir, 'experiments/_newest_ca2'))
        if not parameterNode.GetParameter("followExpParams"):
            parameterNode.SetParameter("followExpParams", "false")
        if not parameterNode.GetParameter("templateFilenamePrefix"):
            parameterNode.SetParameter("templateFilenamePrefix", "combined_v12")
        if not parameterNode.GetParameter("heartModelLoaded"):
            parameterNode.SetParameter("heartModelLoaded", "false")
        if not parameterNode.GetParameter("ca2ModelLoaded"):
            parameterNode.SetParameter("ca2ModelLoaded", "false")
        if not parameterNode.GetParameter("templateLoaded"):
            parameterNode.SetParameter("templateLoaded", "false")
        if not parameterNode.GetParameter("heartRunCompleted"):
            parameterNode.SetParameter("heartRunCompleted", "false")
        if not parameterNode.GetParameter("ca2RunCompleted"):
            parameterNode.SetParameter("ca2RunCompleted", "false")
        if not parameterNode.GetParameter("originalVolumeSpacingDisplay0"):
            parameterNode.SetParameter("originalVolumeSpacingDisplay0", "0")
        if not parameterNode.GetParameter("originalVolumeSpacingDisplay1"):
            parameterNode.SetParameter("originalVolumeSpacingDisplay1", "0")
        if not parameterNode.GetParameter("originalVolumeSpacingDisplay2"):
            parameterNode.SetParameter("originalVolumeSpacingDisplay2", "0")
        if not parameterNode.GetParameter("originalVolumeDimensionsDisplay0"):
            parameterNode.SetParameter("originalVolumeDimensionsDisplay0", "0")
        if not parameterNode.GetParameter("originalVolumeDimensionsDisplay1"):
            parameterNode.SetParameter("originalVolumeDimensionsDisplay1", "0")
        if not parameterNode.GetParameter("originalVolumeDimensionsDisplay2"):
            parameterNode.SetParameter("originalVolumeDimensionsDisplay2", "0")
        if not parameterNode.GetParameter("croppedVolumeSpacingDisplay0"):
            parameterNode.SetParameter("croppedVolumeSpacingDisplay0", "0")
        if not parameterNode.GetParameter("croppedVolumeSpacingDisplay1"):
            parameterNode.SetParameter("croppedVolumeSpacingDisplay1", "0")
        if not parameterNode.GetParameter("croppedVolumeSpacingDisplay2"):
            parameterNode.SetParameter("croppedVolumeSpacingDisplay2", "0")
        if not parameterNode.GetParameter("croppedVolumeDimensionsDisplay0"):
            parameterNode.SetParameter("croppedVolumeDimensionsDisplay0", "0")
        if not parameterNode.GetParameter("croppedVolumeDimensionsDisplay1"):
            parameterNode.SetParameter("croppedVolumeDimensionsDisplay1", "0")
        if not parameterNode.GetParameter("croppedVolumeDimensionsDisplay2"):
            parameterNode.SetParameter("croppedVolumeDimensionsDisplay2", "0")
        if not parameterNode.GetParameter("useGPU"):
            if torch.cuda.is_available():
                parameterNode.SetParameter("useGPU", "true")
            else:
                parameterNode.SetParameter("useGPU", "false")
        
    def process(self, inputVolume, outputVolume, imageThreshold, invert=False, showResult=True):
        pass


#
# RoiPredTest
#

class RoiPredTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """ Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_RoiPred1()

    def test_RoiPred1(self):
        """ Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData
        # registerSampleData()
        # inputVolume = SampleData.downloadSample('RoiPred_test_input')
        inputVolume = SampleData.SampleDataLogic().downloadCTChest()
        inputVolume.SetName('RoiPred_test_input')
        self.delayDisplay('Loaded test data set')

        # inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(inputScalarRange[0], 0)
        # self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        outputVolume.SetName('RoiPred_test_output')

        # Test the module logic

        logic = RoiPredLogic()

        print('test case')

        # # Test algorithm with non-inverted threshold
        # logic.process(inputVolume, outputVolume, threshold, True)
        # outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        # self.assertEqual(outputScalarRange[1], threshold)

        # # Test algorithm with inverted threshold
        # logic.process(inputVolume, outputVolume, threshold, False)
        # outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        # self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay('Test passed')
