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
import sys
import importlib
import traceback

import vtk
import qt
import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import RoiPredSetup.RoiPredSetup
importlib.reload(RoiPredSetup.RoiPredSetup)

curr_file_dir_path = os.path.dirname(os.path.realpath(__file__))
dcvm_parent_dir = os.path.abspath(os.path.join(curr_file_dir_path, '../..'))
if dcvm_parent_dir not in sys.path:
    sys.path.append(dcvm_parent_dir)
SlicerDeepCardiac_dir = os.path.abspath(os.path.join(curr_file_dir_path, '..'))
if SlicerDeepCardiac_dir not in sys.path:
    sys.path.append(SlicerDeepCardiac_dir)

try:
    import numpy as np
    import torch
    import pyvista as pv
    import matplotlib
    colors_stl = matplotlib.colormaps['Accent'].colors

    import dcvm
    import HelperLib as helper_lib
    import RoiPredLib.RoiPredLib as roi_pred_lib
    importlib.reload(roi_pred_lib)
    importlib.reload(roi_pred_lib)
    importlib.reload(roi_pred_lib)
    import types
    def reload_package(package):
        """Recursively reload all modules in the given package."""
        importlib.reload(package)
        for _, module in package.__dict__.items():
            if isinstance(module, types.ModuleType) and (module.__name__.startswith("dcvm") or module.__name__.startswith("HelperLib")):
                reload_package(module)
    reload_package(dcvm)
    reload_package(dcvm)
    reload_package(dcvm)
    reload_package(helper_lib)
    reload_package(helper_lib)
    reload_package(helper_lib)
except:
    traceback.print_exc()
    print(' ')
    print('RoiPred: probably need to pip_install required packages. (Re-)enter or reload module to trigger installation.')
    print(' ')

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

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = RoiPredLogic()
        self.logic.initNamingConvention(self)

        installed_any_pkg, cancelled_any_installation = RoiPredSetup.RoiPredSetup.install_missing_pkgs_in_slicer()
        if installed_any_pkg and (not cancelled_any_installation): ScriptedLoadableModuleWidget.onReload(self)
        self.first_enter = True

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/RoiPred.ui'))
        self.uiWidget = uiWidget
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Connections

        # enter() and exit() should just work without adding observers -- https://discourse.slicer.org/t/scripted-module-leak-addobserver/121/4

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.inputSequenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
        self.ui.inputSequenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.displaySelectedNode)
        self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.displaySelectedNode)
        self.ui.inputSequenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateVisibilityCheckBoxes)
        self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateVisibilityCheckBoxes)
        self.ui.inputSequenceSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateVolumeInfo)
        self.ui.inputVolumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateVolumeInfo)
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
        if not torch.cuda.is_available():
            self.ui.useGPU.enabled = False
        self.ui.heartVisibility.toggled.connect(self.updateParameterNodeFromGUI)
        self.ui.ca2Visibility.toggled.connect(self.updateParameterNodeFromGUI)
        self.ui.heartStlVisibility.toggled.connect(self.updateParameterNodeFromGUI)
        self.ui.templateFilePath.currentPathChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.aortaLvThickness.textChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.leafletThickness.textChanged.connect(self.updateParameterNodeFromGUI)
        self.ui.fuseLeaflets.toggled.connect(self.updateParameterNodeFromGUI)

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
        self.ui.setupDataButton.clicked.connect(self.onSetupDataButton)
        self.ui.heartExpDir.directoryChanged.connect(self.updateHeartExpDir)
        self.ui.ca2ExpDir.directoryChanged.connect(self.updateCa2ExpDir)
        self.ui.useGPU.clicked.connect(self.onUseGpuButton)
        self.ui.cropAndRunButton.clicked.connect(self.onCropAndRunButton)
        self.ui.saveOutputsInpButton.clicked.connect(self.onSaveOutputsInpButton)
        self.ui.modelsToStlButton.clicked.connect(self.onModelsToStlButton)
        self.ui.saveOutputsStlButton.clicked.connect(self.onSaveOutputsStlButton)
        self.ui.removeOutputNodesButton.clicked.connect(self.onRemoveOutputNodesButton)

        self.ui.roiVisibility.clicked.connect(self.onRoiVisibility)
        self.ui.heartVisibility.clicked.connect(self.onHeartVisibility)
        self.ui.ca2Visibility.clicked.connect(self.onCa2Visibility)
        self.ui.heartStlVisibility.clicked.connect(self.onHeartStlVisibility)

        self.updateRoiVisibilityCheckBox = helper_lib.UpdateCheckboxWithDataNodeVisibility(self.ui.roiVisibility)
        self.updateHeartVisibilityCheckBox = helper_lib.UpdateCheckboxWithDataNodeVisibility(self.ui.heartVisibility)
        self.updateCa2VisibilityCheckbox = helper_lib.UpdateCheckboxWithSegmentVisibility(self.ui.ca2Visibility, self.segmentationNodeName_suffix, [self.ca2_segmentName_suffix])
        self.updateHeartStlVisibilityCheckBox = helper_lib.UpdateCheckboxWithDataNodeVisibility(self.ui.heartStlVisibility)

        self.crosshair = slicer.util.getNode('Crosshair')

        self.ui.resetCollapsibleButton.checked = False
        self.ui.stlCollapsibleButton.checked = False
        self.ui.volumeInfoCollapsibleButton.checked = False
        self.ui.aortaLvThickness.lineEdit().setReadOnly(True)
        self.ui.leafletThickness.lineEdit().setReadOnly(True)

        try:
            self.progress_bar_and_run_time = helper_lib.ProgressBarAndRunTime(self.ui.progressBar)
        except Exception:
            uiWidget.setEnabled(False)
            traceback.print_exc()
        
        # Make sure parameter node is initialized (needed for module reload)
        # self.parameterNodeObserved = False
        self.initializeParameterNode()
        roiNode = slicer.util.getFirstNodeByClassByName("vtkMRMLMarkupsROINode", self.roiNodeName)
        if roiNode:
            self._parameterNode.SetNodeReferenceID("roiNode", roiNode.GetID())
        else:
            self._parameterNode.SetNodeReferenceID("roiNode", "None")
        self._parameterNode.SetParameter("heartModelLoaded", "false")
        self._parameterNode.SetParameter("templateLoaded", "false")
        self._parameterNode.SetParameter("ca2ModelLoaded", "false")

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
        if not self.first_enter: # continue asking to install packages if need be (re-enter module to trigger prompt)
            installed_any_pkg, cancelled_any_installation = RoiPredSetup.RoiPredSetup.install_missing_pkgs_in_slicer()
            if installed_any_pkg and (not cancelled_any_installation): ScriptedLoadableModuleWidget.onReload(self)
        self.first_enter = False

        if hasattr(self, 'cropInputSelector'):
            self.displaySelectedNode(displayOutputs=False)
            self.updateVisibilityCheckBoxes()

        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self):
        """
        Called each time the user opens a different module.
        If we update this, need to restart 3D slicer to see the effects
        """
        if self._parameterNode.GetNodeReference("roiNode") is not None:
            roiNode = self._parameterNode.GetNodeReference("roiNode")
            roiNode.GetDisplayNode().SetVisibility(False)

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
        self.ui.templateFilePath.currentPath = self._parameterNode.GetParameter("templateFilePath")

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
        self.ui.heartStlVisibility.checked = True if self._parameterNode.GetParameter("heartStlVisibility") == "true" else False
        self.ui.heartModelLoadedCheck.checked = True if self._parameterNode.GetParameter("heartModelLoaded") == "true" else False
        self.ui.templateLoadedCheck.checked = True if self._parameterNode.GetParameter("templateLoaded") == "true" else False
        self.ui.ca2ModelLoadedCheck.checked = True if self._parameterNode.GetParameter("ca2ModelLoaded") == "true" else False

        self.ui.aortaLvThickness.setValue(float(self._parameterNode.GetParameter("aortaLvThickness")))
        self.ui.leafletThickness.setValue(float(self._parameterNode.GetParameter("leafletThickness")))
        self.ui.fuseLeaflets.checked = True if self._parameterNode.GetParameter("fuseLeaflets") == "true" else False
        
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

        # self.ui.roiVisibility.enabled = True if self._parameterNode.GetNodeReference("roiNode") else False
        # self.ui.heartVisibility.enabled = True if self._parameterNode.GetParameter("heartRunCompleted") == "true" else False
        # self.ui.ca2Visibility.enabled = True if self._parameterNode.GetParameter("ca2RunCompleted") == "true" else False

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

        self._parameterNode.SetParameter("inputSequenceOrVolume", "true" if self.ui.inputSequenceOrVolume.checked else "false")
        self._parameterNode.SetParameter("pytorchInputSpatialDim0", self.ui.pytorchInputSpatialDim0.text)
        self._parameterNode.SetParameter("pytorchInputSpatialDim1", self.ui.pytorchInputSpatialDim1.text)
        self._parameterNode.SetParameter("pytorchInputSpatialDim2", self.ui.pytorchInputSpatialDim2.text)
        self._parameterNode.SetParameter("spacing", self.ui.spacing.text)
        self._parameterNode.SetParameter("roiR", self.ui.roiR.text)
        self._parameterNode.SetParameter("roiA", self.ui.roiA.text)
        self._parameterNode.SetParameter("roiS", self.ui.roiS.text)
        self._parameterNode.SetParameter("roiVisibility", "true" if self.ui.roiVisibility.checked else "false")
        self._parameterNode.SetParameter("heartVisibility", "true" if self.ui.heartVisibility.checked else "false")
        self._parameterNode.SetParameter("ca2Visibility", "true" if self.ui.ca2Visibility.checked else "false")
        self._parameterNode.SetParameter("heartStlVisibility", "true" if self.ui.heartStlVisibility.checked else "false")
        self._parameterNode.SetParameter("useGPU", "true" if self.ui.useGPU.checked else "false")
        self._parameterNode.SetParameter("heartExpDir", self.ui.heartExpDir.directory)
        self._parameterNode.SetParameter("ca2ExpDir", self.ui.ca2ExpDir.directory)
        self._parameterNode.SetParameter("templateFilePath", self.ui.templateFilePath.currentPath)

        self._parameterNode.SetParameter("aortaLvThickness", self.ui.aortaLvThickness.text)
        self._parameterNode.SetParameter("leafletThickness", self.ui.leafletThickness.text)
        self._parameterNode.SetParameter("fuseLeaflets", "true" if self.ui.fuseLeaflets.checked else "false")

        self._parameterNode.EndModify(wasModified)

    def defineOutputNamesFromInputSelector(self):
        if self.cropInputSelector.currentNode():
            # display volume node
            if isinstance(self.cropInputSelector.currentNode(), slicer.vtkMRMLSequenceNode):
                inputVolumeNode = self.cropInputSelector.currentNode().GetNthDataNode(0)
            elif isinstance(self.cropInputSelector.currentNode(), slicer.vtkMRMLScalarVolumeNode):
                inputVolumeNode = self.cropInputSelector.currentNode()

            inputNodeName = self.cropInputSelector.currentNode().GetName()
            if hasattr(self, "heart_mesh_keys"): # to prevent issues for ca2_prediction_only cases
                self.modelNames_dict = {key: '{}_{}'.format(inputNodeName, key) for key in self.heart_mesh_keys}
                self.stlNames_dict = {key: '{}_{}'.format(inputNodeName, key) for key in self.heart_stl_keys}
            self.heartModelFolderName = "{}{}".format(inputNodeName, self.heartModelFolderName_suffix)
            self.segmentationNodeName = "{}{}".format(inputNodeName, self.segmentationNodeName_suffix)
            self.segmentName = "{}{}".format(inputNodeName, self.ca2_segmentName_suffix)
            self.heartStlFolderName = "{}{}".format(inputNodeName, self.heartStlFolderName_suffix)
            self.inputNodeName = inputNodeName

    def updateVisibilityCheckBoxes(self):
        self.defineOutputNamesFromInputSelector()
        
        roiNode = slicer.util.getFirstNodeByClassByName("vtkMRMLMarkupsROINode", self.roiNodeName)
        
        if hasattr(self, "heartModelFolderName"):
            heartFolderNode = slicer.util.getFirstNodeByClassByName("vtkMRMLFolderDisplayNode", self.heartModelFolderName)
        else:
            heartFolderNode = None

        if hasattr(self, "segmentationNodeName"):
            segmentationNode = slicer.util.getFirstNodeByClassByName("vtkMRMLSegmentationNode", self.segmentationNodeName)
        else:
            segmentationNode = None

        if hasattr(self, "heartStlFolderName"):
            heartStlFolderNode = slicer.util.getFirstNodeByClassByName("vtkMRMLFolderDisplayNode", self.heartStlFolderName)
        else:
            heartStlFolderNode = None

        if roiNode:
            self.updateRoiVisibilityCheckBox(roiNode)
        else:
            self.updateRoiVisibilityCheckBox.checkbox.checked = False

        if heartFolderNode:
            self.updateHeartVisibilityCheckBox(heartFolderNode)
        else:
            self.updateHeartVisibilityCheckBox.checkbox.checked = False
            
        if segmentationNode:
            self.updateCa2VisibilityCheckbox(segmentationNode)
        else:
            self.updateCa2VisibilityCheckbox.checkbox.checked = False

        if heartStlFolderNode:
            self.updateHeartStlVisibilityCheckBox(heartStlFolderNode)
        else:
            self.updateHeartStlVisibilityCheckBox.checkbox.checked = False

    def displaySelectedNode(self, caller=None, displayOutputs=True):
        """
        if displayOutputs is True, display ALL available outputs with the names associated with selected input node
        """

        cropInputNode = self.cropInputSelector.currentNode()
        shNode = slicer.mrmlScene.GetSubjectHierarchyNode()

        if cropInputNode is not None:
            # display volume node
            if isinstance(cropInputNode, slicer.vtkMRMLSequenceNode):
                sequenceBrowserNode = helper_lib.update_outputSequenceBrowserNode(self.outputSequenceBrowserNodeName, imgSequenceNode=cropInputNode)
            elif isinstance(cropInputNode, slicer.vtkMRMLScalarVolumeNode):
                inputVolumeNode = self.cropInputSelector.currentNode()
                slicer.util.setSliceViewerLayers(background=inputVolumeNode)

            if displayOutputs:
                # display only the name-associated models and segs
                self.defineOutputNamesFromInputSelector()

                folderNames, _, _ = helper_lib.get_all_folders_containing_suffix(self.heartModelFolderName_suffix)
                for folderName in folderNames:
                    helper_lib.set_folder_visibility(folderName, folderName == self.heartModelFolderName, updateContentsVisibility=True)

                segmentNames, _, _ = helper_lib.get_all_segments_containing_suffix(self.ca2_segmentName_suffix)
                for segmentName in segmentNames:
                    helper_lib.set_segment_visibility(segmentName, segmentName == self.segmentName)

        else: # empty model/seg display when None is selected on inputVolumeSelector
            folderNames, _, _ = helper_lib.get_all_folders_containing_suffix(self.heartModelFolderName_suffix)
            for folderName in folderNames:
                helper_lib.set_folder_visibility(folderName, False)

            segmentNames, _, _ = helper_lib.get_all_segments_containing_suffix(self.ca2_segmentName_suffix)
            for segmentName in segmentNames:
                helper_lib.set_segment_visibility(segmentName, False)

            folderNames, _, _ = helper_lib.get_all_folders_containing_suffix(self.heartStlFolderName_suffix)
            for folderName in folderNames:
                helper_lib.set_folder_visibility(folderName, False)

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
        else: # default
            self.ui.originalVolumeSpacingDisplay0.text = "0"
            self.ui.originalVolumeSpacingDisplay1.text = "0"
            self.ui.originalVolumeSpacingDisplay2.text = "0"
            self.ui.originalVolumeDimensionsDisplay0.text = "0"
            self.ui.originalVolumeDimensionsDisplay1.text = "0"
            self.ui.originalVolumeDimensionsDisplay2.text = "0"
        
        # default - structured this way b/c we want default if _parameterNode is None or self._parameterNode.GetNodeReference("cropOutput") is None
        self.ui.croppedVolumeSpacingDisplay0.text = "0"
        self.ui.croppedVolumeSpacingDisplay1.text = "0"
        self.ui.croppedVolumeSpacingDisplay2.text = "0"
        self.ui.croppedVolumeDimensionsDisplay0.text = "0"
        self.ui.croppedVolumeDimensionsDisplay1.text = "0"
        self.ui.croppedVolumeDimensionsDisplay2.text = "0"
        if self._parameterNode: # prevent error when _parameterNode is deleted but we're still calling updateVolumeInfo
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

    def onInputSequenceOrVolume(self, caller=None, event=None):
        self.initializeParameterNode()
        self.displaySelectedNode(displayOutputs=True)
        self.updateVisibilityCheckBoxes()
        self.updateVolumeInfo()

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
        # delete all roiNode's with the default name
        for roiNode in slicer.util.getNodesByClass('vtkMRMLMarkupsROINode'):
            if roiNode.GetName() == self.roiNodeName:
                roiNode.RemoveAllObservers()
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

        mesh_pv_list = [pv.UnstructuredGrid(modelNode.GetMesh()) for modelNode in slicer.util.getNodesByClass('vtkMRMLModelNode') if modelNode.GetName() in self.modelNames_dict.values()] # pv.UnstructuredGrid to make proxyNode compatible for pv.merge
        mesh_pv_all = pv.merge(mesh_pv_list)
        modelPredCenter = np.array(mesh_pv_all.bounds).reshape(-1,2).mean(axis=1)
        self.ui.roiR.text = round(modelPredCenter[0], 2)
        self.ui.roiA.text = round(modelPredCenter[1], 2)
        self.ui.roiS.text = round(modelPredCenter[2], 2)
        if self._parameterNode.GetNodeReference("roiNode"):
            self._parameterNode.GetNodeReference("roiNode").SetCenter([float(self._parameterNode.GetParameter('roiR')), float(self._parameterNode.GetParameter('roiA')), float(self._parameterNode.GetParameter('roiS'))])

    def onRoiVisibility(self):
        if self.ui.roiVisibility.checked:
            self._parameterNode.GetNodeReference("roiNode").GetDisplayNode().SetVisibility(True)
        else:
            self._parameterNode.GetNodeReference("roiNode").GetDisplayNode().SetVisibility(False)

    def onSetupDataButton(self):
        self.progress_bar_and_run_time.start(maximum=1)
        successful, downloaded_data_dir = RoiPredSetup.RoiPredSetup.relocate_data()
        if not successful:
            slicer.util.messageBox(f'No data folder found at:\n\n{downloaded_data_dir}\n\nPlease first download & extract the .zip from a Github release')

        self.progress_bar_and_run_time.end()

    def updateHeartExpDir(self):
        self._parameterNode.SetParameter("heartExpDir", self.ui.heartExpDir.directory)

    def updateCa2ExpDir(self):
        self._parameterNode.SetParameter("ca2ExpDir", self.ui.ca2ExpDir.directory)

    def onHeartVisibility(self):
        self.defineOutputNamesFromInputSelector()
        folder_exists = helper_lib.set_folder_visibility(self.heartModelFolderName, self.ui.heartVisibility.checked, updateContentsVisibility=True)
        if not folder_exists:
            self.ui.heartVisibility.checked = False
        self.updateVisibilityCheckBoxes() # in case this function affects other visibilities

    def onCa2Visibility(self):
        self.defineOutputNamesFromInputSelector()
        segment_exists = helper_lib.set_segment_visibility(self.segmentName, self.ui.ca2Visibility.checked)
        if not segment_exists:
            self.ui.ca2Visibility.checked = False
        self.updateVisibilityCheckBoxes() # in case this function affects other visibilities

    def onHeartStlVisibility(self):
        self.defineOutputNamesFromInputSelector()
        folder_exists = helper_lib.set_folder_visibility(self.heartStlFolderName, self.ui.heartStlVisibility.checked, updateContentsVisibility=True)
        if not folder_exists:
            self.ui.heartStlVisibility.checked = False
        self.updateVisibilityCheckBoxes() # in case this function affects other visibilities

    def onHeartModelLoadButton(self):
        self.progress_bar_and_run_time.start(maximum=1)

        if self._parameterNode.GetParameter("useGPU") == "true":
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        self.pytorch_model_heart = dcvm.io.load_model(self._parameterNode.GetParameter("heartExpDir"), map_location=map_location)
        self._parameterNode.SetParameter("heartModelLoaded", "true")

        self.progress_bar_and_run_time.end()

    def onCa2ModelLoadButton(self):
        self.progress_bar_and_run_time.start(maximum=1)

        if self._parameterNode.GetParameter("useGPU") == "true":
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        self.pytorch_model_ca2 = dcvm.io.load_model(self._parameterNode.GetParameter("ca2ExpDir"), map_location=map_location)
        self._parameterNode.SetParameter("ca2ModelLoaded", "true")

        self.progress_bar_and_run_time.end()

    def onTemplateLoadButton(self):
        self.progress_bar_and_run_time.start(maximum=1)

        self.heart_verts, self.heart_elems, self.heart_cell_types, self.heart_fiber_ori = torch.load(self._parameterNode.GetParameter("templateFilePath"))
        if isinstance(self.heart_verts, torch.Tensor):
            self.heart_verts = self.heart_verts.squeeze().cpu().numpy()
        if len(self.heart_fiber_ori) == 0:
            self.heart_fiber_ori = None
        self.heart_mesh_keys = list(self.heart_elems.keys())

        self._parameterNode.SetParameter("templateLoaded", "true")
        
        self.progress_bar_and_run_time.end()

    def onUseGpuButton(self):
        # this may not be fully implemented yet
        if self._parameterNode.GetParameter("useGPU") == 'true':
            self.pytorch_model_heart = self.pytorch_model_heart.to(torch.device('cuda'))
            self.pytorch_model_ca2 = self.pytorch_model_ca2.to(torch.device('cuda'))
        else:
            self.pytorch_model_heart = self.pytorch_model_heart.to(torch.device('cpu'))
            self.pytorch_model_ca2 = self.pytorch_model_ca2.to(torch.device('cpu'))

    def onCropAndRunButton(self):
        self.displaySelectedNode(displayOutputs=True)
        self.updateVisibilityCheckBoxes()

        # define relevant inputs
        cropInputNode = self._parameterNode.GetNodeReference("cropInput")
        roiNode = self._parameterNode.GetNodeReference("roiNode")
        spacing = [float(self._parameterNode.GetParameter("spacing"))]*3
        if hasattr(self, "pytorch_model_heart"):
            device = next(self.pytorch_model_heart.parameters()).device
        elif hasattr(self, "pytorch_model_ca2"):
            device = next(self.pytorch_model_ca2.parameters()).device

        # Cropping
        if self._parameterNode.GetParameter("inputSequenceOrVolume") == "false": # single volume image
            cropped_img_torch = roi_pred_lib.run_crop_volume_single(cropInputNode, roiNode, spacing=spacing, device=device) # (n_batch, n_ch, x, y, z)
            cropOutputNode = roi_pred_lib.update_cropOutputNode(cropped_img_torch, self.cropOutputNodeName, roiNode, spacing=spacing)
        else: # sequence of volume images
            cropped_img_torch_list = roi_pred_lib.run_crop_volume_sequence(cropInputNode, roiNode, spacing=spacing, device=device) # list((n_batch, n_ch, x, y, z))
            cropOutputNode = roi_pred_lib.update_cropOutputSequenceNode(cropped_img_torch_list, self.cropOutputSequenceNodeName, roiNode, spacing=spacing)
            # roi_pred_lib.update_cropSequenceBrowserNode(self.cropSequenceBrowserNodeName, cropOutputNode)
        self._parameterNode.SetNodeReferenceID("cropOutput", cropOutputNode.GetID())

        # Start Pytorch Run

        # volume
        if isinstance(cropInputNode, slicer.vtkMRMLScalarVolumeNode):
            self.progress_bar_and_run_time.start(maximum=1)

            # pre-processing img
            pytorch_heart_input_img = roi_pred_lib.preprocess_img(cropped_img_torch, min_bound=-158, max_bound=864)
            pytorch_ca2_input_img = roi_pred_lib.preprocess_img(cropped_img_torch, min_bound=-200, max_bound=1500)

            # run_heart_single only if heart_model and template are loaded
            if (self._parameterNode.GetParameter("heartModelLoaded") == "true") and (self._parameterNode.GetParameter("templateLoaded") == "true"):
                mesh_pv_dict, heart_fiber_ori_transformed = roi_pred_lib.run_heart_single(
                    self.pytorch_model_heart,
                    pytorch_heart_input_img,
                    self.heart_verts, self.heart_elems, self.heart_cell_types, self.heart_fiber_ori,
                    cropOutputNode,
                )
                if heart_fiber_ori_transformed is not None:
                    for key in mesh_pv_dict:
                        mesh_pv_dict[key].cell_data['fiber_ori'] = heart_fiber_ori_transformed[key]
            # run_ca2_single only if ca2_model is loaded
            if self._parameterNode.GetParameter("ca2ModelLoaded") == "true":
                ca2_pv, ca2_seg = roi_pred_lib.run_ca2_single(
                    self.pytorch_model_ca2,
                    pytorch_ca2_input_img,
                    cropInputNode,
                    cropOutputNode,
                )
            self.progress_bar_and_run_time.end()

            if (self._parameterNode.GetParameter("heartModelLoaded") == "true") and (self._parameterNode.GetParameter("templateLoaded") == "true"):
                modelNodes_dict = helper_lib.update_model_nodes_from_pv_dict(mesh_pv_dict, self.modelNames_dict)
                helper_lib.update_model_nodes_display(list(modelNodes_dict.values()))
                helper_lib.put_mrmlNodes_in_folder(self.heartModelFolderName, list(self.modelNames_dict.values()))

                self._parameterNode.SetParameter("heartVisibility", "true")
                self._parameterNode.SetParameter("heartRunCompleted", "true")
            if self._parameterNode.GetParameter("ca2ModelLoaded") == "true":
                # helper_lib.update_model_nodes_from_pv({'ca2': ca2_pv}, {'ca2': 'test_ca2'}) # for debugging. pv-->model conversion implemented before array-->segment
                segmentationNode = helper_lib.update_seg_node_from_np(ca2_seg, self.segmentationNodeName, self.segmentName, cropInputNode)
                helper_lib.update_seg_node_display(segmentationNode, [self.segmentName])
                self._parameterNode.SetParameter("ca2Visibility", "true")
                self._parameterNode.SetParameter("ca2RunCompleted", "true")

            slicer.util.setSliceViewerLayers(background=cropInputNode)

        # sequence
        elif isinstance(cropInputNode, slicer.vtkMRMLSequenceNode):
            self.progress_bar_and_run_time.start(maximum=len(cropped_img_torch_list))

            # pre-processing imgs
            pytorch_heart_input_img_list = [roi_pred_lib.preprocess_img(img, min_bound=-158, max_bound=864) for img in cropped_img_torch_list]
            pytorch_ca2_input_img_list = [roi_pred_lib.preprocess_img(img, min_bound=-200, max_bound=1500) for img in cropped_img_torch_list]

            mesh_pv_dict_list = []
            ca2_seg_list = []
            for run_idx, (pytorch_heart_input_img, pytorch_ca2_input_img) in enumerate(zip(pytorch_heart_input_img_list, pytorch_ca2_input_img_list)):
                if (self._parameterNode.GetParameter("heartModelLoaded") == "true") and (self._parameterNode.GetParameter("templateLoaded") == "true"):
                    mesh_pv_dict, heart_fiber_ori_transformed = roi_pred_lib.run_heart_single(
                        self.pytorch_model_heart,
                        pytorch_heart_input_img,
                        self.heart_verts, self.heart_elems, self.heart_cell_types, self.heart_fiber_ori,
                        cropOutputNode,
                    )
                    if heart_fiber_ori_transformed is not None:
                        for key in mesh_pv_dict:
                            mesh_pv_dict[key].cell_data['fiber_ori'] = heart_fiber_ori_transformed[key]
                    mesh_pv_dict_list.append(mesh_pv_dict)
                if self._parameterNode.GetParameter("ca2ModelLoaded") == "true":
                    ca2_pv, ca2_seg = roi_pred_lib.run_ca2_single(
                        self.pytorch_model_ca2,
                        pytorch_ca2_input_img,
                        cropInputNode,
                        cropOutputNode,
                    )
                    ca2_seg_list.append(ca2_seg)
                self.progress_bar_and_run_time.step(run_idx+1)

            if len(mesh_pv_dict_list) > 0:
                modelSequenceNodes_dict = helper_lib.update_model_sequence_nodes_from_pv_dict_list(mesh_pv_dict_list, self.modelNames_dict)
                browserNode = helper_lib.update_outputSequenceBrowserNode(self.outputSequenceBrowserNodeName, imgSequenceNode=cropInputNode, modelSequenceNodes_dict=modelSequenceNodes_dict)
                helper_lib.put_mrmlNodes_in_folder(self.heartModelFolderName, list(self.modelNames_dict.values()))
                self._parameterNode.SetParameter("heartVisibility", "true")
                self._parameterNode.SetParameter("heartRunCompleted", "true")
            if len(ca2_seg_list) > 0:
                segSequenceNode = helper_lib.update_seg_sequence_node_from_seg_list(ca2_seg_list, self.segmentationNodeName, self.segmentName, cropInputNode)
                browserNode = helper_lib.update_outputSequenceBrowserNode(self.outputSequenceBrowserNodeName, imgSequenceNode=cropInputNode, segSequenceNode=segSequenceNode)
                self._parameterNode.SetParameter("ca2Visibility", "true")
                self._parameterNode.SetParameter("ca2RunCompleted", "true")

        helper_lib.put_outputs_under_same_subject(self.inputNodeName, self.heartModelFolderName_suffix, self.segmentationNodeName_suffix)
        self.displaySelectedNode(displayOutputs=True)
        self.updateVisibilityCheckBoxes()

    def onSaveOutputsInpButton(self):
        self.defineOutputNamesFromInputSelector()
        inputNode = self.cropInputSelector.currentNode()
        inputNodeName = inputNode.GetName()
        # self.defineOutputNamesFromInputSelector() this is already called when changing the drop-down menu choice

        exp_dir = self._parameterNode.GetParameter("heartExpDir")
        save_dir = os.path.join(exp_dir, 'heart_only_inp')
        os.makedirs(save_dir, exist_ok=True)
        
        if isinstance(inputNode, slicer.vtkMRMLSequenceNode):
            # remove sequenceBrowserNode --> sequenceNodes --> proxyNodes
            modelSequenceNodes_dict = {key: node for key, modelName in self.modelNames_dict.items() for node in [node for node in slicer.util.getNodesByClass('vtkMRMLSequenceNode') if node.GetName() == modelName]}
            if len(modelSequenceNodes_dict) > 0: # to avoid error when model doesn't exist
                modelSequence0 = list(modelSequenceNodes_dict.values())[0]
                self.progress_bar_and_run_time.start(maximum=modelSequence0.GetNumberOfDataNodes())
                for itemIndex in range(modelSequence0.GetNumberOfDataNodes()): # all models should have the same number of data nodes
                    mesh_pv_dict_nth_timepoint = {key: pv.UnstructuredGrid(modelSequenceNode.GetNthDataNode(itemIndex).GetMesh()) for key, modelSequenceNode in modelSequenceNodes_dict.items()}
                    save_filepath = os.path.join(save_dir, '{}.inp'.format(inputNode.GetNthDataNode(itemIndex).GetName())) # define save_filepath
                    verts, _ = dcvm.ops.get_verts_faces_from_pyvista(list(mesh_pv_dict_nth_timepoint.values())[0])
                    elems_dict = {key: dcvm.ops.get_verts_faces_from_pyvista(mesh_pv)[1] for key, mesh_pv in mesh_pv_dict_nth_timepoint.items()} # list for 5 different comps at one time point
                    cell_types_dict = {key: mesh_pv.celltypes for key, mesh_pv in mesh_pv_dict_nth_timepoint.items()} # list for 5 different comps at one time point
                    if 'fiber_ori' in list(mesh_pv_dict_nth_timepoint.values())[0].cell_data.keys():
                        dirs_dict = {key: mesh_pv.cell_data['fiber_ori'] for key, mesh_pv in mesh_pv_dict_nth_timepoint.items()}
                    else:
                        dirs_dict = None
                    dcvm.io.write_inp_file(save_filepath, verts, elems_dict, cell_types_dict, dirs_dict)
                    # print('saved {}/{}'.format(itemIndex+1, modelSequence0.GetNumberOfDataNodes()))
                    self.progress_bar_and_run_time.step(itemIndex+1)
                
        elif isinstance(inputNode, slicer.vtkMRMLScalarVolumeNode):
            modelNodes_dict = {key: node for key, modelName in self.modelNames_dict.items() for node in [node for node in slicer.util.getNodesByClass('vtkMRMLModelNode') if node.GetName() == modelName]}
            if len(modelNodes_dict) > 0: # to avoid error when model doesn't exist
                self.progress_bar_and_run_time.start(maximum=1)
                mesh_pv_dict = {key: pv.UnstructuredGrid(modelNode.GetMesh()) for key, modelNode in modelNodes_dict.items()}
                save_filepath = os.path.join(save_dir, '{}.inp'.format(inputNodeName)) # define save_filepath
                verts, _ = dcvm.ops.get_verts_faces_from_pyvista(list(mesh_pv_dict.values())[0])
                elems_dict = {key: dcvm.ops.get_verts_faces_from_pyvista(mesh_pv)[1] for key, mesh_pv in mesh_pv_dict.items()}
                cell_types_dict = {key: mesh_pv.celltypes for key, mesh_pv in mesh_pv_dict.items()}
                if 'fiber_ori' in list(mesh_pv_dict.values())[0].cell_data.keys():
                    dirs_dict = {key: mesh_pv.cell_data['fiber_ori'] for key, mesh_pv in mesh_pv_dict.items()}
                else:
                    dirs_dict = None
                dcvm.io.write_inp_file(save_filepath, verts, elems_dict, cell_types_dict, dirs_dict)

        self.progress_bar_and_run_time.end()

    def onModelsToStlButton(self):
        self.progress_bar_and_run_time.start()

        self.defineOutputNamesFromInputSelector()
        modelNodes_dict = {key: node for key, modelName in self.modelNames_dict.items() for node in [node for node in slicer.util.getNodesByClass('vtkMRMLModelNode') if node.GetName() == modelName]}
        mesh_pv_dict = {key: pv.UnstructuredGrid(modelNode.GetMesh()) for key, modelNode in modelNodes_dict.items()}

        aorta_lv_pv = roi_pred_lib.get_aorta_lv_stl(mesh_pv_dict, offset_dist=float(self._parameterNode.GetParameter("aortaLvThickness")), spacing_scaling=2, pyacvd_cluster_arg=20000)
        leaflets_pv = roi_pred_lib.get_leaflets_stl(mesh_pv_dict, offset_dist=float(self._parameterNode.GetParameter("leafletThickness"))-0.8, spacing_scaling=4, pyacvd_cluster_arg=5000, fuse_leaflets=self.ui.fuseLeaflets.checked)
        mesh_pv_list = [aorta_lv_pv, leaflets_pv]
        
        modelNodes_dict = helper_lib.update_model_nodes_from_pv({name: mesh_pv for name, mesh_pv in zip(self.stlNames_dict.values(), mesh_pv_list)})
        helper_lib.update_model_nodes_display(list(modelNodes_dict.values()), colors_stl)
        helper_lib.put_mrmlNodes_in_folder(self.heartStlFolderName, list(self.stlNames_dict.values()))

        # turn off original model displays
        folderNames, _, _ = helper_lib.get_all_folders_containing_suffix(self.heartModelFolderName_suffix)
        for folderName in folderNames:
            helper_lib.set_folder_visibility(folderName, False)
        
        # turn on only selected input node's stl model displays
        folderNames, _, _ = helper_lib.get_all_folders_containing_suffix(self.heartStlFolderName_suffix)
        for folderName in folderNames:
            helper_lib.set_folder_visibility(folderName, folderName == self.heartStlFolderName, updateContentsVisibility=True)

        helper_lib.put_outputs_under_same_subject(self.inputNodeName, self.heartStlFolderName)
        self.updateVisibilityCheckBoxes()
        
        self.progress_bar_and_run_time.end()

    def onSaveOutputsStlButton(self):
        print('here')

    def onRemoveOutputNodesButton(self):
        inputNode = self.cropInputSelector.currentNode()
        inputNodeName = inputNode.GetName()
        
        if isinstance(inputNode, slicer.vtkMRMLSequenceNode):
            # remove sequenceBrowserNode --> sequenceNodes --> proxyNodes
            outputSequenceBrowserNodes = [node for node in slicer.util.getNodesByClass('vtkMRMLSequenceBrowserNode') if node.GetName() == self.outputSequenceBrowserNodeName]
            modelSequenceNodes = [node for modelName in self.modelNames_dict.values() for node in [node for node in slicer.util.getNodesByClass('vtkMRMLSequenceNode') if node.GetName() == modelName]]
            modelSequenceProxyNodes = [node for modelName in self.modelNames_dict.values() for node in [node for node in slicer.util.getNodesByClass('vtkMRMLModelNode') if node.GetName() == modelName]]
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
            modelNodes = [node for modelName in self.modelNames_dict.values() for node in [node for node in slicer.util.getNodesByClass('vtkMRMLModelNode') if node.GetName() == modelName]]
            if len(modelNodes) > 0: # to avoid error when model doesn't exist
                for modelNode in modelNodes:
                    slicer.mrmlScene.RemoveNode(modelNode)

        _, folderIds, _ = helper_lib.get_all_folders_containing_suffix(self.heartModelFolderName)
        for folderId in folderIds:
            shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
            shNode.RemoveItem(folderId)
        
        segmentationNodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
        if len(segmentationNodes) > 0:
            segmentationNode = [node for node in segmentationNodes if '{}{}'.format(inputNodeName, self.segmentationNodeName_suffix) in node.GetName()][0]
            slicer.mrmlScene.RemoveNode(segmentationNode)

        self.updateVisibilityCheckBoxes()

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

    def initNamingConvention(self, widgetObject):
        # naming convention: setting it all up in one place
        widgetObject.roiNodeName = "RoiPred_ROI"
        widgetObject.cropOutputNodeName = "RoiPred_crop_output"
        widgetObject.cropOutputSequenceNodeName = "RoiPred_crop_output_seq"
        widgetObject.cropSequenceBrowserNodeName = "RoiPred_crop_sequence_browser"
        widgetObject.outputSequenceBrowserNodeName = "RoiPred_output_sequence_browser"
        
        # modelNames_suffix defined by self.heart_mesh_keys after template load # e.g. {inputVolumeName}_lv, etc.
        widgetObject.heartModelFolderName_suffix = "_heart" # e.g. {inputVolumeName}_heart
        widgetObject.segmentationNodeName_suffix = "_Segmentation" # e.g. {inputVolumeName}_Segmentation
        widgetObject.ca2_segmentName_suffix = "_ca2" # e.g. {inputVolumeName}_ca2
        widgetObject.heartStlFolderName_suffix = "_heart_stl" # e.g. {inputVolumeName}_heart_stl
        widgetObject.heart_stl_keys = ['aorta_lv_stl', 'av_leaflets_stl']

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
        if not parameterNode.GetParameter("templateFilePath"):
            parameterNode.SetParameter("templateFilePath", os.path.join(dcvm_parent_dir, "template_for_deform/combined_v12/combined_v12_inference_all.pt"))
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
        if not parameterNode.GetParameter("aortaLvThickness"):
            parameterNode.SetParameter("aortaLvThickness", "1.5")
        if not parameterNode.GetParameter("leafletThickness"):
            parameterNode.SetParameter("leafletThickness", "0.8")
        if not parameterNode.GetParameter("fuseLeaflets"):
            parameterNode.SetParameter("fuseLeaflets", "false")
        
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
