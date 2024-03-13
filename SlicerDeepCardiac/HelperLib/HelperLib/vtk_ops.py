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

import numpy as np
import vtk
import slicer

def get_dims_spacing_origin_from_vtkImageData(geometryImageData):
    dims = geometryImageData.GetDimensions()
    spacing = geometryImageData.GetSpacing()
    origin = geometryImageData.GetOrigin()
    return dims, spacing, origin

def set_dims_spacing_origin_for_vtkImageData(imageData, dims, spacing, origin):
    imageData.SetDimensions(*dims)
    imageData.SetSpacing(*spacing)
    imageData.SetOrigin(*origin)

def get_vtkImageData_from_np(img_np, geometryImageData=None):
    if geometryImageData:
        imageData = slicer.vtkOrientedImageData()
        imageData.DeepCopy(geometryImageData)
    else:
        imageData = slicer.vtkOrientedImageData()
        imageData.SetDimensions(img_np.shape)
        imageData.SetSpacing([1,1,1])
        imageData.SetOrigin([0,0,0])
    imageData.AllocateScalars(vtk.VTK_FLOAT, 1)
    val_pointer = vtk.util.numpy_support.vtk_to_numpy(imageData.GetPointData().GetScalars()).reshape(imageData.GetDimensions()[::-1])
    val_pointer[:] = np.transpose(img_np, [2,1,0])
    return imageData