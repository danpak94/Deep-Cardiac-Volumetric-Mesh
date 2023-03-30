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