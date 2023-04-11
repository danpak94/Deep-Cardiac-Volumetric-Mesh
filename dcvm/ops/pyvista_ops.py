import numpy as np
import pyvista as pv
import vtk
from vtk.util import numpy_support

def mesh_to_PolyData(verts, faces):
    '''
    verts: np.ndarray
    faces: np.ndarray [n_faces, n_verts_per_face]
    OR
    faces: list of lists [ [0,1,2], [2,3,4,5], [2,3,4], ... ]
    '''
    if isinstance(faces, np.ndarray):
        faces_pv = np.hstack(np.concatenate([faces.shape[1]*np.ones([faces.shape[0],1]), faces], axis=1)).astype(int)
    elif isinstance(faces, list):
        """ convert for pyvista, also for vtk """
        faces_pv = []
        for face in faces:
            faces_pv += ([len(face)] + face)
        faces_pv = np.array(faces_pv).astype(int)

    if faces_pv[0] == 2:
        mesh_pv = pv.PolyData(verts)
        mesh_pv.lines = faces_pv
        return mesh_pv
    else:
        return pv.PolyData(verts, faces_pv)

def mesh_to_UnstructuredGrid(verts, faces, cell_types=None):
    """ only compatible with tetrahedron, hexahedron (8 verts, flexible rectangular prism) and wedge (6 verts, flexible triangular prism) for now """
    faces_pv = []
    if cell_types is None:
        initial_cell_types = None
        cell_types = []
    else:
        initial_cell_types = cell_types

    for face in faces:
        faces_pv += ([len(face)] + list(face))
        if initial_cell_types is None:
            if len(face) == 4:
                cell_types.append(vtk.VTK_TETRA)
            elif len(face) == 6:
                cell_types.append(vtk.VTK_WEDGE)
            elif len(face) == 8:
                cell_types.append(vtk.VTK_HEXAHEDRON)
            elif len(face) == 15:
                cell_types.append(vtk.VTK_QUADRATIC_WEDGE)
            elif len(face) == 20:
                cell_types.append(vtk.VTK_QUADRATIC_HEXAHEDRON)
            elif len(face) == 5:
                cell_types.append(vtk.VTK_PYRAMID)
            else:
                raise ValueError('faces must have either 4, 5, 6, or 8 indices for each cell')
            
    faces_pv = np.array(faces_pv).astype(int)
    cell_types = np.array(cell_types).astype(np.uint8)

    return pv.UnstructuredGrid(faces_pv, cell_types, verts)

def get_verts_faces_from_pyvista(mesh_pv):
    verts = mesh_pv.points

    if isinstance(mesh_pv, pv.core.pointset.PolyData):
        faces_pv = mesh_pv.faces
    elif isinstance(mesh_pv, pv.core.pointset.UnstructuredGrid):
        faces_pv = mesh_pv.cells

    """ faces_pv is in vtk format
    e.g. [3,100,101,102,4,202,203,204,205] is tri element [100,101,102] and quad element [202,203,204,205]
    """
    n_verts_0th = faces_pv[0]
    if len(np.unique(faces_pv[::n_verts_0th+1]))==1:
        faces = faces_pv.reshape(-1,n_verts_0th+1)[:,1:]
    else:
        faces = []
        is_n_verts = True
        for val in faces_pv:
            if is_n_verts:
                n_verts_remain = val
                face = []
                is_n_verts = False
                continue

            if n_verts_remain > 0:
                face.append(val)
                n_verts_remain -= 1

            if n_verts_remain == 0:
                faces.append(face)
                is_n_verts = True

    return verts, faces

def unstructured_to_polydata(unstructured_pv):
    return mesh_to_PolyData(*get_verts_faces_from_pyvista(unstructured_pv))

def seg_to_polydata(ca2_seg, isolevel=0.5, dims=None, spacing=[1,1,1], origin=[0,0,0], smooth=True, volume_threshold=0.0, subdivide=0, contour_method='contour'):
    if ca2_seg.sum() == 0: # check seg
        return None
    
    if dims is None:
        dims = ca2_seg.shape
    ca2_seg_pv = pv.UniformGrid(dimensions=dims, spacing=spacing, origin=origin)
    ca2_seg_pv.point_data['values'] = ca2_seg.flatten(order='F')
    ca2_pv = ca2_seg_pv.contour([isolevel], method=contour_method)
    if volume_threshold != 0:
        bodies_orig = ca2_pv.split_bodies()
        multi_bodies = pv.MultiBlock([body for body in bodies_orig if unstructured_to_polydata(body).volume>volume_threshold]) # filter ca2 chunks by volume
        if len(multi_bodies) > 0: # check after volume threshold
            ca2_pv = mesh_to_PolyData(*get_verts_faces_from_pyvista(multi_bodies.combine()))
        else:
            return None
    
    ca2_pv.clear_data()
    if smooth:
        ca2_pv = ca2_pv.smooth(n_iter=5, relaxation_factor=0.01) # smooth ca2 chunks
    if subdivide > 0:
        ca2_pv = ca2_pv.subdivide(subdivide) # makes ca2_smooth higher resolution, much better when removing nodes

    return ca2_pv

def polydata_to_seg(mesh_pv, dims=None, spacing=[1,1,1], origin=[0,0,0], tolerance=0.0, return_dims_spacing_origin=False):
    '''
    NOTE: Use uniform dims and spacing.. idk why, but non-uniform dims mess everything up for the output
    '''
    if dims is None:
        # determine smallest possible UniformGrid without cutting off mesh_pv
        max_len_dim = np.ceil(np.array([
            mesh_pv.bounds[1]-mesh_pv.bounds[0],
            mesh_pv.bounds[3]-mesh_pv.bounds[2],
            mesh_pv.bounds[5]-mesh_pv.bounds[4]
        ])/np.array(spacing)).astype(int).max()
        dims = [max_len_dim + int(20/min(spacing))]*3 # for uniform dimensions.. idk why, but non-uniform dimensions mess everything up for the output

        origin = np.floor(np.array(mesh_pv.bounds)[[0,2,4]]) - 3

    img_background = pv.UniformGrid(dimensions=dims, spacing=spacing, origin=origin)
    img_background.point_data['values'] = np.ones(dims).flatten()

    stencil_filter = vtk.vtkPolyDataToImageStencil()
    stencil_filter.SetInputData(mesh_pv)
    stencil_filter.SetOutputOrigin(origin)
    stencil_filter.SetOutputSpacing(spacing)
    stencil_filter.SetOutputWholeExtent([0, dims[0]-1, 0, dims[1]-1, 0, dims[2]-1])
    stencil_filter.SetTolerance(tolerance)
    stencil_filter.Update()

    occ_grid = vtk.vtkImageStencil()
    occ_grid.SetInputData(img_background)
    occ_grid.SetStencilConnection(stencil_filter.GetOutputPort())
    occ_grid.ReverseStencilOff()
    occ_grid.SetBackgroundValue(0)
    occ_grid.Update()

    seg = vtk.util.numpy_support.vtk_to_numpy(occ_grid.GetOutput().GetPointData().GetScalars()).reshape(dims[::-1])
    seg = seg.transpose(2,1,0)

    if return_dims_spacing_origin:
        return seg, [dims, spacing, origin]
    else:
        return seg