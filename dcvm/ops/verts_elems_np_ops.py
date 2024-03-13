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
import torch
from dcvm.utils import flatten_list_of_lists, unflatten_list_of_lists
from dcvm.ops.pyvista_ops import get_edges_from_pv

def get_elems_in_orig_idxes(faces, original_vert_idxes):
    """
    use this when we have original_vert_idxes where: verts[original_vert_idxes] = new_verts

    faces: list of lists or np.ndarray, cleaned up faces (max(faces)<=max(original_vert_idxes))
    original_vert_idxes: np.ndarray,
    
    for idx, orig_vert_idx enumerate(original_vert_idxes):
        replace faces==idx with orig_vert_idx
    """
    if isinstance(faces, list):
        faces_flat_list, n_verts = flatten_list_of_lists(faces, return_len_each_sublist=True)
        faces_orig_idxes_flat = np.take(original_vert_idxes, np.array(faces_flat_list))
        faces_orig_idxes = unflatten_list_of_lists(faces_orig_idxes_flat.tolist(), n_verts)

    elif isinstance(faces, np.ndarray):
        faces_orig_idxes = np.take(original_vert_idxes, faces, axis=0)

    return faces_orig_idxes

# def get_elems_in_new_idxes(faces, new_vert_idxes):
#     """    
#     use this when we have original_vert_idxes[orig_vert_idx] = new_vert_idx
#     """
#     conversion_arr = -1*np.ones(len(np.unique(new_vert_idxes)))
#     for idx, new_vert_idx in enumerate(new_vert_idxes):
#         conversion_arr[orig_vert_idx] = idx
    

def replace_face_idxes_with_dict(faces, replace_dict):
    if isinstance(faces, list):
        faces_new = []
        for face in faces:
            new_face = [replace_dict[orig_idx] for orig_idx in face]
            faces_new.append(new_face)
    elif isinstance(faces, torch.Tensor):
        faces_new = faces.clone()
        for key, val in replace_dict.items():
            faces_new[faces==key] = int(val)
    elif isinstance(faces, np.ndarray):
        faces_new = faces.copy()
        for key, val in replace_dict.items():
            faces_new[faces==key] = int(val)
    return faces_new

def remove_unused_verts(verts, elems):
    used_vert_idxes = np.unique([vert_idx for elem in elems for vert_idx in elem])
    replace_dict = {used_idx:count for count, used_idx in enumerate(used_vert_idxes)}
    new_verts = verts[used_vert_idxes]
    new_elems = replace_face_idxes_with_dict(elems, replace_dict)
    return new_verts, new_elems

def remove_verts(faces, remove_idxes):
    if isinstance(faces, np.ndarray):
        new_faces = faces[np.logical_not(np.any(np.isin(faces, remove_idxes), axis=1))]
    elif isinstance(faces, list):
        new_faces = [face for face in faces if not np.any(np.isin(face, remove_idxes))]
    
    return new_faces

def get_border_vert_idxes_in_order(exterior_edges, start_idx, reverse=False, max_iter=1000):
    """
    start_idx: should be a corner vertex idx

    this function only works if (1) the orientation of edge is the same, (2) border forms  a loop
    """
    if not reverse:
        init_axis = 1
        step_axis = 0
    else:
        init_axis = 0
        step_axis = 1

    exterior_edges = np.array(exterior_edges)
    # exterior_edges = igl.exterior_edges(faces)

    idx_store = [start_idx]
    curr_idx = start_idx
    n_iter = 0
    next_idx = np.nan

    while (next_idx != start_idx) and n_iter<=max_iter:
        bool_curr_idx = exterior_edges[:, init_axis] == curr_idx
        edge_with_curr_idx = exterior_edges[bool_curr_idx]
        edge_other_vert_idx = edge_with_curr_idx[:, step_axis][0]

        next_idx = edge_other_vert_idx
        idx_store.append(next_idx)
        curr_idx = next_idx

        n_iter += 1

    if n_iter >= max_iter:
        print('get_border_vert_idxes_in_order n_iter >= {}'.format(max_iter))

    return np.array(idx_store)

def find_neighbor_nodes(faces, node_idx):
    """Pass the index of the node in question.
    Returns the vertex indices of the vertices connected with that node."""
    neighbor_faces = np.vstack([face for face in faces if node_idx in face])
    connected = np.unique(neighbor_faces.ravel()) # get unqiue node indices
    return np.delete(connected, np.argwhere(connected == node_idx)) # delete original node_idx

def find_neighbor_vert_idxes(vert_idxes, mesh_pv):
    all_edges = np.array(get_edges_from_pv(mesh_pv.extract_all_edges(use_all_points=True)))

    neighbor_edges = all_edges[np.any(np.isin(all_edges, vert_idxes), axis=1)]
    neighbor_plus_self_vert_idxes = np.unique(neighbor_edges.reshape(-1))
    return np.array(list((set(neighbor_plus_self_vert_idxes) - set(vert_idxes))))

def get_elem_centers(verts, elems):
    if len(np.unique([len(elem) for elem in elems])) == 1:
        centers = verts[np.array(elems)].mean(axis=1)
    else:
        centers = np.array([verts[elem].mean(axis=0) for elem in elems])
    return centers