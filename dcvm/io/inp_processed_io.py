import os
import numpy as np
import torch

from dcvm.io.inp_raw_io import load_hypermesh_abaqus_inp_file

def load_common_data_for_all_evals_old(template_dir, template_filename_prefix):
    device = 'cuda'
    template_verts = np.load(os.path.join(template_dir, '{}_laa_verts_1.25scaled.npy'.format(template_filename_prefix)))
    verts_template_torch = torch.tensor(template_verts, dtype=torch.get_default_dtype(), device=device)

    hypermesh_template_filepath = os.path.join(template_dir, '{}.inp'.format(template_filename_prefix))
    laa_verts, laa_elems, laa_cell_types, dirs_dict = load_hypermesh_abaqus_inp_file(hypermesh_template_filepath)

    # before standardizing .inp files (prior to combined_v11d)
    laa_elems['aorta'] = laa_elems['aorta_offset'] + laa_elems['aw_v4_solid']
    laa_cell_types['aorta'] = laa_cell_types['aorta_offset'] + laa_cell_types['aw_v4_solid']
    for key in ['aorta_offset', 'aw_v4_solid']:
        laa_elems.pop(key)
        laa_cell_types.pop(key)
    replace_key_dict = {
        'lv_shorter_v4': 'lv',
        'leaflet1_v4_solid': 'l3',
        'leaflet2_v4_solid': 'l1',
        'leaflet3_v4_solid': 'l2',
        'aorta': 'aorta',
        'mitral_annulus_cover': 'mitral_annulus_cover',
    }
    laa_elems = {replace_key_dict[key]: val for key, val in laa_elems.items() if key in replace_key_dict.keys()}
    laa_cell_types = {replace_key_dict[key]: val for key, val in laa_cell_types.items() if key in replace_key_dict.keys()}

    return verts_template_torch, laa_elems, laa_cell_types

def load_template_inference(template_dir, template_filename_prefix):
    device = 'cuda'
    verts_template_torch, laa_elems, laa_cell_types, laa_faces = torch.load(os.path.join(template_dir, '{}_inference_all.pt'.format(template_filename_prefix)))
    verts_template_torch = verts_template_torch.to(device)

    return verts_template_torch, laa_elems, laa_cell_types, laa_faces