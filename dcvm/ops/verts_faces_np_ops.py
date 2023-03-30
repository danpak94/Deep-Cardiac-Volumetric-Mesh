import numpy as np
import torch

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