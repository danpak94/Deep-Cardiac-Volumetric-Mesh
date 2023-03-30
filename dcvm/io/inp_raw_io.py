import numpy as np
import vtk
from dcvm.ops import replace_face_idxes_with_dict

def load_hypermesh_abaqus_inp_file(filepath, return_verts_key_dict=False):
    store_verts = False
    store_elems = False
    store_orientation = False

    with open(filepath, 'r') as fstream:
        verts_key_dict = {}
        verts_list = []
        elem_id_list = []
        elems_list = []
        cell_types_list = []
        elem_id_dict = {}
        elems_dict = {}
        cell_types_dict = {}
        orientation_dict = {}
        orientation_part_names = []

        for line_idx, line in enumerate(fstream):

            line = line.rstrip('\n')
            line_parse = line.split('*')

            if len(line_parse) > 1:
                if line_parse[1] == 'NODE':
                    store_verts = True
                    count = 0
                    continue
                
                if len(line.split('**')) > 1:
                    if 'HW_COMPONENT' in line_parse[2]:
                        if len(line.split('NAME=')) > 1:
                            elset_new = line.split('NAME=')[1]

                if 'ELEMENT' in line_parse[1]:
                    if len(line.split(',')) > 2:
                        elset_new = line.split(',')[2].split('=')[1]

                    if 'TYPE=S3' in line.split(',')[1]:
                        store_elems = True
                        elem_type = 'tri'
                        continue

                    elif 'TYPE=S4R' in line.split(',')[1] or 'TYPE=S4' in line.split(',')[1]:
                        store_elems = True
                        elem_type = 'quad'
                        continue

                    elif 'TYPE=C3D8R' in line.split(',')[1] or 'TYPE=C3D8' in line.split(',')[1] or 'TYPE=C3D8I' in line.split(',')[1]:
                        store_elems = True
                        elem_type = 'hex'
                        continue

                    elif 'TYPE=C3D4' in line.split(',')[1]:
                        store_elems = True
                        elem_type = 'tet'
                        continue
                
                if 'Distribution,' in line_parse[1]:
                    store_orientation = True
                    ori_name = line.split('name=')[1].split(',')[0]
                    part_name = '_'.join(ori_name.split('_')[1:-1])
                    orientation_part_names.append(part_name)
                    count = 0
                    continue

            if store_verts:
                if len(line_parse) == 1:
                    line_parse2 = line.split(',  ')
                    verts_key_dict[int(line_parse2[0])] = count
                    verts_list.append(line_parse2[1:4])
                    count += 1
                else:
                    store_verts = False

            if store_elems:
                if len(line_parse) == 1:
                    elset = elset_new
                    elem_id = line.split(', ')[0]
                    elem_id_list.append(int(elem_id))

                    if elem_type == 'tri':
                        idxes_str = line.split(', ')[1:]
                        elems_list.append([int(idx) for idx in idxes_str])
                        cell_types_list.append(vtk.VTK_TRIANGLE)
                    elif elem_type == 'quad':
                        idxes_str = line.split(', ')[1:]
                        elems_list.append([int(idx) for idx in idxes_str])
                        cell_types_list.append(vtk.VTK_QUAD)
                    elif elem_type == 'hex':
                        line_parse2 = line.split(', ')
                        if line_parse2[-1][-1] == ',': # for hypermesh-saved files where last elem is written in a separate line
                            line_save = line_parse2[1:] # part of combining this line with next line
                            line_save[-1] = line_save[-1][:-1] # getting rid of comma in last entry
                        if len(line_parse2) == 1: # for hypermesh-saved files where last elem is written in a separate line
                            idxes_str = line_save + [line]
                            idxes_int = [int(idx) for idx in idxes_str]
                            if idxes_int[-1] == idxes_int[-3]:
                                idxes_int = idxes_int[:-3]
                                cell_types_list.append(vtk.VTK_PYRAMID)
                            else:
                                cell_types_list.append(vtk.VTK_HEXAHEDRON)
                            elems_list.append(idxes_int)
                        if len(line_parse2) == 9: # for DP-saved files where all elems are in one line
                            idxes_int = [int(idx) for idx in line_parse2[1:]]
                            if idxes_int[-1] == idxes_int[-3]:
                                idxes_int = idxes_int[:-3]
                                cell_types_list.append(vtk.VTK_PYRAMID)
                            else:
                                cell_types_list.append(vtk.VTK_HEXAHEDRON)
                            elems_list.append(idxes_int)

                    elif elem_type == 'tet':
                        idxes_str = line.split(', ')[1:]
                        elems_list.append([int(idx) for idx in idxes_str])
                        cell_types_list.append(vtk.VTK_TETRA)
                else:
                    store_elems = False
                    if elems_list:
                        elems_list = replace_face_idxes_with_dict(elems_list, verts_key_dict)
                    if not elset in elems_dict.keys():
                        elem_id_dict[elset] = elem_id_list
                        elems_dict[elset] = elems_list
                        cell_types_dict[elset] = cell_types_list
                    else:
                        elem_id_dict[elset] += elem_id_list
                        elems_dict[elset] += elems_list
                        cell_types_dict[elset] += cell_types_list
                    elem_id_list = []
                    elems_list = []
                    cell_types_list = []
            
            if store_orientation:
                if len(line_parse) == 1:
                    line_parse2 = line.split(', ')
                    if line_parse2[0] == '' or len(line_parse2)<7:
                        continue
                    elem_id = int(line_parse2[0])
                    orientation_coords = [float(x) for x in line_parse2[1:]]
                    orientation_dict[elem_id] = [orientation_coords[0:3], orientation_coords[3:]]
                    count += 1
                else:
                    store_orientation = False
    
    verts = np.array(verts_list).astype(float)

    if len(orientation_dict) > 0:
        dirs_dict = {key: np.array([orientation_dict[elem_id][0] + orientation_dict[elem_id][1] for elem_id in elem_id_dict[key]]) for key in orientation_part_names}
    else:
        dirs_dict = {}

    if return_verts_key_dict:
        return verts, elems_dict, cell_types_dict, dirs_dict, verts_key_dict
    else:
        return verts, elems_dict, cell_types_dict, dirs_dict

def write_inp_file(inp_filepath, verts, elems_dict, cell_types_dict, dirs_dict=None):
    # **
    # ** ABAQUS Input Deck Generated by HyperMesh Version  : 2019.1.0.20
    # ** Generated using HyperMesh-Abaqus Template Version : 2019.1.0.20
    # **
    # **   Template:  ABAQUS/STANDARD 3D
    # **
    # *NODE
    #      14666,  -12.71998018721,  -2.082900663117,  4.4336241171974
    #
    # **HWCOLOR COMP         13     7
    # *ELEMENT,TYPE=C3D8R,ELSET=aw-solid
    #      24156,     17076,     17142,     17143,     17068,     26400,     26401,     26402,
    #      26403
    # *****

    # orientation_fixed_lines = [
    #     '*Orientation, name=ElementOrientation, local direction=2, system=RECTANGULAR',
    #     'ElementOrientationDistribution',
    #     '1, 0.',
    #     '1.0, 0.0, 0',
    #     '1.0, 0.0, 0',
    #     '*Distribution, name=ElementOrientationDistribution, location=ELEMENT, Table=ElementOrientationDistributionTable',
    #     ', 1., 0., 0., 0., 1., 0.'
    # ]
    
    # final_fixed_lines = [
    #     '*Distribution table, name=ElementOrientationDistributionTable',
    #     'COORD3D, COORD3D',
    # ]

    color_list = ['13     7','10    21','18    35','32    26','37    24','38    55','39    58','46    2','45    11']

    with open(inp_filepath, 'w') as f:
        f.write('**\n')
        f.write("** Generated by Daniel Pak's utils_sp.write_inp_file code\n")
        f.write('**\n')
        f.write('**   Template:  ABAQUS/STANDARD 3D\n')
        f.write('**\n')
        # f.write('*Part, name=LH\n')
        f.write('*NODE\n')

        for vert_idx, vert in enumerate(verts):
            vert_line = '{}{},  {}{},  {}{},  {}{}\n'.format(\
                            ' '*(10-len(str(vert_idx+1))), vert_idx+1,
                            '{:.10f}'.format(vert[0]), ' '*(20-len('{:.10f}'.format(vert[0]))),
                            '{:.10f}'.format(vert[1]), ' '*(20-len('{:.10f}'.format(vert[1]))),
                            '{:.10f}'.format(vert[2]), ' '*(20-len('{:.10f}'.format(vert[2]))))
            f.write(vert_line)

        start_idx = 0
        for key, color in zip(elems_dict.keys(), color_list):
            elems = elems_dict[key]
            cell_types = cell_types_dict[key]

            unique_cell_types = np.unique(cell_types)
            for cell_type_ref in unique_cell_types:
                elems_consistent_types = [elem for elem, cell_type in zip(elems, cell_types) if cell_type == cell_type_ref]

                f.write('**HWCOLOR COMP         {}\n'.format(color))
                
                if cell_type_ref == vtk.VTK_TRIANGLE:
                    elem_type = 'S3'
                elif cell_type_ref == vtk.VTK_QUAD:
                    elem_type = 'S4'
                elif cell_type_ref == vtk.VTK_TETRA:
                    elem_type = 'C3D4'
                elif cell_type_ref == vtk.VTK_HEXAHEDRON:
                    elem_type = 'C3D8I'
                elif cell_type_ref == vtk.VTK_PYRAMID:
                    elem_type = 'C3D8I'
                else:                
                    raise NotImplementedError('elems must be (linear) hex, tet, pyramid, or tri')
                f.write('*ELEMENT,TYPE={},ELSET={}\n'.format(elem_type, key))

                for face_idx, elem in enumerate(elems_consistent_types):
                    ''' https://stackoverflow.com/questions/2721521/fastest-way-to-generate-delimited-string-from-1d-numpy-array '''
                    if len(elem) == 5: # inp files have this weird thing where last entry is repeated 3 times for pyramid type. Total of 8 vert_idxes for each elem
                        elem = elem + [elem[-1]]*3
                    face_arrstr = np.char.mod('%i', np.array(elem)+1) # generate an array with strings
                    face_str = ",        ".join(face_arrstr) # define delimiter
                    f.write('     {},        {}\n'.format(start_idx+face_idx+1, face_str))
                start_idx += face_idx+1

            # unique_elem_len = np.unique([len(elem) for elem in elems])
            # for elem_len in unique_elem_len:
            #     elems_consistent_len = [elem for elem in elems if len(elem) == elem_len]

            #     f.write('**HWCOLOR COMP         {}\n'.format(color))

                # if elem_len == 8:
                #     elem_type = 'C3D8I' # difference with 'C3D8R'?
                # elif elem_len == 5:
                #     elem_type = 'C3D8I'
                # elif elem_len == 4:
                #     elem_type = 'C3D4'
                # elif elem_len == 3:
                #     elem_type = 'S3'
                # else:
                #     raise NotImplementedError('elems must be (linear) hex, tet, pyramid, or tri')
                # f.write('*ELEMENT,TYPE={},ELSET={}\n'.format(elem_type, key))
                # for face_idx, elem in enumerate(elems_consistent_len):
                #     ''' https://stackoverflow.com/questions/2721521/fastest-way-to-generate-delimited-string-from-1d-numpy-array '''
                #     if elem_len == 5: # inp files have this weird thing where last entry is repeated 3 times for pyramid type. Total of 8 vert_idxes for each elem
                #         elem = elem + [elem[-1]]*3
                #     face_arrstr = np.char.mod('%i', np.array(elem)+1) # generate an array with strings
                #     face_str = ",        ".join(face_arrstr) # define delimiter
                #     f.write('     {},        {}\n'.format(start_idx+face_idx+1, face_str))
                # start_idx += face_idx+1

        f.write('*****\n')

        if dirs_dict is not None:
            # for line in orientation_fixed_lines:
            #     f.write('{}\n'.format(line))
                
            n_elems_list = [len(elems) for elems in elems_dict.values()]
            end_ids = np.cumsum(n_elems_list)
            start_ids = np.array([1] + (1+end_ids[:-1]).tolist())
            
            for key in dirs_dict.keys():
                f.write('*Distribution, name=ori_{}_dist, location=ELEMENT, Table=ori_{}_dist_table\n'.format(key, key))
                f.write(', 1., 0., 0., 0., 1., 0.\n')

                key_idx = list(elems_dict.keys()).index(key)
                
                for elem_count, elem_id in enumerate(range(start_ids[key_idx], end_ids[key_idx]+1)):
                    ab_arrstr = np.char.mod('%f', dirs_dict[key][elem_count])
                    ab_str = ', '.join(ab_arrstr)
                    f.write('{}, {}\n'.format(elem_id, ab_str))

                f.write('*parameter\n')
                f.write(' Pi = 3.141592654\n')
                f.write(' gamma = 6.78\n')
                f.write(' theta1 = gamma\n')
                f.write(' c1 = cos(theta1*Pi/180)\n')
                f.write(' s1 = sin(theta1*Pi/180)\n')
                f.write(' theta2 = theta1-2*gamma\n')
                f.write(' c2 = cos(theta2*Pi/180)\n')
                f.write(' s2 = sin(theta2*Pi/180)\n')
                f.write('**\n')
                f.write('*Orientation, name=ori_{}, local direction=2, system=RECTANGULAR\n'.format(key))
                f.write('ori_{}_dist\n'.format(key))
                f.write('1, 0.\n')
                f.write(' <c1>, <s1>, 0.0\n')
                f.write(' <c2>, <s2>, 0.0\n')
            
            for key in dirs_dict.keys():
                f.write('*Distribution table, name=ori_{}_dist_table\n'.format(key))
                f.write('COORD3D, COORD3D\n')

            # for line in final_fixed_lines:
            #     f.write('{}\n'.format(line))