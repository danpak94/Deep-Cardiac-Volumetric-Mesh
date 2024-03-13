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
import shutil
import importlib
import types

import slicer

def install_missing_pkgs_in_slicer():
    """
    https://discourse.slicer.org/t/install-python-library-with-extension/10110
    """

    pkg_and_install = {
        "pyvista": "pyvista==0.43.3",
        "matplotlib": "matplotlib==3.8.3",
        "pyacvd": "pyacvd==0.2.10",
        "torch": "torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116",
    }

    installed_any_pkg = False
    cancelled_any_installation = False

    for pkg_name, install_command in pkg_and_install.items():
        try:
            importlib.import_module(pkg_name)
        except ModuleNotFoundError as e:
            slicer.util.setPythonConsoleVisible(True)
            popup_message = "'ROI pred' module requires '{}' Python package. Click OK to install it now. \n\n{}".format(pkg_name, install_command)
            if pkg_name == 'torch':
                popup_message = "ATTENTION: this package may take a while to install. " + popup_message

            if slicer.util.confirmOkCancelDisplay(popup_message):
                # with slicer.util.displayPythonShell():
                slicer.util.pip_install(install_command)
                importlib.import_module(pkg_name)
                installed_any_pkg = True
            else:
                print("RoiPred: Finish installation to use the module. (Re-)enter or reload module to trigger installation.")
                cancelled_any_installation = True
                break

    return installed_any_pkg, cancelled_any_installation

def relocate_data():
    dcvm_parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../..'))
    downloaded_data_dir = os.path.join(dcvm_parent_dir, '../dcvm_data_files')
    if not os.path.exists(downloaded_data_dir):
        return False, downloaded_data_dir # not successful

    downloaded_exps_dir = os.path.join(downloaded_data_dir, 'experiments')
    downloaded_template_dir = os.path.join(downloaded_data_dir, 'template_for_deform')
    
    dcvm_exps_dir = os.path.join(dcvm_parent_dir, 'experiments')
    dcvm_template_dir = os.path.join(dcvm_parent_dir, 'template_for_deform')

    move_matching_files(downloaded_exps_dir, dcvm_exps_dir)
    move_matching_files(downloaded_template_dir, dcvm_template_dir)

    shutil.rmtree(downloaded_data_dir, ignore_errors=True)
    print('Deleted: {}'.format(downloaded_data_dir))
    print(' ')

    return True, downloaded_data_dir # successful

def move_matching_files(dirA, dirB):
    for root, dirs, files in os.walk(dirA):
        for dir_name in dirs:
            source_dir = os.path.join(root, dir_name)
            dest_dir = os.path.join(dirB, dir_name)
            os.makedirs(dest_dir, exist_ok=True)
            # if os.path.exists(dest_dir):
            for file_name in os.listdir(source_dir):
                source_file = os.path.join(source_dir, file_name)
                dest_file = os.path.join(dest_dir, file_name)
                if os.path.isfile(source_file):
                    shutil.move(source_file, dest_file)
                    print('src: {}'.format(source_file))
                    print('dst: {}'.format(dest_file))
                    print(' ')