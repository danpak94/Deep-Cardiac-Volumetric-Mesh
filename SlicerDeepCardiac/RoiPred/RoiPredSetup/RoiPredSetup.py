import importlib
import types

import slicer

def install_missing_pkgs_in_slicer():
    """
    https://discourse.slicer.org/t/install-python-library-with-extension/10110
    """

    pkg_and_install = {
        "pyvista": "pyvista",
        "gdown": "gdown",
        "matplotlib": "matplotlib",
        "torch": "torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116",
    }

    for pkg_name, install_command in pkg_and_install.items():
        try:
            importlib.import_module(pkg_name)
        except ModuleNotFoundError as e:
            if slicer.util.confirmOkCancelDisplay("'ROI pred' module requires '{}' Python package. Click OK to install it now.".format(pkg_name)):
                slicer.util.pip_install(install_command)
                importlib.import_module(pkg_name)