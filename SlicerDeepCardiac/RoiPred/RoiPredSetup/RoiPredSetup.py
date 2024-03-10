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