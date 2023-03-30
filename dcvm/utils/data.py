import os
import shutil
import gdown

def download_data_and_relocate():
    main_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')
    temp_data_dir = os.path.join(main_dir, 'temp_data_downloaded')

    url = "https://drive.google.com/drive/folders/1F0IJlKhpRw7HwvG26X0fFkS9_VgY9ybB?usp=share_link"
    gdown.download_folder(url, output=temp_data_dir, quiet=False, use_cookies=False)

    downloaded_exps_dir = os.path.join(temp_data_dir, 'experiments')
    downloaded_template_dir = os.path.join(temp_data_dir, 'template_for_deform')
    downloaded_roi_pred_resources_dir = os.path.join(temp_data_dir, 'Slicer_RoiPred_Resources')

    dcvm_exps_dir = os.path.join(main_dir, 'experiments')
    dcvm_template_dir = os.path.join(main_dir, 'template_for_deform')
    dcvm_roi_pred_resources_dir = os.path.join(main_dir, 'SlicerDeepCardiac/RoiPred/Resources')

    move_matching_files(downloaded_exps_dir, dcvm_exps_dir)
    move_matching_files(downloaded_template_dir, dcvm_template_dir)
    move_matching_files(downloaded_roi_pred_resources_dir, dcvm_roi_pred_resources_dir)

    shutil.rmtree(temp_data_dir, ignore_errors=True)

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