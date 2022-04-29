
import shutil

from . import config


def copy_file_to_tar_dir(filepath):
    shutil.copyfile(filepath, config.SM_MODEL_DIR)
