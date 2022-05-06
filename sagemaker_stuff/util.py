
import os
import shutil

from . import config


def copy_file_to_tar_dir(filepath):
    shutil.copyfile(
        filepath,
        os.path.join(config.SM_MODEL_DIR, os.path.basename(filepath))
    )
