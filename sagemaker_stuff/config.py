
import os
import torch


def process_environment_variable(environment_variable_name, desired_type=str, default=None):
    environment_variable_value = os.getenv(environment_variable_name)
    if environment_variable_value is None:
        return default

    def processing_failed(msg=None, print_instead=False):
        error_message = f"Environment Variable Processing Failed:\n\t{environment_variable_name=}\n\t{environment_variable_value=}\n\t{default=}\n\t{desired_type=}\n\t{msg=}"
        if print_instead:
            print(error_message)
        else:
            raise Exception(error_message)
    if desired_type is str:
        return environment_variable_value
    elif desired_type is bool:
        environment_variable_value = environment_variable_value.lower()
        strings_bool_true = ['true', 't', '1', 'y', 'yes']
        strings_bool_false = ['false', 'f', '0', 'n', 'no']
        if environment_variable_value in strings_bool_true:
            return True
        elif environment_variable_value in strings_bool_false:
            return False
        else:
            processing_failed(
                f"invalid bool representation - must be among {strings_bool_true+strings_bool_false}")
    elif desired_type is int:
        try:
            return int(environment_variable_value)
        except Exception:
            processing_failed()
    else:
        processing_failed(print_instead=True)
        raise NotImplementedError()


# Local Testing Flag
IS_TESTING_LOCALLY = __file__ == "/workspaces/dev-container/ML-Unsupervised/sagemaker_stuff/config.py"

# Training Job Information
UNIQUE_IMAGE_IDENTIFIER = process_environment_variable(
    "UNIQUE_IMAGE_IDENTIFIER")
DATA_AND_MODEL_DEVICE = process_environment_variable(
    "DATA_AND_MODEL_DEVICE", default=('cuda' if torch.cuda.is_available() else 'cpu'))

# Model Parameters
MODEL_NAME = process_environment_variable("MODEL_NAME")

# Data Handling
BATCH_SIZE = process_environment_variable(
    "BATCH_SIZE", desired_type=int, default=2)

# Display Options
ANNOUNCEMENT_PREFIX = process_environment_variable(
    "ANNOUNCEMENT_PREFIX", default="-->")
LOADING_BARS = process_environment_variable(
    "LOADING_BARS", desired_type=bool, default=IS_TESTING_LOCALLY)
UPDATE_INTERVAL = process_environment_variable(
    "UPDATE_INTERVAL", desired_type=int, default=4)

# Directory Locations
DIR_DATA_TRAIN = process_environment_variable(
    "SM_CHANNEL_TRAIN",
    default="/workspaces/dev-container/testing/data/whole_slide_images/train")
DIR_DATA_TRAIN_SUBDIR = process_environment_variable(
    "DIR_DATA_TRAIN_SUBDIR", default="")
DIR_DATA_TEST = process_environment_variable(
    "SM_CHANNEL_TEST")
DIR_OUTPUT = process_environment_variable(
    "SM_MODEL_DIR",
    default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output'))
DIR_CHECKPOINT = process_environment_variable(
    "SM_CHECKPOINT_DIR")

# Dataset Options
DATASET_SAVE_LABELS = process_environment_variable(
    "DATASET_SAVE_LABELS", desired_type=bool, default=True)
DATASET_SAVE_LABELS_FILENAME = process_environment_variable(
    "DATASET_SAVE_LABELS_FILENAME", default="dataset_labels.json")
DATASET_SAVE_LABELS_IMAGE_BASENAME = process_environment_variable(
    "DATASET_SAVE_LABELS_IMAGE_BASENAME", desired_type=bool, default=False)

# Filtration Cacheing
FILTRATION_CACHE_DOWNLOAD = process_environment_variable(
    "FILTRATION_CACHE_DOWNLOAD", desired_type=bool, default=False)
FILTRATION_CACHE_UPLOAD = process_environment_variable(
    "FILTRATION_CACHE_UPLOAD", desired_type=bool, default=False)

# Region Predictions Intermediate Saving
GET_REGION_PREDICTIONS_SAVE_FREQUENCY = process_environment_variable(
    "GET_REGION_PREDICTION_SAVE_FREQUENCY", desired_type=int, default=10)
