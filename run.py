
"""
    model evaluation job entry
"""

# stdlib imports
import os

# pip imports
import numpy as np
import torch

# proprietary imports
from aws_utils.s3_sagemaker_utils import S3SageMakerUtils
from dataset import Dataset
from filtration import FilterHSV
from model_manager import ModelManager
from model_manager_for_web_app import ModelManager as ModelManagerForWebApp
from unified_image_reader import Image

# local imports
import sagemaker_stuff  # TODO - make job configurable from sagemaker notebook
from model_analysis.by_region_analysis_job import WeightRatioAnalysisJob


def initialize_dataset(filtration_cache_filepath=None):
    dataset = Dataset(
        data_dir=sagemaker_stuff.config.SM_CHANNEL_TRAIN,
        labels=sagemaker_stuff.config.SM_CHANNEL_TRAIN,
        filtration=FilterHSV()
    )
    return dataset


def do_analysis_job_preprocessing(dataset, callback, loadingbars: bool = False):
    job = WeightRatioAnalysisJob(
        dataset=dataset,
        region_prediction_callback=callback
    )
    job.get_region_predictions_from_model(loadingbars=loadingbars)
    return job


def announce(*args, **kwargs):
    for arg in args:
        print(f"\n==> {arg}\n", **kwargs)


def main():
    """ main """

    # change model_name and model_region_prediction_callback as necessary
    announce("Loading Model")
    model_name = os.environ["MODEL_NAME"]
    #mm = ModelManager(os.path.join(os.path.dirname(__file__), "models"))
    mm = ModelManagerForWebApp()
    m = mm.load_model(model_name)
    model_region_prediction_callback = m.model.diagnose_region

    try:
        announce("Downloading Filtration Cache")
        aws_session = S3SageMakerUtils()
        aws_session.download_data('.', 'digpath-cache',
                                  'kevin_supervised/filtration_cache.h5')
    except:
        announce("Downloading Filtration Cache Failed")
    announce("Initializing Dataset")
    dataset = initialize_dataset()

    announce("Get Analysis Job Region Predictions")
    analysis_job = do_analysis_job_preprocessing(
        dataset, model_region_prediction_callback, loadingbars=True)

    announce("Saving Region Predictions to Output")
    sagemaker_stuff.util.copy_file_to_tar_dir(
        analysis_job.region_predictions_store.filename)

    # announce("Do Model Analysis")
    # analysis_job.do_analysis()


if __name__ == "__main__":
    main()
