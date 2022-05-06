
"""
    model evaluation job entry
"""

# stdlib imports
import os
import time

# pip imports
import cv2 as cv
import numpy as np
import torch

# proprietary imports
from aws_utils.s3_sagemaker_utils import S3SageMakerUtils
from dataset import Dataset
from filtration import FilterManager, FilterBlackAndWhite, FilterHSV, FilterFocusMeasure
from model_manager import ModelManager
from model_manager_for_web_app import ModelManager as ModelManagerForWebApp
from unified_image_reader import Image

# local imports
import sagemaker_stuff  # TODO - make job configurable from sagemaker notebook
from model_analysis.by_region_analysis_job import WeightRatioAnalysisJob


def announce(*args, **kwargs):
    print(f"\n{kwargs.get('prefix','')}-->", *args, "\n", **kwargs)


def initialize_dataset():
    try:
        announce("Downloading Filtration Cache")
        aws_session = S3SageMakerUtils()
        aws_session.download_data('.', 'digpath-cache',
                                  'kevin_supervised/filtration_cache.h5')
        announce("Downloading Filtration Cache --> Success")
    except:
        announce("Downloading Filtration Cache --> Failure")
    dataset = Dataset(
        data_dir=sagemaker_stuff.config.SM_CHANNEL_TRAIN,
        labels=sagemaker_stuff.config.SM_CHANNEL_TRAIN,
        filtration=FilterManager([
            FilterBlackAndWhite(),
            FilterHSV(),
            FilterFocusMeasure()
        ])
    )
    # also save the filtration cache to output dir
    sagemaker_stuff.util.copy_file_to_tar_dir(
        dataset.filtration_cache.h5filepath)
    return dataset


def get_model_callback():
    #model_name = os.environ["MODEL_NAME"]
    ##mm = ModelManager(os.path.join(os.path.dirname(__file__), "models"))
    #mm = ModelManagerForWebApp()
    #m = mm.load_model(model_name)
    # return m.daignose_region
    import torch
    import torchvision
    from lazy_model import my_model, utils
    model_densenet = torchvision.models.DenseNet(
        growth_rate=32,
        block_config=(2, 2, 2, 2),
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        num_classes=3
    )
    model_mymodel = my_model.MyModel(
        model=model_densenet,
        loss_fn=None,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        checkpoint_dir=None,
        optimizer=None,
        model_dir=None
    )
    model_path = os.path.join(os.path.dirname(
        __file__), "lazy_model/model.pth")
    model_mymodel.load_model(model_path)
    return model_mymodel.diagnose_region


def do_analysis_job_preprocessing(dataset, callback, loadingbars: bool = False, loadingbars_ascii=False):
    job = WeightRatioAnalysisJob(
        dataset=dataset,
        region_prediction_callback=callback
    )
    job.get_region_predictions_from_model(
        loadingbars=loadingbars, loadingbars_ascii=loadingbars_ascii)
    return job


def main():
    """ main """

    announce("Loading Model")
    model_callback = get_model_callback()

    announce("Initializing Dataset")
    dataset = initialize_dataset()

    announce("Testing Model Functionality")
    img_path = dataset._filepaths[0]
    img = Image(img_path)
    announce(
        f"Testing Model Functionality --> {img_path}, {img.number_of_regions()}")
    region = img.get_region(0)
    region_prediction = model_callback(region)
    announce(f"Testing Model Functionality --> {region_prediction}")

    announce("Get Analysis Job Region Predictions")
    analysis_job = do_analysis_job_preprocessing(
        dataset, model_callback, loadingbars=False, loadingbars_ascii=True)

    announce("Saving Region Predictions to Output")
    sagemaker_stuff.util.copy_file_to_tar_dir(
        analysis_job.region_predictions_store.filename)

    return

    announce("Do Model Analysis")
    analysis_job.do_analysis()


if __name__ == "__main__":
    main()
