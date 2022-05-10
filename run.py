
"""
    model evaluation job entry
"""

# stdlib imports
import json
import os

# pip imports
import cv2 as cv
import numpy as np
import torch

# proprietary imports
from aws_utils.s3_sagemaker_utils import S3SageMakerUtils
from dataset import Dataset, LabelManager
from filtration import FilterManager, FilterBlackAndWhite, FilterHSV, FilterFocusMeasure
from model_manager import ModelManager
from model_manager_for_web_app import ModelManager as ModelManagerForWebApp
from unified_image_reader import Image

# local imports
import sagemaker_stuff  # TODO - make job configurable from sagemaker notebook
from model_analysis.by_region_analysis_job import WeightRatioAnalysisJob
from model_analysis.region_predictions_store import AggregatedRegionPredictionsStore
from util import open_file


def announce(*args, **kwargs):
    print(f" \n{kwargs.get('prefix','')}{sagemaker_stuff.config.ANNOUNCEMENT_PREFIX}",
          *args, " \n", **kwargs)


def announce_testing_status():
    if sagemaker_stuff.config.IS_TESTING_LOCALLY:
        announce(f"Test Run!")


def initialize_dataset():
    if sagemaker_stuff.config.FILTRATION_CACHE_DOWNLOAD:
        try:
            announce("Downloading Filtration Cache")
            aws_session = S3SageMakerUtils()
            aws_session.download_data('.', 'digpath-cache',
                                      'kevin_supervised/filtration_cache.h5')
            announce("Downloading Filtration Cache --> Success")
        except:
            announce("Downloading Filtration Cache --> Failure")
    dataset = Dataset(
        data_dir=sagemaker_stuff.config.DIR_DATA_TRAIN,
        labels=LabelManager(
            path=sagemaker_stuff.config.DIR_DATA_TRAIN,
            label_postprocessor=os.path.basename
        ),
        filtration=FilterManager([
            FilterBlackAndWhite(),
            FilterHSV(),
            FilterFocusMeasure()
        ])
    )
    # also save the filtration cache to output dir
    if sagemaker_stuff.config.FILTRATION_CACHE_UPLOAD:
        sagemaker_stuff.util.copy_file_to_tar_dir(
            dataset.filtration_cache.h5filepath)
    # also save dataset labels
    if sagemaker_stuff.config.DATASET_SAVE_LABELS:
        save_labels_from_dataset(dataset)
    return dataset


def get_model_callback():
    #model_name = os.environ["MODEL_NAME"]
    ##mm = ModelManager(os.path.join(os.path.dirname(__file__), "models"))
    #mm = ModelManagerForWebApp()
    #m = mm.load_model(model_name)
    # return m.daignose_region
    import torch
    import torchvision
    from lazy_model import my_model
    model_densenet = torchvision.models.DenseNet(
        growth_rate=32,
        block_config=(2, 2, 2, 2),
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        num_classes=3
    ).to(torch.device(sagemaker_stuff.config.DATA_AND_MODEL_DEVICE))
    model_densenet.eval()
    model_mymodel = my_model.MyModel(
        model=model_densenet,
        loss_fn=None,
        device=torch.device(sagemaker_stuff.config.DATA_AND_MODEL_DEVICE),
        checkpoint_dir=None,
        optimizer=None,
        model_dir=None
    )
    checkpoint_path = os.path.join(os.path.dirname(
        __file__), "lazy_model/model_5_epochs.pth")  # Kevin's discord message from 5/7 11:17pm
    model_mymodel.load_checkpoint(checkpoint_path, eval_only=True)
    return model_mymodel.diagnose_region


def do_analysis_job_preprocessing(dataset, callback):
    job = WeightRatioAnalysisJob(
        dataset=dataset,
        region_prediction_callback=callback,
        region_predictions_filename=os.path.join(
            sagemaker_stuff.config.DIR_OUTPUT, "region_predictions_aggregated.json")
    )
    job.get_region_predictions_from_model(
        loadingbars=sagemaker_stuff.config.LOADING_BARS,
        save_frequency=sagemaker_stuff.config.GET_REGION_PREDICTIONS_SAVE_FREQUENCY,
        tensor_size=sagemaker_stuff.config.BATCH_SIZE,
        tensor_device=torch.device(sagemaker_stuff.config.DATA_AND_MODEL_DEVICE))
    return job


def save_labels_from_dataset(dataset):
    if not sagemaker_stuff.config.DATASET_SAVE_LABELS:
        return
    labels = {}
    for filepath in dataset._filepaths:
        filepath_key = os.path.basename(
            filepath) if sagemaker_stuff.config.DATASET_SAVE_LABELS_IMAGE_BASENAME else filepath
        labels[filepath_key] = dataset.get_label(filepath)
    labels_path = os.path.join(sagemaker_stuff.config.DIR_OUTPUT,
                               sagemaker_stuff.config.DATASET_SAVE_LABELS_FILENAME)
    with open_file(labels_path) as f:
        json.dump(labels, f)


def main():
    """ main """

    announce_testing_status()

    announce("Loading Model")
    model_callback = get_model_callback()

    announce("Initializing Dataset")
    dataset = initialize_dataset()

    announce("Get Analysis Job Region Predictions")
    analysis_job = do_analysis_job_preprocessing(dataset, model_callback)

    announce("Done!")


if __name__ == "__main__":
    main()
