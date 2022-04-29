
"""
    model evaluation job entry
"""

# stdlib imports
import os

# pip imports
import numpy as np
import torch

# proprietary imports
from dataset import Dataset
from filtration import FilterHSV
from model_manager import ModelManager
from unified_image_reader import Image

# local imports
import sagemaker_stuff  # TODO - make job configurable from sagemaker notebook
from model_analysis.by_region_analysis_job import WeightRatioAnalysisJob


def initialize_dataset():
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


def main():
    """ main """

    filepath = "/opt/ml/input/data//train/Mild/84204T_001.tif"
    img = Image(filepath)
    print(img.number_of_regions())
    print(img.dims)

    filepath = 'temp.txt'
    with open(filepath, 'x') as f:
        f.write('testyboi')
    sagemaker_stuff.util.copy_file_to_tar_dir(filepath)

    return

    # load model
    mm = ModelManager(os.path.join(os.path.dirname(__file__), "models"))
    # throws error trying to import model_manager_for_webapp???
    m = mm.load_model('kevin_initial')

    # initialize dataset and evaluation job
    dataset = initialize_dataset()
    analysis_job = do_analysis_job_preprocessing(
        dataset, m.model.diagnose_region, loadingbars=True)

    # save preprocessing file information
    sagemaker_stuff.util.copy_file_to_tar_dir(
        analysis_job.region_predictions_store.filename)

    # then perform analysis
    # analysis_job.do_analysis()


if __name__ == "__main__":
    main()
