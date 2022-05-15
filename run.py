
"""
    model evaluation job entry
"""

import json
from multiprocessing import Pool, Process
import multiprocessing
import os
import time

from histolab.tile import Tile
from histolab.scorer import NucleiScorer
import PIL
import torch
from tqdm import tqdm as loadingbar

from run_functions import announce, announce_testing_status, pull_dataset_filtraton_cache, pull_tiles_dataset_scoring_data
import sagemaker_stuff
import util

from testing.deit.main import do_train


def main():
    """ main """

    announce_testing_status()

    if not sagemaker_stuff.config.IS_TESTING_LOCALLY:
        announce("Pulling Filtration Cache")
        pull_dataset_filtraton_cache()

        announce("Pulling Tile Scoring Data")
        pull_tiles_dataset_scoring_data()

    announce("Starting the Training!")
    do_train(
        model="deit_tiny_patch16_224",
        batch_size=sagemaker_stuff.config.BATCH_SIZE,
        data_path=sagemaker_stuff.config.DIR_DATA_TRAIN,
        output_dir=sagemaker_stuff.config.DIR_OUTPUT,
        device=sagemaker_stuff.config.DATA_AND_MODEL_DEVICE,
        dataset_filtration_cache_path="filtration_cache.h5",
        scoring_data_path="scoring_data.json"
    )


if __name__ == "__main__":
    main()
