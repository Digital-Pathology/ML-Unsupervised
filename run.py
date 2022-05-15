
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

import run_functions
import sagemaker_stuff
import util


def main():
    """ main """

    run_functions.announce_testing_status()

    run_functions.announce("Loading Model")
    model_callback = run_functions.get_model_callback()

    run_functions.announce("Initializing Dataset")
    dataset = run_functions.initialize_dataset()

    run_functions.announce("Get Analysis Job Region Predictions")
    analysis_job = run_functions.do_analysis_job_preprocessing(
        dataset, model_callback)


if __name__ == "__main__":
    main()
