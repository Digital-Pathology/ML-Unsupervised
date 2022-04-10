
# stdlib imports
import json
import os
import pickle
from typing import Iterable

# pip imports
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# local imports
from . import torchvision_utils

# intrapackage imports
from . import config
from . import predictions


class MyModel:
    """ wraps the model from the tutorial I'm following
    original: https://github.com/pytorch/tutorials/blob/master/_static/torchvision_finetuning_instance_segmentation.ipynb
    my changes: https://colab.research.google.com/drive/1ZWBjCJd06YrCVSCxxkXRiuXQXoAmBFoy?usp=sharing"""

    def __init__(self, model=config.DEFAULT_MODEL_FILEPATH):
        """  """
        if isinstance(model, str):
            self.load_model(model)
        else:
            self.model = model

    def load_model(self, filepath, map_location="cpu"):
        """ loads model from pickle file """
        self.model = torch.load(filepath, map_location=map_location)

    def save_model(self, filepath):
        """ saves the model to a pickle file """
        file_write_mode = 'w' if os.path.exists(filepath) else 'x'
        with open(filepath, file_write_mode + 'b') as f:
            pickle.dump(self.model, f)

    def get_predictions(self, img):
        """ returns wrapped predictions --> use: predictions[i]"""
        p = None
        with torch.no_grad():
            p = self.model([img])
        return predictions.Predictions(p[0])

    def do_epoch(self, optimizer, data_loader, device, epoch_num=0):
        """ does a single epoch """
        torchvision_utils.engine.train_one_epoch(
            self.model, optimizer, data_loader, device, epoch_num, print_freq=5)

    def do_epochs(self, optimizer, data_loader, device, n):
        """ does n epochs """
        for i in range(n):
            self.do_epoch(optimizer, data_loader, device, i)
