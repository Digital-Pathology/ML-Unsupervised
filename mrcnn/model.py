
# stdlib imports
import json
import os
import pickle

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


class Predictions:
    """ wraps the predictions tensor (shape=[100,1,N,N]) with easy indexing """

    def __init__(self, predictions):
        """ stores vanilla predictions """
        self.data = predictions

    def __getitem__(self, index):
        """ easy indexing """
        return self.data[index, 0].byte().numpy()


class MyModel:
    """ wraps the model from the tutorial I'm following 
    original: https://github.com/pytorch/tutorials/blob/master/_static/torchvision_finetuning_instance_segmentation.ipynb
    my changes: https://colab.research.google.com/drive/1ZWBjCJd06YrCVSCxxkXRiuXQXoAmBFoy?usp=sharing"""

    def __init__(self, filepath=config.DEFAULT_MODEL_FILEPATH):
        """  """
        self.load_model(filepath)

    def load_model(self, filepath):
        """ loads model from pickle file """
        self.model = None
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)

    def save_model(self, filepath):
        """ saves the model to a pickle file """
        file_write_mode = 'w' if os.path.exists(filepath) else 'x'
        with open(filepath, file_write_mode + 'b') as f:
            pickle.dump(self.model, f)

    def get_predictions(self, img):
        """ returns wrapped predictions --> use: predictions[i]"""
        prediction = None
        with torch.no_grad():
            prediction = self.model([img])
        return Predictions(prediction[0]['masks'])

    def do_epoch(self, optimizer, data_loader, device, epoch_num=0):
        """ does a single epoch """
        torchvision_utils.engine.train_one_epoch(
            self.model, optimizer, data_loader, device, epoch_num, print_freq=5)

    def do_epochs(self, optimizer, data_loader, device, n):
        """ does n epochs """
        for i in range(n):
            self.do_epoch(optimizer, data_loader, device, i)
