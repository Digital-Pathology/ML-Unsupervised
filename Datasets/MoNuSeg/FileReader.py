
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

import xmltodict

# filepaths and extensions
path_data_dir = __file__.replace(os.path.basename(__file__),"")
path_data_images = path_data_dir + 'Images/'
path_data_masks = path_data_dir + 'Masks/'
path_data_colormasks = path_data_dir + 'ColorMasks/'
path_data_annotations = path_data_dir + 'Annotations/'
path_extension_image = '.png'
path_extension_mask = '-mask.png'
path_extension_colormask = '-mask-color.png'
path_extension_annotation = '.xml'

# get a list of the filenames
files = [f.replace(path_extension_image,'') for f in os.listdir(path_data_images)]

# file reading utils
pil_to_tensor = transforms.ToTensor()
def clean_filepath(filename, path_data_, path_extension_):
    filepath = ("" if path_data_ in filename else path_data_) + filename
    if path_extension_ not in filename: filepath += path_extension_
    return filepath
def get_image(filename):
    filepath = clean_filepath(filename, path_data_images, path_extension_image)
    image = Image.open(filepath)
    image = pil_to_tensor(image)
    return image
def get_mask(filename):
    filepath = clean_filepath(filename, path_data_masks, path_extension_mask)
    mask = Image.open(filepath)
    mask = pil_to_tensor(mask)
    mask = mask.clamp(min=0, max=1)
    return mask
def get_colormask(filename):
    filepath = clean_filepath(filename, path_data_colormasks, path_extension_colormask)
    colormask = Image.open(filepath)
    colormask = pil_to_tensor(colormask)
    return colormask
def get_annotations(filename):
    filepath = clean_filepath(filename, path_data_annotations, path_extension_annotation)
    annotations = None
    with open(filepath, 'r') as f:
        annotations = f.read()
        annotations = xmltodict.parse(annotations)
    return annotations

def process_annotations(annotations):
    regions = []
    regions_iter = annotations['Annotations']['Annotation']['Regions'].items()

