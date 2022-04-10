
"""
    An interface into the files in the dataset
"""

import os

from PIL import Image
from torchvision import transforms

from . import util

# filepaths and extensions
path_data_dir = __file__.replace(os.path.basename(__file__), "")
path_data_images = path_data_dir + 'Images/'
path_data_masks = path_data_dir + 'Masks/'
path_data_colormasks = path_data_dir + 'ColorMasks/'
path_data_polymasks = path_data_dir + 'PolyMasks/'
path_data_annotations = path_data_dir + 'Annotations/'
path_extension_image = '.png'
path_extension_mask = '-mask.png'
path_extension_colormask = '-mask-color.png'
path_extension_polymask = '-mask-poly.png'
path_extension_annotation = '.xml'

# get a list of the filenames
files = [f.replace(path_extension_image, '')
         for f in os.listdir(path_data_images)]
files.sort()

# file reading utils
pil_to_tensor = transforms.ToTensor()


def clean_filepath(filename, path_data_, path_extension_):
    """ ensures the filepath works with the directory structure """
    filepath = ("" if path_data_ in filename else path_data_) + filename
    if path_extension_ not in filename:
        filepath += path_extension_
    return filepath


def get_image(filename, as_tensor=True):
    """ gets the original image """
    filepath = clean_filepath(filename, path_data_images, path_extension_image)
    image = Image.open(filepath)
    if as_tensor:
        image = pil_to_tensor(image)
    return image


def get_mask(filename, as_tensor=True):
    """ gets the binary mask for the file """
    filepath = clean_filepath(filename, path_data_masks, path_extension_mask)
    mask = Image.open(filepath)
    if as_tensor:
        mask = pil_to_tensor(mask)
    mask = mask.clamp(min=0, max=1)
    return mask


def get_colormask(filename, as_tensor=True):
    """ gets the colormask from the original dataset's matlab script """
    filepath = clean_filepath(
        filename, path_data_colormasks, path_extension_colormask)
    colormask = Image.open(filepath)
    if as_tensor:
        colormask = pil_to_tensor(colormask)
    return colormask


def get_polymask(filename, as_tensor=True):
    """ returns the polymask for the file """
    filepath = clean_filepath(
        filename, path_data_polymasks, path_extension_polymask)
    polymask = Image.open(filepath)
    if as_tensor:
        polymask = pil_to_tensor(polymask)
    return polymask


def get_annotations(filename):
    """ returns xmltodict of the annotations """
    filepath = clean_filepath(
        filename, path_data_annotations, path_extension_annotation)
    annotations = None
    with open(filepath, 'r', encoding='utf-8') as f:
        annotations = f.read()
    return util.process_annotations(annotations)
