
from typing import OrderedDict
import numpy as np


class InstancePlaceholder:
    pass


class IncompatibleImageAndMask(Exception):
    pass


def validate_image_and_mask(image, mask):
    """ check dimensions, types, etc. """
    if len(image.shape) == 3 and len(mask.shape) == 2:
        if image.shape[:-1] != mask.shape:
            raise IncompatibleImageAndMask(
                f"{image.shape=} >--< {mask.shape=}")
    return image, mask  # TODO


def apply_mask_to_image(image, mask):
    return np.ma.masked_array(image, mask)


def is_binary(array):
    return True  # TODO


def get_bins(n):
    """ returns n bins from 0 to 1 """
    return [i*(1.0/float(n)) for i in range(n+1)]


def vectorize_dictionary(data: dict):
    new_data = OrderedDict()
    for k, v in data.items():
        if isinstance(v, list):
            for i, obj in enumerate(v):
                new_data[f"{k}{i}"] = obj
        else:
            new_data[k] = v
    return new_data
