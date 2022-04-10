
import numpy as np


class InstancePlaceholder:
    pass


def validate_image_and_mask(image, mask):
    """ check dimensions, types, etc. """
    return image, mask  # TODO


def apply_mask_to_image(image, mask):
    return np.ma.masked_array(image, mask)


def is_binary(array):
    return True  # TODO


def get_bins(n):
    """ returns n bins from 0 to 1 """
    return [i*(1.0/float(n)) for i in range(n+1)]
