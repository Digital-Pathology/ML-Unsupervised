
import abc

from matplotlib.colors import rgb_to_hsv
import numpy as np

from . import feature
from . import util


class ColorFeature(feature.Feature, abc.ABC):
    pass


class MeanHSV(ColorFeature):
    '''
    Uses the mask to calculate the mean of HSV channels only for the cell pixels.

    :param image: 3 Channel image of the cell.
    :param mask_image: 3 Channel image of the mask.
    :returns: A list with mean values of H, S, and V channels respectively.
    '''

    def calculate_feature(self, image, mask):
        hsv = rgb_to_hsv(image)
        if len(mask.shape) == 2:
            mask_3d = np.stack([mask]*3, 2)
        hsv = util.apply_mask_to_image(hsv, mask_3d)
        return [hsv[:, :, 0].mean(), hsv[:, :, 1].mean(), hsv[:, :, 2].mean()]


class HueHistogram(ColorFeature):
    '''
    Quantizes the HSV values and calculates G = 9*h + 3*s + v for every pixel.

    :param image: 3 Channel image of the cell.
    :param mask_image: 3 Channel image of the mask.
    :returns: A histogram of G values.
    '''

    def __init__(self):
        self.bins = util.get_bins(32)

    def calculate_feature(self, image, mask):
        hsv = rgb_to_hsv(image)
        if len(mask.shape) == 2:
            mask_3d = np.stack([mask]*3, 2)
        hsv = util.apply_mask_to_image(hsv, mask_3d)
        return np.histogram(
            hsv[:, :, 0],
            bins=self.bins
        )[0].tolist()
