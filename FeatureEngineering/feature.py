import abc
import numpy as np

class Feature(abc.ABC):
    '''
    A base class that highlights all the important methods for feature calculation.

    :param mask_image: 3 Channel image of the mask.
    :returns: Nothing, abstract class.
    '''

    def __init__(self, image, mask_image):
        self.image = image
        self.mask_image = mask_image
        self.mask = np.sum(self.mask_image, axis=2)/3
        self.bin_mask = np.where(self.mask > 200, 1, 0)
    @abc.abstractmethod
    def _calculate_feature(self):
        pass
    def __call__(self):
        return self._calculate_feature()