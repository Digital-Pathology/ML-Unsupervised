
import numpy as np

from . import util


class Instance:
    """ wrapper on cell segmentation data that standardizes interface """

    def __init__(self, **kwargs):
        """ initialized Instance """
        self.data = kwargs

    def raise_invalid_arguments(self):
        """ raises helpful error """
        raise util.InvalidArguments(list(self.data.keys()))

    @property
    def mask(self):
        """ returns the mask either from self.data or constructed from self.data['points'] """
        mask = None
        if 'mask' in self.data:
            mask = self.data.get('mask')
        elif 'points' in self.data:
            points = self.data.get('points')
            box = self.data.get('bounding_box')
            mask = util.construct_mask_from_points(
                points['x'], points['y'], bounding_box=box)
            self.data['mask'] = mask
        else:
            self.raise_invalid_arguments()
        return mask

    @property
    def points(self):
        """ returns points from self.data or points pulled from self.data['mask'] """
        if 'mask' in self.data:
            mask = self.data.get('mask')
            raise NotImplementedError()
        elif 'points' in self.data:
            points = self.data.get('points')
            return points
        else:
            self.raise_invalid_arguments()

    @property
    def bounding_box(self):
        """ returns the bounding box from self.data """
        if 'box' in self.data:
            box = self.data.get('bounding_box')
            return box
        else:
            self.raise_invalid_arguments()

    @property
    def image(self):
        """ returns the original image data from self.data """
        if 'image' in self.data:
            image = self.data.get('image')
            return image
        else:
            raise NotImplementedError()
