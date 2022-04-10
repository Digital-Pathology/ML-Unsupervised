
import numpy as np

from . import util


class Instance:
    def __init__(self, **kwargs):
        self.data = kwargs

    def raise_invalid_arguments(self):
        raise util.InvalidArguments(list(self.keys()))

    @property
    def mask(self):
        if 'mask' in self.data:
            mask = self.data.get('mask')
            return mask
        elif 'points' in self.data:
            points = self.data.get('points')
            box = self.data.get('bounding_box')
            if box is None:
                box = util.get_box_from_points(points['x'], points['y'])
            mask_dimensions = (box[3]-box[1], box[2]-box[0])
            mask = util.poly2mask(
                [x-box[0] for x in points['x']],
                [y-box[1] for y in points['y']],
                mask_dimensions
            )
            self.data['mask'] = mask
            return mask
        else:
            self.raise_invalid_arguments()

    @property
    def points(self):
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
        if 'box' in self.data:
            box = self.data.get('bounding_box')
            return box
        else:
            self.raise_invalid_arguments()

    @property
    def image(self):
        if 'image' in self.data:
            image = self.data.get('image')
            return image
        else:
            raise NotImplementedError()
