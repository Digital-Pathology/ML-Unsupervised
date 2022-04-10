
'''
Contains classes and methods for calculating various geometric features.
'''

import abc

import numpy as np
from cv2 import distanceTransform, DIST_L2, DIST_MASK_PRECISE

from . import feature
from . import util


class GeometricFeature(feature.Feature, abc.ABC):
    pass


class Area(GeometricFeature):
    '''
    Calculates the Area of the mask by counting cells with value greater than 1.

    :param image: 3 Channel image of the cell.
    :param mask_image: 3 Channel image of the mask.
    :returns: Integer Area of the mask created from the image.
    '''

    def calculate_feature(self, image, mask):
        return np.sum(mask)


class Perimeter(GeometricFeature):
    '''
    Calculates the Perimeter of the mask by counting number of neighbors.

    :param image: 3 Channel image of the cell.
    :param mask_image: 3 Channel image of the mask.
    :returns: Integer Perimeter of the mask created from the image.
    '''

    def calculate_feature(self, image, mask):
        return self._find_perimeter(mask)

    def _find_perimeter(self, mask):
        perimeter = 0
        for i in range(len(mask)):
            for j in range(len(mask[0])):
                if mask[i][j]:
                    perimeter += (4 - self._count_neighbors(mask, i, j))
        return perimeter

    def _count_neighbors(self, mask, i, j):
        count = 0
        # UP
        if (i > 0 and mask[i - 1][j]):
            count += 1
        # LEFT
        if (j > 0 and mask[i][j - 1]):
            count += 1
        # DOWN
        if (i < len(mask)-1 and mask[i + 1][j]):
            count += 1
        # RIGHT
        if (j < len(mask[0])-1 and mask[i][j + 1]):
            count += 1
        return count


class AreaPerimeter(GeometricFeature):
    '''
    Calculates the Area to Perimeter ratio by division.

    :param image: 3 Channel image of the cell.
    :param mask_image: 3 Channel image of the mask.
    :returns: Ratio of Area to Perimeter in decimal format. 
    '''

    def calculate_feature(self, image, mask):
        area = Area().calculate_feature(image, mask)
        perimeter = Perimeter().calculate_feature(image, mask)
        return area / perimeter


class InsideRadialContact(GeometricFeature):
    '''
    Calculates the InsideRadialContact using Euclidean Distance Transform for white pixels.

    :param image: 3 Channel image of the cell.
    :param mask_image: 3 Channel image of the mask.
    :returns: A tuple.
        hist : array
            The values of the histogram excluding black pixels.
        bin_edges : array of dtype float
            Return the bin edges ``(length(hist)+1)``.
    '''

    def __init__(self):
        self.bins = util.get_bins(7)

    def calculate_feature(self, image, mask):
        return self._distance_transform(mask)

    def _distance_transform(self, mask):
        i = np.around(distanceTransform(mask.astype('uint8'),
                                        DIST_L2, DIST_MASK_PRECISE), decimals=2)
        i = i[i > 0]
        i = i / np.max(i)
        return np.histogram(i[i > 0], bins=self.bins)[0].tolist()
