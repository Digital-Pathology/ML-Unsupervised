'''
Contains classes and methods for calculating various geometric features.
'''

import numpy as np
import feature
from cv2 import distanceTransform, DIST_L2, DIST_MASK_PRECISE

class Area(feature.Feature):
    '''
    Calculates the Area of the mask by counting cells with value greater than 1.

    :param image: 3 Channel image of the cell.
    :param mask_image: 3 Channel image of the mask.
    :returns: Integer Area of the mask created from the image.
    '''
    def _calculate_feature(self):
        mask_ = self.bin_mask
        return len(mask_[mask_ > 0])


class Perimeter(feature.Feature):
    '''
    Calculates the Perimeter of the mask by counting number of neighbors.

    :param image: 3 Channel image of the cell.
    :param mask_image: 3 Channel image of the mask.
    :returns: Integer Perimeter of the mask created from the image.
    '''
    def _calculate_feature(self):
        return self._find_perimeter()
    def _find_perimeter(self):
        perimeter = 0

        for i in range(len(self.bin_mask)):
            for j in range(len(self.bin_mask[0])):
                if self.bin_mask[i][j]:
                    perimeter += (4 - self._count_neighbors(i,j))

        return perimeter
    def _count_neighbors(self, i, j):
        count = 0

        # UP
        if (i > 0 and self.bin_mask[i - 1][j]):
            count += 1

        # LEFT
        if (j > 0 and self.bin_mask[i][j - 1]):
            count += 1

        # DOWN
        if (i < len(self.bin_mask)-1 and self.bin_mask[i + 1][j]):
            count += 1

        # RIGHT
        if (j < len(self.bin_mask[0])-1 and self.bin_mask[i][j + 1]):
            count += 1

        return count


class AreaPerimeter(feature.Feature):
    '''
    Calculates the Area to Perimeter ratio by division.

    :param image: 3 Channel image of the cell.
    :param mask_image: 3 Channel image of the mask.
    :returns: Ratio of Area to Perimeter in decimal format. 
    '''
    def _calculate_feature(self):
        ar_ = Area(self.image, self.mask_image)
        pe_ = Perimeter(self.image, self.mask_image)
        return ar_()/pe_()


class InsideRadialContact(feature.Feature):
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
    def _calculate_feature(self):
        return self._distance_transform()
    def _distance_transform(self):
        i = np.around(distanceTransform(self.bin_mask.astype('uint8'),\
            DIST_L2, DIST_MASK_PRECISE), decimals=2)
        return np.histogram(i[i > 0])
