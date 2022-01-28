import feature
from matplotlib.colors import rgb_to_hsv
from numpy import histogram
from numpy.ma import masked_array

class MeanHSV(feature.Feature):
    '''
    Uses the mask to calculate the mean of HSV channels only for the cell pixels. 
    
    :param image: 3 Channel image of the cell.
    :param mask_image: 3 Channel image of the mask.
    :returns: A list with mean values of H, S, and V channels respectively.
    '''
    def _calculate_feature(self):
        hsv = rgb_to_hsv(self.image)
        masked1 = masked_array(hsv[:, :, 0], self.bin_mask)
        masked2 = masked_array(hsv[:, :, 1], self.bin_mask)
        masked3 = masked_array(hsv[:,:,2], self.bin_mask)
        return [masked1.mean(), masked2.mean(), masked3.mean()]

class HistogramHSV(feature.Feature):
    '''
    Quantizes the HSV values and calculates G = 9*h + 3*s + v for every pixel.

    :param image: 3 Channel image of the cell.
    :param mask_image: 3 Channel image of the mask.
    :returns: A histogram of G values.
    '''
    def _calculate_feature(self):
        hsv = rgb_to_hsv(self.image)

        h = masked_array(hsv[:, :, 0], self.bin_mask).data
        s = masked_array(hsv[:, :, 1], self.bin_mask).data
        v = masked_array(hsv[:, :, 2], self.bin_mask).data

        h = self._bin_h(h)
        s = self._bin_s(s)
        v = self._bin_v(v)

        G = 9*h + 3*s + v

        return histogram(G)

    def _bin_h(self,h):
        h[h <= 20] = 0
        h[(h >= 21) & (h <= 40)] = 1
        h[(h >= 41) & (h <= 75)] = 2
        h[(h >= 76) & (h <= 155)] = 3
        h[(h >= 156) & (h <= 190)] = 4
        h[(h >= 191) & (h <= 270)] = 5
        h[(h >= 271) & (h <= 295)] = 6
        h[(h >= 296) & (h <= 315)] = 7
        return h
    def _bin_s(self, s):
        s[(s >= 0) & (s <= 0.2)] = 0
        s[(s > 0.2) & (s <= 0.7)] = 1
        s[(s > 0.7) & (s <= 1)] = 2
        return s
    def _bin_v(self, v):
        v[(v >= 0) & (v <= 0.2)] = 0
        v[(v > 0.2) & (v <= 0.7)] = 1
        v[(v > 0.7) & (v <= 1)] = 2
        return v
