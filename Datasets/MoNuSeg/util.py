
import numpy as np
from skimage import draw


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    """ https://github.com/scikit-image/scikit-image/issues/1103 """
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def correct_axes(img):
    """ corrects the axes of an image to allow for imshow() """
    return np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)
