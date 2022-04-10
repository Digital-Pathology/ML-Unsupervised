
import numpy as np


class InvalidArguments(Exception):
    pass


def get_box_from_points(points_x, points_y):
    x_min = min(points_x)
    x_max = max(points_x)
    y_min = min(points_y)
    y_max = max(points_y)
    return (x_min, y_min, x_max, y_max)


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    """ https://github.com/scikit-image/scikit-image/issues/1103 """
    from skimage import draw
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask
