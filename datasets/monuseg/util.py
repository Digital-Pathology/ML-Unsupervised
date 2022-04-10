
import numpy as np


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    """ https://github.com/scikit-image/scikit-image/issues/1103 """
    from skimage import draw
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def create_polymask(annotations, print_overlapping_items=False):
    """ generates a polymask from the provided region annotations """
    polymask = np.zeros((1000, 1000))
    for i, region in enumerate(annotations):
        mask = poly2mask(
            region['vertices_y'],
            region['vertices_x'],
            (1000, 1000)
        ).astype(int)
        overlapping_mask = ((mask > 0).astype(
            int) + (polymask > 0).astype(int)) > 1
        if np.any(overlapping_mask):
            if print_overlapping_items:
                print(i, end=', ')
        else:
            mask *= i+1
            polymask[mask > 0] = mask[mask > 0]
    return polymask


def process_annotations(annotations):
    """ parses the regions from the annotations xml """
    def process_region(region):
        data = {}
        for key, value in region.items():
            if key[0] == '@':
                data[key[1:]] = value
            elif key == 'Attributes':
                pass
            elif key == 'Vertices':
                data['vertices_x'] = []
                data['vertices_y'] = []
                vertices = value['Vertex']
                for vertex in vertices:
                    data['vertices_x'].append(float(vertex['@X']))
                    data['vertices_y'].append(float(vertex['@Y']))
            else:
                raise Exception(key, value)
        return data
    import xmltodict
    annotations = xmltodict.parse(annotations)
    regions = []
    for region in annotations['Annotations']['Annotation']['Regions']['Region']:
        processed_region = process_region(region)
        regions.append(processed_region)
    return regions


def correct_axes(img):
    """ corrects the axes of an image to allow for imshow() """
    return np.swapaxes(np.swapaxes(img, 0, 2), 0, 1)
