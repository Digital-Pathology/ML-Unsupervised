
# https://stackoverflow.com/a/46336730
def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    return [
        (min(x_coordinates), min(y_coordinates)),
        (max(x_coordinates), max(y_coordinates))
    ]
