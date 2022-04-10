
from collections import OrderedDict
from typing import Dict

from . import feature
from . import color_feature
from . import geometric_feature


def get_feature_map(cls):
    """ returns a dictionary of class_name to class for subclasses of cls """
    return {
        subclass.__name__: subclass
        for subclass in cls.__subclasses__()
    }


FEATURE_MAP = {
    "color_feature": get_feature_map(color_feature.ColorFeature),
    "geometric_feature": get_feature_map(geometric_feature.GeometricFeature)
}


class FeatureManager(feature.Feature):
    '''
    Possible features:
        * geometric_features:
            - Area
            - Perimeter
            - AreaPerimeter
            - InsideRadialContact
        * color_features:
            - MeanHSV
            - HistogramHSV
    '''

    def __init__(self, as_vector: bool = True, **kwargs) -> None:
        self.as_vector = as_vector
        self.features: OrderedDict[str, feature.Feature] = OrderedDict()
        if len(kwargs) == 0:
            kwargs = {feature_type: list(subclasses.keys())
                      for feature_type, subclasses in FEATURE_MAP.items()}
        for feature_type, subcls_iterable in kwargs.items():
            feature_map = FEATURE_MAP.get(feature_type)
            for feature_name in subcls_iterable:
                feature_cls = feature_map.get(feature_name)
                feature_obj = feature_cls()
                self.features[feature_name] = feature_obj

    def calculate_feature(self, image, mask):
        data = OrderedDict({
            feature_name: feature_obj.calculate_feature(image, mask) for
            feature_name, feature_obj in
            self.features.items()
        })
        if self.as_vector:
            for k, v in data.items():
                if isinstance(v, list):
                    del data[k]
                    for i, obj in enumerate(v):
                        data[f"{k}{i}"] = obj
        return data
