import importlib
import pyclbr

class FeatureManager():
    '''
    Calculates and returns features based on the features specified in the constructor. 
    If no features are provided automatically calculates all features.

    :param image: A 3 channel image of the cell. 
    :param mask_image: A black and white 3 channel image of the mask of the cell.
    :param feature_dict: Optional. A dictionary of features to be calculated.

    Example:
        feature_dict = {
            "geometric_features":["Area", "Perimeter"]
        }

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
    def __init__(self, image, mask_image, feature_dict=None) -> None:
        self.features = feature_dict if feature_dict else self._get_feature_dict(["geometric_features", "color_features"])
        self.image = image
        self.mask_image = mask_image
    def __call__(self):
        return self._calculate_features(self.features)
    def _get_feature_dict(self, feature_types):
        output = {}
        for feature_type in feature_types:
            output[feature_type] = pyclbr.readmodule(feature_type).keys()
        return output
    def _calculate_features(self, feature_dict):
        output = {}
        for feature_type in feature_dict.keys():
            gf = importlib.import_module(feature_type)
            output[feature_type] = {}
            for feature in feature_dict[feature_type]:
                c = getattr(gf, feature)
                o = c(mask_image=self.mask_image, image=self.image)
                output[feature_type][feature] = o()
        return output
