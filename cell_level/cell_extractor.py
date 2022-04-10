
"""
    Join feature engineering and segmentation into a single interface
"""

from typing import Callable, Iterable

from . import feature_engineering
from . import instance
from . import segmentation
from . import util

# TODO - docstrings


class CellExtractor:
    """
        Extracts cells from regions
    """

    def __init__(self,
                 features: util.Features,
                 extractor: util.Extractor) -> None:
        """ """
        self.features = features
        self.extractor = extractor
        self._initialize_features()
        self._initialize_extractor()

    def get_cell_representations(self, image, post_processor=None):
        """ """
        cells = self.extract_cells(image)

        def process_cell(cell):
            feature_summary = self.features(cell)
            if post_processor is not None:
                feature_summary = post_processor(feature_summary)
        return [process_cell(cell) for cell in cells]

    def extract_cells(self, image) -> Iterable[instance.Instance]:
        instances = None
        if isinstance(self.extractor, segmentation.ManagedModel):
            instances = self.extractor.get_instances(image)
        else:
            instances = self.extractor(image)
        return instances

    def process_cells(self, instances):
        pass

    def _initialize_features(self):
        if isinstance(self.features, str):
            raise NotImplementedError()
        elif isinstance(self.features, Iterable):
            raise NotImplementedError()
        elif isinstance(self.features, feature_engineering.Feature):
            pass
        elif isinstance(self.features, feature_engineering.FeatureManager):
            pass
        else:
            raise TypeError(type(self.features), self.features)

    def _initialize_extractor(self):
        if isinstance(self.extractor, str):
            self.extractor = segmentation.ModelManager().load_model(self.extractor)
        elif isinstance(self.extractor, segmentation.ManagedModel):
            pass
        elif isinstance(self.extractor, Callable):
            pass
        else:
            raise TypeError(type(self.extractor), self.extractor)
