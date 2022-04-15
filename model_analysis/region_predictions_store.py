
import abc
import json
import os
from typing import Optional


class RegionPredictionsStore(abc.ABC):
    """ stores region predictions """

    def __init__(self, filename: str = 'region_predictions.json', **kwargs):
        """ init """
        self.filename = filename
        self.load_region_predictions()

    @abc.abstractmethod
    def store(self, **kwargs):
        """ stores the prediction for a region from a file """
    @abc.abstractmethod
    def get(self, **kwargs):
        """ returns the prediction(s) associated with a filename and/or region, etc. """
    @abc.abstractmethod
    def has_predictions(self, **kwargs):
        """ checks to see if material might be overwritten """

    def save_region_predictions(self):
        file_mode = 'w' if os.path.exists(
            self.filename) else 'x'
        with open(self.filename, file_mode, encoding='utf-8') as f:
            json.dump(self.data, f)

    def load_region_predictions(self):
        if not os.path.exists(self.filename):
            return {}
        with open(self.filename, 'r', encoding='utf-8') as f:
            self.data = json.load(f)


class IndexedRegionPredictionsStore(RegionPredictionsStore):
    """ self.data = {filename: [region_prediction_0, region_prediction_1, ...], ...}
    region_level predictions are expected to be added sequentially from 0 to N for a given file """

    def __init__(self, filename: str = 'region_predictions_indexed.json', **kwargs):
        super().__init__(filename, **kwargs)

    def store(self, filename: str, region_identifier: int, prediction: int, **kwargs) -> None:
        """ class docstring """
        if filename not in self.data:
            self.data[filename] = []
        if len(self.data[filename]) != region_identifier:
            raise Exception(
                'region predictions should be added sequentially from 0 to N for a given file')
        self.data[filename].append(prediction)

    def get(self, filename: str, region_identifier: Optional[int] = None, **kwargs) -> int:
        """ returns self.data[filename] if region_identifier is None """
        region_predictions = self.data.get(filename, ())
        if region_identifier is None:
            return region_predictions
        else:
            return region_predictions[region_identifier]

    def has_predictions(self, filename: Optional[str] = None, region_identifier: Optional[int] = None, **kwargs):
        """ whether self.data contains info about args """
        if filename is None:
            return len(self.data) > 0
        else:
            region_predictions = self.data.get(filename, ())
            if region_identifier is None:
                return len(region_predictions) > 0
            else:
                return 0 <= region_identifier < len(region_predictions) and \
                    region_predictions[region_identifier] is not None


class AggregatedRegionPredictionsStore(IndexedRegionPredictionsStore):
    """ self.data = {filename: {region_prediction_0: count_0, region_prediction_1: count_1, ...}, ...}
    simply aggregates region-level predictions into counts for each file """

    def __init__(self, filename: str = 'region_predictions_aggregated.json', **kwargs):
        super().__init__(filename, **kwargs)

    def store(self, filename: str, region_identifier: int, prediction: int):
        """ class docstring """
        if filename not in self.data:
            self.data[filename] = {}
        if prediction not in self.data[filename]:
            self.data[filename][prediction] = 0
        self.data[filename][prediction] += 1

    def get(self, filename: str, prediction: int):
        """ class docstring """
        region_predictions = self.data.get(filename, ())
        if prediction is None:
            return region_predictions
        else:
            return region_predictions[prediction]


region_predictions_stores = {
    'indexed': IndexedRegionPredictionsStore,
    'aggregated': AggregatedRegionPredictionsStore
}
