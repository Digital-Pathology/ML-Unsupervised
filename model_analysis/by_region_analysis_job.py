
import abc
from functools import reduce
from itertools import product, accumulate
import json
from numbers import Number
import os
from typing import Callable, Iterable, Union

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from pycm import ConfusionMatrix
from tqdm import tqdm as loadingbar

from .model_analysis_job import ModelAnalysisJob
from .region_predictions_store import RegionPredictionsStore, region_predictions_stores
from . import config


class ByRegionAnalysisJob(ModelAnalysisJob, abc.ABC):
    """  gets region-level predictions from images in a dataset """

    def __init__(self,
                 dataset,
                 region_prediction_callback: Callable,
                 region_predictions_store: Union[str,
                                                 RegionPredictionsStore] = 'indexed',
                 region_predictions_filename='region_predictions.json') -> None:
        """ pass in a function that gives the prediction for a single region
        region_predictions_store is a simple frequent-saving region-level prediction store """
        super().__init__(dataset)
        self.region_predictions_filename = region_predictions_filename
        self.region_prediction_callback = region_prediction_callback
        self.region_predictions_store = region_predictions_store
        if isinstance(self.region_predictions_store, str):
            self.region_predictions_store = region_predictions_stores[self.region_predictions_store]
        elif not isinstance(self.region_predictions_store, RegionPredictionsStore):
            raise TypeError(type(self.region_predictions_store),
                            region_predictions_store)

    @abc.abstractmethod
    def do_analysis(self):
        """ whatever needs to be done """

    def get_region_predictions_from_model(self, overwrite_none=False, overwrite_individuals=False, loadingbars=False):
        """ gets predictions of all regions in the dataset and saves them to file after each file """
        # first check to see if there are any predictions
        if self.region_predictions_store.has_predictions() and overwrite_none:
            return
        # then iterate over each file in the dataset
        for (filename, _, region_generator) in loadingbar(self.dataset.iterate_by_file(),
                                                          total=len(
                                                              self.dataset._filepaths),
                                                          disable=not loadingbars,
                                                          desc="Getting region predictions"):
            # first check whether files in the store should be overwritten (i.e. if the predictions file isn't complete)
            if self.region_predictions_store.has_predictions(filename) and not overwrite_individuals:
                continue
            for region_num, region in loadingbar(enumerate(region_generator),
                                                 disable=not loadingbars,
                                                 total=self.dataset.number_of_regions(
                                                     filename)):
                region_prediction = self.region_prediction_callback(region)
                self.region_predictions_store.store(
                    filename, region_num, region_prediction)
            self.region_predictions_store.save_region_predictions()


class WeightRatioAnalysisJob(ByRegionAnalysisJob):
    """ produces 3d data representing model prediction accuracy with different weight ratios wtih weight_0 always starting at 1 """

    def __init__(self,
                 dataset,
                 region_prediction_callback: Callable,
                 region_predictions_filename='region_predictions.json') -> None:
        """ class docstring """
        region_predictions_store = region_predictions_stores['aggregated']()
        super().__init__(dataset, region_prediction_callback,
                         region_predictions_store, region_predictions_filename)

    def do_analysis(self,
                    ratio_space: Iterable[Iterable[Number]] = (
                        (1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 10),
                        (1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 10)
                    ),
                    metrics=config.ALL_METRICS,
                    results_filepath='model_analysis_results_weight_ratio.csv',
                    loadingbars: bool = False,
                    prediction_cache_overwrite_policy: dict = {
                        'overwrite_none': False,
                        'overwrite_individuals': False
                    }):
        """ yields data about the model with various weights for each label/prediction 
        --> saves lots of files into a directory
        """

        # first get the predictions
        self.get_region_predictions_from_model(
            **prediction_cache_overwrite_policy, loadingbars=loadingbars)

        # now set up the results dataframe

        def get_row(weights, cm):
            row = {f"weight_{i}": w for i, w in enumerate(weights)}
            for metric in metrics:
                metric_value = getattr(cm, metric)
                if isinstance(metric_value, Iterable) and not isinstance(metric_value, str):
                    metric_value_iter = metric_value.items() if isinstance(
                        metric_value, dict) else enumerate(metric_value)
                    for i, v in metric_value_iter:
                        row[f"{metric}_{i}"] = v
                else:
                    row[metric] = metric_value
            return row

        results_dataframe = None

        # iterate over the ratios
        for ratios in loadingbar(product(*ratio_space),
                                 disable=not loadingbars,
                                 total=reduce(lambda a, b: a * b, (len(rs)
                                                                   for rs in ratio_space), 1),
                                 leave=True,
                                 desc=f"Evaluating for each ratio permutation"):
            weights = list(accumulate([1] + list(ratios), lambda a, b: a*b))
            confusion_matrix = self.get_confusion_matrix(
                weights, loadingbars=loadingbars)
            row = get_row(weights, confusion_matrix)
            if results_dataframe is None:
                results_dataframe = pd.DataFrame(columns=list(row.keys()))
                # print(results_dataframe.columns)
            #print({k: row.get(k) for k in [f"weight_{i}" for i in range(3)]})
            results_dataframe = results_dataframe.append(
                row, ignore_index=True)
        results_dataframe.to_csv(results_filepath)

    def get_confusion_matrix(self, weights, loadingbars: bool = False) -> np.ndarray:
        """ returns a confusion matrix whereby axis 0 is ground truth and axis 1 is predicted """
        labels = []
        predictions = []
        for filename, region_predictions in loadingbar(self.region_predictions_store.data.items(),
                                                       disable=not loadingbars,
                                                       total=len(
                self.dataset._filepaths),
                leave=False):
            # get the actual label
            label = self.dataset.get_label(filename)
            if not isinstance(label, int):
                raise TypeError(type(label))
            labels.append(label)
            # now get the predicted label based on the weights
            subtotals = [(weight * region_predictions[str(i)])
                         for i, weight in enumerate(weights)]
            total = sum(subtotals)
            prediction = total / sum(region_predictions.values())
            prediction = round(prediction)
            predictions.append(prediction)
        return ConfusionMatrix(actual_vector=labels, predict_vector=predictions)
