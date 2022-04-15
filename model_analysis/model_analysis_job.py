
import abc
import json
import os
from typing import Callable


class ModelAnalysisJob(abc.ABC):
    """ code reuse for model analysis """

    def __init__(self, dataset) -> None:
        """ class description """
        self.dataset = dataset

    @abc.abstractmethod
    def do_analysis(self):
        """ whatever needs to be done """
