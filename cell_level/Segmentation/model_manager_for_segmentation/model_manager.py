
"""
    A wrapper on ModelManager that requires use of ManagedModel
"""

from types import ModuleType
from typing import Iterable
from model_manager import ModelManager as OriginalModelManager

from . import config
from .managed_model import ManagedModel


class ModelManager(OriginalModelManager):

    """
        Requires saved models to be ManagedModels
    """

    def __init__(self):
        super().__init__(config.DEFAULT_MODELS_DIR)

    def save_model(self,
                   model_name: str,
                   model: ManagedModel,  # no longer Any
                   model_info: dict = None,
                   overwrite_model: bool = False,
                   dependency_modules: Iterable[ModuleType] = None):
        if not isinstance(model, ManagedModel):
            raise TypeError(
                f"model must be of type ManagedModel but is {type(model)=}")
        return super().save_model(
            model_name,
            model,
            model_info,
            overwrite_model,
            dependency_modules
        )
