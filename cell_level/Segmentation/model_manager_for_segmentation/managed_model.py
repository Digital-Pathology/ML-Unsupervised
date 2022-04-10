
"""
    ManagedModel requires diagnosis behavior from subclasses
"""

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from . import util


class ManagedModel(ABC):

    """
        ManagedModel requires diagnosis behavior from subclasses
    """

    @abstractmethod
    def get_instances(self, image: np.ndarray) -> Iterable[util.InstancePlaceholder]:
        """
        """
        return None
