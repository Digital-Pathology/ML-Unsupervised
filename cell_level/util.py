
from typing import Any, Callable, Iterable, NewType, Union

import numpy as np

from . import FeatureEngineering
from . import Instance
from . import Segmentation

Features = NewType('Features',
                   Union[
                       str,
                       Iterable[str],
                       FeatureEngineering.Feature,
                       FeatureEngineering.FeatureManager,
                       Callable[[Instance.Instance], dict[str, Any]]
                   ]
                   )

Extractor = NewType('Extractor',
                    Union[
                        str,
                        Segmentation.ManagedModel,
                        Callable[[np.ndarray], Iterable[Instance.Instance]]
                    ]
                    )
