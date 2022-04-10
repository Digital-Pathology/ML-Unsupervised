
from typing import Any, Callable, Iterable, NewType, Union

import numpy as np

from . import feature_engineering
from . import instance
from . import segmentation

Features = NewType('Features',
                   Union[
                       str,
                       Iterable[str],
                       feature_engineering.Feature,
                       feature_engineering.FeatureManager,
                       Callable[[instance.Instance], dict[str, Any]]
                   ]
                   )

Extractor = NewType('Extractor',
                    Union[
                        str,
                        segmentation.ManagedModel,
                        Callable[[np.ndarray], Iterable[instance.Instance]]
                    ]
                    )
