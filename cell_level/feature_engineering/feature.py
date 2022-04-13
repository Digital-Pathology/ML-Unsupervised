
import abc
from typing import Iterable, Union

from . import util


class Feature(abc.ABC):
    '''
    A base class that highlights all the important methods for feature calculation.

    :param mask_image: 3 Channel image of the mask.
    :returns: Nothing, abstract class.
    '''

    def __init__(self, *args, **kwargs): pass

    def __call__(self, instance: Union[util.InstancePlaceholder, Iterable[util.InstancePlaceholder]]):
        image, mask = util.validate_image_and_mask(
            instance.image, instance.mask)
        return self.calculate_feature(image, mask)

    @abc.abstractmethod
    def calculate_feature(self, image, mask): pass

    @property
    def bins(self): pass
