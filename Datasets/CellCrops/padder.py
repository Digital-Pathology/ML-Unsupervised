
import torch

from . import util


class CellCropPadder(object):
    def __init__(self, size=512, value=0, placement=50) -> None:
        self.size = size
        self.value = value
        self.placement = placement

    def __call__(self, image) -> torch.Tensor:
        image = util.ensure_individual_image(image)
        padded_image = torch.ones(
            size=(image.shape[0], self.size, self.size)) * self.value
        padded_image[:, self.placement:self.placement+image.shape[1],
                     self.placement:self.placement+image.shape[2]] += image
        return padded_image
