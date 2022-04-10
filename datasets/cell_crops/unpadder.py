
import torch

from . import util


class CellCropUnpadder(object):
    def __init__(self, padded_image):
        self.image = util.ensure_individual_image(padded_image)
        padded_img_width = self.image.shape[2]
        padded_img_height = self.image.shape[1]
        # initialize padding
        self.left_pad = 0
        self.right_pad = 0
        self.top_pad = 0
        self.bottom_pad = 0
        # left/right pad deals with columns
        while self.col_is_padding(self.left_pad):
            self.left_pad += 1
        while self.col_is_padding(padded_img_width - 1 - self.right_pad):
            self.right_pad += 1
        # top/bottom pad deals with rows:
        while self.row_is_padding(self.top_pad):
            self.top_pad += 1
        while self.row_is_padding(padded_img_height - 1 - self.bottom_pad):
            self.bottom_pad += 1

    def __call__(self, image):
        if len(image.shape) == 3:
            padded_image_width = image.shape[2]
            padded_image_height = image.shape[1]
        elif len(image.shape) == 2:
            padded_image_width = image.shape[1]
            padded_image_height = image.shape[0]
        else:
            raise Exception(image.shape)
        unpadded_image_width = padded_image_width - \
            (self.left_pad + self.right_pad)
        unpadded_image_height = padded_image_height - \
            (self.top_pad + self.bottom_pad)
        if len(image.shape) == 3:
            unpadded_image = image[
                :,
                self.top_pad: self.top_pad + unpadded_image_height,
                self.left_pad: self.left_pad + unpadded_image_width
            ]
        elif len(image.shape) == 2:
            unpadded_image = image[
                self.top_pad: self.top_pad + unpadded_image_height,
                self.left_pad: self.left_pad + unpadded_image_width
            ]
        else:
            raise Exception(image.shape)
        return unpadded_image

    def row_is_padding(self, row_num): return len(
        torch.unique(self.image[:, row_num, :])) == 1

    def col_is_padding(self, col_num): return len(
        torch.unique(self.image[:, :, col_num])) == 1

    def get_pads(
        self): return self.left_pad, self.right_pad, self.top_pad, self.bottom_pad
