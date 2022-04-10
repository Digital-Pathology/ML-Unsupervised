
from PIL import Image
import torch
import torchvision

from . import padder, unpadder


pil_to_tensor = torchvision.transforms.ToTensor()


def get_cell_crop(cell_crop_filename):
    return pil_to_tensor(Image.open(cell_crop_filename))


def get_padded_cell_crop_and_unpadder(img):
    _padder = padder.CellCropPadder()
    padded_img = _padder(img)
    _unpadder = unpadder.CellCropUnpadder(padded_img)
    return padded_img, _unpadder


# def test_model_with_cell_crop(model_callback, cell_crop_filename, threshold=0.5):
#    img = get_cell_crop(cell_crop_filename)
#    padder = padder.CellCropPadder()
#    padded_img = padder(img)
#    prediction = test_model(model, padded_img)
#    unpadder = unpadder.CellCropUnpadder(padded_img)
#    unpadded_prediction = unpadder(prediction)
#    final_prediction = (unpadded_prediction > threshold).to(
#        unpadded_prediction.dtype)
#    return final_prediction
