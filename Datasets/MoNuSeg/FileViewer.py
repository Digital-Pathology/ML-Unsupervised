
import matplotlib.pyplot as plt
import numpy as np

# fileviewer utils
def show_image(image):
    plt.imshow(image.detach().permute(1, 2, 0))
    plt.show()
def show_mask(mask):
    mask = mask.clamp(min=0, max=1)
    show_image(mask)
def show_colormask(colormask):
    show_image(colormask)
