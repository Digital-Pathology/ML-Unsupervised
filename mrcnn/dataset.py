
"""
    dataset for training mrcnn
"""

# stdlib imports
import os
from typing import Callable

# pip imports
import numpy as np
import torch

# local imports
from . import torchvision_utils


class MyDataset(torch.utils.data.Dataset):
    """ hides all of the logic for polymask handling """

    def __init__(self, imgs_list: list, hook_base_img: Callable, hook_polymask: Callable):
        """"""
        self.initialize_transforms()
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = imgs_list
        self.get_img = hook_base_img
        self.get_polymask = hook_polymask

    def __getitem__(self, idx):
        """"""
        # load images ad masks
        img = self.get_img(self.imgs[idx]).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = np.array(self.get_polymask(self.imgs[idx]))

        # reduce size because G-RAM is limited :(
        N = 512
        img = img.crop((0, 0, N, N))
        mask = mask[:-N, :-N]
        # return img, mask

        # instances are encoded as different colors
        obj_ids = np.unique(mask)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = []
        boxes = []
        for obj_id in obj_ids:
            obj_mask = mask == obj_id
            pos = np.where(obj_mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            if abs(xmax-xmin) > 1 and abs(ymax-ymin) > 1:
                masks.append(obj_mask)
                boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(masks),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        """"""
        return len(self.imgs)

    def initialize_transforms(self):
        """ randomly transform image for training """
        transforms = []
        transforms.append(torchvision_utils.transforms.ToTensor())
        transforms.append(
            torchvision_utils.transforms.RandomHorizontalFlip(0.5))
        self.transforms = torchvision_utils.transforms.Compose(transforms)
