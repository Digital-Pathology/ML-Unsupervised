
"""
    training job entry
"""

# stdlib imports
import os

# pip imports
import numpy as np
import torch

# local imports
from Datasets import MoNuSeg
import mrcnn


def main():
    """ main """

    dataset = mrcnn.dataset.MyDataset(
        imgs_list=MoNuSeg.FileReader.files[:],
        hook_base_img=(lambda filename: MoNuSeg.FileReader.get_image(
            filename, as_tensor=False)),
        hook_polymask=(lambda filename: MoNuSeg.FileReader.get_polymask(
            filename, as_tensor=False))
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1,
        collate_fn=mrcnn.torchvision_utils.utils.collate_fn)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"training on {device=}")
    model = mrcnn.model.MyModel()
    model.model.to(device)

    # construct an optimizer
    params = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    model.do_epoch(optimizer, data_loader, device)


if __name__ == "__main__":
    main()
