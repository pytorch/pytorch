#!/usr/bin/env python3
# Add metrics images to tensorboard summary for easy viewing/comparisons

import argparse
import os

from PIL import Image
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor


def generate_tensorboard_img_summary(logdir, imgdir):
    writer = SummaryWriter(logdir)
    all_metrics_graphs = list(Path(imgdir).rglob("*.png"))

    for img in all_metrics_graphs:
        tag = os.path.basename(img)
        img_tensor = ToTensor()(Image.open(img))
        writer.add_image(tag, img_tensor, 0)
    writer.close()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--logdir', type=str)
    arg_parser.add_argument('--imgdir', type=str)
    args = arg_parser.parse_args()
    generate_tensorboard_img_summary(args.logdir, args.imgdir)
