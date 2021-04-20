#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Runs CIFAR10 training with differential privacy.
"""

import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torch.utils.tensorboard as tensorboard
import torchvision.models as models
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from opacus.utils import stats
from opacus.utils.module_modification import convert_batchnorm_modules
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from torch import vmap
from make_functional import make_functional, load_weights
from functional_utils import grad, grad_with_value
from functools import partial
# from resnet import resnet18

def save_checkpoint(state, is_best, filename="checkpoint.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def accuracy(preds, labels):
    return (preds == labels).mean()


def compute_norms(sample_grads):
    batch_size = sample_grads[0].shape[0]
    norms = [sample_grad.view(batch_size, -1).norm(2, dim=-1) for sample_grad in sample_grads]
    norms = torch.stack(norms, dim=0).norm(2, dim=0)
    return norms

def clip_and_accumulate_and_add_noise(sample_grads, max_per_sample_grad_norm=1.0, noise_multiplier=1.0):
    # step 0: compute the norms
    sample_norms = compute_norms(sample_grads)

    # step 1: compute clipping factors
    clip_factor = max_per_sample_grad_norm / (sample_norms + 1e-6)
    clip_factor = clip_factor.clamp(max=1.0)

    # step 2: clip
    grads = tuple(torch.einsum('i,i...', clip_factor, sample_grad)
                  for sample_grad in sample_grads)

    # step 3: add gaussian noise
    stddev = max_per_sample_grad_norm * noise_multiplier
    noises = tuple(torch.normal(0, stddev, grad_param.shape, device=grad_param.device)
                   for grad_param in grads)
    grads = tuple(noise + grad_param for noise, grad_param in zip(noises, grads))

    return grads

def train(args, model, train_loader, optimizer, epoch, device):
    use_prototype = False
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    for i, (images, target) in enumerate(tqdm(train_loader)):

        images = images.to(device)
        target = target.to(device)

        # Step 1: compute per-sample-grads
        weights, func_model, descriptors = make_functional(model)

        def compute_loss_and_output(weights, image, target):
            images = image.unsqueeze(0)
            targets = target.unsqueeze(0)
            output = func_model(weights, (images,))
            loss = criterion(output, targets)
            return loss, output.squeeze(0)

        # grad_with_value(f) returns a function that returns (1) the grad and
        # (2) the output. `has_aux=True` means that `f` returns a tuple of two values,
        # where the first is to be differentiated and the second is not to be
        # differentiated and further adds a 3rd output.
        #
        # We need to use `grad_with_value(..., has_aux=True)` because we do
        # some analyses on the returned loss and output.
        grads_loss_output = grad_with_value(compute_loss_and_output, has_aux=True)
        sample_grads, sample_loss, output = vmap(partial(grads_loss_output, weights))(images, target)
        loss = sample_loss.mean()

        # Step 2: Clip the per-sample-grads, sum them to form grads, and add noise
        grads = clip_and_accumulate_and_add_noise(
            sample_grads, args.max_per_sample_grad_norm, args.sigma)

        # `load_weights` is the inverse operation of make_functional. We put
        # things back into a model so that we can directly apply optimizers.
        # TODO(rzou): this might not be necessary, optimizers just take
        # the params straight up.
        load_weights(model, descriptors, weights)

        for weight_grad, weight in zip(grads, model.parameters()):
            weight.grad = weight_grad.detach()

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()
        losses.append(loss.item())

        # measure accuracy and record loss
        acc1 = accuracy(preds, labels)

        top1_acc.append(acc1)
        stats.update(stats.StatType.TRAIN, acc1=acc1)

        # make sure we take a step after processing the last mini-batch in the
        # epoch to ensure we start the next epoch with a clean state
        if ((i + 1) % args.n_accumulation_steps == 0) or ((i + 1) == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
        else:
            optimizer.virtual_step()

        if i % args.print_freq == 0:
            print(
                f"\tTrain Epoch: {epoch} \t"
                f"Loss: {np.mean(losses):.6f} "
                f"Acc@1: {np.mean(top1_acc):.6f} "
            )

def test(args, model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in tqdm(test_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc1)

    top1_avg = np.mean(top1_acc)
    stats.update(stats.StatType.TEST, acc1=top1_avg)

    print(f"\tTest set:" f"Loss: {np.mean(losses):.6f} " f"Acc@1: {top1_avg :.6f} ")
    return np.mean(top1_acc)


def main():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")
    parser.add_argument(
        "-j",
        "--workers",
        default=2,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=1,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        # This should be 256, but that OOMs using the prototype.
        default=64,
        type=int,
        metavar="N",
        help="mini-batch size (default: 64), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "-na",
        "--n_accumulation_steps",
        default=1,
        type=int,
        metavar="N",
        help="number of mini-batches to accumulate into an effective batch",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.001,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="SGD momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=5e-4,
        type=float,
        metavar="W",
        help="SGD weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="GPU ID for this process (default: 'cuda')",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier (default 1.0)",
    )
    parser.add_argument(
        "-c",
        "--max-per-sample-grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm (default 1.0)",
    )
    parser.add_argument(
        "--disable-dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--secure-rng",
        action="store_true",
        default=False,
        help="Enable Secure RNG to have trustworthy privacy guarantees. Comes at a performance cost",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta (default: 1e-5)",
    )

    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default="checkpoint",
        help="path to save check points",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../cifar10",
        help="Where CIFAR10 is/will be stored",
    )
    parser.add_argument(
        "--log-dir", type=str, default="", help="Where Tensorboard log will be stored"
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="Adam",
        help="Optimizer to use (Adam, RMSprop, SGD)",
    )

    args = parser.parse_args()
    args.disable_dp = True

    if args.disable_dp and args.n_accumulation_steps > 1:
        raise ValueError("Virtual steps only works with enabled DP")

    # The following few lines, enable stats gathering about the run
    # 1. where the stats should be logged
    stats.set_global_summary_writer(
        tensorboard.SummaryWriter(os.path.join("/tmp/stat", args.log_dir))
    )
    # 2. enable stats
    stats.add(
        # stats about gradient norms aggregated for all layers
        stats.Stat(stats.StatType.GRAD, "AllLayers", frequency=0.1),
        # stats about gradient norms per layer
        stats.Stat(stats.StatType.GRAD, "PerLayer", frequency=0.1),
        # stats about clipping
        stats.Stat(stats.StatType.GRAD, "ClippingStats", frequency=0.1),
        # stats on training accuracy
        stats.Stat(stats.StatType.TRAIN, "accuracy", frequency=0.01),
        # stats on validation accuracy
        stats.Stat(stats.StatType.TEST, "accuracy"),
    )

    # The following lines enable stat gathering for the clipping process
    # and set a default of per layer clipping for the Privacy Engine
    clipping = {"clip_per_layer": False, "enable_stat": True}

    if args.secure_rng:
        assert False
        try:
            import torchcsprng as prng
        except ImportError as e:
            msg = (
                "To use secure RNG, you must install the torchcsprng package! "
                "Check out the instructions here: https://github.com/pytorch/csprng#installation"
            )
            raise ImportError(msg) from e

        generator = prng.create_random_device_generator("/dev/urandom")

    else:
        generator = None

    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    train_transform = transforms.Compose(
        augmentations + normalize if args.disable_dp else normalize
    )

    test_transform = transforms.Compose(normalize)

    train_dataset = CIFAR10(
        root=args.data_root, train=True, download=True, transform=train_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        generator=generator,
    )

    test_dataset = CIFAR10(
        root=args.data_root, train=False, download=True, transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    best_acc1 = 0
    device = torch.device(args.device)
    model = convert_batchnorm_modules(models.resnet18(num_classes=10))
    # model = CIFAR10Model()
    model = model.to(device)

    if args.optim == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError("Optimizer not recognized. Please check spelling")

    if not args.disable_dp:
        privacy_engine = PrivacyEngine(
            model,
            batch_size=args.batch_size * args.n_accumulation_steps,
            sample_size=len(train_dataset),
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=args.sigma,
            max_grad_norm=args.max_per_sample_grad_norm,
            secure_rng=args.secure_rng,
            **clipping,
        )
        privacy_engine.attach(optimizer)

    for epoch in range(args.start_epoch, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch, device)
        top1_acc = test(args, model, test_loader, device)

        # remember best acc@1 and save checkpoint
        is_best = top1_acc > best_acc1
        best_acc1 = max(top1_acc, best_acc1)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": "ResNet18",
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            filename=args.checkpoint_file + ".tar",
        )


if __name__ == "__main__":
    main()
