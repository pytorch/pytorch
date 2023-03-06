#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="test script")

    parser.add_argument(
        "--init-method",
        "--init_method",
        type=str,
        required=True,
        help="init_method to pass to `dist.init_process_group()` (e.g. env://)",
    )
    parser.add_argument(
        "--world-size",
        "--world_size",
        type=int,
        default=os.getenv("WORLD_SIZE", -1),
        help="world_size to pass to `dist.init_process_group()`",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=os.getenv("RANK", -1),
        help="rank to pass to `dist.init_process_group()`",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    dist.init_process_group(
        backend="gloo",
        init_method=args.init_method,
        world_size=args.world_size,
        rank=args.rank,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # one hot (by rank) tensor of size world_size
    # example:
    # rank 0, world_size 4 => [1, 0, 0, 0]
    # rank 1, world_size 4 => [0, 1, 0, 0]
    # ...
    t = F.one_hot(torch.tensor(rank), num_classes=world_size)

    # after all_reduce t = tensor.ones(size=world_size)
    dist.all_reduce(t)

    # adding all elements in t should equal world_size
    derived_world_size = torch.sum(t).item()
    if derived_world_size != world_size:
        raise RuntimeError(
            f"Wrong world size derived. Expected: {world_size}, Got: {derived_world_size}"
        )

    print("Done")


if __name__ == "__main__":
    main()
