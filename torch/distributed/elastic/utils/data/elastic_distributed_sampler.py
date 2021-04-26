#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
from torch.utils.data.distributed import DistributedSampler


class ElasticDistributedSampler(DistributedSampler):
    """
    Sampler that restricts data loading to a subset of
    the dataset for elastic training.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        start_index (optional):  Which index of the dataset to start sampling from
    """

    def __init__(self, dataset, num_replicas=None, rank=None, start_index=0):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank)
        if start_index >= len(dataset):
            raise ValueError(
                "Start index {} should be less than dataset size {}".format(
                    start_index, len(dataset)
                )
            )

        self.start_index = start_index
        self.num_samples = int(
            math.ceil(float(len(self.dataset) - self.start_index) / self.num_replicas)  # type: ignore[arg-type]
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = (
            torch.randperm(len(self.dataset) - self.start_index, generator=g)  # type: ignore[arg-type]
            .add(self.start_index)
            .tolist()
        )

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples
