#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections.abc import Iterator, Sized
from typing import cast, Optional, TypeVar

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


T = TypeVar("T")

__all__ = ["ElasticDistributedSampler"]


class ElasticDistributedSampler(DistributedSampler[T]):
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

    def __init__(
        self,
        dataset: Dataset[T],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        start_index: int = 0,
    ):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank)
        if not isinstance(dataset, Sized):
            raise TypeError("Dataset must be an instance of collections.abc.Sized")

        # Cast to Sized for mypy
        sized_dataset = cast(Sized, dataset)

        if start_index >= len(sized_dataset):
            raise ValueError(
                f"Start index {start_index} should be less than dataset size {len(sized_dataset)}"
            )

        self.start_index = start_index
        sized_dataset = cast(Sized, self.dataset)
        self.num_samples = int(
            math.ceil(float(len(sized_dataset) - self.start_index) / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[T]:
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        sized_dataset = cast(Sized, self.dataset)
        indices = (
            torch.randperm(len(sized_dataset) - self.start_index, generator=g)
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

    def __len__(self) -> int:
        return self.num_samples
