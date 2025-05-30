import math
from collections.abc import Iterator
from typing import Optional, TypeVar, Sequence

import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler


__all__ = ["DistributedSampler"]


_T_co = TypeVar("_T_co", covariant=True)


class DistributedSampler(Sampler[_T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[_T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class DistributedWeightedRandomSampler(Sampler[_T_co]):
    """Sampler that applies weighted random sampling across multiple distributed processes.

    This sampler combines the behavior of `WeightedRandomSampler` (sampling based on weights)
    and `DistributedSampler` (splitting data across multiple processes).

    Args:
        weights (sequence): A sequence of weights, not necessarily summing up to one.
        num_samples (int): Total number of samples across all processes.
        num_replicas (int, optional): Number of processes in distributed training.
        rank (int, optional): Rank of the current process within num_replicas.
        replacement (bool, optional): If True, samples are drawn with replacement.
        shuffle (bool, optional): If True, shuffle indices before applying weights.
        seed (int, optional): Random seed for reproducibility.

    Example:
        >>> import torch
        >>> from torch.utils.data import DataLoader
        >>> from torch.utils.data.distributed import DistributedSampler
        >>> dataset = list(range(10))
        >>> weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        >>> sampler = DistributedWeightedRandomSampler(weights, num_samples=5, num_replicas=2, rank=0)
        >>> loader = DataLoader(dataset, sampler=sampler, batch_size=2)
        >>> for epoch in range(start_epoch, n_epochs):
        >>>     sampler.set_epoch(epoch)
        >>>     for batch in loader:
        >>>         print(batch)  # Each process gets a subset of the sampled indices
    """

    def __init__(
        self,
        weights: Sequence[float],
        num_samples: int,
        num_replicas: Optional[int] = None,
        rank: int = None,
        replacement: bool = True,
        shuffle: bool = True,
        seed: int = 0
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
            
        if len(weights) == 0:
            raise ValueError("Weights must be a non-empty sequence.")
        if any(w < 0 for w in weights):
            raise ValueError("Weights must be non-negative.")

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples  # total
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples_per_proc = self._compute_num_samples_per_rank()
        self.replacement = replacement
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Optional shuffling indices and weights before sampling
        indices = torch.arange(len(self.weights))
        if self.shuffle:
            perm = torch.randperm(len(indices), generator=g)  # Generate permutation
            indices = indices[perm]  # Shuffle indices
            weights = self.weights[perm]  # Shuffle weights in the same order
        else:
            weights = self.weights

        # Perform weighted sampling
        sampled_indices = torch.multinomial(
            weights, self.num_samples, self.replacement, generator=g
        )
        
        # Map sampled indices back to original dataset indices
        sampled_indices = indices[sampled_indices]

        # Distribute samples across processes
        # sampled_indices = sampled_indices[self.rank:self.num_samples_per_proc * self.num_replicas:self.num_replicas]
        start_index = sum(
            self.num_samples // self.num_replicas + (1 if r < self.num_samples % self.num_replicas else 0)
            for r in range(self.rank)
        )
        end_index = start_index + self.num_samples_per_proc
        sampled_indices = sampled_indices[start_index:end_index]
        if len(sampled_indices) != self.num_samples_per_proc:
            raise RuntimeError(
                f"Expected {self.num_samples_per_proc} samples per process, "
                f"but got {len(sampled_indices)}."
            )

        return iter(sampled_indices.tolist())

    def __len__(self) -> int:
        return self.num_samples_per_proc
    
    def _compute_num_samples_per_rank(self):
        # Distribute samples as evenly as possible
        base = self.num_samples // self.num_replicas
        extras = self.num_samples % self.num_replicas
        return base + (1 if self.rank < extras else 0)

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for deterministic shuffling."""
        self.epoch = epoch