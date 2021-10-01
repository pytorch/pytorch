import math
from typing import TypeVar, Optional, Iterator

import torch
from . import Sampler, Dataset, PartitionedDataset
import torch.distributed as dist


T_co = TypeVar('T_co', covariant=True)


class DistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    It supports datasets which are either replicated on all participating
    processes or partitioned and distributed across ranks. Partitioned datasets
    are expected to be an instance of
    :class:`~torch.utils.data.PartitionedDataset`. In the latter case, when
    shuffling, it calls ``dataset.shuffle_inplace`` before getting items from
    the dataset. If ``dataset.shuffle_inplace`` returned ``True`` it will assume
    the data was shuffled and accesses items sequentially from 0-len, otherwise
    it will use the shuffled index order for ``dataset.__getitem__(indices)``.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training each of which is holding a replica of the
            dataset.  By default, :attr:`world_size` is retrieved from the
            current distributed group. Mutually exclusive with :attr:`num_ranks`
            and providing :attr:`dataset=PartitionedDataset`.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`
            or :attr:`num_ranks`. By default, :attr:`rank` is retrieved from the
            current distributed group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
        num_ranks (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved
            from the current distributed group. Mutually exclusive with and
            alternative to :attr:`num_replicas`.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False,
                 num_ranks: Optional[int] = None) -> None:
        if num_replicas is None and num_ranks is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_ranks = dist.get_world_size()
        elif num_replicas is not None and num_ranks is not None:  # num_replicas is not None
            raise RuntimeError("'num_replicas' is mutually exclusive with 'num_ranks'")
        elif num_ranks is None:  # num_replicas is not None
            num_ranks = num_replicas
        if num_ranks is None or num_ranks <= 0:
            raise RuntimeError("'num_ranks' must be > 0")
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_ranks or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_ranks - 1))
        self.dataset = dataset
        self.num_ranks = num_ranks
        if isinstance(self.dataset, PartitionedDataset):
            self.global_len = self.dataset.global_len()
            if num_replicas is not None:
                raise RuntimeError("Providing a PartitionedDataset is mutually exclusive with 'num_replicas'."
                                   " Use 'num_ranks' instead.")
        else:
            self.global_len = len(self.dataset)
            self.num_replicas = self.num_ranks  # BC
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_ranks != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (self.global_len - self.num_ranks) / self.num_ranks
            )
        else:
            self.num_samples = math.ceil(self.global_len / self.num_ranks)
        self.total_size = self.num_samples * self.num_ranks
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.global_len, generator=g).tolist()
        else:
            indices = list(range(self.global_len))

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        # shuffle data in-place?
        if isinstance(self.dataset, PartitionedDataset):
            if self.dataset.shuffle_inplace(indices, self.num_samples):
                # dataset got shuffled so we access items in order
                indices = list(range(self.num_samples))
            else:
                # dataset unchanged so we use the appropriate subset of the shuffled indices
                mx = (self.rank + 1) * self.num_samples
                mn = self.rank * self.num_samples
                indices = [x - mn for x in indices if x >= mn and x < mx]
        else:
            # subsample
            indices = indices[self.rank:self.total_size:self.num_ranks]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
