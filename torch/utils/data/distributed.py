import math
import torch
from . import Sampler
import torch.distributed as dist


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset (Dataset, optional): Dataset used for sampling.
            Mutually exclusive with :attr:`num_indices`. (default: ``None``)
        num_indices (int, optional): Size of sampler
            Mutually exclusive with :attr:`dataset`. (default: ``None``)
        num_replicas (int, optional): Number of processes participating in
            distributed training. (default: ``None``)
        rank (int, optional): Rank of the current process within num_replicas.
            (default: ``None``)
        shuffle (bool, optional): If true (default), sampler will shuffle the indices
            (default: ``True``)
    """

    def __init__(self, dataset=None, num_indices=None, num_replicas=None, rank=None, shuffle=True):
        if dataset is None and num_indices is None:
            raise ValueError("dataset or num_indices must be specified")
        if dataset and num_indices:
            raise ValueError("dataset and num_indices are mutually exclusive")
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_indices = num_indices
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle

        if self.dataset:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = self.num_indices

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        if self.dataset:
            self.num_indices = len(self.dataset)

        if self.shuffle:
            indices = torch.randperm(self.num_indices, generator=g).tolist()
        else:
            indices = list(range(self.num_indices))


        # when dealing with dataset, pad the index list to make it evenly divisible
        if self.dataset:
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]

        # extra validation when dealing with dataset
        if self.dataset:
            assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        if self.dataset:
            return self.num_samples
        else:
            return self.num_indices

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_rank(self, rank):
        assert rank >= 0, 'rank must be >= 0'
        assert rank == 0 or rank <= self.num_replicas, 'rank must <= num_replicas'
        self.rank = rank