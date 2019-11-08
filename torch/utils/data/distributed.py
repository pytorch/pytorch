import math
import torch
from . import Sampler
import torch.distributed as dist


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a `DistributedSampler` instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset (Dataset): Dataset used for sampling. It can be `None` if :attr:`length` is specified
        num_replicas (int, optional): Number of processes participating in
            distributed training (default: `None`)
        rank (int, optional): Rank of the current process within :attr:`num_replicas` (default: `None`)
        shuffle (bool, optional): If `True` sampler will shuffle the indices (default: `True`)
        length (int, optional): length of `dataset`
            If `None`, length is calculated as `len(dataset)` (default: `None`)
            Must be greater than 0 when :attr:`dataset` is a `IterableDataset`
        padding (bool, optional): If `True`, the returned lists will be padded to have the same length
            Padding is done by adding (duplicate) indices from the beggining of the index list
            into the end of it in a circular fashion. (default: `True`)

    .. note::
        Regardless of :attr:`padding` value, :meth:`__len__` will return `ceil(dataset length / num_replicas)`
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, length=None, padding=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        if dataset is None and length is None:
            raise RuntimeError("Either 'dataset' or 'length' must be specified (not `None`)")

        if isinstance(length, int) and length > 0:
            self._dataset_length = length
        elif length is None:
            self._dataset_length = len(dataset)
        else:
            raise RuntimeError("When specified, 'length' must be a strictly positive integer")

        self.num_samples = int(math.ceil(self._dataset_length * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.padding = padding

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(self._dataset_length, generator=g).tolist()
        else:
            indices = list(range(self._dataset_length))


        # add extra samples to make it evenly divisible
        if self.padding:
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]

        if self.padding:
            assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_rank(self, rank):
        assert rank >= 0, 'rank must be >= 0'
        assert rank < self.num_replicas, 'rank must < num_replicas'
        self.rank = rank

    def set_epoch(self, epoch):
        self.epoch = epoch
