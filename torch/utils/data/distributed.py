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

class ChunkDataReader(object):
    r"""Reads a chunk of data given a chunk index.

    A chunk could be full file, section of a large file, folders, URLs, or any other
    abstraction that allows data to be segmented roughly the same size.

    As an example, chunking could be used in a scenario where a dataset is split between
    several CSV files each containing complete records or examples. In this case,
    each file could be a chunk. Similarly a large CSV file can be considered as a
    collection of chunks by defining chunk boundary based on some physical size
    (e.g. 32MB) and seeking to the first record in each chunk during reading.

    Another example would be chunks as individual folders, containing multiple audio files.
    In this scenario, each worker would have access to different folders, 
    where binary files would be read from the filesystem.

    The reading logic must be implemented inside `__call__` method
    """

    def __init__(self):
        pass

    def __call__(self, idx):
        r"""Returns `list(samples)`

        If `samples` contain tensors, numpy arrays, numbers,
        dicts or lists, the default collate function can be implicitly used.
        Otherwise, a custom `collate_fn` must be provided to `DataLoader`

        When no more data is available, `StopIteration` must be raised.
        """
        raise NotImplementedError

class DistributedChunkSampler(Sampler):
    r"""This sampler introduces distributed sampling without padding and dataset dependency.

    This sampler is very similar to the `DistributedSampler`, however without padding and
    the dependency on the size of the dataset. With two levels of sampling, the
    `DistributedChunkSampler` is used by the dataset and hence run by a dataloader worker.

    Python DataLoader uses multi-processing instead of
    multi-threading for parallelism, therefore, each worker
    is a separate process with an identical copy of sampler.
    Because of that, on a distributed environment, different processes
    could read the same portion of the dataset. This sampler uses strides,
    based on the rank concept, to coordinate parallel reading of chunks of data.

    For example, assume 2 workers reading the same `ChunkDataset` dataset.
    Each worker process needs to configure their `DistributedChunkSampler`
    so that one of them reads all even batches (e.g. `rank` 0)
    while the other process reads all odd batches (e.g. `rank` 1).

    Args:
        num_replicas (int): Number of processes participating in the data reading.
            Must be equal to the number of distributed processes * number of DataLoader workers.
        rank (int, optional): Current rank of the worker
            Internally, :attr:`rank` will be recalculated on each `DataLoader` worker
            based on current :attr:`rank`, and `DataLoader` information
            such as number of workers and worker ID, following the expression:
                `rank`=`worker_id`+`current_rank`*`num_workers`. (default: ``0``)
        num_chunks (int, optional): Number of chunks participating in the sampling.
            (default: ``0``)
        shuffle (bool, optional): set to ``True`` to have the chunk indices reshuffled.
            (default: ``False``)
    """

    def __init__(self, num_replicas, rank=0, num_chunks=0, shuffle=False):
        super(DistributedChunkSampler, self).__init__(None)
        assert rank >= 0, 'rank must be >= 0'
        assert num_replicas >= 0, 'num_replicas must be >= 0'
        assert rank <= num_replicas, 'rank must be <= num_replicas'
        assert num_chunks >= num_replicas, 'num_chunks must be >= num_replicas'
        self.rank = rank
        self.num_replicas = num_replicas
        self.num_chunks = num_chunks
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self):
        r"""Stride logic is implemented here"""

        # Deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(self.num_chunks, generator=g).tolist()
        else:
            indices = list(range(self.num_replicas))

        # Create a strided sublist
        indices = indices[self.rank:self.num_chunks:self.num_replicas]
        return iter(indices)

    def __len__(self):
        raise NotImplementedError

    def set_rank(self, rank):
        r"""Sets current global worker rank

        Typically used after new processes are spawned with a copy of this sampler
        to prevent sampling the same indices in different workers
        """
        assert rank >= 0, 'rank must be >= 0'
        assert rank < self.num_replicas, 'rank must < num_replicas'
        self.rank = rank

    def set_epoch(self, epoch):
        self.epoch = epoch
