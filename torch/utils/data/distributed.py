import math
import torch
from . import Sampler, SequentialSampler, RandomSampler
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
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
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
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class ChunkDataReader(object):
    r"""Reads a chunk of data given a chunk index.

    A chunk could be full file, section of a large file, folders, URLs, or any other
    abstraction that allows data to be segmented roughtly the same size.

    As an example, chunking could be used in a scenario where MNIST dataset for training 
    would be converted in CSV text files and equally splitted in several files,
    e.g. 60 files with 1000 lines each.  In this case, each worker would read
    a text file with all 1000 images.

    Another example would be chunks as individual folders, containing multiple audio files.
    In this scenario, each worker would have access to different folders, 
    where binary files would be read from the filesystem.

    A chunk doesn necessarily need to rely in multiple files to split data laoding across workers.
    It is possible to extract chunks from a single large dataset file, where each worker would
    seek to specific positions to load data.

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

    Python DataLoader uses multi-processing instead of
    multi-threading for parallelism, therefore, each worker
    is a separate process with an identical copy of sampler.
    Because of that, on a distributed environment, different processes
    could read the same portion of the dataset. This sampler uses strides,
    based on processes `rank`s, to coordinate parallel reading of chunks of data.

    Each instance of `DistributedChunkSampler` must be configured with different a `rank`
    (aka strides), so that chunk sampling happens in a round-robin fashion on each worker:
    rank0, rank1, ..., rankN, rank0, rank1, ..., rankN, ...

    For example, assume 2 workers reading the same `ChunkDataset` dataset.
    Each worker process needs to configure their `DistributedChunkSampler`
    so that one of them reads all even batches (e.g. rank 0)
    while the other process reads all odd batches (e.g. rank 1).

    Similarly to `DataLoader`, either a custom :attr:`sampler` can be specified or
    the number of chunks (:attr:`num_chunks`) and :attr:`shuffle` flag.
    In the latter case, `SequentialSampler` is used when :attr:`shuffle` is ``False``
    and `RandomSampler` otherwise.

    Args:
        num_replicas (int): Number of workers participating in the sampling.
        rank (int, optional): Current rank for the sampler, starting at 0.
            Typically set during worker initialization function on distributed setup.
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

        if shuffle:
            sampler = RandomSampler(data_source=range(num_chunks))
        else:
            sampler = SequentialSampler(data_source=range(num_chunks))
        self.sampler = sampler

    def __iter__(self):
        r"""Stride logic is implemented here"""
        # Fecthes all indices
        indices = list(self.sampler)

        # Create a strided sublist
        indices = indices[self.rank:self.num_chunks:self.num_replicas]
        return iter(indices)

    def __len__(self):
        raise NotImplementedError

    def set_rank(self, rank):
        r"""Sets current rank

        Typically used after new processes are spawned with a copy of this sampler
        to prevent sampling the same indices in different workers
        """
        assert rank >= 0, 'rank must be >= 0'
        assert rank == 0 or rank < self.num_replicas, 'rank must < num_replicas'
        self.rank = rank