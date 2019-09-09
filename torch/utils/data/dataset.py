import bisect
import warnings
import numpy

from torch._utils import _accumulate
from torch import randperm
from torch.utils.data import DistributedSampler, DistributedChunkSampler
from torch.utils.data._utils.worker import get_worker_info


class Dataset(object):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs a index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])

    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    # in pytorch/torch/utils/data/sampler.py


class IterableDataset(Dataset):
    r"""An iterable Dataset.

    All datasets that represent an iterable of data samples should subclass it.
    Such form of datasets is particularly useful when data come from a stream.

    All subclasses should overrite :meth:`__iter__`, which would return an
    iterator of samples in this dataset.

    When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
    item in the dataset will be yielded from the :class:`~torch.utils.data.DataLoader`
    iterator. When :attr:`num_workers > 0`, each worker process will have a
    different copy of the dataset object, so it is often desired to configure
    each copy independently to avoid having duplicate data returned from the
    workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
    process, returns information about the worker. It can be used in either the
    dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
    :attr:`worker_init_fn` option to modify each copy's behavior.

    Example 1: splitting workload across all workers in :meth:`__iter__`::

        >>> class MyIterableDataset(torch.utils.data.IterableDataset):
        ...     def __init__(self, start, end):
        ...         super(MyIterableDataset).__init__()
        ...         assert end > start, "this example code only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         worker_info = torch.utils.data.get_worker_info()
        ...         if worker_info is None:  # single-process data loading, return the full iterator
        ...             iter_start = self.start
        ...             iter_end = self.end
        ...         else:  # in a worker process
        ...             # split workload
        ...             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
        ...             worker_id = worker_info.id
        ...             iter_start = self.start + worker_id * per_worker
        ...             iter_end = min(iter_start + per_worker, self.end)
        ...         return iter(range(iter_start, iter_end))
        ...
        >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
        >>> ds = MyIterableDataset(start=3, end=7)

        >>> # Single-process loading
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
        [3, 4, 5, 6]

        >>> # Mult-process loading with two worker processes
        >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
        [3, 5, 4, 6]

        >>> # With even more workers
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=20)))
        [3, 4, 5, 6]

    Example 2: splitting workload across all workers using :attr:`worker_init_fn`::

        >>> class MyIterableDataset(torch.utils.data.IterableDataset):
        ...     def __init__(self, start, end):
        ...         super(MyIterableDataset).__init__()
        ...         assert end > start, "this example code only works with end >= start"
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         return iter(range(self.start, self.end))
        ...
        >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
        >>> ds = MyIterableDataset(start=3, end=7)

        >>> # Single-process loading
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
        [3, 4, 5, 6]
        >>>
        >>> # Directly doing multi-process loading yields duplicate data
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
        [3, 3, 4, 4, 5, 5, 6, 6]

        >>> # Define a `worker_init_fn` that configures each dataset copy differently
        >>> def worker_init_fn(worker_id):
        ...     worker_info = torch.utils.data.get_worker_info()
        ...     dataset = worker_info.dataset  # the dataset copy in this worker process
        ...     overall_start = dataset.start
        ...     overall_end = dataset.end
        ...     # configure the dataset to only process the split workload
        ...     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
        ...     worker_id = worker_info.id
        ...     dataset.start = overall_start + worker_id * per_worker
        ...     dataset.end = min(dataset.start + per_worker, overall_end)
        ...

        >>> # Mult-process loading with the custom `worker_init_fn`
        >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
        [3, 5, 4, 6]

        >>> # With even more workers
        >>> print(list(torch.utils.data.DataLoader(ds, num_workers=20, worker_init_fn=worker_init_fn)))
        [3, 4, 5, 6]
    """

    def __iter__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ChainDataset([self, other])

    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]


class TensorDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)


class ConcatDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        for d in self.datasets:
            assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


class ChainDataset(IterableDataset):
    r"""Dataset for chainning multiple :class:`IterableDataset` s.

    This class is useful to assemble different existing dataset streams. The
    chainning operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    Arguments:
        datasets (iterable of IterableDataset): datasets to be chained together
    """

    def __init__(self, datasets):
        super(ChainDataset, self).__init__()
        self.datasets = datasets

    def __iter__(self):
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            for x in d:
                yield x

    def __len__(self):
        total = 0
        for d in self.datasets:
            assert isinstance(d, IterableDataset), "ChainDataset only supports IterableDataset"
            total += len(d)
        return total


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class ChunkDataReader(object):
    r"""Reads a chunk of data for a particular index.

    Chunks can be files, contents of folder, pieces lines
    of a text file, etc

    In a distributed setup, each worker would draw different indices
    by feeding their ID (aka `rank`) into the chunk sampler.

    The reading logic must be implemented inside `__call__` method
    """

    def __init__(self):
        pass

    def __call__(self, idx):
        r"""Returns `tuple(data, target)`

        If `data` and `target` contain tensors, numpy arrays, numbers,
        dicts or lists, the default collate function can be implicitly used.
        Otherwise, a custom `collate_fn` must be provided to `DataLoader`

        When no more data is available, `StopIteration` must be raised.
        """
        raise NotImplementedError


class ChunkDataset(IterableDataset):
    r"""Dataset which uses hierarchical sampling for efficient data reading.

    Instead of reading the full dataset at once, `ChunkDataset` splits the data in
    chunks before loading some of them. Each chunk is then shuffled again before being returned.

    ``ChunkDataset`` extends ``IterableDataset`` because the
    size of the dataset is unknown .

    Arguments:
        chunk_sampler (DistributedSampler or DistributedChunkSampler): Sampler used to split dataset in multiple chunks.
                                       Typically used to split data amongst dataloader workers
        chunk_reader (ChunkDataReader): Specialized reader
        shuffle_cache (bool): Setting `True` forces shuffling of the internal chunk cache
    """

    def __init__(self, chunk_sampler, chunk_reader, shuffle_cache=True):
        super(ChunkDataset, self).__init__()
        assert isinstance(chunk_sampler, DistributedSampler) or isinstance(chunk_sampler, DistributedChunkSampler), 'sampler must be a `DistributedSampler` or `DistributedChunkSampler`'
        assert callable(chunk_reader), 'chunk_reader must be `callable()` and return a container with data'
        assert isinstance(shuffle_cache, bool), 'shuffle_cache must be a `bool`'

        self.chunk_sampler = chunk_sampler
        self.chunk_reader = chunk_reader
        self.shuffle_cache = shuffle_cache

        # Internal state
        self._chunk_sampler_iter = iter(self.chunk_sampler)
        self._data_cache = numpy.array([])
        self._target_cache = numpy.array([])
        self._batch_size = 1
        self._min_cache = 1000

    def __iter__(self):
        return self

    def __len__(self):
        raise NotImplementedError

    def __next__(self):
        r"""Returns a batch or raises exception when exhausted"""

        cache_size = 0
        if len(self._data_cache) > 0:
            cache_size = len(self._data_cache)
        elif len(self._target_cache) > 0:
            cache_size = len(self._target_cache)

        if (cache_size < self._min_cache):
            new_chunk = None
            try:
                new_chunk = self.chunk_reader(next(self._chunk_sampler_iter))
                assert isinstance(new_chunk, tuple), "Chunk data reader must return a `tuple(data, target)`"
            except StopIteration:
                pass

            if new_chunk and cache_size > 0:
                if new_chunk[0] is not None:
                    self._data_cache = numpy.concatenate((self._data_cache, new_chunk[0]))
                if new_chunk[1] is not None:
                    self._target_cache = numpy.concatenate((self._target_cache, new_chunk[1]))
            elif new_chunk:
                if new_chunk[0] is not None:
                    self._data_cache = new_chunk[0]
                if new_chunk[1] is not None:
                    self._target_cache = new_chunk[1]
            elif not new_chunk and cache_size == 0:
                raise StopIteration

            if self.shuffle_cache:
                numpy.random.shuffle(self._data_cache)
                numpy.random.shuffle(self._target_cache)


        if len(self._data_cache) > 0 and len(self._target_cache) > 0:
            batch = self._data_cache[:self._batch_size], self._target_cache[:self._batch_size]
            self._data_cache = self._data_cache[self._batch_size:]
            self._target_cache = self._target_cache[self._batch_size:]
        elif len(self._data_cache) > 0:
            batch = self._data_cache[:self._batch_size], None
            self._data_cache = self._data_cache[self._batch_size:]
        else:
            batch = None, self._target_cache[:self._batch_size]
            self._target_cache = self._target_cache[self._batch_size:]

        return batch

    def reset(self):
        r"""Resets internal state of ChunkDataset

        Typically will be used before a new epoch starts or
        during worker initialization process.
        """
        worker_id = 0
        try:
            worker_id = get_worker_info().id
            self.chunk_sampler.set_rank(worker_id)
        except Exception:
            pass
        self._chunk_sampler_iter = iter(self.chunk_sampler)

    next = __next__  # py2 compatibility


def random_split(dataset, lengths):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths)).tolist()
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]
