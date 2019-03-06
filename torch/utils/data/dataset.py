import bisect
import warnings
import itertools

from torch._utils import _accumulate
from torch import randperm


class Dataset(object):
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overrite ``__getitem__``, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    ``__len__``, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class IterableDataset(Dataset):
    r"""An iterable Dataset.

    All datasets that represent an iterable of data samples should subclass it.
    Such form of datasets is particularly useful when data come from a stream.

    All subclasses should overrite ``__iter__``, which would return an iterator
    of samples in this dataset.

    When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
    item in the dataset will be yielded from the :class:`~torch.utils.data.DataLoader`
    iterator. When ``num_workers > 0``, each worker process will have a
    different copy of the dataset object, so it is often desired to configure
    each copy independently to avoid having duplicate data returned from the
    workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
    process, returns information about the worker. It can be used in either the
    dataset's ``__iter__`` method or the :class:`~torch.utils.data.DataLoader` 's
    :attr:`worker_init_fn` option to modify each copy's behavior.

    Examples::

        >>> # Use `get_worker_info` in `worker_init_fn`
        >>> class MyIterableDataset(torch.utils.data.IterableDataset):
        ...     def __init__(self, start, end):
        ...         super(MyIterableDataset).__init__()
        ...         self.start = start
        ...         self.end = end
        ...
        ...     def __iter__(self):
        ...         return iter(range(self.start, self.end))
        ...
        >>> def worker_init_fn(worker_id):
        ...     # Splits the overall workload across workers using `get_worker_info`
        ...     info = torch.utils.data.get_worker_info()
        ...     dataset = info.dataset  # info.dataset is the dataset copy in this worker process
        ...     overall_start = dataset.start
        ...     overall_end = dataset.end
        ...     # split workload
        ...     per_worker = int(math.ceil((overall_end - overall_start) / float(info.num_workers)))
        ...     worker_id = info.id
        ...     dataset.start = overall_start + worker_id * per_worker
        ...     dataset.end = min(dataset.start + per_worker, overall_end)
        ...
        >>> ds = MyIterableDataset(start=3, end=11)
        >>>
        >>> # The `worker_init_fn` splits workload using `worker_info.num_workers`
        >>> # so it works for different `num_workers` values.
        >>>
        >>> loader = torch.utils.data.DataLoader(ds, num_workers=0, worker_init_fn=worker_init_fn)
        >>> print(list(loader))
        [tensor(3), tensor(4), tensor(5), tensor(6), tensor(7), tensor(8), tensor(9), tensor(10)]
        >>>
        >>> loader = torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)
        >>> print(list(loader))
        [tensor(3), tensor(7), tensor(4), tensor(8), tensor(5), tensor(9), tensor(6), tensor(10)]
        >>>
        >>> loader = torch.utils.data.DataLoader(ds, num_workers=4, worker_init_fn=worker_init_fn)
        >>> print(list(loader))
        [tensor(3), tensor(5), tensor(7), tensor(9), tensor(4), tensor(6), tensor(8), tensor(10)]

        >>> # Use `get_worker_info` in `__iter__`
        >>> class MyIterableDataset(torch.utils.data.IterableDataset):
        ...     def __iter__(self):
        ...         worker_info = torch.utils.data.get_worker_info()
        ...         assert worker_info is not None, "Not in a worker process"
        ...         worker_id = worker_info.id
        ...         return iter([-1, worker_id + 1, (worker_id + 1) * 10])
        ...
        >>> ds = MyIterableDataset()
        >>> loader = torch.utils.data.DataLoader(ds, num_workers=2)
        >>>
        >>> # Worker 0 fetched [-1, 1, 10]. Worker 1 fetched [-1, 2, 20].
        >>> print(list(loader))
        [tensor(-1), tensor(-1), tensor(1), tensor(2), tensor(10), tensor(20)]
    """

    def __iter__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ChainDataset([self, other])

    def __len__(self):
        # Returning `NotImplemented` instead of raising `NotImplementedError`
        # allows for properly triggering some fallback behavior. E.g., the
        # built-in `list(X)` tries to call `len(X)` first, and executes a
        # different code path if `NotImplemented` is returned, while raising
        # `NotImplementedError` will propagate and and make the call fail.
        return NotImplemented


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


def random_split(dataset, lengths):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths))
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]
