import warnings
from torch.utils.data import IterableDataset, _utils
from typing import TypeVar, Callable, Iterator, Sized

T_co = TypeVar('T_co', covariant=True)


# Default function to return each item directly
# In order to keep dataset picklable, eliminates the usage
# of python lambda function
def default_fn(data):
    return data


class CallableIterableDataset(IterableDataset[T_co]):
    r""" :class:`CallableIterableDataset`.

    IterableDataset to run a function over each item from the source dataset.
    args:
        dataset: Source IterableDataset
        fn: Function called over each item
    """
    dataset: IterableDataset
    fn: Callable

    def __init__(self,
                 dataset: IterableDataset,
                 *args,
                 fn: Callable = default_fn,
                 **kwargs,
                 ) -> None:
        super(CallableIterableDataset, self).__init__()
        self.dataset = dataset
        if fn.__name__ == '<lambda>':
            warnings.warn("Lambda function is not supported for pickle, "
                          "please use regular python function instead.")
        self.fn = fn  # type: ignore
        self.args = args
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[T_co]:
        for data in self.dataset:
            yield self.fn(data, *self.args, **self.kwargs)

    def __len__(self) -> int:
        if isinstance(self.dataset, Sized) and len(self.dataset) >= 0:
            return len(self.dataset)
        raise NotImplementedError


class CollateIterableDataset(CallableIterableDataset):
    r""" :class:`CollateIterableDataset`.

    IterableDataset to collate samples from dataset to Tensor(s) by `util_.collate.default_collate`,
    or customized Data Structure by collate_fn.
    args:
        dataset: IterableDataset being collated
        collate_fn: Customized collate function to collect and combine data or a batch of data.
                    Default function collates to Tensor(s) based on data type.

    Example: Convert integer data to float Tensor
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
        ...     def __len__(self):
        ...         return self.end - self.start
        ...
        >>> ds = MyIterableDataset(start=3, end=7)
        >>> print(list(ds))
        [3, 4, 5, 6]

        >>> def collate_fn(batch):
        ...     return torch.tensor(batch, dtype=torch.float)
        ...
        >>> collated_ds = CollateIterableDataset(ds, collate_fn=collate_fn)
        >>> print(list(collated_ds))
        [tensor(3.), tensor(4.), tensor(5.), tensor(6.)]
    """
    def __init__(self,
                 dataset: IterableDataset,
                 *args,
                 collate_fn: Callable = _utils.collate.default_collate,
                 **kwargs,
                 ) -> None:
        super(CollateIterableDataset, self).__init__(dataset, *args, fn=collate_fn, **kwargs)
