import warnings
from torch.utils.data import IterDataPipe, _utils
from typing import TypeVar, Callable, Iterator, Sized

T_co = TypeVar('T_co', covariant=True)


# Default function to return each item directly
# In order to keep datapipe picklable, eliminates the usage
# of python lambda function
def default_fn(data):
    return data


class CallableIterDataPipe(IterDataPipe[T_co]):
    r""" :class:`CallableIterDataPipe`.

    Iterable DataPipe to run a function over each item from the source DataPipe.
    args:
        datapipe: Source Iterable DataPipe
        fn: Function called over each item
    """
    datapipe: IterDataPipe
    fn: Callable

    def __init__(self,
                 datapipe: IterDataPipe,
                 *args,
                 fn: Callable = default_fn,
                 **kwargs,
                 ) -> None:
        super().__init__()
        self.datapipe = datapipe
        if fn.__name__ == '<lambda>':
            warnings.warn("Lambda function is not supported for pickle, "
                          "please use regular python function instead.")
        self.fn = fn  # type: ignore
        self.args = args
        self.kwargs = kwargs

    def __iter__(self) -> Iterator[T_co]:
        for data in self.datapipe:
            yield self.fn(data, *self.args, **self.kwargs)

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized) and len(self.datapipe) >= 0:
            return len(self.datapipe)
        raise NotImplementedError


class CollateIterDataPipe(CallableIterDataPipe):
    r""" :class:`CollateIterDataPipe`.

    Iterable DataPipe to collate samples from datapipe to Tensor(s) by `util_.collate.default_collate`,
    or customized Data Structure by collate_fn.
    args:
        datapipe: Iterable DataPipe being collated
        collate_fn: Customized collate function to collect and combine data or a batch of data.
                    Default function collates to Tensor(s) based on data type.

    Example: Convert integer data to float Tensor
        >>> class MyIterDataPipe(torch.utils.data.IterDataPipe):
        ...     def __init__(self, start, end):
        ...         super(MyIterDataPipe).__init__()
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
        >>> ds = MyIterDataPipe(start=3, end=7)
        >>> print(list(ds))
        [3, 4, 5, 6]

        >>> def collate_fn(batch):
        ...     return torch.tensor(batch, dtype=torch.float)
        ...
        >>> collated_ds = CollateIterDataPipe(ds, collate_fn=collate_fn)
        >>> print(list(collated_ds))
        [tensor(3.), tensor(4.), tensor(5.), tensor(6.)]
    """
    def __init__(self,
                 datapipe: IterDataPipe,
                 *args,
                 collate_fn: Callable = _utils.collate.default_collate,
                 **kwargs,
                 ) -> None:
        super().__init__(datapipe, *args, fn=collate_fn, **kwargs)
