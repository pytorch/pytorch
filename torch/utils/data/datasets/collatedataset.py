from torch.utils.data import IterableDataset, _utils
from typing import TypeVar, Callable, Iterator, Sized

T_co = TypeVar('T_co', covariant=True)
S_co = TypeVar('S_co', covariant=True)


class CollateIterableDataset(IterableDataset[T_co]):
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
                 dataset: IterableDataset[S_co],
                 *,
                 collate_fn: Callable[[S_co], T_co] = _utils.collate.default_collate,
                 ) -> None:
        super(CollateIterableDataset, self).__init__()
        self.dataset = dataset
        self.collate_fn = collate_fn

    def __iter__(self) -> Iterator[T_co]:
        for data in self.dataset:
            yield self.collate_fn(data)

    # `__len__` is attached to class not instance
    # Assume dataset has implemented `__len__` or raise NotImplementedError
    def __len__(self) -> int:
        if isinstance(self.dataset, Sized) and len(self.dataset) >= 0:
            return len(self.dataset)
        raise NotImplementedError
