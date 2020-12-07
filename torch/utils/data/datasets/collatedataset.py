from torch.utils.data import IterableDataset, _utils
from typing import TypeVar, Callable, Union


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
S = TypeVar('S')

class CollateDataset(IterableDataset[S]):
    r""" Prototype of :class:`CollateDataset`.

    IterableDataset to collate samples from dataset to Tensor(s) by `util_.collate.default_collate`,
    or customized Data Structure by collate_fn.
    args:
        dataset: IterableDataset being collated
        collate_fn: Customized collate function to collect and combine data or a batch of data.
                    Default function collates to Tensor(s) based on data type.
        collate_batch_fn: Function creates batch or re-batch from input IterableDataset (bucket, pad).
    """
    def __init__(self,
                 dataset: IterableDataset[T_co],
                 *,
                 collate_fn: Callable[[Union[T, T_co]], S] = _utils.collate.default_collate,
                 collate_batch_fn: Callable[[IterableDataset[T_co]], T] = None,
                 ):
        super().__init__()
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.collate_batch_fn = collate_batch_fn

    def __iter__(self) -> S:
        if self.collate_batch_fn is not None:
            for batch in self.collate_batch_fn(self.dataset):
                yield self.collate_fn(batch)
        else:
            for batch in self.dataset:
                yield self.collate_fn(batch)

    def __len__(self):
        if len(self.dataset) < 0:
            raise NotImplementedError
        else:
            return len(self.dataset)
