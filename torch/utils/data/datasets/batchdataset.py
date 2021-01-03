from torch.utils.data import IterableDataset
from typing import TypeVar, Optional, Iterator, List, Sized

T_co = TypeVar('T_co', covariant=True)


class BatchIterableDataset(IterableDataset[List[T_co]]):
    r""" :class:`BatchIterableDataset`.

    IterableDataset to create mini-batches of data. An outer dimension will be added as
    `batch_size` if `drop_last` is set to `True`, or `length % batch_size` for the
    last batch if `drop_last` is set to `False`.
    args:
        dataset: IterableDataset being batched
        batch_size: The size of each batch
        drop_last: Option to drop the last batch if it's not full
    """
    dataset: IterableDataset[T_co]
    batch_size: int
    drop_last: bool
    length: Optional[int]

    def __init__(self,
                 dataset: IterableDataset[T_co],
                 *,
                 batch_size: int,
                 drop_last: bool = False,
                 ) -> None:
        assert batch_size > 0, "Batch size is required to be larger than 0!"
        super(BatchIterableDataset, self).__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.length = None

    def __iter__(self) -> Iterator[List[T_co]]:
        batch: List[T_co] = []
        for x in self.dataset:
            batch.append(x)
            if len(batch) == self.batch_size:
                yield batch
                batch.clear()
        if len(batch) > 0:
            if not self.drop_last:
                yield batch
            batch.clear()

    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        if isinstance(self.dataset, Sized) and len(self.dataset) >= 0:
            if self.drop_last:
                self.length = len(self.dataset) // self.batch_size
            else:
                self.length = (len(self.dataset) + self.batch_size - 1) // self.batch_size
            return self.length
        raise NotImplementedError
