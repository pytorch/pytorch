# mypy: allow-untyped-defs
from collections.abc import Sized
from typing import TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DataChunk, MapDataPipe


__all__ = ["BatcherMapDataPipe"]


_T = TypeVar("_T")


@functional_datapipe("batch")
class BatcherMapDataPipe(MapDataPipe[DataChunk]):
    r"""
    Create mini-batches of data (functional name: ``batch``).

    An outer dimension will be added as ``batch_size`` if ``drop_last`` is set to ``True``,
    or ``length % batch_size`` for the last batch if ``drop_last`` is set to ``False``.

    Args:
        datapipe: Iterable DataPipe being batched
        batch_size: The size of each batch
        drop_last: Option to drop the last batch if it's not full

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp = SequenceWrapper(range(10))
        >>> batch_dp = dp.batch(batch_size=2)
        >>> list(batch_dp)
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    """

    datapipe: MapDataPipe
    batch_size: int
    drop_last: bool

    def __init__(
        self,
        datapipe: MapDataPipe[_T],
        batch_size: int,
        drop_last: bool = False,
        wrapper_class: type[DataChunk] = DataChunk,
    ) -> None:
        assert batch_size > 0, "Batch size is required to be larger than 0!"
        super().__init__()
        self.datapipe = datapipe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.wrapper_class = wrapper_class

    def __getitem__(self, index) -> DataChunk:
        batch: list = []
        indices = range(index * self.batch_size, (index + 1) * self.batch_size)
        try:
            batch.extend(self.datapipe[i] for i in indices)
            return self.wrapper_class(batch)
        except IndexError as e:
            if not self.drop_last and len(batch) > 0:
                return self.wrapper_class(batch)
            else:
                raise IndexError(f"Index {index} is out of bound.") from e

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            if self.drop_last:
                return len(self.datapipe) // self.batch_size
            else:
                return (len(self.datapipe) + self.batch_size - 1) // self.batch_size
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
