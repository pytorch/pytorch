from torch.utils.data import MapDataPipe, functional_datapipe, DataChunk
from typing import List, Optional, Sized, TypeVar


T = TypeVar('T')

@functional_datapipe('batch')
class BatcherMapDataPipe(MapDataPipe[DataChunk]):
    r""" :class:`BatcherMapDataPipe`.

    Map DataPipe to create mini-batches of data. An outer dimension will be added as
    `batch_size` if `drop_last` is set to `True`, or `length % batch_size` for the
    last batch if `drop_last` is set to `False`.

    Args:
        datapipe: Iterable DataPipe being batched
        batch_size: The size of each batch
        drop_last: Option to drop the last batch if it's not full
        unbatch_level: Specifies if it necessary to unbatch source data before
            applying new batching rule
    """
    datapipe: MapDataPipe
    batch_size: int
    drop_last: bool
    length: Optional[int]

    def __init__(self,
                 datapipe: MapDataPipe[T],
                 batch_size: int,
                 drop_last: bool = False,
                 unbatch_level: int = 0,
                 wrapper_class=DataChunk,
                 ) -> None:
        assert batch_size > 0, "Batch size is required to be larger than 0!"
        super().__init__()
        if unbatch_level == 0:
            self.datapipe = datapipe
        else:
            self.datapipe = datapipe.unbatch(unbatch_level=unbatch_level)
        self.unbatch_level = unbatch_level
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.length = None
        self.wrapper_class = wrapper_class

    def __getitem__(self, index) -> DataChunk:
        batch: List[T] = []
        indices = range(index * self.batch_size, (index + 1) * self.batch_size)
        try:
            for i in indices:
                batch.append(self.datapipe[i])
            return self.wrapper_class(batch)
        except IndexError:
            if not self.drop_last and len(batch) > 0:
                return self.wrapper_class(batch)
            else:
                raise IndexError(f"Index {index} is out of bound.")

    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        if isinstance(self.datapipe, Sized) and self.unbatch_level == 0:
            if self.drop_last:
                self.length = len(self.datapipe) // self.batch_size
            else:
                self.length = (len(self.datapipe) + self.batch_size - 1) // self.batch_size
            return self.length
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))


@functional_datapipe('unbatch')
class UnBatcherMapDataPipe(MapDataPipe[T]):
    r""" :class:`UnBatcherMapDataPipe`.

    Map DataPipe to undo batching of data. In other words, it flattens the data up to the specified level
    within a batched DataPipe.

    Args:
        datapipe: Iterable DataPipe being un-batched
        unbatch_level: Defaults to `1` (only flattening the top level). If set to `2`, it will flatten the top 2 levels,
            and `-1` will flatten the entire DataPipe.
    """

    def __init__(self,
                 datapipe: MapDataPipe[DataChunk],
                 unbatch_level: int = 1):
        self.datapipe = datapipe
        self.unbatch_level = unbatch_level

    def __getitem__(self, index) -> T:
        # TODO: This is not handling unbatch_level, maybe I should do it once up front?
        try:
            batch_size = len(self.datapipe[0])
            if batch_size == 0:
                raise RuntimeError("Batch 0 is empty")
            batch_idx, sub_idx = index // batch_size, index % batch_size
            return self.datapipe[batch_idx][sub_idx]
        except IndexError:
            raise IndexError(f"Index {index} is invalid")

    # def __iter__(self):
    #     for element in self.datapipe:
    #         for i in self._dive(element, unbatch_level=self.unbatch_level):
    #             yield i

    # def _dive(self, element, unbatch_level):
    #     if unbatch_level < -1:
    #         raise ValueError("unbatch_level must be -1 or >= 0")
    #     if unbatch_level == -1:
    #         if isinstance(element, list) or isinstance(element, DataChunk):
    #             for item in element:
    #                 for i in self._dive(item, unbatch_level=-1):
    #                     yield i
    #         else:
    #             yield element
    #     elif unbatch_level == 0:
    #         yield element
    #     else:
    #         if isinstance(element, list) or isinstance(element, DataChunk):
    #             for item in element:
    #                 for i in self._dive(item, unbatch_level=unbatch_level - 1):
    #                     yield i
    #         else:
    #             raise IndexError(f"unbatch_level {self.unbatch_level} exceeds the depth of the DataPipe")
