from torch.utils.data import MapDataPipe, functional_datapipe, DataChunk
from typing import TypeVar


T = TypeVar('T')


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
