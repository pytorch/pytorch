import operator

from functools import reduce
from torch.utils.data import MapDataPipe, functional_datapipe, DataChunk
from typing import List, TypeVar, Union, Optional


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
        if (not isinstance(unbatch_level, int)) or unbatch_level < -1:
            raise RuntimeError(f"unbatch_level must be an integer > -1, got {unbatch_level} instead.")
        self.datapipe = datapipe
        self.deepest_level = self._get_deepest_level()
        self.unbatch_level = unbatch_level if unbatch_level != -1 else self.deepest_level
        self.length: Optional[int] = None

    def _get_batch_sizes(self) -> List[int]:
        sizes = [len(self.datapipe)]
        curr = self.datapipe[0]
        while isinstance(curr, list) or isinstance(curr, DataChunk):
            sizes.append(len(curr))
            curr = curr[0]
        return sizes

    def _get_deepest_level(self) -> int:
        level = 0
        curr: Union[MapDataPipe, DataChunk, List] = self.datapipe
        try:
            while isinstance(curr[0], list) or isinstance(curr[0], DataChunk):
                curr = curr[0]
                level += 1
        except IndexError:
            pass
        return level

    def __getitem__(self, index) -> T:
        batch_sizes: List[int] = self._get_batch_sizes()
        result_index_map, batch_index = [], index
        for i in range(self.unbatch_level):
            relevant_batch_sizes = batch_sizes[i + 1:self.unbatch_level + 1]
            if len(relevant_batch_sizes) == 0:
                raise RuntimeError("Unbatching is not possible because the specified unbatch_level "
                                   f"{self.unbatch_level} exceeds input DataPipe's batch depth.")
            inner_batch_size = reduce(operator.mul, relevant_batch_sizes, 1)
            result_index_map.append(batch_index // inner_batch_size)
            batch_index = batch_index % inner_batch_size
        result_index_map.append(batch_index)
        res = self.datapipe
        for idx in result_index_map:
            try:
                res = res[idx]  # type: ignore[assignment]
            except IndexError:
                raise IndexError(f"Index {index} is not valid. This may be caused by index out of bound"
                                 "or invalid nesting within the input DataPipe.")
            except TypeError:
                raise TypeError("Unable to unbatch this input DataPipe."
                                "This is likely caused by invalid nesting within the input DataPipe.")
        return res  # type: ignore[return-value]

    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        self.length = 0
        while True:
            try:
                self.__getitem__(self.length)
                self.length += 1
            except IndexError:
                break
        return self.length
