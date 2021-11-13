import warnings

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
        if (not isinstance(unbatch_level, int)) or unbatch_level < -1:
            raise RuntimeError(f"unbatch_level must be an integer > -1, got {unbatch_level} instead.")
        self.datapipe = datapipe
        self.deepest_level = self._get_deepest_level()
        self.unbatch_level = unbatch_level if unbatch_level != -1 else self.deepest_level
        self.length = None

    def _get_batch_sizes(self) -> List[int]:
        sizes = [len(self.datapipe)]
        curr = self.datapipe.__getitem__(0)
        while isinstance(curr, list) or isinstance(curr, DataChunk):
            sizes.append(len(curr))
            curr = curr[0]
        return sizes

    def _get_deepest_level(self) -> int:
        level = 0
        curr = self.datapipe
        try:
            while isinstance(curr[0], list) or isinstance(curr[0], DataChunk):
                curr = curr[0]
                level += 1
        except IndexError:
            pass
        return level

    def _unbatch_up_to_level(self, target, unbatch_level: int) -> T:
        # print("UNBATCH UP TO LEVEL")
        # print(f"target: {target}")
        # print(f"unbatch_level: {unbatch_level}")
        if unbatch_level == 0 or not (isinstance(target[0], list) or isinstance(target[0], DataChunk)):
            return target
        else:
            res = []
            for ele in target:
                ele_res = self._unbatch_up_to_level(ele, unbatch_level - 1)
                for child in ele_res:
                    res.append(child)
            # print(res)
            return res

    # This only unbatch a specific level
    # def _unbatch_deeper_level(self, target, unbatch_level: int) -> T:
    #     if unbatch_level == 1:
    #         if isinstance(target[0], list) or isinstance(target[0], DataChunk):  # Unbatching is possible
    #             res = []
    #             for ele in target:
    #                 for child in ele:
    #                     res.append(child)
    #             return res
    #         else:  # Unbatching is impossible
    #             warnings.warn("Unbatching is not possible because the elements are not batches.")
    #             return target
    #     else:
    #         return [self._unbatch_deeper_level(ls, unbatch_level - 1) for ls in target]

    def __getitem__(self, index) -> T:
        batch_sizes: List[int] = self._get_batch_sizes()
        # print(f"__getitem__({index})")
        if self.unbatch_level == 0:
            return self.datapipe[index]
        if self.unbatch_level == 1:
            if len(batch_sizes) == 1:  # No unbatching is possible
                warnings.warn("Unbatching is not possible because the elements are not batches.")
                return self.datapipe[index]
            else:  # Unbatching is possible
                curr_batch_size = batch_sizes[1]
                first_index, second_index = index // curr_batch_size, index % curr_batch_size
                return self.datapipe[first_index][second_index]
        else:  # We should go to a deeper level
            # return self._unbatch_deeper_level(self.datapipe[index], self.unbatch_level - 1)
            # print("Unbatch Level >= 2")
            return self._unbatch_up_to_level(self.datapipe, self.unbatch_level)[index]

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
