from collections import defaultdict
from enum import IntEnum

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe, DataChunk
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn
from typing import Any, Callable, DefaultDict, Dict, Iterator, List, Optional, Sized, Tuple, TypeVar

__all__ = [
    "BatcherIterDataPipe",
    "GrouperIterDataPipe",
    "ShardingFilterIterDataPipe",
    "SHARDING_PRIORITIES",
    "UnBatcherIterDataPipe",
]

T_co = TypeVar('T_co', covariant=True)


class SHARDING_PRIORITIES(IntEnum):
    DEFAULT = 1
    DISTRIBUTED = 2
    MULTIPROCESSING = 3


@functional_datapipe('sharding_filter')
class ShardingFilterIterDataPipe(IterDataPipe):
    r"""
    Wrapper that allows DataPipe to be sharded (functional name: ``sharding_filter``). After ``apply_sharding`` is
    called, each instance of the DataPipe (on different workers) will have every `n`-th element of the
    original DataPipe, where `n` equals to the number of instances.

    Args:
        source_datapipe: Iterable DataPipe that will be sharded
    """

    def __init__(self, source_datapipe: IterDataPipe, sharding_group_filter=None):
        self.source_datapipe = source_datapipe
        self.sharding_group_filter = sharding_group_filter
        self.groups: Dict[int, Tuple[int, int]] = {}
        self.num_of_instances = 1
        self.instance_id = 0
        self._update_num_of_instances()

    def is_shardable(self):
        return True

    def apply_sharding(self, num_of_instances, instance_id, sharding_group=SHARDING_PRIORITIES.DEFAULT):
        if instance_id >= num_of_instances:
            raise ValueError(f"instance_id({instance_id}) should be smaller than num_of_instances({num_of_instances})")
        if sharding_group == SHARDING_PRIORITIES.DEFAULT:
            if len(self.groups) and SHARDING_PRIORITIES.DEFAULT not in self.groups:
                raise Exception('ShardingFilter cannot mix DEFAULT and non DEFAULT groups')
        else:
            if SHARDING_PRIORITIES.DEFAULT in self.groups:
                raise Exception('ShardingFilter cannot mix DEFAULT and non DEFAULT groups')
        self.groups[sharding_group] = (num_of_instances, instance_id)
        self._update_num_of_instances()

    def _update_num_of_instances(self):
        sorted_sharding_groups = []
        for key in sorted(self.groups.keys()):
            if self.sharding_group_filter is None or key == self.sharding_group_filter:
                sorted_sharding_groups.append(self.groups[key])

        sorted_sharding_groups.reverse()

        self.num_of_instances = 1
        self.instance_id = 0

        for group_num_of_instances, group_instance_id in sorted_sharding_groups:
            self.instance_id += self.num_of_instances * group_instance_id
            self.num_of_instances *= group_num_of_instances

    def __iter__(self):
        for i, item in enumerate(self.source_datapipe):
            if i % self.num_of_instances == self.instance_id:
                yield item

    def __len__(self):
        if isinstance(self.source_datapipe, Sized):
            return len(self.source_datapipe) // self.num_of_instances +\
                (1 if (self.instance_id < len(self.source_datapipe) % self.num_of_instances) else 0)
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))


@functional_datapipe('batch')
class BatcherIterDataPipe(IterDataPipe[DataChunk]):
    r"""
    Creates mini-batches of data (functional name: ``batch``). An outer dimension will be added as
    ``batch_size`` if ``drop_last`` is set to ``True``, or ``length % batch_size`` for the
    last batch if ``drop_last`` is set to ``False``.

    Args:
        datapipe: Iterable DataPipe being batched
        batch_size: The size of each batch
        drop_last: Option to drop the last batch if it's not full
        wrapper_class: wrapper to apply onto each batch (type ``List``) before yielding,
            defaults to ``DataChunk``

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp = IterableWrapper(range(10))
        >>> dp = dp.batch(batch_size=3, drop_last=True)
        >>> list(dp)
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """
    datapipe: IterDataPipe
    batch_size: int
    drop_last: bool
    length: Optional[int]

    def __init__(self,
                 datapipe: IterDataPipe,
                 batch_size: int,
                 drop_last: bool = False,
                 wrapper_class=DataChunk,
                 ) -> None:
        assert batch_size > 0, "Batch size is required to be larger than 0!"
        super().__init__()
        self.datapipe = datapipe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.length = None
        self.wrapper_class = wrapper_class

    def __iter__(self) -> Iterator[DataChunk]:
        batch: List = []
        for x in self.datapipe:
            batch.append(x)
            if len(batch) == self.batch_size:
                yield self.wrapper_class(batch)
                batch = []
        if len(batch) > 0:
            if not self.drop_last:
                yield self.wrapper_class(batch)

    def __len__(self) -> int:
        if self.length is not None:
            return self.length
        if isinstance(self.datapipe, Sized):
            if self.drop_last:
                self.length = len(self.datapipe) // self.batch_size
            else:
                self.length = (len(self.datapipe) + self.batch_size - 1) // self.batch_size
            return self.length
        raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))


@functional_datapipe('unbatch')
class UnBatcherIterDataPipe(IterDataPipe):
    r"""
    Undoes batching of data (functional name: ``unbatch``). In other words, it flattens the data up to the specified level
    within a batched DataPipe.

    Args:
        datapipe: Iterable DataPipe being un-batched
        unbatch_level: Defaults to ``1`` (only flattening the top level). If set to ``2``,
            it will flatten the top two levels, and ``-1`` will flatten the entire DataPipe.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper([[[0, 1], [2]], [[3, 4], [5]], [[6]]])
        >>> dp1 = source_dp.unbatch()
        >>> list(dp1)
        [[0, 1], [2], [3, 4], [5], [6]]
        >>> dp2 = source_dp.unbatch(unbatch_level=2)
        >>> list(dp2)
        [0, 1, 2, 3, 4, 5, 6]
    """

    def __init__(self,
                 datapipe: IterDataPipe,
                 unbatch_level: int = 1):
        self.datapipe = datapipe
        self.unbatch_level = unbatch_level

    def __iter__(self):
        for element in self.datapipe:
            for i in self._dive(element, unbatch_level=self.unbatch_level):
                yield i

    def _dive(self, element, unbatch_level):
        if unbatch_level < -1:
            raise ValueError("unbatch_level must be -1 or >= 0")
        if unbatch_level == -1:
            if isinstance(element, list) or isinstance(element, DataChunk):
                for item in element:
                    for i in self._dive(item, unbatch_level=-1):
                        yield i
            else:
                yield element
        elif unbatch_level == 0:
            yield element
        else:
            if isinstance(element, list) or isinstance(element, DataChunk):
                for item in element:
                    for i in self._dive(item, unbatch_level=unbatch_level - 1):
                        yield i
            else:
                raise IndexError(f"unbatch_level {self.unbatch_level} exceeds the depth of the DataPipe")


@functional_datapipe('groupby')
class GrouperIterDataPipe(IterDataPipe[DataChunk]):
    r"""
    Groups data from input IterDataPipe by keys which are generated from ``group_key_fn``,
    and yields a ``DataChunk`` with batch size up to ``group_size`` if defined (functional name: ``groupby``).

    The samples are read sequentially from the source ``datapipe``, and a batch of samples belonging to the same group
    will be yielded as soon as the size of the batch reaches ``group_size``. When the buffer is full,
    the DataPipe will yield the largest batch with the same key, provided that its size is larger
    than ``guaranteed_group_size``. If its size is smaller, it will be dropped if ``drop_remaining=True``.

    After iterating through the entirety of source ``datapipe``, everything not dropped due to the buffer capacity
    will be yielded from the buffer, even if the group sizes are smaller than ``guaranteed_group_size``.

    Args:
        datapipe: Iterable datapipe to be grouped
        group_key_fn: Function used to generate group key from the data of the source datapipe
        buffer_size: The size of buffer for ungrouped data
        group_size: The max size of each group, a batch is yielded as soon as it reaches this size
        guaranteed_group_size: The guaranteed minimum group size to be yielded in case the buffer is full
        drop_remaining: Specifies if the group smaller than ``guaranteed_group_size`` will be dropped from buffer
            when the buffer is full

    Example:
        >>> import os
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def group_fn(file):
        ...    return os.path.basename(file).split(".")[0]
        >>> source_dp = IterableWrapper(["a.png", "b.png", "a.json", "b.json", "a.jpg", "c.json"])
        >>> dp0 = source_dp.groupby(group_key_fn=group_fn)
        >>> list(dp0)
        [['a.png', 'a.json', 'a.jpg'], ['b.png', 'b.json'], ['c.json']]
        >>> # A group is yielded as soon as its size equals to `group_size`
        >>> dp1 = source_dp.groupby(group_key_fn=group_fn, group_size=2)
        >>> list(dp1)
        [['a.png', 'a.json'], ['b.png', 'b.json'], ['a.jpg'], ['c.json']]
        >>> # Scenario where `buffer` is full, and group 'a' needs to be yielded since its size > `guaranteed_group_size`
        >>> dp2 = source_dp.groupby(group_key_fn=group_fn, buffer_size=3, group_size=3, guaranteed_group_size=2)
        >>> list(dp2)
        [['a.png', 'a.json'], ['b.png', 'b.json'], ['a.jpg'], ['c.json']]
    """
    def __init__(self,
                 datapipe: IterDataPipe[T_co],
                 group_key_fn: Callable,
                 *,
                 buffer_size: int = 10000,
                 group_size: Optional[int] = None,
                 guaranteed_group_size: Optional[int] = None,
                 drop_remaining: bool = False):
        _check_unpickable_fn(group_key_fn)
        self.datapipe = datapipe
        self.group_key_fn = group_key_fn

        self.max_buffer_size = buffer_size
        self.buffer_elements: DefaultDict[Any, List] = defaultdict(list)
        self.curr_buffer_size = 0
        self.group_size = group_size
        self.guaranteed_group_size = None
        if group_size is not None and buffer_size is not None:
            assert 0 < group_size <= buffer_size
            self.guaranteed_group_size = group_size
        if guaranteed_group_size is not None:
            assert group_size is not None and 0 < guaranteed_group_size <= group_size
            self.guaranteed_group_size = guaranteed_group_size
        self.drop_remaining = drop_remaining
        self.wrapper_class = DataChunk

    def _remove_biggest_key(self):
        biggest_key = None
        biggest_size = 0
        result_to_yield = None
        for findkey in self.buffer_elements.keys():
            if len(self.buffer_elements[findkey]) > biggest_size:
                biggest_size = len(self.buffer_elements[findkey])
                biggest_key = findkey

        if self.guaranteed_group_size is not None and biggest_size < self.guaranteed_group_size and not self.drop_remaining:
            raise RuntimeError('Failed to group items', str(self.buffer_elements[biggest_key]))

        if self.guaranteed_group_size is None or biggest_size >= self.guaranteed_group_size:
            result_to_yield = self.buffer_elements[biggest_key]

        self.curr_buffer_size -= biggest_size
        del self.buffer_elements[biggest_key]

        return result_to_yield

    def __iter__(self):
        for x in self.datapipe:
            key = self.group_key_fn(x)

            self.buffer_elements[key].append(x)
            self.curr_buffer_size += 1

            if self.group_size is not None and self.group_size == len(self.buffer_elements[key]):
                yield self.wrapper_class(self.buffer_elements[key])
                self.curr_buffer_size -= len(self.buffer_elements[key])
                del self.buffer_elements[key]

            if self.curr_buffer_size == self.max_buffer_size:
                result_to_yield = self._remove_biggest_key()
                if result_to_yield is not None:
                    yield self.wrapper_class(result_to_yield)

        for key in tuple(self.buffer_elements.keys()):
            res = self.buffer_elements.pop(key)
            self.curr_buffer_size -= len(res)
            yield self.wrapper_class(res)

    def reset(self) -> None:
        self.curr_buffer_size = 0
        self.buffer_elements = defaultdict(list)

    def __getstate__(self):
        state = (
            self.datapipe,
            self.group_key_fn,
            self.max_buffer_size,
            self.group_size,
            self.guaranteed_group_size,
            self.drop_remaining,
            self.wrapper_class,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.datapipe,
            self.group_key_fn,
            self.max_buffer_size,
            self.group_size,
            self.guaranteed_group_size,
            self.drop_remaining,
            self.wrapper_class,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self.curr_buffer_size = 0
        self.buffer_elements = defaultdict(list)

    def __del__(self):
        self.buffer_elements.clear()
