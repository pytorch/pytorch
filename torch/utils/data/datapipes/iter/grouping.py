# mypy: allow-untyped-defs
from collections import defaultdict
from collections.abc import Callable, Iterator, Sized
from typing import Any, NoReturn, TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import DataChunk, IterDataPipe
from torch.utils.data.datapipes.utils.common import _check_unpickable_fn


__all__ = [
    "BatcherIterDataPipe",
    "GrouperIterDataPipe",
    "UnBatcherIterDataPipe",
]


_T_co = TypeVar("_T_co", covariant=True)


def __getattr__(name: str) -> NoReturn:
    raise AttributeError(f"module {__name__} has no attribute {name}")


@functional_datapipe("batch")
class BatcherIterDataPipe(IterDataPipe[DataChunk]):
    r"""
    Creates mini-batches of data (functional name: ``batch``).

    An outer dimension will be added as ``batch_size`` if ``drop_last`` is set to ``True``, or ``length % batch_size`` for the
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

    def __init__(
        self,
        datapipe: IterDataPipe,
        batch_size: int,
        drop_last: bool = False,
        wrapper_class: type[DataChunk] = DataChunk,
    ) -> None:
        if batch_size <= 0:
            raise AssertionError("Batch size is required to be larger than 0!")
        super().__init__()
        self.datapipe = datapipe
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.wrapper_class = wrapper_class

    def __iter__(self) -> Iterator[DataChunk]:
        batch: list = []
        for x in self.datapipe:
            batch.append(x)
            if len(batch) == self.batch_size:
                yield self.wrapper_class(batch)
                batch = []
        if len(batch) > 0:
            if not self.drop_last:
                yield self.wrapper_class(batch)

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized):
            if self.drop_last:
                return len(self.datapipe) // self.batch_size
            else:
                return (len(self.datapipe) + self.batch_size - 1) // self.batch_size
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")


@functional_datapipe("unbatch")
class UnBatcherIterDataPipe(IterDataPipe):
    r"""
    Undos batching of data (functional name: ``unbatch``).

    In other words, it flattens the data up to the specified level within a batched DataPipe.

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

    def __init__(self, datapipe: IterDataPipe, unbatch_level: int = 1) -> None:
        self.datapipe = datapipe
        self.unbatch_level = unbatch_level

    def __iter__(self):
        for element in self.datapipe:
            yield from self._dive(element, unbatch_level=self.unbatch_level)

    def _dive(self, element, unbatch_level):
        if unbatch_level < -1:
            raise ValueError("unbatch_level must be -1 or >= 0")
        if unbatch_level == -1:
            if isinstance(element, (list, DataChunk)):
                for item in element:
                    yield from self._dive(item, unbatch_level=-1)
            else:
                yield element
        elif unbatch_level == 0:
            yield element
        else:
            if isinstance(element, (list, DataChunk)):
                for item in element:
                    yield from self._dive(item, unbatch_level=unbatch_level - 1)
            else:
                raise IndexError(
                    f"unbatch_level {self.unbatch_level} exceeds the depth of the DataPipe"
                )


@functional_datapipe("groupby")
class GrouperIterDataPipe(IterDataPipe[DataChunk]):
    r"""
    Groups data from IterDataPipe by keys from ``group_key_fn``, yielding a ``DataChunk`` with batch size up to ``group_size``.

    (functional name: ``groupby``).

    The samples are read sequentially from the source ``datapipe``, and a batch of samples belonging to the same group
    will be yielded as soon as the size of the batch reaches ``group_size``. When the buffer is full,
    the DataPipe will yield the largest batch with the same key, provided that its size is larger
    than ``guaranteed_group_size``. If its size is smaller, it will be dropped if ``drop_remaining=True``.

    After iterating through the entirety of source ``datapipe``, everything not dropped due to the buffer capacity
    will be yielded from the buffer, even if the group sizes are smaller than ``guaranteed_group_size``.

    Args:
        datapipe: Iterable datapipe to be grouped
        group_key_fn: Function used to generate group key from the data of the source datapipe
        keep_key: Option to yield the matching key along with the items in a tuple,
            resulting in `(key, [items])` otherwise returning [items]
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
        ...     return os.path.basename(file).split(".")[0]
        >>> source_dp = IterableWrapper(
        ...     ["a.png", "b.png", "a.json", "b.json", "a.jpg", "c.json"]
        ... )
        >>> dp0 = source_dp.groupby(group_key_fn=group_fn)
        >>> list(dp0)
        [['a.png', 'a.json', 'a.jpg'], ['b.png', 'b.json'], ['c.json']]
        >>> # A group is yielded as soon as its size equals to `group_size`
        >>> dp1 = source_dp.groupby(group_key_fn=group_fn, group_size=2)
        >>> list(dp1)
        [['a.png', 'a.json'], ['b.png', 'b.json'], ['a.jpg'], ['c.json']]
        >>> # Scenario where `buffer` is full, and group 'a' needs to be yielded since its size > `guaranteed_group_size`
        >>> dp2 = source_dp.groupby(
        ...     group_key_fn=group_fn,
        ...     buffer_size=3,
        ...     group_size=3,
        ...     guaranteed_group_size=2,
        ... )
        >>> list(dp2)
        [['a.png', 'a.json'], ['b.png', 'b.json'], ['a.jpg'], ['c.json']]
    """

    def __init__(
        self,
        datapipe: IterDataPipe[_T_co],
        group_key_fn: Callable[[_T_co], Any],
        *,
        keep_key: bool = False,
        buffer_size: int = 10000,
        group_size: int | None = None,
        guaranteed_group_size: int | None = None,
        drop_remaining: bool = False,
    ) -> None:
        _check_unpickable_fn(group_key_fn)
        # pyrefly: ignore [invalid-type-var]
        self.datapipe = datapipe
        # pyrefly: ignore [invalid-type-var]
        self.group_key_fn = group_key_fn

        self.keep_key = keep_key
        self.max_buffer_size = buffer_size
        self.buffer_elements: defaultdict[Any, list] = defaultdict(list)
        self.curr_buffer_size = 0
        self.group_size = group_size
        self.guaranteed_group_size = None
        if group_size is not None and buffer_size is not None:
            if not (0 < group_size <= buffer_size):
                raise AssertionError("group_size must be > 0 and <= buffer_size")
            # pyrefly: ignore [bad-assignment]
            self.guaranteed_group_size = group_size
        if guaranteed_group_size is not None:
            if group_size is None or not (0 < guaranteed_group_size <= group_size):
                raise AssertionError(
                    "guaranteed_group_size must be > 0 and <= group_size and group_size must be set"
                )
            # pyrefly: ignore [bad-assignment]
            self.guaranteed_group_size = guaranteed_group_size
        self.drop_remaining = drop_remaining
        self.wrapper_class = DataChunk

    def _remove_biggest_key(self):
        biggest_key = None
        biggest_size = 0
        result_to_yield = None
        for findkey in self.buffer_elements:
            if len(self.buffer_elements[findkey]) > biggest_size:
                biggest_size = len(self.buffer_elements[findkey])
                biggest_key = findkey

        if (
            self.guaranteed_group_size is not None
            and biggest_size < self.guaranteed_group_size
            and not self.drop_remaining
        ):
            raise RuntimeError(
                "Failed to group items", str(self.buffer_elements[biggest_key])
            )

        if (
            self.guaranteed_group_size is None
            or biggest_size >= self.guaranteed_group_size
        ):
            result_to_yield = self.buffer_elements[biggest_key]

        self.curr_buffer_size -= biggest_size
        del self.buffer_elements[biggest_key]

        return result_to_yield

    def __iter__(self):
        for x in self.datapipe:
            key = self.group_key_fn(x)

            self.buffer_elements[key].append(x)
            self.curr_buffer_size += 1

            if self.group_size is not None and self.group_size == len(
                self.buffer_elements[key]
            ):
                result: DataChunk[Any] = self.wrapper_class(self.buffer_elements[key])
                yield (key, result) if self.keep_key else result
                self.curr_buffer_size -= len(self.buffer_elements[key])
                del self.buffer_elements[key]

            if self.curr_buffer_size == self.max_buffer_size:
                result_to_yield = self._remove_biggest_key()
                if result_to_yield is not None:
                    result = self.wrapper_class(result_to_yield)
                    yield (key, result) if self.keep_key else result

        for key in tuple(self.buffer_elements.keys()):
            result = self.wrapper_class(self.buffer_elements.pop(key))
            self.curr_buffer_size -= len(result)
            yield (key, result) if self.keep_key else result

    def reset(self) -> None:
        self.curr_buffer_size = 0
        self.buffer_elements = defaultdict(list)

    def __getstate__(self):
        state = (
            self.datapipe,
            self.group_key_fn,
            self.keep_key,
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
            self.keep_key,
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

    def __del__(self) -> None:
        self.buffer_elements.clear()
