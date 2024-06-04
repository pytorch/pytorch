# mypy: allow-untyped-defs
import warnings

from abc import ABC, abstractmethod
from collections import deque
import copy as copymodule
from typing import Any, Callable, Iterator, List, Literal, Optional, Sized, Tuple, TypeVar, Deque

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes._hook_iterator import _SnapshotState
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper, _check_unpickable_fn

__all__ = [
    "ConcaterIterDataPipe",
    "DemultiplexerIterDataPipe",
    "ForkerIterDataPipe",
    "MultiplexerIterDataPipe",
    "ZipperIterDataPipe",
]

T_co = TypeVar('T_co', covariant=True)


@functional_datapipe('concat')
class ConcaterIterDataPipe(IterDataPipe):
    r"""
    Concatenates multiple Iterable DataPipes (functional name: ``concat``).

    The resulting DataPipe will yield all the elements from the first input DataPipe, before yielding from the subsequent ones.

    Args:
        datapipes: Iterable DataPipes being concatenated

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> import random
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1 = IterableWrapper(range(3))
        >>> dp2 = IterableWrapper(range(5))
        >>> list(dp1.concat(dp2))
        [0, 1, 2, 0, 1, 2, 3, 4]
    """

    datapipes: Tuple[IterDataPipe]

    def __init__(self, *datapipes: IterDataPipe):
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `IterDataPipe`")
        self.datapipes = datapipes  # type: ignore[assignment]

    def __iter__(self) -> Iterator:
        for dp in self.datapipes:
            yield from dp

    def __len__(self) -> int:
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            return sum(len(dp) for dp in self.datapipes)
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")


@functional_datapipe('fork')
class ForkerIterDataPipe(IterDataPipe):
    r"""
    Creates multiple instances of the same Iterable DataPipe (functional name: ``fork``).

    Args:
        datapipe: Iterable DataPipe being copied
        num_instances: number of instances of the datapipe to create
        buffer_size: this restricts how far ahead the leading child DataPipe
           can read relative to the slowest child DataPipe.
           Defaults to ``1000``. Use ``-1`` for the unlimited buffer.
        copy: copy strategy to use for items yielded by each branch. Supported
            options are ``None`` for no copying, ``"shallow"`` for shallow object
            copies, and ``"deep"`` for deep object copies. Defaults to ``None``.

    Note:
        All branches of the forked pipeline return the identical object unless
        the copy parameter is supplied. If the object is mutable or contains
        mutable objects, changing them in one branch will affect all others.

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(range(5))
        >>> dp1, dp2 = source_dp.fork(num_instances=2)
        >>> list(dp1)
        [0, 1, 2, 3, 4]
        >>> list(dp2)
        [0, 1, 2, 3, 4]
    """

    def __new__(
        cls,
        datapipe: IterDataPipe,
        num_instances: int,
        buffer_size: int = 1000,
        copy: Optional[Literal["shallow", "deep"]] = None
    ):
        if num_instances < 1:
            raise ValueError(f"Expected `num_instances` larger than 0, but {num_instances} is found")
        if num_instances == 1:
            return datapipe
        container = _ForkerIterDataPipe(datapipe, num_instances, buffer_size, copy)  # type: ignore[abstract]
        return [_ChildDataPipe(container, i) for i in range(num_instances)]


class _ContainerTemplate(ABC):
    r"""Abstract class for container ``DataPipes``. The followings are three required methods."""

    @abstractmethod
    def get_next_element_by_instance(self, instance_id: int):
        ...

    @abstractmethod
    def is_every_instance_exhausted(self) -> bool:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def get_length_by_instance(self, instance_id: int):
        r"""Raise TypeError if it's not supposed to be implemented to support `list(datapipe)`."""


def _no_op(x):
    return x


class _ForkerIterDataPipe(IterDataPipe, _ContainerTemplate):
    r"""
    Container to hold instance-specific information on behalf of ForkerIterDataPipe.

    It tracks the state of its child DataPipes, maintains the buffer, and yields the next value
    as requested by the child DataPipes.
    """

    def __init__(
        self,
        datapipe: IterDataPipe,
        num_instances: int,
        buffer_size: int = 1000,
        copy: Optional[Literal["shallow", "deep"]] = None
    ):
        self.main_datapipe = datapipe
        self._datapipe_iterator: Optional[Iterator[Any]] = None
        self.num_instances = num_instances
        self.buffer: Deque = deque()
        self.buffer_size = buffer_size
        if self.buffer_size < 0:
            warnings.warn(
                "Unlimited buffer size is set for `fork`, "
                "please be aware of OOM at random places",
                UserWarning
            )
        if copy is None:
            self.copy_fn = _no_op
        elif copy == "shallow":
            self.copy_fn = copymodule.copy
        elif copy == "deep":
            self.copy_fn = copymodule.deepcopy
        else:
            raise ValueError(f"Unknown copy method `{copy}` requested, choose one of None, `shallow` or `deep`.")

        self.child_pointers: List[int] = [0] * num_instances  # Indicate the indices of the next element to get
        self.slowest_ptr = 0  # The index to read by the slowest child
        self.leading_ptr = 0  # The index to read by the fastest child
        self.end_ptr: Optional[int] = None  # The index to stop child
        self._child_stop: List[bool] = [True for _ in range(num_instances)]

    def __len__(self):
        return len(self.main_datapipe)

    def get_next_element_by_instance(self, instance_id: int):
        if self._datapipe_iterator is None and self._child_stop[instance_id]:
            self._datapipe_iterator = iter(self.main_datapipe)
            self._snapshot_state = _SnapshotState.Iterating
            for i in range(self.num_instances):
                self._child_stop[i] = False
        try:
            while not self._child_stop[instance_id]:
                self.child_pointers[instance_id] += 1
                if self.end_ptr is not None and self.child_pointers[instance_id] == self.end_ptr:
                    self._child_stop[instance_id] = True
                    break
                # Use buffer
                if self.buffer and self.child_pointers[instance_id] <= self.leading_ptr:
                    idx = self.child_pointers[instance_id] - self.slowest_ptr - 1
                    return_val = self.buffer[idx]
                else:  # Retrieve one element from main datapipe
                    self.leading_ptr = self.child_pointers[instance_id]
                    try:
                        return_val = next(self._datapipe_iterator)  # type: ignore[arg-type]
                        self.buffer.append(return_val)
                    except StopIteration:
                        self._child_stop[instance_id] = True
                        self._datapipe_iterator = None
                        self.end_ptr = self.leading_ptr
                        continue
                if self.child_pointers[instance_id] == self.slowest_ptr + 1:
                    new_min = min(self.child_pointers)  # Can optimize by avoiding the call to min()
                    if self.slowest_ptr < new_min:
                        self.slowest_ptr = new_min
                        self.buffer.popleft()
                if self.buffer_size >= 0 and self.leading_ptr > self.buffer_size + self.slowest_ptr:
                    raise BufferError("ForkerIterDataPipe buffer overflow," +
                                      f"buffer size {self.buffer_size} is insufficient.")

                yield self.copy_fn(return_val)  # type: ignore[possibly-undefined]
        finally:
            self._child_stop[instance_id] = True
            # Cleanup _datapipe_iterator for the case that fork exits earlier
            if all(self._child_stop):
                self._datapipe_iterator = None
                self._cleanup()

    def is_every_instance_exhausted(self) -> bool:
        return self.end_ptr is not None and all(self._child_stop)

    def get_length_by_instance(self, instance_id: int) -> int:
        return len(self.main_datapipe)

    def reset(self) -> None:
        self._datapipe_iterator = None
        self.buffer = deque()
        self.child_pointers = [0] * self.num_instances
        self.slowest_ptr = 0
        self.leading_ptr = 0
        self.end_ptr = None
        self._child_stop = [True for _ in range(self.num_instances)]

    def __getstate__(self):
        state = (
            self.main_datapipe,
            self.num_instances,
            self.buffer_size,
            self.copy_fn,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.main_datapipe,
            self.num_instances,
            self.buffer_size,
            self.copy_fn,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self._datapipe_iterator = None
        self.buffer = deque()
        self.child_pointers = [0] * self.num_instances
        self.slowest_ptr = 0
        self.leading_ptr = 0
        self.end_ptr = None
        self._child_stop = [True for _ in range(self.num_instances)]

    def _cleanup(self):
        while self.buffer:
            d = self.buffer.popleft()
            StreamWrapper.close_streams(d)

    def __del__(self):
        self._cleanup()


class _ChildDataPipe(IterDataPipe):
    r"""
    Iterable Datapipe that is a child of a main DataPipe.

    The instance of this class will pass its instance_id to get the next value from its main DataPipe.

    Note:
        ChildDataPipe, like all other IterDataPipe, follows the single iterator per IterDataPipe constraint.
        Since ChildDataPipes share a common buffer, when an iterator is created for one of the ChildDataPipes,
        the previous iterators  for all ChildDataPipes must be invalidated, with the exception when a ChildDataPipe
        hasn't had an iterator created from it since the last invalidation. See the example below.

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> # Singler Iterator per IteraDataPipe Invalidation
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(range(10))
        >>> cdp1, cdp2 = source_dp.fork(num_instances=2)
        >>> it1, it2 = iter(cdp1), iter(cdp2)
        >>> it3 = iter(cdp1)
        >>> # The line above invalidates `it1` and `it2`, and resets `ForkerIterDataPipe`.
        >>> it4 = iter(cdp2)
        >>> # The line above doesn't invalidate `it3`, because an iterator for `cdp2` hasn't been created since
        >>> # the last invalidation.

    Args:
        main_datapipe: Main DataPipe with a method 'get_next_element_by_instance(instance_id)'
        instance_id: integer identifier of this instance
    """

    _is_child_datapipe: bool = True

    def __init__(self, main_datapipe: IterDataPipe, instance_id: int):
        assert isinstance(main_datapipe, _ContainerTemplate)

        self.main_datapipe: IterDataPipe = main_datapipe
        self.instance_id = instance_id

    def __iter__(self):
        # Note that the logic behind setting iterator ID and `reset` are handled within `hook_iterator`
        # We want to separate the code for reset and yield, so that 'reset' executes before __next__ is called
        return self.main_datapipe.get_next_element_by_instance(self.instance_id)

    def __len__(self):
        return self.main_datapipe.get_length_by_instance(self.instance_id)

    # This method is called by `hook_iterator` in `_typing.py`.
    def _set_main_datapipe_valid_iterator_id(self) -> int:
        r"""
        Update the valid iterator ID for both this DataPipe object and `main_datapipe`.

        `main_datapipe.reset()` is called when the ID is incremented to a new generation.
        """
        # 1. First time any child iterator is created
        if self.main_datapipe._valid_iterator_id is None:
            self.main_datapipe._valid_iterator_id = 0  # type: ignore[attr-defined]
        # 2. This instance was already in the same generation as `main_datapipe`,
        #    we need to increment the ID further by 1
        elif self.main_datapipe._valid_iterator_id == self._valid_iterator_id:  # type: ignore[has-type]
            self.main_datapipe._valid_iterator_id += 1  # type: ignore[attr-defined]
            # Whenever a new generation of iterator is created, the `main_datapipe` must reset
            if not self.main_datapipe.is_every_instance_exhausted():
                warnings.warn("Some child DataPipes are not exhausted when __iter__ is called. We are resetting "
                              "the buffer and each child DataPipe will read from the start again.", UserWarning)
            self.main_datapipe.reset()
        # 3. Otherwise, the iterator is behind the others, so it will just need to catch up by setting
        #    the instance's iterator to match that of `main_datapipe`
        self._valid_iterator_id = self.main_datapipe._valid_iterator_id
        return self._valid_iterator_id

    # This method is called by `hook_iterator` in `_typing.py`.
    def _check_valid_iterator_id(self, iterator_id) -> bool:
        r"""Check the valid iterator ID against that of DataPipe object and that of `main_datapipe`."""
        return iterator_id == self._valid_iterator_id and iterator_id == self.main_datapipe._valid_iterator_id


@functional_datapipe('demux')
class DemultiplexerIterDataPipe(IterDataPipe):
    r"""
    Splits the input DataPipe into multiple child DataPipes, using the given classification function (functional name: ``demux``).

    A list of the child DataPipes is returned from this operation.

    Args:
        datapipe: Iterable DataPipe being filtered
        num_instances: number of instances of the DataPipe to create
        classifier_fn: a function that maps values to an integer within the range ``[0, num_instances - 1]`` or ``None``
        drop_none: defaults to ``False``, if ``True``, the function will skip over elements classified as ``None``
        buffer_size: this defines the maximum number of inputs that the buffer can hold across all child
            DataPipes while waiting for their values to be yielded.
            Defaults to ``1000``. Use ``-1`` for the unlimited buffer.

    Examples:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def odd_or_even(n):
        ...     return n % 2
        >>> source_dp = IterableWrapper(range(5))
        >>> dp1, dp2 = source_dp.demux(num_instances=2, classifier_fn=odd_or_even)
        >>> list(dp1)
        [0, 2, 4]
        >>> list(dp2)
        [1, 3]
        >>> # It can also filter out any element that gets `None` from the `classifier_fn`
        >>> def odd_or_even_no_zero(n):
        ...     return n % 2 if n != 0 else None
        >>> dp1, dp2 = source_dp.demux(num_instances=2, classifier_fn=odd_or_even_no_zero, drop_none=True)
        >>> list(dp1)
        [2, 4]
        >>> list(dp2)
        [1, 3]
    """

    def __new__(cls, datapipe: IterDataPipe, num_instances: int,
                classifier_fn: Callable[[T_co], Optional[int]], drop_none: bool = False, buffer_size: int = 1000):
        if num_instances < 1:
            raise ValueError(f"Expected `num_instances` larger than 0, but {num_instances} is found")

        _check_unpickable_fn(classifier_fn)

        # When num_instances == 1, demux can be replaced by filter,
        # but keep it as Demultiplexer for the sake of consistency
        # like throwing Error when classification result is out of o range
        container = _DemultiplexerIterDataPipe(datapipe, num_instances, classifier_fn, drop_none, buffer_size)  # type: ignore[abstract]
        return [_ChildDataPipe(container, i) for i in range(num_instances)]


class _DemultiplexerIterDataPipe(IterDataPipe, _ContainerTemplate):
    r"""
    Container to hold instance-specific information on behalf of DemultiplexerIterDataPipe.

    It tracks the state of its child DataPipes, maintains the buffer, classifies and yields the next correct value
    as requested by the child DataPipes.
    """

    def __init__(self, datapipe: IterDataPipe[T_co], num_instances: int,
                 classifier_fn: Callable[[T_co], Optional[int]], drop_none: bool, buffer_size: int):
        self.main_datapipe = datapipe
        self._datapipe_iterator: Optional[Iterator[Any]] = None
        self.num_instances = num_instances
        self.buffer_size = buffer_size
        if self.buffer_size < 0:
            warnings.warn(
                "Unlimited buffer size is set for `demux`, "
                "please be aware of OOM at random places",
                UserWarning
            )
        self.current_buffer_usage = 0
        self.child_buffers: List[Deque[T_co]] = [deque() for _ in range(num_instances)]
        self.classifier_fn = classifier_fn
        self.drop_none = drop_none
        self.main_datapipe_exhausted = False
        self._child_stop: List[bool] = [True for _ in range(num_instances)]

    def _find_next(self, instance_id: int) -> T_co:  # type: ignore[type-var]
        while True:
            if self.main_datapipe_exhausted or self._child_stop[instance_id]:
                raise StopIteration
            if self._datapipe_iterator is None:
                raise ValueError(
                    "_datapipe_iterator has not been set, likely because this private method is called directly "
                    "without invoking get_next_element_by_instance() first.")
            value = next(self._datapipe_iterator)
            classification = self.classifier_fn(value)
            if classification is None and self.drop_none:
                StreamWrapper.close_streams(value)
                continue
            if classification is None or classification >= self.num_instances or classification < 0:
                raise ValueError(f"Output of the classification fn should be between 0 and {self.num_instances - 1}. " +
                                 f"{classification} is returned.")
            if classification == instance_id:
                return value
            self.child_buffers[classification].append(value)
            self.current_buffer_usage += 1
            if self.buffer_size >= 0 and self.current_buffer_usage > self.buffer_size:
                raise BufferError(
                    f"DemultiplexerIterDataPipe buffer overflow, buffer size {self.buffer_size} is insufficient.")

    def get_next_element_by_instance(self, instance_id: int):
        if self._datapipe_iterator is None and self._child_stop[instance_id]:
            self._datapipe_iterator = iter(self.main_datapipe)
            self._snapshot_state = _SnapshotState.Iterating  # This is necessary for the DataPipe to reset properly.
            self.main_datapipe_exhausted = False
            for i in range(self.num_instances):
                self._child_stop[i] = False

        try:
            while not self._child_stop[instance_id]:
                if self.child_buffers[instance_id]:
                    self.current_buffer_usage -= 1
                    yield self.child_buffers[instance_id].popleft()
                else:
                    try:
                        yield self._find_next(instance_id)
                    except StopIteration:
                        self._child_stop[instance_id] = True
                        self.main_datapipe_exhausted = True
                        self._datapipe_iterator = None
        finally:
            self._child_stop[instance_id] = True
            # Cleanup _datapipe_iterator for the case that demux exits earlier
            if all(self._child_stop):
                self._datapipe_iterator = None
            if self.child_buffers[instance_id]:
                self._cleanup(instance_id)

    def is_every_instance_exhausted(self) -> bool:
        return self.main_datapipe_exhausted and all(self._child_stop)

    def get_length_by_instance(self, instance_id: int) -> int:
        raise TypeError

    def reset(self) -> None:
        self._datapipe_iterator = None
        self.current_buffer_usage = 0
        self.child_buffers = [deque() for _ in range(self.num_instances)]
        self._child_stop = [True for _ in range(self.num_instances)]
        self.main_datapipe_exhausted = False

    def __getstate__(self):
        state = (
            self.main_datapipe,
            self.num_instances,
            self.buffer_size,
            self.classifier_fn,
            self.drop_none,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.main_datapipe,
            self.num_instances,
            self.buffer_size,
            self.classifier_fn,
            self.drop_none,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self._datapipe_iterator = None
        self.current_buffer_usage = 0
        self.child_buffers = [deque() for _ in range(self.num_instances)]
        self._child_stop = [True for _ in range(self.num_instances)]
        self.main_datapipe_exhausted = False

    def _cleanup(self, instance_id: Optional[int] = None):
        ids = range(self.num_instances) if instance_id is None else [instance_id, ]
        for i in ids:
            q = self.child_buffers[i]
            while q:
                d = q.popleft()
                StreamWrapper.close_streams(d)


    def __del__(self):
        self._cleanup()


@functional_datapipe('mux')
class MultiplexerIterDataPipe(IterDataPipe):
    r"""
    Yields one element at a time from each of the input Iterable DataPipes (functional name: ``mux``).

    As in, one element from the 1st input DataPipe, then one element from the 2nd DataPipe in the next iteration,
    and so on. It ends when the shortest input DataPipe is exhausted.

    Args:
        datapipes: Iterable DataPipes that will take turn to yield their elements, until the shortest DataPipe is exhausted

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = IterableWrapper(range(3)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
        >>> list(dp1.mux(dp2, dp3))
        [0, 10, 20, 1, 11, 21, 2, 12, 22]
    """

    def __init__(self, *datapipes):
        self.datapipes = datapipes
        self.buffer: List = []  # Store values to be yielded only when every iterator provides one

    def __iter__(self):
        iterators = [iter(x) for x in self.datapipes]
        while len(iterators):
            for it in iterators:
                try:
                    value = next(it)
                    self.buffer.append(value)
                except StopIteration:
                    self.buffer.clear()
                    return
            yield from self.buffer
            self.buffer.clear()

    def __len__(self):
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            return min(len(dp) for dp in self.datapipes) * len(self.datapipes)
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")

    def reset(self) -> None:
        self.buffer = []

    def __getstate__(self):
        state = (
            self.datapipes,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.datapipes,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self.buffer = []

    def __del__(self):
        self.buffer.clear()


@functional_datapipe('zip')
class ZipperIterDataPipe(IterDataPipe[Tuple[T_co]]):
    r"""
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).

    The output is stopped as soon as the shortest input DataPipe is exhausted.

    Args:
        *datapipes: Iterable DataPipes being aggregated

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = IterableWrapper(range(5)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
        >>> list(dp1.zip(dp2, dp3))
        [(0, 10, 20), (1, 11, 21), (2, 12, 22), (3, 13, 23), (4, 14, 24)]
    """

    datapipes: Tuple[IterDataPipe]

    def __init__(self, *datapipes: IterDataPipe):
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError("All inputs are required to be `IterDataPipe` "
                            "for `ZipIterDataPipe`.")
        super().__init__()
        self.datapipes = datapipes  # type: ignore[assignment]

    def __iter__(self) -> Iterator[Tuple[T_co]]:
        iterators = [iter(datapipe) for datapipe in self.datapipes]
        yield from zip(*iterators)

    def __len__(self) -> int:
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            return min(len(dp) for dp in self.datapipes)
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
