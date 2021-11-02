import warnings

from torch.utils.data import IterDataPipe, functional_datapipe
from typing import Any, Callable, Iterator, List, Optional, Set, Sized, Tuple, TypeVar, Deque
from collections import deque

T_co = TypeVar('T_co', covariant=True)


@functional_datapipe('concat')
class ConcaterIterDataPipe(IterDataPipe):
    r""" :class:`ConcaterIterDataPipe`.

    Iterable DataPipe to concatenate multiple Iterable DataPipes.

    Args:
        datapipes: Iterable DataPipes being concatenated
    """
    datapipes: Tuple[IterDataPipe]
    length: Optional[int]

    def __init__(self, *datapipes: IterDataPipe):
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `IterDataPipe`")
        self.datapipes = datapipes  # type: ignore[assignment]
        self.length = None

    def __iter__(self) -> Iterator:
        for dp in self.datapipes:
            for data in dp:
                yield data

    def __len__(self) -> int:
        if self.length is not None:
            if self.length == -1:
                raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
            return self.length
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            self.length = sum(len(dp) for dp in self.datapipes)
        else:
            self.length = -1
        return len(self)


# This is fake class to show API, going to be replaced by the copy from torchdata
# TODO(VitalyFedyunin): Replace with valid version, documentation and tests
class IterateBuffer(IterDataPipe):

    def __init__(self, buffer):
        self.buffer = buffer

    def __iter__(self):
        for i in self.buffer:
            yield i


@functional_datapipe('fork')
class ForkerIterDataPipe(IterDataPipe):
    r""" :class:`ForkerIterDataPipe`.

        Iterable DataPipe to create multiple instances of the same Iterable DataPipe.

        Args:
            datapipe: Iterable DataPipe being copied
            num_instances: number of instances of the datapipe to create
            buffer_size: this restricts how far ahead the leading child DataPipe
             can read relative to the slowest child DataPipe.
             Use -1 for the unlmited buffer
    """
    def __new__(cls, datapipe: IterDataPipe, num_instances: int, buffer_size: int = 1000):
        if num_instances < 1:
            raise ValueError(f"Expected `num_instaces` larger than 0, but {num_instances} is found")
        if num_instances == 1:
            return datapipe
        container = _ForkerIterDataPipe(datapipe, num_instances, buffer_size)
        return [_ChildDataPipe(container, i) for i in range(num_instances)]


class _ForkerIterDataPipe(IterDataPipe):
    r""" :class:`_ForkerIterDataPipe`.

        Container to hold instance-specific information on behalf of ForkerIterDataPipe. It tracks
        the state of its child DataPipes, maintains the buffer, and yields the next value
        as requested by the child DataPipes.
    """
    def __init__(self, datapipe: IterDataPipe, num_instances: int, buffer_size: int = 1000):
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
        self.child_pointers = [0] * num_instances  # Indicate the indices of the next element to get
        self.slowest_ptr = 0
        self.leading_ptr = 0
        self.end_ptr: Optional[int] = None

    def __len__(self):
        return len(self.main_datapipe)

    def get_next_element_by_instance(self, instance_id: int):
        if self._datapipe_iterator is None:
            self._datapipe_iterator = iter(self.main_datapipe)
        while self.end_ptr is None or self.child_pointers[instance_id] < self.end_ptr:
            if not self.buffer or self.child_pointers[instance_id] > self.leading_ptr:
                self.leading_ptr = self.child_pointers[instance_id]
                if self.buffer_size >= 0 and self.leading_ptr - self.slowest_ptr + 1 > self.buffer_size:
                    raise BufferError("ForkerIterDataPipe buffer overflow," +
                                      f"buffer size {self.buffer_size} is insufficient.")
                try:
                    self.buffer.append(next(self._datapipe_iterator))
                    self.child_pointers[instance_id] += 1
                    yield self.buffer[-1]
                except StopIteration:
                    self.end_ptr = self.leading_ptr
            else:  # Child pointer is slower than or equal to the leading_ptr
                buffer_index = self.child_pointers[instance_id] - self.slowest_ptr
                return_val = self.buffer[buffer_index]
                self.child_pointers[instance_id] += 1
                if self.child_pointers[instance_id] - 1 == self.slowest_ptr:
                    new_min = min(self.child_pointers)  # Can optimize by avoiding the call to min()
                    if self.slowest_ptr < new_min:
                        self.slowest_ptr = new_min
                        self.buffer.popleft()
                yield return_val

    def is_instance_started(self, instance_id: int) -> bool:
        return self.child_pointers[instance_id] != 0

    def is_every_instance_exhausted(self) -> bool:
        return all(self.end_ptr == ptr for ptr in self.child_pointers)

    def reset(self):
        self._datapipe_iterator = iter(self.main_datapipe)
        self.buffer = deque()
        self.child_pointers = [0] * self.num_instances
        self.slowest_ptr = 0
        self.leading_ptr = 0
        self.end_ptr = None

class _ChildDataPipe(IterDataPipe):
    r""" :class:`_ChildDataPipe`.

        Iteratable Datapipe that is a child of a main DataPipe. The instance of this class
        will pass its instance_id to get the next value from its main DataPipe.

        Args:
            main_datapipe: Main DataPipe with a method 'get_next_element_by_instance(instance_id)'
            instance_id: integer identifier of this instance
    """
    def __init__(self, main_datapipe, instance_id: int):
        required_attrs = ["get_next_element_by_instance", "is_instance_started", "is_every_instance_exhausted", "reset"]
        required_ops = [getattr(main_datapipe, attr) for attr in required_attrs]
        if any(not callable(op) for op in required_ops):
            raise NotImplementedError(f"Main Datapipe must have methods {required_attrs} implemented.")
        self.main_datapipe = main_datapipe
        self.instance_id = instance_id

    def __iter__(self):
        if self.main_datapipe.is_instance_started(self.instance_id):  # Only reset if the DataPipe started to read
            if not self.main_datapipe.is_every_instance_exhausted():
                warnings.warn("Some child DataPipes are not exhausted when __iter__ is called. We are resetting "
                              "the buffer and each child DataPipe will read from the start again.", UserWarning)
            self.main_datapipe.reset()
        # We want to separate the code for reset and yield, so that 'reset' exeutes before __next__ is called
        return self.get_generator_by_instance(self.instance_id)

    def __len__(self):
        return len(self.main_datapipe)

    def get_generator_by_instance(self, instance_id: int):
        yield from self.main_datapipe.get_next_element_by_instance(self.instance_id)


@functional_datapipe('demux')
class DemultiplexerIterDataPipe(IterDataPipe):
    r""" :class:`DemultiplexerIterDataPipe`.

        Iterable DataPipe to split the input DataPipe into multiple child DataPipes, using the given
        classification function. A list of the child DataPipes is returned from this operation.

        Args:
            datapipe: Iterable DataPipe being filtered
            num_instances: number of instances of the DataPipe to create
            classifier_fn: a function that maps values to an integer within the range [0, num_instances - 1] or None
            drop_none: defaults to False, if True, the function will skip over elements classified as None
            buffer_size: this defines the maximum number of inputs that the buffer can hold across all child
                DataPipes while waiting for their values to be yielded.
                Use -1 for the unlimited buffer
    """
    def __new__(cls, datapipe: IterDataPipe, num_instances: int,
                classifier_fn: Callable[[T_co], int], drop_none: bool = False, buffer_size: int = 1000):
        if num_instances < 1:
            raise ValueError(f"Expected `num_instaces` larger than 0, but {num_instances} is found")
        # When num_instances == 1, demux can be replaced by filter,
        # but keep it as Demultiplexer for the sake of consistency
        # like throwing Error when classification result is out of o range
        container = _DemultiplexerIterDataPipe(datapipe, num_instances, classifier_fn, drop_none, buffer_size)
        return [_ChildDataPipe(container, i) for i in range(num_instances)]


class _DemultiplexerIterDataPipe(IterDataPipe):
    r""" :class:`_DemultiplexerIterDataPipe`.

        Container to hold instance-specific information on behalf of DemultiplexerIterDataPipe. It tracks
        the state of its child DataPipes, maintains the buffer, classifies and yields the next correct value
        as requested by the child DataPipes.
    """

    def __init__(self, datapipe: IterDataPipe[T_co], num_instances: int,
                 classifier_fn: Callable[[T_co], int], drop_none: bool, buffer_size: int):
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
        self.instance_started: List[bool] = [False] * num_instances
        self.classifier_fn = classifier_fn
        self.drop_none = drop_none
        self.main_datapipe_exhausted = False

    def _find_next(self, instance_id: int) -> T_co:
        while True:
            if self._datapipe_iterator is None:
                raise ValueError("_datapipe_iterator has not been set, likely because this private method is called directly "
                                 "without invoking get_next_element_by_instance() first.")
            value = next(self._datapipe_iterator)
            classification = self.classifier_fn(value)
            if classification is None and self.drop_none:
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
        if self._datapipe_iterator is None:
            self._datapipe_iterator = iter(self.main_datapipe)
        stop = False
        self.instance_started[instance_id] = True
        while not stop:
            if self.child_buffers[instance_id]:
                self.current_buffer_usage -= 1
                yield self.child_buffers[instance_id].popleft()
            else:
                try:
                    yield self._find_next(instance_id)
                except StopIteration:
                    stop = True
                    self.main_datapipe_exhausted = True

    def is_instance_started(self, instance_id: int) -> bool:
        return self.instance_started[instance_id]

    def is_every_instance_exhausted(self) -> bool:
        return self.main_datapipe_exhausted and all(not child_buffer for child_buffer in self.child_buffers)

    def reset(self):
        self._datapipe_iterator = iter(self.main_datapipe)
        self.current_buffer_usage = 0
        self.child_buffers = [deque() for _ in range(self.num_instances)]
        self.instance_started = [False] * self.num_instances
        self.main_datapipe_exhausted = False

@functional_datapipe('mux')
class MultiplexerIterDataPipe(IterDataPipe):
    r""" :class:`MultiplexerIterDataPipe`.

        Iterable DataPipe that yields one element at a time from each input Iterable DataPipe
        (i.e. one element from the 1st input DataPipe, then one element from the 2nd DataPipe in the next iteration,
        and so on). It skips over DataPipes that are exhausted, and ends when all input DataPipes are exhausted.

        Args:
            datapipes: Iterable DataPipes that will take turn to yield their elements, until they are all exhausted
    """
    def __init__(self, *datapipes):
        self.datapipes = datapipes
        self.length: Optional[int] = None

    def __iter__(self):
        iterators = [iter(x) for x in self.datapipes]
        finished: Set[int] = set()
        while len(finished) < len(iterators):
            for i in range(len(iterators)):
                if i not in finished:
                    try:
                        value = next(iterators[i])
                        yield value
                    except StopIteration:
                        finished.add(i)

    def __len__(self):
        if self.length is not None:
            if self.length == -1:
                raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
            return self.length
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            self.length = sum(len(dp) for dp in self.datapipes)
        else:
            self.length = -1
        return len(self)


@functional_datapipe('zip')
class ZipperIterDataPipe(IterDataPipe[Tuple[T_co]]):
    r""" :class:`ZipperIterDataPipe`.

    Iterable DataPipe aggregates elements into a tuple from each of
    the input DataPipe. The output DataPipe is stopped when the
    shortest input DataPipe is exhausted.

    Args:
        *datapipes: Iterable DataPipes being aggregated
    """
    datapipes: Tuple[IterDataPipe]
    length: Optional[int]

    def __init__(self, *datapipes: IterDataPipe):
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError("All inputs are required to be `IterDataPipe` "
                            "for `ZipIterDataPipe`.")
        super().__init__()
        self.datapipes = datapipes  # type: ignore[assignment]
        self.length = None

    def __iter__(self) -> Iterator[Tuple[T_co]]:
        for data in zip(*self.datapipes):
            yield data

    def __len__(self) -> int:
        if self.length is not None:
            if self.length == -1:
                raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
            return self.length
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            self.length = min(len(dp) for dp in self.datapipes)
        else:
            self.length = -1
        return len(self)
