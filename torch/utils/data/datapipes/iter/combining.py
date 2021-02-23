from collections import defaultdict, deque
from torch.utils.data import IterDataPipe
from typing import Iterator, Optional, Sequence, Sized, Tuple, TypeVar

T_co = TypeVar('T_co', covariant=True)


class UnzipIterDataPipe(IterDataPipe):
    r""" :class:`UnzipIterDataPipe`.

    The inverse of :class:`ZipIterDataPipe`, this class disaggregates
    the elements of the zipped iterable DataPipe. The first element
    determines the number of output DataPipe. A :class:`ZipIterDataPipe`
    is often attached later to aggregate these output DataPipes back.
    Return a tuple of split DataPipes.
    args:
        datapipe: Iterable DataPipe being disaggregated
    """
    datapipe: IterDataPipe[Sequence]
    splits: int
    _it: Iterator[Sequence]

    def __new__(cls, datapipe: IterDataPipe[Sequence]):
        source_dp = super().__new__(cls)
        it = iter(datapipe)
        try:
            data = next(it)
            if not isinstance(data, Sequence):
                raise TypeError("Element from `datapipe` is required being a, "
                                "Sequence, but {} is found.".format(type(data)))
            splits = len(data)
            source_dp.__init__(datapipe, splits)
            return tuple(_SplitIterDataPipe(source_dp, i) for i in range(splits))
        except StopIteration:
            raise TypeError("`datapipe` is required having available data "
                            "for `UnzipIterDataPipe.")

    def __init__(self, datapipe, splits):
        self.datapipe = datapipe
        self.splits = splits
        # Status to check if the split has finished processing
        # End: True
        # In-process: False
        self._end = {sp: True for sp in range(self.splits)}
        # Flag to check if all splits are stopped before reset().
        # It prevents the iterator being reset when any other split
        # is still in process.
        self._stopped = True
        self._it = iter(self.datapipe)
        self._buffer = defaultdict(deque)

    def get(self, split_id):
        if len(self._buffer[split_id]) > 0:
            return self._buffer[split_id].popleft()
        try:
            data = next(self._it)
        except StopIteration:
            self._end[split_id] = True
            # Set the flag whenever all splits finish processing
            if all(self._end.values()):
                self._stopped = True
            raise StopIteration
        if not isinstance(data, Sequence):
            raise RuntimeError("Each element from `datapipe` is required being "
                               "a Sequence, but {} is found.".format(type(data)))
        if len(data) != self.splits:
            raise RuntimeError("Each element from `datapipe` is required having "
                               "equal length ({} vs {})."
                               .format(self.splits, len(data)))
        for i in range(self.splits):
            self._buffer[i].append(data[i])
        return self.get(split_id)

    def reset(self, split_id):
        if not self._end[split_id] or not self._stopped:
            raise RuntimeError("Can not reset `UnzipIterDataPipe` when it's "
                               "still in process.")
        else:
            # First reset() will reset the iterator
            if all(self._end.values()):
                self._it = iter(self.datapipe)
            self._end[split_id] = False
            # Last reset() will change flag to prevent any reset() is called
            # by any split when new iteration has started and not finished yet
            if not any(self._end.values()):
                self._stopped = False

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized) and len(self.datapipe) >= 0:
            return len(self.datapipe)
        raise NotImplementedError


class _SplitIterDataPipe(IterDataPipe[T_co]):
    r""" :class:`_SplitIterDataPipe`.

    Iterable DataPipe aggregates elements into a tuple from each of
    the input DataPipe. The output DataPipe is stopped when the
    shortest input DataPipe is exhausted.
    args:
        datapipe: Iterable DataPipes being aggregated
    """
    datapipe: UnzipIterDataPipe
    split_id: int

    def __init__(self, datapipe: UnzipIterDataPipe, split_id: int):
        super().__init__()
        self.datapipe = datapipe
        self.split_id = split_id

    def __iter__(self) -> Iterator[T_co]:
        # Reset UnzipIterDataPipe iterator
        self.datapipe.reset(self.split_id)
        while True:
            try:
                yield self.datapipe.get(self.split_id)
            except StopIteration:
                break

    def __len__(self) -> int:
        if isinstance(self.datapipe, Sized) and len(self.datapipe) >= 0:
            return len(self.datapipe)
        raise NotImplementedError


class ZipIterDataPipe(IterDataPipe[Tuple[T_co]]):
    r""" :class:`ZipIterDataPipe`.

    Iterable DataPipe aggregates elements into a tuple from each of
    the input DataPipe. The output DataPipe is stopped when the
    shortest input DataPipe is exhausted.
    args:
        *datapipes: Iterable DataPipes being aggregated
    """
    datapipes: Tuple[IterDataPipe]
    length: Optional[int]

    def __init__(self, *datapipes: IterDataPipe):
        if not all(tuple(isinstance(dp, IterDataPipe) for dp in datapipes)):
            raise TypeError("All inputs are equired to be `IterDataPipe` "
                            "for `ZipIterDataPipe`.")
        super().__init__()
        self.datapipes = datapipes  # type: ignore
        self.length = None

    def __iter__(self) -> Iterator[Tuple[T_co]]:
        for data in zip(*self.datapipes):
            yield data

    def __len__(self) -> int:
        if self.length is not None:
            if self.length == -1:
                raise NotImplementedError
            return self.length
        if all(list(isinstance(dp, Sized) for dp in self.datapipes)):
            self.length = min(list(len(dp) for dp in self.datapipes))  # type: ignore
        else:
            self.length = -1
        return len(self)
