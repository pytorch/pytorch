import functools

from torch.utils.data import IterDataPipe, functional_datapipe
from typing import Iterator, Optional, Sized, Tuple, TypeVar

T_co = TypeVar('T_co', covariant=True)


@functional_datapipe('concat')
class ConcatIterDataPipe(IterDataPipe):
    r""" :class:`ConcatIterDataPipe`.

    Iterable DataPipe to concatenate multiple Iterable DataPipes.
    args:
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
class ForkIterDataPipe(IterDataPipe):

    def __new__(cls, datapipe, instances):
        result = []
        buffer = list(datapipe)
        return [IterateBuffer(buffer) for i in range(instances)]


@functional_datapipe('demux')
class DemultiplexerIterDataPipe(IterDataPipe):

    def __new__(cls, datapipe, instances, classifier_fn):
        result = []
        buffer = list(datapipe)

        def filter_fn(classifier_fn, i, x):
            return classifier_fn(x) == i
        return [IterateBuffer(buffer).filter(functools.partial(filter_fn, classifier_fn, i)) for i in range(instances)]

@functional_datapipe('mux')
class MultiplexerIterDataPipe(IterDataPipe):

    def __init__(self, *datapipes):
        self.datapipes = datapipes

    def __iter__(self):
        iterators = [iter(x) for x in self.datapipes]
        finished = {}
        had_more = True
        while had_more:
            had_more = False
            for i in range(len(iterators)):
                if i not in finished:
                    try:
                        value = iterators[i].__next__()
                        had_more = True
                        yield value
                    except StopIteration:
                        finished[i] = 1


@functional_datapipe('zip')
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
