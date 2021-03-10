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
        self.datapipes = datapipes  # type: ignore
        self.length = None

    def __iter__(self) -> Iterator:
        for dp in self.datapipes:
            for data in dp:
                yield data

    def __len__(self) -> int:
        if self.length is not None:
            if self.length == -1:
                raise NotImplementedError
            return self.length
        if all(isinstance(dp, Sized) and len(dp) >= 0 for dp in self.datapipes):
            self.length = sum(len(dp) for dp in self.datapipes)  # type: ignore
        else:
            self.length = -1
        return len(self)
