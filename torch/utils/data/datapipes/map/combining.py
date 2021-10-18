from torch.utils.data import MapDataPipe, functional_datapipe
from typing import Sized, Tuple, TypeVar

T_co = TypeVar('T_co', covariant=True)


@functional_datapipe('concat')
class ConcaterMapDataPipe(MapDataPipe):
    r""" :class:`ConcaterMapDataPipe`.

    Map DataPipe to concatenate multiple Map DataPipes.
    The actual index of is the cumulative sum of source datapipes.
    For example, if there are 2 source datapipes both with length 5,
    index 0 to 4 of the resulting `ConcatMapDataPipe` would refer to
    elements of the first datapipe, and 5 to 9 would refer to elements
    of the second datapipe.
    args:
        datapipes: Map DataPipes being concatenated
    """
    datapipes: Tuple[MapDataPipe]
    length: int

    def __init__(self, *datapipes: MapDataPipe):
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, MapDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `MapDataPipe`")
        if not all(isinstance(dp, Sized) for dp in datapipes):
            raise TypeError("Expected all inputs to be `Sized`")
        self.datapipes = datapipes  # type: ignore[assignment]
        self.length = -1

    def __getitem__(self, index) -> T_co:
        offset = 0
        for dp in self.datapipes:
            if index - offset < len(dp):
                return dp[index - offset]
            else:
                offset += len(dp)
        raise IndexError("Index {} is out of range.".format(index))

    def __len__(self) -> int:
        if self.length == -1:
            self.length = sum(len(dp) for dp in self.datapipes)
        return self.length
