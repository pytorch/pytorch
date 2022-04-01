from torch.utils.data import MapDataPipe, functional_datapipe
from typing import Sized, Tuple, TypeVar

T_co = TypeVar('T_co', covariant=True)


@functional_datapipe('concat')
class ConcaterMapDataPipe(MapDataPipe):
    r"""
    Concatenate multiple Map DataPipes (functional name: ``concat``).
    The new index of is the cumulative sum of source DataPipes.
    For example, if there are 2 source DataPipes both with length 5,
    index 0 to 4 of the resulting `ConcatMapDataPipe` would refer to
    elements of the first DataPipe, and 5 to 9 would refer to elements
    of the second DataPipe.

    Args:
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


@functional_datapipe('zip')
class ZipperMapDataPipe(MapDataPipe[Tuple[T_co, ...]]):
    r"""
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).
    This MataPipe is out of bound as soon as the shortest input DataPipe is exhausted.

    Args:
        *datapipes: Map DataPipes being aggregated
    """
    datapipes: Tuple[MapDataPipe[T_co], ...]
    length: int

    def __init__(self, *datapipes: MapDataPipe[T_co]) -> None:
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, MapDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `MapDataPipe`")
        if not all(isinstance(dp, Sized) for dp in datapipes):
            raise TypeError("Expected all inputs to be `Sized`")
        self.datapipes = datapipes
        self.length = -1

    def __getitem__(self, index) -> Tuple[T_co, ...]:
        res = []
        for dp in self.datapipes:
            try:
                res.append(dp[index])
            except IndexError:
                raise IndexError(f"Index {index} is out of range for one of the input MapDataPipes {dp}.")
        return tuple(res)

    def __len__(self) -> int:
        if self.length == -1:
            self.length = min(len(dp) for dp in self.datapipes)
        return self.length
