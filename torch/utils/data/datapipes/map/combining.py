# mypy: allow-untyped-defs
from collections.abc import Sized
from typing import TypeVar

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import MapDataPipe


__all__ = ["ConcaterMapDataPipe", "ZipperMapDataPipe"]

_T_co = TypeVar("_T_co", covariant=True)


@functional_datapipe("concat")
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

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp1 = SequenceWrapper(range(3))
        >>> dp2 = SequenceWrapper(range(3))
        >>> concat_dp = dp1.concat(dp2)
        >>> list(concat_dp)
        [0, 1, 2, 0, 1, 2]
    """

    datapipes: tuple[MapDataPipe]

    def __init__(self, *datapipes: MapDataPipe):
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, MapDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `MapDataPipe`")
        if not all(isinstance(dp, Sized) for dp in datapipes):
            raise TypeError("Expected all inputs to be `Sized`")
        self.datapipes = datapipes  # type: ignore[assignment]

    def __getitem__(self, index) -> _T_co:  # type: ignore[type-var]
        offset = 0
        for dp in self.datapipes:
            # pyrefly: ignore  # bad-argument-type
            if index - offset < len(dp):
                return dp[index - offset]
            else:
                # pyrefly: ignore  # bad-argument-type
                offset += len(dp)
        raise IndexError(f"Index {index} is out of range.")

    def __len__(self) -> int:
        # pyrefly: ignore  # bad-argument-type
        return sum(len(dp) for dp in self.datapipes)


@functional_datapipe("zip")
class ZipperMapDataPipe(MapDataPipe[tuple[_T_co, ...]]):
    r"""
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).

    This MataPipe is out of bound as soon as the shortest input DataPipe is exhausted.

    Args:
        *datapipes: Map DataPipes being aggregated

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.map import SequenceWrapper
        >>> dp1 = SequenceWrapper(range(3))
        >>> dp2 = SequenceWrapper(range(10, 13))
        >>> zip_dp = dp1.zip(dp2)
        >>> list(zip_dp)
        [(0, 10), (1, 11), (2, 12)]
    """

    datapipes: tuple[MapDataPipe[_T_co], ...]

    def __init__(self, *datapipes: MapDataPipe[_T_co]) -> None:
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, MapDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `MapDataPipe`")
        if not all(isinstance(dp, Sized) for dp in datapipes):
            raise TypeError("Expected all inputs to be `Sized`")
        self.datapipes = datapipes

    def __getitem__(self, index) -> tuple[_T_co, ...]:
        res = []
        for dp in self.datapipes:
            try:
                res.append(dp[index])
            except IndexError as e:
                raise IndexError(
                    f"Index {index} is out of range for one of the input MapDataPipes {dp}."
                ) from e
        return tuple(res)

    def __len__(self) -> int:
        # pyrefly: ignore  # bad-argument-type
        return min(len(dp) for dp in self.datapipes)
