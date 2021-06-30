from torch.utils.data import MapDataPipe, functional_datapipe
from typing import Optional, Sized, Tuple, TypeVar

T_co = TypeVar('T_co', covariant=True)


@functional_datapipe('concat')
class ConcatMapDataPipe(MapDataPipe):
    r""" :class:`ConcatMapDataPipe`.

    Map DataPipe to concatenate multiple Map DataPipes.
    args:
        datapipes: Map DataPipes being concatenated
    """
    datapipes: Tuple[MapDataPipe]
    length: Optional[int]

    def __init__(self, *datapipes: MapDataPipe):
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, MapDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `MapDataPipe`")
        self.datapipes = datapipes  # type: ignore[assignment]
        self.length = None

    def __getitem__(self, index) -> T_co:
        offset = 0
        for dp in self.datapipes:
            if isinstance(dp, Sized):
                if index - offset < len(dp):
                    return dp[index - offset]
                else:
                    offset += len(dp)
            else:
                # Regard the datapipe with no valid length as unlimited.
                return dp[index - offset]
        raise ValueError("Index {} is out of range.".format(index))

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
