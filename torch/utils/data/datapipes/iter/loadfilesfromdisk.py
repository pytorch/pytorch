from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.utils.common import get_file_binaries_from_pathnames
from typing import Iterable, Iterator, Tuple
from io import BufferedIOBase

class LoadFilesFromDiskIterDataPipe(IterDataPipe[Tuple[str, BufferedIOBase]]):
    r""" :class:`LoadFilesFromDiskIterDataPipe`.

    Iterable Datapipe to load file binary streams from given pathnames,
    yield pathname and binary stream in a tuple.
    args:
        datapipe: Iterable datapipe that provides pathnames
        length: a nominal length of the datapipe
    """

    def __init__(
            self,
            datapipe : Iterable[str],
            length : int = -1):
        super().__init__()
        self.datapipe : Iterable = datapipe
        self.length : int = length

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]] :
        yield from get_file_binaries_from_pathnames(self.datapipe)

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
