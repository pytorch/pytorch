from io import IOBase
from typing import Iterable, Tuple

from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.utils.common import get_file_binaries_from_pathnames


class LoadFilesFromDiskIterDataPipe(IterDataPipe[Tuple[str, IOBase]]):
    r""" :class:`LoadFilesFromDiskIterDataPipe`.

    Iterable Datapipe to load file streams from given pathnames,
    yield pathname and file stream in a tuple.
    args:
        datapipe: Iterable datapipe that provides pathnames
        mode: An optional string that specifies the mode in which
            the file is opened by `open()`. It defaults to 'b' which
            means open for reading in binary mode. Another option is
            't' for text mode
        length: a nominal length of the datapipe
    """

    def __init__(
            self,
            datapipe : Iterable[str],
            mode: str = 'b',
            length : int = -1):
        super().__init__()
        self.datapipe: Iterable = datapipe
        self.mode: str = mode
        if self.mode not in ('b', 't', 'rb', 'rt', 'r'):
            raise ValueError("Invalid mode {}".format(mode))
        # TODO: enforce typing for each instance based on mode, otherwise
        #       `argument_validation` with this DataPipe may be potentially broken
        self.length: int = length

    # Remove annotation due to 'IOBase' is a general type and true type
    # is determined at runtime based on mode. Some `DataPipe` requiring
    # a subtype would cause mypy error.
    def __iter__(self):
        yield from get_file_binaries_from_pathnames(self.datapipe, self.mode)

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
