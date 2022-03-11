from io import IOBase
from typing import Iterable, Tuple, Optional

from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.utils.common import get_file_binaries_from_pathnames, deprecation_warning


class FileOpenerIterDataPipe(IterDataPipe[Tuple[str, IOBase]]):
    r"""
    Given pathnames, opens files and yield pathname and file stream in a tuple.

    Args:
        datapipe: Iterable datapipe that provides pathnames
        mode: An optional string that specifies the mode in which
            the file is opened by ``open()``. It defaults to ``b`` which
            means open for reading in binary mode. Another option is
            to use ``t`` for text mode
        encoding: An optional string that specifies the encoding of the
            underlying file. It defaults to ``None`` to match the default encoding of ``open``.
        length: Nominal length of the datapipe

    Note:
        The opened file handles will be closed by Python's GC periodically. Users can choose
        to close them explicitly.

    Example:
        >>> from torchdata.datapipes.iter import FileLister, FileOpener, StreamReader
        >>> dp = FileLister(root=".").filter(lambda fname: fname.endswith('.txt'))
        >>> dp = FileOpener(dp)
        >>> dp = StreamReader(dp)
        >>> list(dp)
        [('./abc.txt', 'abc')]
    """

    def __init__(
            self,
            datapipe: Iterable[str],
            mode: str = 'r',
            encoding: Optional[str] = None,
            length: int = -1):
        super().__init__()
        self.datapipe: Iterable = datapipe
        self.mode: str = mode
        self.encoding: Optional[str] = encoding

        if self.mode not in ('b', 't', 'rb', 'rt', 'r'):
            raise ValueError("Invalid mode {}".format(mode))
        # TODO: enforce typing for each instance based on mode, otherwise
        #       `argument_validation` with this DataPipe may be potentially broken

        if 'b' in mode and encoding is not None:
            raise ValueError("binary mode doesn't take an encoding argument")

        self.length: int = length

    # Remove annotation due to 'IOBase' is a general type and true type
    # is determined at runtime based on mode. Some `DataPipe` requiring
    # a subtype would cause mypy error.
    def __iter__(self):
        yield from get_file_binaries_from_pathnames(self.datapipe, self.mode, self.encoding)

    def __len__(self):
        if self.length == -1:
            raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
        return self.length


class FileLoaderIterDataPipe(IterDataPipe[Tuple[str, IOBase]]):

    def __new__(
            cls,
            datapipe: Iterable[str],
            mode: str = 'b',
            length: int = -1):
        deprecation_warning(type(cls).__name__, new_name="FileOpener")
        return FileOpenerIterDataPipe(datapipe=datapipe, mode=mode, length=length)
