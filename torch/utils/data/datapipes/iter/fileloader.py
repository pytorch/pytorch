from collections import defaultdict
from io import IOBase
from typing import DefaultDict, Iterable, Tuple, IO, Any, List, Union

from torch.utils.data import IterDataPipe


class FileLoaderIterDataPipe(IterDataPipe[Tuple[str, IOBase]]):
    r""" :class:`FileLoaderIterDataPipe`.

    Iterable Datapipe to load file streams from given pathnames,
    yield pathname and file stream in a tuple.

    Args:
        datapipe: Iterable datapipe that provides pathnames
        mode: An optional string that specifies the mode in which
            the file is opened by `open()`. It defaults to 'b' which
            means open for reading in binary mode. Another option is
            't' for text mode
        length: Nominal length of the datapipe

    Note:
        The opened file handles will be closed by Python's GC periodly. Users can choose
        to close them explicityly.
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
        self.open_streams: DefaultDict[str, List[Union[IO[Any]]]] = defaultdict(list)

    # Remove annotation due to 'IOBase' is a general type and true type
    # is determined at runtime based on mode. Some `DataPipe` requiring
    # a subtype would cause mypy error.
    def __iter__(self):
        pathnames = self.datapipe if isinstance(self.datapipe, Iterable) else [self.datapipe, ]
        mode = self.mode if self.mode not in ('b', 't') else 'r' + self.mode
        for pathname in pathnames:
            if not isinstance(pathname, str):
                raise TypeError(f"Expected string type for pathname, but got {type(pathname)}")
            stream = open(pathname, mode)
            self.open_streams[pathname].append(stream)
            yield (pathname, stream)

    def __len__(self):
        if self.length == -1:
            raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
        return self.length

    def close_all_streams(self):
        for _name, streams_list in self.open_streams.items():
            for stream in streams_list:
                stream.close()
        self.open_streams = defaultdict(list)
