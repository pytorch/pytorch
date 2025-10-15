from collections.abc import Iterator
from io import IOBase
from typing import Optional

from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe


__all__ = ["StreamReaderIterDataPipe"]


@functional_datapipe("read_from_stream")
class StreamReaderIterDataPipe(IterDataPipe[tuple[str, bytes]]):
    r"""
    Given IO streams and their label names, yield bytes with label name as tuple.

    (functional name: ``read_from_stream``).

    Args:
        datapipe: Iterable DataPipe provides label/URL and byte stream
        chunk: Number of bytes to be read from stream per iteration.
            If ``None``, all bytes will be read until the EOF.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import IterableWrapper, StreamReader
        >>> from io import StringIO
        >>> dp = IterableWrapper([("alphabet", StringIO("abcde"))])
        >>> list(StreamReader(dp, chunk=1))
        [('alphabet', 'a'), ('alphabet', 'b'), ('alphabet', 'c'), ('alphabet', 'd'), ('alphabet', 'e')]
    """

    def __init__(
        self, datapipe: IterDataPipe[tuple[str, IOBase]], chunk: Optional[int] = None
    ):
        self.datapipe = datapipe
        self.chunk = chunk

    def __iter__(self) -> Iterator[tuple[str, bytes]]:
        for furl, stream in self.datapipe:
            while True:
                d = stream.read(self.chunk)
                if not d:
                    stream.close()
                    break
                yield (furl, d)
