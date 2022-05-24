from typing import Optional, Tuple
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe

__all__ = ["StreamReaderIterDataPipe", ]


@functional_datapipe('read_from_stream')
class StreamReaderIterDataPipe(IterDataPipe[Tuple[str, bytes]]):
    r"""
    Given label names and IO streams, yields label
    name and bytes in a tuple (functional name: ``read_from_stream``).

    Args:
        datapipe: Iterable DataPipe provides tuples of label/URL and byte stream
        chunk: Number of bytes to be read from stream per iteration.
            If ``None``, all bytes will be read util the EOF.
        break_if_empty: By default, this is set to ``True``, and the iteration ends as soon as a chunk read is empty.
            Setting to ``False``, is useful for streams that send emply chunks as "keep alive" chunks.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper, StreamReader
        >>> from io import StringIO
        >>> dp = IterableWrapper([("alphabet", StringIO("abcde"))])
        >>> list(StreamReader(dp, chunk=1))
        [('alphabet', 'a'), ('alphabet', 'b'), ('alphabet', 'c'), ('alphabet', 'd'), ('alphabet', 'e')]
    """
    def __init__(self, datapipe: IterDataPipe, chunk: Optional[int] = None, break_if_empty: bool = True):
        self.datapipe: IterDataPipe = datapipe
        self.chunk_size: Optional[int] = chunk
        self.break_if_empty = break_if_empty

    def __iter__(self):
        for furl, stream in self.datapipe:
            while True:
                data = stream.read(self.chunk_size)
                if not data:
                    if self.break_if_empty:
                        break
                    else:
                        continue
                yield furl, data
