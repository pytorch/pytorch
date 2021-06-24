from typing import Tuple
from torch.utils.data import IterDataPipe


class ToBytesIterDataPipe(IterDataPipe[Tuple[str, bytes]]):
    r""" :class:`ToBytesIterDataPipe`

    Iterable DataPipe to load IO stream with label name,
    and to yield bytes with label name in a tuple
    args:
        chunk : bytes to read from stream on each iteration.
                If None, stream reads to the EOF.
    """
    def __init__(self, source_datapipe, chunk=None):
        self.source_datapipe = source_datapipe
        self.chunk = chunk

    def __iter__(self):
        for (furl, stream) in self.source_datapipe:
            while True:
                d = stream.read(self.chunk)
                if not d:
                    break
                yield (furl, d)
