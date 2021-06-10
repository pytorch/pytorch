from typing import Tuple
from torch.utils.data import IterDataPipe


class ToBytesIterDataPipe(IterDataPipe[Tuple[str, bytes]]):
    r""" :class:`ToBytesIterDataPipe`

    Iterable DataPipe to load IO stream with label name,
    yield bytes with label name in a tuple
    args:
        max_limit : maximum bytes to read from stream
    """
    def __init__(self, source_datapipe, max_limit=None):
        self.source_datapipe = source_datapipe
        self.max_limit = max_limit

    def __iter__(self):
        for (furl, stream) in self.source_datapipe:
            d = stream.read(self.max_limit)
            stream.close()
            yield(furl, d)
