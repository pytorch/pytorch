from typing import Tuple
from torch.utils.data import IterDataPipe


class LineReaderIterDataPipe(IterDataPipe[Tuple[str, str]]):
    r""" :class:`LineReaderIterDataPipe`

    Iterable DataPipe to load file name and stream as source IterDataPipe
    and yield filename and line(s).
    """

    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for file_name, stream in self.source_datapipe:
            for line in stream:
                yield file_name, line
