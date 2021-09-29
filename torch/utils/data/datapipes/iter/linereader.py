from typing import Tuple
from torch.utils.data import IterDataPipe


class LineReaderIterDataPipe(IterDataPipe[Tuple[str, str]]):
    r""" :class:`LineReaderIterDataPipe`

    Iterable DataPipe to load file name and stream as source IterDataPipe
    and yield filename and line(s).

    Args:
        datapipe: Iterable DataPipe providing file name and string file stream
    """

    def __init__(self, datapipe):
        self.datapipe = datapipe

    def __iter__(self):
        for file_name, stream in self.datapipe:
            for line in stream:
                yield file_name, line
