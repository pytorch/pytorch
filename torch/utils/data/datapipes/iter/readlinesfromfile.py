from typing import Tuple
from torch.utils.data import IterDataPipe


class ReadLinesFromFileIterDataPipe(IterDataPipe[Tuple[str, str]]):
    r""" :class:`ReadLinesFromFileDataPipe`

    Iterable DataPipe to load file names as source iter data pipe
    and yield filename and line(s).
    """

    def __init__(self, source_datapipe):
        self.source_datapipe = source_datapipe

    def __iter__(self):
        for file_name in self.source_datapipe:
            with open(file_name) as file:
                for line in file:
                    yield (file_name, line)
