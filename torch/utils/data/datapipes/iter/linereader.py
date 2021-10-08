from typing import Tuple
from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.utils.common import deprecation_warning_torchdata


class LineReaderIterDataPipe(IterDataPipe[Tuple[str, str]]):
    r""" :class:`LineReaderIterDataPipe`

    Iterable DataPipe to load file name and stream as source IterDataPipe
    and yield filename and line(s).

    Args:
        datapipe: Iterable DataPipe providing file name and string file stream
    """

    def __init__(self, datapipe):
        self.datapipe = datapipe
        deprecation_warning_torchdata(type(self).__name__)

    def __iter__(self):
        for file_name, stream in self.datapipe:
            for line in stream:
                yield file_name, line
