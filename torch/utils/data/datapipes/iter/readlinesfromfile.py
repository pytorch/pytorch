from torch.utils.data import IterDataPipe


class ReadLinesFromFileIterDataPipe(IterDataPipe[str]):
    r""" :class:`ReadLinesFromFileDataPipe`

    Iterable DataPipe to yield line(s) from the given file path.
    args:
        filepath : file path
    """

    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath) as file:
            for line in file:
                yield line
