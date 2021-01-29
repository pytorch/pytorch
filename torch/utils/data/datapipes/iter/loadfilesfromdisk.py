from torch.utils.data.dataset import IterableDataset as IterDataPipe

from torch.utils.data.datapipes.utils.common import get_file_binaries_from_pathnames

from typing import Iterable, Iterator

class LoadFilesFromDiskIterDataPipe(IterDataPipe):
    r""" :class:`LoadFilesFromDiskIterDataPipe`.

    Iterable Datapipe to load file binary streams from given pathnames,
    yield pathname and binary stream in a tuple.
    args:
        datapipe: Iterable datapipe that provides pathnames
        length: a nominal length of the datapipe
    """

    def __init__(
            self,
            datapipe : Iterable,
            length : int = -1):
        super().__init__()
        self.datapipe : Iterable = datapipe
        self.length : int = length

    def __iter__(self) -> Iterator[tuple] :
        yield from get_file_binaries_from_pathnames(self.datapipe)

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
