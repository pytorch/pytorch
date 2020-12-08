from torch.utils.data.dataset import IterableDataset
from torch.utils.data.datasets.common import get_file_binaries_from_pathnames

from typing import Iterable, Iterator

class LoadFilesFromDiskIterableDataset(IterableDataset):
    r""" :class:`LoadFilesFromDiskIterableDataset`.

    IterableDataset to load file binary streams from given pathnames,
    yield pathname and binary stream in a tuple.
    args:
        dataset: Iterable dataset that provides pathnames
        length: a nominal length of the dataset
    """

    def __init__(
            self,
            dataset : Iterable,
            length : int = -1):
        super().__init__()
        self.dataset : Iterable = dataset
        self.length : int = length

    def __iter__(self) -> Iterator[tuple] :
        yield from get_file_binaries_from_pathnames(self.dataset)

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
