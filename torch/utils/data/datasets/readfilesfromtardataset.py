from torch.utils.data.dataset import IterableDataset
from torch.utils.data.datasets.common import extract_files_from_tar_pathname_binaries

from typing import Iterable, Iterator

class ReadFilesFromTarIterableDataset(IterableDataset):
    r""" :class:`ReadFilesFromTarIterableDataset`.

    IterableDataset to extract tar file into binary streams from given pathnames,
    yield pathname and extracted binary stream in a tuple.
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

    def __iter__(self) -> Iterator[tuple]:
        yield from extract_files_from_tar_pathname_binaries(self.dataset)

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
