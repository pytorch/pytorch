from torch.utils.data.dataset import IterableDataset
from torch.utils.data.datasets.common import (
    extract_files_from_pathname_binaries, extract_files_from_single_tar_pathname_binary)

from typing import Iterable, Iterator

class ReadFilesFromTarIterableDataset(IterableDataset):
    r""" :class:`ReadFilesFromTarIterableDataset`.

    IterableDataset to extract tar binary streams from input iterables
    yield pathname and extracted binary stream in a tuple.
    args:
        dataset: Iterable dataset that provides pathname and tar binary stream in tuples
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
        yield from extract_files_from_pathname_binaries(
            self.dataset, extract_files_from_single_tar_pathname_binary)

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
