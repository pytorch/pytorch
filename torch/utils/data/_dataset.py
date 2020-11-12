# Note: The entire file is in testing phase. Please do not import!

from .dataset import Dataset as MapDataset
from .dataset import IterableDataset as IterDataset

from .common import get_file_pathnames_from_root, get_file_binaries_from_pathnames, extract_files_from_pathname_binaries

from typing import Union, List, Iterable


class ListDirFilesMapDataset(MapDataset):
    def __init__(self, root: str = '.', mask: str = '*.tar'):
        super().__init__()
        self.root : str = root
        self.mask : str = mask


class LoadFilesFromDiskMapDataset(MapDataset):
    def __init__(self, dataset: ListDirFilesMapDataset):
        super().__init__()
        self.dataset = dataset



class ListDirFilesIterDataset(IterDataset):
    def __init__(self, root: str = '.', mask: str = '*.tar', recursive: bool = False, abspath: bool = False):
        super().__init__()
        self.root : str = root
        self.mask : str = mask
        self.recursive : bool = recursive
        self.abspath : bool = abspath
        self.files : Union[None, List] = None

    def _lazy_init(self):
        self.files = get_file_pathnames_from_root(self.root, self.mask, self.recursive, self.abspath)


    def __iter__(self):
        if not self.is_loaded:
            self._lazy_init()
        for file_name in self.files:
            yield file_name

    def __len__(self):
        if not self.is_loaded:
            self._lazy_init()
        return len(self.files)

    @property
    def is_loaded(self):
        return self.files is not None


class LoadFilesFromDiskIterDataset(IterDataset):
    def __init__(
            self,
            input : Iterable,
            auto_extract : bool = False):
        super().__init__()
        self.input : Iterable = input
        self.auto_extract = auto_extract
        self.files : Union[None, List] = None

    def _lazy_init(self):
        self.files = get_file_binaries_from_pathnames(self.input, self.auto_extract)

    def __iter__(self):
        if not self.is_loaded:
            self._lazy_init()
        for file_name_binary_tuple in self.files:
            yield file_name_binary_tuple

    def __len__(self):
        if not self.is_loaded:
            self._lazy_init()
        return len(self.files)

    @property
    def is_loaded(self):
        return self.files is not None


class ExtractFilesIterDataset(IterDataset):
    def __init__(
            self,
            input : Iterable,
            recursive : bool = True):
        super().__init__()
        self.input : Iterable = input
        self.recursive : bool = recursive
        self.files : Union[None, List] = None

    def _lazy_init(self):
        self.files = extract_files_from_pathname_binaries(self.input, self.recursive)

    def __iter__(self):
        if not self.is_loaded:
            self._lazy_init()
        for file_name_binary_tuple in self.files:
            yield file_name_binary_tuple

    def __len__(self):
        if not self.is_loaded:
            self._lazy_init()
        return len(self.files)

    @property
    def is_loaded(self):
        return self.files is not None
