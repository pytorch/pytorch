from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.utils.common import get_file_pathnames_from_root
from typing import List, Union, Iterator

class ListDirFilesIterDataPipe(IterDataPipe):
    r""" :class:`ListDirFilesIterDataPipe`

    Iterable DataPipe to load file pathname(s) (path + filename), yield pathname from given disk root dir.
    args:
        root : root dir
        mask : a unix style filter string or string list for filtering file name(s)
        abspath : whether to return relative pathname or absolute pathname
        length : a nominal length of the datapipe
    """

    def __init__(
            self,
            root: str = '.',
            masks: Union[str, List[str]] = '',
            *,
            recursive: bool = False,
            abspath: bool = False,
            length: int = -1):
        super().__init__()
        self.root : str = root
        self.masks : Union[str, List[str]] = masks
        self.recursive : bool = recursive
        self.abspath : bool = abspath
        self.length : int = length

    def __iter__(self) -> Iterator[str] :
        yield from get_file_pathnames_from_root(self.root, self.masks, self.recursive, self.abspath)

    def __len__(self):
        if self.length == -1:
            raise NotImplementedError
        return self.length
