from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.utils.common import get_file_pathnames_from_root
from typing import List, Union, Iterator


class FileListerIterDataPipe(IterDataPipe[str]):
    r""" :class:`FileListerIterDataPipe`

    Iterable DataPipe to load file pathname(s) (path + filename), yield pathname from given disk root dir.
    The directories and files are traversed in sorted order starting from the root directory, therefore,
    the result from this DataPipe is deterministic given the same set of arguments and no changes to the source disk.

    Args:
        root: Root directory
        mask: Unix style filter string or string list for filtering file name(s)
        abspath: Whether to return relative pathname or absolute pathname
        length: Nominal length of the datapipe
    """

    def __init__(
            self,
            root: str = '.',
            masks: Union[str, List[str]] = '',
            *,
            recursive: bool = False,
            abspath: bool = False,
            length: int = -1) -> None:
        super().__init__()
        self.root: str = root
        self.masks: Union[str, List[str]] = masks
        self.recursive: bool = recursive
        self.abspath: bool = abspath
        self.length: int = length

    def __iter__(self) -> Iterator[str]:
        yield from get_file_pathnames_from_root(self.root, self.masks, self.recursive, self.abspath)

    def __len__(self) -> int:
        if self.length == -1:
            raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
        return self.length
