from typing import Iterator, List, Sequence, Union

from torch.utils.data import IterDataPipe
from torch.utils.data.datapipes.iter import IterableWrapper
from torch.utils.data.datapipes.utils.common import get_file_pathnames_from_root

class FileListerIterDataPipe(IterDataPipe[str]):
    r"""
    Given path(s) to the root directory, yields file pathname(s) (path + filename) of files within the root directory.
    Multiple root directories can be provided.

    Args:
        root: Root directory or a sequence of root directories
        masks: Unix style filter string or string list for filtering file name(s)
        recursive: Whether to return pathname from nested directories or not
        abspath: Whether to return relative pathname or absolute pathname
        non_deterministic: Whether to return pathname in sorted order or not.
            If ``False``, the results yielded from each root directory will be sorted
        length: Nominal length of the datapipe

    Example:
        >>> from torchdata.datapipes.iter import FileLister
        >>> dp = FileLister(root=".", recursive=True)
        >>> list(dp)
        ['example.py', './data/data.tar']
    """

    def __init__(
        self,
        root: Union[str, Sequence[str], IterDataPipe] = '.',
        masks: Union[str, List[str]] = '',
        *,
        recursive: bool = False,
        abspath: bool = False,
        non_deterministic: bool = False,
        length: int = -1
    ) -> None:
        super().__init__()
        if isinstance(root, str):
            root = [root, ]
        if not isinstance(root, IterDataPipe):
            root = IterableWrapper(root)
        self.datapipe: IterDataPipe = root
        self.masks: Union[str, List[str]] = masks
        self.recursive: bool = recursive
        self.abspath: bool = abspath
        self.non_deterministic: bool = non_deterministic
        self.length: int = length

    def __iter__(self) -> Iterator[str] :
        for path in self.datapipe:
            yield from get_file_pathnames_from_root(path, self.masks, self.recursive, self.abspath, self.non_deterministic)

    def __len__(self):
        if self.length == -1:
            raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))
        return self.length
