import os
import fnmatch
import warnings

from io import BufferedIOBase
from typing import Iterable, List, Union


def match_masks(name : str, masks : Union[str, List[str]]) -> bool:
    # empty mask matches any input name
    if not masks:
        return True

    if isinstance(masks, str):
        return fnmatch.fnmatch(name, masks)

    for mask in masks:
        if fnmatch.fnmatch(name, mask):
            return True
    return False


def get_file_pathnames_from_root(
        root: str,
        masks: Union[str, List[str]],
        recursive: bool = False,
        abspath: bool = False) -> Iterable[str]:

    # print out an error message and raise the error out
    def onerror(err : OSError):
        warnings.warn(err.filename + " : " + err.strerror)
        raise err

    for path, dirs, files in os.walk(root, onerror=onerror):
        if abspath:
            path = os.path.abspath(path)
        for f in files:
            if match_masks(f, masks):
                yield os.path.join(path, f)
        if not recursive:
            break


def get_file_binaries_from_pathnames(pathnames: Iterable, mode: str):
    if not isinstance(pathnames, Iterable):
        pathnames = [pathnames, ]

    if mode in ('b', 't'):
        mode = 'r' + mode

    for pathname in pathnames:
        if not isinstance(pathname, str):
            raise TypeError("Expected string type for pathname, but got {}"
                            .format(type(pathname)))
        yield (pathname, open(pathname, mode))


def validate_pathname_binary_tuple(data):
    if not isinstance(data, tuple):
        raise TypeError("pathname binary data should be tuple type, but got {}".format(type(data)))
    if len(data) != 2:
        raise TypeError("pathname binary tuple length should be 2, but got {}".format(str(len(data))))
    if not isinstance(data[0], str):
        raise TypeError("pathname binary tuple should have string type pathname, but got {}".format(type(data[0])))
    if not isinstance(data[1], BufferedIOBase):
        raise TypeError("pathname binary tuple should have BufferedIOBase based binary type, but got {}".format(type(data[1])))
