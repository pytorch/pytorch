import os
import fnmatch
import warnings
from typing import List, Union, Iterable


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


def get_file_binaries_from_pathnames(pathnames : Iterable):

    if not isinstance(pathnames, Iterable):
        warnings.warn("get_file_binaries_from_pathnames needs the input be an Iterable")
        raise TypeError

    for pathname in pathnames:
        if not isinstance(pathname, str):
            warnings.warn("file pathname must be string type, but got {}".format(type(pathname)))
            raise TypeError

        yield (pathname, open(pathname, 'rb'))
