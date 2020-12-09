import os
import fnmatch
import warnings
from typing import List, Union, Iterable
from io import BufferedIOBase

class StreamWrapper:
    # this is a wrapper class which wraps streaming handle
    def __init__(self, stream):
        # if input is a StreamWrapper already, then transfer ownership
        if isinstance(stream, StreamWrapper):
            self.stream = stream()
            stream.reset()
        # only accept streaming obj
        elif isinstance(stream, BufferedIOBase):
            self.stream = stream
        else:
            warnings.warn("StreamWrapper can only wrap BufferedIOBase based obj, but got {}".format(type(stream)))
            raise TypeError

    def reset(self):
        # behavior is undefined if calling any method other than close() after reset() is called
        self.stream = None

    def read(self, *args, **kw):
        # put type ignore here to avoid mypy complaining too many args
        res = self.stream.read(*args, **kw)  # type: ignore
        return res

    def close(self):
        if self.stream:
            self.stream.close()

    def __del__(self):
        self.close()

    def __call__(self):
        return self.stream


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

        yield (pathname, StreamWrapper(open(pathname, 'rb')))
