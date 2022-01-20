import os
import fnmatch
import warnings

from io import IOBase
from typing import Iterable, List, Tuple, Union

try:
    import dill

    # XXX: By default, dill writes the Pickler dispatch table to inject its
    # own logic there. This globally affects the behavior of the standard library
    # pickler for any user who transitively depends on this module!
    # Undo this extension to avoid altering the behavior of the pickler globally.
    dill.extend(use_dill=False)
    DILL_AVAILABLE = True
except ImportError:
    DILL_AVAILABLE = False


def check_lambda_fn(fn):
    # Partial object has no attribute '__name__', but can be pickled
    if hasattr(fn, "__name__") and fn.__name__ == "<lambda>" and not DILL_AVAILABLE:
        warnings.warn(
            "Lambda function is not supported for pickle, please use "
            "regular python function or functools.partial instead."
        )


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
        yield pathname, StreamWrapper(open(pathname, mode))


def validate_pathname_binary_tuple(data: Tuple[str, IOBase]):
    if not isinstance(data, tuple):
        raise TypeError(f"pathname binary data should be tuple type, but it is type {type(data)}")
    if len(data) != 2:
        raise TypeError(f"pathname binary stream tuple length should be 2, but got {len(data)}")
    if not isinstance(data[0], str):
        raise TypeError(f"pathname within the tuple should have string type pathname, but it is type {type(data[0])}")
    if not isinstance(data[1], IOBase) and not isinstance(data[1], StreamWrapper):
        raise TypeError(
            f"binary stream within the tuple should have IOBase or"
            f"its subclasses as type, but it is type {type(data[1])}"
        )


def deprecation_warning(name, new_name: str = ""):
    new_name_statement = ""
    if new_name:
        new_name_statement = f" Please use {new_name} instead."
    warnings.warn(f"{name} and its functional API are deprecated and will be removed from the package `torch`." +
                  new_name_statement, DeprecationWarning)


class StreamWrapper:
    def __init__(self, file_obj):
        self.file_obj = file_obj

    def __getattr__(self, name):
        return getattr(self.file_obj, name)

    def __del__(self):
        self.file_obj.close()

    def __getstate__(self):
        return self.file_obj.__getstate__()

    def __repr__(self):
        return "StreamWrapper<" + repr(self.file_obj) + ">"

    def __iter__(self):
        return self.file_obj.__iter__()

    def __next__(self):
        return self.file_obj.__next__()

    def __setstate__(self, state):
        self.file_obj.__setstate__(state)

    def __sizeof__(self, *args, **kwargs):
        return self.file_obj.__sizeof__(*args, **kwargs)
