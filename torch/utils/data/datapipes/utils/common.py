import fnmatch
import functools
import inspect
import os
import warnings

from io import IOBase

from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from torch.utils._import_utils import dill_available

__all__ = [
    "validate_input_col",
    "StreamWrapper",
    "get_file_binaries_from_pathnames",
    "get_file_pathnames_from_root",
    "match_masks",
    "validate_pathname_binary_tuple",
]


# BC for torchdata
DILL_AVAILABLE = dill_available()


def validate_input_col(fn: Callable, input_col: Optional[Union[int, tuple, list]]):
    """
    Check that function used in a callable datapipe works with the input column.

    This simply ensures that the number of positional arguments matches the size
    of the input column. The function must not contain any non-default
    keyword-only arguments.

    Examples:
        >>> # xdoctest: +SKIP("Failing on some CI machines")
        >>> def f(a, b, *, c=1):
        >>>     return a + b + c
        >>> def f_def(a, b=1, *, c=1):
        >>>     return a + b + c
        >>> assert validate_input_col(f, [1, 2])
        >>> assert validate_input_col(f_def, 1)
        >>> assert validate_input_col(f_def, [1, 2])

    Notes:
        If the function contains variable positional (`inspect.VAR_POSITIONAL`) arguments,
        for example, f(a, *args), the validator will accept any size of input column
        greater than or equal to the number of positional arguments.
        (in this case, 1).

    Args:
        fn: The function to check.
        input_col: The input column to check.

    Raises:
        ValueError: If the function is not compatible with the input column.
    """
    try:
        sig = inspect.signature(fn)
    except ValueError:  # Signature cannot be inspected, likely it is a built-in fn or written in C
        return
    if isinstance(input_col, (list, tuple)):
        input_col_size = len(input_col)
    else:
        input_col_size = 1

    pos = []
    var_positional = False
    non_default_kw_only = []

    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            pos.append(p)
        elif p.kind is inspect.Parameter.VAR_POSITIONAL:
            var_positional = True
        elif p.kind is inspect.Parameter.KEYWORD_ONLY:
            if p.default is p.empty:
                non_default_kw_only.append(p)
        else:
            continue

    if isinstance(fn, functools.partial):
        fn_name = getattr(fn.func, "__name__", repr(fn.func))
    else:
        fn_name = getattr(fn, "__name__", repr(fn))

    if len(non_default_kw_only) > 0:
        raise ValueError(
            f"The function {fn_name} takes {len(non_default_kw_only)} "
            f"non-default keyword-only parameters, which is not allowed."
        )

    if len(sig.parameters) < input_col_size:
        if not var_positional:
            raise ValueError(
                f"The function {fn_name} takes {len(sig.parameters)} "
                f"parameters, but {input_col_size} are required."
            )
    else:
        if len(pos) > input_col_size:
            if any(p.default is p.empty for p in pos[input_col_size:]):
                raise ValueError(
                    f"The function {fn_name} takes {len(pos)} "
                    f"positional parameters, but {input_col_size} are required."
                )
        elif len(pos) < input_col_size:
            if not var_positional:
                raise ValueError(
                    f"The function {fn_name} takes {len(pos)} "
                    f"positional parameters, but {input_col_size} are required."
                )


def _is_local_fn(fn):
    # Functions or Methods
    if hasattr(fn, "__code__"):
        return fn.__code__.co_flags & inspect.CO_NESTED
    # Callable Objects
    else:
        if hasattr(fn, "__qualname__"):
            return "<locals>" in fn.__qualname__
        fn_type = type(fn)
        if hasattr(fn_type, "__qualname__"):
            return "<locals>" in fn_type.__qualname__
    return False


def _check_unpickable_fn(fn: Callable):
    """
    Check function is pickable or not.

    If it is a lambda or local function, a UserWarning will be raised. If it's not a callable function, a TypeError will be raised.
    """
    if not callable(fn):
        raise TypeError(f"A callable function is expected, but {type(fn)} is provided.")

    # Extract function from partial object
    # Nested partial function is automatically expanded as a single partial object
    if isinstance(fn, partial):
        fn = fn.func

    # Local function
    if _is_local_fn(fn) and not dill_available():
        warnings.warn(
            "Local function is not supported by pickle, please use "
            "regular python function or functools.partial instead."
        )
        return

    # Lambda function
    if hasattr(fn, "__name__") and fn.__name__ == "<lambda>" and not dill_available():
        warnings.warn(
            "Lambda function is not supported by pickle, please use "
            "regular python function or functools.partial instead."
        )
        return


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
        abspath: bool = False,
        non_deterministic: bool = False) -> Iterable[str]:

    # print out an error message and raise the error out
    def onerror(err : OSError):
        warnings.warn(err.filename + " : " + err.strerror)
        raise err

    if os.path.isfile(root):
        path = root
        if abspath:
            path = os.path.abspath(path)
        fname = os.path.basename(path)
        if match_masks(fname, masks):
            yield path
    else:
        for path, dirs, files in os.walk(root, onerror=onerror):
            if abspath:
                path = os.path.abspath(path)
            if not non_deterministic:
                files.sort()
            for f in files:
                if match_masks(f, masks):
                    yield os.path.join(path, f)
            if not recursive:
                break
            if not non_deterministic:
                # Note that this is in-place modifying the internal list from `os.walk`
                # This only works because `os.walk` doesn't shallow copy before turn
                # https://github.com/python/cpython/blob/f4c03484da59049eb62a9bf7777b963e2267d187/Lib/os.py#L407
                dirs.sort()


def get_file_binaries_from_pathnames(pathnames: Iterable, mode: str, encoding: Optional[str] = None):
    if not isinstance(pathnames, Iterable):
        pathnames = [pathnames, ]

    if mode in ('b', 't'):
        mode = 'r' + mode

    for pathname in pathnames:
        if not isinstance(pathname, str):
            raise TypeError(f"Expected string type for pathname, but got {type(pathname)}")
        yield pathname, StreamWrapper(open(pathname, mode, encoding=encoding))


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


# Deprecated function names and its corresponding DataPipe type and kwargs for the `_deprecation_warning` function
_iter_deprecated_functional_names: Dict[str, Dict] = {}
_map_deprecated_functional_names: Dict[str, Dict] = {}


def _deprecation_warning(
    old_class_name: str,
    *,
    deprecation_version: str,
    removal_version: str,
    old_functional_name: str = "",
    old_argument_name: str = "",
    new_class_name: str = "",
    new_functional_name: str = "",
    new_argument_name: str = "",
    deprecate_functional_name_only: bool = False,
) -> None:
    if new_functional_name and not old_functional_name:
        raise ValueError("Old functional API needs to be specified for the deprecation warning.")
    if new_argument_name and not old_argument_name:
        raise ValueError("Old argument name needs to be specified for the deprecation warning.")

    if old_functional_name and old_argument_name:
        raise ValueError("Deprecating warning for functional API and argument should be separated.")

    msg = f"`{old_class_name}()`"
    if deprecate_functional_name_only and old_functional_name:
        msg = f"{msg}'s functional API `.{old_functional_name}()` is"
    elif old_functional_name:
        msg = f"{msg} and its functional API `.{old_functional_name}()` are"
    elif old_argument_name:
        msg = f"The argument `{old_argument_name}` of {msg} is"
    else:
        msg = f"{msg} is"
    msg = (
        f"{msg} deprecated since {deprecation_version} and will be removed in {removal_version}."
        f"\nSee https://github.com/pytorch/data/issues/163 for details."
    )

    if new_class_name or new_functional_name:
        msg = f"{msg}\nPlease use"
        if new_class_name:
            msg = f"{msg} `{new_class_name}()`"
        if new_class_name and new_functional_name:
            msg = f"{msg} or"
        if new_functional_name:
            msg = f"{msg} `.{new_functional_name}()`"
        msg = f"{msg} instead."

    if new_argument_name:
        msg = f"{msg}\nPlease use `{old_class_name}({new_argument_name}=)` instead."

    warnings.warn(msg, FutureWarning)


class StreamWrapper:
    """
    StreamWrapper is introduced to wrap file handler generated by DataPipe operation like `FileOpener`.

    StreamWrapper would guarantee the wrapped file handler is closed when it's out of scope.
    """

    session_streams: Dict[Any, int] = {}
    debug_unclosed_streams: bool = False

    def __init__(self, file_obj, parent_stream=None, name=None):
        self.file_obj = file_obj
        self.child_counter = 0
        self.parent_stream = parent_stream
        self.close_on_last_child = False
        self.name = name
        self.closed = False
        if parent_stream is not None:
            if not isinstance(parent_stream, StreamWrapper):
                raise RuntimeError(f'Parent stream should be StreamWrapper, {type(parent_stream)} was given')
            parent_stream.child_counter += 1
            self.parent_stream = parent_stream
        if StreamWrapper.debug_unclosed_streams:
            StreamWrapper.session_streams[self] = 1

    @classmethod
    def close_streams(cls, v, depth=0):
        """Traverse structure and attempts to close all found StreamWrappers on best effort basis."""
        if depth > 10:
            return
        if isinstance(v, StreamWrapper):
            v.close()
        else:
            # Traverse only simple structures
            if isinstance(v, dict):
                for vv in v.values():
                    cls.close_streams(vv, depth=depth + 1)
            elif isinstance(v, (list, tuple)):
                for vv in v:
                    cls.close_streams(vv, depth=depth + 1)

    def __getattr__(self, name):
        file_obj = self.__dict__['file_obj']
        return getattr(file_obj, name)

    def close(self, *args, **kwargs):
        if self.closed:
            return
        if StreamWrapper.debug_unclosed_streams:
            del StreamWrapper.session_streams[self]
        if hasattr(self, "parent_stream") and self.parent_stream is not None:
            self.parent_stream.child_counter -= 1
            if not self.parent_stream.child_counter and self.parent_stream.close_on_last_child:
                self.parent_stream.close()
        try:
            self.file_obj.close(*args, **kwargs)
        except AttributeError:
            pass
        self.closed = True

    def autoclose(self):
        """Automatically close stream when all child streams are closed or if there are none."""
        self.close_on_last_child = True
        if self.child_counter == 0:
            self.close()

    def __dir__(self):
        attrs = list(self.__dict__.keys()) + list(StreamWrapper.__dict__.keys())
        attrs += dir(self.file_obj)
        return list(set(attrs))

    def __del__(self):
        if not self.closed:
            self.close()

    def __iter__(self):
        yield from self.file_obj

    def __next__(self):
        return next(self.file_obj)

    def __repr__(self):
        if self.name is None:
            return f"StreamWrapper<{self.file_obj!r}>"
        else:
            return f"StreamWrapper<{self.name},{self.file_obj!r}>"

    def __getstate__(self):
        return self.file_obj

    def __setstate__(self, obj):
        self.file_obj = obj
