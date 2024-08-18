"""
Python polyfills for os
"""

import os
from typing import AnyStr

from ..decorators import substitute_in_graph


@substitute_in_graph(os.fspath)
def fspath(path: os.PathLike[AnyStr]) -> AnyStr:
    # Copied from os.py in the standard library
    if isinstance(path, (str, bytes)):
        return path

    path_type = type(path)
    try:
        path_repr = path_type.__fspath__(path)
    except AttributeError:
        if hasattr(path_type, "__fspath__"):
            raise
        raise TypeError(
            f"expected str, bytes or os.PathLike object, not {path_type.__name__}",
        ) from None
    if isinstance(path_repr, (str, bytes)):
        return path_repr
    raise TypeError(
        f"expected {path_type.__name__}.__fspath__() to return str or bytes, "
        f"not {type(path_repr).__name__}",
    )
