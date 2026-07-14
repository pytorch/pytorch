"""
Python polyfills for os
"""

from __future__ import annotations

import os
from typing import AnyStr

from ..decorators import substitute_in_graph


__all__ = ["fspath"]


# Copied from os.py in the standard library
# pyrefly: ignore [bad-argument-type]
@substitute_in_graph(os.fspath, can_constant_fold_through=True)
def fspath(path: AnyStr | os.PathLike[AnyStr]) -> AnyStr:
    if isinstance(path, (str, bytes)):
        # pyrefly: ignore [bad-return]
        return path

    path_type = type(path)
    try:
        path_repr = path_type.__fspath__(path)  # type: ignore[arg-type]
    except AttributeError:
        if hasattr(path_type, "__fspath__"):
            raise
        raise TypeError(
            f"expected str, bytes or os.PathLike object, not {path_type.__name__}",
        ) from None
    if isinstance(path_repr, (str, bytes)):
        return path_repr  # type: ignore[return-value]
    raise TypeError(
        f"expected {path_type.__name__}.__fspath__() to return str or bytes, "
        f"not {type(path_repr).__name__}",
    )
