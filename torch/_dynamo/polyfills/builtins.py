"""
Python polyfills for builtins
"""

import builtins
from typing import Iterable

from ..decorators import substitute_in_graph


__all__ = [
    "all",
    "any",
]


@substitute_in_graph(builtins.all, can_constant_fold_through=True)
def all(iterable: Iterable[object], /) -> bool:
    for elem in iterable:
        if not elem:
            return False
    return True


@substitute_in_graph(builtins.any, can_constant_fold_through=True)
def any(iterable: Iterable[object], /) -> bool:
    for elem in iterable:
        if elem:
            return True
    return False
