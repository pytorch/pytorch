"""
Python polyfills for sys
"""

from __future__ import annotations

import sys

from ..decorators import substitute_in_graph


__all__ = [
    "intern",
    "getrecursionlimit",
]


# pyrefly: ignore [bad-argument-type]
@substitute_in_graph(sys.intern, can_constant_fold_through=True)
def intern(string: str, /) -> str:
    return string


@substitute_in_graph(sys.getrecursionlimit, can_constant_fold_through=True)
def getrecursionlimit() -> int:
    return sys.getrecursionlimit()


if sys.version_info >= (3, 11):

    @substitute_in_graph(sys.get_int_max_str_digits, can_constant_fold_through=True)
    def get_int_max_str_digits() -> int:
        return sys.get_int_max_str_digits()

    @substitute_in_graph(sys.set_int_max_str_digits, can_constant_fold_through=True)
    def set_int_max_str_digits(maxdigits: int) -> None:
        sys.set_int_max_str_digits(maxdigits)

    __all__ += ["get_int_max_str_digits", "set_int_max_str_digits"]
