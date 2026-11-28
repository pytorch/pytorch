"""Python polyfills for bisect."""

import bisect

from ..decorators import substitute_in_graph
from . import import_fresh_module


py_bisect = import_fresh_module("bisect", blocked=["_bisect"])

__all__ = ["bisect_left", "bisect_right", "insort_left", "insort_right"]


@substitute_in_graph(bisect.bisect_left)
def bisect_left(a, x, lo=0, hi=None, *, key=None):
    return py_bisect.bisect_left(a, x, lo, hi, key=key)


@substitute_in_graph(bisect.bisect_right)
def bisect_right(a, x, lo=0, hi=None, *, key=None):
    return py_bisect.bisect_right(a, x, lo, hi, key=key)


@substitute_in_graph(bisect.insort_left)
def insort_left(a, x, lo=0, hi=None, *, key=None):
    return py_bisect.insort_left(a, x, lo, hi, key=key)


@substitute_in_graph(bisect.insort_right)
def insort_right(a, x, lo=0, hi=None, *, key=None):
    return py_bisect.insort_right(a, x, lo, hi, key=key)
