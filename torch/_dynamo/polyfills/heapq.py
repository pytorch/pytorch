"""
Python polyfills for builtins
"""

from __future__ import annotations

import heapq

from test.support import import_helper

from ..decorators import substitute_in_graph

py_heapq = import_helper.import_fresh_module('heapq', blocked=['_heapq'])


__all__ = [
    "_heapify_max",
    "_heappop_max",
    "_heapreplace_max",
    "_siftdown",
    "_siftdown_max",
    "_siftup",
    "_siftup_max",
    "heapify",
    "heappop",
    "heappush",
    "heappushpop",
    "heapreplace",
    "merge",
    "nlargest",
    "nsmallest",
]



@substitute_in_graph(heapq._heapify_max)
def _heapify_max(heap):
    return py_heapq._heapify_max(heap)


@substitute_in_graph(heapq._heappop_max)
def _heappop_max(heap):
    return py_heapq._heappop_max(heap)


@substitute_in_graph(heapq._heapreplace_max)
def _heapreplace_max(heap, item):
    return py_heapq._heapreplace_max(heap, item)


@substitute_in_graph(heapq._siftdown)
def _siftdown(heap, startpos, pos):
    return py_heapq._siftdown(heap, startpos, pos)


@substitute_in_graph(heapq._siftdown_max)
def _siftdown_max(heap, startpos, pos):
    return py_heapq._siftdown_max(heap, startpos, pos)


@substitute_in_graph(heapq._siftup)
def _siftup(heap, pos):
    return py_heapq._siftup(heap, pos)


@substitute_in_graph(heapq._siftup_max)
def _siftup_max(heap, pos):
    return py_heapq._siftup_max(heap, pos)


@substitute_in_graph(heapq.heapify)
def heapify(heap, /):
    return py_heapq.heapify(heap)


@substitute_in_graph(heapq.heappop)
def heappop(heap):
    return py_heapq.heappop(heap)


@substitute_in_graph(heapq.heappush)
def heappush(heap, item):
    return py_heapq.heappush(heap, item)


@substitute_in_graph(heapq.heappushpop)
def heappushpop(heap, item):
    return py_heapq.heappushpop(heap, item)


@substitute_in_graph(heapq.heapreplace)
def heapreplace(heap, item):
    return py_heapq.heapreplace(heap, item)


@substitute_in_graph(heapq.merge)
def merge(*iterables, key=None, reverse=False):
    return py_heapq.merge(*iterables, key=key, reverse=reverse)


@substitute_in_graph(heapq.nlargest)
def nlargest(n, iterable, key=None):
    return py_heapq.nlargest(n, iterable, key=key)


@substitute_in_graph(heapq.nsmallest)
def nsmallest(n, iterable, key=None):
    return py_heapq.nsmallest(n, iterable, key=key)
