"""
Python polyfills for builtins
"""

from __future__ import annotations

import _imp

import contextlib
import heapq
import importlib
import sys
from typing import Callable, TypeVar

from ..decorators import substitute_in_graph


_T = TypeVar("_T")
_R = TypeVar("_R")
KeyFn = Callable[[_T], _R]


def _save_and_remove_modules(names):  # type: ignore[no-untyped-def]
    orig_modules = {}
    prefixes = tuple(name + "." for name in names)
    for modname in list(sys.modules):
        if modname in names or modname.startswith(prefixes):
            orig_modules[modname] = sys.modules.pop(modname)
    return orig_modules


@contextlib.contextmanager
def frozen_modules(enabled=True):  # type: ignore[no-untyped-def]
    """Force frozen modules to be used (or not).

    This only applies to modules that haven't been imported yet.
    Also, some essential modules will always be imported frozen.
    """
    _imp._override_frozen_modules_for_tests(1 if enabled else -1)  # type: ignore[attr-defined]
    try:
        yield
    finally:
        _imp._override_frozen_modules_for_tests(0)  # type: ignore[attr-defined]


def import_fresh_module(  # type: ignore[no-untyped-def]
    name, fresh=(), blocked=(), *, deprecated=False, usefrozen=False
):
    # Keep track of modules saved for later restoration as well
    # as those which just need a blocking entry removed
    fresh = list(fresh)
    blocked = list(blocked)
    names = {name, *fresh, *blocked}
    orig_modules = _save_and_remove_modules(names)
    for modname in blocked:
        sys.modules[modname] = None  # type: ignore[assignment]

    try:
        with frozen_modules(usefrozen):
            # Return None when one of the "fresh" modules can not be imported.
            try:
                for modname in fresh:
                    __import__(modname)
            except ImportError:
                return None
            return importlib.import_module(name)
    finally:
        _save_and_remove_modules(names)
        sys.modules.update(orig_modules)


py_heapq = import_fresh_module("heapq", blocked=["_heapq"])


__all__ = [
    "_heapify_max",
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
def _heapify_max(heap: list[_T], /) -> None:
    return py_heapq._heapify_max(heap)


if hasattr(py_heapq, "_heappop_max"):
    __all__ += ["_heappop_max"]

    @substitute_in_graph(heapq._heappop_max)  # type: ignore[attr-defined]
    def _heappop_max(heap: list[_T]) -> _T:
        return py_heapq._heappop_max(heap)


if hasattr(py_heapq, "_heapreplace_max"):
    __all__ += ["_heapreplace_max"]

    @substitute_in_graph(heapq._heapreplace_max)  # type: ignore[attr-defined]
    def _heapreplace_max(heap: list[_T], item: _T) -> _T:
        return py_heapq._heapreplace_max(heap, item)


@substitute_in_graph(heapq.heapify)
def heapify(heap: list[_T], /) -> None:
    return py_heapq.heapify(heap)


@substitute_in_graph(heapq.heappop)
def heappop(heap: list[_T], /) -> _T:
    return py_heapq.heappop(heap)


@substitute_in_graph(heapq.heappush)
def heappush(heap: list[_T], item: _T) -> None:
    return py_heapq.heappush(heap, item)


@substitute_in_graph(heapq.heappushpop)
def heappushpop(heap: list[_T], item: _T) -> _T:
    return py_heapq.heappushpop(heap, item)


@substitute_in_graph(heapq.heapreplace)
def heapreplace(heap: list[_T], item: _T) -> _T:
    return py_heapq.heapreplace(heap, item)


@substitute_in_graph(heapq.merge)  # type: ignore[arg-type]
def merge(*iterables, key=None, reverse=False):  # type: ignore[no-untyped-def]
    return py_heapq.merge(*iterables, key=key, reverse=reverse)


@substitute_in_graph(heapq.nlargest)  # type: ignore[arg-type]
def nlargest(n, iterable, key=None):  # type: ignore[no-untyped-def]
    return py_heapq.nlargest(n, iterable, key=key)


@substitute_in_graph(heapq.nsmallest)  # type: ignore[arg-type]
def nsmallest(n, iterable, key=None):  # type: ignore[no-untyped-def]
    return py_heapq.nsmallest(n, iterable, key=key)
