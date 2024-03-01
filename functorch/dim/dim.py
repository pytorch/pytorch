# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import dis
import inspect

from dataclasses import dataclass
from typing import Union

from . import DimList

_vmap_levels = []


@dataclass
class LevelInfo:
    level: int
    alive: bool = True


class Dim:
    def __init__(self, name: str, size: Union[None, int] = None):
        self.name = name
        self._size = None
        self._vmap_level = None
        if size is not None:
            self.size = size

    def __del__(self):
        if self._vmap_level is not None:
            _vmap_active_levels[self._vmap_stack].alive = False  # noqa: F821
            while (
                not _vmap_levels[-1].alive
                and current_level() == _vmap_levels[-1].level  # noqa: F821
            ):
                _vmap_decrement_nesting()  # noqa: F821
                _vmap_levels.pop()

    @property
    def size(self):
        assert self.is_bound
        return self._size

    @size.setter
    def size(self, size: int):
        from . import DimensionBindError

        if self._size is None:
            self._size = size
            self._vmap_level = _vmap_increment_nesting(size, "same")  # noqa: F821
            self._vmap_stack = len(_vmap_levels)
            _vmap_levels.append(LevelInfo(self._vmap_level))

        elif self._size != size:
            raise DimensionBindError(
                f"Dim '{self}' previously bound to a dimension of size {self._size} cannot bind to a dimension of size {size}"
            )

    @property
    def is_bound(self):
        return self._size is not None

    def __repr__(self):
        return self.name


def extract_name(inst):
    assert inst.opname == "STORE_FAST" or inst.opname == "STORE_NAME"
    return inst.argval


_cache = {}


def dims(lists=0):
    frame = inspect.currentframe()
    assert frame is not None
    calling_frame = frame.f_back
    assert calling_frame is not None
    code, lasti = calling_frame.f_code, calling_frame.f_lasti
    key = (code, lasti)
    if key not in _cache:
        first = lasti // 2 + 1
        instructions = list(dis.get_instructions(calling_frame.f_code))
        unpack = instructions[first]

        if unpack.opname == "STORE_FAST" or unpack.opname == "STORE_NAME":
            # just a single dim, not a list
            name = unpack.argval
            ctor = Dim if lists == 0 else DimList
            _cache[key] = lambda: ctor(name=name)
        else:
            assert unpack.opname == "UNPACK_SEQUENCE"
            ndims = unpack.argval
            names = tuple(
                extract_name(instructions[first + 1 + i]) for i in range(ndims)
            )
            first_list = len(names) - lists
            _cache[key] = lambda: tuple(
                Dim(n) if i < first_list else DimList(name=n)
                for i, n in enumerate(names)
            )
    return _cache[key]()


def _dim_set(positional, arg):
    def convert(a):
        if isinstance(a, Dim):
            return a
        else:
            assert isinstance(a, int)
            return positional[a]

    if arg is None:
        return positional
    elif not isinstance(arg, (Dim, int)):
        return tuple(convert(a) for a in arg)
    else:
        return (convert(arg),)
