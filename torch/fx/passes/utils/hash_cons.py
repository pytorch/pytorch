import enum
import sys
from typing import List, Optional, Tuple

import torch
from torch import fx


class SymRel(enum.Enum):
    EQ = 0
    NE = 1
    LT = 3
    LE = 4
    GT = 5
    GE = 6


class SymExprRange:
    '''
    Assumes a value exists for the range this represents,
    and the calls will follow accordingly.
    e.g. no self.eq(5) & self.ne(5) calls
    '''
    def __init__(self):
        self._eq = None
        self._l = None
        self._l_strict = False
        self._g = None
        self._g_strict = False
        self._ne = set()
        self._is_size = False

    @property
    def static(self):
        return self._eq is not None

    @property
    def is_size(self):
        return self._is_size

    @property
    def val(self):
        return self._eq

    @property
    def min(self):
        return self._g, self._g_strict

    @property
    def max(self):
        return self._l, self._l_strict

    @property
    def not_equals(self):
        return list(self._ne)

    def eq(self, val: int) -> Optional[Tuple[SymRel, int]]:
        self._eq = val
        self._l, self._l_strict = None, None
        self._g, self._g_strict = None, None
        self._ne = set()
        return SymRel.EQ, val

    def ne(self, val: int) -> Optional[Tuple[SymRel, int]]:
        if self.static or val in self._ne:  # static or duplicate
            return
        if val == self._l and not self._l_strict:  # convert to lt call
            return self.lt(val, strict=True)
        if val == self._g and not self._g_strict:  # convert to gt call
            return self.gt(val, strict=True)
        self._ne.add(val)
        return SymRel.NE, val

    def lt(self, val: int, strict: bool) -> Optional[Tuple[SymRel, int]]:
        if self.static:
            return
        for ne in list(self._ne):
            if val < ne:
                self._ne.remove(ne)
        if strict:
            if (
                self._l is None
                or val < self._l
                or (val == self._l and not self._l_strict)
            ):
                self._l = val
                self._l_strict = True
                return SymRel.LT, val
            return None
        else:
            if val == self._g and not self._g_strict:  # equality
                return self.eq(val)
            if val in self._ne:  # convert to strict
                self._ne.remove(val)
                return self.lt(val, strict=True)
            if (
                self._l is None
                or val < self._l
            ):
                self._l = val
                self._l_strict = False
                return SymRel.LE, val
            return None

    def gt(self, val: int, strict: bool) -> Optional[Tuple[SymRel, int]]:
        if self.static:
            return
        for ne in list(self._ne):
            if val > ne:
                self._ne.remove(ne)
        if strict:
            if (
                self._g is None
                or val > self._g
                or (val == self._g and not self._g_strict)
            ):
                self._g = val
                self._g_strict = True
                return SymRel.GT, val
        else:
            if val == self._l and not self._l_strict:  # equality
                return self.eq(val)
            if val in self._ne:  # convert to strict
                self._ne.remove(val)
                return self.gt(val, strict=True)
            if (
                self._g is None
                or val > self._g
            ):
                self._g = val
                self._g_strict = False
                return SymRel.GE, val
        return None

    def set_is_size(self):
        self.gt(0, False)
        self.lt(sys.maxsize - 1, False)
        self._is_size = True
