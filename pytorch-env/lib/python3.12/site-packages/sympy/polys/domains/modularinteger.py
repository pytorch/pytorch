"""Implementation of :class:`ModularInteger` class. """

from __future__ import annotations
from typing import Any

import operator

from sympy.polys.polyutils import PicklableWithSlots
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains.domainelement import DomainElement

from sympy.utilities import public
from sympy.utilities.exceptions import sympy_deprecation_warning

@public
class ModularInteger(PicklableWithSlots, DomainElement):
    """A class representing a modular integer. """

    mod, dom, sym, _parent = None, None, None, None

    __slots__ = ('val',)

    def parent(self):
        return self._parent

    def __init__(self, val):
        if isinstance(val, self.__class__):
            self.val = val.val % self.mod
        else:
            self.val = self.dom.convert(val) % self.mod

    def modulus(self):
        return self.mod

    def __hash__(self):
        return hash((self.val, self.mod))

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.val)

    def __str__(self):
        return "%s mod %s" % (self.val, self.mod)

    def __int__(self):
        return int(self.val)

    def to_int(self):

        sympy_deprecation_warning(
            """ModularInteger.to_int() is deprecated.

            Use int(a) or K = GF(p) and K.to_int(a) instead of a.to_int().
            """,
            deprecated_since_version="1.13",
            active_deprecations_target="modularinteger-to-int",
        )

        if self.sym:
            if self.val <= self.mod // 2:
                return self.val
            else:
                return self.val - self.mod
        else:
            return self.val

    def __pos__(self):
        return self

    def __neg__(self):
        return self.__class__(-self.val)

    @classmethod
    def _get_val(cls, other):
        if isinstance(other, cls):
            return other.val
        else:
            try:
                return cls.dom.convert(other)
            except CoercionFailed:
                return None

    def __add__(self, other):
        val = self._get_val(other)

        if val is not None:
            return self.__class__(self.val + val)
        else:
            return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        val = self._get_val(other)

        if val is not None:
            return self.__class__(self.val - val)
        else:
            return NotImplemented

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        val = self._get_val(other)

        if val is not None:
            return self.__class__(self.val * val)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        val = self._get_val(other)

        if val is not None:
            return self.__class__(self.val * self._invert(val))
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        return self.invert().__mul__(other)

    def __mod__(self, other):
        val = self._get_val(other)

        if val is not None:
            return self.__class__(self.val % val)
        else:
            return NotImplemented

    def __rmod__(self, other):
        val = self._get_val(other)

        if val is not None:
            return self.__class__(val % self.val)
        else:
            return NotImplemented

    def __pow__(self, exp):
        if not exp:
            return self.__class__(self.dom.one)

        if exp < 0:
            val, exp = self.invert().val, -exp
        else:
            val = self.val

        return self.__class__(pow(val, int(exp), self.mod))

    def _compare(self, other, op):
        val = self._get_val(other)

        if val is None:
            return NotImplemented

        return op(self.val, val % self.mod)

    def _compare_deprecated(self, other, op):
        val = self._get_val(other)

        if val is None:
            return NotImplemented

        sympy_deprecation_warning(
            """Ordered comparisons with modular integers are deprecated.

            Use e.g. int(a) < int(b) instead of a < b.
            """,
            deprecated_since_version="1.13",
            active_deprecations_target="modularinteger-compare",
            stacklevel=4,
        )

        return op(self.val, val % self.mod)

    def __eq__(self, other):
        return self._compare(other, operator.eq)

    def __ne__(self, other):
        return self._compare(other, operator.ne)

    def __lt__(self, other):
        return self._compare_deprecated(other, operator.lt)

    def __le__(self, other):
        return self._compare_deprecated(other, operator.le)

    def __gt__(self, other):
        return self._compare_deprecated(other, operator.gt)

    def __ge__(self, other):
        return self._compare_deprecated(other, operator.ge)

    def __bool__(self):
        return bool(self.val)

    @classmethod
    def _invert(cls, value):
        return cls.dom.invert(value, cls.mod)

    def invert(self):
        return self.__class__(self._invert(self.val))

_modular_integer_cache: dict[tuple[Any, Any, Any], type[ModularInteger]] = {}

def ModularIntegerFactory(_mod, _dom, _sym, parent):
    """Create custom class for specific integer modulus."""
    try:
        _mod = _dom.convert(_mod)
    except CoercionFailed:
        ok = False
    else:
        ok = True

    if not ok or _mod < 1:
        raise ValueError("modulus must be a positive integer, got %s" % _mod)

    key = _mod, _dom, _sym

    try:
        cls = _modular_integer_cache[key]
    except KeyError:
        class cls(ModularInteger):
            mod, dom, sym = _mod, _dom, _sym
            _parent = parent

        if _sym:
            cls.__name__ = "SymmetricModularIntegerMod%s" % _mod
        else:
            cls.__name__ = "ModularIntegerMod%s" % _mod

        _modular_integer_cache[key] = cls

    return cls
