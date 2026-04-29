from __future__ import annotations

import functools
import operator
from typing import Any, TYPE_CHECKING

import sympy

from torch.fx.experimental._size_hinting import _sympy_subs
from torch.utils._sympy.symbol import make_symbol, SymT


if TYPE_CHECKING:
    from collections.abc import Iterable


def sympy_product(it: Iterable[sympy.Expr]) -> sympy.Expr:
    return functools.reduce(operator.mul, it, sympy.S.One)


def sympy_index_symbol_with_prefix(prefix: SymT, idx: int) -> sympy.Symbol:
    """
    Used to generate an integer-nonnegative symbol.
    """
    # This should never be used for creating shape/stride symbols, as those
    # should all be allocated before Inductor.
    assert prefix != SymT.SIZE
    # NOTE: shape symbols are positive (> 0), but index variables are only
    # non-negative (>= 0).
    return make_symbol(prefix, idx, integer=True, nonnegative=True)


def sympy_index_symbol(name: str) -> sympy.Symbol:
    """
    Used to generate an integer-nonnegative symbol.
    """
    # This should never be used for creating shape/stride symbols, as those
    # should all be allocated before Inductor.
    assert name[0] != "s"
    # NOTE: shape symbols are positive (> 0), but index variables are only
    # non-negative (>= 0).
    return sympy.Symbol(name, integer=True, nonnegative=True)


def sympy_subs(expr: sympy.Expr, replacements: dict[sympy.Expr, Any]) -> sympy.Expr:
    """
    When the passed replacement symbol v is a string, it is converted to a symbol with name v that
    have the same replaced expression integer and nonnegative properties.
    """
    return _sympy_subs(expr, replacements)
