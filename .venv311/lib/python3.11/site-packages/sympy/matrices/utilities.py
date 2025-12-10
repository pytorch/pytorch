from contextlib import contextmanager
from threading import local

from sympy.core.function import expand_mul


class DotProdSimpState(local):
    def __init__(self):
        self.state = None

_dotprodsimp_state = DotProdSimpState()

@contextmanager
def dotprodsimp(x):
    old = _dotprodsimp_state.state

    try:
        _dotprodsimp_state.state = x
        yield
    finally:
        _dotprodsimp_state.state = old


def _dotprodsimp(expr, withsimp=False):
    """Wrapper for simplify.dotprodsimp to avoid circular imports."""
    from sympy.simplify.simplify import dotprodsimp as dps
    return dps(expr, withsimp=withsimp)


def _get_intermediate_simp(deffunc=lambda x: x, offfunc=lambda x: x,
        onfunc=_dotprodsimp, dotprodsimp=None):
    """Support function for controlling intermediate simplification. Returns a
    simplification function according to the global setting of dotprodsimp
    operation.

    ``deffunc``     - Function to be used by default.
    ``offfunc``     - Function to be used if dotprodsimp has been turned off.
    ``onfunc``      - Function to be used if dotprodsimp has been turned on.
    ``dotprodsimp`` - True, False or None. Will be overridden by global
                      _dotprodsimp_state.state if that is not None.
    """

    if dotprodsimp is False or _dotprodsimp_state.state is False:
        return offfunc
    if dotprodsimp is True or _dotprodsimp_state.state is True:
        return onfunc

    return deffunc # None, None


def _get_intermediate_simp_bool(default=False, dotprodsimp=None):
    """Same as ``_get_intermediate_simp`` but returns bools instead of functions
    by default."""

    return _get_intermediate_simp(default, False, True, dotprodsimp)


def _iszero(x):
    """Returns True if x is zero."""
    return getattr(x, 'is_zero', None)


def _is_zero_after_expand_mul(x):
    """Tests by expand_mul only, suitable for polynomials and rational
    functions."""
    return expand_mul(x) == 0


def _simplify(expr):
    """ Wrapper to avoid circular imports. """
    from sympy.simplify.simplify import simplify
    return simplify(expr)
