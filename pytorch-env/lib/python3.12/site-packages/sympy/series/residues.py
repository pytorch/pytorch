"""
This module implements the Residue function and related tools for working
with residues.
"""

from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.utilities.timeutils import timethis


@timethis('residue')
def residue(expr, x, x0):
    """
    Finds the residue of ``expr`` at the point x=x0.

    The residue is defined as the coefficient of ``1/(x-x0)`` in the power series
    expansion about ``x=x0``.

    Examples
    ========

    >>> from sympy import Symbol, residue, sin
    >>> x = Symbol("x")
    >>> residue(1/x, x, 0)
    1
    >>> residue(1/x**2, x, 0)
    0
    >>> residue(2/sin(x), x, 0)
    2

    This function is essential for the Residue Theorem [1].

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Residue_theorem
    """
    # The current implementation uses series expansion to
    # calculate it. A more general implementation is explained in
    # the section 5.6 of the Bronstein's book {M. Bronstein:
    # Symbolic Integration I, Springer Verlag (2005)}. For purely
    # rational functions, the algorithm is much easier. See
    # sections 2.4, 2.5, and 2.7 (this section actually gives an
    # algorithm for computing any Laurent series coefficient for
    # a rational function). The theory in section 2.4 will help to
    # understand why the resultant works in the general algorithm.
    # For the definition of a resultant, see section 1.4 (and any
    # previous sections for more review).

    from sympy.series.order import Order
    from sympy.simplify.radsimp import collect
    expr = sympify(expr)
    if x0 != 0:
        expr = expr.subs(x, x + x0)
    for n in (0, 1, 2, 4, 8, 16, 32):
        s = expr.nseries(x, n=n)
        if not s.has(Order) or s.getn() >= 0:
            break
    s = collect(s.removeO(), x)
    if s.is_Add:
        args = s.args
    else:
        args = [s]
    res = S.Zero
    for arg in args:
        c, m = arg.as_coeff_mul(x)
        m = Mul(*m)
        if not (m in (S.One, x) or (m.is_Pow and m.exp.is_Integer)):
            raise NotImplementedError('term of unexpected form: %s' % m)
        if m == 1/x:
            res += c
    return res
