from sympy.core import S, sympify
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions import Piecewise, piecewise_fold
from sympy.logic.boolalg import And
from sympy.sets.sets import Interval

from functools import lru_cache


def _ivl(cond, x):
    """return the interval corresponding to the condition

    Conditions in spline's Piecewise give the range over
    which an expression is valid like (lo <= x) & (x <= hi).
    This function returns (lo, hi).
    """
    if isinstance(cond, And) and len(cond.args) == 2:
        a, b = cond.args
        if a.lts == x:
            a, b = b, a
        return a.lts, b.gts
    raise TypeError('unexpected cond type: %s' % cond)


def _add_splines(c, b1, d, b2, x):
    """Construct c*b1 + d*b2."""

    if S.Zero in (b1, c):
        rv = piecewise_fold(d * b2)
    elif S.Zero in (b2, d):
        rv = piecewise_fold(c * b1)
    else:
        new_args = []
        # Just combining the Piecewise without any fancy optimization
        p1 = piecewise_fold(c * b1)
        p2 = piecewise_fold(d * b2)

        # Search all Piecewise arguments except (0, True)
        p2args = list(p2.args[:-1])

        # This merging algorithm assumes the conditions in
        # p1 and p2 are sorted
        for arg in p1.args[:-1]:
            expr = arg.expr
            cond = arg.cond

            lower = _ivl(cond, x)[0]

            # Check p2 for matching conditions that can be merged
            for i, arg2 in enumerate(p2args):
                expr2 = arg2.expr
                cond2 = arg2.cond

                lower_2, upper_2 = _ivl(cond2, x)
                if cond2 == cond:
                    # Conditions match, join expressions
                    expr += expr2
                    # Remove matching element
                    del p2args[i]
                    # No need to check the rest
                    break
                elif lower_2 < lower and upper_2 <= lower:
                    # Check if arg2 condition smaller than arg1,
                    # add to new_args by itself (no match expected
                    # in p1)
                    new_args.append(arg2)
                    del p2args[i]
                    break

            # Checked all, add expr and cond
            new_args.append((expr, cond))

        # Add remaining items from p2args
        new_args.extend(p2args)

        # Add final (0, True)
        new_args.append((0, True))

        rv = Piecewise(*new_args, evaluate=False)

    return rv.expand()


@lru_cache(maxsize=128)
def bspline_basis(d, knots, n, x):
    """
    The $n$-th B-spline at $x$ of degree $d$ with knots.

    Explanation
    ===========

    B-Splines are piecewise polynomials of degree $d$. They are defined on a
    set of knots, which is a sequence of integers or floats.

    Examples
    ========

    The 0th degree splines have a value of 1 on a single interval:

        >>> from sympy import bspline_basis
        >>> from sympy.abc import x
        >>> d = 0
        >>> knots = tuple(range(5))
        >>> bspline_basis(d, knots, 0, x)
        Piecewise((1, (x >= 0) & (x <= 1)), (0, True))

    For a given ``(d, knots)`` there are ``len(knots)-d-1`` B-splines
    defined, that are indexed by ``n`` (starting at 0).

    Here is an example of a cubic B-spline:

        >>> bspline_basis(3, tuple(range(5)), 0, x)
        Piecewise((x**3/6, (x >= 0) & (x <= 1)),
                  (-x**3/2 + 2*x**2 - 2*x + 2/3,
                  (x >= 1) & (x <= 2)),
                  (x**3/2 - 4*x**2 + 10*x - 22/3,
                  (x >= 2) & (x <= 3)),
                  (-x**3/6 + 2*x**2 - 8*x + 32/3,
                  (x >= 3) & (x <= 4)),
                  (0, True))

    By repeating knot points, you can introduce discontinuities in the
    B-splines and their derivatives:

        >>> d = 1
        >>> knots = (0, 0, 2, 3, 4)
        >>> bspline_basis(d, knots, 0, x)
        Piecewise((1 - x/2, (x >= 0) & (x <= 2)), (0, True))

    It is quite time consuming to construct and evaluate B-splines. If
    you need to evaluate a B-spline many times, it is best to lambdify them
    first:

        >>> from sympy import lambdify
        >>> d = 3
        >>> knots = tuple(range(10))
        >>> b0 = bspline_basis(d, knots, 0, x)
        >>> f = lambdify(x, b0)
        >>> y = f(0.5)

    Parameters
    ==========

    d : integer
        degree of bspline

    knots : list of integer values
        list of knots points of bspline

    n : integer
        $n$-th B-spline

    x : symbol

    See Also
    ========

    bspline_basis_set

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/B-spline

    """
    # make sure x has no assumptions so conditions don't evaluate
    xvar = x
    x = Dummy()

    knots = tuple(sympify(k) for k in knots)
    d = int(d)
    n = int(n)
    n_knots = len(knots)
    n_intervals = n_knots - 1
    if n + d + 1 > n_intervals:
        raise ValueError("n + d + 1 must not exceed len(knots) - 1")
    if d == 0:
        result = Piecewise(
            (S.One, Interval(knots[n], knots[n + 1]).contains(x)), (0, True)
        )
    elif d > 0:
        denom = knots[n + d + 1] - knots[n + 1]
        if denom != S.Zero:
            B = (knots[n + d + 1] - x) / denom
            b2 = bspline_basis(d - 1, knots, n + 1, x)
        else:
            b2 = B = S.Zero

        denom = knots[n + d] - knots[n]
        if denom != S.Zero:
            A = (x - knots[n]) / denom
            b1 = bspline_basis(d - 1, knots, n, x)
        else:
            b1 = A = S.Zero

        result = _add_splines(A, b1, B, b2, x)
    else:
        raise ValueError("degree must be non-negative: %r" % n)

    # return result with user-given x
    return result.xreplace({x: xvar})


def bspline_basis_set(d, knots, x):
    """
    Return the ``len(knots)-d-1`` B-splines at *x* of degree *d*
    with *knots*.

    Explanation
    ===========

    This function returns a list of piecewise polynomials that are the
    ``len(knots)-d-1`` B-splines of degree *d* for the given knots.
    This function calls ``bspline_basis(d, knots, n, x)`` for different
    values of *n*.

    Examples
    ========

    >>> from sympy import bspline_basis_set
    >>> from sympy.abc import x
    >>> d = 2
    >>> knots = range(5)
    >>> splines = bspline_basis_set(d, knots, x)
    >>> splines
    [Piecewise((x**2/2, (x >= 0) & (x <= 1)),
               (-x**2 + 3*x - 3/2, (x >= 1) & (x <= 2)),
               (x**2/2 - 3*x + 9/2, (x >= 2) & (x <= 3)),
               (0, True)),
    Piecewise((x**2/2 - x + 1/2, (x >= 1) & (x <= 2)),
              (-x**2 + 5*x - 11/2, (x >= 2) & (x <= 3)),
              (x**2/2 - 4*x + 8, (x >= 3) & (x <= 4)),
              (0, True))]

    Parameters
    ==========

    d : integer
        degree of bspline

    knots : list of integers
        list of knots points of bspline

    x : symbol

    See Also
    ========

    bspline_basis

    """
    n_splines = len(knots) - d - 1
    return [bspline_basis(d, tuple(knots), i, x) for i in range(n_splines)]


def interpolating_spline(d, x, X, Y):
    """
    Return spline of degree *d*, passing through the given *X*
    and *Y* values.

    Explanation
    ===========

    This function returns a piecewise function such that each part is
    a polynomial of degree not greater than *d*. The value of *d*
    must be 1 or greater and the values of *X* must be strictly
    increasing.

    Examples
    ========

    >>> from sympy import interpolating_spline
    >>> from sympy.abc import x
    >>> interpolating_spline(1, x, [1, 2, 4, 7], [3, 6, 5, 7])
    Piecewise((3*x, (x >= 1) & (x <= 2)),
            (7 - x/2, (x >= 2) & (x <= 4)),
            (2*x/3 + 7/3, (x >= 4) & (x <= 7)))
    >>> interpolating_spline(3, x, [-2, 0, 1, 3, 4], [4, 2, 1, 1, 3])
    Piecewise((7*x**3/117 + 7*x**2/117 - 131*x/117 + 2, (x >= -2) & (x <= 1)),
            (10*x**3/117 - 2*x**2/117 - 122*x/117 + 77/39, (x >= 1) & (x <= 4)))

    Parameters
    ==========

    d : integer
        Degree of Bspline strictly greater than equal to one

    x : symbol

    X : list of strictly increasing real values
        list of X coordinates through which the spline passes

    Y : list of real values
        list of corresponding Y coordinates through which the spline passes

    See Also
    ========

    bspline_basis_set, interpolating_poly

    """
    from sympy.solvers.solveset import linsolve
    from sympy.matrices.dense import Matrix

    # Input sanitization
    d = sympify(d)
    if not (d.is_Integer and d.is_positive):
        raise ValueError("Spline degree must be a positive integer, not %s." % d)
    if len(X) != len(Y):
        raise ValueError("Number of X and Y coordinates must be the same.")
    if len(X) < d + 1:
        raise ValueError("Degree must be less than the number of control points.")
    if not all(a < b for a, b in zip(X, X[1:])):
        raise ValueError("The x-coordinates must be strictly increasing.")
    X = [sympify(i) for i in X]

    # Evaluating knots value
    if d.is_odd:
        j = (d + 1) // 2
        interior_knots = X[j:-j]
    else:
        j = d // 2
        interior_knots = [
            (a + b)/2 for a, b in zip(X[j : -j - 1], X[j + 1 : -j])
        ]

    knots = [X[0]] * (d + 1) + list(interior_knots) + [X[-1]] * (d + 1)

    basis = bspline_basis_set(d, knots, x)

    A = [[b.subs(x, v) for b in basis] for v in X]

    coeff = linsolve((Matrix(A), Matrix(Y)), symbols("c0:{}".format(len(X)), cls=Dummy))
    coeff = list(coeff)[0]
    intervals = {c for b in basis for (e, c) in b.args if c != True}

    # Sorting the intervals
    #  ival contains the end-points of each interval
    intervals = sorted(intervals, key=lambda c: _ivl(c, x))

    basis_dicts = [{c: e for (e, c) in b.args} for b in basis]
    spline = []
    for i in intervals:
        piece = sum(
            [c * d.get(i, S.Zero) for (c, d) in zip(coeff, basis_dicts)], S.Zero
        )
        spline.append((piece, i))
    return Piecewise(*spline)
