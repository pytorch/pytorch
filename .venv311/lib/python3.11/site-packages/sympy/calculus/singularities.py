"""
Singularities
=============

This module implements algorithms for finding singularities for a function
and identifying types of functions.

The differential calculus methods in this module include methods to identify
the following function types in the given ``Interval``:
- Increasing
- Strictly Increasing
- Decreasing
- Strictly Decreasing
- Monotonic

"""

from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import sec, csc, cot, tan, cos
from sympy.functions.elementary.hyperbolic import (
    sech, csch, coth, tanh, cosh, asech, acsch, atanh, acoth)
from sympy.utilities.misc import filldedent


def singularities(expression, symbol, domain=None):
    """
    Find singularities of a given function.

    Parameters
    ==========

    expression : Expr
        The target function in which singularities need to be found.
    symbol : Symbol
        The symbol over the values of which the singularity in
        expression in being searched for.

    Returns
    =======

    Set
        A set of values for ``symbol`` for which ``expression`` has a
        singularity. An ``EmptySet`` is returned if ``expression`` has no
        singularities for any given value of ``Symbol``.

    Raises
    ======

    NotImplementedError
        Methods for determining the singularities of this function have
        not been developed.

    Notes
    =====

    This function does not find non-isolated singularities
    nor does it find branch points of the expression.

    Currently supported functions are:
        - univariate continuous (real or complex) functions

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathematical_singularity

    Examples
    ========

    >>> from sympy import singularities, Symbol, log
    >>> x = Symbol('x', real=True)
    >>> y = Symbol('y', real=False)
    >>> singularities(x**2 + x + 1, x)
    EmptySet
    >>> singularities(1/(x + 1), x)
    {-1}
    >>> singularities(1/(y**2 + 1), y)
    {-I, I}
    >>> singularities(1/(y**3 + 1), y)
    {-1, 1/2 - sqrt(3)*I/2, 1/2 + sqrt(3)*I/2}
    >>> singularities(log(x), x)
    {0}

    """
    from sympy.solvers.solveset import solveset

    if domain is None:
        domain = S.Reals if symbol.is_real else S.Complexes
    try:
        sings = S.EmptySet
        e = expression.rewrite([sec, csc, cot, tan], cos)
        e = e.rewrite([sech, csch, coth, tanh], cosh)
        for i in e.atoms(Pow):
            if i.exp.is_infinite:
                raise NotImplementedError
            if i.exp.is_negative:
                # XXX: exponent of varying sign not handled
                sings += solveset(i.base, symbol, domain)
        for i in expression.atoms(log, asech, acsch):
            sings += solveset(i.args[0], symbol, domain)
        for i in expression.atoms(atanh, acoth):
            sings += solveset(i.args[0] - 1, symbol, domain)
            sings += solveset(i.args[0] + 1, symbol, domain)
        return sings
    except NotImplementedError:
        raise NotImplementedError(filldedent('''
            Methods for determining the singularities
            of this function have not been developed.'''))


###########################################################################
#                      DIFFERENTIAL CALCULUS METHODS                      #
###########################################################################


def monotonicity_helper(expression, predicate, interval=S.Reals, symbol=None):
    """
    Helper function for functions checking function monotonicity.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked
    predicate : function
        The property being tested for. The function takes in an integer
        and returns a boolean. The integer input is the derivative and
        the boolean result should be true if the property is being held,
        and false otherwise.
    interval : Set, optional
        The range of values in which we are testing, defaults to all reals.
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    It returns a boolean indicating whether the interval in which
    the function's derivative satisfies given predicate is a superset
    of the given interval.

    Returns
    =======

    Boolean
        True if ``predicate`` is true for all the derivatives when ``symbol``
        is varied in ``range``, False otherwise.

    """
    from sympy.solvers.solveset import solveset

    expression = sympify(expression)
    free = expression.free_symbols

    if symbol is None:
        if len(free) > 1:
            raise NotImplementedError(
                'The function has not yet been implemented'
                ' for all multivariate expressions.'
            )

    variable = symbol or (free.pop() if free else Symbol('x'))
    derivative = expression.diff(variable)
    predicate_interval = solveset(predicate(derivative), variable, S.Reals)
    return interval.is_subset(predicate_interval)


def is_increasing(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is increasing in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is increasing (either strictly increasing or
        constant) in the given ``interval``, False otherwise.

    Examples
    ========

    >>> from sympy import is_increasing
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_increasing(x**3 - 3*x**2 + 4*x, S.Reals)
    True
    >>> is_increasing(-x**2, Interval(-oo, 0))
    True
    >>> is_increasing(-x**2, Interval(0, oo))
    False
    >>> is_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval(-2, 3))
    False
    >>> is_increasing(x**2 + y, Interval(1, 2), x)
    True

    """
    return monotonicity_helper(expression, lambda x: x >= 0, interval, symbol)


def is_strictly_increasing(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is strictly increasing in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is strictly increasing in the given ``interval``,
        False otherwise.

    Examples
    ========

    >>> from sympy import is_strictly_increasing
    >>> from sympy.abc import x, y
    >>> from sympy import Interval, oo
    >>> is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.Ropen(-oo, -2))
    True
    >>> is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.Lopen(3, oo))
    True
    >>> is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.open(-2, 3))
    False
    >>> is_strictly_increasing(-x**2, Interval(0, oo))
    False
    >>> is_strictly_increasing(-x**2 + y, Interval(-oo, 0), x)
    False

    """
    return monotonicity_helper(expression, lambda x: x > 0, interval, symbol)


def is_decreasing(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is decreasing in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is decreasing (either strictly decreasing or
        constant) in the given ``interval``, False otherwise.

    Examples
    ========

    >>> from sympy import is_decreasing
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_decreasing(1/(x**2 - 3*x), Interval.open(S(3)/2, 3))
    True
    >>> is_decreasing(1/(x**2 - 3*x), Interval.open(1.5, 3))
    True
    >>> is_decreasing(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    True
    >>> is_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, S(3)/2))
    False
    >>> is_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, 1.5))
    False
    >>> is_decreasing(-x**2, Interval(-oo, 0))
    False
    >>> is_decreasing(-x**2 + y, Interval(-oo, 0), x)
    False

    """
    return monotonicity_helper(expression, lambda x: x <= 0, interval, symbol)


def is_strictly_decreasing(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is strictly decreasing in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is strictly decreasing in the given ``interval``,
        False otherwise.

    Examples
    ========

    >>> from sympy import is_strictly_decreasing
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_strictly_decreasing(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    True
    >>> is_strictly_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, S(3)/2))
    False
    >>> is_strictly_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, 1.5))
    False
    >>> is_strictly_decreasing(-x**2, Interval(-oo, 0))
    False
    >>> is_strictly_decreasing(-x**2 + y, Interval(-oo, 0), x)
    False

    """
    return monotonicity_helper(expression, lambda x: x < 0, interval, symbol)


def is_monotonic(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is monotonic in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is monotonic in the given ``interval``,
        False otherwise.

    Raises
    ======

    NotImplementedError
        Monotonicity check has not been implemented for the queried function.

    Examples
    ========

    >>> from sympy import is_monotonic
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_monotonic(1/(x**2 - 3*x), Interval.open(S(3)/2, 3))
    True
    >>> is_monotonic(1/(x**2 - 3*x), Interval.open(1.5, 3))
    True
    >>> is_monotonic(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    True
    >>> is_monotonic(x**3 - 3*x**2 + 4*x, S.Reals)
    True
    >>> is_monotonic(-x**2, S.Reals)
    False
    >>> is_monotonic(x**2 + y + 1, Interval(1, 2), x)
    True

    """
    from sympy.solvers.solveset import solveset

    expression = sympify(expression)

    free = expression.free_symbols
    if symbol is None and len(free) > 1:
        raise NotImplementedError(
            'is_monotonic has not yet been implemented'
            ' for all multivariate expressions.'
        )

    variable = symbol or (free.pop() if free else Symbol('x'))
    turning_points = solveset(expression.diff(variable), variable, interval)
    return interval.intersection(turning_points) is S.EmptySet
