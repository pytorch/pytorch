"""Limits of sequences"""

from sympy.calculus.accumulationbounds import AccumulationBounds
from sympy.core.add import Add
from sympy.core.function import PoleError
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.combinatorial.factorials import factorial, subfactorial
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.series.limits import Limit


def difference_delta(expr, n=None, step=1):
    """Difference Operator.

    Explanation
    ===========

    Discrete analog of differential operator. Given a sequence x[n],
    returns the sequence x[n + step] - x[n].

    Examples
    ========

    >>> from sympy import difference_delta as dd
    >>> from sympy.abc import n
    >>> dd(n*(n + 1), n)
    2*n + 2
    >>> dd(n*(n + 1), n, 2)
    4*n + 6

    References
    ==========

    .. [1] https://reference.wolfram.com/language/ref/DifferenceDelta.html
    """
    expr = sympify(expr)

    if n is None:
        f = expr.free_symbols
        if len(f) == 1:
            n = f.pop()
        elif len(f) == 0:
            return S.Zero
        else:
            raise ValueError("Since there is more than one variable in the"
                             " expression, a variable must be supplied to"
                             " take the difference of %s" % expr)
    step = sympify(step)
    if step.is_number is False or step.is_finite is False:
        raise ValueError("Step should be a finite number.")

    if hasattr(expr, '_eval_difference_delta'):
        result = expr._eval_difference_delta(n, step)
        if result:
            return result

    return expr.subs(n, n + step) - expr


def dominant(expr, n):
    """Finds the dominant term in a sum, that is a term that dominates
    every other term.

    Explanation
    ===========

    If limit(a/b, n, oo) is oo then a dominates b.
    If limit(a/b, n, oo) is 0 then b dominates a.
    Otherwise, a and b are comparable.

    If there is no unique dominant term, then returns ``None``.

    Examples
    ========

    >>> from sympy import Sum
    >>> from sympy.series.limitseq import dominant
    >>> from sympy.abc import n, k
    >>> dominant(5*n**3 + 4*n**2 + n + 1, n)
    5*n**3
    >>> dominant(2**n + Sum(k, (k, 0, n)), n)
    2**n

    See Also
    ========

    sympy.series.limitseq.dominant
    """
    terms = Add.make_args(expr.expand(func=True))
    term0 = terms[-1]
    comp = [term0]  # comparable terms
    for t in terms[:-1]:
        r = term0/t
        e = r.gammasimp()
        if e == r:
            e = r.factor()
        l = limit_seq(e, n)
        if l is None:
            return None
        elif l.is_zero:
            term0 = t
            comp = [term0]
        elif l not in [S.Infinity, S.NegativeInfinity]:
            comp.append(t)
    if len(comp) > 1:
        return None
    return term0


def _limit_inf(expr, n):
    try:
        return Limit(expr, n, S.Infinity).doit(deep=False)
    except (NotImplementedError, PoleError):
        return None


def _limit_seq(expr, n, trials):
    from sympy.concrete.summations import Sum

    for i in range(trials):
        if not expr.has(Sum):
            result = _limit_inf(expr, n)
            if result is not None:
                return result

        num, den = expr.as_numer_denom()
        if not den.has(n) or not num.has(n):
            result = _limit_inf(expr.doit(), n)
            if result is not None:
                return result
            return None

        num, den = (difference_delta(t.expand(), n) for t in [num, den])
        expr = (num / den).gammasimp()

        if not expr.has(Sum):
            result = _limit_inf(expr, n)
            if result is not None:
                return result

        num, den = expr.as_numer_denom()

        num = dominant(num, n)
        if num is None:
            return None

        den = dominant(den, n)
        if den is None:
            return None

        expr = (num / den).gammasimp()


def limit_seq(expr, n=None, trials=5):
    """Finds the limit of a sequence as index ``n`` tends to infinity.

    Parameters
    ==========

    expr : Expr
        SymPy expression for the ``n-th`` term of the sequence
    n : Symbol, optional
        The index of the sequence, an integer that tends to positive
        infinity. If None, inferred from the expression unless it has
        multiple symbols.
    trials: int, optional
        The algorithm is highly recursive. ``trials`` is a safeguard from
        infinite recursion in case the limit is not easily computed by the
        algorithm. Try increasing ``trials`` if the algorithm returns ``None``.

    Admissible Terms
    ================

    The algorithm is designed for sequences built from rational functions,
    indefinite sums, and indefinite products over an indeterminate n. Terms of
    alternating sign are also allowed, but more complex oscillatory behavior is
    not supported.

    Examples
    ========

    >>> from sympy import limit_seq, Sum, binomial
    >>> from sympy.abc import n, k, m
    >>> limit_seq((5*n**3 + 3*n**2 + 4) / (3*n**3 + 4*n - 5), n)
    5/3
    >>> limit_seq(binomial(2*n, n) / Sum(binomial(2*k, k), (k, 1, n)), n)
    3/4
    >>> limit_seq(Sum(k**2 * Sum(2**m/m, (m, 1, k)), (k, 1, n)) / (2**n*n), n)
    4

    See Also
    ========

    sympy.series.limitseq.dominant

    References
    ==========

    .. [1] Computing Limits of Sequences - Manuel Kauers
    """

    from sympy.concrete.summations import Sum
    if n is None:
        free = expr.free_symbols
        if len(free) == 1:
            n = free.pop()
        elif not free:
            return expr
        else:
            raise ValueError("Expression has more than one variable. "
                             "Please specify a variable.")
    elif n not in expr.free_symbols:
        return expr

    expr = expr.rewrite(fibonacci, S.GoldenRatio)
    expr = expr.rewrite(factorial, subfactorial, gamma)
    n_ = Dummy("n", integer=True, positive=True)
    n1 = Dummy("n", odd=True, positive=True)
    n2 = Dummy("n", even=True, positive=True)

    # If there is a negative term raised to a power involving n, or a
    # trigonometric function, then consider even and odd n separately.
    powers = (p.as_base_exp() for p in expr.atoms(Pow))
    if (any(b.is_negative and e.has(n) for b, e in powers) or
            expr.has(cos, sin)):
        L1 = _limit_seq(expr.xreplace({n: n1}), n1, trials)
        if L1 is not None:
            L2 = _limit_seq(expr.xreplace({n: n2}), n2, trials)
            if L1 != L2:
                if L1.is_comparable and L2.is_comparable:
                    return AccumulationBounds(Min(L1, L2), Max(L1, L2))
                else:
                    return None
    else:
        L1 = _limit_seq(expr.xreplace({n: n_}), n_, trials)
    if L1 is not None:
        return L1
    else:
        if expr.is_Add:
            limits = [limit_seq(term, n, trials) for term in expr.args]
            if any(result is None for result in limits):
                return None
            else:
                return Add(*limits)
        # Maybe the absolute value is easier to deal with (though not if
        # it has a Sum). If it tends to 0, the limit is 0.
        elif not expr.has(Sum):
            lim = _limit_seq(Abs(expr.xreplace({n: n_})), n_, trials)
            if lim is not None and lim.is_zero:
                return S.Zero
