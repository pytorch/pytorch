from .accumulationbounds import AccumBounds, AccumulationBounds # noqa: F401
from .singularities import singularities
from sympy.core import Pow, S
from sympy.core.function import diff, expand_mul, Function
from sympy.core.kind import NumberKind
from sympy.core.mod import Mod
from sympy.core.numbers import equal_valued
from sympy.core.relational import Relational
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.integers import frac
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (
    TrigonometricFunction, sin, cos, tan, cot, csc, sec,
    asin, acos, acot, atan, asec, acsc)
from sympy.functions.elementary.hyperbolic import (sinh, cosh, tanh, coth,
    sech, csch, asinh, acosh, atanh, acoth, asech, acsch)
from sympy.polys.polytools import degree, lcm_list
from sympy.sets.sets import (Interval, Intersection, FiniteSet, Union,
                             Complement)
from sympy.sets.fancysets import ImageSet
from sympy.sets.conditionset import ConditionSet
from sympy.utilities import filldedent
from sympy.utilities.iterables import iterable
from sympy.matrices.dense import hessian


def continuous_domain(f, symbol, domain):
    """
    Returns the domain on which the function expression f is continuous.

    This function is limited by the ability to determine the various
    singularities and discontinuities of the given function.
    The result is either given as a union of intervals or constructed using
    other set operations.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    symbol : :py:class:`~.Symbol`
        The variable for which the intervals are to be determined.
    domain : :py:class:`~.Interval`
        The domain over which the continuity of the symbol has to be checked.

    Examples
    ========

    >>> from sympy import Interval, Symbol, S, tan, log, pi, sqrt
    >>> from sympy.calculus.util import continuous_domain
    >>> x = Symbol('x')
    >>> continuous_domain(1/x, x, S.Reals)
    Union(Interval.open(-oo, 0), Interval.open(0, oo))
    >>> continuous_domain(tan(x), x, Interval(0, pi))
    Union(Interval.Ropen(0, pi/2), Interval.Lopen(pi/2, pi))
    >>> continuous_domain(sqrt(x - 2), x, Interval(-5, 5))
    Interval(2, 5)
    >>> continuous_domain(log(2*x - 1), x, S.Reals)
    Interval.open(1/2, oo)

    Returns
    =======

    :py:class:`~.Interval`
        Union of all intervals where the function is continuous.

    Raises
    ======

    NotImplementedError
        If the method to determine continuity of such a function
        has not yet been developed.

    """
    from sympy.solvers.inequalities import solve_univariate_inequality

    if not domain.is_subset(S.Reals):
        raise NotImplementedError(filldedent('''
            Domain must be a subset of S.Reals.
            '''))
    implemented = [Pow, exp, log, Abs, frac,
                   sin, cos, tan, cot, sec, csc,
                   asin, acos, atan, acot, asec, acsc,
                   sinh, cosh, tanh, coth, sech, csch,
                   asinh, acosh, atanh, acoth, asech, acsch]
    used = [fct.func for fct in f.atoms(Function) if fct.has(symbol)]
    if any(func not in implemented for func in used):
        raise NotImplementedError(filldedent('''
            Unable to determine the domain of the given function.
            '''))

    x = Symbol('x')
    constraints = {
        log: (x > 0,),
        asin: (x >= -1, x <= 1),
        acos: (x >= -1, x <= 1),
        acosh: (x >= 1,),
        atanh: (x > -1, x < 1),
        asech: (x > 0, x <= 1)
    }
    constraints_union = {
        asec: (x <= -1, x >= 1),
        acsc: (x <= -1, x >= 1),
        acoth: (x < -1, x > 1)
    }

    cont_domain = domain
    for atom in f.atoms(Pow):
        den = atom.exp.as_numer_denom()[1]
        if atom.exp.is_rational and den.is_odd:
            pass    # 0**negative handled by singularities()
        else:
            constraint = solve_univariate_inequality(atom.base >= 0,
                                                        symbol).as_set()
            cont_domain = Intersection(constraint, cont_domain)

    for atom in f.atoms(Function):
        if atom.func in constraints:
            for c in constraints[atom.func]:
                constraint_relational = c.subs(x, atom.args[0])
                constraint_set = solve_univariate_inequality(
                    constraint_relational, symbol).as_set()
                cont_domain = Intersection(constraint_set, cont_domain)
        elif atom.func in constraints_union:
            constraint_set = S.EmptySet
            for c in constraints_union[atom.func]:
                constraint_relational = c.subs(x, atom.args[0])
                constraint_set += solve_univariate_inequality(
                    constraint_relational, symbol).as_set()
            cont_domain = Intersection(constraint_set, cont_domain)
        # XXX: the discontinuities below could be factored out in
        # a new "discontinuities()".
        elif atom.func == acot:
            from sympy.solvers.solveset import solveset_real
            # Sympy's acot() has a step discontinuity at 0. Since it's
            # neither an essential singularity nor a pole, singularities()
            # will not report it. But it's still relevant for determining
            # the continuity of the function f.
            cont_domain -= solveset_real(atom.args[0], symbol)
            # Note that the above may introduce spurious discontinuities, e.g.
            # for abs(acot(x)) at 0.
        elif atom.func == frac:
            from sympy.solvers.solveset import solveset_real
            r = function_range(atom.args[0], symbol, domain)
            r = Intersection(r, S.Integers)
            if r.is_finite_set:
                discont = S.EmptySet
                for n in r:
                    discont += solveset_real(atom.args[0]-n, symbol)
            else:
                discont = ConditionSet(
                    symbol, S.Integers.contains(atom.args[0]), cont_domain)
            cont_domain -= discont

    return cont_domain - singularities(f, symbol, domain)


def function_range(f, symbol, domain):
    """
    Finds the range of a function in a given domain.
    This method is limited by the ability to determine the singularities and
    determine limits.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    symbol : :py:class:`~.Symbol`
        The variable for which the range of function is to be determined.
    domain : :py:class:`~.Interval`
        The domain under which the range of the function has to be found.

    Examples
    ========

    >>> from sympy import Interval, Symbol, S, exp, log, pi, sqrt, sin, tan
    >>> from sympy.calculus.util import function_range
    >>> x = Symbol('x')
    >>> function_range(sin(x), x, Interval(0, 2*pi))
    Interval(-1, 1)
    >>> function_range(tan(x), x, Interval(-pi/2, pi/2))
    Interval(-oo, oo)
    >>> function_range(1/x, x, S.Reals)
    Union(Interval.open(-oo, 0), Interval.open(0, oo))
    >>> function_range(exp(x), x, S.Reals)
    Interval.open(0, oo)
    >>> function_range(log(x), x, S.Reals)
    Interval(-oo, oo)
    >>> function_range(sqrt(x), x, Interval(-5, 9))
    Interval(0, 3)

    Returns
    =======

    :py:class:`~.Interval`
        Union of all ranges for all intervals under domain where function is
        continuous.

    Raises
    ======

    NotImplementedError
        If any of the intervals, in the given domain, for which function
        is continuous are not finite or real,
        OR if the critical points of the function on the domain cannot be found.
    """

    if domain is S.EmptySet:
        return S.EmptySet

    period = periodicity(f, symbol)
    if period == S.Zero:
        # the expression is constant wrt symbol
        return FiniteSet(f.expand())

    from sympy.series.limits import limit
    from sympy.solvers.solveset import solveset

    if period is not None:
        if isinstance(domain, Interval):
            if (domain.inf - domain.sup).is_infinite:
                domain = Interval(0, period)
        elif isinstance(domain, Union):
            for sub_dom in domain.args:
                if isinstance(sub_dom, Interval) and \
                ((sub_dom.inf - sub_dom.sup).is_infinite):
                    domain = Interval(0, period)

    intervals = continuous_domain(f, symbol, domain)
    range_int = S.EmptySet
    if isinstance(intervals,(Interval, FiniteSet)):
        interval_iter = (intervals,)
    elif isinstance(intervals, Union):
        interval_iter = intervals.args
    else:
        raise NotImplementedError("Unable to find range for the given domain.")

    for interval in interval_iter:
        if isinstance(interval, FiniteSet):
            for singleton in interval:
                if singleton in domain:
                    range_int += FiniteSet(f.subs(symbol, singleton))
        elif isinstance(interval, Interval):
            vals = S.EmptySet
            critical_values = S.EmptySet
            bounds = ((interval.left_open, interval.inf, '+'),
                   (interval.right_open, interval.sup, '-'))

            for is_open, limit_point, direction in bounds:
                if is_open:
                    critical_values += FiniteSet(limit(f, symbol, limit_point, direction))
                    vals += critical_values
                else:
                    vals += FiniteSet(f.subs(symbol, limit_point))

            critical_points = solveset(f.diff(symbol), symbol, interval)

            if not iterable(critical_points):
                raise NotImplementedError(
                        'Unable to find critical points for {}'.format(f))
            if isinstance(critical_points, ImageSet):
                raise NotImplementedError(
                        'Infinite number of critical points for {}'.format(f))

            for critical_point in critical_points:
                vals += FiniteSet(f.subs(symbol, critical_point))

            left_open, right_open = False, False

            if critical_values is not S.EmptySet:
                if critical_values.inf == vals.inf:
                    left_open = True

                if critical_values.sup == vals.sup:
                    right_open = True

            range_int += Interval(vals.inf, vals.sup, left_open, right_open)
        else:
            raise NotImplementedError("Unable to find range for the given domain.")

    return range_int


def not_empty_in(finset_intersection, *syms):
    """
    Finds the domain of the functions in ``finset_intersection`` in which the
    ``finite_set`` is not-empty.

    Parameters
    ==========

    finset_intersection : Intersection of FiniteSet
        The unevaluated intersection of FiniteSet containing
        real-valued functions with Union of Sets
    syms : Tuple of symbols
        Symbol for which domain is to be found

    Raises
    ======

    NotImplementedError
        The algorithms to find the non-emptiness of the given FiniteSet are
        not yet implemented.
    ValueError
        The input is not valid.
    RuntimeError
        It is a bug, please report it to the github issue tracker
        (https://github.com/sympy/sympy/issues).

    Examples
    ========

    >>> from sympy import FiniteSet, Interval, not_empty_in, oo
    >>> from sympy.abc import x
    >>> not_empty_in(FiniteSet(x/2).intersect(Interval(0, 1)), x)
    Interval(0, 2)
    >>> not_empty_in(FiniteSet(x, x**2).intersect(Interval(1, 2)), x)
    Union(Interval(1, 2), Interval(-sqrt(2), -1))
    >>> not_empty_in(FiniteSet(x**2/(x + 2)).intersect(Interval(1, oo)), x)
    Union(Interval.Lopen(-2, -1), Interval(2, oo))
    """

    # TODO: handle piecewise defined functions
    # TODO: handle transcendental functions
    # TODO: handle multivariate functions
    if len(syms) == 0:
        raise ValueError("One or more symbols must be given in syms.")

    if finset_intersection is S.EmptySet:
        return S.EmptySet

    if isinstance(finset_intersection, Union):
        elm_in_sets = finset_intersection.args[0]
        return Union(not_empty_in(finset_intersection.args[1], *syms),
                     elm_in_sets)

    if isinstance(finset_intersection, FiniteSet):
        finite_set = finset_intersection
        _sets = S.Reals
    else:
        finite_set = finset_intersection.args[1]
        _sets = finset_intersection.args[0]

    if not isinstance(finite_set, FiniteSet):
        raise ValueError('A FiniteSet must be given, not %s: %s' %
                         (type(finite_set), finite_set))

    if len(syms) == 1:
        symb = syms[0]
    else:
        raise NotImplementedError('more than one variables %s not handled' %
                                  (syms,))

    def elm_domain(expr, intrvl):
        """ Finds the domain of an expression in any given interval """
        from sympy.solvers.solveset import solveset

        _start = intrvl.start
        _end = intrvl.end
        _singularities = solveset(expr.as_numer_denom()[1], symb,
                                  domain=S.Reals)

        if intrvl.right_open:
            if _end is S.Infinity:
                _domain1 = S.Reals
            else:
                _domain1 = solveset(expr < _end, symb, domain=S.Reals)
        else:
            _domain1 = solveset(expr <= _end, symb, domain=S.Reals)

        if intrvl.left_open:
            if _start is S.NegativeInfinity:
                _domain2 = S.Reals
            else:
                _domain2 = solveset(expr > _start, symb, domain=S.Reals)
        else:
            _domain2 = solveset(expr >= _start, symb, domain=S.Reals)

        # domain in the interval
        expr_with_sing = Intersection(_domain1, _domain2)
        expr_domain = Complement(expr_with_sing, _singularities)
        return expr_domain

    if isinstance(_sets, Interval):
        return Union(*[elm_domain(element, _sets) for element in finite_set])

    if isinstance(_sets, Union):
        _domain = S.EmptySet
        for intrvl in _sets.args:
            _domain_element = Union(*[elm_domain(element, intrvl)
                                      for element in finite_set])
            _domain = Union(_domain, _domain_element)
        return _domain


def periodicity(f, symbol, check=False):
    """
    Tests the given function for periodicity in the given symbol.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    symbol : :py:class:`~.Symbol`
        The variable for which the period is to be determined.
    check : bool, optional
        The flag to verify whether the value being returned is a period or not.

    Returns
    =======

    period
        The period of the function is returned.
        ``None`` is returned when the function is aperiodic or has a complex period.
        The value of $0$ is returned as the period of a constant function.

    Raises
    ======

    NotImplementedError
        The value of the period computed cannot be verified.


    Notes
    =====

    Currently, we do not support functions with a complex period.
    The period of functions having complex periodic values such
    as ``exp``, ``sinh`` is evaluated to ``None``.

    The value returned might not be the "fundamental" period of the given
    function i.e. it may not be the smallest periodic value of the function.

    The verification of the period through the ``check`` flag is not reliable
    due to internal simplification of the given expression. Hence, it is set
    to ``False`` by default.

    Examples
    ========
    >>> from sympy import periodicity, Symbol, sin, cos, tan, exp
    >>> x = Symbol('x')
    >>> f = sin(x) + sin(2*x) + sin(3*x)
    >>> periodicity(f, x)
    2*pi
    >>> periodicity(sin(x)*cos(x), x)
    pi
    >>> periodicity(exp(tan(2*x) - 1), x)
    pi/2
    >>> periodicity(sin(4*x)**cos(2*x), x)
    pi
    >>> periodicity(exp(x), x)
    """
    if symbol.kind is not NumberKind:
        raise NotImplementedError("Cannot use symbol of kind %s" % symbol.kind)
    temp = Dummy('x', real=True)
    f = f.subs(symbol, temp)
    symbol = temp

    def _check(orig_f, period):
        '''Return the checked period or raise an error.'''
        new_f = orig_f.subs(symbol, symbol + period)
        if new_f.equals(orig_f):
            return period
        else:
            raise NotImplementedError(filldedent('''
                The period of the given function cannot be verified.
                When `%s` was replaced with `%s + %s` in `%s`, the result
                was `%s` which was not recognized as being the same as
                the original function.
                So either the period was wrong or the two forms were
                not recognized as being equal.
                Set check=False to obtain the value.''' %
                (symbol, symbol, period, orig_f, new_f)))

    orig_f = f
    period = None

    if isinstance(f, Relational):
        f = f.lhs - f.rhs

    f = f.simplify()

    if symbol not in f.free_symbols:
        return S.Zero

    if isinstance(f, TrigonometricFunction):
        try:
            period = f.period(symbol)
        except NotImplementedError:
            pass

    if isinstance(f, Abs):
        arg = f.args[0]
        if isinstance(arg, (sec, csc, cos)):
            # all but tan and cot might have a
            # a period that is half as large
            # so recast as sin
            arg = sin(arg.args[0])
        period = periodicity(arg, symbol)
        if period is not None and isinstance(arg, sin):
            # the argument of Abs was a trigonometric other than
            # cot or tan; test to see if the half-period
            # is valid. Abs(arg) has behaviour equivalent to
            # orig_f, so use that for test:
            orig_f = Abs(arg)
            try:
                return _check(orig_f, period/2)
            except NotImplementedError as err:
                if check:
                    raise NotImplementedError(err)
            # else let new orig_f and period be
            # checked below

    if isinstance(f, exp) or (f.is_Pow and f.base == S.Exp1):
        f = Pow(S.Exp1, expand_mul(f.exp))
        if im(f) != 0:
            period_real = periodicity(re(f), symbol)
            period_imag = periodicity(im(f), symbol)
            if period_real is not None and period_imag is not None:
                period = lcim([period_real, period_imag])

    if f.is_Pow and f.base != S.Exp1:
        base, expo = f.args
        base_has_sym = base.has(symbol)
        expo_has_sym = expo.has(symbol)

        if base_has_sym and not expo_has_sym:
            period = periodicity(base, symbol)

        elif expo_has_sym and not base_has_sym:
            period = periodicity(expo, symbol)

        else:
            period = _periodicity(f.args, symbol)

    elif f.is_Mul:
        coeff, g = f.as_independent(symbol, as_Add=False)
        if isinstance(g, TrigonometricFunction) or not equal_valued(coeff, 1):
            period = periodicity(g, symbol)
        else:
            period = _periodicity(g.args, symbol)

    elif f.is_Add:
        k, g = f.as_independent(symbol)
        if k is not S.Zero:
            return periodicity(g, symbol)

        period = _periodicity(g.args, symbol)

    elif isinstance(f, Mod):
        a, n = f.args

        if a == symbol:
            period = n
        elif isinstance(a, TrigonometricFunction):
            period = periodicity(a, symbol)
        #check if 'f' is linear in 'symbol'
        elif (a.is_polynomial(symbol) and degree(a, symbol) == 1 and
            symbol not in n.free_symbols):
                period = Abs(n / a.diff(symbol))

    elif isinstance(f, Piecewise):
        pass  # not handling Piecewise yet as the return type is not favorable

    elif period is None:
        from sympy.solvers.decompogen import compogen, decompogen
        g_s = decompogen(f, symbol)
        num_of_gs = len(g_s)
        if num_of_gs > 1:
            for index, g in enumerate(reversed(g_s)):
                start_index = num_of_gs - 1 - index
                g = compogen(g_s[start_index:], symbol)
                if g not in (orig_f, f): # Fix for issue 12620
                    period = periodicity(g, symbol)
                    if period is not None:
                        break

    if period is not None:
        if check:
            return _check(orig_f, period)
        return period

    return None


def _periodicity(args, symbol):
    """
    Helper for `periodicity` to find the period of a list of simpler
    functions.
    It uses the `lcim` method to find the least common period of
    all the functions.

    Parameters
    ==========

    args : Tuple of :py:class:`~.Symbol`
        All the symbols present in a function.

    symbol : :py:class:`~.Symbol`
        The symbol over which the function is to be evaluated.

    Returns
    =======

    period
        The least common period of the function for all the symbols
        of the function.
        ``None`` if for at least one of the symbols the function is aperiodic.

    """
    periods = []
    for f in args:
        period = periodicity(f, symbol)
        if period is None:
            return None

        if period is not S.Zero:
            periods.append(period)

    if len(periods) > 1:
        return lcim(periods)

    if periods:
        return periods[0]


def lcim(numbers):
    """Returns the least common integral multiple of a list of numbers.

    The numbers can be rational or irrational or a mixture of both.
    `None` is returned for incommensurable numbers.

    Parameters
    ==========

    numbers : list
        Numbers (rational and/or irrational) for which lcim is to be found.

    Returns
    =======

    number
        lcim if it exists, otherwise ``None`` for incommensurable numbers.

    Examples
    ========

    >>> from sympy.calculus.util import lcim
    >>> from sympy import S, pi
    >>> lcim([S(1)/2, S(3)/4, S(5)/6])
    15/2
    >>> lcim([2*pi, 3*pi, pi, pi/2])
    6*pi
    >>> lcim([S(1), 2*pi])
    """
    result = None
    if all(num.is_irrational for num in numbers):
        factorized_nums = [num.factor() for num in numbers]
        factors_num = [num.as_coeff_Mul() for num in factorized_nums]
        term = factors_num[0][1]
        if all(factor == term for coeff, factor in factors_num):
            common_term = term
            coeffs = [coeff for coeff, factor in factors_num]
            result = lcm_list(coeffs) * common_term

    elif all(num.is_rational for num in numbers):
        result = lcm_list(numbers)

    else:
        pass

    return result

def is_convex(f, *syms, domain=S.Reals):
    r"""Determines the  convexity of the function passed in the argument.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    syms : Tuple of :py:class:`~.Symbol`
        The variables with respect to which the convexity is to be determined.
    domain : :py:class:`~.Interval`, optional
        The domain over which the convexity of the function has to be checked.
        If unspecified, S.Reals will be the default domain.

    Returns
    =======

    bool
        The method returns ``True`` if the function is convex otherwise it
        returns ``False``.

    Raises
    ======

    NotImplementedError
        The check for the convexity of multivariate functions is not implemented yet.

    Notes
    =====

    To determine concavity of a function pass `-f` as the concerned function.
    To determine logarithmic convexity of a function pass `\log(f)` as
    concerned function.
    To determine logarithmic concavity of a function pass `-\log(f)` as
    concerned function.

    Currently, convexity check of multivariate functions is not handled.

    Examples
    ========

    >>> from sympy import is_convex, symbols, exp, oo, Interval
    >>> x = symbols('x')
    >>> is_convex(exp(x), x)
    True
    >>> is_convex(x**3, x, domain = Interval(-1, oo))
    False
    >>> is_convex(1/x**2, x, domain=Interval.open(0, oo))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Convex_function
    .. [2] http://www.ifp.illinois.edu/~angelia/L3_convfunc.pdf
    .. [3] https://en.wikipedia.org/wiki/Logarithmically_convex_function
    .. [4] https://en.wikipedia.org/wiki/Logarithmically_concave_function
    .. [5] https://en.wikipedia.org/wiki/Concave_function

    """
    if len(syms) > 1 :
        return hessian(f, syms).is_positive_semidefinite
    from sympy.solvers.inequalities import solve_univariate_inequality
    f = _sympify(f)
    var = syms[0]
    if any(s in domain for s in singularities(f, var)):
        return False
    condition = f.diff(var, 2) < 0
    if solve_univariate_inequality(condition, var, False, domain):
        return False
    return True


def stationary_points(f, symbol, domain=S.Reals):
    """
    Returns the stationary points of a function (where derivative of the
    function is 0) in the given domain.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    symbol : :py:class:`~.Symbol`
        The variable for which the stationary points are to be determined.
    domain : :py:class:`~.Interval`
        The domain over which the stationary points have to be checked.
        If unspecified, ``S.Reals`` will be the default domain.

    Returns
    =======

    Set
        A set of stationary points for the function. If there are no
        stationary point, an :py:class:`~.EmptySet` is returned.

    Examples
    ========

    >>> from sympy import Interval, Symbol, S, sin, pi, pprint, stationary_points
    >>> x = Symbol('x')

    >>> stationary_points(1/x, x, S.Reals)
    EmptySet

    >>> pprint(stationary_points(sin(x), x), use_unicode=False)
              pi                              3*pi
    {2*n*pi + -- | n in Integers} U {2*n*pi + ---- | n in Integers}
              2                                2

    >>> stationary_points(sin(x),x, Interval(0, 4*pi))
    {pi/2, 3*pi/2, 5*pi/2, 7*pi/2}

    """
    from sympy.solvers.solveset import solveset

    if domain is S.EmptySet:
        return S.EmptySet

    domain = continuous_domain(f, symbol, domain)
    set = solveset(diff(f, symbol), symbol, domain)

    return set


def maximum(f, symbol, domain=S.Reals):
    """
    Returns the maximum value of a function in the given domain.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    symbol : :py:class:`~.Symbol`
        The variable for maximum value needs to be determined.
    domain : :py:class:`~.Interval`
        The domain over which the maximum have to be checked.
        If unspecified, then the global maximum is returned.

    Returns
    =======

    number
        Maximum value of the function in given domain.

    Examples
    ========

    >>> from sympy import Interval, Symbol, S, sin, cos, pi, maximum
    >>> x = Symbol('x')

    >>> f = -x**2 + 2*x + 5
    >>> maximum(f, x, S.Reals)
    6

    >>> maximum(sin(x), x, Interval(-pi, pi/4))
    sqrt(2)/2

    >>> maximum(sin(x)*cos(x), x)
    1/2

    """
    if isinstance(symbol, Symbol):
        if domain is S.EmptySet:
            raise ValueError("Maximum value not defined for empty domain.")

        return function_range(f, symbol, domain).sup
    else:
        raise ValueError("%s is not a valid symbol." % symbol)


def minimum(f, symbol, domain=S.Reals):
    """
    Returns the minimum value of a function in the given domain.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    symbol : :py:class:`~.Symbol`
        The variable for minimum value needs to be determined.
    domain : :py:class:`~.Interval`
        The domain over which the minimum have to be checked.
        If unspecified, then the global minimum is returned.

    Returns
    =======

    number
        Minimum value of the function in the given domain.

    Examples
    ========

    >>> from sympy import Interval, Symbol, S, sin, cos, minimum
    >>> x = Symbol('x')

    >>> f = x**2 + 2*x + 5
    >>> minimum(f, x, S.Reals)
    4

    >>> minimum(sin(x), x, Interval(2, 3))
    sin(3)

    >>> minimum(sin(x)*cos(x), x)
    -1/2

    """
    if isinstance(symbol, Symbol):
        if domain is S.EmptySet:
            raise ValueError("Minimum value not defined for empty domain.")

        return function_range(f, symbol, domain).inf
    else:
        raise ValueError("%s is not a valid symbol." % symbol)
