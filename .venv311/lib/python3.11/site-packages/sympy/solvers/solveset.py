"""
This module contains functions to:

    - solve a single equation for a single variable, in any domain either real or complex.

    - solve a single transcendental equation for a single variable in any domain either real or complex.
      (currently supports solving in real domain only)

    - solve a system of linear equations with N variables and M equations.

    - solve a system of Non Linear Equations with N variables and M equations
"""
from sympy.core.sympify import sympify
from sympy.core import (S, Pow, Dummy, pi, Expr, Wild, Mul,
                        Add, Basic)
from sympy.core.containers import Tuple
from sympy.core.function import (Lambda, expand_complex, AppliedUndef,
                                expand_log, _mexpand, expand_trig, nfloat)
from sympy.core.mod import Mod
from sympy.core.numbers import I, Number, Rational, oo
from sympy.core.intfunc import integer_log
from sympy.core.relational import Eq, Ne, Relational
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, _uniquely_named_symbol
from sympy.core.sympify import _sympify
from sympy.core.traversal import preorder_traversal
from sympy.external.gmpy import gcd as number_gcd, lcm as number_lcm
from sympy.polys.matrices.linsolve import _linear_eq_to_dict
from sympy.polys.polyroots import UnsolvableFactorError
from sympy.simplify.simplify import simplify, fraction, trigsimp, nsimplify
from sympy.simplify import powdenest, logcombine
from sympy.functions import (log, tan, cot, sin, cos, sec, csc, exp,
                             acos, asin, atan, acot, acsc, asec,
                             piecewise_fold, Piecewise)
from sympy.functions.combinatorial.numbers import totient
from sympy.functions.elementary.complexes import Abs, arg, re, im
from sympy.functions.elementary.hyperbolic import (HyperbolicFunction,
                            sinh, cosh, tanh, coth, sech, csch,
                            asinh, acosh, atanh, acoth, asech, acsch)
from sympy.functions.elementary.miscellaneous import real_root
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.logic.boolalg import And, BooleanTrue
from sympy.sets import (FiniteSet, imageset, Interval, Intersection,
                        Union, ConditionSet, ImageSet, Complement, Contains)
from sympy.sets.sets import Set, ProductSet
from sympy.matrices import zeros, Matrix, MatrixBase
from sympy.ntheory.factor_ import divisors
from sympy.ntheory.residue_ntheory import discrete_log, nthroot_mod
from sympy.polys import (roots, Poly, degree, together, PolynomialError,
                         RootOf, factor, lcm, gcd)
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import invert, groebner, poly
from sympy.polys.solvers import (sympy_eqs_to_ring, solve_lin_sys,
    PolyNonlinearError)
from sympy.polys.matrices.linsolve import _linsolve
from sympy.solvers.solvers import (checksol, denoms, unrad,
    _simple_dens, recast_to_symbols)
from sympy.solvers.polysys import solve_poly_system
from sympy.utilities import filldedent
from sympy.utilities.iterables import (numbered_symbols, has_dups,
                                       is_sequence, iterable)
from sympy.calculus.util import periodicity, continuous_domain, function_range


from types import GeneratorType


class NonlinearError(ValueError):
    """Raised when unexpectedly encountering nonlinear equations"""
    pass


def _masked(f, *atoms):
    """Return ``f``, with all objects given by ``atoms`` replaced with
    Dummy symbols, ``d``, and the list of replacements, ``(d, e)``,
    where ``e`` is an object of type given by ``atoms`` in which
    any other instances of atoms have been recursively replaced with
    Dummy symbols, too. The tuples are ordered so that if they are
    applied in sequence, the origin ``f`` will be restored.

    Examples
    ========

    >>> from sympy import cos
    >>> from sympy.abc import x
    >>> from sympy.solvers.solveset import _masked

    >>> f = cos(cos(x) + 1)
    >>> f, reps = _masked(cos(1 + cos(x)), cos)
    >>> f
    _a1
    >>> reps
    [(_a1, cos(_a0 + 1)), (_a0, cos(x))]
    >>> for d, e in reps:
    ...     f = f.xreplace({d: e})
    >>> f
    cos(cos(x) + 1)
    """
    sym = numbered_symbols('a', cls=Dummy, real=True)
    mask = []
    for a in ordered(f.atoms(*atoms)):
        for i in mask:
            a = a.replace(*i)
        mask.append((a, next(sym)))
    for i, (o, n) in enumerate(mask):
        f = f.replace(o, n)
        mask[i] = (n, o)
    mask = list(reversed(mask))
    return f, mask


def _invert(f_x, y, x, domain=S.Complexes):
    r"""
    Reduce the complex valued equation $f(x) = y$ to a set of equations

    $$\left\{g(x) = h_1(y),\  g(x) = h_2(y),\ \dots,\  g(x) = h_n(y) \right\}$$

    where $g(x)$ is a simpler function than $f(x)$.  The return value is a tuple
    $(g(x), \mathrm{set}_h)$, where $g(x)$ is a function of $x$ and $\mathrm{set}_h$ is
    the set of function $\left\{h_1(y), h_2(y), \dots, h_n(y)\right\}$.
    Here, $y$ is not necessarily a symbol.

    $\mathrm{set}_h$ contains the functions, along with the information
    about the domain in which they are valid, through set
    operations. For instance, if :math:`y = |x| - n` is inverted
    in the real domain, then $\mathrm{set}_h$ is not simply
    $\{-n, n\}$ as the nature of `n` is unknown; rather, it is:

    $$ \left(\left[0, \infty\right) \cap \left\{n\right\}\right) \cup
                       \left(\left(-\infty, 0\right] \cap \left\{- n\right\}\right)$$

    By default, the complex domain is used which means that inverting even
    seemingly simple functions like $\exp(x)$ will give very different
    results from those obtained in the real domain.
    (In the case of $\exp(x)$, the inversion via $\log$ is multi-valued
    in the complex domain, having infinitely many branches.)

    If you are working with real values only (or you are not sure which
    function to use) you should probably set the domain to
    ``S.Reals`` (or use ``invert_real`` which does that automatically).


    Examples
    ========

    >>> from sympy.solvers.solveset import invert_complex, invert_real
    >>> from sympy.abc import x, y
    >>> from sympy import exp

    When does exp(x) == y?

    >>> invert_complex(exp(x), y, x)
    (x, ImageSet(Lambda(_n, I*(2*_n*pi + arg(y)) + log(Abs(y))), Integers))
    >>> invert_real(exp(x), y, x)
    (x, Intersection({log(y)}, Reals))

    When does exp(x) == 1?

    >>> invert_complex(exp(x), 1, x)
    (x, ImageSet(Lambda(_n, 2*_n*I*pi), Integers))
    >>> invert_real(exp(x), 1, x)
    (x, {0})

    See Also
    ========
    invert_real, invert_complex
    """
    x = sympify(x)
    if not x.is_Symbol:
        raise ValueError("x must be a symbol")
    f_x = sympify(f_x)
    if x not in f_x.free_symbols:
        raise ValueError("Inverse of constant function doesn't exist")
    y = sympify(y)
    if x in y.free_symbols:
        raise ValueError("y should be independent of x ")

    if domain.is_subset(S.Reals):
        x1, s = _invert_real(f_x, FiniteSet(y), x)
    else:
        x1, s = _invert_complex(f_x, FiniteSet(y), x)

    # f couldn't be inverted completely; return unmodified.
    if  x1 != x:
        return x1, s

    # Avoid adding gratuitous intersections with S.Complexes. Actual
    # conditions should be handled by the respective inverters.
    if domain is S.Complexes:
        return x1, s

    if isinstance(s, FiniteSet):
        return x1, s.intersect(domain)

    # "Fancier" solution sets like those obtained by inversion of trigonometric
    # functions already include general validity conditions (i.e. conditions on
    # the domain of the respective inverse functions), so we should avoid adding
    # blanket intersections with S.Reals. But subsets of R (or C) must still be
    # accounted for.
    if domain is S.Reals:
        return x1, s
    else:
        return x1, s.intersect(domain)


invert_complex = _invert


def invert_real(f_x, y, x):
    """
    Inverts a real-valued function. Same as :func:`invert_complex`, but sets
    the domain to ``S.Reals`` before inverting.
    """
    return _invert(f_x, y, x, S.Reals)


def _invert_real(f, g_ys, symbol):
    """Helper function for _invert."""

    if f == symbol or g_ys is S.EmptySet:
        return (symbol, g_ys)

    n = Dummy('n', real=True)

    if isinstance(f, exp) or (f.is_Pow and f.base == S.Exp1):
        return _invert_real(f.exp,
                            imageset(Lambda(n, log(n)), g_ys),
                            symbol)

    if hasattr(f, 'inverse') and f.inverse() is not None and not isinstance(f, (
            TrigonometricFunction,
            HyperbolicFunction,
            )):
        if len(f.args) > 1:
            raise ValueError("Only functions with one argument are supported.")
        return _invert_real(f.args[0],
                            imageset(Lambda(n, f.inverse()(n)), g_ys),
                            symbol)

    if isinstance(f, Abs):
        return _invert_abs(f.args[0], g_ys, symbol)

    if f.is_Add:
        # f = g + h
        g, h = f.as_independent(symbol)
        if g is not S.Zero:
            return _invert_real(h, imageset(Lambda(n, n - g), g_ys), symbol)

    if f.is_Mul:
        # f = g*h
        g, h = f.as_independent(symbol)

        if g is not S.One:
            return _invert_real(h, imageset(Lambda(n, n/g), g_ys), symbol)

    if f.is_Pow:
        base, expo = f.args
        base_has_sym = base.has(symbol)
        expo_has_sym = expo.has(symbol)

        if not expo_has_sym:

            if expo.is_rational:
                num, den = expo.as_numer_denom()

                if den % 2 == 0 and num % 2 == 1 and den.is_zero is False:
                    # Here we have f(x)**(num/den) = y
                    # where den is nonzero and even and y is an element
                    # of the set g_ys.
                    # den is even, so we are only interested in the cases
                    # where both f(x) and y are positive.
                    # Restricting y to be positive (using the set g_ys_pos)
                    # means that y**(den/num) is always positive.
                    # Therefore it isn't necessary to also constrain f(x)
                    # to be positive because we are only going to
                    # find solutions of f(x) = y**(d/n)
                    # where the rhs is already required to be positive.
                    root = Lambda(n, real_root(n, expo))
                    g_ys_pos = g_ys & Interval(0, oo)
                    res = imageset(root, g_ys_pos)
                    _inv, _set = _invert_real(base, res, symbol)
                    return (_inv, _set)

                if den % 2 == 1:
                    root = Lambda(n, real_root(n, expo))
                    res = imageset(root, g_ys)
                    if num % 2 == 0:
                        neg_res = imageset(Lambda(n, -n), res)
                        return _invert_real(base, res + neg_res, symbol)
                    if num % 2 == 1:
                        return _invert_real(base, res, symbol)

            elif expo.is_irrational:
                root = Lambda(n, real_root(n, expo))
                g_ys_pos = g_ys & Interval(0, oo)
                res = imageset(root, g_ys_pos)
                return _invert_real(base, res, symbol)

            else:
                # indeterminate exponent, e.g. Float or parity of
                # num, den of rational could not be determined
                pass  # use default return

        if not base_has_sym:
            rhs = g_ys.args[0]
            if base.is_positive:
                return _invert_real(expo,
                    imageset(Lambda(n, log(n, base, evaluate=False)), g_ys), symbol)
            elif base.is_negative:
                s, b = integer_log(rhs, base)
                if b:
                    return _invert_real(expo, FiniteSet(s), symbol)
                else:
                    return (expo, S.EmptySet)
            elif base.is_zero:
                one = Eq(rhs, 1)
                if one == S.true:
                    # special case: 0**x - 1
                    return _invert_real(expo, FiniteSet(0), symbol)
                elif one == S.false:
                    return (expo, S.EmptySet)

    if isinstance(f, (TrigonometricFunction, HyperbolicFunction)):
        return _invert_trig_hyp_real(f, g_ys, symbol)

    return (f, g_ys)


# Dictionaries of inverses will be cached after first use.
_trig_inverses = None
_hyp_inverses = None

def _invert_trig_hyp_real(f, g_ys, symbol):
    """Helper function for inverting trigonometric and hyperbolic functions.

    This helper only handles inversion over the reals.

    For trigonometric functions only finite `g_ys` sets are implemented.

    For hyperbolic functions the set `g_ys` is checked against the domain of the
    respective inverse functions. Infinite `g_ys` sets are also supported.
    """

    if isinstance(f, HyperbolicFunction):
        n = Dummy('n', real=True)

        if isinstance(f, sinh):
            # asinh is defined over R.
            return _invert_real(f.args[0], imageset(n, asinh(n), g_ys), symbol)

        if isinstance(f, cosh):
            g_ys_dom = g_ys.intersect(Interval(1, oo))
            if isinstance(g_ys_dom, Intersection):
                # could not properly resolve domain check
                if isinstance(g_ys, FiniteSet):
                    # If g_ys is a `FiniteSet`` it should be sufficient to just
                    # let the calling `_invert_real()` add an intersection with
                    # `S.Reals` (or a subset `domain`) to ensure that only valid
                    # (real) solutions are returned.
                    # This avoids adding "too many" Intersections or
                    # ConditionSets in the returned set.
                    g_ys_dom = g_ys
                else:
                    return (f, g_ys)
            return _invert_real(f.args[0], Union(
                imageset(n, acosh(n), g_ys_dom),
                imageset(n, -acosh(n), g_ys_dom)), symbol)

        if isinstance(f, sech):
            g_ys_dom = g_ys.intersect(Interval.Lopen(0, 1))
            if isinstance(g_ys_dom, Intersection):
                if isinstance(g_ys, FiniteSet):
                    g_ys_dom = g_ys
                else:
                    return (f, g_ys)
            return _invert_real(f.args[0], Union(
                imageset(n, asech(n), g_ys_dom),
                imageset(n, -asech(n), g_ys_dom)), symbol)

        if isinstance(f, tanh):
            g_ys_dom = g_ys.intersect(Interval.open(-1, 1))
            if isinstance(g_ys_dom, Intersection):
                if isinstance(g_ys, FiniteSet):
                    g_ys_dom = g_ys
                else:
                    return (f, g_ys)
            return _invert_real(f.args[0],
                imageset(n, atanh(n), g_ys_dom), symbol)

        if isinstance(f, coth):
            g_ys_dom = g_ys - Interval(-1, 1)
            if isinstance(g_ys_dom, Complement):
                if isinstance(g_ys, FiniteSet):
                    g_ys_dom = g_ys
                else:
                    return (f, g_ys)
            return _invert_real(f.args[0],
                imageset(n, acoth(n), g_ys_dom), symbol)

        if isinstance(f, csch):
            g_ys_dom = g_ys - FiniteSet(0)
            if isinstance(g_ys_dom, Complement):
                if isinstance(g_ys, FiniteSet):
                    g_ys_dom = g_ys
                else:
                    return (f, g_ys)
            return _invert_real(f.args[0],
                imageset(n, acsch(n), g_ys_dom), symbol)

    elif isinstance(f, TrigonometricFunction) and isinstance(g_ys, FiniteSet):
        def _get_trig_inverses(func):
            global _trig_inverses
            if _trig_inverses is None:
                _trig_inverses = {
                    sin : ((asin, lambda y: pi-asin(y)), 2*pi, Interval(-1, 1)),
                    cos : ((acos, lambda y: -acos(y)), 2*pi, Interval(-1, 1)),
                    tan : ((atan,), pi, S.Reals),
                    cot : ((acot,), pi, S.Reals),
                    sec : ((asec, lambda y: -asec(y)), 2*pi,
                        Union(Interval(-oo, -1), Interval(1, oo))),
                    csc : ((acsc, lambda y: pi-acsc(y)), 2*pi,
                        Union(Interval(-oo, -1), Interval(1, oo)))}
            return _trig_inverses[func]

        invs, period, rng = _get_trig_inverses(f.func)
        n = Dummy('n', integer=True)
        def create_return_set(g):
            # returns ConditionSet that will be part of the final (x, set) tuple
            invsimg = Union(*[
                imageset(n, period*n + inv(g), S.Integers) for inv in invs])
            inv_f, inv_g_ys = _invert_real(f.args[0], invsimg, symbol)
            if inv_f == symbol:     # inversion successful
                conds = rng.contains(g)
                return ConditionSet(symbol, conds, inv_g_ys)
            else:
                return ConditionSet(symbol, Eq(f, g), S.Reals)

        retset = Union(*[create_return_set(g) for g in g_ys])
        return (symbol, retset)

    else:
        return (f, g_ys)


def _invert_trig_hyp_complex(f, g_ys, symbol):
    """Helper function for inverting trigonometric and hyperbolic functions.

    This helper only handles inversion over the complex numbers.
    Only finite `g_ys` sets are implemented.

    Handling of singularities is only implemented for hyperbolic equations.
    In case of a symbolic element g in g_ys a ConditionSet may be returned.
    """

    if isinstance(f, TrigonometricFunction) and isinstance(g_ys, FiniteSet):
        def inv(trig):
            if isinstance(trig, (sin, csc)):
                F = asin if isinstance(trig, sin) else acsc
                return (
                    lambda a: 2*n*pi + F(a),
                    lambda a: 2*n*pi + pi - F(a))
            if isinstance(trig, (cos, sec)):
                F = acos if isinstance(trig, cos) else asec
                return (
                    lambda a: 2*n*pi + F(a),
                    lambda a: 2*n*pi - F(a))
            if isinstance(trig, (tan, cot)):
                return (lambda a: n*pi + trig.inverse()(a),)

        n = Dummy('n', integer=True)
        invs = S.EmptySet
        for L in inv(f):
            invs += Union(*[imageset(Lambda(n, L(g)), S.Integers) for g in g_ys])
        return _invert_complex(f.args[0], invs, symbol)

    elif isinstance(f, HyperbolicFunction) and isinstance(g_ys, FiniteSet):
        # There are two main options regarding singularities / domain checking
        # for symbolic elements in g_ys:
        # 1. Add a "catch-all" intersection with S.Complexes.
        # 2. ConditionSets.
        # At present ConditionSets seem to work better and have the additional
        # benefit of representing the precise conditions that must be satisfied.
        # The conditions are also rather straightforward. (At most two isolated
        # points.)
        def _get_hyp_inverses(func):
            global _hyp_inverses
            if _hyp_inverses is None:
                _hyp_inverses = {
                    sinh : ((asinh, lambda y: I*pi-asinh(y)), 2*I*pi, ()),
                    cosh : ((acosh, lambda y: -acosh(y)), 2*I*pi, ()),
                    tanh : ((atanh,), I*pi, (-1, 1)),
                    coth : ((acoth,), I*pi, (-1, 1)),
                    sech : ((asech, lambda y: -asech(y)), 2*I*pi, (0, )),
                    csch : ((acsch, lambda y: I*pi-acsch(y)), 2*I*pi, (0, ))}
            return _hyp_inverses[func]

        # invs: iterable of main inverses, e.g. (acosh, -acosh).
        # excl: iterable of singularities to be checked for.
        invs, period, excl = _get_hyp_inverses(f.func)
        n = Dummy('n', integer=True)
        def create_return_set(g):
            # returns ConditionSet that will be part of the final (x, set) tuple
            invsimg = Union(*[
                imageset(n, period*n + inv(g), S.Integers) for inv in invs])
            inv_f, inv_g_ys = _invert_complex(f.args[0], invsimg, symbol)
            if inv_f == symbol:     # inversion successful
                conds = And(*[Ne(g, e) for e in excl])
                return ConditionSet(symbol, conds, inv_g_ys)
            else:
                return ConditionSet(symbol, Eq(f, g), S.Complexes)

        retset = Union(*[create_return_set(g) for g in g_ys])
        return (symbol, retset)

    else:
        return (f, g_ys)


def _invert_complex(f, g_ys, symbol):
    """Helper function for _invert."""

    if f == symbol or g_ys is S.EmptySet:
        return (symbol, g_ys)

    n = Dummy('n')

    if f.is_Add:
        # f = g + h
        g, h = f.as_independent(symbol)
        if g is not S.Zero:
            return _invert_complex(h, imageset(Lambda(n, n - g), g_ys), symbol)

    if f.is_Mul:
        # f = g*h
        g, h = f.as_independent(symbol)

        if g is not S.One:
            if g in {S.NegativeInfinity, S.ComplexInfinity, S.Infinity}:
                return (h, S.EmptySet)
            return _invert_complex(h, imageset(Lambda(n, n/g), g_ys), symbol)

    if f.is_Pow:
        base, expo = f.args
        # special case: g**r = 0
        # Could be improved like `_invert_real` to handle more general cases.
        if expo.is_Rational and g_ys == FiniteSet(0):
            if expo.is_positive:
                return _invert_complex(base, g_ys, symbol)

    if hasattr(f, 'inverse') and f.inverse() is not None and \
       not isinstance(f, TrigonometricFunction) and \
       not isinstance(f, HyperbolicFunction) and \
       not isinstance(f, exp):
        if len(f.args) > 1:
            raise ValueError("Only functions with one argument are supported.")
        return _invert_complex(f.args[0],
                               imageset(Lambda(n, f.inverse()(n)), g_ys), symbol)

    if isinstance(f, exp) or (f.is_Pow and f.base == S.Exp1):
        if isinstance(g_ys, ImageSet):
            # can solve up to `(d*exp(exp(...(exp(a*x + b))...) + c)` format.
            # Further can be improved to `(d*exp(exp(...(exp(a*x**n + b*x**(n-1) + ... + f))...) + c)`.
            g_ys_expr = g_ys.lamda.expr
            g_ys_vars = g_ys.lamda.variables
            k = Dummy('k{}'.format(len(g_ys_vars)))
            g_ys_vars_1 = (k,) + g_ys_vars
            exp_invs = Union(*[imageset(Lambda((g_ys_vars_1,), (I*(2*k*pi + arg(g_ys_expr))
                                         + log(Abs(g_ys_expr)))), S.Integers**(len(g_ys_vars_1)))])
            return _invert_complex(f.exp, exp_invs, symbol)

        elif isinstance(g_ys, FiniteSet):
            exp_invs = Union(*[imageset(Lambda(n, I*(2*n*pi + arg(g_y)) +
                                               log(Abs(g_y))), S.Integers)
                               for g_y in g_ys if g_y != 0])
            return _invert_complex(f.exp, exp_invs, symbol)

    if isinstance(f, (TrigonometricFunction, HyperbolicFunction)):
        return _invert_trig_hyp_complex(f, g_ys, symbol)

    return (f, g_ys)


def _invert_abs(f, g_ys, symbol):
    """Helper function for inverting absolute value functions.

    Returns the complete result of inverting an absolute value
    function along with the conditions which must also be satisfied.

    If it is certain that all these conditions are met, a :class:`~.FiniteSet`
    of all possible solutions is returned. If any condition cannot be
    satisfied, an :class:`~.EmptySet` is returned. Otherwise, a
    :class:`~.ConditionSet` of the solutions, with all the required conditions
    specified, is returned.

    """
    if not g_ys.is_FiniteSet:
        # this could be used for FiniteSet, but the
        # results are more compact if they aren't, e.g.
        # ConditionSet(x, Contains(n, Interval(0, oo)), {-n, n}) vs
        # Union(Intersection(Interval(0, oo), {n}), Intersection(Interval(-oo, 0), {-n}))
        # for the solution of abs(x) - n
        pos = Intersection(g_ys, Interval(0, S.Infinity))
        parg = _invert_real(f, pos, symbol)
        narg = _invert_real(-f, pos, symbol)
        if parg[0] != narg[0]:
            raise NotImplementedError
        return parg[0], Union(narg[1], parg[1])

    # check conditions: all these must be true. If any are unknown
    # then return them as conditions which must be satisfied
    unknown = []
    for a in g_ys.args:
        ok = a.is_nonnegative if a.is_Number else a.is_positive
        if ok is None:
            unknown.append(a)
        elif not ok:
            return symbol, S.EmptySet
    if unknown:
        conditions = And(*[Contains(i, Interval(0, oo))
            for i in unknown])
    else:
        conditions = True
    n = Dummy('n', real=True)
    # this is slightly different than above: instead of solving
    # +/-f on positive values, here we solve for f on +/- g_ys
    g_x, values = _invert_real(f, Union(
        imageset(Lambda(n, n), g_ys),
        imageset(Lambda(n, -n), g_ys)), symbol)
    return g_x, ConditionSet(g_x, conditions, values)


def domain_check(f, symbol, p):
    """Returns False if point p is infinite or any subexpression of f
    is infinite or becomes so after replacing symbol with p. If none of
    these conditions is met then True will be returned.

    Examples
    ========

    >>> from sympy import Mul, oo
    >>> from sympy.abc import x
    >>> from sympy.solvers.solveset import domain_check
    >>> g = 1/(1 + (1/(x + 1))**2)
    >>> domain_check(g, x, -1)
    False
    >>> domain_check(x**2, x, 0)
    True
    >>> domain_check(1/x, x, oo)
    False

    * The function relies on the assumption that the original form
      of the equation has not been changed by automatic simplification.

    >>> domain_check(x/x, x, 0) # x/x is automatically simplified to 1
    True

    * To deal with automatic evaluations use evaluate=False:

    >>> domain_check(Mul(x, 1/x, evaluate=False), x, 0)
    False
    """
    f, p = sympify(f), sympify(p)
    if p.is_infinite:
        return False
    return _domain_check(f, symbol, p)


def _domain_check(f, symbol, p):
    # helper for domain check
    if f.is_Atom and f.is_finite:
        return True
    elif f.subs(symbol, p).is_infinite:
        return False
    elif isinstance(f, Piecewise):
        # Check the cases of the Piecewise in turn. There might be invalid
        # expressions in later cases that don't apply e.g.
        #    solveset(Piecewise((0, Eq(x, 0)), (1/x, True)), x)
        for expr, cond in f.args:
            condsubs = cond.subs(symbol, p)
            if condsubs is S.false:
                continue
            elif condsubs is S.true:
                return _domain_check(expr, symbol, p)
            else:
                # We don't know which case of the Piecewise holds. On this
                # basis we cannot decide whether any solution is in or out of
                # the domain. Ideally this function would allow returning a
                # symbolic condition for the validity of the solution that
                # could be handled in the calling code. In the mean time we'll
                # give this particular solution the benefit of the doubt and
                # let it pass.
                return True
    else:
        # TODO : We should not blindly recurse through all args of arbitrary expressions like this
        return all(_domain_check(g, symbol, p)
                   for g in f.args)


def _is_finite_with_finite_vars(f, domain=S.Complexes):
    """
    Return True if the given expression is finite. For symbols that
    do not assign a value for `complex` and/or `real`, the domain will
    be used to assign a value; symbols that do not assign a value
    for `finite` will be made finite. All other assumptions are
    left unmodified.
    """
    def assumptions(s):
        A = s.assumptions0
        A.setdefault('finite', A.get('finite', True))
        if domain.is_subset(S.Reals):
            # if this gets set it will make complex=True, too
            A.setdefault('real', True)
        else:
            # don't change 'real' because being complex implies
            # nothing about being real
            A.setdefault('complex', True)
        return A

    reps = {s: Dummy(**assumptions(s)) for s in f.free_symbols}
    return f.xreplace(reps).is_finite


def _is_function_class_equation(func_class, f, symbol):
    """ Tests whether the equation is an equation of the given function class.

    The given equation belongs to the given function class if it is
    comprised of functions of the function class which are multiplied by
    or added to expressions independent of the symbol. In addition, the
    arguments of all such functions must be linear in the symbol as well.

    Examples
    ========

    >>> from sympy.solvers.solveset import _is_function_class_equation
    >>> from sympy import tan, sin, tanh, sinh, exp
    >>> from sympy.abc import x
    >>> from sympy.functions.elementary.trigonometric import TrigonometricFunction
    >>> from sympy.functions.elementary.hyperbolic import HyperbolicFunction
    >>> _is_function_class_equation(TrigonometricFunction, exp(x) + tan(x), x)
    False
    >>> _is_function_class_equation(TrigonometricFunction, tan(x) + sin(x), x)
    True
    >>> _is_function_class_equation(TrigonometricFunction, tan(x**2), x)
    False
    >>> _is_function_class_equation(TrigonometricFunction, tan(x + 2), x)
    True
    >>> _is_function_class_equation(HyperbolicFunction, tanh(x) + sinh(x), x)
    True
    """
    if f.is_Mul or f.is_Add:
        return all(_is_function_class_equation(func_class, arg, symbol)
                   for arg in f.args)

    if f.is_Pow:
        if not f.exp.has(symbol):
            return _is_function_class_equation(func_class, f.base, symbol)
        else:
            return False

    if not f.has(symbol):
        return True

    if isinstance(f, func_class):
        try:
            g = Poly(f.args[0], symbol)
            return g.degree() <= 1
        except PolynomialError:
            return False
    else:
        return False


def _solve_as_rational(f, symbol, domain):
    """ solve rational functions"""
    f = together(_mexpand(f, recursive=True), deep=True)
    g, h = fraction(f)
    if not h.has(symbol):
        try:
            return _solve_as_poly(g, symbol, domain)
        except NotImplementedError:
            # The polynomial formed from g could end up having
            # coefficients in a ring over which finding roots
            # isn't implemented yet, e.g. ZZ[a] for some symbol a
            return ConditionSet(symbol, Eq(f, 0), domain)
        except CoercionFailed:
            # contained oo, zoo or nan
            return S.EmptySet
    else:
        valid_solns = _solveset(g, symbol, domain)
        invalid_solns = _solveset(h, symbol, domain)
        return valid_solns - invalid_solns


class _SolveTrig1Error(Exception):
    """Raised when _solve_trig1 heuristics do not apply"""

def _solve_trig(f, symbol, domain):
    """Function to call other helpers to solve trigonometric equations """
    # If f is composed of a single trig function (potentially appearing multiple
    # times) we should solve by either inverting directly or inverting after a
    # suitable change of variable.
    #
    # _solve_trig is currently only called by _solveset for trig/hyperbolic
    # functions of an argument linear in x. Inverting a symbolic argument should
    # include a guard against division by zero in order to have a result that is
    # consistent with similar processing done by _solve_trig1.
    # (Ideally _invert should add these conditions by itself.)
    trig_expr, count = None, 0
    for expr in preorder_traversal(f):
        if isinstance(expr, (TrigonometricFunction,
                            HyperbolicFunction)) and expr.has(symbol):
            if not trig_expr:
                trig_expr, count = expr, 1
            elif expr == trig_expr:
                count += 1
            else:
                trig_expr, count = False, 0
                break
    if count == 1:
        # direct inversion
        x, sol = _invert(f, 0, symbol, domain)
        if x == symbol:
            cond = True
            if trig_expr.free_symbols - {symbol}:
                a, h = trig_expr.args[0].as_independent(symbol, as_Add=True)
                m, h = h.as_independent(symbol, as_Add=False)
                num, den = m.as_numer_denom()
                cond = Ne(num, 0) & Ne(den, 0)
            return ConditionSet(symbol, cond, sol)
        else:
            return ConditionSet(symbol, Eq(f, 0), domain)
    elif count:
        # solve by change of variable
        y = Dummy('y')
        f_cov = f.subs(trig_expr, y)
        sol_cov = solveset(f_cov, y, domain)
        if isinstance(sol_cov, FiniteSet):
            return Union(
                *[_solve_trig(trig_expr-s, symbol, domain) for s in sol_cov])

    sol = None
    try:
        # multiple trig/hyp functions; solve by rewriting to exp
        sol = _solve_trig1(f, symbol, domain)
    except _SolveTrig1Error:
        try:
            # multiple trig/hyp functions; solve by rewriting to tan(x/2)
            sol = _solve_trig2(f, symbol, domain)
        except ValueError:
            raise NotImplementedError(filldedent('''
                Solution to this kind of trigonometric equations
                is yet to be implemented'''))
    return sol


def _solve_trig1(f, symbol, domain):
    """Primary solver for trigonometric and hyperbolic equations

    Returns either the solution set as a ConditionSet (auto-evaluated to a
    union of ImageSets if no variables besides 'symbol' are involved) or
    raises _SolveTrig1Error if f == 0 cannot be solved.

    Notes
    =====
    Algorithm:
    1. Do a change of variable x -> mu*x in arguments to trigonometric and
    hyperbolic functions, in order to reduce them to small integers. (This
    step is crucial to keep the degrees of the polynomials of step 4 low.)
    2. Rewrite trigonometric/hyperbolic functions as exponentials.
    3. Proceed to a 2nd change of variable, replacing exp(I*x) or exp(x) by y.
    4. Solve the resulting rational equation.
    5. Use invert_complex or invert_real to return to the original variable.
    6. If the coefficients of 'symbol' were symbolic in nature, add the
    necessary consistency conditions in a ConditionSet.

    """
    # Prepare change of variable
    x = Dummy('x')
    if _is_function_class_equation(HyperbolicFunction, f, symbol):
        cov = exp(x)
        inverter = invert_real if domain.is_subset(S.Reals) else invert_complex
    else:
        cov = exp(I*x)
        inverter = invert_complex

    f = trigsimp(f)
    f_original = f
    trig_functions = f.atoms(TrigonometricFunction, HyperbolicFunction)
    trig_arguments = [e.args[0] for e in trig_functions]
    # trigsimp may have reduced the equation to an expression
    # that is independent of 'symbol' (e.g. cos**2+sin**2)
    if not any(a.has(symbol) for a in trig_arguments):
        return solveset(f_original, symbol, domain)

    denominators = []
    numerators = []
    for ar in trig_arguments:
        try:
            poly_ar = Poly(ar, symbol)
        except PolynomialError:
            raise _SolveTrig1Error("trig argument is not a polynomial")
        if poly_ar.degree() > 1:  # degree >1 still bad
            raise _SolveTrig1Error("degree of variable must not exceed one")
        if poly_ar.degree() == 0:  # degree 0, don't care
            continue
        c = poly_ar.all_coeffs()[0]   # got the coefficient of 'symbol'
        numerators.append(fraction(c)[0])
        denominators.append(fraction(c)[1])

    mu = lcm(denominators)/gcd(numerators)
    f = f.subs(symbol, mu*x)
    f = f.rewrite(exp)
    f = together(f)
    g, h = fraction(f)
    y = Dummy('y')
    g, h = g.expand(), h.expand()
    g, h = g.subs(cov, y), h.subs(cov, y)
    if g.has(x) or h.has(x):
        raise _SolveTrig1Error("change of variable not possible")

    solns = solveset_complex(g, y) - solveset_complex(h, y)
    if isinstance(solns, ConditionSet):
        raise _SolveTrig1Error("polynomial has ConditionSet solution")

    if isinstance(solns, FiniteSet):
        if any(isinstance(s, RootOf) for s in solns):
            raise _SolveTrig1Error("polynomial results in RootOf object")
        # revert the change of variable
        cov = cov.subs(x, symbol/mu)
        result = Union(*[inverter(cov, s, symbol)[1] for s in solns])
        # In case of symbolic coefficients, the solution set is only valid
        # if numerator and denominator of mu are non-zero.
        if mu.has(Symbol):
            syms = (mu).atoms(Symbol)
            munum, muden = fraction(mu)
            condnum = munum.as_independent(*syms, as_Add=False)[1]
            condden = muden.as_independent(*syms, as_Add=False)[1]
            cond = And(Ne(condnum, 0), Ne(condden, 0))
        else:
            cond = True
        # Actual conditions are returned as part of the ConditionSet. Adding an
        # intersection with C would only complicate some solution sets due to
        # current limitations of intersection code. (e.g. #19154)
        if domain is S.Complexes:
            # This is a slight abuse of ConditionSet. Ideally this should
            # be some kind of "PiecewiseSet". (See #19507 discussion)
            return ConditionSet(symbol, cond, result)
        else:
            return ConditionSet(symbol, cond, Intersection(result, domain))
    elif solns is S.EmptySet:
        return S.EmptySet
    else:
        raise _SolveTrig1Error("polynomial solutions must form FiniteSet")


def _solve_trig2(f, symbol, domain):
    """Secondary helper to solve trigonometric equations,
    called when first helper fails """
    f = trigsimp(f)
    f_original = f
    trig_functions = f.atoms(sin, cos, tan, sec, cot, csc)
    trig_arguments = [e.args[0] for e in trig_functions]
    denominators = []
    numerators = []

    # todo: This solver can be extended to hyperbolics if the
    # analogous change of variable to tanh (instead of tan)
    # is used.
    if not trig_functions:
        return ConditionSet(symbol, Eq(f_original, 0), domain)

    # todo: The pre-processing below (extraction of numerators, denominators,
    # gcd, lcm, mu, etc.) should be updated to the enhanced version in
    # _solve_trig1. (See #19507)
    for ar in trig_arguments:
        try:
            poly_ar = Poly(ar, symbol)
        except PolynomialError:
            raise ValueError("give up, we cannot solve if this is not a polynomial in x")
        if poly_ar.degree() > 1:  # degree >1 still bad
            raise ValueError("degree of variable inside polynomial should not exceed one")
        if poly_ar.degree() == 0:  # degree 0, don't care
            continue
        c = poly_ar.all_coeffs()[0]   # got the coefficient of 'symbol'
        try:
            numerators.append(Rational(c).p)
            denominators.append(Rational(c).q)
        except TypeError:
            return ConditionSet(symbol, Eq(f_original, 0), domain)

    x = Dummy('x')

    mu = Rational(2)*number_lcm(*denominators)/number_gcd(*numerators)
    f = f.subs(symbol, mu*x)
    f = f.rewrite(tan)
    f = expand_trig(f)
    f = together(f)

    g, h = fraction(f)
    y = Dummy('y')
    g, h = g.expand(), h.expand()
    g, h = g.subs(tan(x), y), h.subs(tan(x), y)

    if g.has(x) or h.has(x):
        return ConditionSet(symbol, Eq(f_original, 0), domain)
    solns = solveset(g, y, S.Reals) - solveset(h, y, S.Reals)

    if isinstance(solns, FiniteSet):
        result = Union(*[invert_real(tan(symbol/mu), s, symbol)[1]
                       for s in solns])
        dsol = invert_real(tan(symbol/mu), oo, symbol)[1]
        if degree(h) > degree(g):                   # If degree(denom)>degree(num) then there
            result = Union(result, dsol)            # would be another sol at Lim(denom-->oo)
        return Intersection(result, domain)
    elif solns is S.EmptySet:
        return S.EmptySet
    else:
        return ConditionSet(symbol, Eq(f_original, 0), S.Reals)


def _solve_as_poly(f, symbol, domain=S.Complexes):
    """
    Solve the equation using polynomial techniques if it already is a
    polynomial equation or, with a change of variables, can be made so.
    """
    result = None
    if f.is_polynomial(symbol):
        solns = roots(f, symbol, cubics=True, quartics=True,
                      quintics=True, domain='EX')
        num_roots = sum(solns.values())
        if degree(f, symbol) <= num_roots:
            result = FiniteSet(*solns.keys())
        else:
            poly = Poly(f, symbol)
            solns = poly.all_roots()
            if poly.degree() <= len(solns):
                result = FiniteSet(*solns)
            else:
                result = ConditionSet(symbol, Eq(f, 0), domain)
    else:
        poly = Poly(f)
        if poly is None:
            result = ConditionSet(symbol, Eq(f, 0), domain)
        gens = [g for g in poly.gens if g.has(symbol)]

        if len(gens) == 1:
            poly = Poly(poly, gens[0])
            gen = poly.gen
            deg = poly.degree()
            poly = Poly(poly.as_expr(), poly.gen, composite=True)
            poly_solns = FiniteSet(*roots(poly, cubics=True, quartics=True,
                                          quintics=True).keys())

            if len(poly_solns) < deg:
                result = ConditionSet(symbol, Eq(f, 0), domain)

            if gen != symbol:
                y = Dummy('y')
                inverter = invert_real if domain.is_subset(S.Reals) else invert_complex
                lhs, rhs_s = inverter(gen, y, symbol)
                if lhs == symbol:
                    result = Union(*[rhs_s.subs(y, s) for s in poly_solns])
                    if isinstance(result, FiniteSet) and isinstance(gen, Pow
                            ) and gen.base.is_Rational:
                        result = FiniteSet(*[expand_log(i) for i in result])
                else:
                    result = ConditionSet(symbol, Eq(f, 0), domain)
        else:
            result = ConditionSet(symbol, Eq(f, 0), domain)

    if result is not None:
        if isinstance(result, FiniteSet):
            # this is to simplify solutions like -sqrt(-I) to sqrt(2)/2
            # - sqrt(2)*I/2. We are not expanding for solution with symbols
            # or undefined functions because that makes the solution more complicated.
            # For example, expand_complex(a) returns re(a) + I*im(a)
            if all(s.atoms(Symbol, AppliedUndef) == set() and not isinstance(s, RootOf)
                   for s in result):
                s = Dummy('s')
                result = imageset(Lambda(s, expand_complex(s)), result)
        if isinstance(result, FiniteSet) and domain != S.Complexes:
            # Avoid adding gratuitous intersections with S.Complexes. Actual
            # conditions should be handled elsewhere.
            result = result.intersection(domain)
        return result
    else:
        return ConditionSet(symbol, Eq(f, 0), domain)


def _solve_radical(f, unradf, symbol, solveset_solver):
    """ Helper function to solve equations with radicals """
    res = unradf
    eq, cov = res if res else (f, [])
    if not cov:
        result = solveset_solver(eq, symbol) - \
            Union(*[solveset_solver(g, symbol) for g in denoms(f, symbol)])
    else:
        y, yeq = cov
        if not solveset_solver(y - I, y):
            yreal = Dummy('yreal', real=True)
            yeq = yeq.xreplace({y: yreal})
            eq = eq.xreplace({y: yreal})
            y = yreal
        g_y_s = solveset_solver(yeq, symbol)
        f_y_sols = solveset_solver(eq, y)
        result = Union(*[imageset(Lambda(y, g_y), f_y_sols)
                         for g_y in g_y_s])

    def check_finiteset(solutions):
        f_set = []  # solutions for FiniteSet
        c_set = []  # solutions for ConditionSet
        for s in solutions:
            if checksol(f, symbol, s):
                f_set.append(s)
            else:
                c_set.append(s)
        return FiniteSet(*f_set) + ConditionSet(symbol, Eq(f, 0), FiniteSet(*c_set))

    def check_set(solutions):
        if solutions is S.EmptySet:
            return solutions
        elif isinstance(solutions, ConditionSet):
            # XXX: Maybe the base set should be checked?
            return solutions
        elif isinstance(solutions, FiniteSet):
            return check_finiteset(solutions)
        elif isinstance(solutions, Complement):
            A, B = solutions.args
            return Complement(check_set(A), B)
        elif isinstance(solutions, Union):
            return Union(*[check_set(s) for s in solutions.args])
        else:
            # XXX: There should be more cases checked here. The cases above
            # are all those that come up in the test suite for now.
            return solutions

    solution_set = check_set(result)

    return solution_set


def _solve_abs(f, symbol, domain):
    """ Helper function to solve equation involving absolute value function """
    if not domain.is_subset(S.Reals):
        raise ValueError(filldedent('''
            Absolute values cannot be inverted in the
            complex domain.'''))
    p, q, r = Wild('p'), Wild('q'), Wild('r')
    pattern_match = f.match(p*Abs(q) + r) or {}
    f_p, f_q, f_r = [pattern_match.get(i, S.Zero) for i in (p, q, r)]

    if not (f_p.is_zero or f_q.is_zero):
        domain = continuous_domain(f_q, symbol, domain)
        from .inequalities import solve_univariate_inequality
        q_pos_cond = solve_univariate_inequality(f_q >= 0, symbol,
                                                 relational=False, domain=domain, continuous=True)
        q_neg_cond = q_pos_cond.complement(domain)

        sols_q_pos = solveset_real(f_p*f_q + f_r,
                                           symbol).intersect(q_pos_cond)
        sols_q_neg = solveset_real(f_p*(-f_q) + f_r,
                                           symbol).intersect(q_neg_cond)
        return Union(sols_q_pos, sols_q_neg)
    else:
        return ConditionSet(symbol, Eq(f, 0), domain)


def solve_decomposition(f, symbol, domain):
    """
    Function to solve equations via the principle of "Decomposition
    and Rewriting".

    Examples
    ========
    >>> from sympy import exp, sin, Symbol, pprint, S
    >>> from sympy.solvers.solveset import solve_decomposition as sd
    >>> x = Symbol('x')
    >>> f1 = exp(2*x) - 3*exp(x) + 2
    >>> sd(f1, x, S.Reals)
    {0, log(2)}
    >>> f2 = sin(x)**2 + 2*sin(x) + 1
    >>> pprint(sd(f2, x, S.Reals), use_unicode=False)
              3*pi
    {2*n*pi + ---- | n in Integers}
               2
    >>> f3 = sin(x + 2)
    >>> pprint(sd(f3, x, S.Reals), use_unicode=False)
    {2*n*pi - 2 | n in Integers} U {2*n*pi - 2 + pi | n in Integers}

    """
    from sympy.solvers.decompogen import decompogen
    # decompose the given function
    g_s = decompogen(f, symbol)
    # `y_s` represents the set of values for which the function `g` is to be
    # solved.
    # `solutions` represent the solutions of the equations `g = y_s` or
    # `g = 0` depending on the type of `y_s`.
    # As we are interested in solving the equation: f = 0
    y_s = FiniteSet(0)
    for g in g_s:
        frange = function_range(g, symbol, domain)
        y_s = Intersection(frange, y_s)
        result = S.EmptySet
        if isinstance(y_s, FiniteSet):
            for y in y_s:
                solutions = solveset(Eq(g, y), symbol, domain)
                if not isinstance(solutions, ConditionSet):
                    result += solutions

        else:
            if isinstance(y_s, ImageSet):
                iter_iset = (y_s,)

            elif isinstance(y_s, Union):
                iter_iset = y_s.args

            elif y_s is S.EmptySet:
                # y_s is not in the range of g in g_s, so no solution exists
                #in the given domain
                return S.EmptySet

            for iset in iter_iset:
                new_solutions = solveset(Eq(iset.lamda.expr, g), symbol, domain)
                dummy_var = tuple(iset.lamda.expr.free_symbols)[0]
                (base_set,) = iset.base_sets
                if isinstance(new_solutions, FiniteSet):
                    new_exprs = new_solutions

                elif isinstance(new_solutions, Intersection):
                    if isinstance(new_solutions.args[1], FiniteSet):
                        new_exprs = new_solutions.args[1]

                for new_expr in new_exprs:
                    result += ImageSet(Lambda(dummy_var, new_expr), base_set)

        if result is S.EmptySet:
            return ConditionSet(symbol, Eq(f, 0), domain)

        y_s = result

    return y_s


def _solveset(f, symbol, domain, _check=False):
    """Helper for solveset to return a result from an expression
    that has already been sympify'ed and is known to contain the
    given symbol."""
    # _check controls whether the answer is checked or not
    from sympy.simplify.simplify import signsimp

    if isinstance(f, BooleanTrue):
        return domain

    orig_f = f
    if f.is_Mul:
        coeff, f = f.as_independent(symbol, as_Add=False)
        if coeff in {S.ComplexInfinity, S.NegativeInfinity, S.Infinity}:
            f = together(orig_f)
    elif f.is_Add:
        a, h = f.as_independent(symbol)
        m, h = h.as_independent(symbol, as_Add=False)
        if m not in {S.ComplexInfinity, S.Zero, S.Infinity,
                              S.NegativeInfinity}:
            f = a/m + h  # XXX condition `m != 0` should be added to soln

    # assign the solvers to use
    solver = lambda f, x, domain=domain: _solveset(f, x, domain)
    inverter = lambda f, rhs, symbol: _invert(f, rhs, symbol, domain)

    result = S.EmptySet

    if f.expand().is_zero:
        return domain
    elif not f.has(symbol):
        return S.EmptySet
    elif f.is_Mul and all(_is_finite_with_finite_vars(m, domain)
            for m in f.args):
        # if f(x) and g(x) are both finite we can say that the solution of
        # f(x)*g(x) == 0 is same as Union(f(x) == 0, g(x) == 0) is not true in
        # general. g(x) can grow to infinitely large for the values where
        # f(x) == 0. To be sure that we are not silently allowing any
        # wrong solutions we are using this technique only if both f and g are
        # finite for a finite input.
        result = Union(*[solver(m, symbol) for m in f.args])
    elif (_is_function_class_equation(TrigonometricFunction, f, symbol) or \
            _is_function_class_equation(HyperbolicFunction, f, symbol)):
        result = _solve_trig(f, symbol, domain)
    elif isinstance(f, arg):
        a = f.args[0]
        result = Intersection(_solveset(re(a) > 0, symbol, domain),
                              _solveset(im(a), symbol, domain))
    elif f.is_Piecewise:
        expr_set_pairs = f.as_expr_set_pairs(domain)
        for (expr, in_set) in expr_set_pairs:
            if in_set.is_Relational:
                in_set = in_set.as_set()
            solns = solver(expr, symbol, in_set)
            result += solns
    elif isinstance(f, Eq):
        result = solver(Add(f.lhs, -f.rhs, evaluate=False), symbol, domain)

    elif f.is_Relational:
        from .inequalities import solve_univariate_inequality
        try:
            result = solve_univariate_inequality(
            f, symbol, domain=domain, relational=False)
        except NotImplementedError:
            result = ConditionSet(symbol, f, domain)
        return result
    elif _is_modular(f, symbol):
        result = _solve_modular(f, symbol, domain)
    else:
        lhs, rhs_s = inverter(f, 0, symbol)
        if lhs == symbol:
            # do some very minimal simplification since
            # repeated inversion may have left the result
            # in a state that other solvers (e.g. poly)
            # would have simplified; this is done here
            # rather than in the inverter since here it
            # is only done once whereas there it would
            # be repeated for each step of the inversion
            if isinstance(rhs_s, FiniteSet):
                rhs_s = FiniteSet(*[Mul(*
                    signsimp(i).as_content_primitive())
                    for i in rhs_s])
            result = rhs_s

        elif isinstance(rhs_s, FiniteSet):
            for equation in [lhs - rhs for rhs in rhs_s]:
                if equation == f:
                    u = unrad(f, symbol)
                    if u:
                        result += _solve_radical(equation, u,
                                                 symbol,
                                                 solver)
                    elif equation.has(Abs):
                        result += _solve_abs(f, symbol, domain)
                    else:
                        result_rational = _solve_as_rational(equation, symbol, domain)
                        if not isinstance(result_rational, ConditionSet):
                            result += result_rational
                        else:
                            # may be a transcendental type equation
                            t_result = _transolve(equation, symbol, domain)
                            if isinstance(t_result, ConditionSet):
                                # might need factoring; this is expensive so we
                                # have delayed until now. To avoid recursion
                                # errors look for a non-trivial factoring into
                                # a product of symbol dependent terms; I think
                                # that something that factors as a Pow would
                                # have already been recognized by now.
                                factored = equation.factor()
                                if factored.is_Mul and equation != factored:
                                    _, dep = factored.as_independent(symbol)
                                    if not dep.is_Add:
                                        # non-trivial factoring of equation
                                        # but use form with constants
                                        # in case they need special handling
                                        t_results = []
                                        for fac in Mul.make_args(factored):
                                            if fac.has(symbol):
                                                t_results.append(solver(fac, symbol))
                                        t_result = Union(*t_results)
                            result += t_result
                else:
                    result += solver(equation, symbol)

        elif rhs_s is not S.EmptySet:
            result = ConditionSet(symbol, Eq(f, 0), domain)

    if isinstance(result, ConditionSet):
        if isinstance(f, Expr):
            num, den = f.as_numer_denom()
            if den.has(symbol):
                _result = _solveset(num, symbol, domain)
                if not isinstance(_result, ConditionSet):
                    singularities = _solveset(den, symbol, domain)
                    result = _result - singularities

    if _check:
        if isinstance(result, ConditionSet):
            # it wasn't solved or has enumerated all conditions
            # -- leave it alone
            return result

        # whittle away all but the symbol-containing core
        # to use this for testing
        if isinstance(orig_f, Expr):
            fx = orig_f.as_independent(symbol, as_Add=True)[1]
            fx = fx.as_independent(symbol, as_Add=False)[1]
        else:
            fx = orig_f

        if isinstance(result, FiniteSet):
            # check the result for invalid solutions
            result = FiniteSet(*[s for s in result
                      if isinstance(s, RootOf)
                      or domain_check(fx, symbol, s)])

    return result


def _is_modular(f, symbol):
    """
    Helper function to check below mentioned types of modular equations.
    ``A - Mod(B, C) = 0``

    A -> This can or cannot be a function of symbol.
    B -> This is surely a function of symbol.
    C -> It is an integer.

    Parameters
    ==========

    f : Expr
        The equation to be checked.

    symbol : Symbol
        The concerned variable for which the equation is to be checked.

    Examples
    ========

    >>> from sympy import symbols, exp, Mod
    >>> from sympy.solvers.solveset import _is_modular as check
    >>> x, y = symbols('x y')
    >>> check(Mod(x, 3) - 1, x)
    True
    >>> check(Mod(x, 3) - 1, y)
    False
    >>> check(Mod(x, 3)**2 - 5, x)
    False
    >>> check(Mod(x, 3)**2 - y, x)
    False
    >>> check(exp(Mod(x, 3)) - 1, x)
    False
    >>> check(Mod(3, y) - 1, y)
    False
    """

    if not f.has(Mod):
        return False

    # extract modterms from f.
    modterms = list(f.atoms(Mod))

    return (len(modterms) == 1 and  # only one Mod should be present
            modterms[0].args[0].has(symbol) and  # B-> function of symbol
            modterms[0].args[1].is_integer and  # C-> to be an integer.
            any(isinstance(term, Mod)
            for term in list(_term_factors(f)))  # free from other funcs
            )


def _invert_modular(modterm, rhs, n, symbol):
    """
    Helper function to invert modular equation.
    ``Mod(a, m) - rhs = 0``

    Generally it is inverted as (a, ImageSet(Lambda(n, m*n + rhs), S.Integers)).
    More simplified form will be returned if possible.

    If it is not invertible then (modterm, rhs) is returned.

    The following cases arise while inverting equation ``Mod(a, m) - rhs = 0``:

    1. If a is symbol then  m*n + rhs is the required solution.

    2. If a is an instance of ``Add`` then we try to find two symbol independent
       parts of a and the symbol independent part gets transferred to the other
       side and again the ``_invert_modular`` is called on the symbol
       dependent part.

    3. If a is an instance of ``Mul`` then same as we done in ``Add`` we separate
       out the symbol dependent and symbol independent parts and transfer the
       symbol independent part to the rhs with the help of invert and again the
       ``_invert_modular`` is called on the symbol dependent part.

    4. If a is an instance of ``Pow`` then two cases arise as following:

        - If a is of type (symbol_indep)**(symbol_dep) then the remainder is
          evaluated with the help of discrete_log function and then the least
          period is being found out with the help of totient function.
          period*n + remainder is the required solution in this case.
          For reference: (https://en.wikipedia.org/wiki/Euler's_theorem)

        - If a is of type (symbol_dep)**(symbol_indep) then we try to find all
          primitive solutions list with the help of nthroot_mod function.
          m*n + rem is the general solution where rem belongs to solutions list
          from nthroot_mod function.

    Parameters
    ==========

    modterm, rhs : Expr
        The modular equation to be inverted, ``modterm - rhs = 0``

    symbol : Symbol
        The variable in the equation to be inverted.

    n : Dummy
        Dummy variable for output g_n.

    Returns
    =======

    A tuple (f_x, g_n) is being returned where f_x is modular independent function
    of symbol and g_n being set of values f_x can have.

    Examples
    ========

    >>> from sympy import symbols, exp, Mod, Dummy, S
    >>> from sympy.solvers.solveset import _invert_modular as invert_modular
    >>> x, y = symbols('x y')
    >>> n = Dummy('n')
    >>> invert_modular(Mod(exp(x), 7), S(5), n, x)
    (Mod(exp(x), 7), 5)
    >>> invert_modular(Mod(x, 7), S(5), n, x)
    (x, ImageSet(Lambda(_n, 7*_n + 5), Integers))
    >>> invert_modular(Mod(3*x + 8, 7), S(5), n, x)
    (x, ImageSet(Lambda(_n, 7*_n + 6), Integers))
    >>> invert_modular(Mod(x**4, 7), S(5), n, x)
    (x, EmptySet)
    >>> invert_modular(Mod(2**(x**2 + x + 1), 7), S(2), n, x)
    (x**2 + x + 1, ImageSet(Lambda(_n, 3*_n + 1), Naturals0))

    """
    a, m = modterm.args

    if rhs.is_integer is False:
        return symbol, S.EmptySet

    if rhs.is_real is False or any(term.is_real is False
            for term in list(_term_factors(a))):
        # Check for complex arguments
        return modterm, rhs

    if abs(rhs) >= abs(m):
        # if rhs has value greater than value of m.
        return symbol, S.EmptySet

    if a == symbol:
        return symbol, ImageSet(Lambda(n, m*n + rhs), S.Integers)

    if a.is_Add:
        # g + h = a
        g, h = a.as_independent(symbol)
        if g is not S.Zero:
            x_indep_term = rhs - Mod(g, m)
            return _invert_modular(Mod(h, m), Mod(x_indep_term, m), n, symbol)

    if a.is_Mul:
        # g*h = a
        g, h = a.as_independent(symbol)
        if g is not S.One:
            x_indep_term = rhs*invert(g, m)
            return _invert_modular(Mod(h, m), Mod(x_indep_term, m), n, symbol)

    if a.is_Pow:
        # base**expo = a
        base, expo = a.args
        if expo.has(symbol) and not base.has(symbol):
            # remainder -> solution independent of n of equation.
            # m, rhs are made coprime by dividing number_gcd(m, rhs)
            if not m.is_Integer and rhs.is_Integer and a.base.is_Integer:
                return modterm, rhs

            mdiv = m.p // number_gcd(m.p, rhs.p)
            try:
                remainder = discrete_log(mdiv, rhs.p, a.base.p)
            except ValueError:  # log does not exist
                return modterm, rhs
            # period -> coefficient of n in the solution and also referred as
            # the least period of expo in which it is repeats itself.
            # (a**(totient(m)) - 1) divides m. Here is link of theorem:
            # (https://en.wikipedia.org/wiki/Euler's_theorem)
            period = totient(m)
            for p in divisors(period):
                # there might a lesser period exist than totient(m).
                if pow(a.base, p, m / number_gcd(m.p, a.base.p)) == 1:
                    period = p
                    break
            # recursion is not applied here since _invert_modular is currently
            # not smart enough to handle infinite rhs as here expo has infinite
            # rhs = ImageSet(Lambda(n, period*n + remainder), S.Naturals0).
            return expo, ImageSet(Lambda(n, period*n + remainder), S.Naturals0)
        elif base.has(symbol) and not expo.has(symbol):
            try:
                remainder_list = nthroot_mod(rhs, expo, m, all_roots=True)
                if remainder_list == []:
                    return symbol, S.EmptySet
            except (ValueError, NotImplementedError):
                return modterm, rhs
            g_n = S.EmptySet
            for rem in remainder_list:
                g_n += ImageSet(Lambda(n, m*n + rem), S.Integers)
            return base, g_n

    return modterm, rhs


def _solve_modular(f, symbol, domain):
    r"""
    Helper function for solving modular equations of type ``A - Mod(B, C) = 0``,
    where A can or cannot be a function of symbol, B is surely a function of
    symbol and C is an integer.

    Currently ``_solve_modular`` is only able to solve cases
    where A is not a function of symbol.

    Parameters
    ==========

    f : Expr
        The modular equation to be solved, ``f = 0``

    symbol : Symbol
        The variable in the equation to be solved.

    domain : Set
        A set over which the equation is solved. It has to be a subset of
        Integers.

    Returns
    =======

    A set of integer solutions satisfying the given modular equation.
    A ``ConditionSet`` if the equation is unsolvable.

    Examples
    ========

    >>> from sympy.solvers.solveset import _solve_modular as solve_modulo
    >>> from sympy import S, Symbol, sin, Intersection, Interval, Mod
    >>> x = Symbol('x')
    >>> solve_modulo(Mod(5*x - 8, 7) - 3, x, S.Integers)
    ImageSet(Lambda(_n, 7*_n + 5), Integers)
    >>> solve_modulo(Mod(5*x - 8, 7) - 3, x, S.Reals)  # domain should be subset of integers.
    ConditionSet(x, Eq(Mod(5*x + 6, 7) - 3, 0), Reals)
    >>> solve_modulo(-7 + Mod(x, 5), x, S.Integers)
    EmptySet
    >>> solve_modulo(Mod(12**x, 21) - 18, x, S.Integers)
    ImageSet(Lambda(_n, 6*_n + 2), Naturals0)
    >>> solve_modulo(Mod(sin(x), 7) - 3, x, S.Integers) # not solvable
    ConditionSet(x, Eq(Mod(sin(x), 7) - 3, 0), Integers)
    >>> solve_modulo(3 - Mod(x, 5), x, Intersection(S.Integers, Interval(0, 100)))
    Intersection(ImageSet(Lambda(_n, 5*_n + 3), Integers), Range(0, 101, 1))
    """
    # extract modterm and g_y from f
    unsolved_result = ConditionSet(symbol, Eq(f, 0), domain)
    modterm = list(f.atoms(Mod))[0]
    rhs = -S.One*(f.subs(modterm, S.Zero))
    if f.as_coefficients_dict()[modterm].is_negative:
        # checks if coefficient of modterm is negative in main equation.
        rhs *= -S.One

    if not domain.is_subset(S.Integers):
        return unsolved_result

    if rhs.has(symbol):
        # TODO Case: A-> function of symbol, can be extended here
        # in future.
        return unsolved_result

    n = Dummy('n', integer=True)
    f_x, g_n = _invert_modular(modterm, rhs, n, symbol)

    if f_x == modterm and g_n == rhs:
        return unsolved_result

    if f_x == symbol:
        if domain is not S.Integers:
            return domain.intersect(g_n)
        return g_n

    if isinstance(g_n, ImageSet):
        lamda_expr = g_n.lamda.expr
        lamda_vars = g_n.lamda.variables
        base_sets = g_n.base_sets
        sol_set = _solveset(f_x - lamda_expr, symbol, S.Integers)
        if isinstance(sol_set, FiniteSet):
            tmp_sol = S.EmptySet
            for sol in sol_set:
                tmp_sol += ImageSet(Lambda(lamda_vars, sol), *base_sets)
            sol_set = tmp_sol
        else:
            sol_set =  ImageSet(Lambda(lamda_vars, sol_set), *base_sets)
        return domain.intersect(sol_set)

    return unsolved_result


def _term_factors(f):
    """
    Iterator to get the factors of all terms present
    in the given equation.

    Parameters
    ==========
    f : Expr
        Equation that needs to be addressed

    Returns
    =======
    Factors of all terms present in the equation.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.solvers.solveset import _term_factors
    >>> x = symbols('x')
    >>> list(_term_factors(-2 - x**2 + x*(x + 1)))
    [-2, -1, x**2, x, x + 1]
    """
    for add_arg in Add.make_args(f):
        yield from Mul.make_args(add_arg)


def _solve_exponential(lhs, rhs, symbol, domain):
    r"""
    Helper function for solving (supported) exponential equations.

    Exponential equations are the sum of (currently) at most
    two terms with one or both of them having a power with a
    symbol-dependent exponent.

    For example

    .. math:: 5^{2x + 3} - 5^{3x - 1}

    .. math:: 4^{5 - 9x} - e^{2 - x}

    Parameters
    ==========

    lhs, rhs : Expr
        The exponential equation to be solved, `lhs = rhs`

    symbol : Symbol
        The variable in which the equation is solved

    domain : Set
        A set over which the equation is solved.

    Returns
    =======

    A set of solutions satisfying the given equation.
    A ``ConditionSet`` if the equation is unsolvable or
    if the assumptions are not properly defined, in that case
    a different style of ``ConditionSet`` is returned having the
    solution(s) of the equation with the desired assumptions.

    Examples
    ========

    >>> from sympy.solvers.solveset import _solve_exponential as solve_expo
    >>> from sympy import symbols, S
    >>> x = symbols('x', real=True)
    >>> a, b = symbols('a b')
    >>> solve_expo(2**x + 3**x - 5**x, 0, x, S.Reals)  # not solvable
    ConditionSet(x, Eq(2**x + 3**x - 5**x, 0), Reals)
    >>> solve_expo(a**x - b**x, 0, x, S.Reals)  # solvable but incorrect assumptions
    ConditionSet(x, (a > 0) & (b > 0), {0})
    >>> solve_expo(3**(2*x) - 2**(x + 3), 0, x, S.Reals)
    {-3*log(2)/(-2*log(3) + log(2))}
    >>> solve_expo(2**x - 4**x, 0, x, S.Reals)
    {0}

    * Proof of correctness of the method

    The logarithm function is the inverse of the exponential function.
    The defining relation between exponentiation and logarithm is:

    .. math:: {\log_b x} = y \enspace if \enspace b^y = x

    Therefore if we are given an equation with exponent terms, we can
    convert every term to its corresponding logarithmic form. This is
    achieved by taking logarithms and expanding the equation using
    logarithmic identities so that it can easily be handled by ``solveset``.

    For example:

    .. math:: 3^{2x} = 2^{x + 3}

    Taking log both sides will reduce the equation to

    .. math:: (2x)\log(3) = (x + 3)\log(2)

    This form can be easily handed by ``solveset``.
    """
    unsolved_result = ConditionSet(symbol, Eq(lhs - rhs, 0), domain)
    newlhs = powdenest(lhs)
    if lhs != newlhs:
        # it may also be advantageous to factor the new expr
        neweq = factor(newlhs - rhs)
        if neweq != (lhs - rhs):
            return _solveset(neweq, symbol, domain)  # try again with _solveset

    if not (isinstance(lhs, Add) and len(lhs.args) == 2):
        # solving for the sum of more than two powers is possible
        # but not yet implemented
        return unsolved_result

    if rhs != 0:
        return unsolved_result

    a, b = list(ordered(lhs.args))
    a_term = a.as_independent(symbol)[1]
    b_term = b.as_independent(symbol)[1]

    a_base, a_exp = a_term.as_base_exp()
    b_base, b_exp = b_term.as_base_exp()

    if domain.is_subset(S.Reals):
        conditions = And(
            a_base > 0,
            b_base > 0,
            Eq(im(a_exp), 0),
            Eq(im(b_exp), 0))
    else:
        conditions = And(
            Ne(a_base, 0),
            Ne(b_base, 0))

    L, R = (expand_log(log(i), force=True) for i in (a, -b))
    solutions = _solveset(L - R, symbol, domain)

    return ConditionSet(symbol, conditions, solutions)


def _is_exponential(f, symbol):
    r"""
    Return ``True`` if one or more terms contain ``symbol`` only in
    exponents, else ``False``.

    Parameters
    ==========

    f : Expr
        The equation to be checked

    symbol : Symbol
        The variable in which the equation is checked

    Examples
    ========

    >>> from sympy import symbols, cos, exp
    >>> from sympy.solvers.solveset import _is_exponential as check
    >>> x, y = symbols('x y')
    >>> check(y, y)
    False
    >>> check(x**y - 1, y)
    True
    >>> check(x**y*2**y - 1, y)
    True
    >>> check(exp(x + 3) + 3**x, x)
    True
    >>> check(cos(2**x), x)
    False

    * Philosophy behind the helper

    The function extracts each term of the equation and checks if it is
    of exponential form w.r.t ``symbol``.
    """
    rv = False
    for expr_arg in _term_factors(f):
        if symbol not in expr_arg.free_symbols:
            continue
        if (isinstance(expr_arg, Pow) and
           symbol not in expr_arg.base.free_symbols or
           isinstance(expr_arg, exp)):
            rv = True  # symbol in exponent
        else:
            return False  # dependent on symbol in non-exponential way
    return rv


def _solve_logarithm(lhs, rhs, symbol, domain):
    r"""
    Helper to solve logarithmic equations which are reducible
    to a single instance of `\log`.

    Logarithmic equations are (currently) the equations that contains
    `\log` terms which can be reduced to a single `\log` term or
    a constant using various logarithmic identities.

    For example:

    .. math:: \log(x) + \log(x - 4)

    can be reduced to:

    .. math:: \log(x(x - 4))

    Parameters
    ==========

    lhs, rhs : Expr
        The logarithmic equation to be solved, `lhs = rhs`

    symbol : Symbol
        The variable in which the equation is solved

    domain : Set
        A set over which the equation is solved.

    Returns
    =======

    A set of solutions satisfying the given equation.
    A ``ConditionSet`` if the equation is unsolvable.

    Examples
    ========

    >>> from sympy import symbols, log, S
    >>> from sympy.solvers.solveset import _solve_logarithm as solve_log
    >>> x = symbols('x')
    >>> f = log(x - 3) + log(x + 3)
    >>> solve_log(f, 0, x, S.Reals)
    {-sqrt(10), sqrt(10)}

    * Proof of correctness

    A logarithm is another way to write exponent and is defined by

    .. math:: {\log_b x} = y \enspace if \enspace b^y = x

    When one side of the equation contains a single logarithm, the
    equation can be solved by rewriting the equation as an equivalent
    exponential equation as defined above. But if one side contains
    more than one logarithm, we need to use the properties of logarithm
    to condense it into a single logarithm.

    Take for example

    .. math:: \log(2x) - 15 = 0

    contains single logarithm, therefore we can directly rewrite it to
    exponential form as

    .. math:: x = \frac{e^{15}}{2}

    But if the equation has more than one logarithm as

    .. math:: \log(x - 3) + \log(x + 3) = 0

    we use logarithmic identities to convert it into a reduced form

    Using,

    .. math:: \log(a) + \log(b) = \log(ab)

    the equation becomes,

    .. math:: \log((x - 3)(x + 3))

    This equation contains one logarithm and can be solved by rewriting
    to exponents.
    """
    new_lhs = logcombine(lhs, force=True)
    new_f = new_lhs - rhs

    return _solveset(new_f, symbol, domain)


def _is_logarithmic(f, symbol):
    r"""
    Return ``True`` if the equation is in the form
    `a\log(f(x)) + b\log(g(x)) + ... + c` else ``False``.

    Parameters
    ==========

    f : Expr
        The equation to be checked

    symbol : Symbol
        The variable in which the equation is checked

    Returns
    =======

    ``True`` if the equation is logarithmic otherwise ``False``.

    Examples
    ========

    >>> from sympy import symbols, tan, log
    >>> from sympy.solvers.solveset import _is_logarithmic as check
    >>> x, y = symbols('x y')
    >>> check(log(x + 2) - log(x + 3), x)
    True
    >>> check(tan(log(2*x)), x)
    False
    >>> check(x*log(x), x)
    False
    >>> check(x + log(x), x)
    False
    >>> check(y + log(x), x)
    True

    * Philosophy behind the helper

    The function extracts each term and checks whether it is
    logarithmic w.r.t ``symbol``.
    """
    rv = False
    for term in Add.make_args(f):
        saw_log = False
        for term_arg in Mul.make_args(term):
            if symbol not in term_arg.free_symbols:
                continue
            if isinstance(term_arg, log):
                if saw_log:
                    return False  # more than one log in term
                saw_log = True
            else:
                return False  # dependent on symbol in non-log way
        if saw_log:
            rv = True
    return rv


def _is_lambert(f, symbol):
    r"""
    If this returns ``False`` then the Lambert solver (``_solve_lambert``) will not be called.

    Explanation
    ===========

    Quick check for cases that the Lambert solver might be able to handle.

    1. Equations containing more than two operands and `symbol`s involving any of
       `Pow`, `exp`, `HyperbolicFunction`,`TrigonometricFunction`, `log` terms.

    2. In `Pow`, `exp` the exponent should have `symbol` whereas for
       `HyperbolicFunction`,`TrigonometricFunction`, `log` should contain `symbol`.

    3. For `HyperbolicFunction`,`TrigonometricFunction` the number of trigonometric functions in
       equation should be less than number of symbols. (since `A*cos(x) + B*sin(x) - c`
       is not the Lambert type).

    Some forms of lambert equations are:
        1. X**X = C
        2. X*(B*log(X) + D)**A = C
        3. A*log(B*X + A) + d*X = C
        4. (B*X + A)*exp(d*X + g) = C
        5. g*exp(B*X + h) - B*X = C
        6. A*D**(E*X + g) - B*X = C
        7. A*cos(X) + B*sin(X) - D*X = C
        8. A*cosh(X) + B*sinh(X) - D*X = C

    Where X is any variable,
          A, B, C, D, E are any constants,
          g, h are linear functions or log terms.

    Parameters
    ==========

    f : Expr
        The equation to be checked

    symbol : Symbol
        The variable in which the equation is checked

    Returns
    =======

    If this returns ``False`` then the Lambert solver (``_solve_lambert``) will not be called.

    Examples
    ========

    >>> from sympy.solvers.solveset import _is_lambert
    >>> from sympy import symbols, cosh, sinh, log
    >>> x = symbols('x')

    >>> _is_lambert(3*log(x) - x*log(3), x)
    True
    >>> _is_lambert(log(log(x - 3)) + log(x-3), x)
    True
    >>> _is_lambert(cosh(x) - sinh(x), x)
    False
    >>> _is_lambert((x**2 - 2*x + 1).subs(x, (log(x) + 3*x)**2 - 1), x)
    True

    See Also
    ========

    _solve_lambert

    """
    term_factors = list(_term_factors(f.expand()))

    # total number of symbols in equation
    no_of_symbols = len([arg for arg in term_factors if arg.has(symbol)])
    # total number of trigonometric terms in equation
    no_of_trig = len([arg for arg in term_factors \
        if arg.has(HyperbolicFunction, TrigonometricFunction)])

    if f.is_Add and no_of_symbols >= 2:
        # `log`, `HyperbolicFunction`, `TrigonometricFunction` should have symbols
        # and no_of_trig < no_of_symbols
        lambert_funcs = (log, HyperbolicFunction, TrigonometricFunction)
        if any(isinstance(arg, lambert_funcs)\
            for arg in term_factors if arg.has(symbol)):
                if no_of_trig < no_of_symbols:
                    return True
        # here, `Pow`, `exp` exponent should have symbols
        elif any(isinstance(arg, (Pow, exp)) \
            for arg in term_factors if (arg.as_base_exp()[1]).has(symbol)):
            return True
    return False


def _transolve(f, symbol, domain):
    r"""
    Function to solve transcendental equations. It is a helper to
    ``solveset`` and should be used internally. ``_transolve``
    currently supports the following class of equations:

        - Exponential equations
        - Logarithmic equations

    Parameters
    ==========

    f : Any transcendental equation that needs to be solved.
        This needs to be an expression, which is assumed
        to be equal to ``0``.

    symbol : The variable for which the equation is solved.
        This needs to be of class ``Symbol``.

    domain : A set over which the equation is solved.
        This needs to be of class ``Set``.

    Returns
    =======

    Set
        A set of values for ``symbol`` for which ``f`` is equal to
        zero. An ``EmptySet`` is returned if ``f`` does not have solutions
        in respective domain. A ``ConditionSet`` is returned as unsolved
        object if algorithms to evaluate complete solution are not
        yet implemented.

    How to use ``_transolve``
    =========================

    ``_transolve`` should not be used as an independent function, because
    it assumes that the equation (``f``) and the ``symbol`` comes from
    ``solveset`` and might have undergone a few modification(s).
    To use ``_transolve`` as an independent function the equation (``f``)
    and the ``symbol`` should be passed as they would have been by
    ``solveset``.

    Examples
    ========

    >>> from sympy.solvers.solveset import _transolve as transolve
    >>> from sympy.solvers.solvers import _tsolve as tsolve
    >>> from sympy import symbols, S, pprint
    >>> x = symbols('x', real=True) # assumption added
    >>> transolve(5**(x - 3) - 3**(2*x + 1), x, S.Reals)
    {-(log(3) + 3*log(5))/(-log(5) + 2*log(3))}

    How ``_transolve`` works
    ========================

    ``_transolve`` uses two types of helper functions to solve equations
    of a particular class:

    Identifying helpers: To determine whether a given equation
    belongs to a certain class of equation or not. Returns either
    ``True`` or ``False``.

    Solving helpers: Once an equation is identified, a corresponding
    helper either solves the equation or returns a form of the equation
    that ``solveset`` might better be able to handle.

    * Philosophy behind the module

    The purpose of ``_transolve`` is to take equations which are not
    already polynomial in their generator(s) and to either recast them
    as such through a valid transformation or to solve them outright.
    A pair of helper functions for each class of supported
    transcendental functions are employed for this purpose. One
    identifies the transcendental form of an equation and the other
    either solves it or recasts it into a tractable form that can be
    solved by  ``solveset``.
    For example, an equation in the form `ab^{f(x)} - cd^{g(x)} = 0`
    can be transformed to
    `\log(a) + f(x)\log(b) - \log(c) - g(x)\log(d) = 0`
    (under certain assumptions) and this can be solved with ``solveset``
    if `f(x)` and `g(x)` are in polynomial form.

    How ``_transolve`` is better than ``_tsolve``
    =============================================

    1) Better output

    ``_transolve`` provides expressions in a more simplified form.

    Consider a simple exponential equation

    >>> f = 3**(2*x) - 2**(x + 3)
    >>> pprint(transolve(f, x, S.Reals), use_unicode=False)
        -3*log(2)
    {------------------}
     -2*log(3) + log(2)
    >>> pprint(tsolve(f, x), use_unicode=False)
         /   3     \
         | --------|
         | log(2/9)|
    [-log\2         /]

    2) Extensible

    The API of ``_transolve`` is designed such that it is easily
    extensible, i.e. the code that solves a given class of
    equations is encapsulated in a helper and not mixed in with
    the code of ``_transolve`` itself.

    3) Modular

    ``_transolve`` is designed to be modular i.e, for every class of
    equation a separate helper for identification and solving is
    implemented. This makes it easy to change or modify any of the
    method implemented directly in the helpers without interfering
    with the actual structure of the API.

    4) Faster Computation

    Solving equation via ``_transolve`` is much faster as compared to
    ``_tsolve``. In ``solve``, attempts are made computing every possibility
    to get the solutions. This series of attempts makes solving a bit
    slow. In ``_transolve``, computation begins only after a particular
    type of equation is identified.

    How to add new class of equations
    =================================

    Adding a new class of equation solver is a three-step procedure:

    - Identify the type of the equations

      Determine the type of the class of equations to which they belong:
      it could be of ``Add``, ``Pow``, etc. types. Separate internal functions
      are used for each type. Write identification and solving helpers
      and use them from within the routine for the given type of equation
      (after adding it, if necessary). Something like:

      .. code-block:: python

        def add_type(lhs, rhs, x):
            ....
            if _is_exponential(lhs, x):
                new_eq = _solve_exponential(lhs, rhs, x)
        ....
        rhs, lhs = eq.as_independent(x)
        if lhs.is_Add:
            result = add_type(lhs, rhs, x)

    - Define the identification helper.

    - Define the solving helper.

    Apart from this, a few other things needs to be taken care while
    adding an equation solver:

    - Naming conventions:
      Name of the identification helper should be as
      ``_is_class`` where class will be the name or abbreviation
      of the class of equation. The solving helper will be named as
      ``_solve_class``.
      For example: for exponential equations it becomes
      ``_is_exponential`` and ``_solve_expo``.
    - The identifying helpers should take two input parameters,
      the equation to be checked and the variable for which a solution
      is being sought, while solving helpers would require an additional
      domain parameter.
    - Be sure to consider corner cases.
    - Add tests for each helper.
    - Add a docstring to your helper that describes the method
      implemented.
      The documentation of the helpers should identify:

      - the purpose of the helper,
      - the method used to identify and solve the equation,
      - a proof of correctness
      - the return values of the helpers
    """

    def add_type(lhs, rhs, symbol, domain):
        """
        Helper for ``_transolve`` to handle equations of
        ``Add`` type, i.e. equations taking the form as
        ``a*f(x) + b*g(x) + .... = c``.
        For example: 4**x + 8**x = 0
        """
        result = ConditionSet(symbol, Eq(lhs - rhs, 0), domain)

        # check if it is exponential type equation
        if _is_exponential(lhs, symbol):
            result = _solve_exponential(lhs, rhs, symbol, domain)
        # check if it is logarithmic type equation
        elif _is_logarithmic(lhs, symbol):
            result = _solve_logarithm(lhs, rhs, symbol, domain)

        return result

    result = ConditionSet(symbol, Eq(f, 0), domain)

    # invert_complex handles the call to the desired inverter based
    # on the domain specified.
    lhs, rhs_s = invert_complex(f, 0, symbol, domain)

    if isinstance(rhs_s, FiniteSet):
        assert (len(rhs_s.args)) == 1
        rhs = rhs_s.args[0]

        if lhs.is_Add:
            result = add_type(lhs, rhs, symbol, domain)
    else:
        result = rhs_s

    return result


def solveset(f, symbol=None, domain=S.Complexes):
    r"""Solves a given inequality or equation with set as output

    Parameters
    ==========

    f : Expr or a relational.
        The target equation or inequality
    symbol : Symbol
        The variable for which the equation is solved
    domain : Set
        The domain over which the equation is solved

    Returns
    =======

    Set
        A set of values for `symbol` for which `f` is True or is equal to
        zero. An :class:`~.EmptySet` is returned if `f` is False or nonzero.
        A :class:`~.ConditionSet` is returned as unsolved object if algorithms
        to evaluate complete solution are not yet implemented.

    ``solveset`` claims to be complete in the solution set that it returns.

    Raises
    ======

    NotImplementedError
        The algorithms to solve inequalities in complex domain  are
        not yet implemented.
    ValueError
        The input is not valid.
    RuntimeError
        It is a bug, please report to the github issue tracker.


    Notes
    =====

    Python interprets 0 and 1 as False and True, respectively, but
    in this function they refer to solutions of an expression. So 0 and 1
    return the domain and EmptySet, respectively, while True and False
    return the opposite (as they are assumed to be solutions of relational
    expressions).


    See Also
    ========

    solveset_real: solver for real domain
    solveset_complex: solver for complex domain

    Examples
    ========

    >>> from sympy import exp, sin, Symbol, pprint, S, Eq
    >>> from sympy.solvers.solveset import solveset, solveset_real

    * The default domain is complex. Not specifying a domain will lead
      to the solving of the equation in the complex domain (and this
      is not affected by the assumptions on the symbol):

    >>> x = Symbol('x')
    >>> pprint(solveset(exp(x) - 1, x), use_unicode=False)
    {2*n*I*pi | n in Integers}

    >>> x = Symbol('x', real=True)
    >>> pprint(solveset(exp(x) - 1, x), use_unicode=False)
    {2*n*I*pi | n in Integers}

    * If you want to use ``solveset`` to solve the equation in the
      real domain, provide a real domain. (Using ``solveset_real``
      does this automatically.)

    >>> R = S.Reals
    >>> x = Symbol('x')
    >>> solveset(exp(x) - 1, x, R)
    {0}
    >>> solveset_real(exp(x) - 1, x)
    {0}

    The solution is unaffected by assumptions on the symbol:

    >>> p = Symbol('p', positive=True)
    >>> pprint(solveset(p**2 - 4))
    {-2, 2}

    When a :class:`~.ConditionSet` is returned, symbols with assumptions that
    would alter the set are replaced with more generic symbols:

    >>> i = Symbol('i', imaginary=True)
    >>> solveset(Eq(i**2 + i*sin(i), 1), i, domain=S.Reals)
    ConditionSet(_R, Eq(_R**2 + _R*sin(_R) - 1, 0), Reals)

    * Inequalities can be solved over the real domain only. Use of a complex
      domain leads to a NotImplementedError.

    >>> solveset(exp(x) > 1, x, R)
    Interval.open(0, oo)

    """
    f = sympify(f)
    symbol = sympify(symbol)

    if f is S.true:
        return domain

    if f is S.false:
        return S.EmptySet

    if not isinstance(f, (Expr, Relational, Number)):
        raise ValueError("%s is not a valid SymPy expression" % f)

    if not isinstance(symbol, (Expr, Relational)) and  symbol is not None:
        raise ValueError("%s is not a valid SymPy symbol" % (symbol,))

    if not isinstance(domain, Set):
        raise ValueError("%s is not a valid domain" %(domain))

    free_symbols = f.free_symbols

    if f.has(Piecewise):
        f = piecewise_fold(f)

    if symbol is None and not free_symbols:
        b = Eq(f, 0)
        if b is S.true:
            return domain
        elif b is S.false:
            return S.EmptySet
        else:
            raise NotImplementedError(filldedent('''
                relationship between value and 0 is unknown: %s''' % b))

    if symbol is None:
        if len(free_symbols) == 1:
            symbol = free_symbols.pop()
        elif free_symbols:
            raise ValueError(filldedent('''
                The independent variable must be specified for a
                multivariate equation.'''))
    elif not isinstance(symbol, Symbol):
        f, s, swap = recast_to_symbols([f], [symbol])
        # the xreplace will be needed if a ConditionSet is returned
        return solveset(f[0], s[0], domain).xreplace(swap)

    # solveset should ignore assumptions on symbols
    newsym = None
    if domain.is_subset(S.Reals):
        if symbol._assumptions_orig != {'real': True}:
            newsym = Dummy('R', real=True)
    elif domain.is_subset(S.Complexes):
        if symbol._assumptions_orig != {'complex': True}:
            newsym = Dummy('C', complex=True)

    if newsym is not None:
        rv = solveset(f.xreplace({symbol: newsym}), newsym, domain)
        # try to use the original symbol if possible
        try:
            _rv = rv.xreplace({newsym: symbol})
        except TypeError:
            _rv = rv
        if rv.dummy_eq(_rv):
            rv = _rv
        return rv

    # Abs has its own handling method which avoids the
    # rewriting property that the first piece of abs(x)
    # is for x >= 0 and the 2nd piece for x < 0 -- solutions
    # can look better if the 2nd condition is x <= 0. Since
    # the solution is a set, duplication of results is not
    # an issue, e.g. {y, -y} when y is 0 will be {0}
    f, mask = _masked(f, Abs)
    f = f.rewrite(Piecewise) # everything that's not an Abs
    for d, e in mask:
        # everything *in* an Abs
        e = e.func(e.args[0].rewrite(Piecewise))
        f = f.xreplace({d: e})
    f = piecewise_fold(f)

    return _solveset(f, symbol, domain, _check=True)


def solveset_real(f, symbol):
    return solveset(f, symbol, S.Reals)


def solveset_complex(f, symbol):
    return solveset(f, symbol, S.Complexes)


def _solveset_multi(eqs, syms, domains):
    '''Basic implementation of a multivariate solveset.

    For internal use (not ready for public consumption)'''

    rep = {}
    for sym, dom in zip(syms, domains):
        if dom is S.Reals:
            rep[sym] = Symbol(sym.name, real=True)
    eqs = [eq.subs(rep) for eq in eqs]
    syms = [sym.subs(rep) for sym in syms]

    syms = tuple(syms)

    if len(eqs) == 0:
        return ProductSet(*domains)

    if len(syms) == 1:
        sym = syms[0]
        domain = domains[0]
        solsets = [solveset(eq, sym, domain) for eq in eqs]
        solset = Intersection(*solsets)
        return ImageSet(Lambda((sym,), (sym,)), solset).doit()

    eqs = sorted(eqs, key=lambda eq: len(eq.free_symbols & set(syms)))

    for n, eq in enumerate(eqs):
        sols = []
        all_handled = True
        for sym in syms:
            if sym not in eq.free_symbols:
                continue
            sol = solveset(eq, sym, domains[syms.index(sym)])

            if isinstance(sol, FiniteSet):
                i = syms.index(sym)
                symsp = syms[:i] + syms[i+1:]
                domainsp = domains[:i] + domains[i+1:]
                eqsp = eqs[:n] + eqs[n+1:]
                for s in sol:
                    eqsp_sub = [eq.subs(sym, s) for eq in eqsp]
                    sol_others = _solveset_multi(eqsp_sub, symsp, domainsp)
                    fun = Lambda((symsp,), symsp[:i] + (s,) + symsp[i:])
                    sols.append(ImageSet(fun, sol_others).doit())
            else:
                all_handled = False
        if all_handled:
            return Union(*sols)


def solvify(f, symbol, domain):
    """Solves an equation using solveset and returns the solution in accordance
    with the `solve` output API.

    Returns
    =======

    We classify the output based on the type of solution returned by `solveset`.

    Solution    |    Output
    ----------------------------------------
    FiniteSet   | list

    ImageSet,   | list (if `f` is periodic)
    Union       |

    Union       | list (with FiniteSet)

    EmptySet    | empty list

    Others      | None


    Raises
    ======

    NotImplementedError
        A ConditionSet is the input.

    Examples
    ========

    >>> from sympy.solvers.solveset import solvify
    >>> from sympy.abc import x
    >>> from sympy import S, tan, sin, exp
    >>> solvify(x**2 - 9, x, S.Reals)
    [-3, 3]
    >>> solvify(sin(x) - 1, x, S.Reals)
    [pi/2]
    >>> solvify(tan(x), x, S.Reals)
    [0]
    >>> solvify(exp(x) - 1, x, S.Complexes)

    >>> solvify(exp(x) - 1, x, S.Reals)
    [0]

    """
    solution_set = solveset(f, symbol, domain)
    result = None
    if solution_set is S.EmptySet:
        result = []

    elif isinstance(solution_set, ConditionSet):
        raise NotImplementedError('solveset is unable to solve this equation.')

    elif isinstance(solution_set, FiniteSet):
        result = list(solution_set)

    else:
        period = periodicity(f, symbol)
        if period is not None:
            solutions = S.EmptySet
            iter_solutions = ()
            if isinstance(solution_set, ImageSet):
                iter_solutions = (solution_set,)
            elif isinstance(solution_set, Union):
                if all(isinstance(i, ImageSet) for i in solution_set.args):
                    iter_solutions = solution_set.args

            for solution in iter_solutions:
                solutions += solution.intersect(Interval(0, period, False, True))

            if isinstance(solutions, FiniteSet):
                result = list(solutions)

        else:
            solution = solution_set.intersect(domain)
            if isinstance(solution, Union):
                # concerned about only FiniteSet with Union but not about ImageSet
                # if required could be extend
                if any(isinstance(i, FiniteSet) for i in solution.args):
                    result = [sol for soln in solution.args \
                     for sol in soln.args if isinstance(soln,FiniteSet)]
                else:
                    return None

            elif isinstance(solution, FiniteSet):
                result += solution

    return result


###############################################################################
################################ LINSOLVE #####################################
###############################################################################


def linear_coeffs(eq, *syms, dict=False):
    """Return a list whose elements are the coefficients of the
    corresponding symbols in the sum of terms in  ``eq``.
    The additive constant is returned as the last element of the
    list.

    Raises
    ======

    NonlinearError
        The equation contains a nonlinear term
    ValueError
        duplicate or unordered symbols are passed

    Parameters
    ==========

    dict - (default False) when True, return coefficients as a
        dictionary with coefficients keyed to syms that were present;
        key 1 gives the constant term

    Examples
    ========

    >>> from sympy.solvers.solveset import linear_coeffs
    >>> from sympy.abc import x, y, z
    >>> linear_coeffs(3*x + 2*y - 1, x, y)
    [3, 2, -1]

    It is not necessary to expand the expression:

        >>> linear_coeffs(x + y*(z*(x*3 + 2) + 3), x)
        [3*y*z + 1, y*(2*z + 3)]

    When nonlinear is detected, an error will be raised:

        * even if they would cancel after expansion (so the
        situation does not pass silently past the caller's
        attention)

        >>> eq = 1/x*(x - 1) + 1/x
        >>> linear_coeffs(eq.expand(), x)
        [0, 1]
        >>> linear_coeffs(eq, x)
        Traceback (most recent call last):
        ...
        NonlinearError:
        nonlinear in given generators

        * when there are cross terms

        >>> linear_coeffs(x*(y + 1), x, y)
        Traceback (most recent call last):
        ...
        NonlinearError:
        symbol-dependent cross-terms encountered

        * when there are terms that contain an expression
        dependent on the symbols that is not linear

        >>> linear_coeffs(x**2, x)
        Traceback (most recent call last):
        ...
        NonlinearError:
        nonlinear in given generators
    """
    eq = _sympify(eq)
    if len(syms) == 1 and iterable(syms[0]) and not isinstance(syms[0], Basic):
        raise ValueError('expecting unpacked symbols, *syms')
    symset = set(syms)
    if len(symset) != len(syms):
        raise ValueError('duplicate symbols given')
    try:
        d, c = _linear_eq_to_dict([eq], symset)
        d = d[0]
        c = c[0]
    except PolyNonlinearError as err:
        raise NonlinearError(str(err))
    if dict:
        if c:
            d[S.One] = c
        return d
    rv = [S.Zero]*(len(syms) + 1)
    rv[-1] = c
    for i, k in enumerate(syms):
        if k not in d:
            continue
        rv[i] = d[k]
    return rv


def linear_eq_to_matrix(equations, *symbols):
    r"""
    Converts a given System of Equations into Matrix form. Here ``equations``
    must be a linear system of equations in ``symbols``. Element ``M[i, j]``
    corresponds to the coefficient of the jth symbol in the ith equation.

    The Matrix form corresponds to the augmented matrix form. For example:

    .. math::

       4x + 2y + 3z & = 1 \\
       3x +  y +  z & = -6 \\
       2x + 4y + 9z & = 2

    This system will return :math:`A` and :math:`b` as:

    .. math::

       A = \left[\begin{array}{ccc}
       4 & 2 & 3 \\
       3 & 1 & 1 \\
       2 & 4 & 9
       \end{array}\right] \\

    .. math::

       b = \left[\begin{array}{c}
       1 \\ -6 \\ 2
       \end{array}\right]

    The only simplification performed is to convert
    ``Eq(a, b)`` :math:`\Rightarrow a - b`.

    Raises
    ======

    NonlinearError
        The equations contain a nonlinear term.
    ValueError
        The symbols are not given or are not unique.

    Examples
    ========

    >>> from sympy import linear_eq_to_matrix, symbols
    >>> c, x, y, z = symbols('c, x, y, z')

    The coefficients (numerical or symbolic) of the symbols will
    be returned as matrices:

    >>> eqns = [c*x + z - 1 - c, y + z, x - y]
    >>> A, b = linear_eq_to_matrix(eqns, [x, y, z])
    >>> A
    Matrix([
    [c,  0, 1],
    [0,  1, 1],
    [1, -1, 0]])
    >>> b
    Matrix([
    [c + 1],
    [    0],
    [    0]])

    This routine does not simplify expressions and will raise an error
    if nonlinearity is encountered:

    >>> eqns = [
    ...     (x**2 - 3*x)/(x - 3) - 3,
    ...     y**2 - 3*y - y*(y - 4) + x - 4]
    >>> linear_eq_to_matrix(eqns, [x, y])
    Traceback (most recent call last):
    ...
    NonlinearError:
    symbol-dependent term can be ignored using `strict=False`

    Simplifying these equations will discard the removable singularity in the
    first and reveal the linear structure of the second:

    >>> [e.simplify() for e in eqns]
    [x - 3, x + y - 4]

    Any such simplification needed to eliminate nonlinear terms must be done
    *before* calling this routine.

    """
    if not symbols:
        raise ValueError(filldedent('''
            Symbols must be given, for which coefficients
            are to be found.
            '''))

    # Check if 'symbols' is a set and raise an error if it is
    if isinstance(symbols[0], set):
        raise TypeError(
            "Unordered 'set' type is not supported as input for symbols.")

    if hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]

    if has_dups(symbols):
        raise ValueError('Symbols must be unique')

    equations = sympify(equations)
    if isinstance(equations, MatrixBase):
        equations = list(equations)
    elif isinstance(equations, (Expr, Eq)):
        equations = [equations]
    elif not is_sequence(equations):
        raise ValueError(filldedent('''
            Equation(s) must be given as a sequence, Expr,
            Eq or Matrix.
            '''))

    # construct the dictionaries
    try:
        eq, c = _linear_eq_to_dict(equations, symbols)
    except PolyNonlinearError as err:
        raise NonlinearError(str(err))
    # prepare output matrices
    n, m = shape = len(eq), len(symbols)
    ix = dict(zip(symbols, range(m)))
    A = zeros(*shape)
    for row, d in enumerate(eq):
        for k in d:
            col = ix[k]
            A[row, col] = d[k]
    b = Matrix(n, 1, [-i for i in c])
    return A, b


def linsolve(system, *symbols):
    r"""
    Solve system of $N$ linear equations with $M$ variables; both
    underdetermined and overdetermined systems are supported.
    The possible number of solutions is zero, one or infinite.
    Zero solutions throws a ValueError, whereas infinite
    solutions are represented parametrically in terms of the given
    symbols. For unique solution a :class:`~.FiniteSet` of ordered tuples
    is returned.

    All standard input formats are supported:
    For the given set of equations, the respective input types
    are given below:

    .. math:: 3x + 2y -   z = 1
    .. math:: 2x - 2y + 4z = -2
    .. math:: 2x -   y + 2z = 0

    * Augmented matrix form, ``system`` given below:

    $$ \text{system} = \left[{array}{cccc}
        3 &  2 & -1 &  1\\
        2 & -2 &  4 & -2\\
        2 & -1 &  2 &  0
        \end{array}\right] $$

    ::

        system = Matrix([[3, 2, -1, 1], [2, -2, 4, -2], [2, -1, 2, 0]])

    * List of equations form

    ::

        system  =  [3x + 2y - z - 1, 2x - 2y + 4z + 2, 2x - y + 2z]

    * Input $A$ and $b$ in matrix form (from $Ax = b$) are given as:

    $$ A = \left[\begin{array}{ccc}
        3 &  2 & -1 \\
        2 & -2 &  4 \\
        2 & -1 &  2
        \end{array}\right] \ \  b = \left[\begin{array}{c}
        1 \\ -2 \\ 0
        \end{array}\right] $$

    ::

        A = Matrix([[3, 2, -1], [2, -2, 4], [2, -1, 2]])
        b = Matrix([[1], [-2], [0]])
        system = (A, b)

    Symbols can always be passed but are actually only needed
    when 1) a system of equations is being passed and 2) the
    system is passed as an underdetermined matrix and one wants
    to control the name of the free variables in the result.
    An error is raised if no symbols are used for case 1, but if
    no symbols are provided for case 2, internally generated symbols
    will be provided. When providing symbols for case 2, there should
    be at least as many symbols are there are columns in matrix A.

    The algorithm used here is Gauss-Jordan elimination, which
    results, after elimination, in a row echelon form matrix.

    Returns
    =======

    A FiniteSet containing an ordered tuple of values for the
    unknowns for which the `system` has a solution. (Wrapping
    the tuple in FiniteSet is used to maintain a consistent
    output format throughout solveset.)

    Returns EmptySet, if the linear system is inconsistent.

    Raises
    ======

    ValueError
        The input is not valid.
        The symbols are not given.

    Examples
    ========

    >>> from sympy import Matrix, linsolve, symbols
    >>> x, y, z = symbols("x, y, z")
    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    >>> b = Matrix([3, 6, 9])
    >>> A
    Matrix([
    [1, 2,  3],
    [4, 5,  6],
    [7, 8, 10]])
    >>> b
    Matrix([
    [3],
    [6],
    [9]])
    >>> linsolve((A, b), [x, y, z])
    {(-1, 2, 0)}

    * Parametric Solution: In case the system is underdetermined, the
      function will return a parametric solution in terms of the given
      symbols. Those that are free will be returned unchanged. e.g. in
      the system below, `z` is returned as the solution for variable z;
      it can take on any value.

    >>> A = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> b = Matrix([3, 6, 9])
    >>> linsolve((A, b), x, y, z)
    {(z - 1, 2 - 2*z, z)}

    If no symbols are given, internally generated symbols will be used.
    The ``tau0`` in the third position indicates (as before) that the third
    variable -- whatever it is named -- can take on any value:

    >>> linsolve((A, b))
    {(tau0 - 1, 2 - 2*tau0, tau0)}

    * List of equations as input

    >>> Eqns = [3*x + 2*y - z - 1, 2*x - 2*y + 4*z + 2, - x + y/2 - z]
    >>> linsolve(Eqns, x, y, z)
    {(1, -2, -2)}

    * Augmented matrix as input

    >>> aug = Matrix([[2, 1, 3, 1], [2, 6, 8, 3], [6, 8, 18, 5]])
    >>> aug
    Matrix([
    [2, 1,  3, 1],
    [2, 6,  8, 3],
    [6, 8, 18, 5]])
    >>> linsolve(aug, x, y, z)
    {(3/10, 2/5, 0)}

    * Solve for symbolic coefficients

    >>> a, b, c, d, e, f = symbols('a, b, c, d, e, f')
    >>> eqns = [a*x + b*y - c, d*x + e*y - f]
    >>> linsolve(eqns, x, y)
    {((-b*f + c*e)/(a*e - b*d), (a*f - c*d)/(a*e - b*d))}

    * A degenerate system returns solution as set of given
      symbols.

    >>> system = Matrix(([0, 0, 0], [0, 0, 0], [0, 0, 0]))
    >>> linsolve(system, x, y)
    {(x, y)}

    * For an empty system linsolve returns empty set

    >>> linsolve([], x)
    EmptySet

    * An error is raised if any nonlinearity is detected, even
      if it could be removed with expansion

    >>> linsolve([x*(1/x - 1)], x)
    Traceback (most recent call last):
    ...
    NonlinearError: nonlinear term: 1/x

    >>> linsolve([x*(y + 1)], x, y)
    Traceback (most recent call last):
    ...
    NonlinearError: nonlinear cross-term: x*(y + 1)

    >>> linsolve([x**2 - 1], x)
    Traceback (most recent call last):
    ...
    NonlinearError: nonlinear term: x**2
    """
    if not system:
        return S.EmptySet

    # If second argument is an iterable
    if symbols and hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]
    sym_gen = isinstance(symbols, GeneratorType)
    dup_msg = 'duplicate symbols given'


    b = None  # if we don't get b the input was bad
    # unpack system

    if hasattr(system, '__iter__'):

        # 1). (A, b)
        if len(system) == 2 and isinstance(system[0], MatrixBase):
            A, b = system

        # 2). (eq1, eq2, ...)
        if not isinstance(system[0], MatrixBase):
            if sym_gen or not symbols:
                raise ValueError(filldedent('''
                    When passing a system of equations, the explicit
                    symbols for which a solution is being sought must
                    be given as a sequence, too.
                '''))
            if len(set(symbols)) != len(symbols):
                raise ValueError(dup_msg)

            #
            # Pass to the sparse solver implemented in polys. It is important
            # that we do not attempt to convert the equations to a matrix
            # because that would be very inefficient for large sparse systems
            # of equations.
            #
            eqs = system
            eqs = [sympify(eq) for eq in eqs]
            try:
                sol = _linsolve(eqs, symbols)
            except PolyNonlinearError as exc:
                # e.g. cos(x) contains an element of the set of generators
                raise NonlinearError(str(exc))

            if sol is None:
                return S.EmptySet

            sol = FiniteSet(Tuple(*(sol.get(sym, sym) for sym in symbols)))
            return sol

    elif isinstance(system, MatrixBase) and not (
            symbols and not isinstance(symbols, GeneratorType) and
            isinstance(symbols[0], MatrixBase)):
        # 3). A augmented with b
        A, b = system[:, :-1], system[:, -1:]

    if b is None:
        raise ValueError("Invalid arguments")
    if sym_gen:
        symbols = [next(symbols) for i in range(A.cols)]
        symset = set(symbols)
        if any(symset & (A.free_symbols | b.free_symbols)):
            raise ValueError(filldedent('''
                At least one of the symbols provided
                already appears in the system to be solved.
                One way to avoid this is to use Dummy symbols in
                the generator, e.g. numbered_symbols('%s', cls=Dummy)
            ''' % symbols[0].name.rstrip('1234567890')))
        elif len(symset) != len(symbols):
            raise ValueError(dup_msg)

    if not symbols:
        symbols = [Dummy() for _ in range(A.cols)]
        name = _uniquely_named_symbol('tau', (A, b),
            compare=lambda i: str(i).rstrip('1234567890')).name
        gen  = numbered_symbols(name)
    else:
        gen = None

    # This is just a wrapper for solve_lin_sys
    eqs = []
    rows = A.tolist()
    for rowi, bi in zip(rows, b):
        terms = [elem * sym for elem, sym in zip(rowi, symbols) if elem]
        terms.append(-bi)
        eqs.append(Add(*terms))

    eqs, ring = sympy_eqs_to_ring(eqs, symbols)
    sol = solve_lin_sys(eqs, ring, _raw=False)
    if sol is None:
        return S.EmptySet
    #sol = {sym:val for sym, val in sol.items() if sym != val}
    sol = FiniteSet(Tuple(*(sol.get(sym, sym) for sym in symbols)))

    if gen is not None:
        solsym = sol.free_symbols
        rep = {sym: next(gen) for sym in symbols if sym in solsym}
        sol = sol.subs(rep)

    return sol


##############################################################################
# ------------------------------nonlinsolve ---------------------------------#
##############################################################################


def _return_conditionset(eqs, symbols):
    # return conditionset
    eqs = (Eq(lhs, 0) for lhs in eqs)
    condition_set = ConditionSet(
        Tuple(*symbols), And(*eqs), S.Complexes**len(symbols))
    return condition_set


def substitution(system, symbols, result=[{}], known_symbols=[],
                 exclude=[], all_symbols=None):
    r"""
    Solves the `system` using substitution method. It is used in
    :func:`~.nonlinsolve`. This will be called from :func:`~.nonlinsolve` when any
    equation(s) is non polynomial equation.

    Parameters
    ==========

    system : list of equations
        The target system of equations
    symbols : list of symbols to be solved.
        The variable(s) for which the system is solved
    known_symbols : list of solved symbols
        Values are known for these variable(s)
    result : An empty list or list of dict
        If No symbol values is known then empty list otherwise
        symbol as keys and corresponding value in dict.
    exclude : Set of expression.
        Mostly denominator expression(s) of the equations of the system.
        Final solution should not satisfy these expressions.
    all_symbols : known_symbols + symbols(unsolved).

    Returns
    =======

    A FiniteSet of ordered tuple of values of `all_symbols` for which the
    `system` has solution. Order of values in the tuple is same as symbols
    present in the parameter `all_symbols`. If parameter `all_symbols` is None
    then same as symbols present in the parameter `symbols`.

    Please note that general FiniteSet is unordered, the solution returned
    here is not simply a FiniteSet of solutions, rather it is a FiniteSet of
    ordered tuple, i.e. the first & only argument to FiniteSet is a tuple of
    solutions, which is ordered, & hence the returned solution is ordered.

    Also note that solution could also have been returned as an ordered tuple,
    FiniteSet is just a wrapper `{}` around the tuple. It has no other
    significance except for the fact it is just used to maintain a consistent
    output format throughout the solveset.

    Raises
    ======

    ValueError
        The input is not valid.
        The symbols are not given.
    AttributeError
        The input symbols are not :class:`~.Symbol` type.

    Examples
    ========

    >>> from sympy import symbols, substitution
    >>> x, y = symbols('x, y', real=True)
    >>> substitution([x + y], [x], [{y: 1}], [y], set([]), [x, y])
    {(-1, 1)}

    * When you want a soln not satisfying $x + 1 = 0$

    >>> substitution([x + y], [x], [{y: 1}], [y], set([x + 1]), [y, x])
    EmptySet
    >>> substitution([x + y], [x], [{y: 1}], [y], set([x - 1]), [y, x])
    {(1, -1)}
    >>> substitution([x + y - 1, y - x**2 + 5], [x, y])
    {(-3, 4), (2, -1)}

    * Returns both real and complex solution

    >>> x, y, z = symbols('x, y, z')
    >>> from sympy import exp, sin
    >>> substitution([exp(x) - sin(y), y**2 - 4], [x, y])
    {(ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2),
     (ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2)}

    >>> eqs = [z**2 + exp(2*x) - sin(y), -3 + exp(-y)]
    >>> substitution(eqs, [y, z])
    {(-log(3), -sqrt(-exp(2*x) - sin(log(3)))),
     (-log(3), sqrt(-exp(2*x) - sin(log(3)))),
     (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
      ImageSet(Lambda(_n, -sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers)),
     (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
      ImageSet(Lambda(_n, sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers))}

    """

    if not system:
        return S.EmptySet

    for i, e in enumerate(system):
        if isinstance(e, Eq):
            system[i] = e.lhs - e.rhs

    if not symbols:
        msg = ('Symbols must be given, for which solution of the '
               'system is to be found.')
        raise ValueError(filldedent(msg))

    if not is_sequence(symbols):
        msg = ('symbols should be given as a sequence, e.g. a list.'
               'Not type %s: %s')
        raise TypeError(filldedent(msg % (type(symbols), symbols)))

    if not getattr(symbols[0], 'is_Symbol', False):
        msg = ('Iterable of symbols must be given as '
               'second argument, not type %s: %s')
        raise ValueError(filldedent(msg % (type(symbols[0]), symbols[0])))

    # By default `all_symbols` will be same as `symbols`
    if all_symbols is None:
        all_symbols = symbols

    old_result = result
    # storing complements and intersection for particular symbol
    complements = {}
    intersections = {}

    # when total_solveset_call equals total_conditionset
    # it means that solveset failed to solve all eqs.
    total_conditionset = -1
    total_solveset_call = -1

    def _unsolved_syms(eq, sort=False):
        """Returns the unsolved symbol present
        in the equation `eq`.
        """
        free = eq.free_symbols
        unsolved = (free - set(known_symbols)) & set(all_symbols)
        if sort:
            unsolved = list(unsolved)
            unsolved.sort(key=default_sort_key)
        return unsolved

    # sort such that equation with the fewest potential symbols is first.
    # means eq with less number of variable first in the list.
    eqs_in_better_order = list(
        ordered(system, lambda _: len(_unsolved_syms(_))))

    def add_intersection_complement(result, intersection_dict, complement_dict):
        # If solveset has returned some intersection/complement
        # for any symbol, it will be added in the final solution.
        final_result = []
        for res in result:
            res_copy = res
            for key_res, value_res in res.items():
                intersect_set, complement_set = None, None
                for key_sym, value_sym in intersection_dict.items():
                    if key_sym == key_res:
                        intersect_set = value_sym
                for key_sym, value_sym in complement_dict.items():
                    if key_sym == key_res:
                        complement_set = value_sym
                if intersect_set or complement_set:
                    new_value = FiniteSet(value_res)
                    if intersect_set and intersect_set != S.Complexes:
                        new_value = Intersection(new_value, intersect_set)
                    if complement_set:
                        new_value = Complement(new_value, complement_set)
                    if new_value is S.EmptySet:
                        res_copy = None
                        break
                    elif new_value.is_FiniteSet and len(new_value) == 1:
                        res_copy[key_res] = set(new_value).pop()
                    else:
                        res_copy[key_res] = new_value

            if res_copy is not None:
                final_result.append(res_copy)
        return final_result

    def _extract_main_soln(sym, sol, soln_imageset):
        """Separate the Complements, Intersections, ImageSet lambda expr and
        its base_set. This function returns the unmasked sol from different classes
        of sets and also returns the appended ImageSet elements in a
        soln_imageset dict: `{unmasked element: ImageSet}`.
        """
        # if there is union, then need to check
        # Complement, Intersection, Imageset.
        # Order should not be changed.
        if isinstance(sol, ConditionSet):
            # extracts any solution in ConditionSet
            sol = sol.base_set

        if isinstance(sol, Complement):
            # extract solution and complement
            complements[sym] = sol.args[1]
            sol = sol.args[0]
            # complement will be added at the end
            # using `add_intersection_complement` method

        # if there is union of Imageset or other in soln.
        # no testcase is written for this if block
        if isinstance(sol, Union):
            sol_args = sol.args
            sol = S.EmptySet
            # We need in sequence so append finteset elements
            # and then imageset or other.
            for sol_arg2 in sol_args:
                if isinstance(sol_arg2, FiniteSet):
                    sol += sol_arg2
                else:
                    # ImageSet, Intersection, complement then
                    # append them directly
                    sol += FiniteSet(sol_arg2)

        if isinstance(sol, Intersection):
            # Interval/Set will be at 0th index always
            if sol.args[0] not in (S.Reals, S.Complexes):
                # Sometimes solveset returns soln with intersection
                # S.Reals or S.Complexes. We don't consider that
                # intersection.
                intersections[sym] = sol.args[0]
            sol = sol.args[1]
        # after intersection and complement Imageset should
        # be checked.
        if isinstance(sol, ImageSet):
            soln_imagest = sol
            expr2 = sol.lamda.expr
            sol = FiniteSet(expr2)
            soln_imageset[expr2] = soln_imagest

        if not isinstance(sol, FiniteSet):
            sol = FiniteSet(sol)
        return sol, soln_imageset

    def _check_exclude(rnew, imgset_yes):
        rnew_ = rnew
        if imgset_yes:
            # replace all dummy variables (Imageset lambda variables)
            # with zero before `checksol`. Considering fundamental soln
            # for `checksol`.
            rnew_copy = rnew.copy()
            dummy_n = imgset_yes[0]
            for key_res, value_res in rnew_copy.items():
                rnew_copy[key_res] = value_res.subs(dummy_n, 0)
            rnew_ = rnew_copy
        # satisfy_exclude == true if it satisfies the expr of `exclude` list.
        try:
            # something like : `Mod(-log(3), 2*I*pi)` can't be
            # simplified right now, so `checksol` returns `TypeError`.
            # when this issue is fixed this try block should be
            # removed. Mod(-log(3), 2*I*pi) == -log(3)
            satisfy_exclude = any(
                checksol(d, rnew_) for d in exclude)
        except TypeError:
            satisfy_exclude = None
        return satisfy_exclude

    def _restore_imgset(rnew, original_imageset, newresult):
        restore_sym = set(rnew.keys()) & \
            set(original_imageset.keys())
        for key_sym in restore_sym:
            img = original_imageset[key_sym]
            rnew[key_sym] = img
        if rnew not in newresult:
            newresult.append(rnew)

    def _append_eq(eq, result, res, delete_soln, n=None):
        u = Dummy('u')
        if n:
            eq = eq.subs(n, 0)
        satisfy = eq if eq in (True, False) else checksol(u, u, eq, minimal=True)
        if satisfy is False:
            delete_soln = True
            res = {}
        else:
            result.append(res)
        return result, res, delete_soln

    def _append_new_soln(rnew, sym, sol, imgset_yes, soln_imageset,
                         original_imageset, newresult, eq=None):
        """If `rnew` (A dict <symbol: soln>) contains valid soln
        append it to `newresult` list.
        `imgset_yes` is (base, dummy_var) if there was imageset in previously
         calculated result(otherwise empty tuple). `original_imageset` is dict
         of imageset expr and imageset from this result.
        `soln_imageset` dict of imageset expr and imageset of new soln.
        """
        satisfy_exclude = _check_exclude(rnew, imgset_yes)
        delete_soln = False
        # soln should not satisfy expr present in `exclude` list.
        if not satisfy_exclude:
            local_n = None
            # if it is imageset
            if imgset_yes:
                local_n = imgset_yes[0]
                base = imgset_yes[1]
                if sym and sol:
                    # when `sym` and `sol` is `None` means no new
                    # soln. In that case we will append rnew directly after
                    # substituting original imagesets in rnew values if present
                    # (second last line of this function using _restore_imgset)
                    dummy_list = list(sol.atoms(Dummy))
                    # use one dummy `n` which is in
                    # previous imageset
                    local_n_list = [
                        local_n for i in range(
                            0, len(dummy_list))]

                    dummy_zip = zip(dummy_list, local_n_list)
                    lam = Lambda(local_n, sol.subs(dummy_zip))
                    rnew[sym] = ImageSet(lam, base)
                if eq is not None:
                    newresult, rnew, delete_soln = _append_eq(
                        eq, newresult, rnew, delete_soln, local_n)
            elif eq is not None:
                newresult, rnew, delete_soln = _append_eq(
                    eq, newresult, rnew, delete_soln)
            elif sol in soln_imageset.keys():
                rnew[sym] = soln_imageset[sol]
                # restore original imageset
                _restore_imgset(rnew, original_imageset, newresult)
            else:
                newresult.append(rnew)
        elif satisfy_exclude:
            delete_soln = True
            rnew = {}
        _restore_imgset(rnew, original_imageset, newresult)
        return newresult, delete_soln

    def _new_order_result(result, eq):
        # separate first, second priority. `res` that makes `eq` value equals
        # to zero, should be used first then other result(second priority).
        # If it is not done then we may miss some soln.
        first_priority = []
        second_priority = []
        for res in result:
            if not any(isinstance(val, ImageSet) for val in res.values()):
                if eq.subs(res) == 0:
                    first_priority.append(res)
                else:
                    second_priority.append(res)
        if first_priority or second_priority:
            return first_priority + second_priority
        return result

    def _solve_using_known_values(result, solver):
        """Solves the system using already known solution
        (result contains the dict <symbol: value>).
        solver is :func:`~.solveset_complex` or :func:`~.solveset_real`.
        """
        # stores imageset <expr: imageset(Lambda(n, expr), base)>.
        soln_imageset = {}
        total_solvest_call = 0
        total_conditionst = 0

        # sort equations so the one with the fewest potential
        # symbols appears first
        for index, eq in enumerate(eqs_in_better_order):
            newresult = []
            # if imageset, expr is used to solve for other symbol
            imgset_yes = False
            for res in result:
                original_imageset = {}
                got_symbol = set()  # symbols solved in one iteration
                # find the imageset and use its expr.
                for k, v in res.items():
                    if isinstance(v, ImageSet):
                        res[k] = v.lamda.expr
                        original_imageset[k] = v
                        dummy_n = v.lamda.expr.atoms(Dummy).pop()
                        (base,) = v.base_sets
                        imgset_yes = (dummy_n, base)
                    assert not isinstance(v, FiniteSet)  # if so, internal error
                # update eq with everything that is known so far
                eq2 = eq.subs(res).expand()
                if imgset_yes and not eq2.has(imgset_yes[0]):
                    # The substituted equation simplified in such a way that
                    # it's no longer necessary to encapsulate a potential new
                    # solution in an ImageSet. (E.g. at the previous step some
                    # {n*2*pi} was found as partial solution for one of the
                    # unknowns, but its main solution expression n*2*pi has now
                    # been substituted in a trigonometric function.)
                    imgset_yes = False

                unsolved_syms = _unsolved_syms(eq2, sort=True)
                if not unsolved_syms:
                    if res:
                        newresult, delete_res = _append_new_soln(
                            res, None, None, imgset_yes, soln_imageset,
                            original_imageset, newresult, eq2)
                        if delete_res:
                            # `delete_res` is true, means substituting `res` in
                            # eq2 doesn't return `zero` or deleting the `res`
                            # (a soln) since it satisfies expr of `exclude`
                            # list.
                            result.remove(res)
                    continue  # skip as it's independent of desired symbols
                depen1, depen2 = eq2.as_independent(*unsolved_syms)
                if (depen1.has(Abs) or depen2.has(Abs)) and solver == solveset_complex:
                    # Absolute values cannot be inverted in the
                    # complex domain
                    continue
                soln_imageset = {}
                for sym in unsolved_syms:
                    not_solvable = False
                    try:
                        soln = solver(eq2, sym)
                        total_solvest_call += 1
                        soln_new = S.EmptySet
                        if isinstance(soln, Complement):
                            # separate solution and complement
                            complements[sym] = soln.args[1]
                            soln = soln.args[0]
                            # complement will be added at the end
                        if isinstance(soln, Intersection):
                            # Interval will be at 0th index always
                            if soln.args[0] != Interval(-oo, oo):
                                # sometimes solveset returns soln
                                # with intersection S.Reals, to confirm that
                                # soln is in domain=S.Reals
                                intersections[sym] = soln.args[0]
                            soln_new += soln.args[1]
                        soln = soln_new if soln_new else soln
                        if index > 0 and solver == solveset_real:
                            # one symbol's real soln, another symbol may have
                            # corresponding complex soln.
                            if not isinstance(soln, (ImageSet, ConditionSet)):
                                soln += solveset_complex(eq2, sym)  # might give ValueError with Abs
                    except (NotImplementedError, ValueError):
                        # If solveset is not able to solve equation `eq2`. Next
                        # time we may get soln using next equation `eq2`
                        continue
                    if isinstance(soln, ConditionSet):
                        if soln.base_set in (S.Reals, S.Complexes):
                            soln = S.EmptySet
                            # don't do `continue` we may get soln
                            # in terms of other symbol(s)
                            not_solvable = True
                            total_conditionst += 1
                        else:
                            soln = soln.base_set

                    if soln is not S.EmptySet:
                        soln, soln_imageset = _extract_main_soln(
                            sym, soln, soln_imageset)

                    for sol in soln:
                        # sol is not a `Union` since we checked it
                        # before this loop
                        sol, soln_imageset = _extract_main_soln(
                            sym, sol, soln_imageset)
                        sol = set(sol).pop()  # XXX what if there are more solutions?
                        free = sol.free_symbols
                        if got_symbol and any(
                            ss in free for ss in got_symbol
                        ):
                            # sol depends on previously solved symbols
                            # then continue
                            continue
                        rnew = res.copy()
                        # put each solution in res and append the new  result
                        # in the new result list (solution for symbol `s`)
                        # along with old results.
                        for k, v in res.items():
                            if isinstance(v, Expr) and isinstance(sol, Expr):
                                # if any unsolved symbol is present
                                # Then subs known value
                                rnew[k] = v.subs(sym, sol)
                        # and add this new solution
                        if sol in soln_imageset.keys():
                            # replace all lambda variables with 0.
                            imgst = soln_imageset[sol]
                            rnew[sym] = imgst.lamda(
                                *[0 for i in range(0, len(
                                    imgst.lamda.variables))])
                        else:
                            rnew[sym] = sol
                        newresult, delete_res = _append_new_soln(
                            rnew, sym, sol, imgset_yes, soln_imageset,
                            original_imageset, newresult)
                        if delete_res:
                            # deleting the `res` (a soln) since it satisfies
                            # eq of `exclude` list
                            result.remove(res)
                    # solution got for sym
                    if not not_solvable:
                        got_symbol.add(sym)
            # next time use this new soln
            if newresult:
                result = newresult
        return result, total_solvest_call, total_conditionst

    new_result_real, solve_call1, cnd_call1 = _solve_using_known_values(
        old_result, solveset_real)
    new_result_complex, solve_call2, cnd_call2 = _solve_using_known_values(
        old_result, solveset_complex)

    # If total_solveset_call is equal to total_conditionset
    # then solveset failed to solve all of the equations.
    # In this case we return a ConditionSet here.
    total_conditionset += (cnd_call1 + cnd_call2)
    total_solveset_call += (solve_call1 + solve_call2)

    if total_conditionset == total_solveset_call and total_solveset_call != -1:
        return _return_conditionset(eqs_in_better_order, all_symbols)

    # don't keep duplicate solutions
    filtered_complex = []
    for i in list(new_result_complex):
        for j in list(new_result_real):
            if i.keys() != j.keys():
                continue
            if all(a.dummy_eq(b) for a, b in zip(i.values(), j.values()) \
                if not (isinstance(a, int) and isinstance(b, int))):
                break
        else:
            filtered_complex.append(i)
    # overall result
    result = new_result_real + filtered_complex

    result_all_variables = []
    result_infinite = []
    for res in result:
        if not res:
            # means {None : None}
            continue
        # If length < len(all_symbols) means infinite soln.
        # Some or all the soln is dependent on 1 symbol.
        # eg. {x: y+2} then final soln {x: y+2, y: y}
        if len(res) < len(all_symbols):
            solved_symbols = res.keys()
            unsolved = list(filter(
                lambda x: x not in solved_symbols, all_symbols))
            for unsolved_sym in unsolved:
                res[unsolved_sym] = unsolved_sym
            result_infinite.append(res)
        if res not in result_all_variables:
            result_all_variables.append(res)

    if result_infinite:
        # we have general soln
        # eg : [{x: -1, y : 1}, {x : -y, y: y}] then
        # return [{x : -y, y : y}]
        result_all_variables = result_infinite
    if intersections or complements:
        result_all_variables = add_intersection_complement(
            result_all_variables, intersections, complements)

    # convert to ordered tuple
    result = S.EmptySet
    for r in result_all_variables:
        temp = [r[symb] for symb in all_symbols]
        result += FiniteSet(tuple(temp))
    return result


def _solveset_work(system, symbols):
    soln = solveset(system[0], symbols[0])
    if isinstance(soln, FiniteSet):
        _soln = FiniteSet(*[(s,) for s in soln])
        return _soln
    else:
        return FiniteSet(tuple(FiniteSet(soln)))


def _handle_positive_dimensional(polys, symbols, denominators):
    from sympy.polys.polytools import groebner
    # substitution method where new system is groebner basis of the system
    _symbols = list(symbols)
    _symbols.sort(key=default_sort_key)
    basis = groebner(polys, _symbols, polys=True)
    new_system = []
    for poly_eq in basis:
        new_system.append(poly_eq.as_expr())
    result = [{}]
    result = substitution(
        new_system, symbols, result, [],
        denominators)
    return result


def _handle_zero_dimensional(polys, symbols, system):
    # solve 0 dimensional poly system using `solve_poly_system`
    result = solve_poly_system(polys, *symbols)
    # May be some extra soln is added because
    # we used `unrad` in `_separate_poly_nonpoly`, so
    # need to check and remove if it is not a soln.
    result_update = S.EmptySet
    for res in result:
        dict_sym_value = dict(list(zip(symbols, res)))
        if all(checksol(eq, dict_sym_value) for eq in system):
            result_update += FiniteSet(res)
    return result_update


def _separate_poly_nonpoly(system, symbols):
    polys = []
    polys_expr = []
    nonpolys = []
    # unrad_changed stores a list of expressions containing
    # radicals that were processed using unrad
    # this is useful if solutions need to be checked later.
    unrad_changed = []
    denominators = set()
    poly = None
    for eq in system:
        # Store denom expressions that contain symbols
        denominators.update(_simple_dens(eq, symbols))
        # Convert equality to expression
        if isinstance(eq, Eq):
            eq = eq.lhs - eq.rhs
        # try to remove sqrt and rational power
        without_radicals = unrad(simplify(eq), *symbols)
        if without_radicals:
            unrad_changed.append(eq)
            eq_unrad, cov = without_radicals
            if not cov:
                eq = eq_unrad
        if isinstance(eq, Expr):
            eq = eq.as_numer_denom()[0]
            poly = eq.as_poly(*symbols, extension=True)
        elif simplify(eq).is_number:
            continue
        if poly is not None:
            polys.append(poly)
            polys_expr.append(poly.as_expr())
        else:
            nonpolys.append(eq)
    return polys, polys_expr, nonpolys, denominators, unrad_changed


def _handle_poly(polys, symbols):
    # _handle_poly(polys, symbols) -> (poly_sol, poly_eqs)
    #
    # We will return possible solution information to nonlinsolve as well as a
    # new system of polynomial equations to be solved if we cannot solve
    # everything directly here. The new system of polynomial equations will be
    # a lex-order Groebner basis for the original system. The lex basis
    # hopefully separate some of the variables and equations and give something
    # easier for substitution to work with.

    # The format for representing solution sets in nonlinsolve and substitution
    # is a list of dicts. These are the special cases:
    no_information = [{}]   # No equations solved yet
    no_solutions = []       # The system is inconsistent and has no solutions.

    # If there is no need to attempt further solution of these equations then
    # we return no equations:
    no_equations = []

    inexact = any(not p.domain.is_Exact for p in polys)
    if inexact:
        # The use of Groebner over RR is likely to result incorrectly in an
        # inconsistent Groebner basis. So, convert any float coefficients to
        # Rational before computing the Groebner basis.
        polys = [poly(nsimplify(p, rational=True)) for p in polys]

    # Compute a Groebner basis in grevlex order wrt the ordering given. We will
    # try to convert this to lex order later. Usually it seems to be more
    # efficient to compute a lex order basis by computing a grevlex basis and
    # converting to lex with fglm.
    basis = groebner(polys, symbols, order='grevlex', polys=False)

    #
    # No solutions (inconsistent equations)?
    #
    if 1 in basis:

        # No solutions:
        poly_sol = no_solutions
        poly_eqs = no_equations

    #
    # Finite number of solutions (zero-dimensional case)
    #
    elif basis.is_zero_dimensional:

        # Convert Groebner basis to lex ordering
        basis = basis.fglm('lex')

        # Convert polynomial coefficients back to float before calling
        # solve_poly_system
        if inexact:
            basis = [nfloat(p) for p in basis]

        # Solve the zero-dimensional case using solve_poly_system if possible.
        # If some polynomials have factors that cannot be solved in radicals
        # then this will fail. Using solve_poly_system(..., strict=True)
        # ensures that we either get a complete solution set in radicals or
        # UnsolvableFactorError will be raised.
        try:
            result = solve_poly_system(basis, *symbols, strict=True)
        except UnsolvableFactorError:
            # Failure... not fully solvable in radicals. Return the lex-order
            # basis for substitution to handle.
            poly_sol = no_information
            poly_eqs = list(basis)
        else:
            # Success! We have a finite solution set and solve_poly_system has
            # succeeded in finding all solutions. Return the solutions and also
            # an empty list of remaining equations to be solved.
            poly_sol = [dict(zip(symbols, res)) for res in result]
            poly_eqs = no_equations

    #
    # Infinite families of solutions (positive-dimensional case)
    #
    else:
        # In this case the grevlex basis cannot be converted to lex using the
        # fglm method and also solve_poly_system cannot solve the equations. We
        # would like to return a lex basis but since we can't use fglm we
        # compute the lex basis directly here. The time required to recompute
        # the basis is generally significantly less than the time required by
        # substitution to solve the new system.
        poly_sol = no_information
        poly_eqs = list(groebner(polys, symbols, order='lex', polys=False))

        if inexact:
            poly_eqs = [nfloat(p) for p in poly_eqs]

    return poly_sol, poly_eqs


def nonlinsolve(system, *symbols):
    r"""
    Solve system of $N$ nonlinear equations with $M$ variables, which means both
    under and overdetermined systems are supported. Positive dimensional
    system is also supported (A system with infinitely many solutions is said
    to be positive-dimensional). In a positive dimensional system the solution will
    be dependent on at least one symbol. Returns both real solution
    and complex solution (if they exist).

    Parameters
    ==========

    system : list of equations
        The target system of equations
    symbols : list of Symbols
        symbols should be given as a sequence eg. list

    Returns
    =======

    A :class:`~.FiniteSet` of ordered tuple of values of `symbols` for which the `system`
    has solution. Order of values in the tuple is same as symbols present in
    the parameter `symbols`.

    Please note that general :class:`~.FiniteSet` is unordered, the solution
    returned here is not simply a :class:`~.FiniteSet` of solutions, rather it
    is a :class:`~.FiniteSet` of ordered tuple, i.e. the first and only
    argument to :class:`~.FiniteSet` is a tuple of solutions, which is
    ordered, and, hence ,the returned solution is ordered.

    Also note that solution could also have been returned as an ordered tuple,
    FiniteSet is just a wrapper ``{}`` around the tuple. It has no other
    significance except for the fact it is just used to maintain a consistent
    output format throughout the solveset.

    For the given set of equations, the respective input types
    are given below:

    .. math:: xy - 1 = 0
    .. math:: 4x^2 + y^2 - 5 = 0

    ::

       system  = [x*y - 1, 4*x**2 + y**2 - 5]
       symbols = [x, y]

    Raises
    ======

    ValueError
        The input is not valid.
        The symbols are not given.
    AttributeError
        The input symbols are not `Symbol` type.

    Examples
    ========

    >>> from sympy import symbols, nonlinsolve
    >>> x, y, z = symbols('x, y, z', real=True)
    >>> nonlinsolve([x*y - 1, 4*x**2 + y**2 - 5], [x, y])
    {(-1, -1), (-1/2, -2), (1/2, 2), (1, 1)}

    1. Positive dimensional system and complements:

    >>> from sympy import pprint
    >>> from sympy.polys.polytools import is_zero_dimensional
    >>> a, b, c, d = symbols('a, b, c, d', extended_real=True)
    >>> eq1 =  a + b + c + d
    >>> eq2 = a*b + b*c + c*d + d*a
    >>> eq3 = a*b*c + b*c*d + c*d*a + d*a*b
    >>> eq4 = a*b*c*d - 1
    >>> system = [eq1, eq2, eq3, eq4]
    >>> is_zero_dimensional(system)
    False
    >>> pprint(nonlinsolve(system, [a, b, c, d]), use_unicode=False)
      -1       1               1      -1
    {(---, -d, -, {d} \ {0}), (-, -d, ---, {d} \ {0})}
       d       d               d       d
    >>> nonlinsolve([(x+y)**2 - 4, x + y - 2], [x, y])
    {(2 - y, y)}

    2. If some of the equations are non-polynomial then `nonlinsolve`
    will call the ``substitution`` function and return real and complex solutions,
    if present.

    >>> from sympy import exp, sin
    >>> nonlinsolve([exp(x) - sin(y), y**2 - 4], [x, y])
    {(ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2),
     (ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2)}

    3. If system is non-linear polynomial and zero-dimensional then it
    returns both solution (real and complex solutions, if present) using
    :func:`~.solve_poly_system`:

    >>> from sympy import sqrt
    >>> nonlinsolve([x**2 - 2*y**2 -2, x*y - 2], [x, y])
    {(-2, -1), (2, 1), (-sqrt(2)*I, sqrt(2)*I), (sqrt(2)*I, -sqrt(2)*I)}

    4. ``nonlinsolve`` can solve some linear (zero or positive dimensional)
    system (because it uses the :func:`sympy.polys.polytools.groebner` function to get the
    groebner basis and then uses the ``substitution`` function basis as the
    new `system`). But it is not recommended to solve linear system using
    ``nonlinsolve``, because :func:`~.linsolve` is better for general linear systems.

    >>> nonlinsolve([x + 2*y -z - 3, x - y - 4*z + 9, y + z - 4], [x, y, z])
    {(3*z - 5, 4 - z, z)}

    5. System having polynomial equations and only real solution is
    solved using :func:`~.solve_poly_system`:

    >>> e1 = sqrt(x**2 + y**2) - 10
    >>> e2 = sqrt(y**2 + (-x + 10)**2) - 3
    >>> nonlinsolve((e1, e2), (x, y))
    {(191/20, -3*sqrt(391)/20), (191/20, 3*sqrt(391)/20)}
    >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [x, y])
    {(1, 2), (1 - sqrt(5), 2 + sqrt(5)), (1 + sqrt(5), 2 - sqrt(5))}
    >>> nonlinsolve([x**2 + 2/y - 2, x + y - 3], [y, x])
    {(2, 1), (2 - sqrt(5), 1 + sqrt(5)), (2 + sqrt(5), 1 - sqrt(5))}

    6. It is better to use symbols instead of trigonometric functions or
    :class:`~.Function`. For example, replace $\sin(x)$ with a symbol, replace
    $f(x)$ with a symbol and so on. Get a solution from ``nonlinsolve`` and then
    use :func:`~.solveset` to get the value of $x$.

    How nonlinsolve is better than old solver ``_solve_system`` :
    =============================================================

    1. A positive dimensional system solver: nonlinsolve can return
    solution for positive dimensional system. It finds the
    Groebner Basis of the positive dimensional system(calling it as
    basis) then we can start solving equation(having least number of
    variable first in the basis) using solveset and substituting that
    solved solutions into other equation(of basis) to get solution in
    terms of minimum variables. Here the important thing is how we
    are substituting the known values and in which equations.

    2. Real and complex solutions: nonlinsolve returns both real
    and complex solution. If all the equations in the system are polynomial
    then using :func:`~.solve_poly_system` both real and complex solution is returned.
    If all the equations in the system are not polynomial equation then goes to
    ``substitution`` method with this polynomial and non polynomial equation(s),
    to solve for unsolved variables. Here to solve for particular variable
    solveset_real and solveset_complex is used. For both real and complex
    solution ``_solve_using_known_values`` is used inside ``substitution``
    (``substitution`` will be called when any non-polynomial equation is present).
    If a solution is valid its general solution is added to the final result.

    3. :class:`~.Complement` and :class:`~.Intersection` will be added:
    nonlinsolve maintains dict for complements and intersections. If solveset
    find complements or/and intersections with any interval or set during the
    execution of ``substitution`` function, then complement or/and
    intersection for that variable is added before returning final solution.

    """
    if not system:
        return S.EmptySet

    if not symbols:
        msg = ('Symbols must be given, for which solution of the '
               'system is to be found.')
        raise ValueError(filldedent(msg))

    if hasattr(symbols[0], '__iter__'):
        symbols = symbols[0]

    if not is_sequence(symbols) or not symbols:
        msg = ('Symbols must be given, for which solution of the '
               'system is to be found.')
        raise IndexError(filldedent(msg))

    symbols = list(map(_sympify, symbols))
    system, symbols, swap = recast_to_symbols(system, symbols)
    if swap:
        soln = nonlinsolve(system, symbols)
        return FiniteSet(*[tuple(i.xreplace(swap) for i in s) for s in soln])

    if len(system) == 1 and len(symbols) == 1:
        return _solveset_work(system, symbols)

    # main code of def nonlinsolve() starts from here

    polys, polys_expr, nonpolys, denominators, unrad_changed = \
        _separate_poly_nonpoly(system, symbols)

    poly_eqs = []
    poly_sol = [{}]

    if polys:
        poly_sol, poly_eqs = _handle_poly(polys, symbols)
        if poly_sol and poly_sol[0]:
            poly_syms = set().union(*(eq.free_symbols for eq in polys))
            unrad_syms = set().union(*(eq.free_symbols for eq in unrad_changed))
            if unrad_syms == poly_syms and unrad_changed:
                # if all the symbols have been solved by _handle_poly
                # and unrad has been used then check solutions
                poly_sol = [sol for sol in poly_sol if checksol(unrad_changed, sol)]

    # Collect together the unsolved polynomials with the non-polynomial
    # equations.
    remaining = poly_eqs + nonpolys

    # to_tuple converts a solution dictionary to a tuple containing the
    # value for each symbol
    to_tuple = lambda sol: tuple(sol[s] for s in symbols)

    if not remaining:
        # If there is nothing left to solve then return the solution from
        # solve_poly_system directly.
        return FiniteSet(*map(to_tuple, poly_sol))
    else:
        # Here we handle:
        #
        #  1. The Groebner basis if solve_poly_system failed.
        #  2. The Groebner basis in the positive-dimensional case.
        #  3. Any non-polynomial equations
        #
        # If solve_poly_system did succeed then we pass those solutions in as
        # preliminary results.
        subs_res = substitution(remaining, symbols, result=poly_sol, exclude=denominators)

        if not isinstance(subs_res, FiniteSet):
            return subs_res

        # check solutions produced by substitution. Currently, checking is done for
        # only those solutions which have non-Set variable values.
        if unrad_changed:
            result = [dict(zip(symbols, sol)) for sol in subs_res.args]
            correct_sols = [sol for sol in result if any(isinstance(v, Set) for v in sol)
                            or checksol(unrad_changed, sol) != False]
            return FiniteSet(*map(to_tuple, correct_sols))
        else:
            return subs_res
