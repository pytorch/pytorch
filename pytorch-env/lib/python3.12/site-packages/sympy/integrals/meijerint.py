"""
Integrate functions by rewriting them as Meijer G-functions.

There are three user-visible functions that can be used by other parts of the
sympy library to solve various integration problems:

- meijerint_indefinite
- meijerint_definite
- meijerint_inversion

They can be used to compute, respectively, indefinite integrals, definite
integrals over intervals of the real line, and inverse laplace-type integrals
(from c-I*oo to c+I*oo). See the respective docstrings for details.

The main references for this are:

[L] Luke, Y. L. (1969), The Special Functions and Their Approximations,
    Volume 1

[R] Kelly B. Roach.  Meijer G Function Representations.
    In: Proceedings of the 1997 International Symposium on Symbolic and
    Algebraic Computation, pages 205-211, New York, 1997. ACM.

[P] A. P. Prudnikov, Yu. A. Brychkov and O. I. Marichev (1990).
    Integrals and Series: More Special Functions, Vol. 3,.
    Gordon and Breach Science Publisher
"""

from __future__ import annotations
import itertools

from sympy import SYMPY_DEBUG
from sympy.core import S, Expr
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.function import (expand, expand_mul, expand_power_base,
                                 expand_trig, Function)
from sympy.core.mul import Mul
from sympy.core.intfunc import ilcm
from sympy.core.numbers import Rational, pi
from sympy.core.relational import Eq, Ne, _canonical_coeff
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Dummy, symbols, Wild, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (re, im, arg, Abs, sign,
        unpolarify, polarify, polar_lift, principal_branch, unbranched_argument,
        periodic_argument)
from sympy.functions.elementary.exponential import exp, exp_polar, log
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.hyperbolic import (cosh, sinh,
        _rewrite_hyperbolics_as_exp, HyperbolicFunction)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise, piecewise_fold
from sympy.functions.elementary.trigonometric import (cos, sin, sinc,
        TrigonometricFunction)
from sympy.functions.special.bessel import besselj, bessely, besseli, besselk
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.elliptic_integrals import elliptic_k, elliptic_e
from sympy.functions.special.error_functions import (erf, erfc, erfi, Ei,
        expint, Si, Ci, Shi, Chi, fresnels, fresnelc)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg
from sympy.functions.special.singularity_functions import SingularityFunction
from .integrals import Integral
from sympy.logic.boolalg import And, Or, BooleanAtom, Not, BooleanFunction
from sympy.polys import cancel, factor
from sympy.utilities.iterables import multiset_partitions
from sympy.utilities.misc import debug as _debug
from sympy.utilities.misc import debugf as _debugf

# keep this at top for easy reference
z = Dummy('z')


def _has(res, *f):
    # return True if res has f; in the case of Piecewise
    # only return True if *all* pieces have f
    res = piecewise_fold(res)
    if getattr(res, 'is_Piecewise', False):
        return all(_has(i, *f) for i in res.args)
    return res.has(*f)


def _create_lookup_table(table):
    """ Add formulae for the function -> meijerg lookup table. """
    def wild(n):
        return Wild(n, exclude=[z])
    p, q, a, b, c = list(map(wild, 'pqabc'))
    n = Wild('n', properties=[lambda x: x.is_Integer and x > 0])
    t = p*z**q

    def add(formula, an, ap, bm, bq, arg=t, fac=S.One, cond=True, hint=True):
        table.setdefault(_mytype(formula, z), []).append((formula,
                                     [(fac, meijerg(an, ap, bm, bq, arg))], cond, hint))

    def addi(formula, inst, cond, hint=True):
        table.setdefault(
            _mytype(formula, z), []).append((formula, inst, cond, hint))

    def constant(a):
        return [(a, meijerg([1], [], [], [0], z)),
                (a, meijerg([], [1], [0], [], z))]
    table[()] = [(a, constant(a), True, True)]

    # [P], Section 8.
    class IsNonPositiveInteger(Function):

        @classmethod
        def eval(cls, arg):
            arg = unpolarify(arg)
            if arg.is_Integer is True:
                return arg <= 0

    # Section 8.4.2
    # TODO this needs more polar_lift (c/f entry for exp)
    add(Heaviside(t - b)*(t - b)**(a - 1), [a], [], [], [0], t/b,
        gamma(a)*b**(a - 1), And(b > 0))
    add(Heaviside(b - t)*(b - t)**(a - 1), [], [a], [0], [], t/b,
        gamma(a)*b**(a - 1), And(b > 0))
    add(Heaviside(z - (b/p)**(1/q))*(t - b)**(a - 1), [a], [], [], [0], t/b,
        gamma(a)*b**(a - 1), And(b > 0))
    add(Heaviside((b/p)**(1/q) - z)*(b - t)**(a - 1), [], [a], [0], [], t/b,
        gamma(a)*b**(a - 1), And(b > 0))
    add((b + t)**(-a), [1 - a], [], [0], [], t/b, b**(-a)/gamma(a),
        hint=Not(IsNonPositiveInteger(a)))
    add(Abs(b - t)**(-a), [1 - a], [(1 - a)/2], [0], [(1 - a)/2], t/b,
        2*sin(pi*a/2)*gamma(1 - a)*Abs(b)**(-a), re(a) < 1)
    add((t**a - b**a)/(t - b), [0, a], [], [0, a], [], t/b,
        b**(a - 1)*sin(a*pi)/pi)

    # 12
    def A1(r, sign, nu):
        return pi**Rational(-1, 2)*(-sign*nu/2)**(1 - 2*r)

    def tmpadd(r, sgn):
        # XXX the a**2 is bad for matching
        add((sqrt(a**2 + t) + sgn*a)**b/(a**2 + t)**r,
            [(1 + b)/2, 1 - 2*r + b/2], [],
            [(b - sgn*b)/2], [(b + sgn*b)/2], t/a**2,
            a**(b - 2*r)*A1(r, sgn, b))
    tmpadd(0, 1)
    tmpadd(0, -1)
    tmpadd(S.Half, 1)
    tmpadd(S.Half, -1)

    # 13
    def tmpadd(r, sgn):
        add((sqrt(a + p*z**q) + sgn*sqrt(p)*z**(q/2))**b/(a + p*z**q)**r,
            [1 - r + sgn*b/2], [1 - r - sgn*b/2], [0, S.Half], [],
            p*z**q/a, a**(b/2 - r)*A1(r, sgn, b))
    tmpadd(0, 1)
    tmpadd(0, -1)
    tmpadd(S.Half, 1)
    tmpadd(S.Half, -1)
    # (those after look obscure)

    # Section 8.4.3
    add(exp(polar_lift(-1)*t), [], [], [0], [])

    # TODO can do sin^n, sinh^n by expansion ... where?
    # 8.4.4 (hyperbolic functions)
    add(sinh(t), [], [1], [S.Half], [1, 0], t**2/4, pi**Rational(3, 2))
    add(cosh(t), [], [S.Half], [0], [S.Half, S.Half], t**2/4, pi**Rational(3, 2))

    # Section 8.4.5
    # TODO can do t + a. but can also do by expansion... (XXX not really)
    add(sin(t), [], [], [S.Half], [0], t**2/4, sqrt(pi))
    add(cos(t), [], [], [0], [S.Half], t**2/4, sqrt(pi))

    # Section 8.4.6 (sinc function)
    add(sinc(t), [], [], [0], [Rational(-1, 2)], t**2/4, sqrt(pi)/2)

    # Section 8.5.5
    def make_log1(subs):
        N = subs[n]
        return [(S.NegativeOne**N*factorial(N),
                 meijerg([], [1]*(N + 1), [0]*(N + 1), [], t))]

    def make_log2(subs):
        N = subs[n]
        return [(factorial(N),
                 meijerg([1]*(N + 1), [], [], [0]*(N + 1), t))]
    # TODO these only hold for positive p, and can be made more general
    #      but who uses log(x)*Heaviside(a-x) anyway ...
    # TODO also it would be nice to derive them recursively ...
    addi(log(t)**n*Heaviside(1 - t), make_log1, True)
    addi(log(t)**n*Heaviside(t - 1), make_log2, True)

    def make_log3(subs):
        return make_log1(subs) + make_log2(subs)
    addi(log(t)**n, make_log3, True)
    addi(log(t + a),
         constant(log(a)) + [(S.One, meijerg([1, 1], [], [1], [0], t/a))],
         True)
    addi(log(Abs(t - a)), constant(log(Abs(a))) +
         [(pi, meijerg([1, 1], [S.Half], [1], [0, S.Half], t/a))],
         True)
    # TODO log(x)/(x+a) and log(x)/(x-1) can also be done. should they
    #      be derivable?
    # TODO further formulae in this section seem obscure

    # Sections 8.4.9-10
    # TODO

    # Section 8.4.11
    addi(Ei(t),
         constant(-S.ImaginaryUnit*pi) + [(S.NegativeOne, meijerg([], [1], [0, 0], [],
                  t*polar_lift(-1)))],
         True)

    # Section 8.4.12
    add(Si(t), [1], [], [S.Half], [0, 0], t**2/4, sqrt(pi)/2)
    add(Ci(t), [], [1], [0, 0], [S.Half], t**2/4, -sqrt(pi)/2)

    # Section 8.4.13
    add(Shi(t), [S.Half], [], [0], [Rational(-1, 2), Rational(-1, 2)], polar_lift(-1)*t**2/4,
        t*sqrt(pi)/4)
    add(Chi(t), [], [S.Half, 1], [0, 0], [S.Half, S.Half], t**2/4, -
        pi**S('3/2')/2)

    # generalized exponential integral
    add(expint(a, t), [], [a], [a - 1, 0], [], t)

    # Section 8.4.14
    add(erf(t), [1], [], [S.Half], [0], t**2, 1/sqrt(pi))
    # TODO exp(-x)*erf(I*x) does not work
    add(erfc(t), [], [1], [0, S.Half], [], t**2, 1/sqrt(pi))
    # This formula for erfi(z) yields a wrong(?) minus sign
    #add(erfi(t), [1], [], [S.Half], [0], -t**2, I/sqrt(pi))
    add(erfi(t), [S.Half], [], [0], [Rational(-1, 2)], -t**2, t/sqrt(pi))

    # Fresnel Integrals
    add(fresnels(t), [1], [], [Rational(3, 4)], [0, Rational(1, 4)], pi**2*t**4/16, S.Half)
    add(fresnelc(t), [1], [], [Rational(1, 4)], [0, Rational(3, 4)], pi**2*t**4/16, S.Half)

    ##### bessel-type functions #####
    # Section 8.4.19
    add(besselj(a, t), [], [], [a/2], [-a/2], t**2/4)

    # all of the following are derivable
    #add(sin(t)*besselj(a, t), [Rational(1, 4), Rational(3, 4)], [], [(1+a)/2],
    #    [-a/2, a/2, (1-a)/2], t**2, 1/sqrt(2))
    #add(cos(t)*besselj(a, t), [Rational(1, 4), Rational(3, 4)], [], [a/2],
    #    [-a/2, (1+a)/2, (1-a)/2], t**2, 1/sqrt(2))
    #add(besselj(a, t)**2, [S.Half], [], [a], [-a, 0], t**2, 1/sqrt(pi))
    #add(besselj(a, t)*besselj(b, t), [0, S.Half], [], [(a + b)/2],
    #    [-(a+b)/2, (a - b)/2, (b - a)/2], t**2, 1/sqrt(pi))

    # Section 8.4.20
    add(bessely(a, t), [], [-(a + 1)/2], [a/2, -a/2], [-(a + 1)/2], t**2/4)

    # TODO all of the following should be derivable
    #add(sin(t)*bessely(a, t), [Rational(1, 4), Rational(3, 4)], [(1 - a - 1)/2],
    #    [(1 + a)/2, (1 - a)/2], [(1 - a - 1)/2, (1 - 1 - a)/2, (1 - 1 + a)/2],
    #    t**2, 1/sqrt(2))
    #add(cos(t)*bessely(a, t), [Rational(1, 4), Rational(3, 4)], [(0 - a - 1)/2],
    #    [(0 + a)/2, (0 - a)/2], [(0 - a - 1)/2, (1 - 0 - a)/2, (1 - 0 + a)/2],
    #    t**2, 1/sqrt(2))
    #add(besselj(a, t)*bessely(b, t), [0, S.Half], [(a - b - 1)/2],
    #    [(a + b)/2, (a - b)/2], [(a - b - 1)/2, -(a + b)/2, (b - a)/2],
    #    t**2, 1/sqrt(pi))
    #addi(bessely(a, t)**2,
    #     [(2/sqrt(pi), meijerg([], [S.Half, S.Half - a], [0, a, -a],
    #                           [S.Half - a], t**2)),
    #      (1/sqrt(pi), meijerg([S.Half], [], [a], [-a, 0], t**2))],
    #     True)
    #addi(bessely(a, t)*bessely(b, t),
    #     [(2/sqrt(pi), meijerg([], [0, S.Half, (1 - a - b)/2],
    #                           [(a + b)/2, (a - b)/2, (b - a)/2, -(a + b)/2],
    #                           [(1 - a - b)/2], t**2)),
    #      (1/sqrt(pi), meijerg([0, S.Half], [], [(a + b)/2],
    #                           [-(a + b)/2, (a - b)/2, (b - a)/2], t**2))],
    #     True)

    # Section 8.4.21 ?
    # Section 8.4.22
    add(besseli(a, t), [], [(1 + a)/2], [a/2], [-a/2, (1 + a)/2], t**2/4, pi)
    # TODO many more formulas. should all be derivable

    # Section 8.4.23
    add(besselk(a, t), [], [], [a/2, -a/2], [], t**2/4, S.Half)
    # TODO many more formulas. should all be derivable

    # Complete elliptic integrals K(z) and E(z)
    add(elliptic_k(t), [S.Half, S.Half], [], [0], [0], -t, S.Half)
    add(elliptic_e(t), [S.Half, 3*S.Half], [], [0], [0], -t, Rational(-1, 2)/2)


####################################################################
# First some helper functions.
####################################################################

from sympy.utilities.timeutils import timethis
timeit = timethis('meijerg')


def _mytype(f: Basic, x: Symbol) -> tuple[type[Basic], ...]:
    """ Create a hashable entity describing the type of f. """
    def key(x: type[Basic]) -> tuple[int, int, str]:
        return x.class_key()

    if x not in f.free_symbols:
        return ()
    elif f.is_Function:
        return type(f),
    return tuple(sorted((t for a in f.args for t in _mytype(a, x)), key=key))


class _CoeffExpValueError(ValueError):
    """
    Exception raised by _get_coeff_exp, for internal use only.
    """
    pass


def _get_coeff_exp(expr, x):
    """
    When expr is known to be of the form c*x**b, with c and/or b possibly 1,
    return c, b.

    Examples
    ========

    >>> from sympy.abc import x, a, b
    >>> from sympy.integrals.meijerint import _get_coeff_exp
    >>> _get_coeff_exp(a*x**b, x)
    (a, b)
    >>> _get_coeff_exp(x, x)
    (1, 1)
    >>> _get_coeff_exp(2*x, x)
    (2, 1)
    >>> _get_coeff_exp(x**3, x)
    (1, 3)
    """
    from sympy.simplify import powsimp
    (c, m) = expand_power_base(powsimp(expr)).as_coeff_mul(x)
    if not m:
        return c, S.Zero
    [m] = m
    if m.is_Pow:
        if m.base != x:
            raise _CoeffExpValueError('expr not of form a*x**b')
        return c, m.exp
    elif m == x:
        return c, S.One
    else:
        raise _CoeffExpValueError('expr not of form a*x**b: %s' % expr)


def _exponents(expr, x):
    """
    Find the exponents of ``x`` (not including zero) in ``expr``.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _exponents
    >>> from sympy.abc import x, y
    >>> from sympy import sin
    >>> _exponents(x, x)
    {1}
    >>> _exponents(x**2, x)
    {2}
    >>> _exponents(x**2 + x, x)
    {1, 2}
    >>> _exponents(x**3*sin(x + x**y) + 1/x, x)
    {-1, 1, 3, y}
    """
    def _exponents_(expr, x, res):
        if expr == x:
            res.update([1])
            return
        if expr.is_Pow and expr.base == x:
            res.update([expr.exp])
            return
        for argument in expr.args:
            _exponents_(argument, x, res)
    res = set()
    _exponents_(expr, x, res)
    return res


def _functions(expr, x):
    """ Find the types of functions in expr, to estimate the complexity. """
    return {e.func for e in expr.atoms(Function) if x in e.free_symbols}


def _find_splitting_points(expr, x):
    """
    Find numbers a such that a linear substitution x -> x + a would
    (hopefully) simplify expr.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _find_splitting_points as fsp
    >>> from sympy import sin
    >>> from sympy.abc import x
    >>> fsp(x, x)
    {0}
    >>> fsp((x-1)**3, x)
    {1}
    >>> fsp(sin(x+3)*x, x)
    {-3, 0}
    """
    p, q = [Wild(n, exclude=[x]) for n in 'pq']

    def compute_innermost(expr, res):
        if not isinstance(expr, Expr):
            return
        m = expr.match(p*x + q)
        if m and m[p] != 0:
            res.add(-m[q]/m[p])
            return
        if expr.is_Atom:
            return
        for argument in expr.args:
            compute_innermost(argument, res)
    innermost = set()
    compute_innermost(expr, innermost)
    return innermost


def _split_mul(f, x):
    """
    Split expression ``f`` into fac, po, g, where fac is a constant factor,
    po = x**s for some s independent of s, and g is "the rest".

    Examples
    ========

    >>> from sympy.integrals.meijerint import _split_mul
    >>> from sympy import sin
    >>> from sympy.abc import s, x
    >>> _split_mul((3*x)**s*sin(x**2)*x, x)
    (3**s, x*x**s, sin(x**2))
    """
    fac = S.One
    po = S.One
    g = S.One
    f = expand_power_base(f)

    args = Mul.make_args(f)
    for a in args:
        if a == x:
            po *= x
        elif x not in a.free_symbols:
            fac *= a
        else:
            if a.is_Pow and x not in a.exp.free_symbols:
                c, t = a.base.as_coeff_mul(x)
                if t != (x,):
                    c, t = expand_mul(a.base).as_coeff_mul(x)
                if t == (x,):
                    po *= x**a.exp
                    fac *= unpolarify(polarify(c**a.exp, subs=False))
                    continue
            g *= a

    return fac, po, g


def _mul_args(f):
    """
    Return a list ``L`` such that ``Mul(*L) == f``.

    If ``f`` is not a ``Mul`` or ``Pow``, ``L=[f]``.
    If ``f=g**n`` for an integer ``n``, ``L=[g]*n``.
    If ``f`` is a ``Mul``, ``L`` comes from applying ``_mul_args`` to all factors of ``f``.
    """
    args = Mul.make_args(f)
    gs = []
    for g in args:
        if g.is_Pow and g.exp.is_Integer:
            n = g.exp
            base = g.base
            if n < 0:
                n = -n
                base = 1/base
            gs += [base]*n
        else:
            gs.append(g)
    return gs


def _mul_as_two_parts(f):
    """
    Find all the ways to split ``f`` into a product of two terms.
    Return None on failure.

    Explanation
    ===========

    Although the order is canonical from multiset_partitions, this is
    not necessarily the best order to process the terms. For example,
    if the case of len(gs) == 2 is removed and multiset is allowed to
    sort the terms, some tests fail.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _mul_as_two_parts
    >>> from sympy import sin, exp, ordered
    >>> from sympy.abc import x
    >>> list(ordered(_mul_as_two_parts(x*sin(x)*exp(x))))
    [(x, exp(x)*sin(x)), (x*exp(x), sin(x)), (x*sin(x), exp(x))]
    """

    gs = _mul_args(f)
    if len(gs) < 2:
        return None
    if len(gs) == 2:
        return [tuple(gs)]
    return [(Mul(*x), Mul(*y)) for (x, y) in multiset_partitions(gs, 2)]


def _inflate_g(g, n):
    """ Return C, h such that h is a G function of argument z**n and
        g = C*h. """
    # TODO should this be a method of meijerg?
    # See: [L, page 150, equation (5)]
    def inflate(params, n):
        """ (a1, .., ak) -> (a1/n, (a1+1)/n, ..., (ak + n-1)/n) """
        return [(a + i)/n for a, i in itertools.product(params, range(n))]
    v = S(len(g.ap) - len(g.bq))
    C = n**(1 + g.nu + v/2)
    C /= (2*pi)**((n - 1)*g.delta)
    return C, meijerg(inflate(g.an, n), inflate(g.aother, n),
                      inflate(g.bm, n), inflate(g.bother, n),
                      g.argument**n * n**(n*v))


def _flip_g(g):
    """ Turn the G function into one of inverse argument
        (i.e. G(1/x) -> G'(x)) """
    # See [L], section 5.2
    def tr(l):
        return [1 - a for a in l]
    return meijerg(tr(g.bm), tr(g.bother), tr(g.an), tr(g.aother), 1/g.argument)


def _inflate_fox_h(g, a):
    r"""
    Let d denote the integrand in the definition of the G function ``g``.
    Consider the function H which is defined in the same way, but with
    integrand d/Gamma(a*s) (contour conventions as usual).

    If ``a`` is rational, the function H can be written as C*G, for a constant C
    and a G-function G.

    This function returns C, G.
    """
    if a < 0:
        return _inflate_fox_h(_flip_g(g), -a)
    p = S(a.p)
    q = S(a.q)
    # We use the substitution s->qs, i.e. inflate g by q. We are left with an
    # extra factor of Gamma(p*s), for which we use Gauss' multiplication
    # theorem.
    D, g = _inflate_g(g, q)
    z = g.argument
    D /= (2*pi)**((1 - p)/2)*p**Rational(-1, 2)
    z /= p**p
    bs = [(n + 1)/p for n in range(p)]
    return D, meijerg(g.an, g.aother, g.bm, list(g.bother) + bs, z)

_dummies: dict[tuple[str, str], Dummy]  = {}


def _dummy(name, token, expr, **kwargs):
    """
    Return a dummy. This will return the same dummy if the same token+name is
    requested more than once, and it is not already in expr.
    This is for being cache-friendly.
    """
    d = _dummy_(name, token, **kwargs)
    if d in expr.free_symbols:
        return Dummy(name, **kwargs)
    return d


def _dummy_(name, token, **kwargs):
    """
    Return a dummy associated to name and token. Same effect as declaring
    it globally.
    """
    global _dummies
    if not (name, token) in _dummies:
        _dummies[(name, token)] = Dummy(name, **kwargs)
    return _dummies[(name, token)]


def _is_analytic(f, x):
    """ Check if f(x), when expressed using G functions on the positive reals,
        will in fact agree with the G functions almost everywhere """
    return not any(x in expr.free_symbols for expr in f.atoms(Heaviside, Abs))


def _condsimp(cond, first=True):
    """
    Do naive simplifications on ``cond``.

    Explanation
    ===========

    Note that this routine is completely ad-hoc, simplification rules being
    added as need arises rather than following any logical pattern.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _condsimp as simp
    >>> from sympy import Or, Eq
    >>> from sympy.abc import x, y
    >>> simp(Or(x < y, Eq(x, y)))
    x <= y
    """
    if first:
        cond = cond.replace(lambda _: _.is_Relational, _canonical_coeff)
        first = False
    if not isinstance(cond, BooleanFunction):
        return cond
    p, q, r = symbols('p q r', cls=Wild)
    # transforms tests use 0, 4, 5 and 11-14
    # meijer tests use 0, 2, 11, 14
    # joint_rv uses 6, 7
    rules = [
        (Or(p < q, Eq(p, q)), p <= q),  # 0
        # The next two obviously are instances of a general pattern, but it is
        # easier to spell out the few cases we care about.
        (And(Abs(arg(p)) <= pi, Abs(arg(p) - 2*pi) <= pi),
         Eq(arg(p) - pi, 0)),  # 1
        (And(Abs(2*arg(p) + pi) <= pi, Abs(2*arg(p) - pi) <= pi),
         Eq(arg(p), 0)), # 2
        (And(Abs(2*arg(p) + pi) < pi, Abs(2*arg(p) - pi) <= pi),
         S.false),  # 3
        (And(Abs(arg(p) - pi/2) <= pi/2, Abs(arg(p) + pi/2) <= pi/2),
         Eq(arg(p), 0)),  # 4
        (And(Abs(arg(p) - pi/2) <= pi/2, Abs(arg(p) + pi/2) < pi/2),
         S.false),  # 5
        (And(Abs(arg(p**2/2 + 1)) < pi, Ne(Abs(arg(p**2/2 + 1)), pi)),
         S.true),  # 6
        (Or(Abs(arg(p**2/2 + 1)) < pi, Ne(1/(p**2/2 + 1), 0)),
         S.true),  # 7
        (And(Abs(unbranched_argument(p)) <= pi,
           Abs(unbranched_argument(exp_polar(-2*pi*S.ImaginaryUnit)*p)) <= pi),
         Eq(unbranched_argument(exp_polar(-S.ImaginaryUnit*pi)*p), 0)),  # 8
        (And(Abs(unbranched_argument(p)) <= pi/2,
           Abs(unbranched_argument(exp_polar(-pi*S.ImaginaryUnit)*p)) <= pi/2),
         Eq(unbranched_argument(exp_polar(-S.ImaginaryUnit*pi/2)*p), 0)),  # 9
        (Or(p <= q, And(p < q, r)), p <= q),  # 10
        (Ne(p**2, 1) & (p**2 > 1), p**2 > 1),  # 11
        (Ne(1/p, 1) & (cos(Abs(arg(p)))*Abs(p) > 1), Abs(p) > 1),  # 12
        (Ne(p, 2) & (cos(Abs(arg(p)))*Abs(p) > 2), Abs(p) > 2),  # 13
        ((Abs(arg(p)) < pi/2) & (cos(Abs(arg(p)))*sqrt(Abs(p**2)) > 1), p**2 > 1),  # 14
    ]
    cond = cond.func(*[_condsimp(_, first) for _ in cond.args])
    change = True
    while change:
        change = False
        for irule, (fro, to) in enumerate(rules):
            if fro.func != cond.func:
                continue
            for n, arg1 in enumerate(cond.args):
                if r in fro.args[0].free_symbols:
                    m = arg1.match(fro.args[1])
                    num = 1
                else:
                    num = 0
                    m = arg1.match(fro.args[0])
                if not m:
                    continue
                otherargs = [x.subs(m) for x in fro.args[:num] + fro.args[num + 1:]]
                otherlist = [n]
                for arg2 in otherargs:
                    for k, arg3 in enumerate(cond.args):
                        if k in otherlist:
                            continue
                        if arg2 == arg3:
                            otherlist += [k]
                            break
                        if isinstance(arg3, And) and arg2.args[1] == r and \
                                isinstance(arg2, And) and arg2.args[0] in arg3.args:
                            otherlist += [k]
                            break
                        if isinstance(arg3, And) and arg2.args[0] == r and \
                                isinstance(arg2, And) and arg2.args[1] in arg3.args:
                            otherlist += [k]
                            break
                if len(otherlist) != len(otherargs) + 1:
                    continue
                newargs = [arg_ for (k, arg_) in enumerate(cond.args)
                           if k not in otherlist] + [to.subs(m)]
                if SYMPY_DEBUG:
                    if irule not in (0, 2, 4, 5, 6, 7, 11, 12, 13, 14):
                        print('used new rule:', irule)
                cond = cond.func(*newargs)
                change = True
                break

    # final tweak
    def rel_touchup(rel):
        if rel.rel_op != '==' or rel.rhs != 0:
            return rel

        # handle Eq(*, 0)
        LHS = rel.lhs
        m = LHS.match(arg(p)**q)
        if not m:
            m = LHS.match(unbranched_argument(polar_lift(p)**q))
        if not m:
            if isinstance(LHS, periodic_argument) and not LHS.args[0].is_polar \
                    and LHS.args[1] is S.Infinity:
                return (LHS.args[0] > 0)
            return rel
        return (m[p] > 0)
    cond = cond.replace(lambda _: _.is_Relational, rel_touchup)
    if SYMPY_DEBUG:
        print('_condsimp: ', cond)
    return cond

def _eval_cond(cond):
    """ Re-evaluate the conditions. """
    if isinstance(cond, bool):
        return cond
    return _condsimp(cond.doit())

####################################################################
# Now the "backbone" functions to do actual integration.
####################################################################


def _my_principal_branch(expr, period, full_pb=False):
    """ Bring expr nearer to its principal branch by removing superfluous
        factors.
        This function does *not* guarantee to yield the principal branch,
        to avoid introducing opaque principal_branch() objects,
        unless full_pb=True. """
    res = principal_branch(expr, period)
    if not full_pb:
        res = res.replace(principal_branch, lambda x, y: x)
    return res


def _rewrite_saxena_1(fac, po, g, x):
    """
    Rewrite the integral fac*po*g dx, from zero to infinity, as
    integral fac*G, where G has argument a*x. Note po=x**s.
    Return fac, G.
    """
    _, s = _get_coeff_exp(po, x)
    a, b = _get_coeff_exp(g.argument, x)
    period = g.get_period()
    a = _my_principal_branch(a, period)

    # We substitute t = x**b.
    C = fac/(Abs(b)*a**((s + 1)/b - 1))
    # Absorb a factor of (at)**((1 + s)/b - 1).

    def tr(l):
        return [a + (1 + s)/b - 1 for a in l]
    return C, meijerg(tr(g.an), tr(g.aother), tr(g.bm), tr(g.bother),
                      a*x)


def _check_antecedents_1(g, x, helper=False):
    r"""
    Return a condition under which the mellin transform of g exists.
    Any power of x has already been absorbed into the G function,
    so this is just $\int_0^\infty g\, dx$.

    See [L, section 5.6.1]. (Note that s=1.)

    If ``helper`` is True, only check if the MT exists at infinity, i.e. if
    $\int_1^\infty g\, dx$ exists.
    """
    # NOTE if you update these conditions, please update the documentation as well
    delta = g.delta
    eta, _ = _get_coeff_exp(g.argument, x)
    m, n, p, q = S([len(g.bm), len(g.an), len(g.ap), len(g.bq)])

    if p > q:
        def tr(l):
            return [1 - x for x in l]
        return _check_antecedents_1(meijerg(tr(g.bm), tr(g.bother),
                                            tr(g.an), tr(g.aother), x/eta),
                                    x)

    tmp = [-re(b) < 1 for b in g.bm] + [1 < 1 - re(a) for a in g.an]
    cond_3 = And(*tmp)

    tmp += [-re(b) < 1 for b in g.bother]
    tmp += [1 < 1 - re(a) for a in g.aother]
    cond_3_star = And(*tmp)

    cond_4 = (-re(g.nu) + (q + 1 - p)/2 > q - p)

    def debug(*msg):
        _debug(*msg)

    def debugf(string, arg):
        _debugf(string, arg)

    debug('Checking antecedents for 1 function:')
    debugf('  delta=%s, eta=%s, m=%s, n=%s, p=%s, q=%s',
           (delta, eta, m, n, p, q))
    debugf('  ap = %s, %s', (list(g.an), list(g.aother)))
    debugf('  bq = %s, %s', (list(g.bm), list(g.bother)))
    debugf('  cond_3=%s, cond_3*=%s, cond_4=%s', (cond_3, cond_3_star, cond_4))

    conds = []

    # case 1
    case1 = []
    tmp1 = [1 <= n, p < q, 1 <= m]
    tmp2 = [1 <= p, 1 <= m, Eq(q, p + 1), Not(And(Eq(n, 0), Eq(m, p + 1)))]
    tmp3 = [1 <= p, Eq(q, p)]
    for k in range(ceiling(delta/2) + 1):
        tmp3 += [Ne(Abs(unbranched_argument(eta)), (delta - 2*k)*pi)]
    tmp = [delta > 0, Abs(unbranched_argument(eta)) < delta*pi]
    extra = [Ne(eta, 0), cond_3]
    if helper:
        extra = []
    for t in [tmp1, tmp2, tmp3]:
        case1 += [And(*(t + tmp + extra))]
    conds += case1
    debug('  case 1:', case1)

    # case 2
    extra = [cond_3]
    if helper:
        extra = []
    case2 = [And(Eq(n, 0), p + 1 <= m, m <= q,
                 Abs(unbranched_argument(eta)) < delta*pi, *extra)]
    conds += case2
    debug('  case 2:', case2)

    # case 3
    extra = [cond_3, cond_4]
    if helper:
        extra = []
    case3 = [And(p < q, 1 <= m, delta > 0, Eq(Abs(unbranched_argument(eta)), delta*pi),
                 *extra)]
    case3 += [And(p <= q - 2, Eq(delta, 0), Eq(Abs(unbranched_argument(eta)), 0), *extra)]
    conds += case3
    debug('  case 3:', case3)

    # TODO altered cases 4-7

    # extra case from wofram functions site:
    # (reproduced verbatim from Prudnikov, section 2.24.2)
    # https://functions.wolfram.com/HypergeometricFunctions/MeijerG/21/02/01/
    case_extra = []
    case_extra += [Eq(p, q), Eq(delta, 0), Eq(unbranched_argument(eta), 0), Ne(eta, 0)]
    if not helper:
        case_extra += [cond_3]
    s = []
    for a, b in zip(g.ap, g.bq):
        s += [b - a]
    case_extra += [re(Add(*s)) < 0]
    case_extra = And(*case_extra)
    conds += [case_extra]
    debug('  extra case:', [case_extra])

    case_extra_2 = [And(delta > 0, Abs(unbranched_argument(eta)) < delta*pi)]
    if not helper:
        case_extra_2 += [cond_3]
    case_extra_2 = And(*case_extra_2)
    conds += [case_extra_2]
    debug('  second extra case:', [case_extra_2])

    # TODO This leaves only one case from the three listed by Prudnikov.
    #      Investigate if these indeed cover everything; if so, remove the rest.

    return Or(*conds)


def _int0oo_1(g, x):
    r"""
    Evaluate $\int_0^\infty g\, dx$ using G functions,
    assuming the necessary conditions are fulfilled.

    Examples
    ========

    >>> from sympy.abc import a, b, c, d, x, y
    >>> from sympy import meijerg
    >>> from sympy.integrals.meijerint import _int0oo_1
    >>> _int0oo_1(meijerg([a], [b], [c], [d], x*y), x)
    gamma(-a)*gamma(c + 1)/(y*gamma(-d)*gamma(b + 1))
    """
    from sympy.simplify import gammasimp
    # See [L, section 5.6.1]. Note that s=1.
    eta, _ = _get_coeff_exp(g.argument, x)
    res = 1/eta
    # XXX TODO we should reduce order first
    for b in g.bm:
        res *= gamma(b + 1)
    for a in g.an:
        res *= gamma(1 - a - 1)
    for b in g.bother:
        res /= gamma(1 - b - 1)
    for a in g.aother:
        res /= gamma(a + 1)
    return gammasimp(unpolarify(res))


def _rewrite_saxena(fac, po, g1, g2, x, full_pb=False):
    """
    Rewrite the integral ``fac*po*g1*g2`` from 0 to oo in terms of G
    functions with argument ``c*x``.

    Explanation
    ===========

    Return C, f1, f2 such that integral C f1 f2 from 0 to infinity equals
    integral fac ``po``, ``g1``, ``g2`` from 0 to infinity.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _rewrite_saxena
    >>> from sympy.abc import s, t, m
    >>> from sympy import meijerg
    >>> g1 = meijerg([], [], [0], [], s*t)
    >>> g2 = meijerg([], [], [m/2], [-m/2], t**2/4)
    >>> r = _rewrite_saxena(1, t**0, g1, g2, t)
    >>> r[0]
    s/(4*sqrt(pi))
    >>> r[1]
    meijerg(((), ()), ((-1/2, 0), ()), s**2*t/4)
    >>> r[2]
    meijerg(((), ()), ((m/2,), (-m/2,)), t/4)
    """
    def pb(g):
        a, b = _get_coeff_exp(g.argument, x)
        per = g.get_period()
        return meijerg(g.an, g.aother, g.bm, g.bother,
                       _my_principal_branch(a, per, full_pb)*x**b)

    _, s = _get_coeff_exp(po, x)
    _, b1 = _get_coeff_exp(g1.argument, x)
    _, b2 = _get_coeff_exp(g2.argument, x)
    if (b1 < 0) == True:
        b1 = -b1
        g1 = _flip_g(g1)
    if (b2 < 0) == True:
        b2 = -b2
        g2 = _flip_g(g2)
    if not b1.is_Rational or not b2.is_Rational:
        return
    m1, n1 = b1.p, b1.q
    m2, n2 = b2.p, b2.q
    tau = ilcm(m1*n2, m2*n1)
    r1 = tau//(m1*n2)
    r2 = tau//(m2*n1)

    C1, g1 = _inflate_g(g1, r1)
    C2, g2 = _inflate_g(g2, r2)
    g1 = pb(g1)
    g2 = pb(g2)

    fac *= C1*C2
    a1, b = _get_coeff_exp(g1.argument, x)
    a2, _ = _get_coeff_exp(g2.argument, x)

    # arbitrarily tack on the x**s part to g1
    # TODO should we try both?
    exp = (s + 1)/b - 1
    fac = fac/(Abs(b) * a1**exp)

    def tr(l):
        return [a + exp for a in l]
    g1 = meijerg(tr(g1.an), tr(g1.aother), tr(g1.bm), tr(g1.bother), a1*x)
    g2 = meijerg(g2.an, g2.aother, g2.bm, g2.bother, a2*x)

    from sympy.simplify import powdenest
    return powdenest(fac, polar=True), g1, g2


def _check_antecedents(g1, g2, x):
    """ Return a condition under which the integral theorem applies. """
    #  Yes, this is madness.
    # XXX TODO this is a testing *nightmare*
    # NOTE if you update these conditions, please update the documentation as well

    # The following conditions are found in
    # [P], Section 2.24.1
    #
    # They are also reproduced (verbatim!) at
    # https://functions.wolfram.com/HypergeometricFunctions/MeijerG/21/02/03/
    #
    # Note: k=l=r=alpha=1
    sigma, _ = _get_coeff_exp(g1.argument, x)
    omega, _ = _get_coeff_exp(g2.argument, x)
    s, t, u, v = S([len(g1.bm), len(g1.an), len(g1.ap), len(g1.bq)])
    m, n, p, q = S([len(g2.bm), len(g2.an), len(g2.ap), len(g2.bq)])
    bstar = s + t - (u + v)/2
    cstar = m + n - (p + q)/2
    rho = g1.nu + (u - v)/2 + 1
    mu = g2.nu + (p - q)/2 + 1
    phi = q - p - (v - u)
    eta = 1 - (v - u) - mu - rho
    psi = (pi*(q - m - n) + Abs(unbranched_argument(omega)))/(q - p)
    theta = (pi*(v - s - t) + Abs(unbranched_argument(sigma)))/(v - u)

    _debug('Checking antecedents:')
    _debugf('  sigma=%s, s=%s, t=%s, u=%s, v=%s, b*=%s, rho=%s',
            (sigma, s, t, u, v, bstar, rho))
    _debugf('  omega=%s, m=%s, n=%s, p=%s, q=%s, c*=%s, mu=%s,',
            (omega, m, n, p, q, cstar, mu))
    _debugf('  phi=%s, eta=%s, psi=%s, theta=%s', (phi, eta, psi, theta))

    def _c1():
        for g in [g1, g2]:
            for i, j in itertools.product(g.an, g.bm):
                diff = i - j
                if diff.is_integer and diff.is_positive:
                    return False
        return True
    c1 = _c1()
    c2 = And(*[re(1 + i + j) > 0 for i in g1.bm for j in g2.bm])
    c3 = And(*[re(1 + i + j) < 1 + 1 for i in g1.an for j in g2.an])
    c4 = And(*[(p - q)*re(1 + i - 1) - re(mu) > Rational(-3, 2) for i in g1.an])
    c5 = And(*[(p - q)*re(1 + i) - re(mu) > Rational(-3, 2) for i in g1.bm])
    c6 = And(*[(u - v)*re(1 + i - 1) - re(rho) > Rational(-3, 2) for i in g2.an])
    c7 = And(*[(u - v)*re(1 + i) - re(rho) > Rational(-3, 2) for i in g2.bm])
    c8 = (Abs(phi) + 2*re((rho - 1)*(q - p) + (v - u)*(q - p) + (mu -
          1)*(v - u)) > 0)
    c9 = (Abs(phi) - 2*re((rho - 1)*(q - p) + (v - u)*(q - p) + (mu -
          1)*(v - u)) > 0)
    c10 = (Abs(unbranched_argument(sigma)) < bstar*pi)
    c11 = Eq(Abs(unbranched_argument(sigma)), bstar*pi)
    c12 = (Abs(unbranched_argument(omega)) < cstar*pi)
    c13 = Eq(Abs(unbranched_argument(omega)), cstar*pi)

    # The following condition is *not* implemented as stated on the wolfram
    # function site. In the book of Prudnikov there is an additional part
    # (the And involving re()). However, I only have this book in russian, and
    # I don't read any russian. The following condition is what other people
    # have told me it means.
    # Worryingly, it is different from the condition implemented in REDUCE.
    # The REDUCE implementation:
    #   https://reduce-algebra.svn.sourceforge.net/svnroot/reduce-algebra/trunk/packages/defint/definta.red
    #   (search for tst14)
    # The Wolfram alpha version:
    #   https://functions.wolfram.com/HypergeometricFunctions/MeijerG/21/02/03/03/0014/
    z0 = exp(-(bstar + cstar)*pi*S.ImaginaryUnit)
    zos = unpolarify(z0*omega/sigma)
    zso = unpolarify(z0*sigma/omega)
    if zos == 1/zso:
        c14 = And(Eq(phi, 0), bstar + cstar <= 1,
                  Or(Ne(zos, 1), re(mu + rho + v - u) < 1,
                     re(mu + rho + q - p) < 1))
    else:
        def _cond(z):
            '''Returns True if abs(arg(1-z)) < pi, avoiding arg(0).

            Explanation
            ===========

            If ``z`` is 1 then arg is NaN. This raises a
            TypeError on `NaN < pi`. Previously this gave `False` so
            this behavior has been hardcoded here but someone should
            check if this NaN is more serious! This NaN is triggered by
            test_meijerint() in test_meijerint.py:
            `meijerint_definite(exp(x), x, 0, I)`
            '''
            return z != 1 and Abs(arg(1 - z)) < pi

        c14 = And(Eq(phi, 0), bstar - 1 + cstar <= 0,
                  Or(And(Ne(zos, 1), _cond(zos)),
                     And(re(mu + rho + v - u) < 1, Eq(zos, 1))))

        c14_alt = And(Eq(phi, 0), cstar - 1 + bstar <= 0,
                  Or(And(Ne(zso, 1), _cond(zso)),
                     And(re(mu + rho + q - p) < 1, Eq(zso, 1))))

        # Since r=k=l=1, in our case there is c14_alt which is the same as calling
        # us with (g1, g2) = (g2, g1). The conditions below enumerate all cases
        # (i.e. we don't have to try arguments reversed by hand), and indeed try
        # all symmetric cases. (i.e. whenever there is a condition involving c14,
        # there is also a dual condition which is exactly what we would get when g1,
        # g2 were interchanged, *but c14 was unaltered*).
        # Hence the following seems correct:
        c14 = Or(c14, c14_alt)

    '''
    When `c15` is NaN (e.g. from `psi` being NaN as happens during
    'test_issue_4992' and/or `theta` is NaN as in 'test_issue_6253',
    both in `test_integrals.py`) the comparison to 0 formerly gave False
    whereas now an error is raised. To keep the old behavior, the value
    of NaN is replaced with False but perhaps a closer look at this condition
    should be made: XXX how should conditions leading to c15=NaN be handled?
    '''
    try:
        lambda_c = (q - p)*Abs(omega)**(1/(q - p))*cos(psi) \
            + (v - u)*Abs(sigma)**(1/(v - u))*cos(theta)
        # the TypeError might be raised here, e.g. if lambda_c is NaN
        if _eval_cond(lambda_c > 0) != False:
            c15 = (lambda_c > 0)
        else:
            def lambda_s0(c1, c2):
                return c1*(q - p)*Abs(omega)**(1/(q - p))*sin(psi) \
                    + c2*(v - u)*Abs(sigma)**(1/(v - u))*sin(theta)
            lambda_s = Piecewise(
                ((lambda_s0(+1, +1)*lambda_s0(-1, -1)),
                 And(Eq(unbranched_argument(sigma), 0), Eq(unbranched_argument(omega), 0))),
                (lambda_s0(sign(unbranched_argument(omega)), +1)*lambda_s0(sign(unbranched_argument(omega)), -1),
                 And(Eq(unbranched_argument(sigma), 0), Ne(unbranched_argument(omega), 0))),
                (lambda_s0(+1, sign(unbranched_argument(sigma)))*lambda_s0(-1, sign(unbranched_argument(sigma))),
                 And(Ne(unbranched_argument(sigma), 0), Eq(unbranched_argument(omega), 0))),
                (lambda_s0(sign(unbranched_argument(omega)), sign(unbranched_argument(sigma))), True))
            tmp = [lambda_c > 0,
                   And(Eq(lambda_c, 0), Ne(lambda_s, 0), re(eta) > -1),
                   And(Eq(lambda_c, 0), Eq(lambda_s, 0), re(eta) > 0)]
            c15 = Or(*tmp)
    except TypeError:
        c15 = False
    for cond, i in [(c1, 1), (c2, 2), (c3, 3), (c4, 4), (c5, 5), (c6, 6),
                    (c7, 7), (c8, 8), (c9, 9), (c10, 10), (c11, 11),
                    (c12, 12), (c13, 13), (c14, 14), (c15, 15)]:
        _debugf('  c%s: %s', (i, cond))

    # We will return Or(*conds)
    conds = []

    def pr(count):
        _debugf('  case %s: %s', (count, conds[-1]))
    conds += [And(m*n*s*t != 0, bstar.is_positive is True, cstar.is_positive is True, c1, c2, c3, c10,
                  c12)]  # 1
    pr(1)
    conds += [And(Eq(u, v), Eq(bstar, 0), cstar.is_positive is True, sigma.is_positive is True, re(rho) < 1,
                  c1, c2, c3, c12)]  # 2
    pr(2)
    conds += [And(Eq(p, q), Eq(cstar, 0), bstar.is_positive is True, omega.is_positive is True, re(mu) < 1,
                  c1, c2, c3, c10)]  # 3
    pr(3)
    conds += [And(Eq(p, q), Eq(u, v), Eq(bstar, 0), Eq(cstar, 0),
                  sigma.is_positive is True, omega.is_positive is True, re(mu) < 1, re(rho) < 1,
                  Ne(sigma, omega), c1, c2, c3)]  # 4
    pr(4)
    conds += [And(Eq(p, q), Eq(u, v), Eq(bstar, 0), Eq(cstar, 0),
                  sigma.is_positive is True, omega.is_positive is True, re(mu + rho) < 1,
                  Ne(omega, sigma), c1, c2, c3)]  # 5
    pr(5)
    conds += [And(p > q, s.is_positive is True, bstar.is_positive is True, cstar >= 0,
                  c1, c2, c3, c5, c10, c13)]  # 6
    pr(6)
    conds += [And(p < q, t.is_positive is True, bstar.is_positive is True, cstar >= 0,
                  c1, c2, c3, c4, c10, c13)]  # 7
    pr(7)
    conds += [And(u > v, m.is_positive is True, cstar.is_positive is True, bstar >= 0,
                  c1, c2, c3, c7, c11, c12)]  # 8
    pr(8)
    conds += [And(u < v, n.is_positive is True, cstar.is_positive is True, bstar >= 0,
                  c1, c2, c3, c6, c11, c12)]  # 9
    pr(9)
    conds += [And(p > q, Eq(u, v), Eq(bstar, 0), cstar >= 0, sigma.is_positive is True,
                  re(rho) < 1, c1, c2, c3, c5, c13)]  # 10
    pr(10)
    conds += [And(p < q, Eq(u, v), Eq(bstar, 0), cstar >= 0, sigma.is_positive is True,
                  re(rho) < 1, c1, c2, c3, c4, c13)]  # 11
    pr(11)
    conds += [And(Eq(p, q), u > v, bstar >= 0, Eq(cstar, 0), omega.is_positive is True,
                  re(mu) < 1, c1, c2, c3, c7, c11)]  # 12
    pr(12)
    conds += [And(Eq(p, q), u < v, bstar >= 0, Eq(cstar, 0), omega.is_positive is True,
                  re(mu) < 1, c1, c2, c3, c6, c11)]  # 13
    pr(13)
    conds += [And(p < q, u > v, bstar >= 0, cstar >= 0,
                  c1, c2, c3, c4, c7, c11, c13)]  # 14
    pr(14)
    conds += [And(p > q, u < v, bstar >= 0, cstar >= 0,
                  c1, c2, c3, c5, c6, c11, c13)]  # 15
    pr(15)
    conds += [And(p > q, u > v, bstar >= 0, cstar >= 0,
                  c1, c2, c3, c5, c7, c8, c11, c13, c14)]  # 16
    pr(16)
    conds += [And(p < q, u < v, bstar >= 0, cstar >= 0,
                  c1, c2, c3, c4, c6, c9, c11, c13, c14)]  # 17
    pr(17)
    conds += [And(Eq(t, 0), s.is_positive is True, bstar.is_positive is True, phi.is_positive is True, c1, c2, c10)]  # 18
    pr(18)
    conds += [And(Eq(s, 0), t.is_positive is True, bstar.is_positive is True, phi.is_negative is True, c1, c3, c10)]  # 19
    pr(19)
    conds += [And(Eq(n, 0), m.is_positive is True, cstar.is_positive is True, phi.is_negative is True, c1, c2, c12)]  # 20
    pr(20)
    conds += [And(Eq(m, 0), n.is_positive is True, cstar.is_positive is True, phi.is_positive is True, c1, c3, c12)]  # 21
    pr(21)
    conds += [And(Eq(s*t, 0), bstar.is_positive is True, cstar.is_positive is True,
                  c1, c2, c3, c10, c12)]  # 22
    pr(22)
    conds += [And(Eq(m*n, 0), bstar.is_positive is True, cstar.is_positive is True,
                  c1, c2, c3, c10, c12)]  # 23
    pr(23)

    # The following case is from [Luke1969]. As far as I can tell, it is *not*
    # covered by Prudnikov's.
    # Let G1 and G2 be the two G-functions. Suppose the integral exists from
    # 0 to a > 0 (this is easy the easy part), that G1 is exponential decay at
    # infinity, and that the mellin transform of G2 exists.
    # Then the integral exists.
    mt1_exists = _check_antecedents_1(g1, x, helper=True)
    mt2_exists = _check_antecedents_1(g2, x, helper=True)
    conds += [And(mt2_exists, Eq(t, 0), u < s, bstar.is_positive is True, c10, c1, c2, c3)]
    pr('E1')
    conds += [And(mt2_exists, Eq(s, 0), v < t, bstar.is_positive is True, c10, c1, c2, c3)]
    pr('E2')
    conds += [And(mt1_exists, Eq(n, 0), p < m, cstar.is_positive is True, c12, c1, c2, c3)]
    pr('E3')
    conds += [And(mt1_exists, Eq(m, 0), q < n, cstar.is_positive is True, c12, c1, c2, c3)]
    pr('E4')

    # Let's short-circuit if this worked ...
    # the rest is corner-cases and terrible to read.
    r = Or(*conds)
    if _eval_cond(r) != False:
        return r

    conds += [And(m + n > p, Eq(t, 0), Eq(phi, 0), s.is_positive is True, bstar.is_positive is True, cstar.is_negative is True,
                  Abs(unbranched_argument(omega)) < (m + n - p + 1)*pi,
                  c1, c2, c10, c14, c15)]  # 24
    pr(24)
    conds += [And(m + n > q, Eq(s, 0), Eq(phi, 0), t.is_positive is True, bstar.is_positive is True, cstar.is_negative is True,
                  Abs(unbranched_argument(omega)) < (m + n - q + 1)*pi,
                  c1, c3, c10, c14, c15)]  # 25
    pr(25)
    conds += [And(Eq(p, q - 1), Eq(t, 0), Eq(phi, 0), s.is_positive is True, bstar.is_positive is True,
                  cstar >= 0, cstar*pi < Abs(unbranched_argument(omega)),
                  c1, c2, c10, c14, c15)]  # 26
    pr(26)
    conds += [And(Eq(p, q + 1), Eq(s, 0), Eq(phi, 0), t.is_positive is True, bstar.is_positive is True,
                  cstar >= 0, cstar*pi < Abs(unbranched_argument(omega)),
                  c1, c3, c10, c14, c15)]  # 27
    pr(27)
    conds += [And(p < q - 1, Eq(t, 0), Eq(phi, 0), s.is_positive is True, bstar.is_positive is True,
                  cstar >= 0, cstar*pi < Abs(unbranched_argument(omega)),
                  Abs(unbranched_argument(omega)) < (m + n - p + 1)*pi,
                  c1, c2, c10, c14, c15)]  # 28
    pr(28)
    conds += [And(
        p > q + 1, Eq(s, 0), Eq(phi, 0), t.is_positive is True, bstar.is_positive is True, cstar >= 0,
                  cstar*pi < Abs(unbranched_argument(omega)),
                  Abs(unbranched_argument(omega)) < (m + n - q + 1)*pi,
                  c1, c3, c10, c14, c15)]  # 29
    pr(29)
    conds += [And(Eq(n, 0), Eq(phi, 0), s + t > 0, m.is_positive is True, cstar.is_positive is True, bstar.is_negative is True,
                  Abs(unbranched_argument(sigma)) < (s + t - u + 1)*pi,
                  c1, c2, c12, c14, c15)]  # 30
    pr(30)
    conds += [And(Eq(m, 0), Eq(phi, 0), s + t > v, n.is_positive is True, cstar.is_positive is True, bstar.is_negative is True,
                  Abs(unbranched_argument(sigma)) < (s + t - v + 1)*pi,
                  c1, c3, c12, c14, c15)]  # 31
    pr(31)
    conds += [And(Eq(n, 0), Eq(phi, 0), Eq(u, v - 1), m.is_positive is True, cstar.is_positive is True,
                  bstar >= 0, bstar*pi < Abs(unbranched_argument(sigma)),
                  Abs(unbranched_argument(sigma)) < (bstar + 1)*pi,
                  c1, c2, c12, c14, c15)]  # 32
    pr(32)
    conds += [And(Eq(m, 0), Eq(phi, 0), Eq(u, v + 1), n.is_positive is True, cstar.is_positive is True,
                  bstar >= 0, bstar*pi < Abs(unbranched_argument(sigma)),
                  Abs(unbranched_argument(sigma)) < (bstar + 1)*pi,
                  c1, c3, c12, c14, c15)]  # 33
    pr(33)
    conds += [And(
        Eq(n, 0), Eq(phi, 0), u < v - 1, m.is_positive is True, cstar.is_positive is True, bstar >= 0,
        bstar*pi < Abs(unbranched_argument(sigma)),
        Abs(unbranched_argument(sigma)) < (s + t - u + 1)*pi,
        c1, c2, c12, c14, c15)]  # 34
    pr(34)
    conds += [And(
        Eq(m, 0), Eq(phi, 0), u > v + 1, n.is_positive is True, cstar.is_positive is True, bstar >= 0,
        bstar*pi < Abs(unbranched_argument(sigma)),
        Abs(unbranched_argument(sigma)) < (s + t - v + 1)*pi,
        c1, c3, c12, c14, c15)]  # 35
    pr(35)

    return Or(*conds)

    # NOTE An alternative, but as far as I can tell weaker, set of conditions
    #      can be found in [L, section 5.6.2].


def _int0oo(g1, g2, x):
    """
    Express integral from zero to infinity g1*g2 using a G function,
    assuming the necessary conditions are fulfilled.

    Examples
    ========

    >>> from sympy.integrals.meijerint import _int0oo
    >>> from sympy.abc import s, t, m
    >>> from sympy import meijerg, S
    >>> g1 = meijerg([], [], [-S(1)/2, 0], [], s**2*t/4)
    >>> g2 = meijerg([], [], [m/2], [-m/2], t/4)
    >>> _int0oo(g1, g2, t)
    4*meijerg(((0, 1/2), ()), ((m/2,), (-m/2,)), s**(-2))/s**2
    """
    # See: [L, section 5.6.2, equation (1)]
    eta, _ = _get_coeff_exp(g1.argument, x)
    omega, _ = _get_coeff_exp(g2.argument, x)

    def neg(l):
        return [-x for x in l]
    a1 = neg(g1.bm) + list(g2.an)
    a2 = list(g2.aother) + neg(g1.bother)
    b1 = neg(g1.an) + list(g2.bm)
    b2 = list(g2.bother) + neg(g1.aother)
    return meijerg(a1, a2, b1, b2, omega/eta)/eta


def _rewrite_inversion(fac, po, g, x):
    """ Absorb ``po`` == x**s into g. """
    _, s = _get_coeff_exp(po, x)
    a, b = _get_coeff_exp(g.argument, x)

    def tr(l):
        return [t + s/b for t in l]
    from sympy.simplify import powdenest
    return (powdenest(fac/a**(s/b), polar=True),
            meijerg(tr(g.an), tr(g.aother), tr(g.bm), tr(g.bother), g.argument))


def _check_antecedents_inversion(g, x):
    """ Check antecedents for the laplace inversion integral. """
    _debug('Checking antecedents for inversion:')
    z = g.argument
    _, e = _get_coeff_exp(z, x)
    if e < 0:
        _debug('  Flipping G.')
        # We want to assume that argument gets large as |x| -> oo
        return _check_antecedents_inversion(_flip_g(g), x)

    def statement_half(a, b, c, z, plus):
        coeff, exponent = _get_coeff_exp(z, x)
        a *= exponent
        b *= coeff**c
        c *= exponent
        conds = []
        wp = b*exp(S.ImaginaryUnit*re(c)*pi/2)
        wm = b*exp(-S.ImaginaryUnit*re(c)*pi/2)
        if plus:
            w = wp
        else:
            w = wm
        conds += [And(Or(Eq(b, 0), re(c) <= 0), re(a) <= -1)]
        conds += [And(Ne(b, 0), Eq(im(c), 0), re(c) > 0, re(w) < 0)]
        conds += [And(Ne(b, 0), Eq(im(c), 0), re(c) > 0, re(w) <= 0,
                      re(a) <= -1)]
        return Or(*conds)

    def statement(a, b, c, z):
        """ Provide a convergence statement for z**a * exp(b*z**c),
             c/f sphinx docs. """
        return And(statement_half(a, b, c, z, True),
                   statement_half(a, b, c, z, False))

    # Notations from [L], section 5.7-10
    m, n, p, q = S([len(g.bm), len(g.an), len(g.ap), len(g.bq)])
    tau = m + n - p
    nu = q - m - n
    rho = (tau - nu)/2
    sigma = q - p
    if sigma == 1:
        epsilon = S.Half
    elif sigma > 1:
        epsilon = 1
    else:
        epsilon = S.NaN
    theta = ((1 - sigma)/2 + Add(*g.bq) - Add(*g.ap))/sigma
    delta = g.delta
    _debugf('  m=%s, n=%s, p=%s, q=%s, tau=%s, nu=%s, rho=%s, sigma=%s',
            (m, n, p, q, tau, nu, rho, sigma))
    _debugf('  epsilon=%s, theta=%s, delta=%s', (epsilon, theta, delta))

    # First check if the computation is valid.
    if not (g.delta >= e/2 or (p >= 1 and p >= q)):
        _debug('  Computation not valid for these parameters.')
        return False

    # Now check if the inversion integral exists.

    # Test "condition A"
    for a, b in itertools.product(g.an, g.bm):
        if (a - b).is_integer and a > b:
            _debug('  Not a valid G function.')
            return False

    # There are two cases. If p >= q, we can directly use a slater expansion
    # like [L], 5.2 (11). Note in particular that the asymptotics of such an
    # expansion even hold when some of the parameters differ by integers, i.e.
    # the formula itself would not be valid! (b/c G functions are cts. in their
    # parameters)
    # When p < q, we need to use the theorems of [L], 5.10.

    if p >= q:
        _debug('  Using asymptotic Slater expansion.')
        return And(*[statement(a - 1, 0, 0, z) for a in g.an])

    def E(z):
        return And(*[statement(a - 1, 0, 0, z) for a in g.an])

    def H(z):
        return statement(theta, -sigma, 1/sigma, z)

    def Hp(z):
        return statement_half(theta, -sigma, 1/sigma, z, True)

    def Hm(z):
        return statement_half(theta, -sigma, 1/sigma, z, False)

    # [L], section 5.10
    conds = []
    # Theorem 1 -- p < q from test above
    conds += [And(1 <= n, 1 <= m, rho*pi - delta >= pi/2, delta > 0,
                  E(z*exp(S.ImaginaryUnit*pi*(nu + 1))))]
    # Theorem 2, statements (2) and (3)
    conds += [And(p + 1 <= m, m + 1 <= q, delta > 0, delta < pi/2, n == 0,
                  (m - p + 1)*pi - delta >= pi/2,
                  Hp(z*exp(S.ImaginaryUnit*pi*(q - m))),
                  Hm(z*exp(-S.ImaginaryUnit*pi*(q - m))))]
    # Theorem 2, statement (5)  -- p < q from test above
    conds += [And(m == q, n == 0, delta > 0,
                  (sigma + epsilon)*pi - delta >= pi/2, H(z))]
    # Theorem 3, statements (6) and (7)
    conds += [And(Or(And(p <= q - 2, 1 <= tau, tau <= sigma/2),
                     And(p + 1 <= m + n, m + n <= (p + q)/2)),
                  delta > 0, delta < pi/2, (tau + 1)*pi - delta >= pi/2,
                  Hp(z*exp(S.ImaginaryUnit*pi*nu)),
                  Hm(z*exp(-S.ImaginaryUnit*pi*nu)))]
    # Theorem 4, statements (10) and (11)  -- p < q from test above
    conds += [And(1 <= m, rho > 0, delta > 0, delta + rho*pi < pi/2,
                  (tau + epsilon)*pi - delta >= pi/2,
                  Hp(z*exp(S.ImaginaryUnit*pi*nu)),
                  Hm(z*exp(-S.ImaginaryUnit*pi*nu)))]
    # Trivial case
    conds += [m == 0]

    # TODO
    # Theorem 5 is quite general
    # Theorem 6 contains special cases for q=p+1

    return Or(*conds)


def _int_inversion(g, x, t):
    """
    Compute the laplace inversion integral, assuming the formula applies.
    """
    b, a = _get_coeff_exp(g.argument, x)
    C, g = _inflate_fox_h(meijerg(g.an, g.aother, g.bm, g.bother, b/t**a), -a)
    return C/t*g


####################################################################
# Finally, the real meat.
####################################################################

_lookup_table = None


@cacheit
@timeit
def _rewrite_single(f, x, recursive=True):
    """
    Try to rewrite f as a sum of single G functions of the form
    C*x**s*G(a*x**b), where b is a rational number and C is independent of x.
    We guarantee that result.argument.as_coeff_mul(x) returns (a, (x**b,))
    or (a, ()).
    Returns a list of tuples (C, s, G) and a condition cond.
    Returns None on failure.
    """
    from .transforms import (mellin_transform, inverse_mellin_transform,
        IntegralTransformError, MellinTransformStripError)

    global _lookup_table
    if not _lookup_table:
        _lookup_table = {}
        _create_lookup_table(_lookup_table)

    if isinstance(f, meijerg):
        coeff, m = factor(f.argument, x).as_coeff_mul(x)
        if len(m) > 1:
            return None
        m = m[0]
        if m.is_Pow:
            if m.base != x or not m.exp.is_Rational:
                return None
        elif m != x:
            return None
        return [(1, 0, meijerg(f.an, f.aother, f.bm, f.bother, coeff*m))], True

    f_ = f
    f = f.subs(x, z)
    t = _mytype(f, z)
    if t in _lookup_table:
        l = _lookup_table[t]
        for formula, terms, cond, hint in l:
            subs = f.match(formula, old=True)
            if subs:
                subs_ = {}
                for fro, to in subs.items():
                    subs_[fro] = unpolarify(polarify(to, lift=True),
                                            exponents_only=True)
                subs = subs_
                if not isinstance(hint, bool):
                    hint = hint.subs(subs)
                if hint == False:
                    continue
                if not isinstance(cond, (bool, BooleanAtom)):
                    cond = unpolarify(cond.subs(subs))
                if _eval_cond(cond) == False:
                    continue
                if not isinstance(terms, list):
                    terms = terms(subs)
                res = []
                for fac, g in terms:
                    r1 = _get_coeff_exp(unpolarify(fac.subs(subs).subs(z, x),
                                                   exponents_only=True), x)
                    try:
                        g = g.subs(subs).subs(z, x)
                    except ValueError:
                        continue
                    # NOTE these substitutions can in principle introduce oo,
                    #      zoo and other absurdities. It shouldn't matter,
                    #      but better be safe.
                    if Tuple(*(r1 + (g,))).has(S.Infinity, S.ComplexInfinity, S.NegativeInfinity):
                        continue
                    g = meijerg(g.an, g.aother, g.bm, g.bother,
                                unpolarify(g.argument, exponents_only=True))
                    res.append(r1 + (g,))
                if res:
                    return res, cond

    # try recursive mellin transform
    if not recursive:
        return None
    _debug('Trying recursive Mellin transform method.')

    def my_imt(F, s, x, strip):
        """ Calling simplify() all the time is slow and not helpful, since
            most of the time it only factors things in a way that has to be
            un-done anyway. But sometimes it can remove apparent poles. """
        # XXX should this be in inverse_mellin_transform?
        try:
            return inverse_mellin_transform(F, s, x, strip,
                                            as_meijerg=True, needeval=True)
        except MellinTransformStripError:
            from sympy.simplify import simplify
            return inverse_mellin_transform(
                simplify(cancel(expand(F))), s, x, strip,
                as_meijerg=True, needeval=True)
    f = f_
    s = _dummy('s', 'rewrite-single', f)
    # to avoid infinite recursion, we have to force the two g functions case

    def my_integrator(f, x):
        r = _meijerint_definite_4(f, x, only_double=True)
        if r is not None:
            from sympy.simplify import hyperexpand
            res, cond = r
            res = _my_unpolarify(hyperexpand(res, rewrite='nonrepsmall'))
            return Piecewise((res, cond),
                             (Integral(f, (x, S.Zero, S.Infinity)), True))
        return Integral(f, (x, S.Zero, S.Infinity))
    try:
        F, strip, _ = mellin_transform(f, x, s, integrator=my_integrator,
                                       simplify=False, needeval=True)
        g = my_imt(F, s, x, strip)
    except IntegralTransformError:
        g = None
    if g is None:
        # We try to find an expression by analytic continuation.
        # (also if the dummy is already in the expression, there is no point in
        #  putting in another one)
        a = _dummy_('a', 'rewrite-single')
        if a not in f.free_symbols and _is_analytic(f, x):
            try:
                F, strip, _ = mellin_transform(f.subs(x, a*x), x, s,
                                               integrator=my_integrator,
                                               needeval=True, simplify=False)
                g = my_imt(F, s, x, strip).subs(a, 1)
            except IntegralTransformError:
                g = None
    if g is None or g.has(S.Infinity, S.NaN, S.ComplexInfinity):
        _debug('Recursive Mellin transform failed.')
        return None
    args = Add.make_args(g)
    res = []
    for f in args:
        c, m = f.as_coeff_mul(x)
        if len(m) > 1:
            raise NotImplementedError('Unexpected form...')
        g = m[0]
        a, b = _get_coeff_exp(g.argument, x)
        res += [(c, 0, meijerg(g.an, g.aother, g.bm, g.bother,
                               unpolarify(polarify(
                                   a, lift=True), exponents_only=True)
                               *x**b))]
    _debug('Recursive Mellin transform worked:', g)
    return res, True


def _rewrite1(f, x, recursive=True):
    """
    Try to rewrite ``f`` using a (sum of) single G functions with argument a*x**b.
    Return fac, po, g such that f = fac*po*g, fac is independent of ``x``.
    and po = x**s.
    Here g is a result from _rewrite_single.
    Return None on failure.
    """
    fac, po, g = _split_mul(f, x)
    g = _rewrite_single(g, x, recursive)
    if g:
        return fac, po, g[0], g[1]


def _rewrite2(f, x):
    """
    Try to rewrite ``f`` as a product of two G functions of arguments a*x**b.
    Return fac, po, g1, g2 such that f = fac*po*g1*g2, where fac is
    independent of x and po is x**s.
    Here g1 and g2 are results of _rewrite_single.
    Returns None on failure.
    """
    fac, po, g = _split_mul(f, x)
    if any(_rewrite_single(expr, x, False) is None for expr in _mul_args(g)):
        return None
    l = _mul_as_two_parts(g)
    if not l:
        return None
    l = list(ordered(l, [
        lambda p: max(len(_exponents(p[0], x)), len(_exponents(p[1], x))),
        lambda p: max(len(_functions(p[0], x)), len(_functions(p[1], x))),
        lambda p: max(len(_find_splitting_points(p[0], x)),
                      len(_find_splitting_points(p[1], x)))]))

    for recursive, (fac1, fac2) in itertools.product((False, True), l):
        g1 = _rewrite_single(fac1, x, recursive)
        g2 = _rewrite_single(fac2, x, recursive)
        if g1 and g2:
            cond = And(g1[1], g2[1])
            if cond != False:
                return fac, po, g1[0], g2[0], cond


def meijerint_indefinite(f, x):
    """
    Compute an indefinite integral of ``f`` by rewriting it as a G function.

    Examples
    ========

    >>> from sympy.integrals.meijerint import meijerint_indefinite
    >>> from sympy import sin
    >>> from sympy.abc import x
    >>> meijerint_indefinite(sin(x), x)
    -cos(x)
    """
    f = sympify(f)
    results = []
    for a in sorted(_find_splitting_points(f, x) | {S.Zero}, key=default_sort_key):
        res = _meijerint_indefinite_1(f.subs(x, x + a), x)
        if not res:
            continue
        res = res.subs(x, x - a)
        if _has(res, hyper, meijerg):
            results.append(res)
        else:
            return res
    if f.has(HyperbolicFunction):
        _debug('Try rewriting hyperbolics in terms of exp.')
        rv = meijerint_indefinite(
            _rewrite_hyperbolics_as_exp(f), x)
        if rv:
            if not isinstance(rv, list):
                from sympy.simplify.radsimp import collect
                return collect(factor_terms(rv), rv.atoms(exp))
            results.extend(rv)
    if results:
        return next(ordered(results))


def _meijerint_indefinite_1(f, x):
    """ Helper that does not attempt any substitution. """
    _debug('Trying to compute the indefinite integral of', f, 'wrt', x)
    from sympy.simplify import hyperexpand, powdenest

    gs = _rewrite1(f, x)
    if gs is None:
        # Note: the code that calls us will do expand() and try again
        return None

    fac, po, gl, cond = gs
    _debug(' could rewrite:', gs)
    res = S.Zero
    for C, s, g in gl:
        a, b = _get_coeff_exp(g.argument, x)
        _, c = _get_coeff_exp(po, x)
        c += s

        # we do a substitution t=a*x**b, get integrand fac*t**rho*g
        fac_ = fac * C * x**(1 + c) / b
        rho = (c + 1)/b

        # we now use t**rho*G(params, t) = G(params + rho, t)
        # [L, page 150, equation (4)]
        # and integral G(params, t) dt = G(1, params+1, 0, t)
        #   (or a similar expression with 1 and 0 exchanged ... pick the one
        #    which yields a well-defined function)
        # [R, section 5]
        # (Note that this dummy will immediately go away again, so we
        #  can safely pass S.One for ``expr``.)
        t = _dummy('t', 'meijerint-indefinite', S.One)

        def tr(p):
            return [a + rho for a in p]
        if any(b.is_integer and (b <= 0) == True for b in tr(g.bm)):
            r = -meijerg(
                list(g.an), list(g.aother) + [1-rho], list(g.bm) + [-rho], list(g.bother), t)
        else:
            r = meijerg(
                list(g.an) + [1-rho], list(g.aother), list(g.bm), list(g.bother) + [-rho], t)
        # The antiderivative is most often expected to be defined
        # in the neighborhood of  x = 0.
        if b.is_extended_nonnegative and not f.subs(x, 0).has(S.NaN, S.ComplexInfinity):
            place = 0  # Assume we can expand at zero
        else:
            place = None
        r = hyperexpand(r.subs(t, a*x**b), place=place)

        # now substitute back
        # Note: we really do want the powers of x to combine.
        res += powdenest(fac_*r, polar=True)

    def _clean(res):
        """This multiplies out superfluous powers of x we created, and chops off
        constants:

            >> _clean(x*(exp(x)/x - 1/x) + 3)
            exp(x)

        cancel is used before mul_expand since it is possible for an
        expression to have an additive constant that does not become isolated
        with simple expansion. Such a situation was identified in issue 6369:

        Examples
        ========

        >>> from sympy import sqrt, cancel
        >>> from sympy.abc import x
        >>> a = sqrt(2*x + 1)
        >>> bad = (3*x*a**5 + 2*x - a**5 + 1)/a**2
        >>> bad.expand().as_independent(x)[0]
        0
        >>> cancel(bad).expand().as_independent(x)[0]
        1
        """
        res = expand_mul(cancel(res), deep=False)
        return Add._from_args(res.as_coeff_add(x)[1])

    res = piecewise_fold(res, evaluate=None)
    if res.is_Piecewise:
        newargs = []
        for e, c in res.args:
            e = _my_unpolarify(_clean(e))
            newargs += [(e, c)]
        res = Piecewise(*newargs, evaluate=False)
    else:
        res = _my_unpolarify(_clean(res))
    return Piecewise((res, _my_unpolarify(cond)), (Integral(f, x), True))


@timeit
def meijerint_definite(f, x, a, b):
    """
    Integrate ``f`` over the interval [``a``, ``b``], by rewriting it as a product
    of two G functions, or as a single G function.

    Return res, cond, where cond are convergence conditions.

    Examples
    ========

    >>> from sympy.integrals.meijerint import meijerint_definite
    >>> from sympy import exp, oo
    >>> from sympy.abc import x
    >>> meijerint_definite(exp(-x**2), x, -oo, oo)
    (sqrt(pi), True)

    This function is implemented as a succession of functions
    meijerint_definite, _meijerint_definite_2, _meijerint_definite_3,
    _meijerint_definite_4. Each function in the list calls the next one
    (presumably) several times. This means that calling meijerint_definite
    can be very costly.
    """
    # This consists of three steps:
    # 1) Change the integration limits to 0, oo
    # 2) Rewrite in terms of G functions
    # 3) Evaluate the integral
    #
    # There are usually several ways of doing this, and we want to try all.
    # This function does (1), calls _meijerint_definite_2 for step (2).
    _debugf('Integrating %s wrt %s from %s to %s.', (f, x, a, b))
    f = sympify(f)
    if f.has(DiracDelta):
        _debug('Integrand has DiracDelta terms - giving up.')
        return None

    if f.has(SingularityFunction):
        _debug('Integrand has Singularity Function terms - giving up.')
        return None

    f_, x_, a_, b_ = f, x, a, b

    # Let's use a dummy in case any of the boundaries has x.
    d = Dummy('x')
    f = f.subs(x, d)
    x = d

    if a == b:
        return (S.Zero, True)

    results = []
    if a is S.NegativeInfinity and b is not S.Infinity:
        return meijerint_definite(f.subs(x, -x), x, -b, -a)

    elif a is S.NegativeInfinity:
        # Integrating -oo to oo. We need to find a place to split the integral.
        _debug('  Integrating -oo to +oo.')
        innermost = _find_splitting_points(f, x)
        _debug('  Sensible splitting points:', innermost)
        for c in sorted(innermost, key=default_sort_key, reverse=True) + [S.Zero]:
            _debug('  Trying to split at', c)
            if not c.is_extended_real:
                _debug('  Non-real splitting point.')
                continue
            res1 = _meijerint_definite_2(f.subs(x, x + c), x)
            if res1 is None:
                _debug('  But could not compute first integral.')
                continue
            res2 = _meijerint_definite_2(f.subs(x, c - x), x)
            if res2 is None:
                _debug('  But could not compute second integral.')
                continue
            res1, cond1 = res1
            res2, cond2 = res2
            cond = _condsimp(And(cond1, cond2))
            if cond == False:
                _debug('  But combined condition is always false.')
                continue
            res = res1 + res2
            return res, cond

    elif a is S.Infinity:
        res = meijerint_definite(f, x, b, S.Infinity)
        return -res[0], res[1]

    elif (a, b) == (S.Zero, S.Infinity):
        # This is a common case - try it directly first.
        res = _meijerint_definite_2(f, x)
        if res:
            if _has(res[0], meijerg):
                results.append(res)
            else:
                return res

    else:
        if b is S.Infinity:
            for split in _find_splitting_points(f, x):
                if (a - split >= 0) == True:
                    _debugf('Trying x -> x + %s', split)
                    res = _meijerint_definite_2(f.subs(x, x + split)
                                                *Heaviside(x + split - a), x)
                    if res:
                        if _has(res[0], meijerg):
                            results.append(res)
                        else:
                            return res

        f = f.subs(x, x + a)
        b = b - a
        a = 0
        if b is not S.Infinity:
            phi = exp(S.ImaginaryUnit*arg(b))
            b = Abs(b)
            f = f.subs(x, phi*x)
            f *= Heaviside(b - x)*phi
            b = S.Infinity

        _debug('Changed limits to', a, b)
        _debug('Changed function to', f)
        res = _meijerint_definite_2(f, x)
        if res:
            if _has(res[0], meijerg):
                results.append(res)
            else:
                return res
    if f_.has(HyperbolicFunction):
        _debug('Try rewriting hyperbolics in terms of exp.')
        rv = meijerint_definite(
            _rewrite_hyperbolics_as_exp(f_), x_, a_, b_)
        if rv:
            if not isinstance(rv, list):
                from sympy.simplify.radsimp import collect
                rv = (collect(factor_terms(rv[0]), rv[0].atoms(exp)),) + rv[1:]
                return rv
            results.extend(rv)
    if results:
        return next(ordered(results))


def _guess_expansion(f, x):
    """ Try to guess sensible rewritings for integrand f(x). """
    res = [(f, 'original integrand')]

    orig = res[-1][0]
    saw = {orig}
    expanded = expand_mul(orig)
    if expanded not in saw:
        res += [(expanded, 'expand_mul')]
        saw.add(expanded)

    expanded = expand(orig)
    if expanded not in saw:
        res += [(expanded, 'expand')]
        saw.add(expanded)

    if orig.has(TrigonometricFunction, HyperbolicFunction):
        expanded = expand_mul(expand_trig(orig))
        if expanded not in saw:
            res += [(expanded, 'expand_trig, expand_mul')]
            saw.add(expanded)

    if orig.has(cos, sin):
        from sympy.simplify.fu import sincos_to_sum
        reduced = sincos_to_sum(orig)
        if reduced not in saw:
            res += [(reduced, 'trig power reduction')]
            saw.add(reduced)

    return res


def _meijerint_definite_2(f, x):
    """
    Try to integrate f dx from zero to infinity.

    The body of this function computes various 'simplifications'
    f1, f2, ... of f (e.g. by calling expand_mul(), trigexpand()
    - see _guess_expansion) and calls _meijerint_definite_3 with each of
    these in succession.
    If _meijerint_definite_3 succeeds with any of the simplified functions,
    returns this result.
    """
    # This function does preparation for (2), calls
    # _meijerint_definite_3 for (2) and (3) combined.

    # use a positive dummy - we integrate from 0 to oo
    # XXX if a nonnegative symbol is used there will be test failures
    dummy = _dummy('x', 'meijerint-definite2', f, positive=True)
    f = f.subs(x, dummy)
    x = dummy

    if f == 0:
        return S.Zero, True

    for g, explanation in _guess_expansion(f, x):
        _debug('Trying', explanation)
        res = _meijerint_definite_3(g, x)
        if res:
            return res


def _meijerint_definite_3(f, x):
    """
    Try to integrate f dx from zero to infinity.

    This function calls _meijerint_definite_4 to try to compute the
    integral. If this fails, it tries using linearity.
    """
    res = _meijerint_definite_4(f, x)
    if res and res[1] != False:
        return res
    if f.is_Add:
        _debug('Expanding and evaluating all terms.')
        ress = [_meijerint_definite_4(g, x) for g in f.args]
        if all(r is not None for r in ress):
            conds = []
            res = S.Zero
            for r, c in ress:
                res += r
                conds += [c]
            c = And(*conds)
            if c != False:
                return res, c


def _my_unpolarify(f):
    return _eval_cond(unpolarify(f))


@timeit
def _meijerint_definite_4(f, x, only_double=False):
    """
    Try to integrate f dx from zero to infinity.

    Explanation
    ===========

    This function tries to apply the integration theorems found in literature,
    i.e. it tries to rewrite f as either one or a product of two G-functions.

    The parameter ``only_double`` is used internally in the recursive algorithm
    to disable trying to rewrite f as a single G-function.
    """
    from sympy.simplify import hyperexpand
    # This function does (2) and (3)
    _debug('Integrating', f)
    # Try single G function.
    if not only_double:
        gs = _rewrite1(f, x, recursive=False)
        if gs is not None:
            fac, po, g, cond = gs
            _debug('Could rewrite as single G function:', fac, po, g)
            res = S.Zero
            for C, s, f in g:
                if C == 0:
                    continue
                C, f = _rewrite_saxena_1(fac*C, po*x**s, f, x)
                res += C*_int0oo_1(f, x)
                cond = And(cond, _check_antecedents_1(f, x))
                if cond == False:
                    break
            cond = _my_unpolarify(cond)
            if cond == False:
                _debug('But cond is always False.')
            else:
                _debug('Result before branch substitutions is:', res)
                return _my_unpolarify(hyperexpand(res)), cond

    # Try two G functions.
    gs = _rewrite2(f, x)
    if gs is not None:
        for full_pb in [False, True]:
            fac, po, g1, g2, cond = gs
            _debug('Could rewrite as two G functions:', fac, po, g1, g2)
            res = S.Zero
            for C1, s1, f1 in g1:
                for C2, s2, f2 in g2:
                    r = _rewrite_saxena(fac*C1*C2, po*x**(s1 + s2),
                                        f1, f2, x, full_pb)
                    if r is None:
                        _debug('Non-rational exponents.')
                        return
                    C, f1_, f2_ = r
                    _debug('Saxena subst for yielded:', C, f1_, f2_)
                    cond = And(cond, _check_antecedents(f1_, f2_, x))
                    if cond == False:
                        break
                    res += C*_int0oo(f1_, f2_, x)
                else:
                    continue
                break
            cond = _my_unpolarify(cond)
            if cond == False:
                _debugf('But cond is always False (full_pb=%s).', full_pb)
            else:
                _debugf('Result before branch substitutions is: %s', (res, ))
                if only_double:
                    return res, cond
                return _my_unpolarify(hyperexpand(res)), cond


def meijerint_inversion(f, x, t):
    r"""
    Compute the inverse laplace transform
    $\int_{c+i\infty}^{c-i\infty} f(x) e^{tx}\, dx$,
    for real c larger than the real part of all singularities of ``f``.

    Note that ``t`` is always assumed real and positive.

    Return None if the integral does not exist or could not be evaluated.

    Examples
    ========

    >>> from sympy.abc import x, t
    >>> from sympy.integrals.meijerint import meijerint_inversion
    >>> meijerint_inversion(1/x, x, t)
    Heaviside(t)
    """
    f_ = f
    t_ = t
    t = Dummy('t', polar=True)  # We don't want sqrt(t**2) = abs(t) etc
    f = f.subs(t_, t)
    _debug('Laplace-inverting', f)
    if not _is_analytic(f, x):
        _debug('But expression is not analytic.')
        return None
    # Exponentials correspond to shifts; we filter them out and then
    # shift the result later.  If we are given an Add this will not
    # work, but the calling code will take care of that.
    shift = S.Zero

    if f.is_Mul:
        args = list(f.args)
    elif isinstance(f, exp):
        args = [f]
    else:
        args = None

    if args:
        newargs = []
        exponentials = []
        while args:
            arg = args.pop()
            if isinstance(arg, exp):
                arg2 = expand(arg)
                if arg2.is_Mul:
                    args += arg2.args
                    continue
                try:
                    a, b = _get_coeff_exp(arg.args[0], x)
                except _CoeffExpValueError:
                    b = 0
                if b == 1:
                    exponentials.append(a)
                else:
                    newargs.append(arg)
            elif arg.is_Pow:
                arg2 = expand(arg)
                if arg2.is_Mul:
                    args += arg2.args
                    continue
                if x not in arg.base.free_symbols:
                    try:
                        a, b = _get_coeff_exp(arg.exp, x)
                    except _CoeffExpValueError:
                        b = 0
                    if b == 1:
                        exponentials.append(a*log(arg.base))
                newargs.append(arg)
            else:
                newargs.append(arg)
        shift = Add(*exponentials)
        f = Mul(*newargs)

    if x not in f.free_symbols:
        _debug('Expression consists of constant and exp shift:', f, shift)
        cond = Eq(im(shift), 0)
        if cond == False:
            _debug('but shift is nonreal, cannot be a Laplace transform')
            return None
        res = f*DiracDelta(t + shift)
        _debug('Result is a delta function, possibly conditional:', res, cond)
        # cond is True or Eq
        return Piecewise((res.subs(t, t_), cond))

    gs = _rewrite1(f, x)
    if gs is not None:
        fac, po, g, cond = gs
        _debug('Could rewrite as single G function:', fac, po, g)
        res = S.Zero
        for C, s, f in g:
            C, f = _rewrite_inversion(fac*C, po*x**s, f, x)
            res += C*_int_inversion(f, x, t)
            cond = And(cond, _check_antecedents_inversion(f, x))
            if cond == False:
                break
        cond = _my_unpolarify(cond)
        if cond == False:
            _debug('But cond is always False.')
        else:
            _debug('Result before branch substitution:', res)
            from sympy.simplify import hyperexpand
            res = _my_unpolarify(hyperexpand(res))
            if not res.has(Heaviside):
                res *= Heaviside(t)
            res = res.subs(t, t + shift)
            if not isinstance(cond, bool):
                cond = cond.subs(t, t + shift)
            from .transforms import InverseLaplaceTransform
            return Piecewise((res.subs(t, t_), cond),
                             (InverseLaplaceTransform(f_.subs(t, t_), x, t_, None), True))
