from math import isclose

from sympy.calculus.util import stationary_points
from sympy.core.containers import Tuple
from sympy.core.function import (Function, Lambda, nfloat, diff)
from sympy.core.mod import Mod
from sympy.core.numbers import (E, I, Rational, oo, pi, Integer, all_close)
from sympy.core.relational import (Eq, Gt, Ne, Ge)
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import (Abs, arg, im, re, sign, conjugate)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (HyperbolicFunction,
    sinh, cosh, tanh, coth, sech, csch, asinh, acosh, atanh, acoth, asech, acsch)
from sympy.functions.elementary.miscellaneous import sqrt, Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (
    TrigonometricFunction, acos, acot, acsc, asec, asin, atan, atan2,
    cos, cot, csc, sec, sin, tan)
from sympy.functions.special.error_functions import (erf, erfc,
    erfcinv, erfinv)
from sympy.logic.boolalg import And
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.matrices.immutable import ImmutableDenseMatrix
from sympy.polys.polytools import Poly
from sympy.polys.rootoftools import CRootOf
from sympy.sets.contains import Contains
from sympy.sets.conditionset import ConditionSet
from sympy.sets.fancysets import ImageSet, Range
from sympy.sets.sets import (Complement, FiniteSet,
    Intersection, Interval, Union, imageset, ProductSet)
from sympy.simplify import simplify
from sympy.tensor.indexed import Indexed
from sympy.utilities.iterables import numbered_symbols

from sympy.testing.pytest import (XFAIL, raises, skip, slow, SKIP, _both_exp_pow)
from sympy.core.random import verify_numerically as tn
from sympy.physics.units import cm

from sympy.solvers import solve
from sympy.solvers.solveset import (
    solveset_real, domain_check, solveset_complex, linear_eq_to_matrix,
    linsolve, _is_function_class_equation, invert_real, invert_complex,
    _invert_trig_hyp_real, solveset, solve_decomposition, substitution,
    nonlinsolve, solvify,
    _is_finite_with_finite_vars, _transolve, _is_exponential,
    _solve_exponential, _is_logarithmic, _is_lambert,
    _solve_logarithm, _term_factors, _is_modular, NonlinearError)

from sympy.abc import (a, b, c, d, e, f, g, h, i, j, k, l, m, n, q, r,
    t, w, x, y, z)


def dumeq(i, j):
    if type(i) in (list, tuple):
        return all(dumeq(i, j) for i, j in zip(i, j))
    return i == j or i.dummy_eq(j)


def assert_close_ss(sol1, sol2):
    """Test solutions with floats from solveset are close"""
    sol1 = sympify(sol1)
    sol2 = sympify(sol2)
    assert isinstance(sol1, FiniteSet)
    assert isinstance(sol2, FiniteSet)
    assert len(sol1) == len(sol2)
    assert all(isclose(v1, v2) for v1, v2 in zip(sol1, sol2))


def assert_close_nl(sol1, sol2):
    """Test solutions with floats from nonlinsolve are close"""
    sol1 = sympify(sol1)
    sol2 = sympify(sol2)
    assert isinstance(sol1, FiniteSet)
    assert isinstance(sol2, FiniteSet)
    assert len(sol1) == len(sol2)
    for s1, s2 in zip(sol1, sol2):
        assert len(s1) == len(s2)
        assert all(isclose(v1, v2) for v1, v2 in zip(s1, s2))


@_both_exp_pow
def test_invert_real():
    x = Symbol('x', real=True)

    def ireal(x, s=S.Reals):
        return Intersection(s, x)

    assert invert_real(exp(x), z, x) == (x, ireal(FiniteSet(log(z))))

    y = Symbol('y', positive=True)
    n = Symbol('n', real=True)
    assert invert_real(x + 3, y, x) == (x, FiniteSet(y - 3))
    assert invert_real(x*3, y, x) == (x, FiniteSet(y / 3))

    assert invert_real(exp(x), y, x) == (x, FiniteSet(log(y)))
    assert invert_real(exp(3*x), y, x) == (x, FiniteSet(log(y) / 3))
    assert invert_real(exp(x + 3), y, x) == (x, FiniteSet(log(y) - 3))

    assert invert_real(exp(x) + 3, y, x) == (x, ireal(FiniteSet(log(y - 3))))
    assert invert_real(exp(x)*3, y, x) == (x, FiniteSet(log(y / 3)))

    assert invert_real(log(x), y, x) == (x, FiniteSet(exp(y)))
    assert invert_real(log(3*x), y, x) == (x, FiniteSet(exp(y) / 3))
    assert invert_real(log(x + 3), y, x) == (x, FiniteSet(exp(y) - 3))

    assert invert_real(Abs(x), y, x) == (x, FiniteSet(y, -y))

    assert invert_real(2**x, y, x) == (x, FiniteSet(log(y)/log(2)))
    assert invert_real(2**exp(x), y, x) == (x, ireal(FiniteSet(log(log(y)/log(2)))))

    assert invert_real(x**2, y, x) == (x, FiniteSet(sqrt(y), -sqrt(y)))
    assert invert_real(x**S.Half, y, x) == (x, FiniteSet(y**2))

    raises(ValueError, lambda: invert_real(x, x, x))

    # issue 21236
    assert invert_real(x**pi, y, x) == (x, FiniteSet(y**(1/pi)))
    assert invert_real(x**pi, -E, x) == (x, S.EmptySet)
    assert invert_real(x**Rational(3/2), 1000, x) == (x, FiniteSet(100))
    assert invert_real(x**1.0, 1, x) == (x**1.0, FiniteSet(1))

    raises(ValueError, lambda: invert_real(S.One, y, x))

    assert invert_real(x**31 + x, y, x) == (x**31 + x, FiniteSet(y))

    lhs = x**31 + x
    base_values =  FiniteSet(y - 1, -y - 1)
    assert invert_real(Abs(x**31 + x + 1), y, x) == (lhs, base_values)

    assert dumeq(invert_real(sin(x), y, x), (x,
        ConditionSet(x, (S(-1) <= y) & (y <= S(1)), Union(
            ImageSet(Lambda(n, 2*n*pi + asin(y)), S.Integers),
            ImageSet(Lambda(n, pi*2*n + pi - asin(y)), S.Integers)))))

    assert dumeq(invert_real(sin(exp(x)), y, x), (x,
        ConditionSet(x, (S(-1) <= y) & (y <= S(1)), Union(
            ImageSet(Lambda(n, log(2*n*pi + asin(y))), S.Integers),
            ImageSet(Lambda(n, log(pi*2*n + pi - asin(y))), S.Integers)))))

    assert dumeq(invert_real(csc(x), y, x), (x,
        ConditionSet(x, ((S(1) <= y) & (y < oo)) | ((-oo < y) & (y <= S(-1))),
            Union(ImageSet(Lambda(n, 2*n*pi + acsc(y)), S.Integers),
                ImageSet(Lambda(n, 2*n*pi - acsc(y) + pi), S.Integers)))))

    assert dumeq(invert_real(csc(exp(x)), y, x), (x,
        ConditionSet(x, ((S(1) <= y) & (y < oo)) | ((-oo < y) & (y <= S(-1))),
            Union(ImageSet(Lambda(n, log(2*n*pi + acsc(y))), S.Integers),
                ImageSet(Lambda(n, log(2*n*pi - acsc(y) + pi)), S.Integers)))))

    assert dumeq(invert_real(cos(x), y, x), (x,
        ConditionSet(x, (S(-1) <= y) & (y <= S(1)), Union(
            ImageSet(Lambda(n, 2*n*pi + acos(y)), S.Integers),
            ImageSet(Lambda(n, 2*n*pi - acos(y)), S.Integers)))))

    assert dumeq(invert_real(cos(exp(x)), y, x), (x,
        ConditionSet(x, (S(-1) <= y) & (y <= S(1)), Union(
            ImageSet(Lambda(n, log(2*n*pi + acos(y))), S.Integers),
            ImageSet(Lambda(n, log(2*n*pi - acos(y))), S.Integers)))))

    assert dumeq(invert_real(sec(x), y, x), (x,
        ConditionSet(x, ((S(1) <= y) & (y < oo)) | ((-oo < y) & (y <= S(-1))),
            Union(ImageSet(Lambda(n, 2*n*pi + asec(y)), S.Integers), \
                ImageSet(Lambda(n, 2*n*pi - asec(y)), S.Integers)))))

    assert dumeq(invert_real(sec(exp(x)), y, x), (x,
        ConditionSet(x, ((S(1) <= y) & (y < oo)) | ((-oo < y) & (y <= S(-1))),
            Union(ImageSet(Lambda(n, log(2*n*pi - asec(y))), S.Integers),
                ImageSet(Lambda(n, log(2*n*pi + asec(y))), S.Integers)))))

    assert dumeq(invert_real(tan(x), y, x), (x,
        ConditionSet(x, (-oo < y) & (y < oo),
            ImageSet(Lambda(n, n*pi + atan(y)), S.Integers))))

    assert dumeq(invert_real(tan(exp(x)), y, x), (x,
        ConditionSet(x, (-oo < y) & (y < oo),
            ImageSet(Lambda(n, log(n*pi + atan(y))), S.Integers))))

    assert dumeq(invert_real(cot(x), y, x), (x,
        ConditionSet(x, (-oo < y) & (y < oo),
            ImageSet(Lambda(n, n*pi + acot(y)), S.Integers))))

    assert dumeq(invert_real(cot(exp(x)), y, x), (x,
        ConditionSet(x, (-oo < y) & (y < oo),
            ImageSet(Lambda(n, log(n*pi + acot(y))), S.Integers))))

    assert dumeq(invert_real(tan(tan(x)), y, x),
        (x, ConditionSet(x, Eq(tan(tan(x)), y), S.Reals)))
        # slight regression compared to previous result:
        # (tan(x), imageset(Lambda(n, n*pi + atan(y)), S.Integers)))

    x = Symbol('x', positive=True)
    assert invert_real(x**pi, y, x) == (x, FiniteSet(y**(1/pi)))

    r = Symbol('r', real=True)
    p = Symbol('p', positive=True)
    assert invert_real(sinh(x), r, x) == (x, FiniteSet(asinh(r)))
    assert invert_real(sinh(log(x)), p, x) == (x, FiniteSet(exp(asinh(p))))

    assert invert_real(cosh(x), r, x) == (x, Intersection(
        FiniteSet(-acosh(r), acosh(r)), S.Reals))
    assert invert_real(cosh(x), p + 1, x) == (x,
        FiniteSet(-acosh(p + 1), acosh(p + 1)))

    assert invert_real(tanh(x), r, x) == (x, Intersection(FiniteSet(atanh(r)), S.Reals))
    assert invert_real(coth(x), p+1, x) == (x, FiniteSet(acoth(p+1)))
    assert invert_real(sech(x), r, x) == (x, Intersection(
        FiniteSet(-asech(r), asech(r)), S.Reals))
    assert invert_real(csch(x), p, x) == (x, FiniteSet(acsch(p)))

    assert dumeq(invert_real(tanh(sin(x)), r, x), (x,
        ConditionSet(x, (S(-1) <= atanh(r)) & (atanh(r) <= S(1)), Union(
            ImageSet(Lambda(n, 2*n*pi + asin(atanh(r))), S.Integers),
            ImageSet(Lambda(n, 2*n*pi - asin(atanh(r)) + pi), S.Integers)))))


def test_invert_trig_hyp_real():
    # check some codepaths that are not as easily reached otherwise
    n = Dummy('n')
    assert _invert_trig_hyp_real(cosh(x), Range(-5, 10, 1), x)[1].dummy_eq(Union(
        ImageSet(Lambda(n, -acosh(n)), Range(1, 10, 1)),
        ImageSet(Lambda(n, acosh(n)), Range(1, 10, 1))))
    assert _invert_trig_hyp_real(coth(x), Interval(-3, 2), x) == (x, Union(
        Interval(-oo, -acoth(3)), Interval(acoth(2), oo)))
    assert _invert_trig_hyp_real(tanh(x), Interval(-S.Half, 1), x) == (x,
        Interval(-atanh(S.Half), oo))
    assert _invert_trig_hyp_real(sech(x), imageset(n, S.Half + n/3, S.Naturals0), x) == \
        (x, FiniteSet(-asech(S(1)/2), asech(S(1)/2), -asech(S(5)/6), asech(S(5)/6)))
    assert _invert_trig_hyp_real(csch(x), S.Reals, x) == (x,
        Union(Interval.open(-oo, 0), Interval.open(0, oo)))


def test_invert_complex():
    assert invert_complex(x + 3, y, x) == (x, FiniteSet(y - 3))
    assert invert_complex(x*3, y, x) == (x, FiniteSet(y / 3))
    assert invert_complex((x - 1)**3, 0, x) == (x, FiniteSet(1))

    assert dumeq(invert_complex(exp(x), y, x),
        (x, imageset(Lambda(n, I*(2*pi*n + arg(y)) + log(Abs(y))), S.Integers)))

    assert invert_complex(log(x), y, x) == (x, FiniteSet(exp(y)))

    raises(ValueError, lambda: invert_real(1, y, x))
    raises(ValueError, lambda: invert_complex(x, x, x))
    raises(ValueError, lambda: invert_complex(x, x, 1))

    assert dumeq(invert_complex(sin(x), I, x), (x, Union(
        ImageSet(Lambda(n, 2*n*pi + I*log(1 + sqrt(2))), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + pi - I*log(1 + sqrt(2))), S.Integers))))
    assert dumeq(invert_complex(cos(x), 1+I, x), (x, Union(
        ImageSet(Lambda(n, 2*n*pi - acos(1 + I)), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + acos(1 + I)), S.Integers))))
    assert dumeq(invert_complex(tan(2*x), 1, x), (x,
        ImageSet(Lambda(n, n*pi/2 + pi/8), S.Integers)))
    assert dumeq(invert_complex(cot(x), 2*I, x), (x,
        ImageSet(Lambda(n, n*pi - I*acoth(2)), S.Integers)))

    assert dumeq(invert_complex(sinh(x), 0, x), (x, Union(
        ImageSet(Lambda(n, 2*n*I*pi), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi + I*pi), S.Integers))))
    assert dumeq(invert_complex(cosh(x), 0, x), (x, Union(
        ImageSet(Lambda(n, 2*n*I*pi + I*pi/2), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi + 3*I*pi/2), S.Integers))))
    assert invert_complex(tanh(x), 1, x) == (x, S.EmptySet)
    assert dumeq(invert_complex(tanh(x), a, x), (x,
        ConditionSet(x, Ne(a, -1) & Ne(a, 1),
        ImageSet(Lambda(n, n*I*pi + atanh(a)), S.Integers))))
    assert invert_complex(coth(x), 1, x) == (x, S.EmptySet)
    assert dumeq(invert_complex(coth(x), a, x), (x,
        ConditionSet(x, Ne(a, -1) & Ne(a, 1),
        ImageSet(Lambda(n, n*I*pi + acoth(a)), S.Integers))))
    assert dumeq(invert_complex(sech(x), 2, x), (x, Union(
        ImageSet(Lambda(n, 2*n*I*pi + I*pi/3), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi + 5*I*pi/3), S.Integers))))


def test_domain_check():
    assert domain_check(1/(1 + (1/(x+1))**2), x, -1) is False
    assert domain_check(x**2, x, 0) is True
    assert domain_check(x, x, oo) is False
    assert domain_check(0, x, oo) is False


def test_issue_11536():
    assert solveset(0**x - 100, x, S.Reals) == S.EmptySet
    assert solveset(0**x - 1, x, S.Reals) == FiniteSet(0)


def test_issue_17479():
    f = (x**2 + y**2)**2 + (x**2 + z**2)**2 - 2*(2*x**2 + y**2 + z**2)
    fx = f.diff(x)
    fy = f.diff(y)
    fz = f.diff(z)
    sol = nonlinsolve([fx, fy, fz], [x, y, z])
    assert len(sol) >= 4 and len(sol) <= 20
    # nonlinsolve has been giving a varying number of solutions
    # (originally 18, then 20, now 19) due to various internal changes.
    # Unfortunately not all the solutions are actually valid and some are
    # redundant. Since the original issue was that an exception was raised,
    # this first test only checks that nonlinsolve returns a "plausible"
    # solution set. The next test checks the result for correctness.


@XFAIL
def test_issue_18449():
    x, y, z = symbols("x, y, z")
    f = (x**2 + y**2)**2 + (x**2 + z**2)**2 - 2*(2*x**2 + y**2 + z**2)
    fx = diff(f, x)
    fy = diff(f, y)
    fz = diff(f, z)
    sol = nonlinsolve([fx, fy, fz], [x, y, z])
    for (xs, ys, zs) in sol:
        d = {x: xs, y: ys, z: zs}
        assert tuple(_.subs(d).simplify() for _ in (fx, fy, fz)) == (0, 0, 0)
    # After simplification and removal of duplicate elements, there should
    # only be 4 parametric solutions left:
    # simplifiedsolutions = FiniteSet((sqrt(1 - z**2), z, z),
    #                                 (-sqrt(1 - z**2), z, z),
    #                                 (sqrt(1 - z**2), -z, z),
    #                                 (-sqrt(1 - z**2), -z, z))
    # TODO: Is the above solution set definitely complete?


def test_issue_21047():
    f = (2 - x)**2 + (sqrt(x - 1) - 1)**6
    assert solveset(f, x, S.Reals) == FiniteSet(2)

    f = (sqrt(x)-1)**2 + (sqrt(x)+1)**2 -2*x**2 + sqrt(2)
    assert solveset(f, x, S.Reals) == FiniteSet(
        S.Half - sqrt(2*sqrt(2) + 5)/2, S.Half + sqrt(2*sqrt(2) + 5)/2)


def test_is_function_class_equation():
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x), x) is True
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x) - 1, x) is True
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x) + sin(x), x) is True
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x) + sin(x) - a, x) is True
    assert _is_function_class_equation(TrigonometricFunction,
                                       sin(x)*tan(x) + sin(x), x) is True
    assert _is_function_class_equation(TrigonometricFunction,
                                       sin(x)*tan(x + a) + sin(x), x) is True
    assert _is_function_class_equation(TrigonometricFunction,
                                       sin(x)*tan(x*a) + sin(x), x) is True
    assert _is_function_class_equation(TrigonometricFunction,
                                       a*tan(x) - 1, x) is True
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x)**2 + sin(x) - 1, x) is True
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x) + x, x) is False
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x**2), x) is False
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x**2) + sin(x), x) is False
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(x)**sin(x), x) is False
    assert _is_function_class_equation(TrigonometricFunction,
                                       tan(sin(x)) + sin(x), x) is False
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x), x) is True
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x) - 1, x) is True
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x) + sinh(x), x) is True
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x) + sinh(x) - a, x) is True
    assert _is_function_class_equation(HyperbolicFunction,
                                       sinh(x)*tanh(x) + sinh(x), x) is True
    assert _is_function_class_equation(HyperbolicFunction,
                                       sinh(x)*tanh(x + a) + sinh(x), x) is True
    assert _is_function_class_equation(HyperbolicFunction,
                                       sinh(x)*tanh(x*a) + sinh(x), x) is True
    assert _is_function_class_equation(HyperbolicFunction,
                                       a*tanh(x) - 1, x) is True
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x)**2 + sinh(x) - 1, x) is True
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x) + x, x) is False
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x**2), x) is False
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x**2) + sinh(x), x) is False
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(x)**sinh(x), x) is False
    assert _is_function_class_equation(HyperbolicFunction,
                                       tanh(sinh(x)) + sinh(x), x) is False


def test_garbage_input():
    raises(ValueError, lambda: solveset_real([y], y))
    x = Symbol('x', real=True)
    assert solveset_real(x, 1) == S.EmptySet
    assert solveset_real(x - 1, 1) == FiniteSet(x)
    assert solveset_real(x, pi) == S.EmptySet
    assert solveset_real(x, x**2) == S.EmptySet

    raises(ValueError, lambda: solveset_complex([x], x))
    assert solveset_complex(x, pi) == S.EmptySet

    raises(ValueError, lambda: solveset((x, y), x))
    raises(ValueError, lambda: solveset(x + 1, S.Reals))
    raises(ValueError, lambda: solveset(x + 1, x, 2))


def test_solve_mul():
    assert solveset_real((a*x + b)*(exp(x) - 3), x) == \
        Union({log(3)}, Intersection({-b/a}, S.Reals))
    anz = Symbol('anz', nonzero=True)
    bb = Symbol('bb', real=True)
    assert solveset_real((anz*x + bb)*(exp(x) - 3), x) == \
        FiniteSet(-bb/anz, log(3))
    assert solveset_real((2*x + 8)*(8 + exp(x)), x) == FiniteSet(S(-4))
    assert solveset_real(x/log(x), x) is S.EmptySet


def test_solve_invert():
    assert solveset_real(exp(x) - 3, x) == FiniteSet(log(3))
    assert solveset_real(log(x) - 3, x) == FiniteSet(exp(3))

    assert solveset_real(3**(x + 2), x) == FiniteSet()
    assert solveset_real(3**(2 - x), x) == FiniteSet()

    assert solveset_real(y - b*exp(a/x), x) == Intersection(
        S.Reals, FiniteSet(a/log(y/b)))

    # issue 4504
    assert solveset_real(2**x - 10, x) == FiniteSet(1 + log(5)/log(2))


def test_issue_25768():
    assert dumeq(solveset_real(sin(x) - S.Half, x), Union(
        ImageSet(Lambda(n, pi*2*n + pi/6), S.Integers),
        ImageSet(Lambda(n, pi*2*n + pi*5/6), S.Integers)))
    n1 = solveset_real(sin(x) - 0.5, x).n(5)
    n2 = solveset_real(sin(x) - S.Half, x).n(5)
    # help pass despite fp differences
    eq = [i.replace(
        lambda x:x.is_Float,
        lambda x:Rational(x).limit_denominator(1000)) for i in (n1, n2)]
    assert dumeq(*eq),(n1,n2)


def test_errorinverses():
    assert solveset_real(erf(x) - S.Half, x) == \
        FiniteSet(erfinv(S.Half))
    assert solveset_real(erfinv(x) - 2, x) == \
        FiniteSet(erf(2))
    assert solveset_real(erfc(x) - S.One, x) == \
        FiniteSet(erfcinv(S.One))
    assert solveset_real(erfcinv(x) - 2, x) == FiniteSet(erfc(2))


def test_solve_polynomial():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    assert solveset_real(3*x - 2, x) == FiniteSet(Rational(2, 3))

    assert solveset_real(x**2 - 1, x) == FiniteSet(-S.One, S.One)
    assert solveset_real(x - y**3, x) == FiniteSet(y ** 3)

    assert solveset_real(x**3 - 15*x - 4, x) == FiniteSet(
        -2 + 3 ** S.Half,
        S(4),
        -2 - 3 ** S.Half)

    assert solveset_real(sqrt(x) - 1, x) == FiniteSet(1)
    assert solveset_real(sqrt(x) - 2, x) == FiniteSet(4)
    assert solveset_real(x**Rational(1, 4) - 2, x) == FiniteSet(16)
    assert solveset_real(x**Rational(1, 3) - 3, x) == FiniteSet(27)
    assert len(solveset_real(x**5 + x**3 + 1, x)) == 1
    assert len(solveset_real(-2*x**3 + 4*x**2 - 2*x + 6, x)) > 0
    assert solveset_real(x**6 + x**4 + I, x) is S.EmptySet


def test_return_root_of():
    f = x**5 - 15*x**3 - 5*x**2 + 10*x + 20
    s = list(solveset_complex(f, x))
    for root in s:
        assert root.func == CRootOf

    # if one uses solve to get the roots of a polynomial that has a CRootOf
    # solution, make sure that the use of nfloat during the solve process
    # doesn't fail. Note: if you want numerical solutions to a polynomial
    # it is *much* faster to use nroots to get them than to solve the
    # equation only to get CRootOf solutions which are then numerically
    # evaluated. So for eq = x**5 + 3*x + 7 do Poly(eq).nroots() rather
    # than [i.n() for i in solve(eq)] to get the numerical roots of eq.
    assert nfloat(list(solveset_complex(x**5 + 3*x**3 + 7, x))[0],
                  exponent=False) == CRootOf(x**5 + 3*x**3 + 7, 0).n()

    sol = list(solveset_complex(x**6 - 2*x + 2, x))
    assert all(isinstance(i, CRootOf) for i in sol) and len(sol) == 6

    f = x**5 - 15*x**3 - 5*x**2 + 10*x + 20
    s = list(solveset_complex(f, x))
    for root in s:
        assert root.func == CRootOf

    s = x**5 + 4*x**3 + 3*x**2 + Rational(7, 4)
    assert solveset_complex(s, x) == \
        FiniteSet(*Poly(s*4, domain='ZZ').all_roots())

    # Refer issue #7876
    eq = x*(x - 1)**2*(x + 1)*(x**6 - x + 1)
    assert solveset_complex(eq, x) == \
        FiniteSet(-1, 0, 1, CRootOf(x**6 - x + 1, 0),
                       CRootOf(x**6 - x + 1, 1),
                       CRootOf(x**6 - x + 1, 2),
                       CRootOf(x**6 - x + 1, 3),
                       CRootOf(x**6 - x + 1, 4),
                       CRootOf(x**6 - x + 1, 5))


def test_solveset_sqrt_1():
    assert solveset_real(sqrt(5*x + 6) - 2 - x, x) == \
        FiniteSet(-S.One, S(2))
    assert solveset_real(sqrt(x - 1) - x + 7, x) == FiniteSet(10)
    assert solveset_real(sqrt(x - 2) - 5, x) == FiniteSet(27)
    assert solveset_real(sqrt(x) - 2 - 5, x) == FiniteSet(49)
    assert solveset_real(sqrt(x**3), x) == FiniteSet(0)
    assert solveset_real(sqrt(x - 1), x) == FiniteSet(1)
    assert solveset_real(sqrt((x-3)/x), x) == FiniteSet(3)
    assert solveset_real(sqrt((x-3)/x)-Rational(1, 2), x) == \
        FiniteSet(4)

def test_solveset_sqrt_2():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    # http://tutorial.math.lamar.edu/Classes/Alg/SolveRadicalEqns.aspx#Solve_Rad_Ex2_a
    assert solveset_real(sqrt(2*x - 1) - sqrt(x - 4) - 2, x) == \
        FiniteSet(S(5), S(13))
    assert solveset_real(sqrt(x + 7) + 2 - sqrt(3 - x), x) == \
        FiniteSet(-6)

    # http://www.purplemath.com/modules/solverad.htm
    assert solveset_real(sqrt(17*x - sqrt(x**2 - 5)) - 7, x) == \
        FiniteSet(3)

    eq = x + 1 - (x**4 + 4*x**3 - x)**Rational(1, 4)
    assert solveset_real(eq, x) == FiniteSet(Rational(-1, 2), Rational(-1, 3))

    eq = sqrt(2*x + 9) - sqrt(x + 1) - sqrt(x + 4)
    assert solveset_real(eq, x) == FiniteSet(0)

    eq = sqrt(x + 4) + sqrt(2*x - 1) - 3*sqrt(x - 1)
    assert solveset_real(eq, x) == FiniteSet(5)

    eq = sqrt(x)*sqrt(x - 7) - 12
    assert solveset_real(eq, x) == FiniteSet(16)

    eq = sqrt(x - 3) + sqrt(x) - 3
    assert solveset_real(eq, x) == FiniteSet(4)

    eq = sqrt(2*x**2 - 7) - (3 - x)
    assert solveset_real(eq, x) == FiniteSet(-S(8), S(2))

    # others
    eq = sqrt(9*x**2 + 4) - (3*x + 2)
    assert solveset_real(eq, x) == FiniteSet(0)

    assert solveset_real(sqrt(x - 3) - sqrt(x) - 3, x) == FiniteSet()

    eq = (2*x - 5)**Rational(1, 3) - 3
    assert solveset_real(eq, x) == FiniteSet(16)

    assert solveset_real(sqrt(x) + sqrt(sqrt(x)) - 4, x) == \
        FiniteSet((Rational(-1, 2) + sqrt(17)/2)**4)

    eq = sqrt(x) - sqrt(x - 1) + sqrt(sqrt(x))
    assert solveset_real(eq, x) == FiniteSet()

    eq = (x - 4)**2 + (sqrt(x) - 2)**4
    assert solveset_real(eq, x) == FiniteSet(-4, 4)

    eq = (sqrt(x) + sqrt(x + 1) + sqrt(1 - x) - 6*sqrt(5)/5)
    ans = solveset_real(eq, x)
    ra = S('''-1484/375 - 4*(-S(1)/2 + sqrt(3)*I/2)*(-12459439/52734375 +
    114*sqrt(12657)/78125)**(S(1)/3) - 172564/(140625*(-S(1)/2 +
    sqrt(3)*I/2)*(-12459439/52734375 + 114*sqrt(12657)/78125)**(S(1)/3))''')
    rb = Rational(4, 5)
    assert all(abs(eq.subs(x, i).n()) < 1e-10 for i in (ra, rb)) and \
        len(ans) == 2 and \
        {i.n(chop=True) for i in ans} == \
        {i.n(chop=True) for i in (ra, rb)}

    assert solveset_real(sqrt(x) + x**Rational(1, 3) +
                                 x**Rational(1, 4), x) == FiniteSet(0)

    assert solveset_real(x/sqrt(x**2 + 1), x) == FiniteSet(0)

    eq = (x - y**3)/((y**2)*sqrt(1 - y**2))
    assert solveset_real(eq, x) == FiniteSet(y**3)

    # issue 4497
    assert solveset_real(1/(5 + x)**Rational(1, 5) - 9, x) == \
        FiniteSet(Rational(-295244, 59049))


@XFAIL
def test_solve_sqrt_fail():
    # this only works if we check real_root(eq.subs(x, Rational(1, 3)))
    # but checksol doesn't work like that
    eq = (x**3 - 3*x**2)**Rational(1, 3) + 1 - x
    assert solveset_real(eq, x) == FiniteSet(Rational(1, 3))


@slow
def test_solve_sqrt_3():
    R = Symbol('R')
    eq = sqrt(2)*R*sqrt(1/(R + 1)) + (R + 1)*(sqrt(2)*sqrt(1/(R + 1)) - 1)
    sol = solveset_complex(eq, R)
    fset = [Rational(5, 3) + 4*sqrt(10)*cos(atan(3*sqrt(111)/251)/3)/3,
            -sqrt(10)*cos(atan(3*sqrt(111)/251)/3)/3 +
            40*re(1/((Rational(-1, 2) - sqrt(3)*I/2)*(Rational(251, 27) + sqrt(111)*I/9)**Rational(1, 3)))/9 +
            sqrt(30)*sin(atan(3*sqrt(111)/251)/3)/3 + Rational(5, 3) +
            I*(-sqrt(30)*cos(atan(3*sqrt(111)/251)/3)/3 -
               sqrt(10)*sin(atan(3*sqrt(111)/251)/3)/3 +
               40*im(1/((Rational(-1, 2) - sqrt(3)*I/2)*(Rational(251, 27) + sqrt(111)*I/9)**Rational(1, 3)))/9)]
    cset = [40*re(1/((Rational(-1, 2) + sqrt(3)*I/2)*(Rational(251, 27) + sqrt(111)*I/9)**Rational(1, 3)))/9 -
            sqrt(10)*cos(atan(3*sqrt(111)/251)/3)/3 - sqrt(30)*sin(atan(3*sqrt(111)/251)/3)/3 +
            Rational(5, 3) +
            I*(40*im(1/((Rational(-1, 2) + sqrt(3)*I/2)*(Rational(251, 27) + sqrt(111)*I/9)**Rational(1, 3)))/9 -
               sqrt(10)*sin(atan(3*sqrt(111)/251)/3)/3 +
               sqrt(30)*cos(atan(3*sqrt(111)/251)/3)/3)]

    fs = FiniteSet(*fset)
    cs = ConditionSet(R, Eq(eq, 0), FiniteSet(*cset))
    assert sol == (fs - {-1}) | (cs - {-1})

    # the number of real roots will depend on the value of m: for m=1 there are 4
    # and for m=-1 there are none.
    eq = -sqrt((m - q)**2 + (-m/(2*q) + S.Half)**2) + sqrt((-m**2/2 - sqrt(
        4*m**4 - 4*m**2 + 8*m + 1)/4 - Rational(1, 4))**2 + (m**2/2 - m - sqrt(
            4*m**4 - 4*m**2 + 8*m + 1)/4 - Rational(1, 4))**2)
    unsolved_object = ConditionSet(q, Eq(sqrt((m - q)**2 + (-m/(2*q) + S.Half)**2) -
        sqrt((-m**2/2 - sqrt(4*m**4 - 4*m**2 + 8*m + 1)/4 - Rational(1, 4))**2 + (m**2/2 - m -
        sqrt(4*m**4 - 4*m**2 + 8*m + 1)/4 - Rational(1, 4))**2), 0), S.Reals)
    assert solveset_real(eq, q) == unsolved_object


def test_solve_polynomial_symbolic_param():
    assert solveset_complex((x**2 - 1)**2 - a, x) == \
        FiniteSet(sqrt(1 + sqrt(a)), -sqrt(1 + sqrt(a)),
                  sqrt(1 - sqrt(a)), -sqrt(1 - sqrt(a)))

    # issue 4507
    assert solveset_complex(y - b/(1 + a*x), x) == \
        FiniteSet((b/y - 1)/a) - FiniteSet(-1/a)

    # issue 4508
    assert solveset_complex(y - b*x/(a + x), x) == \
        FiniteSet(-a*y/(y - b)) - FiniteSet(-a)


def test_solve_rational():
    assert solveset_real(1/x + 1, x) == FiniteSet(-S.One)
    assert solveset_real(1/exp(x) - 1, x) == FiniteSet(0)
    assert solveset_real(x*(1 - 5/x), x) == FiniteSet(5)
    assert solveset_real(2*x/(x + 2) - 1, x) == FiniteSet(2)
    assert solveset_real((x**2/(7 - x)).diff(x), x) == \
        FiniteSet(S.Zero, S(14))


def test_solveset_real_gen_is_pow():
    assert solveset_real(sqrt(1) + 1, x) is S.EmptySet


def test_no_sol():
    assert solveset(1 - oo*x) is S.EmptySet
    assert solveset(oo*x, x) is S.EmptySet
    assert solveset(oo*x - oo, x) is S.EmptySet
    assert solveset_real(4, x) is S.EmptySet
    assert solveset_real(exp(x), x) is S.EmptySet
    assert solveset_real(x**2 + 1, x) is S.EmptySet
    assert solveset_real(-3*a/sqrt(x), x) is S.EmptySet
    assert solveset_real(1/x, x) is S.EmptySet
    assert solveset_real(-(1 + x)/(2 + x)**2 + 1/(2 + x), x
        ) is S.EmptySet


def test_sol_zero_real():
    assert solveset_real(0, x) == S.Reals
    assert solveset(0, x, Interval(1, 2)) == Interval(1, 2)
    assert solveset_real(-x**2 - 2*x + (x + 1)**2 - 1, x) == S.Reals


def test_no_sol_rational_extragenous():
    assert solveset_real((x/(x + 1) + 3)**(-2), x) is S.EmptySet
    assert solveset_real((x - 1)/(1 + 1/(x - 1)), x) is S.EmptySet


def test_solve_polynomial_cv_1a():
    """
    Test for solving on equations that can be converted to
    a polynomial equation using the change of variable y -> x**Rational(p, q)
    """
    assert solveset_real(sqrt(x) - 1, x) == FiniteSet(1)
    assert solveset_real(sqrt(x) - 2, x) == FiniteSet(4)
    assert solveset_real(x**Rational(1, 4) - 2, x) == FiniteSet(16)
    assert solveset_real(x**Rational(1, 3) - 3, x) == FiniteSet(27)
    assert solveset_real(x*(x**(S.One / 3) - 3), x) == \
        FiniteSet(S.Zero, S(27))


def test_solveset_real_rational():
    """Test solveset_real for rational functions"""
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    assert solveset_real((x - y**3) / ((y**2)*sqrt(1 - y**2)), x) \
        == FiniteSet(y**3)
    # issue 4486
    assert solveset_real(2*x/(x + 2) - 1, x) == FiniteSet(2)


def test_solveset_real_log():
    assert solveset_real(log((x-1)*(x+1)), x) == \
        FiniteSet(sqrt(2), -sqrt(2))


def test_poly_gens():
    assert solveset_real(4**(2*(x**2) + 2*x) - 8, x) == \
        FiniteSet(Rational(-3, 2), S.Half)


def test_solve_abs():
    n = Dummy('n')
    raises(ValueError, lambda: solveset(Abs(x) - 1, x))
    assert solveset(Abs(x) - n, x, S.Reals).dummy_eq(
        ConditionSet(x, Contains(n, Interval(0, oo)), {-n, n}))
    assert solveset_real(Abs(x) - 2, x) == FiniteSet(-2, 2)
    assert solveset_real(Abs(x) + 2, x) is S.EmptySet
    assert solveset_real(Abs(x + 3) - 2*Abs(x - 3), x) == \
        FiniteSet(1, 9)
    assert solveset_real(2*Abs(x) - Abs(x - 1), x) == \
        FiniteSet(-1, Rational(1, 3))

    sol = ConditionSet(
            x,
            And(
                Contains(b, Interval(0, oo)),
                Contains(a + b, Interval(0, oo)),
                Contains(a - b, Interval(0, oo))),
            FiniteSet(-a - b - 3, -a + b - 3, a - b - 3, a + b - 3))
    eq = Abs(Abs(x + 3) - a) - b
    assert invert_real(eq, 0, x)[1] == sol
    reps = {a: 3, b: 1}
    eqab = eq.subs(reps)
    for si in sol.subs(reps):
        assert not eqab.subs(x, si)
    assert dumeq(solveset(Eq(sin(Abs(x)), 1), x, domain=S.Reals), Union(
        Intersection(Interval(0, oo), Union(
        Intersection(ImageSet(Lambda(n, 2*n*pi + 3*pi/2), S.Integers),
            Interval(-oo, 0)),
        Intersection(ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers),
            Interval(0, oo))))))


def test_issue_9824():
    assert dumeq(solveset(sin(x)**2 - 2*sin(x) + 1, x), ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers))
    assert dumeq(solveset(cos(x)**2 - 2*cos(x) + 1, x), ImageSet(Lambda(n, 2*n*pi), S.Integers))


def test_issue_9565():
    assert solveset_real(Abs((x - 1)/(x - 5)) <= Rational(1, 3), x) == Interval(-1, 2)


def test_issue_10069():
    eq = abs(1/(x - 1)) - 1 > 0
    assert solveset_real(eq, x) == Union(
        Interval.open(0, 1), Interval.open(1, 2))


def test_real_imag_splitting():
    a, b = symbols('a b', real=True)
    assert solveset_real(sqrt(a**2 - b**2) - 3, a) == \
        FiniteSet(-sqrt(b**2 + 9), sqrt(b**2 + 9))
    assert solveset_real(sqrt(a**2 + b**2) - 3, a) != \
        S.EmptySet


def test_units():
    assert solveset_real(1/x - 1/(2*cm), x) == FiniteSet(2*cm)


def test_solve_only_exp_1():
    y = Symbol('y', positive=True)
    assert solveset_real(exp(x) - y, x) == FiniteSet(log(y))
    assert solveset_real(exp(x) + exp(-x) - 4, x) == \
        FiniteSet(log(-sqrt(3) + 2), log(sqrt(3) + 2))
    assert solveset_real(exp(x) + exp(-x) - y, x) != S.EmptySet


def test_atan2():
    # The .inverse() method on atan2 works only if x.is_real is True and the
    # second argument is a real constant
    assert solveset_real(atan2(x, 2) - pi/3, x) == FiniteSet(2*sqrt(3))


def test_piecewise_solveset():
    eq = Piecewise((x - 2, Gt(x, 2)), (2 - x, True)) - 3
    assert set(solveset_real(eq, x)) == set(FiniteSet(-1, 5))

    absxm3 = Piecewise(
        (x - 3, 0 <= x - 3),
        (3 - x, 0 > x - 3))
    y = Symbol('y', positive=True)
    assert solveset_real(absxm3 - y, x) == FiniteSet(-y + 3, y + 3)

    f = Piecewise(((x - 2)**2, x >= 0), (0, True))
    assert solveset(f, x, domain=S.Reals) == Union(FiniteSet(2), Interval(-oo, 0, True, True))

    assert solveset(
        Piecewise((x + 1, x > 0), (I, True)) - I, x, S.Reals
        ) == Interval(-oo, 0)

    assert solveset(Piecewise((x - 1, Ne(x, I)), (x, True)), x) == FiniteSet(1)

    # issue 19718
    g = Piecewise((1, x > 10), (0, True))
    assert solveset(g > 0, x, S.Reals) == Interval.open(10, oo)

    from sympy.logic.boolalg import BooleanTrue
    f = BooleanTrue()
    assert solveset(f, x, domain=Interval(-3, 10)) == Interval(-3, 10)

    # issue 20552
    f = Piecewise((0, Eq(x, 0)), (x**2/Abs(x), True))
    g = Piecewise((0, Eq(x, pi)), ((x - pi)/sin(x), True))
    assert solveset(f, x, domain=S.Reals) == FiniteSet(0)
    assert solveset(g) == FiniteSet(pi)


def test_solveset_complex_polynomial():
    assert solveset_complex(a*x**2 + b*x + c, x) == \
        FiniteSet(-b/(2*a) - sqrt(-4*a*c + b**2)/(2*a),
                  -b/(2*a) + sqrt(-4*a*c + b**2)/(2*a))

    assert solveset_complex(x - y**3, y) == FiniteSet(
        (-x**Rational(1, 3))/2 + I*sqrt(3)*x**Rational(1, 3)/2,
        x**Rational(1, 3),
        (-x**Rational(1, 3))/2 - I*sqrt(3)*x**Rational(1, 3)/2)

    assert solveset_complex(x + 1/x - 1, x) == \
        FiniteSet(S.Half + I*sqrt(3)/2, S.Half - I*sqrt(3)/2)


def test_sol_zero_complex():
    assert solveset_complex(0, x) is S.Complexes


def test_solveset_complex_rational():
    assert solveset_complex((x - 1)*(x - I)/(x - 3), x) == \
        FiniteSet(1, I)

    assert solveset_complex((x - y**3)/((y**2)*sqrt(1 - y**2)), x) == \
        FiniteSet(y**3)
    assert solveset_complex(-x**2 - I, x) == \
        FiniteSet(-sqrt(2)/2 + sqrt(2)*I/2, sqrt(2)/2 - sqrt(2)*I/2)


def test_solve_quintics():
    skip("This test is too slow")
    f = x**5 - 110*x**3 - 55*x**2 + 2310*x + 979
    s = solveset_complex(f, x)
    for root in s:
        res = f.subs(x, root.n()).n()
        assert tn(res, 0)

    f = x**5 + 15*x + 12
    s = solveset_complex(f, x)
    for root in s:
        res = f.subs(x, root.n()).n()
        assert tn(res, 0)


def test_solveset_complex_exp():
    assert dumeq(solveset_complex(exp(x) - 1, x),
        imageset(Lambda(n, I*2*n*pi), S.Integers))
    assert dumeq(solveset_complex(exp(x) - I, x),
        imageset(Lambda(n, I*(2*n*pi + pi/2)), S.Integers))
    assert solveset_complex(1/exp(x), x) == S.EmptySet
    assert dumeq(solveset_complex(sinh(x).rewrite(exp), x),
        imageset(Lambda(n, n*pi*I), S.Integers))


def test_solveset_real_exp():
    assert solveset(Eq((-2)**x, 4), x, S.Reals) == FiniteSet(2)
    assert solveset(Eq(-2**x, 4), x, S.Reals) == S.EmptySet
    assert solveset(Eq((-3)**x, 27), x, S.Reals) == S.EmptySet
    assert solveset(Eq((-5)**(x+1), 625), x, S.Reals) == FiniteSet(3)
    assert solveset(Eq(2**(x-3), -16), x, S.Reals) == S.EmptySet
    assert solveset(Eq((-3)**(x - 3), -3**39), x, S.Reals) == FiniteSet(42)
    assert solveset(Eq(2**x, y), x, S.Reals) == Intersection(S.Reals, FiniteSet(log(y)/log(2)))

    assert invert_real((-2)**(2*x) - 16, 0, x) == (x, FiniteSet(2))


def test_solve_complex_log():
    assert solveset_complex(log(x), x) == FiniteSet(1)
    assert solveset_complex(1 - log(a + 4*x**2), x) == \
        FiniteSet(-sqrt(-a + E)/2, sqrt(-a + E)/2)


def test_solve_complex_sqrt():
    assert solveset_complex(sqrt(5*x + 6) - 2 - x, x) == \
        FiniteSet(-S.One, S(2))
    assert solveset_complex(sqrt(5*x + 6) - (2 + 2*I) - x, x) == \
        FiniteSet(-S(2), 3 - 4*I)
    assert solveset_complex(4*x*(1 - a * sqrt(x)), x) == \
        FiniteSet(S.Zero, 1 / a ** 2)


def test_solveset_complex_tan():
    s = solveset_complex(tan(x).rewrite(exp), x)
    assert dumeq(s, imageset(Lambda(n, pi*n), S.Integers) - \
        imageset(Lambda(n, pi*n + pi/2), S.Integers))


@_both_exp_pow
def test_solve_trig():
    assert dumeq(solveset_real(sin(x), x),
        Union(imageset(Lambda(n, 2*pi*n), S.Integers),
              imageset(Lambda(n, 2*pi*n + pi), S.Integers)))

    assert dumeq(solveset_real(sin(x) - 1, x),
        imageset(Lambda(n, 2*pi*n + pi/2), S.Integers))

    assert dumeq(solveset_real(cos(x), x),
        Union(imageset(Lambda(n, 2*pi*n + pi/2), S.Integers),
              imageset(Lambda(n, 2*pi*n + pi*Rational(3, 2)), S.Integers)))

    assert dumeq(solveset_real(sin(x) + cos(x), x),
        Union(imageset(Lambda(n, 2*n*pi + pi*Rational(3, 4)), S.Integers),
              imageset(Lambda(n, 2*n*pi + pi*Rational(7, 4)), S.Integers)))

    assert solveset_real(sin(x)**2 + cos(x)**2, x) == S.EmptySet

    assert dumeq(solveset_complex(cos(x) - S.Half, x),
        Union(imageset(Lambda(n, 2*n*pi + pi*Rational(5, 3)), S.Integers),
              imageset(Lambda(n, 2*n*pi + pi/3), S.Integers)))

    assert dumeq(solveset(sin(y + a) - sin(y), a, domain=S.Reals),
        ConditionSet(a, (S(-1) <= sin(y)) & (sin(y) <= S(1)), Union(
            ImageSet(Lambda(n, 2*n*pi - y + asin(sin(y))), S.Integers),
            ImageSet(Lambda(n, 2*n*pi - y - asin(sin(y)) + pi), S.Integers))))

    assert dumeq(solveset_real(sin(2*x)*cos(x) + cos(2*x)*sin(x)-1, x),
        ImageSet(Lambda(n, n*pi*Rational(2, 3) + pi/6), S.Integers))

    assert dumeq(solveset_real(2*tan(x)*sin(x) + 1, x), Union(
        ImageSet(Lambda(n, 2*n*pi + atan(sqrt(2)*sqrt(-1 + sqrt(17))/
            (1 - sqrt(17))) + pi), S.Integers),
        ImageSet(Lambda(n, 2*n*pi - atan(sqrt(2)*sqrt(-1 + sqrt(17))/
            (1 - sqrt(17))) + pi), S.Integers)))

    assert dumeq(solveset_real(cos(2*x)*cos(4*x) - 1, x),
                            ImageSet(Lambda(n, n*pi), S.Integers))

    assert dumeq(solveset(sin(x/10) + Rational(3, 4)), Union(
        ImageSet(Lambda(n, 20*n*pi - 10*asin(S(3)/4) + 20*pi), S.Integers),
        ImageSet(Lambda(n, 20*n*pi + 10*asin(S(3)/4) + 10*pi), S.Integers)))

    assert dumeq(solveset(cos(x/15) + cos(x/5)), Union(
        ImageSet(Lambda(n, 30*n*pi + 15*pi/2), S.Integers),
        ImageSet(Lambda(n, 30*n*pi + 45*pi/2), S.Integers),
        ImageSet(Lambda(n, 30*n*pi + 75*pi/4), S.Integers),
        ImageSet(Lambda(n, 30*n*pi + 45*pi/4), S.Integers),
        ImageSet(Lambda(n, 30*n*pi + 105*pi/4), S.Integers),
        ImageSet(Lambda(n, 30*n*pi + 15*pi/4), S.Integers)))

    assert dumeq(solveset(sec(sqrt(2)*x/3) + 5), Union(
        ImageSet(Lambda(n, 3*sqrt(2)*(2*n*pi - asec(-5))/2), S.Integers),
        ImageSet(Lambda(n, 3*sqrt(2)*(2*n*pi + asec(-5))/2), S.Integers)))

    assert dumeq(simplify(solveset(tan(pi*x) - cot(pi/2*x))), Union(
        ImageSet(Lambda(n, 4*n + 1), S.Integers),
        ImageSet(Lambda(n, 4*n + 3), S.Integers),
        ImageSet(Lambda(n, 4*n + Rational(7, 3)), S.Integers),
        ImageSet(Lambda(n, 4*n + Rational(5, 3)), S.Integers),
        ImageSet(Lambda(n, 4*n + Rational(11, 3)), S.Integers),
        ImageSet(Lambda(n, 4*n + Rational(1, 3)), S.Integers)))

    assert dumeq(solveset(cos(9*x)), Union(
        ImageSet(Lambda(n, 2*n*pi/9 + pi/18), S.Integers),
        ImageSet(Lambda(n, 2*n*pi/9 + pi/6), S.Integers)))

    assert dumeq(solveset(sin(8*x) + cot(12*x), x, S.Reals), Union(
        ImageSet(Lambda(n, n*pi/2 + pi/8), S.Integers),
        ImageSet(Lambda(n, n*pi/2 + 3*pi/8), S.Integers),
        ImageSet(Lambda(n, n*pi/2 + 5*pi/16), S.Integers),
        ImageSet(Lambda(n, n*pi/2 + 3*pi/16), S.Integers),
        ImageSet(Lambda(n, n*pi/2 + 7*pi/16), S.Integers),
        ImageSet(Lambda(n, n*pi/2 + pi/16), S.Integers)))

    # This is the only remaining solveset test that actually ends up being solved
    # by _solve_trig2(). All others are handled by the improved _solve_trig1.
    assert dumeq(solveset_real(2*cos(x)*cos(2*x) - 1, x),
          Union(ImageSet(Lambda(n, 2*n*pi + 2*atan(sqrt(-2*2**Rational(1, 3)*(67 +
                  9*sqrt(57))**Rational(2, 3) + 8*2**Rational(2, 3) + 11*(67 +
                  9*sqrt(57))**Rational(1, 3))/(3*(67 + 9*sqrt(57))**Rational(1, 6)))), S.Integers),
                  ImageSet(Lambda(n, 2*n*pi - 2*atan(sqrt(-2*2**Rational(1, 3)*(67 +
                  9*sqrt(57))**Rational(2, 3) + 8*2**Rational(2, 3) + 11*(67 +
                  9*sqrt(57))**Rational(1, 3))/(3*(67 + 9*sqrt(57))**Rational(1, 6))) +
                  2*pi), S.Integers)))

    # issue #16870
    assert dumeq(simplify(solveset(sin(x/180*pi) - S.Half, x, S.Reals)), Union(
        ImageSet(Lambda(n, 360*n + 150), S.Integers),
        ImageSet(Lambda(n, 360*n + 30), S.Integers)))


def test_solve_trig_hyp_by_inversion():
    n = Dummy('n')
    assert solveset_real(sin(2*x + 3) - S(1)/2, x).dummy_eq(Union(
        ImageSet(Lambda(n, n*pi - S(3)/2 + 13*pi/12), S.Integers),
        ImageSet(Lambda(n, n*pi - S(3)/2 + 17*pi/12), S.Integers)))
    assert solveset_complex(sin(2*x + 3) - S(1)/2, x).dummy_eq(Union(
        ImageSet(Lambda(n, n*pi - S(3)/2 + 13*pi/12), S.Integers),
        ImageSet(Lambda(n, n*pi - S(3)/2 + 17*pi/12), S.Integers)))
    assert solveset_real(tan(x) - tan(pi/10), x).dummy_eq(
        ImageSet(Lambda(n, n*pi + pi/10), S.Integers))
    assert solveset_complex(tan(x) - tan(pi/10), x).dummy_eq(
        ImageSet(Lambda(n, n*pi + pi/10), S.Integers))

    assert solveset_real(3*cosh(2*x) - 5, x) == FiniteSet(
        -acosh(S(5)/3)/2, acosh(S(5)/3)/2)
    assert solveset_complex(3*cosh(2*x) - 5, x).dummy_eq(Union(
        ImageSet(Lambda(n, n*I*pi - acosh(S(5)/3)/2), S.Integers),
        ImageSet(Lambda(n, n*I*pi + acosh(S(5)/3)/2), S.Integers)))
    assert solveset_real(sinh(x - 3) - 2, x) == FiniteSet(
        asinh(2) + 3)
    assert solveset_complex(sinh(x - 3) - 2, x).dummy_eq(Union(
        ImageSet(Lambda(n, 2*n*I*pi + asinh(2) + 3), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi - asinh(2) + 3 + I*pi), S.Integers)))

    assert solveset_real(cos(sinh(x))-cos(pi/12), x).dummy_eq(Union(
        ImageSet(Lambda(n, asinh(2*n*pi + pi/12)), S.Integers),
        ImageSet(Lambda(n, asinh(2*n*pi + 23*pi/12)), S.Integers)))
    assert solveset(cos(sinh(x))-cos(pi/12), x, Interval(2,3)) == \
        FiniteSet(asinh(23*pi/12), asinh(25*pi/12))
    assert solveset_real(cosh(x**2-1)-2, x) == FiniteSet(
        -sqrt(1 + acosh(2)), sqrt(1 + acosh(2)))

    assert solveset_real(sin(x) - 2, x) == S.EmptySet   # issue #17334
    assert solveset_real(cos(x) + 2, x) == S.EmptySet
    assert solveset_real(sec(x), x) == S.EmptySet
    assert solveset_real(csc(x), x) == S.EmptySet
    assert solveset_real(cosh(x) + 1, x) == S.EmptySet
    assert solveset_real(coth(x), x) == S.EmptySet
    assert solveset_real(sech(x) - 2, x) == S.EmptySet
    assert solveset_real(sech(x), x) == S.EmptySet
    assert solveset_real(tanh(x) + 1, x) == S.EmptySet
    assert solveset_complex(tanh(x), 1) == S.EmptySet
    assert solveset_complex(coth(x), -1) == S.EmptySet
    assert solveset_complex(sech(x), 0) == S.EmptySet
    assert solveset_complex(csch(x), 0) == S.EmptySet

    assert solveset_real(abs(csch(x)) - 3, x) == FiniteSet(-acsch(3), acsch(3))

    assert solveset_real(tanh(x**2 - 1) - exp(-9), x) == FiniteSet(
        -sqrt(atanh(exp(-9)) + 1), sqrt(atanh(exp(-9)) + 1))

    assert solveset_real(coth(log(x)) + 2, x) == FiniteSet(exp(-acoth(2)))
    assert solveset_real(coth(exp(x)) + 2, x) == S.EmptySet

    assert solveset_complex(sinh(x) - I/2, x).dummy_eq(Union(
        ImageSet(Lambda(n, 2*I*pi*n + 5*I*pi/6), S.Integers),
        ImageSet(Lambda(n, 2*I*pi*n + I*pi/6), S.Integers)))
    assert solveset_complex(sinh(x/10) + Rational(3, 4), x).dummy_eq(Union(
        ImageSet(Lambda(n, 20*n*I*pi - 10*asinh(S(3)/4)), S.Integers),
        ImageSet(Lambda(n, 20*n*I*pi + 10*asinh(S(3)/4) + 10*I*pi), S.Integers)))
    assert solveset_complex(sech(sqrt(2)*x/3) + 5, x).dummy_eq(Union(
        ImageSet(Lambda(n, 3*sqrt(2)*(2*n*I*pi - asech(-5))/2), S.Integers),
        ImageSet(Lambda(n, 3*sqrt(2)*(2*n*I*pi + asech(-5))/2), S.Integers)))
    assert solveset_complex(cosh(9*x), x).dummy_eq(Union(
        ImageSet(Lambda(n, 2*n*I*pi/9 + I*pi/18), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi/9 + I*pi/6), S.Integers)))

    eq = (x**5 -4*x + 1).subs(x, coth(z))
    assert solveset(eq, z, S.Complexes).dummy_eq(Union(
        ImageSet(Lambda(n, n*I*pi + acoth(CRootOf(x**5 -4*x + 1, 0))), S.Integers),
        ImageSet(Lambda(n, n*I*pi + acoth(CRootOf(x**5 -4*x + 1, 1))), S.Integers),
        ImageSet(Lambda(n, n*I*pi + acoth(CRootOf(x**5 -4*x + 1, 2))), S.Integers),
        ImageSet(Lambda(n, n*I*pi + acoth(CRootOf(x**5 -4*x + 1, 3))), S.Integers),
        ImageSet(Lambda(n, n*I*pi + acoth(CRootOf(x**5 -4*x + 1, 4))), S.Integers)))
    assert solveset(eq, z, S.Reals) == FiniteSet(
        acoth(CRootOf(x**5 - 4*x + 1, 0)), acoth(CRootOf(x**5 - 4*x + 1, 2)))

    eq = ((x-sqrt(3)/2)*(x+2)).expand().subs(x, cos(x))
    assert solveset(eq, x, S.Complexes).dummy_eq(Union(
        ImageSet(Lambda(n, 2*n*pi - acos(-2)), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + acos(-2)), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + pi/6), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + 11*pi/6), S.Integers)))
    assert solveset(eq, x, S.Reals).dummy_eq(Union(
        ImageSet(Lambda(n, 2*n*pi + pi/6), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + 11*pi/6), S.Integers)))

    assert solveset((1+sec(sqrt(3)*x+4)**2)/(1-sec(sqrt(3)*x+4))).dummy_eq(Union(
        ImageSet(Lambda(n, sqrt(3)*(2*n*pi - 4 - asec(I))/3), S.Integers),
        ImageSet(Lambda(n, sqrt(3)*(2*n*pi - 4 + asec(I))/3), S.Integers),
        ImageSet(Lambda(n, sqrt(3)*(2*n*pi - 4 - asec(-I))/3), S.Integers),
        ImageSet(Lambda(n, sqrt(3)*(2*n*pi - 4 + asec(-I))/3), S.Integers)))

    assert all_close(solveset(tan(3.14*x)**(S(3)/2)-5.678, x, Interval(0, 3)),
        FiniteSet(0.403301114561067, 0.403301114561067 + 0.318471337579618*pi,
                0.403301114561067 + 0.636942675159236*pi))


def test_old_trig_issues():
    # issues #9606 / #9531:
    assert solveset(sinh(x), x, S.Reals) == FiniteSet(0)
    assert solveset(sinh(x), x, S.Complexes).dummy_eq(Union(
        ImageSet(Lambda(n, 2*n*I*pi), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi + I*pi), S.Integers)))

    # issues #11218 / #18427
    assert solveset(sin(pi*x), x, S.Reals).dummy_eq(Union(
        ImageSet(Lambda(n, (2*n*pi + pi)/pi), S.Integers),
        ImageSet(Lambda(n, 2*n), S.Integers)))
    assert solveset(sin(pi*x), x).dummy_eq(Union(
        ImageSet(Lambda(n, (2*n*pi + pi)/pi), S.Integers),
        ImageSet(Lambda(n, 2*n), S.Integers)))

    # issue #17543
    assert solveset(I*cot(8*x - 8*E), x).dummy_eq(
        ImageSet(Lambda(n, pi*n/8 - 13*pi/16 + E), S.Integers))

    # issue #20798
    assert all_close(solveset(cos(2*x) - 0.5, x, Interval(0, 2*pi)), FiniteSet(
        0.523598775598299, -0.523598775598299 + pi,
        -0.523598775598299 + 2*pi, 0.523598775598299 + pi))
    sol = Union(ImageSet(Lambda(n, n*pi - 0.523598775598299), S.Integers),
                ImageSet(Lambda(n, n*pi + 0.523598775598299), S.Integers))
    ret = solveset(cos(2*x) - 0.5, x, S.Reals)
    # replace Dummy n by the regular Symbol n to allow all_close comparison.
    ret = ret.subs(ret.atoms(Dummy).pop(), n)
    assert all_close(ret, sol)
    ret = solveset(cos(2*x) - 0.5, x, S.Complexes)
    ret = ret.subs(ret.atoms(Dummy).pop(), n)
    assert all_close(ret, sol)

    # issue #21296 / #17667
    assert solveset(tan(x)-sqrt(2), x, Interval(0, pi/2)) == FiniteSet(atan(sqrt(2)))
    assert solveset(tan(x)-pi, x, Interval(0, pi/2)) == FiniteSet(atan(pi))

    # issue #17667
    # not yet working properly:
    # solveset(cos(x)-y, x, Interval(0, pi))
    assert solveset(cos(x)-y, x, S.Reals).dummy_eq(
        ConditionSet(x,(S(-1) <= y) & (y <= S(1)), Union(
            ImageSet(Lambda(n, 2*n*pi - acos(y)), S.Integers),
            ImageSet(Lambda(n, 2*n*pi + acos(y)), S.Integers))))

    # issue #17579
    # Valid result, but the intersection could potentially be simplified.
    assert solveset(sin(log(x)), x, Interval(0,1, True, False)).dummy_eq(
        Union(Intersection(ImageSet(Lambda(n, exp(2*n*pi)), S.Integers), Interval.Lopen(0, 1)),
              Intersection(ImageSet(Lambda(n, exp(2*n*pi + pi)), S.Integers), Interval.Lopen(0, 1))))

    # issue #17334
    assert solveset(sin(x) - sin(1), x, S.Reals).dummy_eq(Union(
        ImageSet(Lambda(n, 2*n*pi + 1), S.Integers),
        ImageSet(Lambda(n, 2*n*pi - 1 + pi), S.Integers)))
    assert solveset(sin(x) - sqrt(5)/3, x, S.Reals).dummy_eq(Union(
        ImageSet(Lambda(n, 2*n*pi + asin(sqrt(5)/3)), S.Integers),
        ImageSet(Lambda(n, 2*n*pi - asin(sqrt(5)/3) + pi), S.Integers)))
    assert solveset(sinh(x)-cosh(2), x, S.Reals) == FiniteSet(asinh(cosh(2)))

    # issue 9825
    assert solveset(Eq(tan(x), y), x, domain=S.Reals).dummy_eq(
        ConditionSet(x, (-oo < y) & (y < oo),
                     ImageSet(Lambda(n, n*pi + atan(y)), S.Integers)))
    r = Symbol('r', real=True)
    assert solveset(Eq(tan(x), r), x, domain=S.Reals).dummy_eq(
        ImageSet(Lambda(n, n*pi + atan(r)), S.Integers))


def test_solve_hyperbolic():
    # actual solver: _solve_trig1
    n = Dummy('n')
    assert solveset(sinh(x) + cosh(x), x) == S.EmptySet
    assert solveset(sinh(x) + cos(x), x) == ConditionSet(x,
        Eq(cos(x) + sinh(x), 0), S.Complexes)
    assert solveset_real(sinh(x) + sech(x), x) == FiniteSet(
        log(sqrt(sqrt(5) - 2)))
    assert solveset_real(cosh(2*x) + 2*sinh(x) - 5, x) == FiniteSet(
        log(-2 + sqrt(5)), log(1 + sqrt(2)))
    assert solveset_real((coth(x) + sinh(2*x))/cosh(x) - 3, x) == FiniteSet(
        log(S.Half + sqrt(5)/2), log(1 + sqrt(2)))
    assert solveset_real(cosh(x)*sinh(x) - 2, x) == FiniteSet(
        log(4 + sqrt(17))/2)
    assert solveset_real(sinh(x) + tanh(x) - 1, x) == FiniteSet(
        log(sqrt(2)/2 + sqrt(-S(1)/2 + sqrt(2))))

    assert dumeq(solveset_complex(sinh(x) + sech(x), x), Union(
        ImageSet(Lambda(n, 2*n*I*pi + log(sqrt(-2 + sqrt(5)))), S.Integers),
        ImageSet(Lambda(n, I*(2*n*pi + pi/2) + log(sqrt(2 + sqrt(5)))), S.Integers),
        ImageSet(Lambda(n, I*(2*n*pi + pi) + log(sqrt(-2 + sqrt(5)))), S.Integers),
        ImageSet(Lambda(n, I*(2*n*pi - pi/2) + log(sqrt(2 + sqrt(5)))), S.Integers)))

    assert dumeq(solveset(cosh(x/15) + cosh(x/5)), Union(
        ImageSet(Lambda(n, 15*I*(2*n*pi + pi/2)), S.Integers),
        ImageSet(Lambda(n, 15*I*(2*n*pi - pi/2)), S.Integers),
        ImageSet(Lambda(n, 15*I*(2*n*pi - 3*pi/4)), S.Integers),
        ImageSet(Lambda(n, 15*I*(2*n*pi + 3*pi/4)), S.Integers),
        ImageSet(Lambda(n, 15*I*(2*n*pi - pi/4)), S.Integers),
        ImageSet(Lambda(n, 15*I*(2*n*pi + pi/4)), S.Integers)))

    assert dumeq(solveset(tanh(pi*x) - coth(pi/2*x)), Union(
        ImageSet(Lambda(n, 2*I*(2*n*pi + pi/2)/pi), S.Integers),
        ImageSet(Lambda(n, 2*I*(2*n*pi - pi/2)/pi), S.Integers)))

    # issues #18490 / #19489
    assert solveset(cosh(x) + cosh(3*x) - cosh(5*x), x, S.Reals
        ).dummy_eq(ConditionSet(x,
        Eq(cosh(x) + cosh(3*x) - cosh(5*x), 0), S.Reals))
    assert solveset(sinh(8*x) + coth(12*x)).dummy_eq(
        ConditionSet(x, Eq(sinh(8*x) + coth(12*x), 0), S.Complexes))


def test_solve_trig_hyp_symbolic():
    # actual solver: invert_trig_hyp
    assert dumeq(solveset(sin(a*x), x), ConditionSet(x, Ne(a, 0), Union(
        ImageSet(Lambda(n, (2*n*pi + pi)/a), S.Integers),
        ImageSet(Lambda(n, 2*n*pi/a), S.Integers))))

    assert dumeq(solveset(cosh(x/a), x), ConditionSet(x, Ne(a, 0), Union(
        ImageSet(Lambda(n, a*(2*n*I*pi + I*pi/2)), S.Integers),
        ImageSet(Lambda(n, a*(2*n*I*pi + 3*I*pi/2)), S.Integers))))

    assert dumeq(solveset(sin(2*sqrt(3)/3*a**2/(b*pi)*x)
        + cos(4*sqrt(3)/3*a**2/(b*pi)*x), x),
       ConditionSet(x, Ne(b, 0) & Ne(a**2, 0), Union(
           ImageSet(Lambda(n, sqrt(3)*pi*b*(2*n*pi + pi/2)/(2*a**2)), S.Integers),
           ImageSet(Lambda(n, sqrt(3)*pi*b*(2*n*pi - 5*pi/6)/(2*a**2)), S.Integers),
           ImageSet(Lambda(n, sqrt(3)*pi*b*(2*n*pi - pi/6)/(2*a**2)), S.Integers))))

    assert dumeq(solveset(cosh((a**2 + 1)*x) - 3, x), ConditionSet(
        x, Ne(a**2 + 1, 0), Union(
            ImageSet(Lambda(n, (2*n*I*pi - acosh(3))/(a**2 + 1)), S.Integers),
            ImageSet(Lambda(n, (2*n*I*pi + acosh(3))/(a**2 + 1)), S.Integers))))

    ar = Symbol('ar', real=True)
    assert solveset(cosh((ar**2 + 1)*x) - 2, x, S.Reals) == FiniteSet(
        -acosh(2)/(ar**2 + 1), acosh(2)/(ar**2 + 1))

    # actual solver: _solve_trig1
    assert dumeq(simplify(solveset(cot((1 + I)*x) - cot((3 + 3*I)*x), x)), Union(
        ImageSet(Lambda(n, pi*(1 - I)*(4*n + 1)/4), S.Integers),
        ImageSet(Lambda(n, pi*(1 - I)*(4*n - 1)/4), S.Integers)))


def test_issue_9616():
    assert dumeq(solveset(sinh(x) + tanh(x) - 1, x), Union(
        ImageSet(Lambda(n, 2*n*I*pi + log(sqrt(2)/2 + sqrt(-S.Half + sqrt(2)))), S.Integers),
        ImageSet(Lambda(n, I*(2*n*pi - atan(sqrt(2)*sqrt(S.Half + sqrt(2))) + pi)
            + log(sqrt(1 + sqrt(2)))), S.Integers),
        ImageSet(Lambda(n, I*(2*n*pi + pi) + log(-sqrt(2)/2 + sqrt(-S.Half + sqrt(2)))), S.Integers),
        ImageSet(Lambda(n, I*(2*n*pi - pi + atan(sqrt(2)*sqrt(S.Half + sqrt(2))))
            + log(sqrt(1 + sqrt(2)))), S.Integers)))
    f1 = (sinh(x)).rewrite(exp)
    f2 = (tanh(x)).rewrite(exp)
    assert dumeq(solveset(f1 + f2 - 1, x), Union(
        Complement(ImageSet(
            Lambda(n, I*(2*n*pi + pi) + log(-sqrt(2)/2 + sqrt(-S.Half + sqrt(2)))), S.Integers),
            ImageSet(Lambda(n, I*(2*n*pi + pi)/2), S.Integers)),
        Complement(ImageSet(Lambda(n, I*(2*n*pi - pi + atan(sqrt(2)*sqrt(S.Half + sqrt(2))))
                + log(sqrt(1 + sqrt(2)))), S.Integers),
            ImageSet(Lambda(n, I*(2*n*pi + pi)/2), S.Integers)),
        Complement(ImageSet(Lambda(n, I*(2*n*pi - atan(sqrt(2)*sqrt(S.Half + sqrt(2))) + pi)
                + log(sqrt(1 + sqrt(2)))), S.Integers),
            ImageSet(Lambda(n, I*(2*n*pi + pi)/2), S.Integers)),
        Complement(
            ImageSet(Lambda(n, 2*n*I*pi + log(sqrt(2)/2 + sqrt(-S.Half + sqrt(2)))), S.Integers),
            ImageSet(Lambda(n, I*(2*n*pi + pi)/2), S.Integers))))


def test_solve_invalid_sol():
    assert 0 not in solveset_real(sin(x)/x, x)
    assert 0 not in solveset_complex((exp(x) - 1)/x, x)


@XFAIL
def test_solve_trig_simplified():
    n = Dummy('n')
    assert dumeq(solveset_real(sin(x), x),
        imageset(Lambda(n, n*pi), S.Integers))

    assert dumeq(solveset_real(cos(x), x),
        imageset(Lambda(n, n*pi + pi/2), S.Integers))

    assert dumeq(solveset_real(cos(x) + sin(x), x),
        imageset(Lambda(n, n*pi - pi/4), S.Integers))


@XFAIL
def test_solve_lambert():
    assert solveset_real(x*exp(x) - 1, x) == FiniteSet(LambertW(1))
    assert solveset_real(exp(x) + x, x) == FiniteSet(-LambertW(1))
    assert solveset_real(x + 2**x, x) == \
        FiniteSet(-LambertW(log(2))/log(2))

    # issue 4739
    ans = solveset_real(3*x + 5 + 2**(-5*x + 3), x)
    assert ans == FiniteSet(Rational(-5, 3) +
                            LambertW(-10240*2**Rational(1, 3)*log(2)/3)/(5*log(2)))

    eq = 2*(3*x + 4)**5 - 6*7**(3*x + 9)
    result = solveset_real(eq, x)
    ans = FiniteSet((log(2401) +
                     5*LambertW(-log(7**(7*3**Rational(1, 5)/5))))/(3*log(7))/-1)
    assert result == ans
    assert solveset_real(eq.expand(), x) == result

    assert solveset_real(5*x - 1 + 3*exp(2 - 7*x), x) == \
        FiniteSet(Rational(1, 5) + LambertW(-21*exp(Rational(3, 5))/5)/7)

    assert solveset_real(2*x + 5 + log(3*x - 2), x) == \
        FiniteSet(Rational(2, 3) + LambertW(2*exp(Rational(-19, 3))/3)/2)

    assert solveset_real(3*x + log(4*x), x) == \
        FiniteSet(LambertW(Rational(3, 4))/3)

    assert solveset_real(x**x - 2) == FiniteSet(exp(LambertW(log(2))))

    a = Symbol('a')
    assert solveset_real(-a*x + 2*x*log(x), x) == FiniteSet(exp(a/2))
    a = Symbol('a', real=True)
    assert solveset_real(a/x + exp(x/2), x) == \
        FiniteSet(2*LambertW(-a/2))
    assert solveset_real((a/x + exp(x/2)).diff(x), x) == \
        FiniteSet(4*LambertW(sqrt(2)*sqrt(a)/4))

    # coverage test
    assert solveset_real(tanh(x + 3)*tanh(x - 3) - 1, x) is S.EmptySet

    assert solveset_real((x**2 - 2*x + 1).subs(x, log(x) + 3*x), x) == \
        FiniteSet(LambertW(3*S.Exp1)/3)
    assert solveset_real((x**2 - 2*x + 1).subs(x, (log(x) + 3*x)**2 - 1), x) == \
        FiniteSet(LambertW(3*exp(-sqrt(2)))/3, LambertW(3*exp(sqrt(2)))/3)
    assert solveset_real((x**2 - 2*x - 2).subs(x, log(x) + 3*x), x) == \
        FiniteSet(LambertW(3*exp(1 + sqrt(3)))/3, LambertW(3*exp(-sqrt(3) + 1))/3)
    assert solveset_real(x*log(x) + 3*x + 1, x) == \
        FiniteSet(exp(-3 + LambertW(-exp(3))))
    eq = (x*exp(x) - 3).subs(x, x*exp(x))
    assert solveset_real(eq, x) == \
        FiniteSet(LambertW(3*exp(-LambertW(3))))

    assert solveset_real(3*log(a**(3*x + 5)) + a**(3*x + 5), x) == \
        FiniteSet(-((log(a**5) + LambertW(Rational(1, 3)))/(3*log(a))))
    p = symbols('p', positive=True)
    assert solveset_real(3*log(p**(3*x + 5)) + p**(3*x + 5), x) == \
        FiniteSet(
        log((-3**Rational(1, 3) - 3**Rational(5, 6)*I)*LambertW(Rational(1, 3))**Rational(1, 3)/(2*p**Rational(5, 3)))/log(p),
        log((-3**Rational(1, 3) + 3**Rational(5, 6)*I)*LambertW(Rational(1, 3))**Rational(1, 3)/(2*p**Rational(5, 3)))/log(p),
        log((3*LambertW(Rational(1, 3))/p**5)**(1/(3*log(p)))),)  # checked numerically
    # check collection
    b = Symbol('b')
    eq = 3*log(a**(3*x + 5)) + b*log(a**(3*x + 5)) + a**(3*x + 5)
    assert solveset_real(eq, x) == FiniteSet(
        -((log(a**5) + LambertW(1/(b + 3)))/(3*log(a))))

    # issue 4271
    assert solveset_real((a/x + exp(x/2)).diff(x, 2), x) == FiniteSet(
        6*LambertW((-1)**Rational(1, 3)*a**Rational(1, 3)/3))

    assert solveset_real(x**3 - 3**x, x) == \
        FiniteSet(-3/log(3)*LambertW(-log(3)/3))
    assert solveset_real(3**cos(x) - cos(x)**3) == FiniteSet(
        acos(-3*LambertW(-log(3)/3)/log(3)))

    assert solveset_real(x**2 - 2**x, x) == \
        solveset_real(-x**2 + 2**x, x)

    assert solveset_real(3*log(x) - x*log(3)) == FiniteSet(
        -3*LambertW(-log(3)/3)/log(3),
        -3*LambertW(-log(3)/3, -1)/log(3))

    assert solveset_real(LambertW(2*x) - y) == FiniteSet(
        y*exp(y)/2)


@XFAIL
def test_other_lambert():
    a = Rational(6, 5)
    assert solveset_real(x**a - a**x, x) == FiniteSet(
        a, -a*LambertW(-log(a)/a)/log(a))


@_both_exp_pow
def test_solveset():
    f = Function('f')
    raises(ValueError, lambda: solveset(x + y))
    assert solveset(x, 1) == S.EmptySet
    assert solveset(f(1)**2 + y + 1, f(1)
        ) == FiniteSet(-sqrt(-y - 1), sqrt(-y - 1))
    assert solveset(f(1)**2 - 1, f(1), S.Reals) == FiniteSet(-1, 1)
    assert solveset(f(1)**2 + 1, f(1)) == FiniteSet(-I, I)
    assert solveset(x - 1, 1) == FiniteSet(x)
    assert solveset(sin(x) - cos(x), sin(x)) == FiniteSet(cos(x))

    assert solveset(0, domain=S.Reals) == S.Reals
    assert solveset(1) == S.EmptySet
    assert solveset(True, domain=S.Reals) == S.Reals  # issue 10197
    assert solveset(False, domain=S.Reals) == S.EmptySet

    assert solveset(exp(x) - 1, domain=S.Reals) == FiniteSet(0)
    assert solveset(exp(x) - 1, x, S.Reals) == FiniteSet(0)
    assert solveset(Eq(exp(x), 1), x, S.Reals) == FiniteSet(0)
    assert solveset(exp(x) - 1, exp(x), S.Reals) == FiniteSet(1)
    A = Indexed('A', x)
    assert solveset(A - 1, A, S.Reals) == FiniteSet(1)

    assert solveset(x - 1 >= 0, x, S.Reals) == Interval(1, oo)
    assert solveset(exp(x) - 1 >= 0, x, S.Reals) == Interval(0, oo)

    assert dumeq(solveset(exp(x) - 1, x), imageset(Lambda(n, 2*I*pi*n), S.Integers))
    assert dumeq(solveset(Eq(exp(x), 1), x), imageset(Lambda(n, 2*I*pi*n),
                                                  S.Integers))
    # issue 13825
    assert solveset(x**2 + f(0) + 1, x) == {-sqrt(-f(0) - 1), sqrt(-f(0) - 1)}

    # issue 19977
    assert solveset(atan(log(x)) > 0, x, domain=Interval.open(0, oo)) == Interval.open(1, oo)


@_both_exp_pow
def test_multi_exp():
    k1, k2, k3 = symbols('k1, k2, k3')
    assert dumeq(solveset(exp(exp(x)) - 5, x),\
         imageset(Lambda(((k1, n),), I*(2*k1*pi + arg(2*n*I*pi + log(5))) + log(Abs(2*n*I*pi + log(5)))),\
             ProductSet(S.Integers, S.Integers)))
    assert dumeq(solveset((d*exp(exp(a*x + b)) + c), x),\
        imageset(Lambda(x, (-b + x)/a), ImageSet(Lambda(((k1, n),), \
            I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))) + log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d))))), \
                ProductSet(S.Integers, S.Integers))))

    assert dumeq(solveset((d*exp(exp(exp(a*x + b))) + c), x),\
        imageset(Lambda(x, (-b + x)/a), ImageSet(Lambda(((k2, k1, n),), \
            I*(2*k2*pi + arg(I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))) + \
                log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))))) + log(Abs(I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + \
                    log(Abs(c/d)))) + log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d))))))), \
                        ProductSet(S.Integers, S.Integers, S.Integers))))

    assert dumeq(solveset((d*exp(exp(exp(exp(a*x + b)))) + c), x),\
        ImageSet(Lambda(x, (-b + x)/a), ImageSet(Lambda(((k3, k2, k1, n),), \
            I*(2*k3*pi + arg(I*(2*k2*pi + arg(I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))) + \
                log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))))) + log(Abs(I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + \
                    log(Abs(c/d)))) + log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))))))) + log(Abs(I*(2*k2*pi + \
                        arg(I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))) + log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))))) + \
                            log(Abs(I*(2*k1*pi + arg(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d)))) + log(Abs(I*(2*n*pi + arg(-c/d)) + log(Abs(c/d))))))))), \
             ProductSet(S.Integers, S.Integers, S.Integers, S.Integers))))


def test__solveset_multi():
    from sympy.solvers.solveset import _solveset_multi
    from sympy.sets import Reals

    # Basic univariate case:
    assert _solveset_multi([x**2-1], [x], [S.Reals]) == FiniteSet((1,), (-1,))

    # Linear systems of two equations
    assert _solveset_multi([x+y, x+1], [x, y], [Reals, Reals]) == FiniteSet((-1, 1))
    assert _solveset_multi([x+y, x+1], [y, x], [Reals, Reals]) == FiniteSet((1, -1))
    assert _solveset_multi([x+y, x-y-1], [x, y], [Reals, Reals]) == FiniteSet((S(1)/2, -S(1)/2))
    assert _solveset_multi([x-1, y-2], [x, y], [Reals, Reals]) == FiniteSet((1, 2))
    # assert dumeq(_solveset_multi([x+y], [x, y], [Reals, Reals]), ImageSet(Lambda(x, (x, -x)), Reals))
    assert dumeq(_solveset_multi([x+y], [x, y], [Reals, Reals]), Union(
            ImageSet(Lambda(((x,),), (x, -x)), ProductSet(Reals)),
            ImageSet(Lambda(((y,),), (-y, y)), ProductSet(Reals))))
    assert _solveset_multi([x+y, x+y+1], [x, y], [Reals, Reals]) == S.EmptySet
    assert _solveset_multi([x+y, x-y, x-1], [x, y], [Reals, Reals]) == S.EmptySet
    assert _solveset_multi([x+y, x-y, x-1], [y, x], [Reals, Reals]) == S.EmptySet

    # Systems of three equations:
    assert _solveset_multi([x+y+z-1, x+y-z-2, x-y-z-3], [x, y, z], [Reals,
        Reals, Reals]) == FiniteSet((2, -S.Half, -S.Half))

    # Nonlinear systems:
    from sympy.abc import theta
    assert _solveset_multi([x**2+y**2-2, x+y], [x, y], [Reals, Reals]) == FiniteSet((-1, 1), (1, -1))
    assert _solveset_multi([x**2-1, y], [x, y], [Reals, Reals]) == FiniteSet((1, 0), (-1, 0))
    #assert _solveset_multi([x**2-y**2], [x, y], [Reals, Reals]) == Union(
    #        ImageSet(Lambda(x, (x, -x)), Reals), ImageSet(Lambda(x, (x, x)), Reals))
    assert dumeq(_solveset_multi([x**2-y**2], [x, y], [Reals, Reals]), Union(
            ImageSet(Lambda(((x,),), (x, -Abs(x))), ProductSet(Reals)),
            ImageSet(Lambda(((x,),), (x, Abs(x))), ProductSet(Reals)),
            ImageSet(Lambda(((y,),), (-Abs(y), y)), ProductSet(Reals)),
            ImageSet(Lambda(((y,),), (Abs(y), y)), ProductSet(Reals))))
    assert _solveset_multi([r*cos(theta)-1, r*sin(theta)], [theta, r],
            [Interval(0, pi), Interval(-1, 1)]) == FiniteSet((0, 1), (pi, -1))
    assert _solveset_multi([r*cos(theta)-1, r*sin(theta)], [r, theta],
            [Interval(0, 1), Interval(0, pi)]) == FiniteSet((1, 0))
    assert _solveset_multi([r*cos(theta)-r, r*sin(theta)], [r, theta],
           [Interval(0, 1), Interval(0, pi)]) == Union(
           ImageSet(Lambda(((r,),), (r, 0)),
           ImageSet(Lambda(r, (r,)), Interval(0, 1))),
           ImageSet(Lambda(((theta,),), (0, theta)),
           ImageSet(Lambda(theta, (theta,)), Interval(0, pi))))


def test_conditionset():
    assert solveset(Eq(sin(x)**2 + cos(x)**2, 1), x, domain=S.Reals
        ) is S.Reals

    assert solveset(Eq(x**2 + x*sin(x), 1), x, domain=S.Reals
        ).dummy_eq(ConditionSet(x, Eq(x**2 + x*sin(x) - 1, 0), S.Reals))

    assert dumeq(solveset(Eq(-I*(exp(I*x) - exp(-I*x))/2, 1), x
        ), imageset(Lambda(n, 2*n*pi + pi/2), S.Integers))

    assert solveset(x + sin(x) > 1, x, domain=S.Reals
        ).dummy_eq(ConditionSet(x, x + sin(x) > 1, S.Reals))

    assert solveset(Eq(sin(Abs(x)), x), x, domain=S.Reals
        ).dummy_eq(ConditionSet(x, Eq(-x + sin(Abs(x)), 0), S.Reals))

    assert solveset(y**x-z, x, S.Reals
        ).dummy_eq(ConditionSet(x, Eq(y**x - z, 0), S.Reals))


@XFAIL
def test_conditionset_equality():
    ''' Checking equality of different representations of ConditionSet'''
    assert solveset(Eq(tan(x), y), x) == ConditionSet(x, Eq(tan(x), y), S.Complexes)


def test_solveset_domain():
    assert solveset(x**2 - x - 6, x, Interval(0, oo)) == FiniteSet(3)
    assert solveset(x**2 - 1, x, Interval(0, oo)) == FiniteSet(1)
    assert solveset(x**4 - 16, x, Interval(0, 10)) == FiniteSet(2)


def test_improve_coverage():
    solution = solveset(exp(x) + sin(x), x, S.Reals)
    unsolved_object = ConditionSet(x, Eq(exp(x) + sin(x), 0), S.Reals)
    assert solution.dummy_eq(unsolved_object)


def test_issue_9522():
    expr1 = Eq(1/(x**2 - 4) + x, 1/(x**2 - 4) + 2)
    expr2 = Eq(1/x + x, 1/x)

    assert solveset(expr1, x, S.Reals) is S.EmptySet
    assert solveset(expr2, x, S.Reals) is S.EmptySet


def test_solvify():
    assert solvify(x**2 + 10, x, S.Reals) == []
    assert solvify(x**3 + 1, x, S.Complexes) == [-1, S.Half - sqrt(3)*I/2,
                                                 S.Half + sqrt(3)*I/2]
    assert solvify(log(x), x, S.Reals) == [1]
    assert solvify(cos(x), x, S.Reals) == [pi/2, pi*Rational(3, 2)]
    assert solvify(sin(x) + 1, x, S.Reals) == [pi*Rational(3, 2)]
    raises(NotImplementedError, lambda: solvify(sin(exp(x)), x, S.Complexes))


def test_solvify_piecewise():
    p1 = Piecewise((0, x < -1), (x**2, x <= 1), (log(x), True))
    p2 = Piecewise((0, x < -10), (x**2 + 5*x - 6, x >= -9))
    p3 = Piecewise((0, Eq(x, 0)), (x**2/Abs(x), True))
    p4 = Piecewise((0, Eq(x, pi)), ((x - pi)/sin(x), True))

    # issue 21079
    assert solvify(p1, x, S.Reals) == [0]
    assert solvify(p2, x, S.Reals) == [-6, 1]
    assert solvify(p3, x, S.Reals) == [0]
    assert solvify(p4, x, S.Reals) == [pi]


def test_abs_invert_solvify():

    x = Symbol('x',positive=True)
    assert solvify(sin(Abs(x)), x, S.Reals) == [0, pi]
    x = Symbol('x')
    assert solvify(sin(Abs(x)), x, S.Reals) is None


def test_linear_eq_to_matrix():
    assert linear_eq_to_matrix(0, x) == (Matrix([[0]]), Matrix([[0]]))
    assert linear_eq_to_matrix(1, x) == (Matrix([[0]]), Matrix([[-1]]))

    # integer coefficients
    eqns1 = [2*x + y - 2*z - 3, x - y - z, x + y + 3*z - 12]
    eqns2 = [Eq(3*x + 2*y - z, 1), Eq(2*x - 2*y + 4*z, -2), -2*x + y - 2*z]

    A, B = linear_eq_to_matrix(eqns1, x, y, z)
    assert A == Matrix([[2, 1, -2], [1, -1, -1], [1, 1, 3]])
    assert B == Matrix([[3], [0], [12]])

    A, B = linear_eq_to_matrix(eqns2, x, y, z)
    assert A == Matrix([[3, 2, -1], [2, -2, 4], [-2, 1, -2]])
    assert B == Matrix([[1], [-2], [0]])

    # Pure symbolic coefficients
    eqns3 = [a*b*x + b*y + c*z - d, e*x + d*x + f*y + g*z - h, i*x + j*y + k*z - l]
    A, B = linear_eq_to_matrix(eqns3, x, y, z)
    assert A == Matrix([[a*b, b, c], [d + e, f, g], [i, j, k]])
    assert B == Matrix([[d], [h], [l]])

    # raise Errors if
    # 1) no symbols are given
    raises(ValueError, lambda: linear_eq_to_matrix(eqns3))
    # 2) there are duplicates
    raises(ValueError, lambda: linear_eq_to_matrix(eqns3, [x, x, y]))
    # 3) a nonlinear term is detected in the original expression
    raises(NonlinearError, lambda: linear_eq_to_matrix(Eq(1/x + x, 1/x), [x]))
    raises(NonlinearError, lambda: linear_eq_to_matrix([x**2], [x]))
    raises(NonlinearError, lambda: linear_eq_to_matrix([x*y], [x, y]))
    # 4) Eq being used to represent equations autoevaluates
    # (use unevaluated Eq instead)
    raises(ValueError, lambda: linear_eq_to_matrix(Eq(x, x), x))
    raises(ValueError, lambda: linear_eq_to_matrix(Eq(x, x + 1), x))


    # if non-symbols are passed, the user is responsible for interpreting
    assert linear_eq_to_matrix([x], [1/x]) == (Matrix([[0]]), Matrix([[-x]]))

    # issue 15195
    assert linear_eq_to_matrix(x + y*(z*(3*x + 2) + 3), x) == (
        Matrix([[3*y*z + 1]]), Matrix([[-y*(2*z + 3)]]))
    assert linear_eq_to_matrix(Matrix(
        [[a*x + b*y - 7], [5*x + 6*y - c]]), x, y) == (
        Matrix([[a, b], [5, 6]]), Matrix([[7], [c]]))

    # issue 15312
    assert linear_eq_to_matrix(Eq(x + 2, 1), x) == (
        Matrix([[1]]), Matrix([[-1]]))

    # issue 25423
    raises(TypeError, lambda: linear_eq_to_matrix([], {x, y}))
    raises(TypeError, lambda: linear_eq_to_matrix([x + y], {x, y}))
    raises(ValueError, lambda: linear_eq_to_matrix({x + y}, (x, y)))


def test_issue_16577():
    assert linear_eq_to_matrix(Eq(a*(2*x + 3*y) + 4*y, 5), x, y) == (
        Matrix([[2*a, 3*a + 4]]), Matrix([[5]]))


def test_issue_10085():
    assert invert_real(exp(x),0,x) == (x, S.EmptySet)


def test_linsolve():
    x1, x2, x3, x4 = symbols('x1, x2, x3, x4')

    # Test for different input forms

    M = Matrix([[1, 2, 1, 1, 7], [1, 2, 2, -1, 12], [2, 4, 0, 6, 4]])
    system1 = A, B = M[:, :-1], M[:, -1]
    Eqns = [x1 + 2*x2 + x3 + x4 - 7, x1 + 2*x2 + 2*x3 - x4 - 12,
            2*x1 + 4*x2 + 6*x4 - 4]

    sol = FiniteSet((-2*x2 - 3*x4 + 2, x2, 2*x4 + 5, x4))
    assert linsolve(Eqns, (x1, x2, x3, x4)) == sol
    assert linsolve(Eqns, *(x1, x2, x3, x4)) == sol
    assert linsolve(system1, (x1, x2, x3, x4)) == sol
    assert linsolve(system1, *(x1, x2, x3, x4)) == sol
    # issue 9667 - symbols can be Dummy symbols
    x1, x2, x3, x4 = symbols('x:4', cls=Dummy)
    assert linsolve(system1, x1, x2, x3, x4) == FiniteSet(
        (-2*x2 - 3*x4 + 2, x2, 2*x4 + 5, x4))

    # raise ValueError for garbage value
    raises(ValueError, lambda: linsolve(Eqns))
    raises(ValueError, lambda: linsolve(x1))
    raises(ValueError, lambda: linsolve(x1, x2))
    raises(ValueError, lambda: linsolve((A,), x1, x2))
    raises(ValueError, lambda: linsolve(A, B, x1, x2))
    raises(ValueError, lambda: linsolve([x1], x1, x1))
    raises(ValueError, lambda: linsolve([x1], (i for i in (x1, x1))))

    #raise ValueError if equations are non-linear in given variables
    raises(NonlinearError, lambda: linsolve([x + y - 1, x ** 2 + y - 3], [x, y]))
    raises(NonlinearError, lambda: linsolve([cos(x) + y, x + y], [x, y]))
    assert linsolve([x + z - 1, x ** 2 + y - 3], [z, y]) == {(-x + 1, -x**2 + 3)}

    # Fully symbolic test
    A = Matrix([[a, b], [c, d]])
    B = Matrix([[e], [g]])
    system2 = (A, B)
    sol = FiniteSet(((-b*g + d*e)/(a*d - b*c), (a*g - c*e)/(a*d - b*c)))
    assert linsolve(system2, [x, y]) == sol

    # No solution
    A = Matrix([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    B = Matrix([0, 0, 1])
    assert linsolve((A, B), (x, y, z)) is S.EmptySet

    # Issue #10056
    A, B, J1, J2 = symbols('A B J1 J2')
    Augmatrix = Matrix([
        [2*I*J1, 2*I*J2, -2/J1],
        [-2*I*J2, -2*I*J1, 2/J2],
        [0, 2, 2*I/(J1*J2)],
        [2, 0,  0],
        ])

    assert linsolve(Augmatrix, A, B) == FiniteSet((0, I/(J1*J2)))

    # Issue #10121 - Assignment of free variables
    Augmatrix = Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
    assert linsolve(Augmatrix, a, b, c, d, e) == FiniteSet((a, 0, c, 0, e))
    #raises(IndexError, lambda: linsolve(Augmatrix, a, b, c))

    x0, x1, x2, _x0 = symbols('tau0 tau1 tau2 _tau0')
    assert linsolve(Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, _x0]])
        ) == FiniteSet((x0, 0, x1, _x0, x2))
    x0, x1, x2, _x0 = symbols('tau00 tau01 tau02 tau0')
    assert linsolve(Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, _x0]])
        ) == FiniteSet((x0, 0, x1, _x0, x2))
    x0, x1, x2, _x0 = symbols('tau00 tau01 tau02 tau1')
    assert linsolve(Matrix([[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, _x0]])
        ) == FiniteSet((x0, 0, x1, _x0, x2))
    # symbols can be given as generators
    x0, x2, x4 = symbols('x0, x2, x4')
    assert linsolve(Augmatrix, numbered_symbols('x')
        ) == FiniteSet((x0, 0, x2, 0, x4))
    Augmatrix[-1, -1] = x0
    # use Dummy to avoid clash; the names may clash but the symbols
    # will not
    Augmatrix[-1, -1] = symbols('_x0')
    assert len(linsolve(
        Augmatrix, numbered_symbols('x', cls=Dummy)).free_symbols) == 4

    # Issue #12604
    f = Function('f')
    assert linsolve([f(x) - 5], f(x)) == FiniteSet((5,))

    # Issue #14860
    from sympy.physics.units import meter, newton, kilo
    kN = kilo*newton
    Eqns = [8*kN + x + y, 28*kN*meter + 3*x*meter]
    assert linsolve(Eqns, x, y) == {
            (kilo*newton*Rational(-28, 3), kN*Rational(4, 3))}

    # linsolve does not allow expansion (real or implemented)
    # to remove singularities, but it will cancel linear terms
    assert linsolve([Eq(x, x + y)], [x, y]) == {(x, 0)}
    assert linsolve([Eq(x + x*y, 1 + y)], [x]) == {(1,)}
    assert linsolve([Eq(1 + y, x + x*y)], [x]) == {(1,)}
    raises(NonlinearError, lambda:
        linsolve([Eq(x**2, x**2 + y)], [x, y]))

    # corner cases
    #
    # XXX: The case below should give the same as for [0]
    # assert linsolve([], [x]) == {(x,)}
    assert linsolve([], [x]) is S.EmptySet
    assert linsolve([0], [x]) == {(x,)}
    assert linsolve([x], [x, y]) == {(0, y)}
    assert linsolve([x, 0], [x, y]) == {(0, y)}


def test_linsolve_large_sparse():
    #
    # This is mainly a performance test
    #

    def _mk_eqs_sol(n):
        xs = symbols('x:{}'.format(n))
        ys = symbols('y:{}'.format(n))
        syms = xs + ys
        eqs = []
        sol = (-S.Half,) * n + (S.Half,) * n
        for xi, yi in zip(xs, ys):
            eqs.extend([xi + yi, xi - yi + 1])
        return eqs, syms, FiniteSet(sol)

    n = 500
    eqs, syms, sol = _mk_eqs_sol(n)
    assert linsolve(eqs, syms) == sol


def test_linsolve_immutable():
    A = ImmutableDenseMatrix([[1, 1, 2], [0, 1, 2], [0, 0, 1]])
    B = ImmutableDenseMatrix([2, 1, -1])
    assert linsolve([A, B], (x, y, z)) == FiniteSet((1, 3, -1))

    A = ImmutableDenseMatrix([[1, 1, 7], [1, -1, 3]])
    assert linsolve(A) == FiniteSet((5, 2))


def test_solve_decomposition():
    n = Dummy('n')

    f1 = exp(3*x) - 6*exp(2*x) + 11*exp(x) - 6
    f2 = sin(x)**2 - 2*sin(x) + 1
    f3 = sin(x)**2 - sin(x)
    f4 = sin(x + 1)
    f5 = exp(x + 2) - 1
    f6 = 1/log(x)
    f7 = 1/x

    s1 = ImageSet(Lambda(n, 2*n*pi), S.Integers)
    s2 = ImageSet(Lambda(n, 2*n*pi + pi), S.Integers)
    s3 = ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers)
    s4 = ImageSet(Lambda(n, 2*n*pi - 1), S.Integers)
    s5 = ImageSet(Lambda(n, 2*n*pi - 1 + pi), S.Integers)

    assert solve_decomposition(f1, x, S.Reals) == FiniteSet(0, log(2), log(3))
    assert dumeq(solve_decomposition(f2, x, S.Reals), s3)
    assert dumeq(solve_decomposition(f3, x, S.Reals), Union(s1, s2, s3))
    assert dumeq(solve_decomposition(f4, x, S.Reals), Union(s4, s5))
    assert solve_decomposition(f5, x, S.Reals) == FiniteSet(-2)
    assert solve_decomposition(f6, x, S.Reals) == S.EmptySet
    assert solve_decomposition(f7, x, S.Reals) == S.EmptySet
    assert solve_decomposition(x, x, Interval(1, 2)) == S.EmptySet


# nonlinsolve testcases
def test_nonlinsolve_basic():
    assert nonlinsolve([],[]) == S.EmptySet
    assert nonlinsolve([],[x, y]) == S.EmptySet

    system = [x, y - x - 5]
    assert nonlinsolve([x],[x, y]) == FiniteSet((0, y))
    assert nonlinsolve(system, [y]) == S.EmptySet
    soln = (ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers),)
    assert dumeq(nonlinsolve([sin(x) - 1], [x]), FiniteSet(tuple(soln)))
    soln = ((ImageSet(Lambda(n, 2*n*pi + pi), S.Integers), 1),
            (ImageSet(Lambda(n, 2*n*pi), S.Integers), 1))
    assert dumeq(nonlinsolve([sin(x), y - 1], [x, y]), FiniteSet(*soln))
    assert nonlinsolve([x**2 - 1], [x]) == FiniteSet((-1,), (1,))

    soln = FiniteSet((y, y))
    assert nonlinsolve([x - y, 0], x, y) == soln
    assert nonlinsolve([0, x - y], x, y) == soln
    assert nonlinsolve([x - y, x - y], x, y) == soln
    assert nonlinsolve([x, 0], x, y) == FiniteSet((0, y))
    f = Function('f')
    assert nonlinsolve([f(x), 0], f(x), y) == FiniteSet((0, y))
    assert nonlinsolve([f(x), 0], f(x), f(y)) == FiniteSet((0, f(y)))
    A = Indexed('A', x)
    assert nonlinsolve([A, 0], A, y) == FiniteSet((0, y))
    assert nonlinsolve([x**2 -1], [sin(x)]) == FiniteSet((S.EmptySet,))
    assert nonlinsolve([x**2 -1], sin(x)) == FiniteSet((S.EmptySet,))
    assert nonlinsolve([x**2 -1], 1) == FiniteSet((x**2,))
    assert nonlinsolve([x**2 -1], x + y) == FiniteSet((S.EmptySet,))
    assert nonlinsolve([Eq(1, x + y), Eq(1, -x + y - 1), Eq(1, -x + y - 1)], x, y) == FiniteSet(
        (-S.Half, 3*S.Half))


def test_nonlinsolve_abs():
    soln = FiniteSet((y, y), (-y, y))
    assert nonlinsolve([Abs(x) - y], x, y) == soln


def test_raise_exception_nonlinsolve():
    raises(IndexError, lambda: nonlinsolve([x**2 -1], []))
    raises(ValueError, lambda: nonlinsolve([x**2 -1]))


def test_trig_system():
    # TODO: add more simple testcases when solveset returns
    # simplified soln for Trig eq
    assert nonlinsolve([sin(x) - 1, cos(x) -1 ], x) == S.EmptySet
    soln1 = (ImageSet(Lambda(n, 2*n*pi + pi/2), S.Integers),)
    soln = FiniteSet(soln1)
    assert dumeq(nonlinsolve([sin(x) - 1, cos(x)], x), soln)


@XFAIL
def test_trig_system_fail():
    # fails because solveset trig solver is not much smart.
    sys = [x + y - pi/2, sin(x) + sin(y) - 1]
    # solveset returns conditionset for sin(x) + sin(y) - 1
    soln_1 = (ImageSet(Lambda(n, n*pi + pi/2), S.Integers),
        ImageSet(Lambda(n, n*pi), S.Integers))
    soln_1 = FiniteSet(soln_1)
    soln_2 = (ImageSet(Lambda(n, n*pi), S.Integers),
        ImageSet(Lambda(n, n*pi+ pi/2), S.Integers))
    soln_2 = FiniteSet(soln_2)
    soln = soln_1 + soln_2
    assert dumeq(nonlinsolve(sys, [x, y]), soln)

    # Add more cases from here
    # http://www.vitutor.com/geometry/trigonometry/equations_systems.html#uno
    sys = [sin(x) + sin(y) - (sqrt(3)+1)/2, sin(x) - sin(y) - (sqrt(3) - 1)/2]
    soln_x = Union(ImageSet(Lambda(n, 2*n*pi + pi/3), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + pi*Rational(2, 3)), S.Integers))
    soln_y = Union(ImageSet(Lambda(n, 2*n*pi + pi/6), S.Integers),
        ImageSet(Lambda(n, 2*n*pi + pi*Rational(5, 6)), S.Integers))
    assert dumeq(nonlinsolve(sys, [x, y]), FiniteSet((soln_x, soln_y)))


def test_nonlinsolve_positive_dimensional():
    x, y, a, b, c, d = symbols('x, y, a, b, c, d', extended_real=True)
    assert nonlinsolve([x*y, x*y - x], [x, y]) == FiniteSet((0, y))

    system = [a**2 + a*c, a - b]
    assert nonlinsolve(system, [a, b]) == FiniteSet((0, 0), (-c, -c))
    # here (a= 0, b = 0) is independent soln so both is printed.
    # if symbols = [a, b, c] then only {a : -c ,b : -c}

    eq1 =  a + b + c + d
    eq2 = a*b + b*c + c*d + d*a
    eq3 = a*b*c + b*c*d + c*d*a + d*a*b
    eq4 = a*b*c*d - 1
    system = [eq1, eq2, eq3, eq4]
    sol1 = (-1/d, -d, 1/d, FiniteSet(d) - FiniteSet(0))
    sol2 = (1/d, -d, -1/d, FiniteSet(d) - FiniteSet(0))
    soln = FiniteSet(sol1, sol2)
    assert nonlinsolve(system, [a, b, c, d]) == soln

    assert nonlinsolve([x**4 - 3*x**2 + y*x, x*z**2, y*z - 1], [x, y, z]) == \
           {(0, 1/z, z)}


def test_nonlinsolve_polysys():
    x, y, z = symbols('x, y, z', real=True)
    assert nonlinsolve([x**2 + y - 2, x**2 + y], [x, y]) == S.EmptySet

    s = (-y + 2, y)
    assert nonlinsolve([(x + y)**2 - 4, x + y - 2], [x, y]) == FiniteSet(s)

    system = [x**2 - y**2]
    soln_real = FiniteSet((-y, y), (y, y))
    soln_complex = FiniteSet((-Abs(y), y), (Abs(y), y))
    soln =soln_real + soln_complex
    assert nonlinsolve(system, [x, y]) == soln

    system = [x**2 - y**2]
    soln_real= FiniteSet((y, -y), (y, y))
    soln_complex = FiniteSet((y, -Abs(y)), (y, Abs(y)))
    soln = soln_real + soln_complex
    assert nonlinsolve(system, [y, x]) == soln

    system = [x**2 + y - 3, x - y - 4]
    assert nonlinsolve(system, (x, y)) != nonlinsolve(system, (y, x))

    assert nonlinsolve([-x**2 - y**2 + z, -2*x, -2*y, S.One], [x, y, z]) == S.EmptySet
    assert nonlinsolve([x + y + z, S.One, S.One, S.One], [x, y, z]) == S.EmptySet

    system = [-x**2*z**2 + x*y*z + y**4, -2*x*z**2 + y*z, x*z + 4*y**3, -2*x**2*z + x*y]
    assert nonlinsolve(system, [x, y, z]) == FiniteSet((0, 0, z), (x, 0, 0))


def test_nonlinsolve_using_substitution():
    x, y, z, n = symbols('x, y, z, n', real = True)
    system = [(x + y)*n - y**2 + 2]
    s_x = (n*y - y**2 + 2)/n
    soln = (-s_x, y)
    assert nonlinsolve(system, [x, y]) == FiniteSet(soln)

    system = [z**2*x**2 - z**2*y**2/exp(x)]
    soln_real_1 = (y, x, 0)
    soln_real_2 = (-exp(x/2)*Abs(x), x, z)
    soln_real_3 = (exp(x/2)*Abs(x), x, z)
    soln_complex_1 = (-x*exp(x/2), x, z)
    soln_complex_2 = (x*exp(x/2), x, z)
    syms = [y, x, z]
    soln = FiniteSet(soln_real_1, soln_complex_1, soln_complex_2,\
        soln_real_2, soln_real_3)
    assert nonlinsolve(system,syms) == soln


def test_nonlinsolve_complex():
    n = Dummy('n')
    assert dumeq(nonlinsolve([exp(x) - sin(y), 1/y - 3], [x, y]), {
        (ImageSet(Lambda(n, 2*n*I*pi + log(sin(Rational(1, 3)))), S.Integers), Rational(1, 3))})

    system = [exp(x) - sin(y), 1/exp(y) - 3]
    assert dumeq(nonlinsolve(system, [x, y]), {
        (ImageSet(Lambda(n, I*(2*n*pi + pi)
                         + log(sin(log(3)))), S.Integers), -log(3)),
        (ImageSet(Lambda(n, I*(2*n*pi + arg(sin(2*n*I*pi - log(3))))
                         + log(Abs(sin(2*n*I*pi - log(3))))), S.Integers),
        ImageSet(Lambda(n, 2*n*I*pi - log(3)), S.Integers))})

    system = [exp(x) - sin(y), y**2 - 4]
    assert dumeq(nonlinsolve(system, [x, y]), {
        (ImageSet(Lambda(n, I*(2*n*pi + pi) + log(sin(2))), S.Integers), -2),
        (ImageSet(Lambda(n, 2*n*I*pi + log(sin(2))), S.Integers), 2)})

    system = [exp(x) - 2, y ** 2 - 2]
    assert dumeq(nonlinsolve(system, [x, y]), {
        (log(2), -sqrt(2)), (log(2), sqrt(2)),
        (ImageSet(Lambda(n, 2*n*I*pi + log(2)), S.Integers), -sqrt(2)),
        (ImageSet(Lambda(n, 2 * n * I * pi + log(2)), S.Integers), sqrt(2))})


def test_nonlinsolve_radical():
    assert nonlinsolve([sqrt(y) - x - z, y - 1], [x, y, z]) == {(1 - z, 1, z)}


def test_nonlinsolve_inexact():
    sol = [(-1.625, -1.375), (1.625, 1.375)]
    res = nonlinsolve([(x + y)**2 - 9, x**2 - y**2 - 0.75], [x, y])
    assert all(abs(res.args[i][j]-sol[i][j]) < 1e-9
               for i in range(2) for j in range(2))

    assert nonlinsolve([(x + y)**2 - 9, (x + y)**2 - 0.75], [x, y]) == S.EmptySet

    assert nonlinsolve([y**2 + (x - 0.5)**2 - 0.0625, 2*x - 1.0, 2*y], [x, y]) == \
           S.EmptySet

    res = nonlinsolve([x**2 + y - 0.5, (x + y)**2, log(z)], [x, y, z])
    sol = [(-0.366025403784439, 0.366025403784439, 1),
           (-0.366025403784439, 0.366025403784439, 1),
           (1.36602540378444, -1.36602540378444, 1)]
    assert all(abs(res.args[i][j]-sol[i][j]) < 1e-9
               for i in range(3) for j in range(3))

    res = nonlinsolve([y - x**2, x**5 - x + 1.0], [x, y])
    sol = [(-1.16730397826142, 1.36259857766493),
           (-0.181232444469876 - 1.08395410131771*I,
            -1.14211129483496 + 0.392895302949911*I),
           (-0.181232444469876 + 1.08395410131771*I,
            -1.14211129483496 - 0.392895302949911*I),
           (0.764884433600585 - 0.352471546031726*I,
            0.460812006002492 - 0.539199997693599*I),
           (0.764884433600585 + 0.352471546031726*I,
            0.460812006002492 + 0.539199997693599*I)]
    assert all(abs(res.args[i][j] - sol[i][j]) < 1e-9
               for i in range(5) for j in range(2))

@XFAIL
def test_solve_nonlinear_trans():
    # After the transcendental equation solver these will work
    x, y = symbols('x, y', real=True)
    soln1 = FiniteSet((2*LambertW(y/2), y))
    soln2 = FiniteSet((-x*sqrt(exp(x)), y), (x*sqrt(exp(x)), y))
    soln3 = FiniteSet((x*exp(x/2), x))
    soln4 = FiniteSet(2*LambertW(y/2), y)
    assert nonlinsolve([x**2 - y**2/exp(x)], [x, y]) == soln1
    assert nonlinsolve([x**2 - y**2/exp(x)], [y, x]) == soln2
    assert nonlinsolve([x**2 - y**2/exp(x)], [y, x]) == soln3
    assert nonlinsolve([x**2 - y**2/exp(x)], [x, y]) == soln4


def test_nonlinsolve_issue_25182():
    a1, b1, c1, ca, cb, cg = symbols('a1, b1, c1, ca, cb, cg')
    eq1 = a1*a1 + b1*b1 - 2.*a1*b1*cg - c1*c1
    eq2 = a1*a1 + c1*c1 - 2.*a1*c1*cb - b1*b1
    eq3 = b1*b1 + c1*c1 - 2.*b1*c1*ca - a1*a1
    assert nonlinsolve([eq1, eq2, eq3], [c1, cb, cg]) == FiniteSet(
        (1.0*b1*ca - 1.0*sqrt(a1**2 + b1**2*ca**2 - b1**2),
        -1.0*sqrt(a1**2 + b1**2*ca**2 - b1**2)/a1,
        -1.0*b1*(ca - 1)*(ca + 1)/a1 + 1.0*ca*sqrt(a1**2 + b1**2*ca**2 - b1**2)/a1),
        (1.0*b1*ca + 1.0*sqrt(a1**2 + b1**2*ca**2 - b1**2),
        1.0*sqrt(a1**2 + b1**2*ca**2 - b1**2)/a1,
        -1.0*b1*(ca - 1)*(ca + 1)/a1 - 1.0*ca*sqrt(a1**2 + b1**2*ca**2 - b1**2)/a1))


def test_issue_14642():
    x = Symbol('x')
    n1 = 0.5*x**3+x**2+0.5+I #add I in the Polynomials
    solution = solveset(n1, x)
    assert abs(solution.args[0] - (-2.28267560928153 - 0.312325580497716*I)) <= 1e-9
    assert abs(solution.args[1] - (-0.297354141679308 + 1.01904778618762*I)) <= 1e-9
    assert abs(solution.args[2] - (0.580029750960839 - 0.706722205689907*I)) <= 1e-9

    # Symbolic
    n1 = S.Half*x**3+x**2+S.Half+I
    res = FiniteSet(-((3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 +
            S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(S(172)/49)
            /2)/2)**2)**(S(1)/6)*cos(atan((27 + 3*sqrt(3)*31985**(S(1)/4)*
            cos(atan(S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*sin(atan(
            S(172)/49)/2)/2 + S(43)/2))/3)/3 - S(2)/3 - 4*cos(atan((27 +
            3*sqrt(3)*31985**(S(1)/4)*cos(atan(S(172)/49)/2)/2)/(3*sqrt(3)*
            31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 + S(43)/2))/3)/(3*((3*
            sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 + S(43)/2)**2 +
            (27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(S(172)/49)/2)/2)**2)**(S(1)/
            6)) + I*(-((3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 +
            S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(S(172)/49)/
            2)/2)**2)**(S(1)/6)*sin(atan((27 + 3*sqrt(3)*31985**(S(1)/4)*cos(
            atan(S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)
            /2)/2 + S(43)/2))/3)/3 + 4*sin(atan((27 + 3*sqrt(3)*31985**(S(1)/4)*
            cos(atan(S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)
            /49)/2)/2 + S(43)/2))/3)/(3*((3*sqrt(3)*31985**(S(1)/4)*sin(atan(
            S(172)/49)/2)/2 + S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*
            cos(atan(S(172)/49)/2)/2)**2)**(S(1)/6))), -S(2)/3 - sqrt(3)*((3*
            sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 + S(43)/2)**2 +
            (27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(S(172)/49)/2)/2)**2)**(S(1)
            /6)*sin(atan((27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(S(172)/49)/2)
            /2)/(3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 + S(43)/2))
            /3)/6 - 4*re(1/((-S(1)/2 - sqrt(3)*I/2)*(S(43)/2 + 27*I + sqrt(-256 +
            (43 + 54*I)**2)/2)**(S(1)/3)))/3 + ((3*sqrt(3)*31985**(S(1)/4)*sin(
            atan(S(172)/49)/2)/2 + S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*
            cos(atan(S(172)/49)/2)/2)**2)**(S(1)/6)*cos(atan((27 + 3*sqrt(3)*
            31985**(S(1)/4)*cos(atan(S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*
            sin(atan(S(172)/49)/2)/2 + S(43)/2))/3)/6 + I*(-4*im(1/((-S(1)/2 -
            sqrt(3)*I/2)*(S(43)/2 + 27*I + sqrt(-256 + (43 + 54*I)**2)/2)**(S(1)/
            3)))/3 + ((3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 +
            S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(S(172)/49)/2)
            /2)**2)**(S(1)/6)*sin(atan((27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(
            S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 +
            S(43)/2))/3)/6 + sqrt(3)*((3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/
            49)/2)/2 + S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(
            S(172)/49)/2)/2)**2)**(S(1)/6)*cos(atan((27 + 3*sqrt(3)*31985**(S(1)/
            4)*cos(atan(S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*sin(atan(
            S(172)/49)/2)/2 + S(43)/2))/3)/6), -S(2)/3 - 4*re(1/((-S(1)/2 +
            sqrt(3)*I/2)*(S(43)/2 + 27*I + sqrt(-256 + (43 + 54*I)**2)/2)**(S(1)
            /3)))/3 + sqrt(3)*((3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 +
            S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(S(172)/49)/2)
            /2)**2)**(S(1)/6)*sin(atan((27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(
            S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 +
            S(43)/2))/3)/6 + ((3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 +
            S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(S(172)/49)/2)
            /2)**2)**(S(1)/6)*cos(atan((27 + 3*sqrt(3)*31985**(S(1)/4)*cos(atan(
            S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*sin(atan(S(172)/49)/2)/2 +
            S(43)/2))/3)/6 + I*(-sqrt(3)*((3*sqrt(3)*31985**(S(1)/4)*sin(atan(
            S(172)/49)/2)/2 + S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*cos(
            atan(S(172)/49)/2)/2)**2)**(S(1)/6)*cos(atan((27 + 3*sqrt(3)*31985**(
            S(1)/4)*cos(atan(S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*sin(
            atan(S(172)/49)/2)/2 + S(43)/2))/3)/6 + ((3*sqrt(3)*31985**(S(1)/4)*
            sin(atan(S(172)/49)/2)/2 + S(43)/2)**2 + (27 + 3*sqrt(3)*31985**(S(1)/4)*
            cos(atan(S(172)/49)/2)/2)**2)**(S(1)/6)*sin(atan((27 + 3*sqrt(3)*31985**(
            S(1)/4)*cos(atan(S(172)/49)/2)/2)/(3*sqrt(3)*31985**(S(1)/4)*sin(
            atan(S(172)/49)/2)/2 + S(43)/2))/3)/6 - 4*im(1/((-S(1)/2 + sqrt(3)*I/2)*
            (S(43)/2 + 27*I + sqrt(-256 + (43 + 54*I)**2)/2)**(S(1)/3)))/3))

    assert solveset(n1, x) == res


def test_issue_13961():
    V = (ax, bx, cx, gx, jx, lx, mx, nx, q) = symbols('ax bx cx gx jx lx mx nx q')
    S = (ax*q - lx*q - mx, ax - gx*q - lx, bx*q**2 + cx*q - jx*q - nx, q*(-ax*q + lx*q + mx), q*(-ax + gx*q + lx))

    sol = FiniteSet((lx + mx/q, (-cx*q + jx*q + nx)/q**2, cx, mx/q**2, jx, lx, mx, nx, Complement({q}, {0})),
                    (lx + mx/q, (cx*q - jx*q - nx)/q**2*-1, cx, mx/q**2, jx, lx, mx, nx, Complement({q}, {0})))
    assert nonlinsolve(S, *V) == sol
    # The two solutions are in fact identical, so even better if only one is returned


def test_issue_14541():
    solutions = solveset(sqrt(-x**2 - 2.0), x)
    assert abs(solutions.args[0]+1.4142135623731*I) <= 1e-9
    assert abs(solutions.args[1]-1.4142135623731*I) <= 1e-9


def test_issue_13396():
    expr = -2*y*exp(-x**2 - y**2)*Abs(x)
    sol = FiniteSet(0)

    assert solveset(expr, y, domain=S.Reals) == sol

    # Related type of equation also solved here
    assert solveset(atan(x**2 - y**2)-pi/2, y, S.Reals) is S.EmptySet


def test_issue_12032():
    sol = FiniteSet(-sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                          2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))/2 +
                    sqrt(Abs(-2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)) +
                             2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                             2/sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                                    2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))))/2,
                    -sqrt(Abs(-2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)) +
                              2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                              2/sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                                     2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))))/2 -
                    sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                         2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))/2,
                    sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                         2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))/2 -
                    I*sqrt(Abs(-2/sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                                       2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) -
                               2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)) +
                               2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))))/2,
                    sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                         2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)))/2 +
                    I*sqrt(Abs(-2/sqrt(-2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) +
                                       2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3))) -
                               2*(Rational(1, 16) + sqrt(849)/144)**(Rational(1, 3)) +
                               2/(3*(Rational(1, 16) + sqrt(849)/144)**(Rational(1,3)))))/2)
    assert solveset(x**4 + x - 1, x) == sol


def test_issue_10876():
    assert solveset(1/sqrt(x), x) == S.EmptySet


def test_issue_19050():
    # test_issue_19050 --> TypeError removed
    assert dumeq(nonlinsolve([x + y, sin(y)], [x, y]),
        FiniteSet((ImageSet(Lambda(n, -2*n*pi), S.Integers), ImageSet(Lambda(n, 2*n*pi), S.Integers)),\
             (ImageSet(Lambda(n, -2*n*pi - pi), S.Integers), ImageSet(Lambda(n, 2*n*pi + pi), S.Integers))))
    assert dumeq(nonlinsolve([x + y, sin(y) + cos(y)], [x, y]),
        FiniteSet((ImageSet(Lambda(n, -2*n*pi - 3*pi/4), S.Integers), ImageSet(Lambda(n, 2*n*pi + 3*pi/4), S.Integers)), \
            (ImageSet(Lambda(n, -2*n*pi - 7*pi/4), S.Integers), ImageSet(Lambda(n, 2*n*pi + 7*pi/4), S.Integers))))


def test_issue_16618():
    eqn = [sin(x)*sin(y), cos(x)*cos(y) - 1]
    # nonlinsolve's answer is still suspicious since it contains only three
    # distinct Dummys instead of 4. (Both 'x' ImageSets share the same Dummy.)
    ans = FiniteSet((ImageSet(Lambda(n, 2*n*pi), S.Integers), ImageSet(Lambda(n, 2*n*pi), S.Integers)),
        (ImageSet(Lambda(n, 2*n*pi + pi), S.Integers), ImageSet(Lambda(n, 2*n*pi + pi), S.Integers)))
    sol = nonlinsolve(eqn, [x, y])

    for i0, j0 in zip(ordered(sol), ordered(ans)):
        assert len(i0) == len(j0) == 2
        assert all(a.dummy_eq(b) for a, b in zip(i0, j0))
    assert len(sol) == len(ans)


def test_issue_17566():
    assert nonlinsolve([32*(2**x)/2**(-y) - 4**y, 27*(3**x) - S(1)/3**y], x, y) ==\
        FiniteSet((-log(81)/log(3), 1))


def test_issue_16643():
    n = Dummy('n')
    assert solveset(x**2*sin(x), x).dummy_eq(Union(ImageSet(Lambda(n, 2*n*pi + pi), S.Integers),
                                                   ImageSet(Lambda(n, 2*n*pi), S.Integers)))


def test_issue_19587():
    n,m = symbols('n m')
    assert nonlinsolve([32*2**m*2**n - 4**n, 27*3**m - 3**(-n)], m, n) ==\
        FiniteSet((-log(81)/log(3), 1))


def test_issue_5132_1():
    system = [sqrt(x**2 + y**2) - sqrt(10), x + y - 4]
    assert nonlinsolve(system, [x, y]) == FiniteSet((1, 3), (3, 1))

    n = Dummy('n')
    eqs = [exp(x)**2 - sin(y) + z**2, 1/exp(y) - 3]
    s_real_y = -log(3)
    s_real_z = sqrt(-exp(2*x) - sin(log(3)))
    soln_real = FiniteSet((s_real_y, s_real_z), (s_real_y, -s_real_z))
    lam = Lambda(n, 2*n*I*pi + -log(3))
    s_complex_y = ImageSet(lam, S.Integers)
    lam = Lambda(n, sqrt(-exp(2*x) + sin(2*n*I*pi + -log(3))))
    s_complex_z_1 = ImageSet(lam, S.Integers)
    lam = Lambda(n, -sqrt(-exp(2*x) + sin(2*n*I*pi + -log(3))))
    s_complex_z_2 = ImageSet(lam, S.Integers)
    soln_complex = FiniteSet(
                                            (s_complex_y, s_complex_z_1),
                                            (s_complex_y, s_complex_z_2)
                                        )
    soln = soln_real + soln_complex
    assert dumeq(nonlinsolve(eqs, [y, z]), soln)


def test_issue_5132_2():
    x, y = symbols('x, y', real=True)
    eqs = [exp(x)**2 - sin(y) + z**2]
    n = Dummy('n')
    soln_real = (log(-z**2 + sin(y))/2, z)
    lam = Lambda( n, I*(2*n*pi + arg(-z**2 + sin(y)))/2 + log(Abs(z**2 - sin(y)))/2)
    img = ImageSet(lam, S.Integers)
    # not sure about the complex soln. But it looks correct.
    soln_complex = (img, z)
    soln = FiniteSet(soln_real, soln_complex)
    assert dumeq(nonlinsolve(eqs, [x, z]), soln)

    system = [r - x**2 - y**2, tan(t) - y/x]
    s_x = sqrt(r/(tan(t)**2 + 1))
    s_y = sqrt(r/(tan(t)**2 + 1))*tan(t)
    soln = FiniteSet((s_x, s_y), (-s_x, -s_y))
    assert nonlinsolve(system, [x, y]) == soln


def test_issue_6752():
    a, b = symbols('a, b', real=True)
    assert nonlinsolve([a**2 + a, a - b], [a, b]) == {(-1, -1), (0, 0)}


@SKIP("slow")
def test_issue_5114_solveset():
    # slow testcase
    from sympy.abc import o, p

    # there is no 'a' in the equation set but this is how the
    # problem was originally posed
    syms = [a, b, c, f, h, k, n]
    eqs = [b + r/d - c/d,
    c*(1/d + 1/e + 1/g) - f/g - r/d,
        f*(1/g + 1/i + 1/j) - c/g - h/i,
        h*(1/i + 1/l + 1/m) - f/i - k/m,
        k*(1/m + 1/o + 1/p) - h/m - n/p,
        n*(1/p + 1/q) - k/p]
    assert len(nonlinsolve(eqs, syms)) == 1


@SKIP("Hangs")
def _test_issue_5335():
    # Not able to check zero dimensional system.
    # is_zero_dimensional Hangs
    lam, a0, conc = symbols('lam a0 conc')
    eqs = [lam + 2*y - a0*(1 - x/2)*x - 0.005*x/2*x,
           a0*(1 - x/2)*x - 1*y - 0.743436700916726*y,
           x + y - conc]
    sym = [x, y, a0]
    # there are 4 solutions but only two are valid
    assert len(nonlinsolve(eqs, sym)) == 2
    # float
    eqs = [lam + 2*y - a0*(1 - x/2)*x - 0.005*x/2*x,
           a0*(1 - x/2)*x - 1*y - 0.743436700916726*y,
           x + y - conc]
    sym = [x, y, a0]
    assert len(nonlinsolve(eqs, sym)) == 2


def test_issue_2777():
    # the equations represent two circles
    x, y = symbols('x y', real=True)
    e1, e2 = sqrt(x**2 + y**2) - 10, sqrt(y**2 + (-x + 10)**2) - 3
    a, b = Rational(191, 20), 3*sqrt(391)/20
    ans = {(a, -b), (a, b)}
    assert nonlinsolve((e1, e2), (x, y)) == ans
    assert nonlinsolve((e1, e2/(x - a)), (x, y)) == S.EmptySet
    # make the 2nd circle's radius be -3
    e2 += 6
    assert nonlinsolve((e1, e2), (x, y)) == S.EmptySet


def test_issue_8828():
    x1 = 0
    y1 = -620
    r1 = 920
    x2 = 126
    y2 = 276
    x3 = 51
    y3 = 205
    r3 = 104
    v = [x, y, z]

    f1 = (x - x1)**2 + (y - y1)**2 - (r1 - z)**2
    f2 = (x2 - x)**2 + (y2 - y)**2 - z**2
    f3 = (x - x3)**2 + (y - y3)**2 - (r3 - z)**2
    F = [f1, f2, f3]

    g1 = sqrt((x - x1)**2 + (y - y1)**2) + z - r1
    g2 = f2
    g3 = sqrt((x - x3)**2 + (y - y3)**2) + z - r3
    G = [g1, g2, g3]

    # both soln same
    A = nonlinsolve(F, v)
    B = nonlinsolve(G, v)
    assert A == B


def test_nonlinsolve_conditionset():
    # when solveset failed to solve all the eq
    # return conditionset
    f = Function('f')
    f1 = f(x) - pi/2
    f2 = f(y) - pi*Rational(3, 2)
    intermediate_system = Eq(2*f(x) - pi, 0) & Eq(2*f(y) - 3*pi, 0)
    syms = Tuple(x, y)
    soln = ConditionSet(
        syms,
        intermediate_system,
        S.Complexes**2)
    assert nonlinsolve([f1, f2], [x, y]) == soln


def test_substitution_basic():
    assert substitution([], [x, y]) == S.EmptySet
    assert substitution([], []) == S.EmptySet
    system = [2*x**2 + 3*y**2 - 30, 3*x**2 - 2*y**2 - 19]
    soln = FiniteSet((-3, -2), (-3, 2), (3, -2), (3, 2))
    assert substitution(system, [x, y]) == soln

    soln = FiniteSet((-1, 1))
    assert substitution([x + y], [x], [{y: 1}], [y], set(), [x, y]) == soln
    assert substitution(
        [x + y], [x], [{y: 1}], [y],
        {x + 1}, [y, x]) == S.EmptySet


def test_substitution_incorrect():
    # the solutions in the following two tests are incorrect. The
    # correct result is EmptySet in both cases.
    assert substitution([h - 1, k - 1, f - 2, f - 4, -2 * k],
                        [h, k, f]) == {(1, 1, f)}
    assert substitution([x + y + z, S.One, S.One, S.One], [x, y, z]) == \
                        {(-y - z, y, z)}

    # the correct result in the test below is {(-I, I, I, -I),
    # (I, -I, -I, I)}
    assert substitution([a - d, b + d, c + d, d**2 + 1], [a, b, c, d]) == \
                        {(d, -d, -d, d)}

    # the result in the test below is incomplete. The complete result
    # is {(0, b), (log(2), 2)}
    assert substitution([a*(a - log(b)), a*(b - 2)], [a, b]) == \
           {(0, b)}

    # The system in the test below is zero-dimensional, so the result
    # should have no free symbols
    assert substitution([-k*y + 6*x - 4*y, -81*k + 49*y**2 - 270,
                         -3*k*z + k + z**3, k**2 - 2*k + 4],
                        [x, y, z, k]).free_symbols == {z}


def test_substitution_redundant():
    # the third and fourth solutions are redundant in the test below
    assert substitution([x**2 - y**2, z - 1], [x, z]) == \
           {(-y, 1), (y, 1), (-sqrt(y**2), 1), (sqrt(y**2), 1)}

    # the system below has three solutions. Two of the solutions
    # returned by substitution are redundant.
    res = substitution([x - y, y**3 - 3*y**2 + 1], [x, y])
    assert len(res) == 5


def test_issue_5132_substitution():
    x, y, z, r, t = symbols('x, y, z, r, t', real=True)
    system = [r - x**2 - y**2, tan(t) - y/x]
    s_x_1 = Complement(FiniteSet(-sqrt(r/(tan(t)**2 + 1))), FiniteSet(0))
    s_x_2 = Complement(FiniteSet(sqrt(r/(tan(t)**2 + 1))), FiniteSet(0))
    s_y = sqrt(r/(tan(t)**2 + 1))*tan(t)
    soln = FiniteSet((s_x_2, s_y)) + FiniteSet((s_x_1, -s_y))
    assert substitution(system, [x, y]) == soln

    n = Dummy('n')
    eqs = [exp(x)**2 - sin(y) + z**2, 1/exp(y) - 3]
    s_real_y = -log(3)
    s_real_z = sqrt(-exp(2*x) - sin(log(3)))
    soln_real = FiniteSet((s_real_y, s_real_z), (s_real_y, -s_real_z))
    lam = Lambda(n, 2*n*I*pi + -log(3))
    s_complex_y = ImageSet(lam, S.Integers)
    lam = Lambda(n, sqrt(-exp(2*x) + sin(2*n*I*pi + -log(3))))
    s_complex_z_1 = ImageSet(lam, S.Integers)
    lam = Lambda(n, -sqrt(-exp(2*x) + sin(2*n*I*pi + -log(3))))
    s_complex_z_2 = ImageSet(lam, S.Integers)
    soln_complex = FiniteSet(
        (s_complex_y, s_complex_z_1),
        (s_complex_y, s_complex_z_2))
    soln = soln_real + soln_complex
    assert dumeq(substitution(eqs, [y, z]), soln)


def test_raises_substitution():
    raises(ValueError, lambda: substitution([x**2 -1], []))
    raises(TypeError, lambda: substitution([x**2 -1]))
    raises(ValueError, lambda: substitution([x**2 -1], [sin(x)]))
    raises(TypeError, lambda: substitution([x**2 -1], x))
    raises(TypeError, lambda: substitution([x**2 -1], 1))


def test_issue_21022():
    from sympy.core.sympify import sympify

    eqs = [
    'k-16',
    'p-8',
    'y*y+z*z-x*x',
    'd - x + p',
    'd*d+k*k-y*y',
    'z*z-p*p-k*k',
    'abc-efg',
    ]
    efg = Symbol('efg')
    eqs = [sympify(x) for x in eqs]

    syb = list(ordered(set.union(*[x.free_symbols for x in eqs])))
    res = nonlinsolve(eqs, syb)

    ans = FiniteSet(
    (efg, 32, efg, 16, 8, 40, -16*sqrt(5), -8*sqrt(5)),
    (efg, 32, efg, 16, 8, 40, -16*sqrt(5), 8*sqrt(5)),
    (efg, 32, efg, 16, 8, 40, 16*sqrt(5), -8*sqrt(5)),
    (efg, 32, efg, 16, 8, 40, 16*sqrt(5), 8*sqrt(5)),
    )
    assert len(res) == len(ans) == 4
    assert res == ans
    for result in res.args:
        assert len(result) == 8


def test_issue_17940():
    n = Dummy('n')
    k1 = Dummy('k1')
    sol = ImageSet(Lambda(((k1, n),), I*(2*k1*pi + arg(2*n*I*pi + log(5)))
                          + log(Abs(2*n*I*pi + log(5)))),
                   ProductSet(S.Integers, S.Integers))
    assert solveset(exp(exp(x)) - 5, x).dummy_eq(sol)


def test_issue_17906():
    assert solveset(7**(x**2 - 80) - 49**x, x) == FiniteSet(-8, 10)


@XFAIL
def test_issue_17933():
    eq1 = x*sin(45) - y*cos(q)
    eq2 = x*cos(45) - y*sin(q)
    eq3 = 9*x*sin(45)/10 + y*cos(q)
    eq4 = 9*x*cos(45)/10 + y*sin(z) - z
    assert nonlinsolve([eq1, eq2, eq3, eq4], x, y, z, q) ==\
        FiniteSet((0, 0, 0, q))

def test_issue_17933_bis():
    # nonlinsolve's result depends on the 'default_sort_key' ordering of
    # the unknowns.
    eq1 = x*sin(45) - y*cos(q)
    eq2 = x*cos(45) - y*sin(q)
    eq3 = 9*x*sin(45)/10 + y*cos(q)
    eq4 = 9*x*cos(45)/10 + y*sin(z) - z
    zz = Symbol('zz')
    eqs = [e.subs(q, zz) for e in (eq1, eq2, eq3, eq4)]
    assert nonlinsolve(eqs, x, y, z, zz) == FiniteSet((0, 0, 0, zz))


def test_issue_14565():
    # removed redundancy
    assert dumeq(nonlinsolve([k + m, k + m*exp(-2*pi*k)], [k, m]) ,
        FiniteSet((-n*I, ImageSet(Lambda(n, n*I), S.Integers))))


# end of tests for nonlinsolve


def test_issue_9556():
    b = Symbol('b', positive=True)

    assert solveset(Abs(x) + 1, x, S.Reals) is S.EmptySet
    assert solveset(Abs(x) + b, x, S.Reals) is S.EmptySet
    assert solveset(Eq(b, -1), b, S.Reals) is S.EmptySet


def test_issue_9611():
    assert solveset(Eq(x - x + a, a), x, S.Reals) == S.Reals
    assert solveset(Eq(y - y + a, a), y) == S.Complexes


def test_issue_9557():
    assert solveset(x**2 + a, x, S.Reals) == Intersection(S.Reals,
        FiniteSet(-sqrt(-a), sqrt(-a)))


def test_issue_9778():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    assert solveset(x**3 + 1, x, S.Reals) == FiniteSet(-1)
    assert solveset(x**Rational(3, 5) + 1, x, S.Reals) == S.EmptySet
    assert solveset(x**3 + y, x, S.Reals) == \
        FiniteSet(-Abs(y)**Rational(1, 3)*sign(y))


def test_issue_10214():
    assert solveset(x**Rational(3, 2) + 4, x, S.Reals) == S.EmptySet
    assert solveset(x**(Rational(-3, 2)) + 4, x, S.Reals) == S.EmptySet

    ans = FiniteSet(-2**Rational(2, 3))
    assert solveset(x**(S(3)) + 4, x, S.Reals) == ans
    assert (x**(S(3)) + 4).subs(x,list(ans)[0]) == 0 # substituting ans and verifying the result.
    assert (x**(S(3)) + 4).subs(x,-(-2)**Rational(2, 3)) == 0


def test_issue_9849():
    assert solveset(Abs(sin(x)) + 1, x, S.Reals) == S.EmptySet


def test_issue_9953():
    assert linsolve([ ], x) == S.EmptySet


def test_issue_9913():
    assert solveset(2*x + 1/(x - 10)**2, x, S.Reals) == \
        FiniteSet(-(3*sqrt(24081)/4 + Rational(4027, 4))**Rational(1, 3)/3 - 100/
                (3*(3*sqrt(24081)/4 + Rational(4027, 4))**Rational(1, 3)) + Rational(20, 3))


def test_issue_10397():
    assert solveset(sqrt(x), x, S.Complexes) == FiniteSet(0)


def test_issue_14987():
    raises(ValueError, lambda: linear_eq_to_matrix(
        [x**2], x))
    raises(ValueError, lambda: linear_eq_to_matrix(
        [x*(-3/x + 1) + 2*y - a], [x, y]))
    raises(ValueError, lambda: linear_eq_to_matrix(
        [(x**2 - 3*x)/(x - 3) - 3], x))
    raises(ValueError, lambda: linear_eq_to_matrix(
        [(x + 1)**3 - x**3 - 3*x**2 + 7], x))
    raises(ValueError, lambda: linear_eq_to_matrix(
        [x*(1/x + 1) + y], [x, y]))
    raises(ValueError, lambda: linear_eq_to_matrix(
        [(x + 1)*y], [x, y]))
    raises(ValueError, lambda: linear_eq_to_matrix(
        [Eq(1/x, 1/x + y)], [x, y]))
    raises(ValueError, lambda: linear_eq_to_matrix(
        [Eq(y/x, y/x + y)], [x, y]))
    raises(ValueError, lambda: linear_eq_to_matrix(
        [Eq(x*(x + 1), x**2 + y)], [x, y]))


def test_simplification():
    eq = x + (a - b)/(-2*a + 2*b)
    assert solveset(eq, x) == FiniteSet(S.Half)
    assert solveset(eq, x, S.Reals) == Intersection({-((a - b)/(-2*a + 2*b))}, S.Reals)
    # So that ap - bn is not zero:
    ap = Symbol('ap', positive=True)
    bn = Symbol('bn', negative=True)
    eq = x + (ap - bn)/(-2*ap + 2*bn)
    assert solveset(eq, x) == FiniteSet(S.Half)
    assert solveset(eq, x, S.Reals) == FiniteSet(S.Half)


def test_integer_domain_relational():
    eq1 = 2*x + 3 > 0
    eq2 = x**2 + 3*x - 2 >= 0
    eq3 = x + 1/x > -2 + 1/x
    eq4 = x + sqrt(x**2 - 5) > 0
    eq = x + 1/x > -2 + 1/x
    eq5 = eq.subs(x,log(x))
    eq6 = log(x)/x <= 0
    eq7 = log(x)/x < 0
    eq8 = x/(x-3) < 3
    eq9 = x/(x**2-3) < 3

    assert solveset(eq1, x, S.Integers) == Range(-1, oo, 1)
    assert solveset(eq2, x, S.Integers) == Union(Range(-oo, -3, 1), Range(1, oo, 1))
    assert solveset(eq3, x, S.Integers) == Union(Range(-1, 0, 1), Range(1, oo, 1))
    assert solveset(eq4, x, S.Integers) == Range(3, oo, 1)
    assert solveset(eq5, x, S.Integers) == Range(2, oo, 1)
    assert solveset(eq6, x, S.Integers) == Range(1, 2, 1)
    assert solveset(eq7, x, S.Integers) == S.EmptySet
    assert solveset(eq8, x, domain=Range(0,5)) == Range(0, 3, 1)
    assert solveset(eq9, x, domain=Range(0,5)) == Union(Range(0, 2, 1), Range(2, 5, 1))

    # test_issue_19794
    assert solveset(x + 2 < 0, x, S.Integers) == Range(-oo, -2, 1)


def test_issue_10555():
    f = Function('f')
    g = Function('g')
    assert solveset(f(x) - pi/2, x, S.Reals).dummy_eq(
        ConditionSet(x, Eq(f(x) - pi/2, 0), S.Reals))
    assert solveset(f(g(x)) - pi/2, g(x), S.Reals).dummy_eq(
        ConditionSet(g(x), Eq(f(g(x)) - pi/2, 0), S.Reals))


def test_issue_8715():
    eq = x + 1/x > -2 + 1/x
    assert solveset(eq, x, S.Reals) == \
        (Interval.open(-2, oo) - FiniteSet(0))
    assert solveset(eq.subs(x,log(x)), x, S.Reals) == \
        Interval.open(exp(-2), oo) - FiniteSet(1)


def test_issue_11174():
    eq = z**2 + exp(2*x) - sin(y)
    soln = Intersection(S.Reals, FiniteSet(log(-z**2 + sin(y))/2))
    assert solveset(eq, x, S.Reals) == soln

    eq = sqrt(r)*Abs(tan(t))/sqrt(tan(t)**2 + 1) + x*tan(t)
    s = -sqrt(r)*Abs(tan(t))/(sqrt(tan(t)**2 + 1)*tan(t))
    soln = Intersection(S.Reals, FiniteSet(s))
    assert solveset(eq, x, S.Reals) == soln


def test_issue_11534():
    # eq1 and eq2 should not have the same solutions because squaring both
    # sides of the radical equation introduces a spurious solution branch.
    # The equations have a symbolic parameter y and it is easy to see that for
    # y != 0 the solution s1 will not be valid for eq1.
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    eq1 = -y + x/sqrt(-x**2 + 1)
    eq2 = -y**2 + x**2/(-x**2 + 1)

    # We get a ConditionSet here because s1 works in eq1 if y is equal to zero
    # although not for any other value of y. That case is redundant though
    # because if y=0 then s1=s2 so the solution for eq1 could just be returned
    # as s2 - {-1, 1}. In fact we have
    #   |y/sqrt(y**2 + 1)| < 1
    # So the complements are not needed either. The ideal output here would be
    #   sol1 = s2
    #   sol2 = s1 | s2.
    s1, s2 = FiniteSet(-y/sqrt(y**2 + 1)), FiniteSet(y/sqrt(y**2 + 1))
    cset = ConditionSet(x, Eq(eq1, 0), s1)
    sol1 = (s2 - {-1, 1}) | (cset - {-1, 1})
    sol2 = (s1 | s2) - {-1, 1}

    assert solveset(eq1, x, S.Reals) == sol1
    assert solveset(eq2, x, S.Reals) == sol2


def test_issue_10477():
    assert solveset((x**2 + 4*x - 3)/x < 2, x, S.Reals) == \
        Union(Interval.open(-oo, -3), Interval.open(0, 1))


def test_issue_10671():
    assert solveset(sin(y), y, Interval(0, pi)) == FiniteSet(0, pi)
    i = Interval(1, 10)
    assert solveset((1/x).diff(x) < 0, x, i) == i


def test_issue_11064():
    eq = x + sqrt(x**2 - 5)
    assert solveset(eq > 0, x, S.Reals) == \
        Interval(sqrt(5), oo)
    assert solveset(eq < 0, x, S.Reals) == \
        Interval(-oo, -sqrt(5))
    assert solveset(eq > sqrt(5), x, S.Reals) == \
        Interval.Lopen(sqrt(5), oo)


def test_issue_12478():
    eq = sqrt(x - 2) + 2
    soln = solveset_real(eq, x)
    assert soln is S.EmptySet
    assert solveset(eq < 0, x, S.Reals) is S.EmptySet
    assert solveset(eq > 0, x, S.Reals) == Interval(2, oo)


def test_issue_12429():
    eq = solveset(log(x)/x <= 0, x, S.Reals)
    sol = Interval.Lopen(0, 1)
    assert eq == sol


def test_issue_19506():
    eq = arg(x + I)
    C = Dummy('C')
    assert solveset(eq).dummy_eq(Intersection(ConditionSet(C, Eq(im(C) + 1, 0), S.Complexes),
                                                 ConditionSet(C, re(C) > 0, S.Complexes)))


def test_solveset_arg():
    assert solveset(arg(x), x, S.Reals)  == Interval.open(0, oo)
    assert solveset(arg(4*x -3), x, S.Reals) == Interval.open(Rational(3, 4), oo)


def test__is_finite_with_finite_vars():
    f = _is_finite_with_finite_vars
    # issue 12482
    assert all(f(1/x) is None for x in (
        Dummy(), Dummy(real=True), Dummy(complex=True)))
    assert f(1/Dummy(real=False)) is True  # b/c it's finite but not 0


def test_issue_13550():
    assert solveset(x**2 - 2*x - 15, symbol = x, domain = Interval(-oo, 0)) == FiniteSet(-3)


def test_issue_13849():
    assert nonlinsolve((t*(sqrt(5) + sqrt(2)) - sqrt(2), t), t) is S.EmptySet


def test_issue_14223():
    assert solveset((Abs(x + Min(x, 2)) - 2).rewrite(Piecewise), x,
        S.Reals) == FiniteSet(-1, 1)
    assert solveset((Abs(x + Min(x, 2)) - 2).rewrite(Piecewise), x,
        Interval(0, 2)) == FiniteSet(1)
    assert solveset(x, x, FiniteSet(1, 2)) is S.EmptySet


def test_issue_10158():
    dom = S.Reals
    assert solveset(x*Max(x, 15) - 10, x, dom) == FiniteSet(Rational(2, 3))
    assert solveset(x*Min(x, 15) - 10, x, dom) == FiniteSet(-sqrt(10), sqrt(10))
    assert solveset(Max(Abs(x - 3) - 1, x + 2) - 3, x, dom) == FiniteSet(-1, 1)
    assert solveset(Abs(x - 1) - Abs(y), x, dom) == FiniteSet(-Abs(y) + 1, Abs(y) + 1)
    assert solveset(Abs(x + 4*Abs(x + 1)), x, dom) == FiniteSet(Rational(-4, 3), Rational(-4, 5))
    assert solveset(2*Abs(x + Abs(x + Max(3, x))) - 2, x, S.Reals) == FiniteSet(-1, -2)
    dom = S.Complexes
    raises(ValueError, lambda: solveset(x*Max(x, 15) - 10, x, dom))
    raises(ValueError, lambda: solveset(x*Min(x, 15) - 10, x, dom))
    raises(ValueError, lambda: solveset(Max(Abs(x - 3) - 1, x + 2) - 3, x, dom))
    raises(ValueError, lambda: solveset(Abs(x - 1) - Abs(y), x, dom))
    raises(ValueError, lambda: solveset(Abs(x + 4*Abs(x + 1)), x, dom))


def test_issue_14300():
    f = 1 - exp(-18000000*x) - y
    a1 = FiniteSet(-log(-y + 1)/18000000)

    assert solveset(f, x, S.Reals) == \
        Intersection(S.Reals, a1)
    assert dumeq(solveset(f, x),
        ImageSet(Lambda(n, -I*(2*n*pi + arg(-y + 1))/18000000 -
            log(Abs(y - 1))/18000000), S.Integers))


def test_issue_14454():
    number = CRootOf(x**4 + x - 1, 2)
    raises(ValueError, lambda: invert_real(number, 0, x))
    assert invert_real(x**2, number, x)  # no error


def test_issue_17882():
    assert solveset(-8*x**2/(9*(x**2 - 1)**(S(4)/3)) + 4/(3*(x**2 - 1)**(S(1)/3)), x, S.Complexes) == \
        FiniteSet(sqrt(3), -sqrt(3))


def test_term_factors():
    assert list(_term_factors(3**x - 2)) == [-2, 3**x]
    expr = 4**(x + 1) + 4**(x + 2) + 4**(x - 1) - 3**(x + 2) - 3**(x + 3)
    assert set(_term_factors(expr)) == {
        3**(x + 2), 4**(x + 2), 3**(x + 3), 4**(x - 1), -1, 4**(x + 1)}


#################### tests for transolve and its helpers ###############

def test_transolve():

    assert _transolve(3**x, x, S.Reals) == S.EmptySet
    assert _transolve(3**x - 9**(x + 5), x, S.Reals) == FiniteSet(-10)


def test_issue_21276():
    eq = (2*x*(y - z) - y*erf(y - z) - y + z*erf(y - z) + z)**2
    assert solveset(eq.expand(), y) == FiniteSet(z, z + erfinv(2*x - 1))


# exponential tests
def test_exponential_real():
    from sympy.abc import y

    e1 = 3**(2*x) - 2**(x + 3)
    e2 = 4**(5 - 9*x) - 8**(2 - x)
    e3 = 2**x + 4**x
    e4 = exp(log(5)*x) - 2**x
    e5 = exp(x/y)*exp(-z/y) - 2
    e6 = 5**(x/2) - 2**(x/3)
    e7 = 4**(x + 1) + 4**(x + 2) + 4**(x - 1) - 3**(x + 2) - 3**(x + 3)
    e8 = -9*exp(-2*x + 5) + 4*exp(3*x + 1)
    e9 = 2**x + 4**x + 8**x - 84
    e10 = 29*2**(x + 1)*615**(x) - 123*2726**(x)

    assert solveset(e1, x, S.Reals) == FiniteSet(
        -3*log(2)/(-2*log(3) + log(2)))
    assert solveset(e2, x, S.Reals) == FiniteSet(Rational(4, 15))
    assert solveset(e3, x, S.Reals) == S.EmptySet
    assert solveset(e4, x, S.Reals) == FiniteSet(0)
    assert solveset(e5, x, S.Reals) == Intersection(
        S.Reals, FiniteSet(y*log(2*exp(z/y))))
    assert solveset(e6, x, S.Reals) == FiniteSet(0)
    assert solveset(e7, x, S.Reals) == FiniteSet(2)
    assert solveset(e8, x, S.Reals) == FiniteSet(-2*log(2)/5 + 2*log(3)/5 + Rational(4, 5))
    assert solveset(e9, x, S.Reals) == FiniteSet(2)
    assert solveset(e10,x, S.Reals) == FiniteSet((-log(29) - log(2) + log(123))/(-log(2726) + log(2) + log(615)))

    assert solveset_real(-9*exp(-2*x + 5) + 2**(x + 1), x) == FiniteSet(
        -((-5 - 2*log(3) + log(2))/(log(2) + 2)))
    assert solveset_real(4**(x/2) - 2**(x/3), x) == FiniteSet(0)
    b = sqrt(6)*sqrt(log(2))/sqrt(log(5))
    assert solveset_real(5**(x/2) - 2**(3/x), x) == FiniteSet(-b, b)

    # coverage test
    C1, C2 = symbols('C1 C2')
    f = Function('f')
    assert solveset_real(C1 + C2/x**2 - exp(-f(x)), f(x)) == Intersection(
        S.Reals, FiniteSet(-log(C1 + C2/x**2)))
    y = symbols('y', positive=True)
    assert solveset_real(x**2 - y**2/exp(x), y) == Intersection(
        S.Reals, FiniteSet(-sqrt(x**2*exp(x)), sqrt(x**2*exp(x))))
    p = Symbol('p', positive=True)
    assert solveset_real((1/p + 1)**(p + 1), p).dummy_eq(
        ConditionSet(x, Eq((1 + 1/x)**(x + 1), 0), S.Reals))
    assert solveset(2**x - 4**x + 12, x, S.Reals) == {2}
    assert solveset(2**x - 2**(2*x) + 12, x, S.Reals) == {2}


@XFAIL
def test_exponential_complex():
    n = Dummy('n')

    assert dumeq(solveset_complex(2**x + 4**x, x),imageset(
        Lambda(n, I*(2*n*pi + pi)/log(2)), S.Integers))
    assert solveset_complex(x**z*y**z - 2, z) == FiniteSet(
        log(2)/(log(x) + log(y)))
    assert dumeq(solveset_complex(4**(x/2) - 2**(x/3), x), imageset(
        Lambda(n, 3*n*I*pi/log(2)), S.Integers))
    assert dumeq(solveset(2**x + 32, x), imageset(
        Lambda(n, (I*(2*n*pi + pi) + 5*log(2))/log(2)), S.Integers))

    eq = (2**exp(y**2/x) + 2)/(x**2 + 15)
    a = sqrt(x)*sqrt(-log(log(2)) + log(log(2) + 2*n*I*pi))
    assert solveset_complex(eq, y) == FiniteSet(-a, a)

    union1 = imageset(Lambda(n, I*(2*n*pi - pi*Rational(2, 3))/log(2)), S.Integers)
    union2 = imageset(Lambda(n, I*(2*n*pi + pi*Rational(2, 3))/log(2)), S.Integers)
    assert dumeq(solveset(2**x + 4**x + 8**x, x), Union(union1, union2))

    eq = 4**(x + 1) + 4**(x + 2) + 4**(x - 1) - 3**(x + 2) - 3**(x + 3)
    res = solveset(eq, x)
    num = 2*n*I*pi - 4*log(2) + 2*log(3)
    den = -2*log(2) + log(3)
    ans = imageset(Lambda(n, num/den), S.Integers)
    assert dumeq(res, ans)


def test_expo_conditionset():

    f1 = (exp(x) + 1)**x - 2
    f2 = (x + 2)**y*x - 3
    f3 = 2**x - exp(x) - 3
    f4 = log(x) - exp(x)
    f5 = 2**x + 3**x - 5**x

    assert solveset(f1, x, S.Reals).dummy_eq(ConditionSet(
        x, Eq((exp(x) + 1)**x - 2, 0), S.Reals))
    assert solveset(f2, x, S.Reals).dummy_eq(ConditionSet(
        x, Eq(x*(x + 2)**y - 3, 0), S.Reals))
    assert solveset(f3, x, S.Reals).dummy_eq(ConditionSet(
        x, Eq(2**x - exp(x) - 3, 0), S.Reals))
    assert solveset(f4, x, S.Reals).dummy_eq(ConditionSet(
        x, Eq(-exp(x) + log(x), 0), S.Reals))
    assert solveset(f5, x, S.Reals).dummy_eq(ConditionSet(
        x, Eq(2**x + 3**x - 5**x, 0), S.Reals))


def test_exponential_symbols():
    x, y, z = symbols('x y z', positive=True)
    xr, zr = symbols('xr, zr', real=True)

    assert solveset(z**x - y, x, S.Reals) == Intersection(
        S.Reals, FiniteSet(log(y)/log(z)))

    f1 = 2*x**w - 4*y**w
    f2 = (x/y)**w - 2
    sol1 = Intersection({log(2)/(log(x) - log(y))}, S.Reals)
    sol2 = Intersection({log(2)/log(x/y)}, S.Reals)
    assert solveset(f1, w, S.Reals) == sol1, solveset(f1, w, S.Reals)
    assert solveset(f2, w, S.Reals) == sol2, solveset(f2, w, S.Reals)

    assert solveset(x**x, x, Interval.Lopen(0,oo)).dummy_eq(
        ConditionSet(w, Eq(w**w, 0), Interval.open(0, oo)))
    assert solveset(x**y - 1, y, S.Reals) == FiniteSet(0)
    assert solveset(exp(x/y)*exp(-z/y) - 2, y, S.Reals) == \
    Complement(ConditionSet(y, Eq(im(x)/y, 0) & Eq(im(z)/y, 0), \
    Complement(Intersection(FiniteSet((x - z)/log(2)), S.Reals), FiniteSet(0))), FiniteSet(0))
    assert solveset(exp(xr/y)*exp(-zr/y) - 2, y, S.Reals) == \
        Complement(FiniteSet((xr - zr)/log(2)), FiniteSet(0))

    assert solveset(a**x - b**x, x).dummy_eq(ConditionSet(
        w, Ne(a, 0) & Ne(b, 0), FiniteSet(0)))


def test_ignore_assumptions():
    # make sure assumptions are ignored
    xpos = symbols('x', positive=True)
    x = symbols('x')
    assert solveset_complex(xpos**2 - 4, xpos
        ) == solveset_complex(x**2 - 4, x)


@XFAIL
def test_issue_10864():
    assert solveset(x**(y*z) - x, x, S.Reals) == FiniteSet(1)


@XFAIL
def test_solve_only_exp_2():
    assert solveset_real(sqrt(exp(x)) + sqrt(exp(-x)) - 4, x) == \
        FiniteSet(2*log(-sqrt(3) + 2), 2*log(sqrt(3) + 2))


def test_is_exponential():
    assert _is_exponential(y, x) is False
    assert _is_exponential(3**x - 2, x) is True
    assert _is_exponential(5**x - 7**(2 - x), x) is True
    assert _is_exponential(sin(2**x) - 4*x, x) is False
    assert _is_exponential(x**y - z, y) is True
    assert _is_exponential(x**y - z, x) is False
    assert _is_exponential(2**x + 4**x - 1, x) is True
    assert _is_exponential(x**(y*z) - x, x) is False
    assert _is_exponential(x**(2*x) - 3**x, x) is False
    assert _is_exponential(x**y - y*z, y) is False
    assert _is_exponential(x**y - x*z, y) is True


def test_solve_exponential():
    assert _solve_exponential(3**(2*x) - 2**(x + 3), 0, x, S.Reals) == \
        FiniteSet(-3*log(2)/(-2*log(3) + log(2)))
    assert _solve_exponential(2**y + 4**y, 1, y, S.Reals) == \
        FiniteSet(log(Rational(-1, 2) + sqrt(5)/2)/log(2))
    assert _solve_exponential(2**y + 4**y, 0, y, S.Reals) == \
        S.EmptySet
    assert _solve_exponential(2**x + 3**x - 5**x, 0, x, S.Reals) == \
        ConditionSet(x, Eq(2**x + 3**x - 5**x, 0), S.Reals)

# end of exponential tests


# logarithmic tests
def test_logarithmic():
    assert solveset_real(log(x - 3) + log(x + 3), x) == FiniteSet(
        -sqrt(10), sqrt(10))
    assert solveset_real(log(x + 1) - log(2*x - 1), x) == FiniteSet(2)
    assert solveset_real(log(x + 3) + log(1 + 3/x) - 3, x) == FiniteSet(
        -3 + sqrt(-12 + exp(3))*exp(Rational(3, 2))/2 + exp(3)/2,
        -sqrt(-12 + exp(3))*exp(Rational(3, 2))/2 - 3 + exp(3)/2)

    eq = z - log(x) + log(y/(x*(-1 + y**2/x**2)))
    assert solveset_real(eq, x) == \
        Intersection(S.Reals, FiniteSet(-sqrt(y**2 - y*exp(z)),
            sqrt(y**2 - y*exp(z)))) - \
        Intersection(S.Reals, FiniteSet(-sqrt(y**2), sqrt(y**2)))
    assert solveset_real(
        log(3*x) - log(-x + 1) - log(4*x + 1), x) == FiniteSet(Rational(-1, 2), S.Half)
    assert solveset(log(x**y) - y*log(x), x, S.Reals) == S.Reals

@XFAIL
def test_uselogcombine_2():
    eq = log(exp(2*x) + 1) + log(-tanh(x) + 1) - log(2)
    assert solveset_real(eq, x) is S.EmptySet
    eq = log(8*x) - log(sqrt(x) + 1) - 2
    assert solveset_real(eq, x) is S.EmptySet


def test_is_logarithmic():
    assert _is_logarithmic(y, x) is False
    assert _is_logarithmic(log(x), x) is True
    assert _is_logarithmic(log(x) - 3, x) is True
    assert _is_logarithmic(log(x)*log(y), x) is True
    assert _is_logarithmic(log(x)**2, x) is False
    assert _is_logarithmic(log(x - 3) + log(x + 3), x) is True
    assert _is_logarithmic(log(x**y) - y*log(x), x) is True
    assert _is_logarithmic(sin(log(x)), x) is False
    assert _is_logarithmic(x + y, x) is False
    assert _is_logarithmic(log(3*x) - log(1 - x) + 4, x) is True
    assert _is_logarithmic(log(x) + log(y) + x, x) is False
    assert _is_logarithmic(log(log(x - 3)) + log(x - 3), x) is True
    assert _is_logarithmic(log(log(3) + x) + log(x), x) is True
    assert _is_logarithmic(log(x)*(y + 3) + log(x), y) is False


def test_solve_logarithm():
    y = Symbol('y')
    assert _solve_logarithm(log(x**y) - y*log(x), 0, x, S.Reals) == S.Reals
    y = Symbol('y', positive=True)
    assert _solve_logarithm(log(x)*log(y), 0, x, S.Reals) == FiniteSet(1)

# end of logarithmic tests


# lambert tests
def test_is_lambert():
    a, b, c = symbols('a,b,c')
    assert _is_lambert(x**2, x) is False
    assert _is_lambert(a**x**2+b*x+c, x) is True
    assert _is_lambert(E**2, x) is False
    assert _is_lambert(x*E**2, x) is False
    assert _is_lambert(3*log(x) - x*log(3), x) is True
    assert _is_lambert(log(log(x - 3)) + log(x-3), x) is True
    assert _is_lambert(5*x - 1 + 3*exp(2 - 7*x), x) is True
    assert _is_lambert((a/x + exp(x/2)).diff(x, 2), x) is True
    assert _is_lambert((x**2 - 2*x + 1).subs(x, (log(x) + 3*x)**2 - 1), x) is True
    assert _is_lambert(x*sinh(x) - 1, x) is True
    assert _is_lambert(x*cos(x) - 5, x) is True
    assert _is_lambert(tanh(x) - 5*x, x) is True
    assert _is_lambert(cosh(x) - sinh(x), x) is False

# end of lambert tests


def test_linear_coeffs():
    from sympy.solvers.solveset import linear_coeffs
    assert linear_coeffs(0, x) == [0, 0]
    assert all(i is S.Zero for i in linear_coeffs(0, x))
    assert linear_coeffs(x + 2*y + 3, x, y) == [1, 2, 3]
    assert linear_coeffs(x + 2*y + 3, y, x) == [2, 1, 3]
    assert linear_coeffs(x + 2*x**2 + 3, x, x**2) == [1, 2, 3]
    raises(ValueError, lambda:
        linear_coeffs(x + 2*x**2 + x**3, x, x**2))
    raises(ValueError, lambda:
        linear_coeffs(1/x*(x - 1) + 1/x, x))
    raises(ValueError, lambda:
        linear_coeffs(x, x, x))
    assert linear_coeffs(a*(x + y), x, y) == [a, a, 0]
    assert linear_coeffs(1.0, x, y) == [0, 0, 1.0]
    # don't include coefficients of 0
    assert linear_coeffs(Eq(x, x + y), x, y, dict=True) == {y: -1}
    assert linear_coeffs(0, x, y, dict=True) == {}


def test_is_modular():
    assert _is_modular(y, x) is False
    assert _is_modular(Mod(x, 3) - 1, x) is True
    assert _is_modular(Mod(x**3 - 3*x**2 - x + 1, 3) - 1, x) is True
    assert _is_modular(Mod(exp(x + y), 3) - 2, x) is True
    assert _is_modular(Mod(exp(x + y), 3) - log(x), x) is True
    assert _is_modular(Mod(x, 3) - 1, y) is False
    assert _is_modular(Mod(x, 3)**2 - 5, x) is False
    assert _is_modular(Mod(x, 3)**2 - y, x) is False
    assert _is_modular(exp(Mod(x, 3)) - 1, x) is False
    assert _is_modular(Mod(3, y) - 1, y) is False


def test_invert_modular():
    n = Dummy('n', integer=True)
    from sympy.solvers.solveset import _invert_modular as invert_modular

    # no solutions
    assert invert_modular(Mod(x, 12), S(1)/2, n, x) == (x, S.EmptySet)
    # non invertible cases
    assert invert_modular(Mod(sin(x), 7), S(5), n, x) == (Mod(sin(x), 7), 5)
    assert invert_modular(Mod(exp(x), 7), S(5), n, x) == (Mod(exp(x), 7), 5)
    assert invert_modular(Mod(log(x), 7), S(5), n, x) == (Mod(log(x), 7), 5)
    # a is symbol
    assert dumeq(invert_modular(Mod(x, 7), S(5), n, x),
            (x, ImageSet(Lambda(n, 7*n + 5), S.Integers)))
    # a.is_Add
    assert dumeq(invert_modular(Mod(x + 8, 7), S(5), n, x),
            (x, ImageSet(Lambda(n, 7*n + 4), S.Integers)))
    assert invert_modular(Mod(x**2 + x, 7), S(5), n, x) == \
            (Mod(x**2 + x, 7), 5)
    # a.is_Mul
    assert dumeq(invert_modular(Mod(3*x, 7), S(5), n, x),
            (x, ImageSet(Lambda(n, 7*n + 4), S.Integers)))
    assert invert_modular(Mod((x + 1)*(x + 2), 7), S(5), n, x) == \
            (Mod((x + 1)*(x + 2), 7), 5)
    # a.is_Pow
    assert invert_modular(Mod(x**4, 7), S(5), n, x) == \
            (x, S.EmptySet)
    assert dumeq(invert_modular(Mod(3**x, 4), S(3), n, x),
            (x, ImageSet(Lambda(n, 2*n + 1), S.Naturals0)))
    assert dumeq(invert_modular(Mod(2**(x**2 + x + 1), 7), S(2), n, x),
            (x**2 + x + 1, ImageSet(Lambda(n, 3*n + 1), S.Naturals0)))
    assert invert_modular(Mod(sin(x)**4, 7), S(5), n, x) == (x, S.EmptySet)


def test_solve_modular():
    n = Dummy('n', integer=True)
    # if rhs has symbol (need to be implemented in future).
    assert solveset(Mod(x, 4) - x, x, S.Integers
        ).dummy_eq(
            ConditionSet(x, Eq(-x + Mod(x, 4), 0),
            S.Integers))
    # when _invert_modular fails to invert
    assert solveset(3 - Mod(sin(x), 7), x, S.Integers
        ).dummy_eq(
            ConditionSet(x, Eq(Mod(sin(x), 7) - 3, 0), S.Integers))
    assert solveset(3 - Mod(log(x), 7), x, S.Integers
        ).dummy_eq(
            ConditionSet(x, Eq(Mod(log(x), 7) - 3, 0), S.Integers))
    assert solveset(3 - Mod(exp(x), 7), x, S.Integers
        ).dummy_eq(ConditionSet(x, Eq(Mod(exp(x), 7) - 3, 0),
        S.Integers))
    # EmptySet solution definitely
    assert solveset(7 - Mod(x, 5), x, S.Integers) is S.EmptySet
    assert solveset(5 - Mod(x, 5), x, S.Integers) is S.EmptySet
    # Negative m
    assert dumeq(solveset(2 + Mod(x, -3), x, S.Integers),
            ImageSet(Lambda(n, -3*n - 2), S.Integers))
    assert solveset(4 + Mod(x, -3), x, S.Integers) is S.EmptySet
    # linear expression in Mod
    assert dumeq(solveset(3 - Mod(x, 5), x, S.Integers),
        ImageSet(Lambda(n, 5*n + 3), S.Integers))
    assert dumeq(solveset(3 - Mod(5*x - 8, 7), x, S.Integers),
                ImageSet(Lambda(n, 7*n + 5), S.Integers))
    assert dumeq(solveset(3 - Mod(5*x, 7), x, S.Integers),
                ImageSet(Lambda(n, 7*n + 2), S.Integers))
    # higher degree expression in Mod
    assert dumeq(solveset(Mod(x**2, 160) - 9, x, S.Integers),
            Union(ImageSet(Lambda(n, 160*n + 3), S.Integers),
            ImageSet(Lambda(n, 160*n + 13), S.Integers),
            ImageSet(Lambda(n, 160*n + 67), S.Integers),
            ImageSet(Lambda(n, 160*n + 77), S.Integers),
            ImageSet(Lambda(n, 160*n + 83), S.Integers),
            ImageSet(Lambda(n, 160*n + 93), S.Integers),
            ImageSet(Lambda(n, 160*n + 147), S.Integers),
            ImageSet(Lambda(n, 160*n + 157), S.Integers)))
    assert solveset(3 - Mod(x**4, 7), x, S.Integers) is S.EmptySet
    assert dumeq(solveset(Mod(x**4, 17) - 13, x, S.Integers),
            Union(ImageSet(Lambda(n, 17*n + 3), S.Integers),
            ImageSet(Lambda(n, 17*n + 5), S.Integers),
            ImageSet(Lambda(n, 17*n + 12), S.Integers),
            ImageSet(Lambda(n, 17*n + 14), S.Integers)))
    # a.is_Pow tests
    assert dumeq(solveset(Mod(7**x, 41) - 15, x, S.Integers),
            ImageSet(Lambda(n, 40*n + 3), S.Naturals0))
    assert dumeq(solveset(Mod(12**x, 21) - 18, x, S.Integers),
            ImageSet(Lambda(n, 6*n + 2), S.Naturals0))
    assert dumeq(solveset(Mod(3**x, 4) - 3, x, S.Integers),
            ImageSet(Lambda(n, 2*n + 1), S.Naturals0))
    assert dumeq(solveset(Mod(2**x, 7) - 2 , x, S.Integers),
            ImageSet(Lambda(n, 3*n + 1), S.Naturals0))
    assert dumeq(solveset(Mod(3**(3**x), 4) - 3, x, S.Integers),
            Intersection(ImageSet(Lambda(n, Intersection({log(2*n + 1)/log(3)},
            S.Integers)), S.Naturals0), S.Integers))
    # Implemented for m without primitive root
    assert solveset(Mod(x**3, 7) - 2, x, S.Integers) is S.EmptySet
    assert dumeq(solveset(Mod(x**3, 8) - 1, x, S.Integers),
            ImageSet(Lambda(n, 8*n + 1), S.Integers))
    assert dumeq(solveset(Mod(x**4, 9) - 4, x, S.Integers),
            Union(ImageSet(Lambda(n, 9*n + 4), S.Integers),
            ImageSet(Lambda(n, 9*n + 5), S.Integers)))
    # domain intersection
    assert dumeq(solveset(3 - Mod(5*x - 8, 7), x, S.Naturals0),
            Intersection(ImageSet(Lambda(n, 7*n + 5), S.Integers), S.Naturals0))
    # Complex args
    assert solveset(Mod(x, 3) - I, x, S.Integers) == \
            S.EmptySet
    assert solveset(Mod(I*x, 3) - 2, x, S.Integers
        ).dummy_eq(
            ConditionSet(x, Eq(Mod(I*x, 3) - 2, 0), S.Integers))
    assert solveset(Mod(I + x, 3) - 2, x, S.Integers
        ).dummy_eq(
            ConditionSet(x, Eq(Mod(x + I, 3) - 2, 0), S.Integers))

    # issue 17373 (https://github.com/sympy/sympy/issues/17373)
    assert dumeq(solveset(Mod(x**4, 14) - 11, x, S.Integers),
            Union(ImageSet(Lambda(n, 14*n + 3), S.Integers),
            ImageSet(Lambda(n, 14*n + 11), S.Integers)))
    assert dumeq(solveset(Mod(x**31, 74) - 43, x, S.Integers),
            ImageSet(Lambda(n, 74*n + 31), S.Integers))

    # issue 13178
    n = symbols('n', integer=True)
    a = 742938285
    b = 1898888478
    m = 2**31 - 1
    c = 20170816
    assert dumeq(solveset(c - Mod(a**n*b, m), n, S.Integers),
            ImageSet(Lambda(n, 2147483646*n + 100), S.Naturals0))
    assert dumeq(solveset(c - Mod(a**n*b, m), n, S.Naturals0),
            Intersection(ImageSet(Lambda(n, 2147483646*n + 100), S.Naturals0),
            S.Naturals0))
    assert dumeq(solveset(c - Mod(a**(2*n)*b, m), n, S.Integers),
            Intersection(ImageSet(Lambda(n, 1073741823*n + 50), S.Naturals0),
            S.Integers))
    assert solveset(c - Mod(a**(2*n + 7)*b, m), n, S.Integers) is S.EmptySet
    assert dumeq(solveset(c - Mod(a**(n - 4)*b, m), n, S.Integers),
            Intersection(ImageSet(Lambda(n, 2147483646*n + 104), S.Naturals0),
            S.Integers))

# end of modular tests

def test_issue_17276():
    assert nonlinsolve([Eq(x, 5**(S(1)/5)), Eq(x*y, 25*sqrt(5))], x, y) == \
     FiniteSet((5**(S(1)/5), 25*5**(S(3)/10)))


def test_issue_10426():
    x = Dummy('x')
    a = Symbol('a')
    n = Dummy('n')
    assert (solveset(sin(x + a) - sin(x), a)).dummy_eq(Dummy('x')) == (Union(
        ImageSet(Lambda(n, 2*n*pi), S.Integers),
        Intersection(S.Complexes, ImageSet(Lambda(n, -I*(I*(2*n*pi + arg(-exp(-2*I*x))) + 2*im(x))),
        S.Integers)))).dummy_eq(Dummy('x,n'))


def test_solveset_conjugate():
    """Test solveset for simple conjugate functions"""
    assert solveset(conjugate(x) -3 + I) == FiniteSet(3 + I)


def test_issue_18208():
    variables = symbols('x0:16') + symbols('y0:12')
    x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15,\
    y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11 = variables

    eqs = [x0 + x1 + x2 + x3 - 51,
           x0 + x1 + x4 + x5 - 46,
           x2 + x3 + x6 + x7 - 39,
           x0 + x3 + x4 + x7 - 50,
           x1 + x2 + x5 + x6 - 35,
           x4 + x5 + x6 + x7 - 34,
           x4 + x5 + x8 + x9 - 46,
           x10 + x11 + x6 + x7 - 23,
           x11 + x4 + x7 + x8 - 25,
           x10 + x5 + x6 + x9 - 44,
           x10 + x11 + x8 + x9 - 35,
           x12 + x13 + x8 + x9 - 35,
           x10 + x11 + x14 + x15 - 29,
           x11 + x12 + x15 + x8 - 35,
           x10 + x13 + x14 + x9 - 29,
           x12 + x13 + x14 + x15 - 29,
           y0 + y1 + y2 + y3 - 55,
           y0 + y1 + y4 + y5 - 53,
           y2 + y3 + y6 + y7 - 56,
           y0 + y3 + y4 + y7 - 57,
           y1 + y2 + y5 + y6 - 52,
           y4 + y5 + y6 + y7 - 54,
           y4 + y5 + y8 + y9 - 48,
           y10 + y11 + y6 + y7 - 60,
           y11 + y4 + y7 + y8 - 51,
           y10 + y5 + y6 + y9 - 57,
           y10 + y11 + y8 + y9 - 54,
           x10 - 2,
           x11 - 5,
           x12 - 1,
           x13 - 6,
           x14 - 1,
           x15 - 21,
           y0 - 12,
           y1 - 20]

    expected = [38 - x3, x3 - 10, 23 - x3, x3, 12 - x7, x7 + 6, 16 - x7, x7,
                8, 20, 2, 5, 1, 6, 1, 21, 12, 20, -y11 + y9 + 2, y11 - y9 + 21,
                -y11 - y7 + y9 + 24, y11 + y7 - y9 - 3, 33 - y7, y7, 27 - y9, y9,
                27 - y11, y11]

    A, b = linear_eq_to_matrix(eqs, variables)

    # solve
    solve_expected = {v:eq for v, eq in zip(variables, expected) if v != eq}

    assert solve(eqs, variables) == solve_expected

    # linsolve
    linsolve_expected = FiniteSet(Tuple(*expected))

    assert linsolve(eqs, variables) == linsolve_expected
    assert linsolve((A, b), variables) == linsolve_expected

    # gauss_jordan_solve
    gj_solve, new_vars = A.gauss_jordan_solve(b)
    gj_solve = list(gj_solve)

    gj_expected = linsolve_expected.subs(zip([x3, x7, y7, y9, y11], new_vars))

    assert FiniteSet(Tuple(*gj_solve)) == gj_expected

    # nonlinsolve
    # The solution set of nonlinsolve is currently equivalent to linsolve and is
    # also correct. However, we would prefer to use the same symbols as parameters
    # for the solution to the underdetermined system in all cases if possible.
    # We want a solution that is not just equivalent but also given in the same form.
    # This test may be changed should nonlinsolve be modified in this way.

    nonlinsolve_expected = FiniteSet((38 - x3, x3 - 10, 23 - x3, x3, 12 - x7, x7 + 6,
                                      16 - x7, x7, 8, 20, 2, 5, 1, 6, 1, 21, 12, 20,
                                      -y5 + y7 - 1, y5 - y7 + 24, 21 - y5, y5, 33 - y7,
                                      y7, 27 - y9, y9, -y5 + y7 - y9 + 24, y5 - y7 + y9 + 3))

    assert nonlinsolve(eqs, variables) == nonlinsolve_expected


def test_substitution_with_infeasible_solution():
    a00, a01, a10, a11, l0, l1, l2, l3, m0, m1, m2, m3, m4, m5, m6, m7, c00, c01, c10, c11, p00, p01, p10, p11 = symbols(
        'a00, a01, a10, a11, l0, l1, l2, l3, m0, m1, m2, m3, m4, m5, m6, m7, c00, c01, c10, c11, p00, p01, p10, p11'
    )
    solvefor = [p00, p01, p10, p11, c00, c01, c10, c11, m0, m1, m3, l0, l1, l2, l3]
    system = [
        -l0 * c00 - l1 * c01 + m0 + c00 + c01,
        -l0 * c10 - l1 * c11 + m1,
        -l2 * c00 - l3 * c01 + c00 + c01,
        -l2 * c10 - l3 * c11 + m3,
        -l0 * p00 - l2 * p10 + p00 + p10,
        -l1 * p00 - l3 * p10 + p00 + p10,
        -l0 * p01 - l2 * p11,
        -l1 * p01 - l3 * p11,
        -a00 + c00 * p00 + c10 * p01,
        -a01 + c01 * p00 + c11 * p01,
        -a10 + c00 * p10 + c10 * p11,
        -a11 + c01 * p10 + c11 * p11,
        -m0 * p00,
        -m1 * p01,
        -m2 * p10,
        -m3 * p11,
        -m4 * c00,
        -m5 * c01,
        -m6 * c10,
        -m7 * c11,
        m2,
        m4,
        m5,
        m6,
        m7
    ]
    sol = FiniteSet(
        (0, Complement(FiniteSet(p01), FiniteSet(0)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, l2, l3),
        (p00, Complement(FiniteSet(p01), FiniteSet(0)), 0, p11, 0, 0, 0, 0, 0, 0, 0, 1, 1, -p01/p11, -p01/p11),
        (0, Complement(FiniteSet(p01), FiniteSet(0)), 0, p11, 0, 0, 0, 0, 0, 0, 0, 1, -l3*p11/p01, -p01/p11, l3),
        (0, Complement(FiniteSet(p01), FiniteSet(0)), 0, p11, 0, 0, 0, 0, 0, 0, 0, -l2*p11/p01, -l3*p11/p01, l2, l3),
    )
    assert sol != nonlinsolve(system, solvefor)


def test_issue_20097():
    assert solveset(1/sqrt(x)) is S.EmptySet


def test_issue_15350():
    assert solveset(diff(sqrt(1/x+x))) == FiniteSet(-1, 1)


def test_issue_18359():
    c1 = Piecewise((0, x < 0), (Min(1, x)/2 - Min(2, x)/2 + Min(3, x)/2, True))
    c2 = Piecewise((Piecewise((0, x < 0), (Min(1, x)/2 - Min(2, x)/2 + Min(3, x)/2, True)), x >= 0), (0, True))
    correct_result = Interval(1, 2)
    result1 = solveset(c1 - Rational(1, 2), x, Interval(0, 3))
    result2 = solveset(c2 - Rational(1, 2), x, Interval(0, 3))
    assert result1 == correct_result
    assert result2 == correct_result


def test_issue_17604():
    lhs = -2**(3*x/11)*exp(x/11) + pi**(x/11)
    assert _is_exponential(lhs, x)
    assert _solve_exponential(lhs, 0, x, S.Complexes) == FiniteSet(0)


def test_issue_17580():
    assert solveset(1/(1 - x**3)**2, x, S.Reals) is S.EmptySet


def test_issue_17566_actual():
    sys = [2**x + 2**y - 3, 4**x + 9**y - 5]
    # Not clear this is the correct result, but at least no recursion error
    assert nonlinsolve(sys, x, y) == FiniteSet((log(3 - 2**y)/log(2), y))


def test_issue_17565():
    eq = Ge(2*(x - 2)**2/(3*(x + 1)**(Integer(1)/3)) + 2*(x - 2)*(x + 1)**(Integer(2)/3), 0)
    res = Union(Interval.Lopen(-1, -Rational(1, 4)), Interval(2, oo))
    assert solveset(eq, x, S.Reals) == res


def test_issue_15024():
    function = (x + 5)/sqrt(-x**2 - 10*x)
    assert solveset(function, x, S.Reals) == FiniteSet(Integer(-5))


def test_issue_16877():
    assert dumeq(nonlinsolve([x - 1, sin(y)], x, y),
                 FiniteSet((1, ImageSet(Lambda(n, 2*n*pi), S.Integers)),
                           (1, ImageSet(Lambda(n, 2*n*pi + pi), S.Integers))))
    # Even better if (1, ImageSet(Lambda(n, n*pi), S.Integers)) is obtained


def test_issue_16876():
    assert dumeq(nonlinsolve([sin(x), 2*x - 4*y], x, y),
                 FiniteSet((ImageSet(Lambda(n, 2*n*pi), S.Integers),
                            ImageSet(Lambda(n, n*pi), S.Integers)),
                           (ImageSet(Lambda(n, 2*n*pi + pi), S.Integers),
                            ImageSet(Lambda(n, n*pi + pi/2), S.Integers))))
    # Even better if (ImageSet(Lambda(n, n*pi), S.Integers),
    #                 ImageSet(Lambda(n, n*pi/2), S.Integers)) is obtained

def test_issue_21236():
    x, z = symbols("x z")
    y = symbols('y', rational=True)
    assert solveset(x**y - z, x, S.Reals) == ConditionSet(x, Eq(x**y - z, 0), S.Reals)
    e1, e2 = symbols('e1 e2', even=True)
    y = e1/e2  # don't know if num or den will be odd and the other even
    assert solveset(x**y - z, x, S.Reals) == ConditionSet(x, Eq(x**y - z, 0), S.Reals)


def test_issue_21908():
    assert nonlinsolve([(x**2 + 2*x - y**2)*exp(x), -2*y*exp(x)], x, y
                      ) == {(-2, 0), (0, 0)}


def test_issue_19144():
    # test case 1
    expr1 = [x + y - 1, y**2 + 1]
    eq1 = [Eq(i, 0) for i in expr1]
    soln1 = {(1 - I, I), (1 + I, -I)}
    soln_expr1 = nonlinsolve(expr1, [x, y])
    soln_eq1 = nonlinsolve(eq1, [x, y])
    assert soln_eq1 == soln_expr1 == soln1
    # test case 2 - with denoms
    expr2 = [x/y - 1, y**2 + 1]
    eq2 = [Eq(i, 0) for i in expr2]
    soln2 = {(-I, -I), (I, I)}
    soln_expr2 = nonlinsolve(expr2, [x, y])
    soln_eq2 = nonlinsolve(eq2, [x, y])
    assert soln_eq2 == soln_expr2 == soln2
    # denominators that cancel in expression
    assert nonlinsolve([Eq(x + 1/x, 1/x)], [x]) == FiniteSet((S.EmptySet,))


def test_issue_22413():
    res =  nonlinsolve((4*y*(2*x + 2*exp(y) + 1)*exp(2*x),
                         4*x*exp(2*x) + 4*y*exp(2*x + y) + 4*exp(2*x + y) + 1),
                        x, y)
    # First solution is not correct, but the issue was an exception
    sols = FiniteSet((x, S.Zero), (-exp(y) - S.Half, y))
    assert res == sols


def test_issue_23318():
    eqs_eq = [
        Eq(53.5780461486929, x * log(y / (5.0 - y) + 1) / y),
        Eq(x, 0.0015 * z),
        Eq(0.0015, 7845.32 * y / z),
    ]
    eqs_expr = [eq.lhs - eq.rhs for eq in eqs_eq]

    sol = {(266.97755814852, 0.0340301680681629, 177985.03876568)}

    assert_close_nl(nonlinsolve(eqs_eq, [x, y, z]), sol)
    assert_close_nl(nonlinsolve(eqs_expr, [x, y, z]), sol)

    logterm = log(1.91196789933362e-7*z/(5.0 - 1.91196789933362e-7*z) + 1)
    eq = -0.0015*z*logterm + 1.02439504345316e-5*z
    assert_close_ss(solveset(eq, z), {0, 177985.038765679})


def test_issue_19814():
    assert nonlinsolve([ 2**m - 2**(2*n), 4*2**m - 2**(4*n)], m, n
                      ) == FiniteSet((log(2**(2*n))/log(2), S.Complexes))


def test_issue_22058():
    sol = solveset(-sqrt(t)*x**2 + 2*x + sqrt(t), x, S.Reals)
    # doesn't fail (and following numerical check)
    assert sol.xreplace({t: 1}) == {1 - sqrt(2), 1 + sqrt(2)}, sol.xreplace({t: 1})


def test_issue_11184():
    assert solveset(20*sqrt(y**2 + (sqrt(-(y - 10)*(y + 10)) + 10)**2) - 60, y, S.Reals) is S.EmptySet


def test_issue_21890():
    e = S(2)/3
    assert nonlinsolve([4*x**3*y**4 - 2*y, 4*x**4*y**3 - 2*x], x, y) == {
        (2**e/(2*y), y), ((-2**e/4 - 2**e*sqrt(3)*I/4)/y, y),
        ((-2**e/4 + 2**e*sqrt(3)*I/4)/y, y)}
    assert nonlinsolve([(1 - 4*x**2)*exp(-2*x**2 - 2*y**2),
        -4*x*y*exp(-2*x**2)*exp(-2*y**2)], x, y) == {(-S(1)/2, 0), (S(1)/2, 0)}
    rx, ry = symbols('x y', real=True)
    sol = nonlinsolve([4*rx**3*ry**4 - 2*ry, 4*rx**4*ry**3 - 2*rx], rx, ry)
    ans = {(2**(S(2)/3)/(2*ry), ry),
        ((-2**(S(2)/3)/4 - 2**(S(2)/3)*sqrt(3)*I/4)/ry, ry),
        ((-2**(S(2)/3)/4 + 2**(S(2)/3)*sqrt(3)*I/4)/ry, ry)}
    assert sol == ans


def test_issue_22628():
    assert nonlinsolve([h - 1, k - 1, f - 2, f - 4, -2*k], h, k, f) == S.EmptySet
    assert nonlinsolve([x**3 - 1, x + y, x**2 - 4], [x, y]) == S.EmptySet


def test_issue_25781():
    assert solve(sqrt(x/2) - x) == [0, S.Half]


def test_issue_26077():
    _n = Symbol('_n')
    function = x*cot(5*x)
    critical_points = stationary_points(function, x, S.Reals)
    excluded_points = Union(
        ImageSet(Lambda(_n, 2*_n*pi/5), S.Integers),
        ImageSet(Lambda(_n, 2*_n*pi/5 + pi/5), S.Integers)
    )
    solution = ConditionSet(x,
        Eq(x*(-5*cot(5*x)**2 - 5) + cot(5*x), 0),
        Complement(S.Reals, excluded_points)
    )
    assert solution.as_dummy() == critical_points.as_dummy()
