from sympy.core.function import Lambda
from sympy.core.numbers import (E, I, Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.elementary.complexes import (Abs, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import frac
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (
    cos, cot, csc, sec, sin, tan, asin, acos, atan, acot, asec, acsc)
from sympy.functions.elementary.hyperbolic import (sinh, cosh, tanh, coth,
    sech, csch, asinh, acosh, atanh, acoth, asech, acsch)
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.error_functions import expint
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.simplify.simplify import simplify
from sympy.calculus.util import (function_range, continuous_domain, not_empty_in,
                                 periodicity, lcim, is_convex,
                                 stationary_points, minimum, maximum)
from sympy.sets.sets import (Interval, FiniteSet, Complement, Union)
from sympy.sets.fancysets import ImageSet
from sympy.sets.conditionset import ConditionSet
from sympy.testing.pytest import XFAIL, raises, _both_exp_pow, slow
from sympy.abc import x, y

a = Symbol('a', real=True)

def test_function_range():
    assert function_range(sin(x), x, Interval(-pi/2, pi/2)
        ) == Interval(-1, 1)
    assert function_range(sin(x), x, Interval(0, pi)
        ) == Interval(0, 1)
    assert function_range(tan(x), x, Interval(0, pi)
        ) == Interval(-oo, oo)
    assert function_range(tan(x), x, Interval(pi/2, pi)
        ) == Interval(-oo, 0)
    assert function_range((x + 3)/(x - 2), x, Interval(-5, 5)
        ) == Union(Interval(-oo, Rational(2, 7)), Interval(Rational(8, 3), oo))
    assert function_range(1/(x**2), x, Interval(-1, 1)
        ) == Interval(1, oo)
    assert function_range(exp(x), x, Interval(-1, 1)
        ) == Interval(exp(-1), exp(1))
    assert function_range(log(x) - x, x, S.Reals
        ) == Interval(-oo, -1)
    assert function_range(sqrt(3*x - 1), x, Interval(0, 2)
        ) == Interval(0, sqrt(5))
    assert function_range(x*(x - 1) - (x**2 - x), x, S.Reals
        ) == FiniteSet(0)
    assert function_range(x*(x - 1) - (x**2 - x) + y, x, S.Reals
        ) == FiniteSet(y)
    assert function_range(sin(x), x, Union(Interval(-5, -3), FiniteSet(4))
        ) == Union(Interval(-sin(3), 1), FiniteSet(sin(4)))
    assert function_range(cos(x), x, Interval(-oo, -4)
        ) == Interval(-1, 1)
    assert function_range(cos(x), x, S.EmptySet) == S.EmptySet
    assert function_range(x/sqrt(x**2+1), x, S.Reals) == Interval.open(-1,1)
    raises(NotImplementedError, lambda : function_range(
        exp(x)*(sin(x) - cos(x))/2 - x, x, S.Reals))
    raises(NotImplementedError, lambda : function_range(
        sin(x) + x, x, S.Reals)) # issue 13273
    raises(NotImplementedError, lambda : function_range(
        log(x), x, S.Integers))
    raises(NotImplementedError, lambda : function_range(
        sin(x)/2, x, S.Naturals))


@slow
def test_function_range1():
    assert function_range(tan(x)**2 + tan(3*x)**2 + 1, x, S.Reals) == Interval(1,oo)


def test_continuous_domain():
    assert continuous_domain(sin(x), x, Interval(0, 2*pi)) == Interval(0, 2*pi)
    assert continuous_domain(tan(x), x, Interval(0, 2*pi)) == \
        Union(Interval(0, pi/2, False, True), Interval(pi/2, pi*Rational(3, 2), True, True),
              Interval(pi*Rational(3, 2), 2*pi, True, False))
    assert continuous_domain(cot(x), x, Interval(0, 2*pi)) == Union(
        Interval.open(0, pi), Interval.open(pi, 2*pi))
    assert continuous_domain((x - 1)/((x - 1)**2), x, S.Reals) == \
        Union(Interval(-oo, 1, True, True), Interval(1, oo, True, True))
    assert continuous_domain(log(x) + log(4*x - 1), x, S.Reals) == \
        Interval(Rational(1, 4), oo, True, True)
    assert continuous_domain(1/sqrt(x - 3), x, S.Reals) == Interval(3, oo, True, True)
    assert continuous_domain(1/x - 2, x, S.Reals) == \
        Union(Interval.open(-oo, 0), Interval.open(0, oo))
    assert continuous_domain(1/(x**2 - 4) + 2, x, S.Reals) == \
        Union(Interval.open(-oo, -2), Interval.open(-2, 2), Interval.open(2, oo))
    assert continuous_domain((x+1)**pi, x, S.Reals) == Interval(-1, oo)
    assert continuous_domain((x+1)**(pi/2), x, S.Reals) == Interval(-1, oo)
    assert continuous_domain(x**x, x, S.Reals) == Interval(0, oo)
    assert continuous_domain((x+1)**log(x**2), x, S.Reals) == Union(
        Interval.Ropen(-1, 0), Interval.open(0, oo))
    domain = continuous_domain(log(tan(x)**2 + 1), x, S.Reals)
    assert not domain.contains(3*pi/2)
    assert domain.contains(5)
    d = Symbol('d', even=True, zero=False)
    assert continuous_domain(x**(1/d), x, S.Reals) == Interval(0, oo)
    n = Dummy('n')
    assert continuous_domain(1/sin(x), x, S.Reals).dummy_eq(Complement(
        S.Reals, Union(ImageSet(Lambda(n, 2*n*pi + pi), S.Integers),
                       ImageSet(Lambda(n, 2*n*pi), S.Integers))))
    assert continuous_domain(sin(x) + cos(x), x, S.Reals) == S.Reals
    assert continuous_domain(asin(x), x, S.Reals) == Interval(-1, 1) # issue #21786
    assert continuous_domain(1/acos(log(x)), x, S.Reals) == Interval.Ropen(exp(-1), E)
    assert continuous_domain(sinh(x)+cosh(x), x, S.Reals) == S.Reals
    assert continuous_domain(tanh(x)+sech(x), x, S.Reals) == S.Reals
    assert continuous_domain(atan(x)+asinh(x), x, S.Reals) == S.Reals
    assert continuous_domain(acosh(x), x, S.Reals) == Interval(1, oo)
    assert continuous_domain(atanh(x), x, S.Reals) == Interval.open(-1, 1)
    assert continuous_domain(atanh(x)+acosh(x), x, S.Reals) == S.EmptySet
    assert continuous_domain(asech(x), x, S.Reals) == Interval.Lopen(0, 1)
    assert continuous_domain(acoth(x), x, S.Reals) == Union(
        Interval.open(-oo, -1), Interval.open(1, oo))
    assert continuous_domain(asec(x), x, S.Reals) == Union(
        Interval(-oo, -1), Interval(1, oo))
    assert continuous_domain(acsc(x), x, S.Reals) == Union(
        Interval(-oo, -1), Interval(1, oo))
    for f in (coth, acsch, csch):
        assert continuous_domain(f(x), x, S.Reals) == Union(
            Interval.open(-oo, 0), Interval.open(0, oo))
    assert continuous_domain(acot(x), x, S.Reals).contains(0) == False
    assert continuous_domain(1/(exp(x) - x), x, S.Reals) == Complement(
        S.Reals, ConditionSet(x, Eq(-x + exp(x), 0), S.Reals))
    assert continuous_domain(frac(x**2), x, Interval(-2,-1)) == Union(
        Interval.open(-2, -sqrt(3)), Interval.open(-sqrt(2), -1),
        Interval.open(-sqrt(3), -sqrt(2)))
    assert continuous_domain(frac(x), x, S.Reals) == Complement(
        S.Reals, S.Integers)
    raises(NotImplementedError, lambda : continuous_domain(
        1/(x**2+1), x, S.Complexes))
    raises(NotImplementedError, lambda : continuous_domain(
        gamma(x), x, Interval(-5,0)))
    assert continuous_domain(x + gamma(pi), x, S.Reals) == S.Reals


@XFAIL
def test_continuous_domain_acot():
    acot_cont = Piecewise((pi+acot(x), x<0), (acot(x), True))
    assert continuous_domain(acot_cont, x, S.Reals) == S.Reals

@XFAIL
def test_continuous_domain_gamma():
    assert continuous_domain(gamma(x), x, S.Reals).contains(-1) == False

@XFAIL
def test_continuous_domain_neg_power():
    assert continuous_domain((x-2)**(1-x), x, S.Reals) == Interval.open(2, oo)


def test_not_empty_in():
    assert not_empty_in(FiniteSet(x, 2*x).intersect(Interval(1, 2, True, False)), x) == \
        Interval(S.Half, 2, True, False)
    assert not_empty_in(FiniteSet(x, x**2).intersect(Interval(1, 2)), x) == \
        Union(Interval(-sqrt(2), -1), Interval(1, 2))
    assert not_empty_in(FiniteSet(x**2 + x, x).intersect(Interval(2, 4)), x) == \
        Union(Interval(-sqrt(17)/2 - S.Half, -2),
              Interval(1, Rational(-1, 2) + sqrt(17)/2), Interval(2, 4))
    assert not_empty_in(FiniteSet(x/(x - 1)).intersect(S.Reals), x) == \
        Complement(S.Reals, FiniteSet(1))
    assert not_empty_in(FiniteSet(a/(a - 1)).intersect(S.Reals), a) == \
        Complement(S.Reals, FiniteSet(1))
    assert not_empty_in(FiniteSet((x**2 - 3*x + 2)/(x - 1)).intersect(S.Reals), x) == \
        Complement(S.Reals, FiniteSet(1))
    assert not_empty_in(FiniteSet(3, 4, x/(x - 1)).intersect(Interval(2, 3)), x) == \
        Interval(-oo, oo)
    assert not_empty_in(FiniteSet(4, x/(x - 1)).intersect(Interval(2, 3)), x) == \
        Interval(S(3)/2, 2)
    assert not_empty_in(FiniteSet(x/(x**2 - 1)).intersect(S.Reals), x) == \
        Complement(S.Reals, FiniteSet(-1, 1))
    assert not_empty_in(FiniteSet(x, x**2).intersect(Union(Interval(1, 3, True, True),
                                                           Interval(4, 5))), x) == \
        Union(Interval(-sqrt(5), -2), Interval(-sqrt(3), -1, True, True),
              Interval(1, 3, True, True), Interval(4, 5))
    assert not_empty_in(FiniteSet(1).intersect(Interval(3, 4)), x) == S.EmptySet
    assert not_empty_in(FiniteSet(x**2/(x + 2)).intersect(Interval(1, oo)), x) == \
        Union(Interval(-2, -1, True, False), Interval(2, oo))
    raises(ValueError, lambda: not_empty_in(x))
    raises(ValueError, lambda: not_empty_in(Interval(0, 1), x))
    raises(NotImplementedError,
           lambda: not_empty_in(FiniteSet(x).intersect(S.Reals), x, a))


@_both_exp_pow
def test_periodicity():
    assert periodicity(sin(2*x), x) == pi
    assert periodicity((-2)*tan(4*x), x) == pi/4
    assert periodicity(sin(x)**2, x) == 2*pi
    assert periodicity(3**tan(3*x), x) == pi/3
    assert periodicity(tan(x)*cos(x), x) == 2*pi
    assert periodicity(sin(x)**(tan(x)), x) == 2*pi
    assert periodicity(tan(x)*sec(x), x) == 2*pi
    assert periodicity(sin(2*x)*cos(2*x) - y, x) == pi/2
    assert periodicity(tan(x) + cot(x), x) == pi
    assert periodicity(sin(x) - cos(2*x), x) == 2*pi
    assert periodicity(sin(x) - 1, x) == 2*pi
    assert periodicity(sin(4*x) + sin(x)*cos(x), x) == pi
    assert periodicity(exp(sin(x)), x) == 2*pi
    assert periodicity(log(cot(2*x)) - sin(cos(2*x)), x) == pi
    assert periodicity(sin(2*x)*exp(tan(x) - csc(2*x)), x) == pi
    assert periodicity(cos(sec(x) - csc(2*x)), x) == 2*pi
    assert periodicity(tan(sin(2*x)), x) == pi
    assert periodicity(2*tan(x)**2, x) == pi
    assert periodicity(sin(x%4), x) == 4
    assert periodicity(sin(x)%4, x) == 2*pi
    assert periodicity(tan((3*x-2)%4), x) == Rational(4, 3)
    assert periodicity((sqrt(2)*(x+1)+x) % 3, x) == 3 / (sqrt(2)+1)
    assert periodicity((x**2+1) % x, x) is None
    assert periodicity(sin(re(x)), x) == 2*pi
    assert periodicity(sin(x)**2 + cos(x)**2, x) is S.Zero
    assert periodicity(tan(x), y) is S.Zero
    assert periodicity(sin(x) + I*cos(x), x) == 2*pi
    assert periodicity(x - sin(2*y), y) == pi

    assert periodicity(exp(x), x) is None
    assert periodicity(exp(I*x), x) == 2*pi
    assert periodicity(exp(I*a), a) == 2*pi
    assert periodicity(exp(a), a) is None
    assert periodicity(exp(log(sin(a) + I*cos(2*a)), evaluate=False), a) == 2*pi
    assert periodicity(exp(log(sin(2*a) + I*cos(a)), evaluate=False), a) == 2*pi
    assert periodicity(exp(sin(a)), a) == 2*pi
    assert periodicity(exp(2*I*a), a) == pi
    assert periodicity(exp(a + I*sin(a)), a) is None
    assert periodicity(exp(cos(a/2) + sin(a)), a) == 4*pi
    assert periodicity(log(x), x) is None
    assert periodicity(exp(x)**sin(x), x) is None
    assert periodicity(sin(x)**y, y) is None

    assert periodicity(Abs(sin(Abs(sin(x)))), x) == pi
    assert all(periodicity(Abs(f(x)), x) == pi for f in (
        cos, sin, sec, csc, tan, cot))
    assert periodicity(Abs(sin(tan(x))), x) == pi
    assert periodicity(Abs(sin(sin(x) + tan(x))), x) == 2*pi
    assert periodicity(sin(x) > S.Half, x) == 2*pi

    assert periodicity(x > 2, x) is None
    assert periodicity(x**3 - x**2 + 1, x) is None
    assert periodicity(Abs(x), x) is None
    assert periodicity(Abs(x**2 - 1), x) is None

    assert periodicity((x**2 + 4)%2, x) is None
    assert periodicity((E**x)%3, x) is None

    assert periodicity(sin(expint(1, x))/expint(1, x), x) is None
    # returning `None` for any Piecewise
    p = Piecewise((0, x < -1), (x**2, x <= 1), (log(x), True))
    assert periodicity(p, x) is None

    m = MatrixSymbol('m', 3, 3)
    raises(NotImplementedError, lambda: periodicity(sin(m), m))
    raises(NotImplementedError, lambda: periodicity(sin(m[0, 0]), m))
    raises(NotImplementedError, lambda: periodicity(sin(m), m[0, 0]))
    raises(NotImplementedError, lambda: periodicity(sin(m[0, 0]), m[0, 0]))


def test_periodicity_check():
    assert periodicity(tan(x), x, check=True) == pi
    assert periodicity(sin(x) + cos(x), x, check=True) == 2*pi
    assert periodicity(sec(x), x) == 2*pi
    assert periodicity(sin(x*y), x) == 2*pi/abs(y)
    assert periodicity(Abs(sec(sec(x))), x) == pi


def test_lcim():
    assert lcim([S.Half, S(2), S(3)]) == 6
    assert lcim([pi/2, pi/4, pi]) == pi
    assert lcim([2*pi, pi/2]) == 2*pi
    assert lcim([S.One, 2*pi]) is None
    assert lcim([S(2) + 2*E, E/3 + Rational(1, 3), S.One + E]) == S(2) + 2*E


def test_is_convex():
    assert is_convex(1/x, x, domain=Interval.open(0, oo)) == True
    assert is_convex(1/x, x, domain=Interval(-oo, 0)) == False
    assert is_convex(x**2, x, domain=Interval(0, oo)) == True
    assert is_convex(1/x**3, x, domain=Interval.Lopen(0, oo)) == True
    assert is_convex(-1/x**3, x, domain=Interval.Ropen(-oo, 0)) == True
    assert is_convex(log(x) ,x) == False
    assert is_convex(x**2+y**2, x, y) == True
    assert is_convex(cos(x) + cos(y), x) == False
    assert is_convex(8*x**2 - 2*y**2, x, y) == False


def test_stationary_points():
    assert stationary_points(sin(x), x, Interval(-pi/2, pi/2)
        ) == {-pi/2, pi/2}
    assert  stationary_points(sin(x), x, Interval.Ropen(0, pi/4)
        ) is S.EmptySet
    assert stationary_points(tan(x), x,
        ) is S.EmptySet
    assert stationary_points(sin(x)*cos(x), x, Interval(0, pi)
        ) == {pi/4, pi*Rational(3, 4)}
    assert stationary_points(sec(x), x, Interval(0, pi)
        ) == {0, pi}
    assert stationary_points((x+3)*(x-2), x
        ) == FiniteSet(Rational(-1, 2))
    assert stationary_points((x + 3)/(x - 2), x, Interval(-5, 5)
        ) is S.EmptySet
    assert stationary_points((x**2+3)/(x-2), x
        ) == {2 - sqrt(7), 2 + sqrt(7)}
    assert stationary_points((x**2+3)/(x-2), x, Interval(0, 5)
        ) == {2 + sqrt(7)}
    assert stationary_points(x**4 + x**3 - 5*x**2, x, S.Reals
        ) == FiniteSet(-2, 0, Rational(5, 4))
    assert stationary_points(exp(x), x
        ) is S.EmptySet
    assert stationary_points(log(x) - x, x, S.Reals
        ) == {1}
    assert stationary_points(cos(x), x, Union(Interval(0, 5), Interval(-6, -3))
        ) == {0, -pi, pi}
    assert stationary_points(y, x, S.Reals
        ) == S.Reals
    assert stationary_points(y, x, S.EmptySet) == S.EmptySet


def test_maximum():
    assert maximum(sin(x), x) is S.One
    assert maximum(sin(x), x, Interval(0, 1)) == sin(1)
    assert maximum(tan(x), x) is oo
    assert maximum(tan(x), x, Interval(-pi/4, pi/4)) is S.One
    assert maximum(sin(x)*cos(x), x, S.Reals) == S.Half
    assert simplify(maximum(sin(x)*cos(x), x, Interval(pi*Rational(3, 8), pi*Rational(5, 8)))
        ) == sqrt(2)/4
    assert maximum((x+3)*(x-2), x) is oo
    assert maximum((x+3)*(x-2), x, Interval(-5, 0)) == S(14)
    assert maximum((x+3)/(x-2), x, Interval(-5, 0)) == Rational(2, 7)
    assert simplify(maximum(-x**4-x**3+x**2+10, x)
        ) == 41*sqrt(41)/512 + Rational(5419, 512)
    assert maximum(exp(x), x, Interval(-oo, 2)) == exp(2)
    assert maximum(log(x) - x, x, S.Reals) is S.NegativeOne
    assert maximum(cos(x), x, Union(Interval(0, 5), Interval(-6, -3))
        ) is S.One
    assert maximum(cos(x)-sin(x), x, S.Reals) == sqrt(2)
    assert maximum(y, x, S.Reals) == y
    assert maximum(abs(a**3 + a), a, Interval(0, 2)) == 10
    assert maximum(abs(60*a**3 + 24*a), a, Interval(0, 2)) == 528
    assert maximum(abs(12*a*(5*a**2 + 2)), a, Interval(0, 2)) == 528
    assert maximum(x/sqrt(x**2+1), x, S.Reals) == 1

    raises(ValueError, lambda : maximum(sin(x), x, S.EmptySet))
    raises(ValueError, lambda : maximum(log(cos(x)), x, S.EmptySet))
    raises(ValueError, lambda : maximum(1/(x**2 + y**2 + 1), x, S.EmptySet))
    raises(ValueError, lambda : maximum(sin(x), sin(x)))
    raises(ValueError, lambda : maximum(sin(x), x*y, S.EmptySet))
    raises(ValueError, lambda : maximum(sin(x), S.One))


def test_minimum():
    assert minimum(sin(x), x) is S.NegativeOne
    assert minimum(sin(x), x, Interval(1, 4)) == sin(4)
    assert minimum(tan(x), x) is -oo
    assert minimum(tan(x), x, Interval(-pi/4, pi/4)) is S.NegativeOne
    assert minimum(sin(x)*cos(x), x, S.Reals) == Rational(-1, 2)
    assert simplify(minimum(sin(x)*cos(x), x, Interval(pi*Rational(3, 8), pi*Rational(5, 8)))
        ) == -sqrt(2)/4
    assert minimum((x+3)*(x-2), x) == Rational(-25, 4)
    assert minimum((x+3)/(x-2), x, Interval(-5, 0)) == Rational(-3, 2)
    assert minimum(x**4-x**3+x**2+10, x) == S(10)
    assert minimum(exp(x), x, Interval(-2, oo)) == exp(-2)
    assert minimum(log(x) - x, x, S.Reals) is -oo
    assert minimum(cos(x), x, Union(Interval(0, 5), Interval(-6, -3))
        ) is S.NegativeOne
    assert minimum(cos(x)-sin(x), x, S.Reals) == -sqrt(2)
    assert minimum(y, x, S.Reals) == y
    assert minimum(x/sqrt(x**2+1), x, S.Reals) == -1

    raises(ValueError, lambda : minimum(sin(x), x, S.EmptySet))
    raises(ValueError, lambda : minimum(log(cos(x)), x, S.EmptySet))
    raises(ValueError, lambda : minimum(1/(x**2 + y**2 + 1), x, S.EmptySet))
    raises(ValueError, lambda : minimum(sin(x), sin(x)))
    raises(ValueError, lambda : minimum(sin(x), x*y, S.EmptySet))
    raises(ValueError, lambda : minimum(sin(x), S.One))


def test_issue_19869():
    assert (maximum(sqrt(3)*(x - 1)/(3*sqrt(x**2 + 1)), x)
        ) == sqrt(3)/3


def test_issue_16469():
    f = abs(a)
    assert function_range(f, a, S.Reals) == Interval(0, oo, False, True)


@_both_exp_pow
def test_issue_18747():
    assert periodicity(exp(pi*I*(x/4 + S.Half/2)), x) == 8


def test_issue_25942():
    assert (acos(x) > pi/3).as_set() == Interval.Ropen(-1, S(1)/2)
