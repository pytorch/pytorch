from sympy.sets.setexpr import SetExpr
from sympy.sets import Interval, FiniteSet, Intersection, ImageSet, Union

from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import cos
from sympy.sets.sets import Set


a, x = symbols("a, x")
_d = Dummy("d")


def test_setexpr():
    se = SetExpr(Interval(0, 1))
    assert isinstance(se.set, Set)
    assert isinstance(se, Expr)


def test_scalar_funcs():
    assert SetExpr(Interval(0, 1)).set == Interval(0, 1)
    a, b = Symbol('a', real=True), Symbol('b', real=True)
    a, b = 1, 2
    # TODO: add support for more functions in the future:
    for f in [exp, log]:
        input_se = f(SetExpr(Interval(a, b)))
        output = input_se.set
        expected = Interval(Min(f(a), f(b)), Max(f(a), f(b)))
        assert output == expected


def test_Add_Mul():
    assert (SetExpr(Interval(0, 1)) + 1).set == Interval(1, 2)
    assert (SetExpr(Interval(0, 1))*2).set == Interval(0, 2)


def test_Pow():
    assert (SetExpr(Interval(0, 2))**2).set == Interval(0, 4)


def test_compound():
    assert (exp(SetExpr(Interval(0, 1))*2 + 1)).set == \
           Interval(exp(1), exp(3))


def test_Interval_Interval():
    assert (SetExpr(Interval(1, 2)) + SetExpr(Interval(10, 20))).set == \
           Interval(11, 22)
    assert (SetExpr(Interval(1, 2))*SetExpr(Interval(10, 20))).set == \
           Interval(10, 40)


def test_FiniteSet_FiniteSet():
    assert (SetExpr(FiniteSet(1, 2, 3)) + SetExpr(FiniteSet(1, 2))).set == \
           FiniteSet(2, 3, 4, 5)
    assert (SetExpr(FiniteSet(1, 2, 3))*SetExpr(FiniteSet(1, 2))).set == \
           FiniteSet(1, 2, 3, 4, 6)


def test_Interval_FiniteSet():
    assert (SetExpr(FiniteSet(1, 2)) + SetExpr(Interval(0, 10))).set == \
           Interval(1, 12)


def test_Many_Sets():
    assert (SetExpr(Interval(0, 1)) +
            SetExpr(Interval(2, 3)) +
            SetExpr(FiniteSet(10, 11, 12))).set == Interval(12, 16)


def test_same_setexprs_are_not_identical():
    a = SetExpr(FiniteSet(0, 1))
    b = SetExpr(FiniteSet(0, 1))
    assert (a + b).set == FiniteSet(0, 1, 2)

    # Cannot detect the set being the same:
    # assert (a + a).set == FiniteSet(0, 2)


def test_Interval_arithmetic():
    i12cc = SetExpr(Interval(1, 2))
    i12lo = SetExpr(Interval.Lopen(1, 2))
    i12ro = SetExpr(Interval.Ropen(1, 2))
    i12o = SetExpr(Interval.open(1, 2))

    n23cc = SetExpr(Interval(-2, 3))
    n23lo = SetExpr(Interval.Lopen(-2, 3))
    n23ro = SetExpr(Interval.Ropen(-2, 3))
    n23o = SetExpr(Interval.open(-2, 3))

    n3n2cc = SetExpr(Interval(-3, -2))

    assert i12cc + i12cc == SetExpr(Interval(2, 4))
    assert i12cc - i12cc == SetExpr(Interval(-1, 1))
    assert i12cc*i12cc == SetExpr(Interval(1, 4))
    assert i12cc/i12cc == SetExpr(Interval(S.Half, 2))
    assert i12cc**2 == SetExpr(Interval(1, 4))
    assert i12cc**3 == SetExpr(Interval(1, 8))

    assert i12lo + i12ro == SetExpr(Interval.open(2, 4))
    assert i12lo - i12ro == SetExpr(Interval.Lopen(-1, 1))
    assert i12lo*i12ro == SetExpr(Interval.open(1, 4))
    assert i12lo/i12ro == SetExpr(Interval.Lopen(S.Half, 2))
    assert i12lo + i12lo == SetExpr(Interval.Lopen(2, 4))
    assert i12lo - i12lo == SetExpr(Interval.open(-1, 1))
    assert i12lo*i12lo == SetExpr(Interval.Lopen(1, 4))
    assert i12lo/i12lo == SetExpr(Interval.open(S.Half, 2))
    assert i12lo + i12cc == SetExpr(Interval.Lopen(2, 4))
    assert i12lo - i12cc == SetExpr(Interval.Lopen(-1, 1))
    assert i12lo*i12cc == SetExpr(Interval.Lopen(1, 4))
    assert i12lo/i12cc == SetExpr(Interval.Lopen(S.Half, 2))
    assert i12lo + i12o == SetExpr(Interval.open(2, 4))
    assert i12lo - i12o == SetExpr(Interval.open(-1, 1))
    assert i12lo*i12o == SetExpr(Interval.open(1, 4))
    assert i12lo/i12o == SetExpr(Interval.open(S.Half, 2))
    assert i12lo**2 == SetExpr(Interval.Lopen(1, 4))
    assert i12lo**3 == SetExpr(Interval.Lopen(1, 8))

    assert i12ro + i12ro == SetExpr(Interval.Ropen(2, 4))
    assert i12ro - i12ro == SetExpr(Interval.open(-1, 1))
    assert i12ro*i12ro == SetExpr(Interval.Ropen(1, 4))
    assert i12ro/i12ro == SetExpr(Interval.open(S.Half, 2))
    assert i12ro + i12cc == SetExpr(Interval.Ropen(2, 4))
    assert i12ro - i12cc == SetExpr(Interval.Ropen(-1, 1))
    assert i12ro*i12cc == SetExpr(Interval.Ropen(1, 4))
    assert i12ro/i12cc == SetExpr(Interval.Ropen(S.Half, 2))
    assert i12ro + i12o == SetExpr(Interval.open(2, 4))
    assert i12ro - i12o == SetExpr(Interval.open(-1, 1))
    assert i12ro*i12o == SetExpr(Interval.open(1, 4))
    assert i12ro/i12o == SetExpr(Interval.open(S.Half, 2))
    assert i12ro**2 == SetExpr(Interval.Ropen(1, 4))
    assert i12ro**3 == SetExpr(Interval.Ropen(1, 8))

    assert i12o + i12lo == SetExpr(Interval.open(2, 4))
    assert i12o - i12lo == SetExpr(Interval.open(-1, 1))
    assert i12o*i12lo == SetExpr(Interval.open(1, 4))
    assert i12o/i12lo == SetExpr(Interval.open(S.Half, 2))
    assert i12o + i12ro == SetExpr(Interval.open(2, 4))
    assert i12o - i12ro == SetExpr(Interval.open(-1, 1))
    assert i12o*i12ro == SetExpr(Interval.open(1, 4))
    assert i12o/i12ro == SetExpr(Interval.open(S.Half, 2))
    assert i12o + i12cc == SetExpr(Interval.open(2, 4))
    assert i12o - i12cc == SetExpr(Interval.open(-1, 1))
    assert i12o*i12cc == SetExpr(Interval.open(1, 4))
    assert i12o/i12cc == SetExpr(Interval.open(S.Half, 2))
    assert i12o**2 == SetExpr(Interval.open(1, 4))
    assert i12o**3 == SetExpr(Interval.open(1, 8))

    assert n23cc + n23cc == SetExpr(Interval(-4, 6))
    assert n23cc - n23cc == SetExpr(Interval(-5, 5))
    assert n23cc*n23cc == SetExpr(Interval(-6, 9))
    assert n23cc/n23cc == SetExpr(Interval.open(-oo, oo))
    assert n23cc + n23ro == SetExpr(Interval.Ropen(-4, 6))
    assert n23cc - n23ro == SetExpr(Interval.Lopen(-5, 5))
    assert n23cc*n23ro == SetExpr(Interval.Ropen(-6, 9))
    assert n23cc/n23ro == SetExpr(Interval.Lopen(-oo, oo))
    assert n23cc + n23lo == SetExpr(Interval.Lopen(-4, 6))
    assert n23cc - n23lo == SetExpr(Interval.Ropen(-5, 5))
    assert n23cc*n23lo == SetExpr(Interval(-6, 9))
    assert n23cc/n23lo == SetExpr(Interval.open(-oo, oo))
    assert n23cc + n23o == SetExpr(Interval.open(-4, 6))
    assert n23cc - n23o == SetExpr(Interval.open(-5, 5))
    assert n23cc*n23o == SetExpr(Interval.open(-6, 9))
    assert n23cc/n23o == SetExpr(Interval.open(-oo, oo))
    assert n23cc**2 == SetExpr(Interval(0, 9))
    assert n23cc**3 == SetExpr(Interval(-8, 27))

    n32cc = SetExpr(Interval(-3, 2))
    n32lo = SetExpr(Interval.Lopen(-3, 2))
    n32ro = SetExpr(Interval.Ropen(-3, 2))
    assert n32cc*n32lo == SetExpr(Interval.Ropen(-6, 9))
    assert n32cc*n32cc == SetExpr(Interval(-6, 9))
    assert n32lo*n32cc == SetExpr(Interval.Ropen(-6, 9))
    assert n32cc*n32ro == SetExpr(Interval(-6, 9))
    assert n32lo*n32ro == SetExpr(Interval.Ropen(-6, 9))
    assert n32cc/n32lo == SetExpr(Interval.Ropen(-oo, oo))
    assert i12cc/n32lo == SetExpr(Interval.Ropen(-oo, oo))

    assert n3n2cc**2 == SetExpr(Interval(4, 9))
    assert n3n2cc**3 == SetExpr(Interval(-27, -8))

    assert n23cc + i12cc == SetExpr(Interval(-1, 5))
    assert n23cc - i12cc == SetExpr(Interval(-4, 2))
    assert n23cc*i12cc == SetExpr(Interval(-4, 6))
    assert n23cc/i12cc == SetExpr(Interval(-2, 3))


def test_SetExpr_Intersection():
    x, y, z, w = symbols("x y z w")
    set1 = Interval(x, y)
    set2 = Interval(w, z)
    inter = Intersection(set1, set2)
    se = SetExpr(inter)
    assert exp(se).set == Intersection(
        ImageSet(Lambda(x, exp(x)), set1),
        ImageSet(Lambda(x, exp(x)), set2))
    assert cos(se).set == ImageSet(Lambda(x, cos(x)), inter)


def test_SetExpr_Interval_div():
    # TODO: some expressions cannot be calculated due to bugs (currently
    # commented):
    assert SetExpr(Interval(-3, -2))/SetExpr(Interval(-2, 1)) == SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(2, 3))/SetExpr(Interval(-2, 2)) == SetExpr(Interval(-oo, oo))

    assert SetExpr(Interval(-3, -2))/SetExpr(Interval(0, 4)) == SetExpr(Interval(-oo, Rational(-1, 2)))
    assert SetExpr(Interval(2, 4))/SetExpr(Interval(-3, 0)) == SetExpr(Interval(-oo, Rational(-2, 3)))
    assert SetExpr(Interval(2, 4))/SetExpr(Interval(0, 3)) == SetExpr(Interval(Rational(2, 3), oo))

    # assert SetExpr(Interval(0, 1))/SetExpr(Interval(0, 1)) == SetExpr(Interval(0, oo))
    # assert SetExpr(Interval(-1, 0))/SetExpr(Interval(0, 1)) == SetExpr(Interval(-oo, 0))
    assert SetExpr(Interval(-1, 2))/SetExpr(Interval(-2, 2)) == SetExpr(Interval(-oo, oo))

    assert 1/SetExpr(Interval(-1, 2)) == SetExpr(Union(Interval(-oo, -1), Interval(S.Half, oo)))

    assert 1/SetExpr(Interval(0, 2)) == SetExpr(Interval(S.Half, oo))
    assert (-1)/SetExpr(Interval(0, 2)) == SetExpr(Interval(-oo, Rational(-1, 2)))
    assert 1/SetExpr(Interval(-oo, 0)) == SetExpr(Interval.open(-oo, 0))
    assert 1/SetExpr(Interval(-1, 0)) == SetExpr(Interval(-oo, -1))
    # assert (-2)/SetExpr(Interval(-oo, 0)) == SetExpr(Interval(0, oo))
    # assert 1/SetExpr(Interval(-oo, -1)) == SetExpr(Interval(-1, 0))

    # assert SetExpr(Interval(1, 2))/a == Mul(SetExpr(Interval(1, 2)), 1/a, evaluate=False)

    # assert SetExpr(Interval(1, 2))/0 == SetExpr(Interval(1, 2))*zoo
    # assert SetExpr(Interval(1, oo))/oo == SetExpr(Interval(0, oo))
    # assert SetExpr(Interval(1, oo))/(-oo) == SetExpr(Interval(-oo, 0))
    # assert SetExpr(Interval(-oo, -1))/oo == SetExpr(Interval(-oo, 0))
    # assert SetExpr(Interval(-oo, -1))/(-oo) == SetExpr(Interval(0, oo))
    # assert SetExpr(Interval(-oo, oo))/oo == SetExpr(Interval(-oo, oo))
    # assert SetExpr(Interval(-oo, oo))/(-oo) == SetExpr(Interval(-oo, oo))
    # assert SetExpr(Interval(-1, oo))/oo == SetExpr(Interval(0, oo))
    # assert SetExpr(Interval(-1, oo))/(-oo) == SetExpr(Interval(-oo, 0))
    # assert SetExpr(Interval(-oo, 1))/oo == SetExpr(Interval(-oo, 0))
    # assert SetExpr(Interval(-oo, 1))/(-oo) == SetExpr(Interval(0, oo))


def test_SetExpr_Interval_pow():
    assert SetExpr(Interval(0, 2))**2 == SetExpr(Interval(0, 4))
    assert SetExpr(Interval(-1, 1))**2 == SetExpr(Interval(0, 1))
    assert SetExpr(Interval(1, 2))**2 == SetExpr(Interval(1, 4))
    assert SetExpr(Interval(-1, 2))**3 == SetExpr(Interval(-1, 8))
    assert SetExpr(Interval(-1, 1))**0 == SetExpr(FiniteSet(1))


    assert SetExpr(Interval(1, 2))**Rational(5, 2) == SetExpr(Interval(1, 4*sqrt(2)))
    #assert SetExpr(Interval(-1, 2))**Rational(1, 3) == SetExpr(Interval(-1, 2**Rational(1, 3)))
    #assert SetExpr(Interval(0, 2))**S.Half == SetExpr(Interval(0, sqrt(2)))

    #assert SetExpr(Interval(-4, 2))**Rational(2, 3) == SetExpr(Interval(0, 2*2**Rational(1, 3)))

    #assert SetExpr(Interval(-1, 5))**S.Half == SetExpr(Interval(0, sqrt(5)))
    #assert SetExpr(Interval(-oo, 2))**S.Half == SetExpr(Interval(0, sqrt(2)))
    #assert SetExpr(Interval(-2, 3))**(Rational(-1, 4)) == SetExpr(Interval(0, oo))

    assert SetExpr(Interval(1, 5))**(-2) == SetExpr(Interval(Rational(1, 25), 1))
    assert SetExpr(Interval(-1, 3))**(-2) == SetExpr(Interval(0, oo))

    assert SetExpr(Interval(0, 2))**(-2) == SetExpr(Interval(Rational(1, 4), oo))
    assert SetExpr(Interval(-1, 2))**(-3) == SetExpr(Union(Interval(-oo, -1), Interval(Rational(1, 8), oo)))
    assert SetExpr(Interval(-3, -2))**(-3) == SetExpr(Interval(Rational(-1, 8), Rational(-1, 27)))
    assert SetExpr(Interval(-3, -2))**(-2) == SetExpr(Interval(Rational(1, 9), Rational(1, 4)))
    #assert SetExpr(Interval(0, oo))**S.Half == SetExpr(Interval(0, oo))
    #assert SetExpr(Interval(-oo, -1))**Rational(1, 3) == SetExpr(Interval(-oo, -1))
    #assert SetExpr(Interval(-2, 3))**(Rational(-1, 3)) == SetExpr(Interval(-oo, oo))

    assert SetExpr(Interval(-oo, 0))**(-2) == SetExpr(Interval.open(0, oo))
    assert SetExpr(Interval(-2, 0))**(-2) == SetExpr(Interval(Rational(1, 4), oo))

    assert SetExpr(Interval(Rational(1, 3), S.Half))**oo == SetExpr(FiniteSet(0))
    assert SetExpr(Interval(0, S.Half))**oo == SetExpr(FiniteSet(0))
    assert SetExpr(Interval(S.Half, 1))**oo == SetExpr(Interval(0, oo))
    assert SetExpr(Interval(0, 1))**oo == SetExpr(Interval(0, oo))
    assert SetExpr(Interval(2, 3))**oo == SetExpr(FiniteSet(oo))
    assert SetExpr(Interval(1, 2))**oo == SetExpr(Interval(0, oo))
    assert SetExpr(Interval(S.Half, 3))**oo == SetExpr(Interval(0, oo))
    assert SetExpr(Interval(Rational(-1, 3), Rational(-1, 4)))**oo == SetExpr(FiniteSet(0))
    assert SetExpr(Interval(-1, Rational(-1, 2)))**oo == SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(-3, -2))**oo == SetExpr(FiniteSet(-oo, oo))
    assert SetExpr(Interval(-2, -1))**oo == SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(-2, Rational(-1, 2)))**oo == SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(Rational(-1, 2), S.Half))**oo == SetExpr(FiniteSet(0))
    assert SetExpr(Interval(Rational(-1, 2), 1))**oo == SetExpr(Interval(0, oo))
    assert SetExpr(Interval(Rational(-2, 3), 2))**oo == SetExpr(Interval(0, oo))
    assert SetExpr(Interval(-1, 1))**oo == SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(-1, S.Half))**oo == SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(-1, 2))**oo == SetExpr(Interval(-oo, oo))
    assert SetExpr(Interval(-2, S.Half))**oo == SetExpr(Interval(-oo, oo))

    assert (SetExpr(Interval(1, 2))**x).dummy_eq(SetExpr(ImageSet(Lambda(_d, _d**x), Interval(1, 2))))

    assert SetExpr(Interval(2, 3))**(-oo) == SetExpr(FiniteSet(0))
    assert SetExpr(Interval(0, 2))**(-oo) == SetExpr(Interval(0, oo))
    assert (SetExpr(Interval(-1, 2))**(-oo)).dummy_eq(SetExpr(ImageSet(Lambda(_d, _d**(-oo)), Interval(-1, 2))))


def test_SetExpr_Integers():
    assert SetExpr(S.Integers) + 1 == SetExpr(S.Integers)
    assert (SetExpr(S.Integers) + I).dummy_eq(
        SetExpr(ImageSet(Lambda(_d, _d + I), S.Integers)))
    assert SetExpr(S.Integers)*(-1) == SetExpr(S.Integers)
    assert (SetExpr(S.Integers)*2).dummy_eq(
        SetExpr(ImageSet(Lambda(_d, 2*_d), S.Integers)))
    assert (SetExpr(S.Integers)*I).dummy_eq(
        SetExpr(ImageSet(Lambda(_d, I*_d), S.Integers)))
    # issue #18050:
    assert SetExpr(S.Integers)._eval_func(Lambda(x, I*x + 1)).dummy_eq(
        SetExpr(ImageSet(Lambda(_d, I*_d + 1), S.Integers)))
    # needs improvement:
    assert (SetExpr(S.Integers)*I + 1).dummy_eq(
        SetExpr(ImageSet(Lambda(x, x + 1),
        ImageSet(Lambda(_d, _d*I), S.Integers))))
