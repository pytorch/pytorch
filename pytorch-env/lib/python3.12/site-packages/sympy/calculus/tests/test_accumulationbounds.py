from sympy.core.numbers import (E, Rational, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core import Add, Mul, Pow
from sympy.core.expr import unchanged
from sympy.testing.pytest import raises, XFAIL
from sympy.abc import x

a = Symbol('a', real=True)
B = AccumBounds


def test_AccumBounds():
    assert B(1, 2).args == (1, 2)
    assert B(1, 2).delta is S.One
    assert B(1, 2).mid == Rational(3, 2)
    assert B(1, 3).is_real == True

    assert B(1, 1) is S.One

    assert B(1, 2) + 1 == B(2, 3)
    assert 1 + B(1, 2) == B(2, 3)
    assert B(1, 2) + B(2, 3) == B(3, 5)

    assert -B(1, 2) == B(-2, -1)

    assert B(1, 2) - 1 == B(0, 1)
    assert 1 - B(1, 2) == B(-1, 0)
    assert B(2, 3) - B(1, 2) == B(0, 2)

    assert x + B(1, 2) == Add(B(1, 2), x)
    assert a + B(1, 2) == B(1 + a, 2 + a)
    assert B(1, 2) - x == Add(B(1, 2), -x)

    assert B(-oo, 1) + oo == B(-oo, oo)
    assert B(1, oo) + oo is oo
    assert B(1, oo) - oo == B(-oo, oo)
    assert (-oo - B(-1, oo)) is -oo
    assert B(-oo, 1) - oo is -oo

    assert B(1, oo) - oo == B(-oo, oo)
    assert B(-oo, 1) - (-oo) == B(-oo, oo)
    assert (oo - B(1, oo)) == B(-oo, oo)
    assert (-oo - B(1, oo)) is -oo

    assert B(1, 2)/2 == B(S.Half, 1)
    assert 2/B(2, 3) == B(Rational(2, 3), 1)
    assert 1/B(-1, 1) == B(-oo, oo)

    assert abs(B(1, 2)) == B(1, 2)
    assert abs(B(-2, -1)) == B(1, 2)
    assert abs(B(-2, 1)) == B(0, 2)
    assert abs(B(-1, 2)) == B(0, 2)
    c = Symbol('c')
    raises(ValueError, lambda: B(0, c))
    raises(ValueError, lambda: B(1, -1))
    r = Symbol('r', real=True)
    raises(ValueError, lambda: B(r, r - 1))


def test_AccumBounds_mul():
    assert B(1, 2)*2 == B(2, 4)
    assert 2*B(1, 2) == B(2, 4)
    assert B(1, 2)*B(2, 3) == B(2, 6)
    assert B(0, 2)*B(2, oo) == B(0, oo)
    l, r = B(-oo, oo), B(-a, a)
    assert l*r == B(-oo, oo)
    assert r*l == B(-oo, oo)
    l, r = B(1, oo), B(-3, -2)
    assert l*r == B(-oo, -2)
    assert r*l == B(-oo, -2)
    assert B(1, 2)*0 == 0
    assert B(1, oo)*0 == B(0, oo)
    assert B(-oo, 1)*0 == B(-oo, 0)
    assert B(-oo, oo)*0 == B(-oo, oo)

    assert B(1, 2)*x == Mul(B(1, 2), x, evaluate=False)

    assert B(0, 2)*oo == B(0, oo)
    assert B(-2, 0)*oo == B(-oo, 0)
    assert B(0, 2)*(-oo) == B(-oo, 0)
    assert B(-2, 0)*(-oo) == B(0, oo)
    assert B(-1, 1)*oo == B(-oo, oo)
    assert B(-1, 1)*(-oo) == B(-oo, oo)
    assert B(-oo, oo)*oo == B(-oo, oo)


def test_AccumBounds_div():
    assert B(-1, 3)/B(3, 4) == B(Rational(-1, 3), 1)
    assert B(-2, 4)/B(-3, 4) == B(-oo, oo)
    assert B(-3, -2)/B(-4, 0) == B(S.Half, oo)

    # these two tests can have a better answer
    # after Union of B is improved
    assert B(-3, -2)/B(-2, 1) == B(-oo, oo)
    assert B(2, 3)/B(-2, 2) == B(-oo, oo)

    assert B(-3, -2)/B(0, 4) == B(-oo, Rational(-1, 2))
    assert B(2, 4)/B(-3, 0) == B(-oo, Rational(-2, 3))
    assert B(2, 4)/B(0, 3) == B(Rational(2, 3), oo)

    assert B(0, 1)/B(0, 1) == B(0, oo)
    assert B(-1, 0)/B(0, 1) == B(-oo, 0)
    assert B(-1, 2)/B(-2, 2) == B(-oo, oo)

    assert 1/B(-1, 2) == B(-oo, oo)
    assert 1/B(0, 2) == B(S.Half, oo)
    assert (-1)/B(0, 2) == B(-oo, Rational(-1, 2))
    assert 1/B(-oo, 0) == B(-oo, 0)
    assert 1/B(-1, 0) == B(-oo, -1)
    assert (-2)/B(-oo, 0) == B(0, oo)
    assert 1/B(-oo, -1) == B(-1, 0)

    assert B(1, 2)/a == Mul(B(1, 2), 1/a, evaluate=False)

    assert B(1, 2)/0 == B(1, 2)*zoo
    assert B(1, oo)/oo == B(0, oo)
    assert B(1, oo)/(-oo) == B(-oo, 0)
    assert B(-oo, -1)/oo == B(-oo, 0)
    assert B(-oo, -1)/(-oo) == B(0, oo)
    assert B(-oo, oo)/oo == B(-oo, oo)
    assert B(-oo, oo)/(-oo) == B(-oo, oo)
    assert B(-1, oo)/oo == B(0, oo)
    assert B(-1, oo)/(-oo) == B(-oo, 0)
    assert B(-oo, 1)/oo == B(-oo, 0)
    assert B(-oo, 1)/(-oo) == B(0, oo)


def test_issue_18795():
    r = Symbol('r', real=True)
    a = B(-1,1)
    c = B(7, oo)
    b = B(-oo, oo)
    assert c - tan(r) == B(7-tan(r), oo)
    assert b + tan(r) == B(-oo, oo)
    assert (a + r)/a == B(-oo, oo)*B(r - 1, r + 1)
    assert (b + a)/a == B(-oo, oo)


def test_AccumBounds_func():
    assert (x**2 + 2*x + 1).subs(x, B(-1, 1)) == B(-1, 4)
    assert exp(B(0, 1)) == B(1, E)
    assert exp(B(-oo, oo)) == B(0, oo)
    assert log(B(3, 6)) == B(log(3), log(6))


@XFAIL
def test_AccumBounds_powf():
    nn = Symbol('nn', nonnegative=True)
    assert B(1 + nn, 2 + nn)**B(1, 2) == B(1 + nn, (2 + nn)**2)
    i = Symbol('i', integer=True, negative=True)
    assert B(1, 2)**i == B(2**i, 1)


def test_AccumBounds_pow():
    assert B(0, 2)**2 == B(0, 4)
    assert B(-1, 1)**2 == B(0, 1)
    assert B(1, 2)**2 == B(1, 4)
    assert B(-1, 2)**3 == B(-1, 8)
    assert B(-1, 1)**0 == 1

    assert B(1, 2)**Rational(5, 2) == B(1, 4*sqrt(2))
    assert B(0, 2)**S.Half == B(0, sqrt(2))

    neg = Symbol('neg', negative=True)
    assert unchanged(Pow, B(neg, 1), S.Half)
    nn = Symbol('nn', nonnegative=True)
    assert B(nn, nn + 1)**S.Half == B(sqrt(nn), sqrt(nn + 1))
    assert B(nn, nn + 1)**nn == B(nn**nn, (nn + 1)**nn)
    assert unchanged(Pow, B(nn, nn + 1), x)
    i = Symbol('i', integer=True)
    assert B(1, 2)**i == B(Min(1, 2**i), Max(1, 2**i))
    i = Symbol('i', integer=True, nonnegative=True)
    assert B(1, 2)**i == B(1, 2**i)
    assert B(0, 1)**i == B(0**i, 1)

    assert B(1, 5)**(-2) == B(Rational(1, 25), 1)
    assert B(-1, 3)**(-2) == B(0, oo)
    assert B(0, 2)**(-3) == B(Rational(1, 8), oo)
    assert B(-2, 0)**(-3) == B(-oo, -Rational(1, 8))
    assert B(0, 2)**(-2) == B(Rational(1, 4), oo)
    assert B(-1, 2)**(-3) == B(-oo, oo)
    assert B(-3, -2)**(-3) == B(Rational(-1, 8), Rational(-1, 27))
    assert B(-3, -2)**(-2) == B(Rational(1, 9), Rational(1, 4))
    assert B(0, oo)**S.Half == B(0, oo)
    assert B(-oo, 0)**(-2) == B(0, oo)
    assert B(-2, 0)**(-2) == B(Rational(1, 4), oo)

    assert B(Rational(1, 3), S.Half)**oo is S.Zero
    assert B(0, S.Half)**oo is S.Zero
    assert B(S.Half, 1)**oo == B(0, oo)
    assert B(0, 1)**oo == B(0, oo)
    assert B(2, 3)**oo is oo
    assert B(1, 2)**oo == B(0, oo)
    assert B(S.Half, 3)**oo == B(0, oo)
    assert B(Rational(-1, 3), Rational(-1, 4))**oo is S.Zero
    assert B(-1, Rational(-1, 2))**oo is S.NaN
    assert B(-3, -2)**oo is zoo
    assert B(-2, -1)**oo is S.NaN
    assert B(-2, Rational(-1, 2))**oo is S.NaN
    assert B(Rational(-1, 2), S.Half)**oo is S.Zero
    assert B(Rational(-1, 2), 1)**oo == B(0, oo)
    assert B(Rational(-2, 3), 2)**oo == B(0, oo)
    assert B(-1, 1)**oo == B(-oo, oo)
    assert B(-1, S.Half)**oo == B(-oo, oo)
    assert B(-1, 2)**oo == B(-oo, oo)
    assert B(-2, S.Half)**oo == B(-oo, oo)

    assert B(1, 2)**x == Pow(B(1, 2), x, evaluate=False)

    assert B(2, 3)**(-oo) is S.Zero
    assert B(0, 2)**(-oo) == B(0, oo)
    assert B(-1, 2)**(-oo) == B(-oo, oo)

    assert (tan(x)**sin(2*x)).subs(x, B(0, pi/2)) == \
        Pow(B(-oo, oo), B(0, 1))


def test_AccumBounds_exponent():
    # base is 0
    z = 0**B(a, a + S.Half)
    assert z.subs(a, 0) == B(0, 1)
    assert z.subs(a, 1) == 0
    p = z.subs(a, -1)
    assert p.is_Pow and p.args == (0, B(-1, -S.Half))
    # base > 0
    #   when base is 1 the type of bounds does not matter
    assert 1**B(a, a + 1) == 1
    #  otherwise we need to know if 0 is in the bounds
    assert S.Half**B(-2, 2) == B(S(1)/4, 4)
    assert 2**B(-2, 2) == B(S(1)/4, 4)

    # +eps may introduce +oo
    # if there is a negative integer exponent
    assert B(0, 1)**B(S(1)/2, 1) == B(0, 1)
    assert B(0, 1)**B(0, 1) == B(0, 1)

    # positive bases have positive bounds
    assert B(2, 3)**B(-3, -2) == B(S(1)/27, S(1)/4)
    assert B(2, 3)**B(-3, 2) == B(S(1)/27, 9)

    # bounds generating imaginary parts unevaluated
    assert unchanged(Pow, B(-1, 1), B(1, 2))
    assert B(0, S(1)/2)**B(1, oo) == B(0, S(1)/2)
    assert B(0, 1)**B(1, oo) == B(0, oo)
    assert B(0, 2)**B(1, oo) == B(0, oo)
    assert B(0, oo)**B(1, oo) == B(0, oo)
    assert B(S(1)/2, 1)**B(1, oo) == B(0, oo)
    assert B(S(1)/2, 1)**B(-oo, -1) == B(0, oo)
    assert B(S(1)/2, 1)**B(-oo, oo) == B(0, oo)
    assert B(S(1)/2, 2)**B(1, oo) == B(0, oo)
    assert B(S(1)/2, 2)**B(-oo, -1) == B(0, oo)
    assert B(S(1)/2, 2)**B(-oo, oo) == B(0, oo)
    assert B(S(1)/2, oo)**B(1, oo) == B(0, oo)
    assert B(S(1)/2, oo)**B(-oo, -1) == B(0, oo)
    assert B(S(1)/2, oo)**B(-oo, oo) == B(0, oo)
    assert B(1, 2)**B(1, oo) == B(0, oo)
    assert B(1, 2)**B(-oo, -1) == B(0, oo)
    assert B(1, 2)**B(-oo, oo) == B(0, oo)
    assert B(1, oo)**B(1, oo) == B(0, oo)
    assert B(1, oo)**B(-oo, -1) == B(0, oo)
    assert B(1, oo)**B(-oo, oo) == B(0, oo)
    assert B(2, oo)**B(1, oo) == B(2, oo)
    assert B(2, oo)**B(-oo, -1) == B(0, S(1)/2)
    assert B(2, oo)**B(-oo, oo) == B(0, oo)


def test_comparison_AccumBounds():
    assert (B(1, 3) < 4) == S.true
    assert (B(1, 3) < -1) == S.false
    assert (B(1, 3) < 2).rel_op == '<'
    assert (B(1, 3) <= 2).rel_op == '<='

    assert (B(1, 3) > 4) == S.false
    assert (B(1, 3) > -1) == S.true
    assert (B(1, 3) > 2).rel_op == '>'
    assert (B(1, 3) >= 2).rel_op == '>='

    assert (B(1, 3) < B(4, 6)) == S.true
    assert (B(1, 3) < B(2, 4)).rel_op == '<'
    assert (B(1, 3) < B(-2, 0)) == S.false

    assert (B(1, 3) <= B(4, 6)) == S.true
    assert (B(1, 3) <= B(-2, 0)) == S.false

    assert (B(1, 3) > B(4, 6)) == S.false
    assert (B(1, 3) > B(-2, 0)) == S.true

    assert (B(1, 3) >= B(4, 6)) == S.false
    assert (B(1, 3) >= B(-2, 0)) == S.true

    # issue 13499
    assert (cos(x) > 0).subs(x, oo) == (B(-1, 1) > 0)

    c = Symbol('c')
    raises(TypeError, lambda: (B(0, 1) < c))
    raises(TypeError, lambda: (B(0, 1) <= c))
    raises(TypeError, lambda: (B(0, 1) > c))
    raises(TypeError, lambda: (B(0, 1) >= c))


def test_contains_AccumBounds():
    assert (1 in B(1, 2)) == S.true
    raises(TypeError, lambda: a in B(1, 2))
    assert 0 in B(-1, 0)
    raises(TypeError, lambda:
        (cos(1)**2 + sin(1)**2 - 1) in B(-1, 0))
    assert (-oo in B(1, oo)) == S.true
    assert (oo in B(-oo, 0)) == S.true

    # issue 13159
    assert Mul(0, B(-1, 1)) == Mul(B(-1, 1), 0) == 0
    import itertools
    for perm in itertools.permutations([0, B(-1, 1), x]):
        assert Mul(*perm) == 0


def test_intersection_AccumBounds():
    assert B(0, 3).intersection(B(1, 2)) == B(1, 2)
    assert B(0, 3).intersection(B(1, 4)) == B(1, 3)
    assert B(0, 3).intersection(B(-1, 2)) == B(0, 2)
    assert B(0, 3).intersection(B(-1, 4)) == B(0, 3)
    assert B(0, 1).intersection(B(2, 3)) == S.EmptySet
    raises(TypeError, lambda: B(0, 3).intersection(1))


def test_union_AccumBounds():
    assert B(0, 3).union(B(1, 2)) == B(0, 3)
    assert B(0, 3).union(B(1, 4)) == B(0, 4)
    assert B(0, 3).union(B(-1, 2)) == B(-1, 3)
    assert B(0, 3).union(B(-1, 4)) == B(-1, 4)
    raises(TypeError, lambda: B(0, 3).union(1))
