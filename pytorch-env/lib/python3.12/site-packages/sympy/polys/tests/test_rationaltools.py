"""Tests for tools for manipulation of rational expressions. """

from sympy.polys.rationaltools import together

from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import Integral
from sympy.abc import x, y, z

A, B = symbols('A,B', commutative=False)


def test_together():
    assert together(0) == 0
    assert together(1) == 1

    assert together(x*y*z) == x*y*z
    assert together(x + y) == x + y

    assert together(1/x) == 1/x

    assert together(1/x + 1) == (x + 1)/x
    assert together(1/x + 3) == (3*x + 1)/x
    assert together(1/x + x) == (x**2 + 1)/x

    assert together(1/x + S.Half) == (x + 2)/(2*x)
    assert together(S.Half + x/2) == Mul(S.Half, x + 1, evaluate=False)

    assert together(1/x + 2/y) == (2*x + y)/(y*x)
    assert together(1/(1 + 1/x)) == x/(1 + x)
    assert together(x/(1 + 1/x)) == x**2/(1 + x)

    assert together(1/x + 1/y + 1/z) == (x*y + x*z + y*z)/(x*y*z)
    assert together(1/(1 + x + 1/y + 1/z)) == y*z/(y + z + y*z + x*y*z)

    assert together(1/(x*y) + 1/(x*y)**2) == y**(-2)*x**(-2)*(1 + x*y)
    assert together(1/(x*y) + 1/(x*y)**4) == y**(-4)*x**(-4)*(1 + x**3*y**3)
    assert together(1/(x**7*y) + 1/(x*y)**4) == y**(-4)*x**(-7)*(x**3 + y**3)

    assert together(5/(2 + 6/(3 + 7/(4 + 8/(5 + 9/x))))) == \
        Rational(5, 2)*((171 + 119*x)/(279 + 203*x))

    assert together(1 + 1/(x + 1)**2) == (1 + (x + 1)**2)/(x + 1)**2
    assert together(1 + 1/(x*(1 + x))) == (1 + x*(1 + x))/(x*(1 + x))
    assert together(
        1/(x*(x + 1)) + 1/(x*(x + 2))) == (3 + 2*x)/(x*(1 + x)*(2 + x))
    assert together(1 + 1/(2*x + 2)**2) == (4*(x + 1)**2 + 1)/(4*(x + 1)**2)

    assert together(sin(1/x + 1/y)) == sin(1/x + 1/y)
    assert together(sin(1/x + 1/y), deep=True) == sin((x + y)/(x*y))

    assert together(1/exp(x) + 1/(x*exp(x))) == (1 + x)/(x*exp(x))
    assert together(1/exp(2*x) + 1/(x*exp(3*x))) == (1 + exp(x)*x)/(x*exp(3*x))

    assert together(Integral(1/x + 1/y, x)) == Integral((x + y)/(x*y), x)
    assert together(Eq(1/x + 1/y, 1 + 1/z)) == Eq((x + y)/(x*y), (z + 1)/z)

    assert together((A*B)**-1 + (B*A)**-1) == (A*B)**-1 + (B*A)**-1
