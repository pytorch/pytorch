from sympy.core.function import Function
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cot, sin, tan)
from sympy.series.residues import residue
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import x, z, a, s, k


def test_basic1():
    assert residue(1/x, x, 0) == 1
    assert residue(-2/x, x, 0) == -2
    assert residue(81/x, x, 0) == 81
    assert residue(1/x**2, x, 0) == 0
    assert residue(0, x, 0) == 0
    assert residue(5, x, 0) == 0
    assert residue(x, x, 0) == 0
    assert residue(x**2, x, 0) == 0


def test_basic2():
    assert residue(1/x, x, 1) == 0
    assert residue(-2/x, x, 1) == 0
    assert residue(81/x, x, -1) == 0
    assert residue(1/x**2, x, 1) == 0
    assert residue(0, x, 1) == 0
    assert residue(5, x, 1) == 0
    assert residue(x, x, 1) == 0
    assert residue(x**2, x, 5) == 0


def test_f():
    f = Function("f")
    assert residue(f(x)/x**5, x, 0) == f(x).diff(x, 4).subs(x, 0)/24


def test_functions():
    assert residue(1/sin(x), x, 0) == 1
    assert residue(2/sin(x), x, 0) == 2
    assert residue(1/sin(x)**2, x, 0) == 0
    assert residue(1/sin(x)**5, x, 0) == Rational(3, 8)


def test_expressions():
    assert residue(1/(x + 1), x, 0) == 0
    assert residue(1/(x + 1), x, -1) == 1
    assert residue(1/(x**2 + 1), x, -1) == 0
    assert residue(1/(x**2 + 1), x, I) == -I/2
    assert residue(1/(x**2 + 1), x, -I) == I/2
    assert residue(1/(x**4 + 1), x, 0) == 0
    assert residue(1/(x**4 + 1), x, exp(I*pi/4)).equals(-(Rational(1, 4) + I/4)/sqrt(2))
    assert residue(1/(x**2 + a**2)**2, x, a*I) == -I/4/a**3


@XFAIL
def test_expressions_failing():
    n = Symbol('n', integer=True, positive=True)
    assert residue(exp(z)/(z - pi*I/4*a)**n, z, I*pi*a) == \
        exp(I*pi*a/4)/factorial(n - 1)


def test_NotImplemented():
    raises(NotImplementedError, lambda: residue(exp(1/z), z, 0))


def test_bug():
    assert residue(2**(z)*(s + z)*(1 - s - z)/z**2, z, 0) == \
        1 + s*log(2) - s**2*log(2) - 2*s


def test_issue_5654():
    assert residue(1/(x**2 + a**2)**2, x, a*I) == -I/(4*a**3)
    assert residue(1/s*1/(z - exp(s)), s, 0) == 1/(z - 1)
    assert residue((1 + k)/s*1/(z - exp(s)), s, 0) == k/(z - 1) + 1/(z - 1)


def test_issue_6499():
    assert residue(1/(exp(z) - 1), z, 0) == 1


def test_issue_14037():
    assert residue(sin(x**50)/x**51, x, 0) == 1


def test_issue_21176():
    f = x**2*cot(pi*x)/(x**4 + 1)
    assert residue(f, x, -sqrt(2)/2 - sqrt(2)*I/2).cancel().together(deep=True)\
        == sqrt(2)*(1 - I)/(8*tan(sqrt(2)*pi*(1 + I)/2))


def test_issue_21177():
    r = -sqrt(3)*tanh(sqrt(3)*pi/2)/3
    a = residue(cot(pi*x)/((x - 1)*(x - 2) + 1), x, S(3)/2 - sqrt(3)*I/2)
    b = residue(cot(pi*x)/(x**2 - 3*x + 3), x, S(3)/2 - sqrt(3)*I/2)
    assert a == r
    assert (b - a).cancel() == 0
