from sympy.solvers.decompogen import decompogen, compogen
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.testing.pytest import XFAIL, raises

x, y = symbols('x y')


def test_decompogen():
    assert decompogen(sin(cos(x)), x) == [sin(x), cos(x)]
    assert decompogen(sin(x)**2 + sin(x) + 1, x) == [x**2 + x + 1, sin(x)]
    assert decompogen(sqrt(6*x**2 - 5), x) == [sqrt(x), 6*x**2 - 5]
    assert decompogen(sin(sqrt(cos(x**2 + 1))), x) == [sin(x), sqrt(x), cos(x), x**2 + 1]
    assert decompogen(Abs(cos(x)**2 + 3*cos(x) - 4), x) == [Abs(x), x**2 + 3*x - 4, cos(x)]
    assert decompogen(sin(x)**2 + sin(x) - sqrt(3)/2, x) == [x**2 + x - sqrt(3)/2, sin(x)]
    assert decompogen(Abs(cos(y)**2 + 3*cos(x) - 4), x) == [Abs(x), 3*x + cos(y)**2 - 4, cos(x)]
    assert decompogen(x, y) == [x]
    assert decompogen(1, x) == [1]
    assert decompogen(Max(3, x), x) == [Max(3, x)]
    raises(TypeError, lambda: decompogen(x < 5, x))
    u = 2*x + 3
    assert decompogen(Max(sqrt(u),(u)**2), x) == [Max(sqrt(x), x**2), u]
    assert decompogen(Max(u, u**2, y), x) == [Max(x, x**2, y), u]
    assert decompogen(Max(sin(x), u), x) == [Max(2*x + 3, sin(x))]


def test_decompogen_poly():
    assert decompogen(x**4 + 2*x**2 + 1, x) == [x**2 + 2*x + 1, x**2]
    assert decompogen(x**4 + 2*x**3 - x - 1, x) == [x**2 - x - 1, x**2 + x]


@XFAIL
def test_decompogen_fails():
    A = lambda x: x**2 + 2*x + 3
    B = lambda x: 4*x**2 + 5*x + 6
    assert decompogen(A(x*exp(x)), x) == [x**2 + 2*x + 3, x*exp(x)]
    assert decompogen(A(B(x)), x) == [x**2 + 2*x + 3, 4*x**2 + 5*x + 6]
    assert decompogen(A(1/x + 1/x**2), x) == [x**2 + 2*x + 3, 1/x + 1/x**2]
    assert decompogen(A(1/x + 2/(x + 1)), x) == [x**2 + 2*x + 3, 1/x + 2/(x + 1)]


def test_compogen():
    assert compogen([sin(x), cos(x)], x) == sin(cos(x))
    assert compogen([x**2 + x + 1, sin(x)], x) == sin(x)**2 + sin(x) + 1
    assert compogen([sqrt(x), 6*x**2 - 5], x) == sqrt(6*x**2 - 5)
    assert compogen([sin(x), sqrt(x), cos(x), x**2 + 1], x) == sin(sqrt(
                                                                cos(x**2 + 1)))
    assert compogen([Abs(x), x**2 + 3*x - 4, cos(x)], x) == Abs(cos(x)**2 +
                                                                3*cos(x) - 4)
    assert compogen([x**2 + x - sqrt(3)/2, sin(x)], x) == (sin(x)**2 + sin(x) -
                                                           sqrt(3)/2)
    assert compogen([Abs(x), 3*x + cos(y)**2 - 4, cos(x)], x) == \
        Abs(3*cos(x) + cos(y)**2 - 4)
    assert compogen([x**2 + 2*x + 1, x**2], x) == x**4 + 2*x**2 + 1
    # the result is in unsimplified form
    assert compogen([x**2 - x - 1, x**2 + x], x) == -x**2 - x + (x**2 + x)**2 - 1
