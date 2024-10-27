from sympy.core.numbers import E
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.geometry.curve import Curve
from sympy.integrals.integrals import line_integrate

s, t, x, y, z = symbols('s,t,x,y,z')


def test_lineintegral():
    c = Curve([E**t + 1, E**t - 1], (t, 0, log(2)))
    assert line_integrate(x + y, c, [x, y]) == 3*sqrt(2)
