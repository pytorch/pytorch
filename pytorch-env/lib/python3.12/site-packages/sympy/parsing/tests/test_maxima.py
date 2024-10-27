from sympy.parsing.maxima import parse_maxima
from sympy.core.numbers import (E, Rational, oo)
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.abc import x

n = Symbol('n', integer=True)


def test_parser():
    assert Abs(parse_maxima('float(1/3)') - 0.333333333) < 10**(-5)
    assert parse_maxima('13^26') == 91733330193268616658399616009
    assert parse_maxima('sin(%pi/2) + cos(%pi/3)') == Rational(3, 2)
    assert parse_maxima('log(%e)') == 1


def test_injection():
    parse_maxima('c: x+1', globals=globals())
    # c created by parse_maxima
    assert c == x + 1 # noqa:F821

    parse_maxima('g: sqrt(81)', globals=globals())
    # g created by parse_maxima
    assert g == 9 # noqa:F821


def test_maxima_functions():
    assert parse_maxima('expand( (x+1)^2)') == x**2 + 2*x + 1
    assert parse_maxima('factor( x**2 + 2*x + 1)') == (x + 1)**2
    assert parse_maxima('2*cos(x)^2 + sin(x)^2') == 2*cos(x)**2 + sin(x)**2
    assert parse_maxima('trigexpand(sin(2*x)+cos(2*x))') == \
        -1 + 2*cos(x)**2 + 2*cos(x)*sin(x)
    assert parse_maxima('solve(x^2-4,x)') == [-2, 2]
    assert parse_maxima('limit((1+1/x)^x,x,inf)') == E
    assert parse_maxima('limit(sqrt(-x)/x,x,0,minus)') is -oo
    assert parse_maxima('diff(x^x, x)') == x**x*(1 + log(x))
    assert parse_maxima('sum(k, k, 1, n)', name_dict={
        "n": Symbol('n', integer=True),
        "k": Symbol('k', integer=True)
    }) == (n**2 + n)/2
    assert parse_maxima('product(k, k, 1, n)', name_dict={
        "n": Symbol('n', integer=True),
        "k": Symbol('k', integer=True)
    }) == factorial(n)
    assert parse_maxima('ratsimp((x^2-1)/(x+1))') == x - 1
    assert Abs( parse_maxima(
        'float(sec(%pi/3) + csc(%pi/3))') - 3.154700538379252) < 10**(-5)
