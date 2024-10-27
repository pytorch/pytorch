from sympy.series import approximants
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.combinatorial.numbers import (fibonacci, lucas)


def test_approximants():
    x, t = symbols("x,t")
    g = [lucas(k) for k in range(16)]
    assert list(approximants(g)) == (
        [2, -4/(x - 2), (5*x - 2)/(3*x - 1), (x - 2)/(x**2 + x - 1)] )
    g = [lucas(k)+fibonacci(k+2) for k in range(16)]
    assert list(approximants(g)) == (
        [3, -3/(x - 1), (3*x - 3)/(2*x - 1), -3/(x**2 + x - 1)] )
    g = [lucas(k)**2 for k in range(16)]
    assert list(approximants(g)) == (
        [4, -16/(x - 4), (35*x - 4)/(9*x - 1), (37*x - 28)/(13*x**2 + 11*x - 7),
        (50*x**2 + 63*x - 52)/(37*x**2 + 19*x - 13),
        (-x**2 - 7*x + 4)/(x**3 - 2*x**2 - 2*x + 1)] )
    p = [sum(binomial(k,i)*x**i for i in range(k+1)) for k in range(16)]
    y = approximants(p, t, simplify=True)
    assert next(y) == 1
    assert next(y) == -1/(t*(x + 1) - 1)
