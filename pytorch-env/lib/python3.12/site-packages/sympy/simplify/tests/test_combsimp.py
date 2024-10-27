from sympy.core.numbers import Rational
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import (FallingFactorial, RisingFactorial, binomial, factorial)
from sympy.functions.special.gamma_functions import gamma
from sympy.simplify.combsimp import combsimp
from sympy.abc import x


def test_combsimp():
    k, m, n = symbols('k m n', integer = True)

    assert combsimp(factorial(n)) == factorial(n)
    assert combsimp(binomial(n, k)) == binomial(n, k)

    assert combsimp(factorial(n)/factorial(n - 3)) == n*(-1 + n)*(-2 + n)
    assert combsimp(binomial(n + 1, k + 1)/binomial(n, k)) == (1 + n)/(1 + k)

    assert combsimp(binomial(3*n + 4, n + 1)/binomial(3*n + 1, n)) == \
        Rational(3, 2)*((3*n + 2)*(3*n + 4)/((n + 1)*(2*n + 3)))

    assert combsimp(factorial(n)**2/factorial(n - 3)) == \
        factorial(n)*n*(-1 + n)*(-2 + n)
    assert combsimp(factorial(n)*binomial(n + 1, k + 1)/binomial(n, k)) == \
        factorial(n + 1)/(1 + k)

    assert combsimp(gamma(n + 3)) == factorial(n + 2)

    assert combsimp(factorial(x)) == gamma(x + 1)

    # issue 9699
    assert combsimp((n + 1)*factorial(n)) == factorial(n + 1)
    assert combsimp(factorial(n)/n) == factorial(n-1)

    # issue 6658
    assert combsimp(binomial(n, n - k)) == binomial(n, k)

    # issue 6341, 7135
    assert combsimp(factorial(n)/(factorial(k)*factorial(n - k))) == \
        binomial(n, k)
    assert combsimp(factorial(k)*factorial(n - k)/factorial(n)) == \
        1/binomial(n, k)
    assert combsimp(factorial(2*n)/factorial(n)**2) == binomial(2*n, n)
    assert combsimp(factorial(2*n)*factorial(k)*factorial(n - k)/
        factorial(n)**3) == binomial(2*n, n)/binomial(n, k)

    assert combsimp(factorial(n*(1 + n) - n**2 - n)) == 1

    assert combsimp(6*FallingFactorial(-4, n)/factorial(n)) == \
        (-1)**n*(n + 1)*(n + 2)*(n + 3)
    assert combsimp(6*FallingFactorial(-4, n - 1)/factorial(n - 1)) == \
        (-1)**(n - 1)*n*(n + 1)*(n + 2)
    assert combsimp(6*FallingFactorial(-4, n - 3)/factorial(n - 3)) == \
        (-1)**(n - 3)*n*(n - 1)*(n - 2)
    assert combsimp(6*FallingFactorial(-4, -n - 1)/factorial(-n - 1)) == \
        -(-1)**(-n - 1)*n*(n - 1)*(n - 2)

    assert combsimp(6*RisingFactorial(4, n)/factorial(n)) == \
        (n + 1)*(n + 2)*(n + 3)
    assert combsimp(6*RisingFactorial(4, n - 1)/factorial(n - 1)) == \
        n*(n + 1)*(n + 2)
    assert combsimp(6*RisingFactorial(4, n - 3)/factorial(n - 3)) == \
        n*(n - 1)*(n - 2)
    assert combsimp(6*RisingFactorial(4, -n - 1)/factorial(-n - 1)) == \
        -n*(n - 1)*(n - 2)


def test_issue_6878():
    n = symbols('n', integer=True)
    assert combsimp(RisingFactorial(-10, n)) == 3628800*(-1)**n/factorial(10 - n)


def test_issue_14528():
    p = symbols("p", integer=True, positive=True)
    assert combsimp(binomial(1,p)) == 1/(factorial(p)*factorial(1-p))
    assert combsimp(factorial(2-p)) == factorial(2-p)
