from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import (binomial, factorial, subfactorial)
from sympy.functions.combinatorial.numbers import (fibonacci, harmonic)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.series.limitseq import limit_seq
from sympy.series.limitseq import difference_delta as dd
from sympy.testing.pytest import raises, XFAIL
from sympy.calculus.accumulationbounds import AccumulationBounds

n, m, k = symbols('n m k', integer=True)


def test_difference_delta():
    e = n*(n + 1)
    e2 = e * k

    assert dd(e) == 2*n + 2
    assert dd(e2, n, 2) == k*(4*n + 6)

    raises(ValueError, lambda: dd(e2))
    raises(ValueError, lambda: dd(e2, n, oo))


def test_difference_delta__Sum():
    e = Sum(1/k, (k, 1, n))
    assert dd(e, n) == 1/(n + 1)
    assert dd(e, n, 5) == Add(*[1/(i + n + 1) for i in range(5)])

    e = Sum(1/k, (k, 1, 3*n))
    assert dd(e, n) == Add(*[1/(i + 3*n + 1) for i in range(3)])

    e = n * Sum(1/k, (k, 1, n))
    assert dd(e, n) == 1 + Sum(1/k, (k, 1, n))

    e = Sum(1/k, (k, 1, n), (m, 1, n))
    assert dd(e, n) == harmonic(n)


def test_difference_delta__Add():
    e = n + n*(n + 1)
    assert dd(e, n) == 2*n + 3
    assert dd(e, n, 2) == 4*n + 8

    e = n + Sum(1/k, (k, 1, n))
    assert dd(e, n) == 1 + 1/(n + 1)
    assert dd(e, n, 5) == 5 + Add(*[1/(i + n + 1) for i in range(5)])


def test_difference_delta__Pow():
    e = 4**n
    assert dd(e, n) == 3*4**n
    assert dd(e, n, 2) == 15*4**n

    e = 4**(2*n)
    assert dd(e, n) == 15*4**(2*n)
    assert dd(e, n, 2) == 255*4**(2*n)

    e = n**4
    assert dd(e, n) == (n + 1)**4 - n**4

    e = n**n
    assert dd(e, n) == (n + 1)**(n + 1) - n**n


def test_limit_seq():
    e = binomial(2*n, n) / Sum(binomial(2*k, k), (k, 1, n))
    assert limit_seq(e) == S(3) / 4
    assert limit_seq(e, m) == e

    e = (5*n**3 + 3*n**2 + 4) / (3*n**3 + 4*n - 5)
    assert limit_seq(e, n) == S(5) / 3

    e = (harmonic(n) * Sum(harmonic(k), (k, 1, n))) / (n * harmonic(2*n)**2)
    assert limit_seq(e, n) == 1

    e = Sum(k**2 * Sum(2**m/m, (m, 1, k)), (k, 1, n)) / (2**n*n)
    assert limit_seq(e, n) == 4

    e = (Sum(binomial(3*k, k) * binomial(5*k, k), (k, 1, n)) /
         (binomial(3*n, n) * binomial(5*n, n)))
    assert limit_seq(e, n) == S(84375) / 83351

    e = Sum(harmonic(k)**2/k, (k, 1, 2*n)) / harmonic(n)**3
    assert limit_seq(e, n) == S.One / 3

    raises(ValueError, lambda: limit_seq(e * m))


def test_alternating_sign():
    assert limit_seq((-1)**n/n**2, n) == 0
    assert limit_seq((-2)**(n+1)/(n + 3**n), n) == 0
    assert limit_seq((2*n + (-1)**n)/(n + 1), n) == 2
    assert limit_seq(sin(pi*n), n) == 0
    assert limit_seq(cos(2*pi*n), n) == 1
    assert limit_seq((S.NegativeOne/5)**n, n) == 0
    assert limit_seq((Rational(-1, 5))**n, n) == 0
    assert limit_seq((I/3)**n, n) == 0
    assert limit_seq(sqrt(n)*(I/2)**n, n) == 0
    assert limit_seq(n**7*(I/3)**n, n) == 0
    assert limit_seq(n/(n + 1) + (I/2)**n, n) == 1


def test_accum_bounds():
    assert limit_seq((-1)**n, n) == AccumulationBounds(-1, 1)
    assert limit_seq(cos(pi*n), n) == AccumulationBounds(-1, 1)
    assert limit_seq(sin(pi*n/2)**2, n) == AccumulationBounds(0, 1)
    assert limit_seq(2*(-3)**n/(n + 3**n), n) == AccumulationBounds(-2, 2)
    assert limit_seq(3*n/(n + 1) + 2*(-1)**n, n) == AccumulationBounds(1, 5)


def test_limitseq_sum():
    from sympy.abc import x, y, z
    assert limit_seq(Sum(1/x, (x, 1, y)) - log(y), y) == S.EulerGamma
    assert limit_seq(Sum(1/x, (x, 1, y)) - 1/y, y) is S.Infinity
    assert (limit_seq(binomial(2*x, x) / Sum(binomial(2*y, y), (y, 1, x)), x) ==
            S(3) / 4)
    assert (limit_seq(Sum(y**2 * Sum(2**z/z, (z, 1, y)), (y, 1, x)) /
                  (2**x*x), x) == 4)


def test_issue_9308():
    assert limit_seq(subfactorial(n)/factorial(n), n) == exp(-1)


def test_issue_10382():
    n = Symbol('n', integer=True)
    assert limit_seq(fibonacci(n+1)/fibonacci(n), n).together() == S.GoldenRatio


def test_issue_11672():
    assert limit_seq(Rational(-1, 2)**n, n) == 0


def test_issue_14196():
    k, n  = symbols('k, n', positive=True)
    m = Symbol('m')
    assert limit_seq(Sum(m**k, (m, 1, n)).doit()/(n**(k + 1)), n) == 1/(k + 1)


def test_issue_16735():
    assert limit_seq(5**n/factorial(n), n) == 0


def test_issue_19868():
    assert limit_seq(1/gamma(n + S.One/2), n) == 0


@XFAIL
def test_limit_seq_fail():
    # improve Summation algorithm or add ad-hoc criteria
    e = (harmonic(n)**3 * Sum(1/harmonic(k), (k, 1, n)) /
         (n * Sum(harmonic(k)/k, (k, 1, n))))
    assert limit_seq(e, n) == 2

    # No unique dominant term
    e = (Sum(2**k * binomial(2*k, k) / k**2, (k, 1, n)) /
         (Sum(2**k/k*2, (k, 1, n)) * Sum(binomial(2*k, k), (k, 1, n))))
    assert limit_seq(e, n) == S(3) / 7

    # Simplifications of summations needs to be improved.
    e = n**3*Sum(2**k/k**2, (k, 1, n))**2 / (2**n * Sum(2**k/k, (k, 1, n)))
    assert limit_seq(e, n) == 2

    e = (harmonic(n) * Sum(2**k/k, (k, 1, n)) /
         (n * Sum(2**k*harmonic(k)/k**2, (k, 1, n))))
    assert limit_seq(e, n) == 1

    e = (Sum(2**k*factorial(k) / k**2, (k, 1, 2*n)) /
         (Sum(4**k/k**2, (k, 1, n)) * Sum(factorial(k), (k, 1, 2*n))))
    assert limit_seq(e, n) == S(3) / 16
