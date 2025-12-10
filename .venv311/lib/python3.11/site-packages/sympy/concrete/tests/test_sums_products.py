from math import prod

from sympy.concrete.expr_with_intlimits import ReorderError
from sympy.concrete.products import (Product, product)
from sympy.concrete.summations import (Sum, summation, telescopic,
     eval_sum_residue, _dummy_with_inherited_properties_concrete)
from sympy.core.function import (Derivative, Function)
from sympy.core import (Catalan, EulerGamma)
from sympy.core.facts import InconsistentAssumptions
from sympy.core.mod import Mod
from sympy.core.numbers import (E, I, Rational, nan, oo, pi)
from sympy.core.relational import Eq, Ne
from sympy.core.numbers import Float
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (rf, binomial, factorial)
from sympy.functions.combinatorial.numbers import harmonic
from sympy.functions.elementary.complexes import Abs, re
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (sinh, tanh)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin, atan)
from sympy.functions.special.gamma_functions import (gamma, lowergamma)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.functions.special.zeta_functions import zeta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And, Or
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.matrices import (Matrix, SparseMatrix,
    ImmutableDenseMatrix, ImmutableSparseMatrix, diag)
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import Interval
from sympy.simplify.combsimp import combsimp
from sympy.simplify.simplify import simplify
from sympy.tensor.indexed import (Idx, Indexed, IndexedBase)
from sympy.testing.pytest import XFAIL, raises, slow
from sympy.abc import a, b, c, d, k, m, x, y, z

n = Symbol('n', integer=True)
f, g = symbols('f g', cls=Function)

def test_karr_convention():
    # Test the Karr summation convention that we want to hold.
    # See his paper "Summation in Finite Terms" for a detailed
    # reasoning why we really want exactly this definition.
    # The convention is described on page 309 and essentially
    # in section 1.4, definition 3:
    #
    # \sum_{m <= i < n} f(i) 'has the obvious meaning'   for m < n
    # \sum_{m <= i < n} f(i) = 0                         for m = n
    # \sum_{m <= i < n} f(i) = - \sum_{n <= i < m} f(i)  for m > n
    #
    # It is important to note that he defines all sums with
    # the upper limit being *exclusive*.
    # In contrast, SymPy and the usual mathematical notation has:
    #
    # sum_{i = a}^b f(i) = f(a) + f(a+1) + ... + f(b-1) + f(b)
    #
    # with the upper limit *inclusive*. So translating between
    # the two we find that:
    #
    # \sum_{m <= i < n} f(i) = \sum_{i = m}^{n-1} f(i)
    #
    # where we intentionally used two different ways to typeset the
    # sum and its limits.

    i = Symbol("i", integer=True)
    k = Symbol("k", integer=True)
    j = Symbol("j", integer=True)

    # A simple example with a concrete summand and symbolic limits.

    # The normal sum: m = k and n = k + j and therefore m < n:
    m = k
    n = k + j

    a = m
    b = n - 1
    S1 = Sum(i**2, (i, a, b)).doit()

    # The reversed sum: m = k + j and n = k and therefore m > n:
    m = k + j
    n = k

    a = m
    b = n - 1
    S2 = Sum(i**2, (i, a, b)).doit()

    assert simplify(S1 + S2) == 0

    # Test the empty sum: m = k and n = k and therefore m = n:
    m = k
    n = k

    a = m
    b = n - 1
    Sz = Sum(i**2, (i, a, b)).doit()

    assert Sz == 0

    # Another example this time with an unspecified summand and
    # numeric limits. (We can not do both tests in the same example.)

    # The normal sum with m < n:
    m = 2
    n = 11

    a = m
    b = n - 1
    S1 = Sum(f(i), (i, a, b)).doit()

    # The reversed sum with m > n:
    m = 11
    n = 2

    a = m
    b = n - 1
    S2 = Sum(f(i), (i, a, b)).doit()

    assert simplify(S1 + S2) == 0

    # Test the empty sum with m = n:
    m = 5
    n = 5

    a = m
    b = n - 1
    Sz = Sum(f(i), (i, a, b)).doit()

    assert Sz == 0

    e = Piecewise((exp(-i), Mod(i, 2) > 0), (0, True))
    s = Sum(e, (i, 0, 11))
    assert s.n(3) == s.doit().n(3)

    # issue #27893
    n = Symbol('n', integer=True)
    assert Sum(1/(x**2 + 1), (x, oo, 0)).doit(deep=False) == Rational(-1, 2) + pi / (2 * tanh(pi))
    assert Sum(c**x/factorial(x), (x, oo, 0)).doit(deep=False).simplify() == exp(c) - 1 # exponential series
    assert Sum((-1)**x/x, (x, oo,0)).doit() == -log(2) # alternating harmnic series
    assert Sum((1/2)**x,(x, oo, -1)).doit() == S(2) # geometric series
    assert Sum(1/x, (x, oo, 0)).doit() == oo # harmonic series, divergent
    assert Sum((-1)**x/(2*x+1), (x, oo, -1)).doit() == pi/4 # leibniz series
    assert Sum((((-1)**x) * c**(2*x+1)) / factorial(2*x+1), (x, oo, -1)).doit() == sin(c) # sinusoidal series
    assert Sum((((-1)**x) * c**(2*x+1)) / (2*x+1), (x, 0, oo)).doit() \
        == Piecewise((atan(c), Ne(c**2, -1) & (Abs(c**2) <= 1)), \
                     (Sum((-1)**x*c**(2*x + 1)/(2*x + 1), (x, 0, oo)), True)) # arctangent series
    assert Sum(binomial(n, x) * c**x, (x, 0, oo)).doit() \
        == Piecewise(((c + 1)**n, \
                     ((n <= -1) & (Abs(c) < 1)) \
                        | ((n > 0) & (Abs(c) <= 1)) \
                        | ((n <= 0) & (n > -1) & Ne(c, -1) & (Abs(c) <= 1))), \
                     (Sum(c**x*binomial(n, x), (x, 0, oo)), True)) # binomial series
    assert Sum(1/x**n, (x, oo, 0)).doit() \
        == Piecewise((zeta(n), n > 1), (Sum(x**(-n), (x, oo, 0)), True)) # Euler's zeta function

def test_karr_proposition_2a():
    # Test Karr, page 309, proposition 2, part a
    i = Symbol("i", integer=True)
    u = Symbol("u", integer=True)
    v = Symbol("v", integer=True)

    def test_the_sum(m, n):
        # g
        g = i**3 + 2*i**2 - 3*i
        # f = Delta g
        f = simplify(g.subs(i, i+1) - g)
        # The sum
        a = m
        b = n - 1
        S = Sum(f, (i, a, b)).doit()
        # Test if Sum_{m <= i < n} f(i) = g(n) - g(m)
        assert simplify(S - (g.subs(i, n) - g.subs(i, m))) == 0

    # m < n
    test_the_sum(u,   u+v)
    # m = n
    test_the_sum(u,   u  )
    # m > n
    test_the_sum(u+v, u  )


def test_karr_proposition_2b():
    # Test Karr, page 309, proposition 2, part b
    i = Symbol("i", integer=True)
    u = Symbol("u", integer=True)
    v = Symbol("v", integer=True)
    w = Symbol("w", integer=True)

    def test_the_sum(l, n, m):
        # Summand
        s = i**3
        # First sum
        a = l
        b = n - 1
        S1 = Sum(s, (i, a, b)).doit()
        # Second sum
        a = l
        b = m - 1
        S2 = Sum(s, (i, a, b)).doit()
        # Third sum
        a = m
        b = n - 1
        S3 = Sum(s, (i, a, b)).doit()
        # Test if S1 = S2 + S3 as required
        assert S1 - (S2 + S3) == 0

    # l < m < n
    test_the_sum(u,     u+v,   u+v+w)
    # l < m = n
    test_the_sum(u,     u+v,   u+v  )
    # l < m > n
    test_the_sum(u,     u+v+w, v    )
    # l = m < n
    test_the_sum(u,     u,     u+v  )
    # l = m = n
    test_the_sum(u,     u,     u    )
    # l = m > n
    test_the_sum(u+v,   u+v,   u    )
    # l > m < n
    test_the_sum(u+v,   u,     u+w  )
    # l > m = n
    test_the_sum(u+v,   u,     u    )
    # l > m > n
    test_the_sum(u+v+w, u+v,   u    )


def test_arithmetic_sums():
    assert summation(1, (n, a, b)) == b - a + 1
    assert Sum(S.NaN, (n, a, b)) is S.NaN
    assert Sum(x, (n, a, a)).doit() == x
    assert Sum(x, (x, a, a)).doit() == a
    assert Sum(x, (n, 1, a)).doit() == a*x
    assert Sum(x, (x, Range(1, 11))).doit() == 55
    assert Sum(x, (x, Range(1, 11, 2))).doit() == 25
    assert Sum(x, (x, Range(1, 10, 2))) == Sum(x, (x, Range(9, 0, -2)))
    lo, hi = 1, 2
    s1 = Sum(n, (n, lo, hi))
    s2 = Sum(n, (n, hi, lo))
    assert s1 != s2
    assert s1.doit() == 3 and s2.doit() == 0
    lo, hi = x, x + 1
    s1 = Sum(n, (n, lo, hi))
    s2 = Sum(n, (n, hi, lo))
    assert s1 != s2
    assert s1.doit() == 2*x + 1 and s2.doit() == 0
    assert Sum(Integral(x, (x, 1, y)) + x, (x, 1, 2)).doit() == \
        y**2 + 2
    assert summation(1, (n, 1, 10)) == 10
    assert summation(2*n, (n, 0, 10**10)) == 100000000010000000000
    assert summation(4*n*m, (n, a, 1), (m, 1, d)).expand() == \
        2*d + 2*d**2 + a*d + a*d**2 - d*a**2 - a**2*d**2
    assert summation(cos(n), (n, -2, 1)) == cos(-2) + cos(-1) + cos(0) + cos(1)
    assert summation(cos(n), (n, x, x + 2)) == cos(x) + cos(x + 1) + cos(x + 2)
    assert isinstance(summation(cos(n), (n, x, x + S.Half)), Sum)
    assert summation(k, (k, 0, oo)) is oo
    assert summation(k, (k, Range(1, 11))) == 55


def test_polynomial_sums():
    assert summation(n**2, (n, 3, 8)) == 199
    assert summation(n, (n, a, b)) == \
        ((a + b)*(b - a + 1)/2).expand()
    assert summation(n**2, (n, 1, b)) == \
        ((2*b**3 + 3*b**2 + b)/6).expand()
    assert summation(n**3, (n, 1, b)) == \
        ((b**4 + 2*b**3 + b**2)/4).expand()
    assert summation(n**6, (n, 1, b)) == \
        ((6*b**7 + 21*b**6 + 21*b**5 - 7*b**3 + b)/42).expand()


def test_geometric_sums():
    assert summation(pi**n, (n, 0, b)) == (1 - pi**(b + 1)) / (1 - pi)
    assert summation(2 * 3**n, (n, 0, b)) == 3**(b + 1) - 1
    assert summation(S.Half**n, (n, 1, oo)) == 1
    assert summation(2**n, (n, 0, b)) == 2**(b + 1) - 1
    assert summation(2**n, (n, 1, oo)) is oo
    assert summation(2**(-n), (n, 1, oo)) == 1
    assert summation(3**(-n), (n, 4, oo)) == Rational(1, 54)
    assert summation(2**(-4*n + 3), (n, 1, oo)) == Rational(8, 15)
    assert summation(2**(n + 1), (n, 1, b)).expand() == 4*(2**b - 1)

    # issue 6664:
    assert summation(x**n, (n, 0, oo)) == \
        Piecewise((1/(-x + 1), Abs(x) < 1), (Sum(x**n, (n, 0, oo)), True))

    assert summation(-2**n, (n, 0, oo)) is -oo
    assert summation(I**n, (n, 0, oo)) == Sum(I**n, (n, 0, oo))

    # issue 6802:
    assert summation((-1)**(2*x + 2), (x, 0, n)) == n + 1
    assert summation((-2)**(2*x + 2), (x, 0, n)) == 4*4**(n + 1)/S(3) - Rational(4, 3)
    assert summation((-1)**x, (x, 0, n)) == -(-1)**(n + 1)/S(2) + S.Half
    assert summation(y**x, (x, a, b)) == \
        Piecewise((-a + b + 1, Eq(y, 1)), ((y**a - y**(b + 1))/(-y + 1), True))
    assert summation((-2)**(y*x + 2), (x, 0, n)) == \
        4*Piecewise((n + 1, Eq((-2)**y, 1)),
                    ((-(-2)**(y*(n + 1)) + 1)/(-(-2)**y + 1), True))

    # issue 8251:
    assert summation((1/(n + 1)**2)*n**2, (n, 0, oo)) is oo

    #issue 9908:
    assert Sum(1/(n**3 - 1), (n, -oo, -2)).doit() == summation(1/(n**3 - 1), (n, -oo, -2))

    #issue 11642:
    result = Sum(0.5**n, (n, 1, oo)).doit()
    assert result == 1.0
    assert result.is_Float

    result = Sum(0.25**n, (n, 1, oo)).doit()
    assert result == 1/3.
    assert result.is_Float

    result = Sum(0.99999**n, (n, 1, oo)).doit()
    assert result == 99999.0
    assert result.is_Float

    result = Sum(S.Half**n, (n, 1, oo)).doit()
    assert result == 1
    assert not result.is_Float

    result = Sum(Rational(3, 5)**n, (n, 1, oo)).doit()
    assert result == Rational(3, 2)
    assert not result.is_Float

    assert Sum(1.0**n, (n, 1, oo)).doit() is oo
    assert Sum(2.43**n, (n, 1, oo)).doit() is oo

    # Issue 13979
    i, k, q = symbols('i k q', integer=True)
    result = summation(
        exp(-2*I*pi*k*i/n) * exp(2*I*pi*q*i/n) / n, (i, 0, n - 1)
    )
    assert result.simplify() == Piecewise(
            (1, Eq(exp(-2*I*pi*(k - q)/n), 1)), (0, True)
    )

    #Issue 23491
    assert Sum(1/(n**2 + 1), (n, 1, oo)).doit() == S(-1)/2 + pi/(2*tanh(pi))

def test_harmonic_sums():
    assert summation(1/k, (k, 0, n)) == Sum(1/k, (k, 0, n))
    assert summation(1/k, (k, 1, n)) == harmonic(n)
    assert summation(n/k, (k, 1, n)) == n*harmonic(n)
    assert summation(1/k, (k, 5, n)) == harmonic(n) - harmonic(4)


def test_composite_sums():
    f = S.Half*(7 - 6*n + Rational(1, 7)*n**3)
    s = summation(f, (n, a, b))
    assert not isinstance(s, Sum)
    A = 0
    for i in range(-3, 5):
        A += f.subs(n, i)
    B = s.subs(a, -3).subs(b, 4)
    assert A == B


def test_hypergeometric_sums():
    assert summation(
        binomial(2*k, k)/4**k, (k, 0, n)) == (1 + 2*n)*binomial(2*n, n)/4**n
    assert summation(binomial(2*k, k)/5**k, (k, -oo, oo)) == sqrt(5)


def test_other_sums():
    f = m**2 + m*exp(m)
    g = 3*exp(Rational(3, 2))/2 + exp(S.Half)/2 - exp(Rational(-1, 2))/2 - 3*exp(Rational(-3, 2))/2 + 5

    assert summation(f, (m, Rational(-3, 2), Rational(3, 2))) == g
    assert summation(f, (m, -1.5, 1.5)).evalf().epsilon_eq(g.evalf(), 1e-10)

fac = factorial


def NS(e, n=15, **options):
    return str(sympify(e).evalf(n, **options))


def test_evalf_fast_series():
    # Euler transformed series for sqrt(1+x)
    assert NS(Sum(
        fac(2*n + 1)/fac(n)**2/2**(3*n + 1), (n, 0, oo)), 100) == NS(sqrt(2), 100)

    # Some series for exp(1)
    estr = NS(E, 100)
    assert NS(Sum(1/fac(n), (n, 0, oo)), 100) == estr
    assert NS(1/Sum((1 - 2*n)/fac(2*n), (n, 0, oo)), 100) == estr
    assert NS(Sum((2*n + 1)/fac(2*n), (n, 0, oo)), 100) == estr
    assert NS(Sum((4*n + 3)/2**(2*n + 1)/fac(2*n + 1), (n, 0, oo))**2, 100) == estr

    pistr = NS(pi, 100)
    # Ramanujan series for pi
    assert NS(9801/sqrt(8)/Sum(fac(
        4*n)*(1103 + 26390*n)/fac(n)**4/396**(4*n), (n, 0, oo)), 100) == pistr
    assert NS(1/Sum(
        binomial(2*n, n)**3 * (42*n + 5)/2**(12*n + 4), (n, 0, oo)), 100) == pistr
    # Machin's formula for pi
    assert NS(16*Sum((-1)**n/(2*n + 1)/5**(2*n + 1), (n, 0, oo)) -
        4*Sum((-1)**n/(2*n + 1)/239**(2*n + 1), (n, 0, oo)), 100) == pistr

    # Apery's constant
    astr = NS(zeta(3), 100)
    P = 126392*n**5 + 412708*n**4 + 531578*n**3 + 336367*n**2 + 104000* \
        n + 12463
    assert NS(Sum((-1)**n * P / 24 * (fac(2*n + 1)*fac(2*n)*fac(
        n))**3 / fac(3*n + 2) / fac(4*n + 3)**3, (n, 0, oo)), 100) == astr
    assert NS(Sum((-1)**n * (205*n**2 + 250*n + 77)/64 * fac(n)**10 /
              fac(2*n + 1)**5, (n, 0, oo)), 100) == astr


def test_evalf_fast_series_issue_4021():
    # Catalan's constant
    assert NS(Sum((-1)**(n - 1)*2**(8*n)*(40*n**2 - 24*n + 3)*fac(2*n)**3*
        fac(n)**2/n**3/(2*n - 1)/fac(4*n)**2, (n, 1, oo))/64, 100) == \
        NS(Catalan, 100)
    astr = NS(zeta(3), 100)
    assert NS(5*Sum(
        (-1)**(n - 1)*fac(n)**2 / n**3 / fac(2*n), (n, 1, oo))/2, 100) == astr
    assert NS(Sum((-1)**(n - 1)*(56*n**2 - 32*n + 5) / (2*n - 1)**2 * fac(n - 1)
              **3 / fac(3*n), (n, 1, oo))/4, 100) == astr


def test_evalf_slow_series():
    assert NS(Sum((-1)**n / n, (n, 1, oo)), 15) == NS(-log(2), 15)
    assert NS(Sum((-1)**n / n, (n, 1, oo)), 50) == NS(-log(2), 50)
    assert NS(Sum(1/n**2, (n, 1, oo)), 15) == NS(pi**2/6, 15)
    assert NS(Sum(1/n**2, (n, 1, oo)), 100) == NS(pi**2/6, 100)
    assert NS(Sum(1/n**2, (n, 1, oo)), 500) == NS(pi**2/6, 500)
    assert NS(Sum((-1)**n / (2*n + 1)**3, (n, 0, oo)), 15) == NS(pi**3/32, 15)
    assert NS(Sum((-1)**n / (2*n + 1)**3, (n, 0, oo)), 50) == NS(pi**3/32, 50)


def test_evalf_oo_to_oo():
    # There used to be an error in certain cases
    # Does not evaluate, but at least do not throw an error
    # Evaluates symbolically to 0, which is not correct
    assert Sum(1/(n**2+1), (n, -oo, oo)).evalf() == Sum(1/(n**2+1), (n, -oo, oo))
    # This evaluates if from 1 to oo and symbolically
    assert Sum(1/(factorial(abs(n))), (n, -oo, -1)).evalf() == Sum(1/(factorial(abs(n))), (n, -oo, -1))


def test_euler_maclaurin():
    # Exact polynomial sums with E-M
    def check_exact(f, a, b, m, n):
        A = Sum(f, (k, a, b))
        s, e = A.euler_maclaurin(m, n)
        assert (e == 0) and (s.expand() == A.doit())
    check_exact(k**4, a, b, 0, 2)
    check_exact(k**4 + 2*k, a, b, 1, 2)
    check_exact(k**4 + k**2, a, b, 1, 5)
    check_exact(k**5, 2, 6, 1, 2)
    check_exact(k**5, 2, 6, 1, 3)
    assert Sum(x-1, (x, 0, 2)).euler_maclaurin(m=30, n=30, eps=2**-15) == (0, 0)
    # Not exact
    assert Sum(k**6, (k, a, b)).euler_maclaurin(0, 2)[1] != 0
    # Numerical test
    for mi, ni in [(2, 4), (2, 20), (10, 20), (18, 20)]:
        A = Sum(1/k**3, (k, 1, oo))
        s, e = A.euler_maclaurin(mi, ni)
        assert abs((s - zeta(3)).evalf()) < e.evalf()

    raises(ValueError, lambda: Sum(1, (x, 0, 1), (k, 0, 1)).euler_maclaurin())


@slow
def test_evalf_euler_maclaurin():
    assert NS(Sum(1/k**k, (k, 1, oo)), 15) == '1.29128599706266'
    assert NS(Sum(1/k**k, (k, 1, oo)),
              50) == '1.2912859970626635404072825905956005414986193682745'
    assert NS(Sum(1/k - log(1 + 1/k), (k, 1, oo)), 15) == NS(EulerGamma, 15)
    assert NS(Sum(1/k - log(1 + 1/k), (k, 1, oo)), 50) == NS(EulerGamma, 50)
    assert NS(Sum(log(k)/k**2, (k, 1, oo)), 15) == '0.937548254315844'
    assert NS(Sum(log(k)/k**2, (k, 1, oo)),
              50) == '0.93754825431584375370257409456786497789786028861483'
    assert NS(Sum(1/k, (k, 1000000, 2000000)), 15) == '0.693147930560008'
    assert NS(Sum(1/k, (k, 1000000, 2000000)),
              50) == '0.69314793056000780941723211364567656807940638436025'


def test_evalf_symbolic():
    # issue 6328
    expr = Sum(f(x), (x, 1, 3)) + Sum(g(x), (x, 1, 3))
    assert expr.evalf() == expr


def test_evalf_issue_3273():
    assert Sum(0, (k, 1, oo)).evalf() == 0


def test_simple_products():
    assert Product(S.NaN, (x, 1, 3)) is S.NaN
    assert product(S.NaN, (x, 1, 3)) is S.NaN
    assert Product(x, (n, a, a)).doit() == x
    assert Product(x, (x, a, a)).doit() == a
    assert Product(x, (y, 1, a)).doit() == x**a

    lo, hi = 1, 2
    s1 = Product(n, (n, lo, hi))
    s2 = Product(n, (n, hi, lo))
    assert s1 != s2
    # This IS correct according to Karr product convention
    assert s1.doit() == 2
    assert s2.doit() == 1

    lo, hi = x, x + 1
    s1 = Product(n, (n, lo, hi))
    s2 = Product(n, (n, hi, lo))
    s3 = 1 / Product(n, (n, hi + 1, lo - 1))
    assert s1 != s2
    # This IS correct according to Karr product convention
    assert s1.doit() == x*(x + 1)
    assert s2.doit() == 1
    assert s3.doit() == x*(x + 1)

    assert Product(Integral(2*x, (x, 1, y)) + 2*x, (x, 1, 2)).doit() == \
        (y**2 + 1)*(y**2 + 3)
    assert product(2, (n, a, b)) == 2**(b - a + 1)
    assert product(n, (n, 1, b)) == factorial(b)
    assert product(n**3, (n, 1, b)) == factorial(b)**3
    assert product(3**(2 + n), (n, a, b)) \
        == 3**(2*(1 - a + b) + b/2 + (b**2)/2 + a/2 - (a**2)/2)
    assert product(cos(n), (n, 3, 5)) == cos(3)*cos(4)*cos(5)
    assert product(cos(n), (n, x, x + 2)) == cos(x)*cos(x + 1)*cos(x + 2)
    assert isinstance(product(cos(n), (n, x, x + S.Half)), Product)
    # If Product managed to evaluate this one, it most likely got it wrong!
    assert isinstance(Product(n**n, (n, 1, b)), Product)


def test_rational_products():
    assert combsimp(product(1 + 1/n, (n, a, b))) == (1 + b)/a
    assert combsimp(product(n + 1, (n, a, b))) == gamma(2 + b)/gamma(1 + a)
    assert combsimp(product((n + 1)/(n - 1), (n, a, b))) == b*(1 + b)/(a*(a - 1))
    assert combsimp(product(n/(n + 1)/(n + 2), (n, a, b))) == \
        a*gamma(a + 2)/(b + 1)/gamma(b + 3)
    assert combsimp(product(n*(n + 1)/(n - 1)/(n - 2), (n, a, b))) == \
        b**2*(b - 1)*(1 + b)/(a - 1)**2/(a*(a - 2))


def test_wallis_product():
    # Wallis product, given in two different forms to ensure that Product
    # can factor simple rational expressions
    A = Product(4*n**2 / (4*n**2 - 1), (n, 1, b))
    B = Product((2*n)*(2*n)/(2*n - 1)/(2*n + 1), (n, 1, b))
    R = pi*gamma(b + 1)**2/(2*gamma(b + S.Half)*gamma(b + Rational(3, 2)))
    assert simplify(A.doit()) == R
    assert simplify(B.doit()) == R
    # This one should eventually also be doable (Euler's product formula for sin)
    # assert Product(1+x/n**2, (n, 1, b)) == ...


def test_telescopic_sums():
    #checks also input 2 of comment 1 issue 4127
    assert Sum(1/k - 1/(k + 1), (k, 1, n)).doit() == 1 - 1/(1 + n)
    assert Sum(
        f(k) - f(k + 2), (k, m, n)).doit() == -f(1 + n) - f(2 + n) + f(m) + f(1 + m)
    assert Sum(cos(k) - cos(k + 3), (k, 1, n)).doit() == -cos(1 + n) - \
        cos(2 + n) - cos(3 + n) + cos(1) + cos(2) + cos(3)

    # dummy variable shouldn't matter
    assert telescopic(1/m, -m/(1 + m), (m, n - 1, n)) == \
        telescopic(1/k, -k/(1 + k), (k, n - 1, n))

    assert Sum(1/x/(x - 1), (x, a, b)).doit() == 1/(a - 1) - 1/b
    eq = 1/((5*n + 2)*(5*(n + 1) + 2))
    assert Sum(eq, (n, 0, oo)).doit() == S(1)/10
    nz = symbols('nz', nonzero=True)
    v = Sum(eq.subs(5, nz), (n, 0, oo)).doit()
    assert v.subs(nz, 5).simplify() == S(1)/10
    # check that apart is being used in non-symbolic case
    s = Sum(eq, (n, 0, k)).doit()
    v = Sum(eq, (n, 0, 10**100)).doit()
    assert v == s.subs(k, 10**100)


def test_sum_reconstruct():
    s = Sum(n**2, (n, -1, 1))
    assert s == Sum(*s.args)
    raises(ValueError, lambda: Sum(x, x))
    raises(ValueError, lambda: Sum(x, (x, 1)))


def test_limit_subs():
    for F in (Sum, Product, Integral):
        assert F(a*exp(a), (a, -2, 2)) == F(a*exp(a), (a, -b, b)).subs(b, 2)
        assert F(a, (a, F(b, (b, 1, 2)), 4)).subs(F(b, (b, 1, 2)), c) == \
            F(a, (a, c, 4))
        assert F(x, (x, 1, x + y)).subs(x, 1) == F(x, (x, 1, y + 1))


def test_function_subs():
    S = Sum(x*f(y),(x,0,oo),(y,0,oo))
    assert S.subs(f(y),y) == Sum(x*y,(x,0,oo),(y,0,oo))
    assert S.subs(f(x),x) == S
    raises(ValueError, lambda: S.subs(f(y),x+y) )
    S = Sum(x*log(y),(x,0,oo),(y,0,oo))
    assert S.subs(log(y),y) == S
    S = Sum(x*f(y),(x,0,oo),(y,0,oo))
    assert S.subs(f(y),y) == Sum(x*y,(x,0,oo),(y,0,oo))


def test_equality():
    # if this fails remove special handling below
    raises(ValueError, lambda: Sum(x, x))
    r = symbols('x', real=True)
    for F in (Sum, Product, Integral):
        try:
            assert F(x, x) != F(y, y)
            assert F(x, (x, 1, 2)) != F(x, x)
            assert F(x, (x, x)) != F(x, x)  # or else they print the same
            assert F(1, x) != F(1, y)
        except ValueError:
            pass
        assert F(a, (x, 1, 2)) != F(a, (x, 1, 3))  # diff limit
        assert F(a, (x, 1, x)) != F(a, (y, 1, y))
        assert F(a, (x, 1, 2)) != F(b, (x, 1, 2))  # diff expression
        assert F(x, (x, 1, 2)) != F(r, (r, 1, 2))  # diff assumptions
        assert F(1, (x, 1, x)) != F(1, (y, 1, x))  # only dummy is diff
        assert F(1, (x, 1, x)).dummy_eq(F(1, (y, 1, x)))

    # issue 5265
    assert Sum(x, (x, 1, x)).subs(x, a) == Sum(x, (x, 1, a))


def test_Sum_doit():
    assert Sum(n*Integral(a**2), (n, 0, 2)).doit() == a**3
    assert Sum(n*Integral(a**2), (n, 0, 2)).doit(deep=False) == \
        3*Integral(a**2)
    assert summation(n*Integral(a**2), (n, 0, 2)) == 3*Integral(a**2)

    # test nested sum evaluation
    s = Sum( Sum( Sum(2,(z,1,n+1)), (y,x+1,n)), (x,1,n))
    assert 0 == (s.doit() - n*(n+1)*(n-1)).factor()

    # Integer assumes finite
    assert Sum(KroneckerDelta(x, y), (x, -oo, oo)).doit() == Piecewise((1, And(-oo < y, y < oo)), (0, True))
    assert Sum(KroneckerDelta(m, n), (m, -oo, oo)).doit() == 1
    assert Sum(m*KroneckerDelta(x, y), (x, -oo, oo)).doit() == Piecewise((m, And(-oo < y, y < oo)), (0, True))
    assert Sum(x*KroneckerDelta(m, n), (m, -oo, oo)).doit() == x
    assert Sum(Sum(KroneckerDelta(m, n), (m, 1, 3)), (n, 1, 3)).doit() == 3
    assert Sum(Sum(KroneckerDelta(k, m), (m, 1, 3)), (n, 1, 3)).doit() == \
           3 * Piecewise((1, And(1 <= k, k <= 3)), (0, True))
    assert Sum(f(n) * Sum(KroneckerDelta(m, n), (m, 0, oo)), (n, 1, 3)).doit() == \
           f(1) + f(2) + f(3)
    assert Sum(f(n) * Sum(KroneckerDelta(m, n), (m, 0, oo)), (n, 1, oo)).doit() == \
           Sum(f(n), (n, 1, oo))

    # issue 2597
    nmax = symbols('N', integer=True, positive=True)
    pw = Piecewise((1, And(1 <= n, n <= nmax)), (0, True))
    assert Sum(pw, (n, 1, nmax)).doit() == Sum(Piecewise((1, nmax >= n),
                    (0, True)), (n, 1, nmax))

    q, s = symbols('q, s')
    assert summation(1/n**(2*s), (n, 1, oo)) == Piecewise((zeta(2*s), 2*re(s) > 1),
        (Sum(n**(-2*s), (n, 1, oo)), True))
    assert summation(1/(n+1)**s, (n, 0, oo)) == Piecewise((zeta(s), re(s) > 1),
        (Sum((n + 1)**(-s), (n, 0, oo)), True))
    assert summation(1/(n+q)**s, (n, 0, oo)) == Piecewise(
        (zeta(s, q), And(~Contains(-q, S.Naturals0), re(s) > 1)),
        (Sum((n + q)**(-s), (n, 0, oo)), True))
    assert summation(1/(n+q)**s, (n, q, oo)) == Piecewise(
        (zeta(s, 2*q), And(~Contains(-2*q, S.Naturals0), re(s) > 1)),
        (Sum((n + q)**(-s), (n, q, oo)), True))
    assert summation(1/n**2, (n, 1, oo)) == zeta(2)
    assert summation(1/n**s, (n, 0, oo)) == Sum(n**(-s), (n, 0, oo))
    assert summation(1/(n+1)**(2+I), (n, 0, oo)) == zeta(2+I)
    t = symbols('t', real=True, positive=True)
    assert summation(1/(n+I)**(t+1), (n, 0, oo)) == zeta(t+1, I)


def test_Product_doit():
    assert Product(n*Integral(a**2), (n, 1, 3)).doit() == 2 * a**9 / 9
    assert Product(n*Integral(a**2), (n, 1, 3)).doit(deep=False) == \
        6*Integral(a**2)**3
    assert product(n*Integral(a**2), (n, 1, 3)) == 6*Integral(a**2)**3


def test_Sum_interface():
    assert isinstance(Sum(0, (n, 0, 2)), Sum)
    assert Sum(nan, (n, 0, 2)) is nan
    assert Sum(nan, (n, 0, oo)) is nan
    assert Sum(0, (n, 0, 2)).doit() == 0
    assert isinstance(Sum(0, (n, 0, oo)), Sum)
    assert Sum(0, (n, 0, oo)).doit() == 0
    raises(ValueError, lambda: Sum(1))
    raises(ValueError, lambda: summation(1))


def test_diff():
    assert Sum(x, (x, 1, 2)).diff(x) == 0
    assert Sum(x*y, (x, 1, 2)).diff(x) == 0
    assert Sum(x*y, (y, 1, 2)).diff(x) == Sum(y, (y, 1, 2))
    e = Sum(x*y, (x, 1, a))
    assert e.diff(a) == Derivative(e, a)
    assert Sum(x*y, (x, 1, 3), (a, 2, 5)).diff(y).doit() == \
        Sum(x*y, (x, 1, 3), (a, 2, 5)).doit().diff(y) == 24
    assert Sum(x, (x, 1, 2)).diff(y) == 0


def test_hypersum():
    assert simplify(summation(x**n/fac(n), (n, 1, oo))) == -1 + exp(x)
    assert summation((-1)**n * x**(2*n) / fac(2*n), (n, 0, oo)) == cos(x)
    assert simplify(summation((-1)**n*x**(2*n + 1) /
        factorial(2*n + 1), (n, 3, oo))) == -x + sin(x) + x**3/6 - x**5/120

    assert summation(1/(n + 2)**3, (n, 1, oo)) == Rational(-9, 8) + zeta(3)
    assert summation(1/n**4, (n, 1, oo)) == pi**4/90

    s = summation(x**n*n, (n, -oo, 0))
    assert s.is_Piecewise
    assert s.args[0].args[0] == -1/(x*(1 - 1/x)**2)
    assert s.args[0].args[1] == (abs(1/x) < 1)

    m = Symbol('n', integer=True, positive=True)
    assert summation(binomial(m, k), (k, 0, m)) == 2**m


def test_issue_4170():
    assert summation(1/factorial(k), (k, 0, oo)) == E


def test_is_commutative():
    from sympy.physics.secondquant import NO, F, Fd
    m = Symbol('m', commutative=False)
    for f in (Sum, Product, Integral):
        assert f(z, (z, 1, 1)).is_commutative is True
        assert f(z*y, (z, 1, 6)).is_commutative is True
        assert f(m*x, (x, 1, 2)).is_commutative is False

        assert f(NO(Fd(x)*F(y))*z, (z, 1, 2)).is_commutative is False


def test_is_zero():
    for func in [Sum, Product]:
        assert func(0, (x, 1, 1)).is_zero is True
        assert func(x, (x, 1, 1)).is_zero is None

    assert Sum(0, (x, 1, 0)).is_zero is True
    assert Product(0, (x, 1, 0)).is_zero is False


def test_is_number():
    # is number should not rely on evaluation or assumptions,
    # it should be equivalent to `not foo.free_symbols`
    assert Sum(1, (x, 1, 1)).is_number is True
    assert Sum(1, (x, 1, x)).is_number is False
    assert Sum(0, (x, y, z)).is_number is False
    assert Sum(x, (y, 1, 2)).is_number is False
    assert Sum(x, (y, 1, 1)).is_number is False
    assert Sum(x, (x, 1, 2)).is_number is True
    assert Sum(x*y, (x, 1, 2), (y, 1, 3)).is_number is True

    assert Product(2, (x, 1, 1)).is_number is True
    assert Product(2, (x, 1, y)).is_number is False
    assert Product(0, (x, y, z)).is_number is False
    assert Product(1, (x, y, z)).is_number is False
    assert Product(x, (y, 1, x)).is_number is False
    assert Product(x, (y, 1, 2)).is_number is False
    assert Product(x, (y, 1, 1)).is_number is False
    assert Product(x, (x, 1, 2)).is_number is True


def test_free_symbols():
    for func in [Sum, Product]:
        assert func(1, (x, 1, 2)).free_symbols == set()
        assert func(0, (x, 1, y)).free_symbols == {y}
        assert func(2, (x, 1, y)).free_symbols == {y}
        assert func(x, (x, 1, 2)).free_symbols == set()
        assert func(x, (x, 1, y)).free_symbols == {y}
        assert func(x, (y, 1, y)).free_symbols == {x, y}
        assert func(x, (y, 1, 2)).free_symbols == {x}
        assert func(x, (y, 1, 1)).free_symbols == {x}
        assert func(x, (y, 1, z)).free_symbols == {x, z}
        assert func(x, (x, 1, y), (y, 1, 2)).free_symbols == set()
        assert func(x, (x, 1, y), (y, 1, z)).free_symbols == {z}
        assert func(x, (x, 1, y), (y, 1, y)).free_symbols == {y}
        assert func(x, (y, 1, y), (y, 1, z)).free_symbols == {x, z}
    assert Sum(1, (x, 1, y)).free_symbols == {y}
    # free_symbols answers whether the object *as written* has free symbols,
    # not whether the evaluated expression has free symbols
    assert Product(1, (x, 1, y)).free_symbols == {y}
    # don't count free symbols that are not independent of integration
    # variable(s)
    assert func(f(x), (f(x), 1, 2)).free_symbols == set()
    assert func(f(x), (f(x), 1, x)).free_symbols == {x}
    assert func(f(x), (f(x), 1, y)).free_symbols == {y}
    assert func(f(x), (z, 1, y)).free_symbols == {x, y}


def test_conjugate_transpose():
    A, B = symbols("A B", commutative=False)
    p = Sum(A*B**n, (n, 1, 3))
    assert p.adjoint().doit() == p.doit().adjoint()
    assert p.conjugate().doit() == p.doit().conjugate()
    assert p.transpose().doit() == p.doit().transpose()

    p = Sum(B**n*A, (n, 1, 3))
    assert p.adjoint().doit() == p.doit().adjoint()
    assert p.conjugate().doit() == p.doit().conjugate()
    assert p.transpose().doit() == p.doit().transpose()


def test_noncommutativity_honoured():
    A, B = symbols("A B", commutative=False)
    M = symbols('M', integer=True, positive=True)
    p = Sum(A*B**n, (n, 1, M))
    assert p.doit() == A*Piecewise((M, Eq(B, 1)),
                                   ((B - B**(M + 1))*(1 - B)**(-1), True))

    p = Sum(B**n*A, (n, 1, M))
    assert p.doit() == Piecewise((M, Eq(B, 1)),
                                 ((B - B**(M + 1))*(1 - B)**(-1), True))*A

    p = Sum(B**n*A*B**n, (n, 1, M))
    assert p.doit() == p


def test_issue_4171():
    assert summation(factorial(2*k + 1)/factorial(2*k), (k, 0, oo)) is oo
    assert summation(2*k + 1, (k, 0, oo)) is oo


def test_issue_6273():
    assert Sum(x, (x, 1, n)).n(2, subs={n: 1}) == Float(1, 2)


def test_issue_6274():
    assert Sum(x, (x, 1, 0)).doit() == 0
    assert NS(Sum(x, (x, 1, 0))) == '0'
    assert Sum(n, (n, 10, 5)).doit() == -30
    assert NS(Sum(n, (n, 10, 5))) == '-30.0000000000000'


def test_simplify_sum():
    y, t, v = symbols('y, t, v')

    _simplify = lambda e: simplify(e, doit=False)
    assert _simplify(Sum(x*y, (x, n, m), (y, a, k)) + \
        Sum(y, (x, n, m), (y, a, k))) == Sum(y * (x + 1), (x, n, m), (y, a, k))
    assert _simplify(Sum(x, (x, n, m)) + Sum(x, (x, m + 1, a))) == \
        Sum(x, (x, n, a))
    assert _simplify(Sum(x, (x, k + 1, a)) + Sum(x, (x, n, k))) == \
        Sum(x, (x, n, a))
    assert _simplify(Sum(x, (x, k + 1, a)) + Sum(x + 1, (x, n, k))) == \
        Sum(x, (x, n, a)) + Sum(1, (x, n, k))
    assert _simplify(Sum(x, (x, 0, 3)) * 3 + 3 * Sum(x, (x, 4, 6)) + \
        4 * Sum(z, (z, 0, 1))) == 4*Sum(z, (z, 0, 1)) + 3*Sum(x, (x, 0, 6))
    assert _simplify(3*Sum(x**2, (x, a, b)) + Sum(x, (x, a, b))) == \
        Sum(x*(3*x + 1), (x, a, b))
    assert _simplify(Sum(x**3, (x, n, k)) * 3 + 3 * Sum(x, (x, n, k)) + \
        4 * y * Sum(z, (z, n, k))) + 1 == \
            4*y*Sum(z, (z, n, k)) + 3*Sum(x**3 + x, (x, n, k)) + 1
    assert _simplify(Sum(x, (x, a, b)) + 1 + Sum(x, (x, b + 1, c))) == \
        1 + Sum(x, (x, a, c))
    assert _simplify(Sum(x, (t, a, b)) + Sum(y, (t, a, b)) + \
        Sum(x, (t, b+1, c))) == x * Sum(1, (t, a, c)) + y * Sum(1, (t, a, b))
    assert _simplify(Sum(x, (t, a, b)) + Sum(x, (t, b+1, c)) + \
        Sum(y, (t, a, b))) == x * Sum(1, (t, a, c)) + y * Sum(1, (t, a, b))
    assert _simplify(Sum(x, (t, a, b)) + 2 * Sum(x, (t, b+1, c))) == \
        _simplify(Sum(x, (t, a, b)) + Sum(x, (t, b+1, c)) + Sum(x, (t, b+1, c)))
    assert _simplify(Sum(x, (x, a, b))*Sum(x**2, (x, a, b))) == \
        Sum(x, (x, a, b)) * Sum(x**2, (x, a, b))
    assert _simplify(Sum(x, (t, a, b)) + Sum(y, (t, a, b)) + Sum(z, (t, a, b))) \
        == (x + y + z) * Sum(1, (t, a, b))          # issue 8596
    assert _simplify(Sum(x, (t, a, b)) + Sum(y, (t, a, b)) + Sum(z, (t, a, b)) + \
        Sum(v, (t, a, b))) == (x + y + z + v) * Sum(1, (t, a, b))  # issue 8596
    assert _simplify(Sum(x * y, (x, a, b)) / (3 * y)) == \
        (Sum(x, (x, a, b)) / 3)
    assert _simplify(Sum(f(x) * y * z, (x, a, b)) / (y * z)) \
        == Sum(f(x), (x, a, b))
    assert _simplify(Sum(c * x, (x, a, b)) - c * Sum(x, (x, a, b))) == 0
    assert _simplify(c * (Sum(x, (x, a, b))  + y)) == c * (y + Sum(x, (x, a, b)))
    assert _simplify(c * (Sum(x, (x, a, b)) + y * Sum(x, (x, a, b)))) == \
        c * (y + 1) * Sum(x, (x, a, b))
    assert _simplify(Sum(Sum(c * x, (x, a, b)), (y, a, b))) == \
                c * Sum(x, (x, a, b), (y, a, b))
    assert _simplify(Sum((3 + y) * Sum(c * x, (x, a, b)), (y, a, b))) == \
                c * Sum((3 + y), (y, a, b)) * Sum(x, (x, a, b))
    assert _simplify(Sum((3 + t) * Sum(c * t, (x, a, b)), (y, a, b))) == \
                c*t*(t + 3)*Sum(1, (x, a, b))*Sum(1, (y, a, b))
    assert _simplify(Sum(Sum(d * t, (x, a, b - 1)) + \
                Sum(d * t, (x, b, c)), (t, a, b))) == \
                    d * Sum(1, (x, a, c)) * Sum(t, (t, a, b))
    assert _simplify(Sum(sin(t)**2 + cos(t)**2 + 1, (t, a, b))) == \
        2 * Sum(1, (t, a, b))


def test_change_index():
    b, v, w = symbols('b, v, w', integer = True)

    assert Sum(x, (x, a, b)).change_index(x, x + 1, y) == \
        Sum(y - 1, (y, a + 1, b + 1))
    assert Sum(x**2, (x, a, b)).change_index( x, x - 1) == \
        Sum((x+1)**2, (x, a - 1, b - 1))
    assert Sum(x**2, (x, a, b)).change_index( x, -x, y) == \
        Sum((-y)**2, (y, -b, -a))
    assert Sum(x, (x, a, b)).change_index( x, -x - 1) == \
        Sum(-x - 1, (x, -b - 1, -a - 1))
    assert Sum(x*y, (x, a, b), (y, c, d)).change_index( x, x - 1, z) == \
        Sum((z + 1)*y, (z, a - 1, b - 1), (y, c, d))
    assert Sum(x, (x, a, b)).change_index( x, x + v) == \
        Sum(-v + x, (x, a + v, b + v))
    assert Sum(x, (x, a, b)).change_index( x, -x - v) == \
        Sum(-v - x, (x, -b - v, -a - v))
    assert Sum(x, (x, a, b)).change_index(x, w*x, v) == \
        Sum(v/w, (v, b*w, a*w))
    raises(ValueError, lambda: Sum(x, (x, a, b)).change_index(x, 2*x))


def test_reorder():
    b, y, c, d, z = symbols('b, y, c, d, z', integer = True)

    assert Sum(x*y, (x, a, b), (y, c, d)).reorder((0, 1)) == \
        Sum(x*y, (y, c, d), (x, a, b))
    assert Sum(x, (x, a, b), (x, c, d)).reorder((0, 1)) == \
        Sum(x, (x, c, d), (x, a, b))
    assert Sum(x*y + z, (x, a, b), (z, m, n), (y, c, d)).reorder(\
        (2, 0), (0, 1)) == Sum(x*y + z, (z, m, n), (y, c, d), (x, a, b))
    assert Sum(x*y*z, (x, a, b), (y, c, d), (z, m, n)).reorder(\
        (0, 1), (1, 2), (0, 2)) == Sum(x*y*z, (x, a, b), (z, m, n), (y, c, d))
    assert Sum(x*y*z, (x, a, b), (y, c, d), (z, m, n)).reorder(\
        (x, y), (y, z), (x, z)) == Sum(x*y*z, (x, a, b), (z, m, n), (y, c, d))
    assert Sum(x*y, (x, a, b), (y, c, d)).reorder((x, 1)) == \
        Sum(x*y, (y, c, d), (x, a, b))
    assert Sum(x*y, (x, a, b), (y, c, d)).reorder((y, x)) == \
        Sum(x*y, (y, c, d), (x, a, b))


def test_reverse_order():
    assert Sum(x, (x, 0, 3)).reverse_order(0) == Sum(-x, (x, 4, -1))
    assert Sum(x*y, (x, 1, 5), (y, 0, 6)).reverse_order(0, 1) == \
           Sum(x*y, (x, 6, 0), (y, 7, -1))
    assert Sum(x, (x, 1, 2)).reverse_order(0) == Sum(-x, (x, 3, 0))
    assert Sum(x, (x, 1, 3)).reverse_order(0) == Sum(-x, (x, 4, 0))
    assert Sum(x, (x, 1, a)).reverse_order(0) == Sum(-x, (x, a + 1, 0))
    assert Sum(x, (x, a, 5)).reverse_order(0) == Sum(-x, (x, 6, a - 1))
    assert Sum(x, (x, a + 1, a + 5)).reverse_order(0) == \
                         Sum(-x, (x, a + 6, a))
    assert Sum(x, (x, a + 1, a + 2)).reverse_order(0) == \
           Sum(-x, (x, a + 3, a))
    assert Sum(x, (x, a + 1, a + 1)).reverse_order(0) == \
           Sum(-x, (x, a + 2, a))
    assert Sum(x, (x, a, b)).reverse_order(0) == Sum(-x, (x, b + 1, a - 1))
    assert Sum(x, (x, a, b)).reverse_order(x) == Sum(-x, (x, b + 1, a - 1))
    assert Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(x, 1) == \
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))
    assert Sum(x*y, (x, a, b), (y, 2, 5)).reverse_order(y, x) == \
        Sum(x*y, (x, b + 1, a - 1), (y, 6, 1))


def test_issue_7097():
    assert sum(x**n/n for n in range(1, 401)) == summation(x**n/n, (n, 1, 400))


def test_factor_expand_subs():
    # test factoring
    assert Sum(4 * x, (x, 1, y)).factor() == 4 * Sum(x, (x, 1, y))
    assert Sum(x * a, (x, 1, y)).factor() == a * Sum(x, (x, 1, y))
    assert Sum(4 * x * a, (x, 1, y)).factor() == 4 * a * Sum(x, (x, 1, y))
    assert Sum(4 * x * y, (x, 1, y)).factor() == 4 * y * Sum(x, (x, 1, y))

    # test expand
    _x = Symbol('x', zero=False)
    assert Sum(x+1,(x,1,y)).expand() == Sum(x,(x,1,y)) + Sum(1,(x,1,y))
    assert Sum(x+a*x**2,(x,1,y)).expand() == Sum(x,(x,1,y)) + Sum(a*x**2,(x,1,y))
    assert Sum(_x**(n + 1)*(n + 1), (n, -1, oo)).expand() \
        == Sum(n*_x*_x**n + _x*_x**n, (n, -1, oo))
    assert Sum(x**(n + 1)*(n + 1), (n, -1, oo)).expand(power_exp=False) \
        == Sum(n*x**(n + 1) + x**(n + 1), (n, -1, oo))
    assert Sum(x**(n + 1)*(n + 1), (n, -1, oo)).expand(force=True) \
           == Sum(x*x**n, (n, -1, oo)) + Sum(n*x*x**n, (n, -1, oo))
    assert Sum(a*n+a*n**2,(n,0,4)).expand() \
        == Sum(a*n,(n,0,4)) + Sum(a*n**2,(n,0,4))
    assert Sum(_x**a*_x**n,(x,0,3)) \
        == Sum(_x**(a+n),(x,0,3)).expand(power_exp=True)
    _a, _n = symbols('a n', positive=True)
    assert Sum(x**(_a+_n),(x,0,3)).expand(power_exp=True) \
        == Sum(x**_a*x**_n, (x, 0, 3))
    assert Sum(x**(_a-_n),(x,0,3)).expand(power_exp=True) \
        == Sum(x**(_a-_n),(x,0,3)).expand(power_exp=False)

    # test subs
    assert Sum(1/(1+a*x**2),(x,0,3)).subs([(a,3)]) == Sum(1/(1+3*x**2),(x,0,3))
    assert Sum(x*y,(x,0,y),(y,0,x)).subs([(x,3)]) == Sum(x*y,(x,0,y),(y,0,3))
    assert Sum(x,(x,1,10)).subs([(x,y-2)]) == Sum(x,(x,1,10))
    assert Sum(1/x,(x,1,10)).subs([(x,(3+n)**3)]) == Sum(1/x,(x,1,10))
    assert Sum(1/x,(x,1,10)).subs([(x,3*x-2)]) == Sum(1/x,(x,1,10))


def test_distribution_over_equality():
    assert Product(Eq(x*2, f(x)), (x, 1, 3)).doit() == Eq(48, f(1)*f(2)*f(3))
    assert Sum(Eq(f(x), x**2), (x, 0, y)) == \
        Eq(Sum(f(x), (x, 0, y)), Sum(x**2, (x, 0, y)))


def test_issue_2787():
    n, k = symbols('n k', positive=True, integer=True)
    p = symbols('p', positive=True)
    binomial_dist = binomial(n, k)*p**k*(1 - p)**(n - k)
    s = Sum(binomial_dist*k, (k, 0, n))
    res = s.doit().simplify()
    ans = Piecewise(
        (n*p, x),
        (Sum(k*p**k*binomial(n, k)*(1 - p)**(n - k), (k, 0, n)),
        True)).subs(x, (Eq(n, 1) | (n > 1)) & (p/Abs(p - 1) <= 1))
    ans2 = Piecewise(
        (n*p, x),
        (factorial(n)*Sum(p**k*(1 - p)**(-k + n)/
        (factorial(-k + n)*factorial(k - 1)), (k, 0, n)),
        True)).subs(x, (Eq(n, 1) | (n > 1)) & (p/Abs(p - 1) <= 1))
    assert res in [ans, ans2]  # XXX system dependent
    # Issue #17165: make sure that another simplify does not complicate
    # the result by much. Why didn't first simplify replace
    # Eq(n, 1) | (n > 1) with True?
    assert res.simplify().count_ops() <= res.count_ops() + 2


def test_issue_4668():
    assert summation(1/n, (n, 2, oo)) is oo


def test_matrix_sum():
    A = Matrix([[0, 1], [n, 0]])

    result = Sum(A, (n, 0, 3)).doit()
    assert result == Matrix([[0, 4], [6, 0]])
    assert result.__class__ == ImmutableDenseMatrix

    A = SparseMatrix([[0, 1], [n, 0]])

    result = Sum(A, (n, 0, 3)).doit()
    assert result.__class__ == ImmutableSparseMatrix


def test_failing_matrix_sum():
    n = Symbol('n')
    # TODO Implement matrix geometric series summation.
    A = Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    assert Sum(A ** n, (n, 1, 4)).doit() == \
        Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    # issue sympy/sympy#16989
    assert summation(A**n, (n, 1, 1)) == A


def test_indexed_idx_sum():
    i = symbols('i', cls=Idx)
    r = Indexed('r', i)
    assert Sum(r, (i, 0, 3)).doit() == sum(r.xreplace({i: j}) for j in range(4))
    assert Product(r, (i, 0, 3)).doit() == prod([r.xreplace({i: j}) for j in range(4)])

    j = symbols('j', integer=True)
    assert Sum(r, (i, j, j+2)).doit() == sum(r.xreplace({i: j+k}) for k in range(3))
    assert Product(r, (i, j, j+2)).doit() == prod([r.xreplace({i: j+k}) for k in range(3)])

    k = Idx('k', range=(1, 3))
    A = IndexedBase('A')
    assert Sum(A[k], k).doit() == sum(A[Idx(j, (1, 3))] for j in range(1, 4))
    assert Product(A[k], k).doit() == prod([A[Idx(j, (1, 3))] for j in range(1, 4)])

    raises(ValueError, lambda: Sum(A[k], (k, 1, 4)))
    raises(ValueError, lambda: Sum(A[k], (k, 0, 3)))
    raises(ValueError, lambda: Sum(A[k], (k, 2, oo)))

    raises(ValueError, lambda: Product(A[k], (k, 1, 4)))
    raises(ValueError, lambda: Product(A[k], (k, 0, 3)))
    raises(ValueError, lambda: Product(A[k], (k, 2, oo)))


@slow
def test_is_convergent():
    # divergence tests --
    assert Sum(n/(2*n + 1), (n, 1, oo)).is_convergent() is S.false
    assert Sum(factorial(n)/5**n, (n, 1, oo)).is_convergent() is S.false
    assert Sum(3**(-2*n - 1)*n**n, (n, 1, oo)).is_convergent() is S.false
    assert Sum((-1)**n*n, (n, 3, oo)).is_convergent() is S.false
    assert Sum((-1)**n, (n, 1, oo)).is_convergent() is S.false
    assert Sum(log(1/n), (n, 2, oo)).is_convergent() is S.false
    assert Sum(sin(n), (n, 1, oo)).is_convergent() is S.false

    # Raabe's test --
    assert Sum(Product((3*m),(m,1,n))/Product((3*m+4),(m,1,n)),(n,1,oo)).is_convergent() is S.true

    # root test --
    assert Sum((-12)**n/n, (n, 1, oo)).is_convergent() is S.false

    # integral test --

    # p-series test --
    assert Sum(1/(n**2 + 1), (n, 1, oo)).is_convergent() is S.true
    assert Sum(1/n**Rational(6, 5), (n, 1, oo)).is_convergent() is S.true
    assert Sum(2/(n*sqrt(n - 1)), (n, 2, oo)).is_convergent() is S.true
    assert Sum(1/(sqrt(n)*sqrt(n)), (n, 2, oo)).is_convergent() is S.false
    assert Sum(factorial(n) / factorial(n+2), (n, 1, oo)).is_convergent() is S.true
    assert Sum(rf(5,n)/rf(7,n),(n,1,oo)).is_convergent() is S.true
    assert Sum((rf(1, n)*rf(2, n))/(rf(3, n)*factorial(n)),(n,1,oo)).is_convergent() is S.false

    # comparison test --
    assert Sum(1/(n + log(n)), (n, 1, oo)).is_convergent() is S.false
    assert Sum(1/(n**2*log(n)), (n, 2, oo)).is_convergent() is S.true
    assert Sum(1/(n*log(n)), (n, 2, oo)).is_convergent() is S.false
    assert Sum(2/(n*log(n)*log(log(n))**2), (n, 5, oo)).is_convergent() is S.true
    assert Sum(2/(n*log(n)**2), (n, 2, oo)).is_convergent() is S.true
    assert Sum((n - 1)/(n**2*log(n)**3), (n, 2, oo)).is_convergent() is S.true
    assert Sum(1/(n*log(n)*log(log(n))), (n, 5, oo)).is_convergent() is S.false
    assert Sum((n - 1)/(n*log(n)**3), (n, 3, oo)).is_convergent() is S.false
    assert Sum(2/(n**2*log(n)), (n, 2, oo)).is_convergent() is S.true
    assert Sum(1/(n*sqrt(log(n))*log(log(n))), (n, 100, oo)).is_convergent() is S.false
    assert Sum(log(log(n))/(n*log(n)**2), (n, 100, oo)).is_convergent() is S.true
    assert Sum(log(n)/n**2, (n, 5, oo)).is_convergent() is S.true

    # alternating series tests --
    assert Sum((-1)**(n - 1)/(n**2 - 1), (n, 3, oo)).is_convergent() is S.true

    # with -negativeInfinite Limits
    assert Sum(1/(n**2 + 1), (n, -oo, 1)).is_convergent() is S.true
    assert Sum(1/(n - 1), (n, -oo, -1)).is_convergent() is S.false
    assert Sum(1/(n**2 - 1), (n, -oo, -5)).is_convergent() is S.true
    assert Sum(1/(n**2 - 1), (n, -oo, 2)).is_convergent() is S.true
    assert Sum(1/(n**2 - 1), (n, -oo, oo)).is_convergent() is S.true

    # piecewise functions
    f = Piecewise((n**(-2), n <= 1), (n**2, n > 1))
    assert Sum(f, (n, 1, oo)).is_convergent() is S.false
    assert Sum(f, (n, -oo, oo)).is_convergent() is S.false
    assert Sum(f, (n, 1, 100)).is_convergent() is S.true
    #assert Sum(f, (n, -oo, 1)).is_convergent() is S.true

    # integral test

    assert Sum(log(n)/n**3, (n, 1, oo)).is_convergent() is S.true
    assert Sum(-log(n)/n**3, (n, 1, oo)).is_convergent() is S.true
    # the following function has maxima located at (x, y) =
    # (1.2, 0.43), (3.0, -0.25) and (6.8, 0.050)
    eq = (x - 2)*(x**2 - 6*x + 4)*exp(-x)
    assert Sum(eq, (x, 1, oo)).is_convergent() is S.true
    assert Sum(eq, (x, 1, 2)).is_convergent() is S.true
    assert Sum(1/(x**3), (x, 1, oo)).is_convergent() is S.true
    assert Sum(1/(x**S.Half), (x, 1, oo)).is_convergent() is S.false

    # issue 19545
    assert Sum(1/n - 3/(3*n +2), (n, 1, oo)).is_convergent() is S.true

    # issue 19836
    assert Sum(4/(n + 2) - 5/(n + 1) + 1/n,(n, 7, oo)).is_convergent() is S.true


def test_is_absolutely_convergent():
    assert Sum((-1)**n, (n, 1, oo)).is_absolutely_convergent() is S.false
    assert Sum((-1)**n/n**2, (n, 1, oo)).is_absolutely_convergent() is S.true


@XFAIL
def test_convergent_failing():
    # dirichlet tests
    assert Sum(sin(n)/n, (n, 1, oo)).is_convergent() is S.true
    assert Sum(sin(2*n)/n, (n, 1, oo)).is_convergent() is S.true


def test_issue_6966():
    i, k, m = symbols('i k m', integer=True)
    z_i, q_i = symbols('z_i q_i')
    a_k = Sum(-q_i*z_i/k,(i,1,m))
    b_k = a_k.diff(z_i)
    assert isinstance(b_k, Sum)
    assert b_k == Sum(-q_i/k,(i,1,m))


def test_issue_10156():
    cx = Sum(2*y**2*x, (x, 1,3))
    e = 2*y*Sum(2*cx*x**2, (x, 1, 9))
    assert e.factor() == \
        8*y**3*Sum(x, (x, 1, 3))*Sum(x**2, (x, 1, 9))


def test_issue_10973():
    assert Sum((-n + (n**3 + 1)**(S(1)/3))/log(n), (n, 1, oo)).is_convergent() is S.true


def test_issue_14103():
    assert Sum(sin(n)**2 + cos(n)**2 - 1, (n, 1, oo)).is_convergent() is S.true
    assert Sum(sin(pi*n), (n, 1, oo)).is_convergent() is S.true


def test_issue_14129():
    x = Symbol('x', zero=False)
    assert Sum( k*x**k, (k, 0, n-1)).doit() == \
        Piecewise((n**2/2 - n/2, Eq(x, 1)), ((n*x*x**n -
            n*x**n - x*x**n + x)/(x - 1)**2, True))
    assert Sum( x**k, (k, 0, n-1)).doit() == \
        Piecewise((n, Eq(x, 1)), ((-x**n + 1)/(-x + 1), True))
    assert Sum( k*(x/y+x)**k, (k, 0, n-1)).doit() == \
        Piecewise((n*(n - 1)/2, Eq(x, y/(y + 1))),
        (x*(y + 1)*(n*x*y*(x + x/y)**(n - 1) +
        n*x*(x + x/y)**(n - 1) - n*y*(x + x/y)**(n - 1) -
        x*y*(x + x/y)**(n - 1) - x*(x + x/y)**(n - 1) + y)/
        (x*y + x - y)**2, True))


def test_issue_14112():
    assert Sum((-1)**n/sqrt(n), (n, 1, oo)).is_absolutely_convergent() is S.false
    assert Sum((-1)**(2*n)/n, (n, 1, oo)).is_convergent() is S.false
    assert Sum((-2)**n + (-3)**n, (n, 1, oo)).is_convergent() is S.false


def test_issue_14219():
    A = diag(0, 2, -3)
    res = diag(1, 15, -20)
    assert Sum(A**n, (n, 0, 3)).doit() == res


def test_sin_times_absolutely_convergent():
    assert Sum(sin(n) / n**3, (n, 1, oo)).is_convergent() is S.true
    assert Sum(sin(n) * log(n) / n**3, (n, 1, oo)).is_convergent() is S.true


def test_issue_14111():
    assert Sum(1/log(log(n)), (n, 22, oo)).is_convergent() is S.false


def test_issue_14484():
    assert Sum(sin(n)/log(log(n)), (n, 22, oo)).is_convergent() is S.false


def test_issue_14640():
    i, n = symbols("i n", integer=True)
    a, b, c = symbols("a b c", zero=False)

    assert Sum(a**-i/(a - b), (i, 0, n)).doit() == Sum(
        1/(a*a**i - a**i*b), (i, 0, n)).doit() == Piecewise(
            (n + 1, Eq(1/a, 1)),
            ((-a**(-n - 1) + 1)/(1 - 1/a), True))/(a - b)

    assert Sum((b*a**i - c*a**i)**-2, (i, 0, n)).doit() == Piecewise(
        (n + 1, Eq(a**(-2), 1)),
        ((-a**(-2*n - 2) + 1)/(1 - 1/a**2), True))/(b - c)**2

    s = Sum(i*(a**(n - i) - b**(n - i))/(a - b), (i, 0, n)).doit()
    assert not s.has(Sum)
    assert s.subs({a: 2, b: 3, n: 5}) == 122


def test_issue_15943():
    s = Sum(binomial(n, k)*factorial(n - k), (k, 0, n)).doit().rewrite(gamma)
    assert s == -E*(n + 1)*gamma(n + 1)*lowergamma(n + 1, 1)/gamma(n + 2
        ) + E*gamma(n + 1)
    assert s.simplify() == E*(factorial(n) - lowergamma(n + 1, 1))


def test_Sum_dummy_eq():
    assert not Sum(x, (x, a, b)).dummy_eq(1)
    assert not Sum(x, (x, a, b)).dummy_eq(Sum(x, (x, a, b), (a, 1, 2)))
    assert not Sum(x, (x, a, b)).dummy_eq(Sum(x, (x, a, c)))
    assert Sum(x, (x, a, b)).dummy_eq(Sum(x, (x, a, b)))
    d = Dummy()
    assert Sum(x, (x, a, d)).dummy_eq(Sum(x, (x, a, c)), c)
    assert not Sum(x, (x, a, d)).dummy_eq(Sum(x, (x, a, c)))
    assert Sum(x, (x, a, c)).dummy_eq(Sum(y, (y, a, c)))
    assert Sum(x, (x, a, d)).dummy_eq(Sum(y, (y, a, c)), c)
    assert not Sum(x, (x, a, d)).dummy_eq(Sum(y, (y, a, c)))


def test_issue_15852():
    assert summation(x**y*y, (y, -oo, oo)).doit() == Sum(x**y*y, (y, -oo, oo))


def test_exceptions():
    S = Sum(x, (x, a, b))
    raises(ValueError, lambda: S.change_index(x, x**2, y))
    S = Sum(x, (x, a, b), (x, 1, 4))
    raises(ValueError, lambda: S.index(x))
    S = Sum(x, (x, a, b), (y, 1, 4))
    raises(ValueError, lambda: S.reorder([x]))
    S = Sum(x, (x, y, b), (y, 1, 4))
    raises(ReorderError, lambda: S.reorder_limit(0, 1))
    S = Sum(x*y, (x, a, b), (y, 1, 4))
    raises(NotImplementedError, lambda: S.is_convergent())


def test_sumproducts_assumptions():
    M = Symbol('M', integer=True, positive=True)

    m = Symbol('m', integer=True)
    for func in [Sum, Product]:
        assert func(m, (m, -M, M)).is_positive is None
        assert func(m, (m, -M, M)).is_nonpositive is None
        assert func(m, (m, -M, M)).is_negative is None
        assert func(m, (m, -M, M)).is_nonnegative is None
        assert func(m, (m, -M, M)).is_finite is True

    m = Symbol('m', integer=True, nonnegative=True)
    for func in [Sum, Product]:
        assert func(m, (m, 0, M)).is_positive is None
        assert func(m, (m, 0, M)).is_nonpositive is None
        assert func(m, (m, 0, M)).is_negative is False
        assert func(m, (m, 0, M)).is_nonnegative is True
        assert func(m, (m, 0, M)).is_finite is True

    m = Symbol('m', integer=True, positive=True)
    for func in [Sum, Product]:
        assert func(m, (m, 1, M)).is_positive is True
        assert func(m, (m, 1, M)).is_nonpositive is False
        assert func(m, (m, 1, M)).is_negative is False
        assert func(m, (m, 1, M)).is_nonnegative is True
        assert func(m, (m, 1, M)).is_finite is True

    m = Symbol('m', integer=True, negative=True)
    assert Sum(m, (m, -M, -1)).is_positive is False
    assert Sum(m, (m, -M, -1)).is_nonpositive is True
    assert Sum(m, (m, -M, -1)).is_negative is True
    assert Sum(m, (m, -M, -1)).is_nonnegative is False
    assert Sum(m, (m, -M, -1)).is_finite is True
    assert Product(m, (m, -M, -1)).is_positive is None
    assert Product(m, (m, -M, -1)).is_nonpositive is None
    assert Product(m, (m, -M, -1)).is_negative is None
    assert Product(m, (m, -M, -1)).is_nonnegative is None
    assert Product(m, (m, -M, -1)).is_finite is True

    m = Symbol('m', integer=True, nonpositive=True)
    assert Sum(m, (m, -M, 0)).is_positive is False
    assert Sum(m, (m, -M, 0)).is_nonpositive is True
    assert Sum(m, (m, -M, 0)).is_negative is None
    assert Sum(m, (m, -M, 0)).is_nonnegative is None
    assert Sum(m, (m, -M, 0)).is_finite is True
    assert Product(m, (m, -M, 0)).is_positive is None
    assert Product(m, (m, -M, 0)).is_nonpositive is None
    assert Product(m, (m, -M, 0)).is_negative is None
    assert Product(m, (m, -M, 0)).is_nonnegative is None
    assert Product(m, (m, -M, 0)).is_finite is True

    m = Symbol('m', integer=True)
    assert Sum(2, (m, 0, oo)).is_positive is None
    assert Sum(2, (m, 0, oo)).is_nonpositive is None
    assert Sum(2, (m, 0, oo)).is_negative is None
    assert Sum(2, (m, 0, oo)).is_nonnegative is None
    assert Sum(2, (m, 0, oo)).is_finite is None

    assert Product(2, (m, 0, oo)).is_positive is None
    assert Product(2, (m, 0, oo)).is_nonpositive is None
    assert Product(2, (m, 0, oo)).is_negative is False
    assert Product(2, (m, 0, oo)).is_nonnegative is None
    assert Product(2, (m, 0, oo)).is_finite is None

    assert Product(0, (x, M, M-1)).is_positive is True
    assert Product(0, (x, M, M-1)).is_finite is True


def test_expand_with_assumptions():
    M = Symbol('M', integer=True, positive=True)
    x = Symbol('x', positive=True)
    m = Symbol('m', nonnegative=True)
    assert log(Product(x**m, (m, 0, M))).expand() == Sum(m*log(x), (m, 0, M))
    assert log(Product(exp(x**m), (m, 0, M))).expand() == Sum(x**m, (m, 0, M))
    assert log(Product(x**m, (m, 0, M))).rewrite(Sum).expand() == Sum(m*log(x), (m, 0, M))
    assert log(Product(exp(x**m), (m, 0, M))).rewrite(Sum).expand() == Sum(x**m, (m, 0, M))

    n = Symbol('n', nonnegative=True)
    i, j = symbols('i,j', positive=True, integer=True)
    x, y = symbols('x,y', positive=True)
    assert log(Product(x**i*y**j, (i, 1, n), (j, 1, m))).expand() \
        == Sum(i*log(x) + j*log(y), (i, 1, n), (j, 1, m))

    m = Symbol('m', nonnegative=True, integer=True)
    s = Sum(x**m, (m, 0, M))
    s_as_product = s.rewrite(Product)
    assert s_as_product.has(Product)
    assert s_as_product == log(Product(exp(x**m), (m, 0, M)))
    assert s_as_product.expand() == s
    s5 = s.subs(M, 5)
    s5_as_product = s5.rewrite(Product)
    assert s5_as_product.has(Product)
    assert s5_as_product.doit().expand() == s5.doit()


def test_has_finite_limits():
    x = Symbol('x')
    assert Sum(1, (x, 1, 9)).has_finite_limits is True
    assert Sum(1, (x, 1, oo)).has_finite_limits is False
    M = Symbol('M')
    assert Sum(1, (x, 1, M)).has_finite_limits is None
    M = Symbol('M', positive=True)
    assert Sum(1, (x, 1, M)).has_finite_limits is True
    x = Symbol('x', positive=True)
    M = Symbol('M')
    assert Sum(1, (x, 1, M)).has_finite_limits is True

    assert Sum(1, (x, 1, M), (y, -oo, oo)).has_finite_limits is False

def test_has_reversed_limits():
    assert Sum(1, (x, 1, 1)).has_reversed_limits is False
    assert Sum(1, (x, 1, 9)).has_reversed_limits is False
    assert Sum(1, (x, 1, -9)).has_reversed_limits is True
    assert Sum(1, (x, 1, 0)).has_reversed_limits is True
    assert Sum(1, (x, 1, oo)).has_reversed_limits is False
    M = Symbol('M')
    assert Sum(1, (x, 1, M)).has_reversed_limits is None
    M = Symbol('M', positive=True, integer=True)
    assert Sum(1, (x, 1, M)).has_reversed_limits is False
    assert Sum(1, (x, 1, M), (y, -oo, oo)).has_reversed_limits is False
    M = Symbol('M', negative=True)
    assert Sum(1, (x, 1, M)).has_reversed_limits is True

    assert Sum(1, (x, 1, M), (y, -oo, oo)).has_reversed_limits is True
    assert Sum(1, (x, oo, oo)).has_reversed_limits is None


def test_has_empty_sequence():
    assert Sum(1, (x, 1, 1)).has_empty_sequence is False
    assert Sum(1, (x, 1, 9)).has_empty_sequence is False
    assert Sum(1, (x, 1, -9)).has_empty_sequence is False
    assert Sum(1, (x, 1, 0)).has_empty_sequence is True
    assert Sum(1, (x, y, y - 1)).has_empty_sequence is True
    assert Sum(1, (x, 3, 2), (y, -oo, oo)).has_empty_sequence is True
    assert Sum(1, (y, -oo, oo), (x, 3, 2)).has_empty_sequence is True
    assert Sum(1, (x, oo, oo)).has_empty_sequence is False


def test_empty_sequence():
    assert Product(x*y, (x, -oo, oo), (y, 1, 0)).doit() == 1
    assert Product(x*y, (y, 1, 0), (x, -oo, oo)).doit() == 1
    assert Sum(x, (x, -oo, oo), (y, 1, 0)).doit() == 0
    assert Sum(x, (y, 1, 0), (x, -oo, oo)).doit() == 0


def test_issue_8016():
    k = Symbol('k', integer=True)
    n, m = symbols('n, m', integer=True, positive=True)
    s = Sum(binomial(m, k)*binomial(m, n - k)*(-1)**k, (k, 0, n))
    assert s.doit().simplify() == \
        cos(pi*n/2)*gamma(m + 1)/gamma(n/2 + 1)/gamma(m - n/2 + 1)


def test_issue_14313():
    assert Sum(S.Half**floor(n/2), (n, 1, oo)).is_convergent()


def test_issue_14563():
    # The assertion was failing due to no assumptions methods in Sums and Product
    assert 1 % Sum(1, (x, 0, 1)) == 1


def test_issue_16735():
    assert Sum(5**n/gamma(n+1), (n, 1, oo)).is_convergent() is S.true


def test_issue_14871():
    assert Sum((Rational(1, 10))**n*rf(0, n)/factorial(n), (n, 0, oo)).rewrite(factorial).doit() == 1


def test_issue_17165():
    n = symbols("n", integer=True)
    x = symbols('x')
    s = (x*Sum(x**n, (n, -1, oo)))
    ssimp = s.doit().simplify()

    assert ssimp == Piecewise((-1/(x - 1), (x > -1) & (x < 1)),
                              (x*Sum(x**n, (n, -1, oo)), True)), ssimp
    assert ssimp.simplify() == ssimp


def test_issue_19379():
    assert Sum(factorial(n)/factorial(n + 2), (n, 1, oo)).is_convergent() is S.true


def test_issue_20777():
    assert Sum(exp(x*sin(n/m)), (n, 1, m)).doit() == Sum(exp(x*sin(n/m)), (n, 1, m))


def test__dummy_with_inherited_properties_concrete():
    x = Symbol('x')

    from sympy.core.containers import Tuple
    d = _dummy_with_inherited_properties_concrete(Tuple(x, 0, 5))
    assert d.is_real
    assert d.is_integer
    assert d.is_nonnegative
    assert d.is_extended_nonnegative

    d = _dummy_with_inherited_properties_concrete(Tuple(x, 1, 9))
    assert d.is_real
    assert d.is_integer
    assert d.is_positive
    assert d.is_odd is None

    d = _dummy_with_inherited_properties_concrete(Tuple(x, -5, 5))
    assert d.is_real
    assert d.is_integer
    assert d.is_positive is None
    assert d.is_extended_nonnegative is None
    assert d.is_odd is None

    d = _dummy_with_inherited_properties_concrete(Tuple(x, -1.5, 1.5))
    assert d.is_real
    assert d.is_integer is None
    assert d.is_positive is None
    assert d.is_extended_nonnegative is None

    N = Symbol('N', integer=True, positive=True)
    d = _dummy_with_inherited_properties_concrete(Tuple(x, 2, N))
    assert d.is_real
    assert d.is_positive
    assert d.is_integer

    # Return None if no assumptions are added
    N = Symbol('N', integer=True, positive=True)
    d = _dummy_with_inherited_properties_concrete(Tuple(N, 2, 4))
    assert d is None

    x = Symbol('x', negative=True)
    raises(InconsistentAssumptions,
           lambda: _dummy_with_inherited_properties_concrete(Tuple(x, 1, 5)))


def test_matrixsymbol_summation_numerical_limits():
    A = MatrixSymbol('A', 3, 3)
    n = Symbol('n', integer=True)

    assert Sum(A**n, (n, 0, 2)).doit() == Identity(3) + A + A**2
    assert Sum(A, (n, 0, 2)).doit() == 3*A
    assert Sum(n*A, (n, 0, 2)).doit() == 3*A

    B = Matrix([[0, n, 0], [-1, 0, 0], [0, 0, 2]])
    ans = Matrix([[0, 6, 0], [-4, 0, 0], [0, 0, 8]]) + 4*A
    assert Sum(A+B, (n, 0, 3)).doit() == ans
    ans = A*Matrix([[0, 6, 0], [-4, 0, 0], [0, 0, 8]])
    assert Sum(A*B, (n, 0, 3)).doit() == ans

    ans = (A**2*Matrix([[-2, 0, 0], [0,-2, 0], [0, 0, 4]]) +
           A**3*Matrix([[0, -9, 0], [3, 0, 0], [0, 0, 8]]) +
           A*Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 2]]))
    assert Sum(A**n*B**n, (n, 1, 3)).doit() == ans


def test_issue_21651():
    i = Symbol('i')
    a = Sum(floor(2*2**(-i)), (i, S.One, 2))
    assert a.doit() == S.One


@XFAIL
def test_matrixsymbol_summation_symbolic_limits():
    N = Symbol('N', integer=True, positive=True)

    A = MatrixSymbol('A', 3, 3)
    n = Symbol('n', integer=True)
    assert Sum(A, (n, 0, N)).doit() == (N+1)*A
    assert Sum(n*A, (n, 0, N)).doit() == (N**2/2+N/2)*A


def test_summation_by_residues():
    x = Symbol('x')

    # Examples from Nakhle H. Asmar, Loukas Grafakos,
    # Complex Analysis with Applications
    assert eval_sum_residue(1 / (x**2 + 1), (x, -oo, oo)) == pi/tanh(pi)
    assert eval_sum_residue(1 / x**6, (x, S(1), oo)) == pi**6/945
    assert eval_sum_residue(1 / (x**2 + 9), (x, -oo, oo)) == pi/(3*tanh(3*pi))
    assert eval_sum_residue(1 / (x**2 + 1)**2, (x, -oo, oo)).cancel() == \
        (-pi**2*tanh(pi)**2 + pi*tanh(pi) + pi**2)/(2*tanh(pi)**2)
    assert eval_sum_residue(x**2 / (x**2 + 1)**2, (x, -oo, oo)).cancel() == \
        (-pi**2 + pi*tanh(pi) + pi**2*tanh(pi)**2)/(2*tanh(pi)**2)
    assert eval_sum_residue(1 / (4*x**2 - 1), (x, -oo, oo)) == 0
    assert eval_sum_residue(x**2 / (x**2 - S(1)/4)**2, (x, -oo, oo)) == pi**2/2
    assert eval_sum_residue(1 / (4*x**2 - 1)**2, (x, -oo, oo)) == pi**2/8
    assert eval_sum_residue(1 / ((x - S(1)/2)**2 + 1), (x, -oo, oo)) == pi*tanh(pi)
    assert eval_sum_residue(1 / x**2, (x, S(1), oo)) == pi**2/6
    assert eval_sum_residue(1 / x**4, (x, S(1), oo)) == pi**4/90
    assert eval_sum_residue(1 / x**2 / (x**2 + 4), (x, S(1), oo)) == \
        -pi*(-pi/12 - 1/(16*pi) + 1/(8*tanh(2*pi)))/2

    # Some examples made from 1 / (x**2 + 1)
    assert eval_sum_residue(1 / (x**2 + 1), (x, S(0), oo)) == \
        S(1)/2 + pi/(2*tanh(pi))
    assert eval_sum_residue(1 / (x**2 + 1), (x, S(1), oo)) == \
        -S(1)/2 + pi/(2*tanh(pi))
    assert eval_sum_residue(1 / (x**2 + 1), (x, S(-1), oo)) == \
        1 + pi/(2*tanh(pi))
    assert eval_sum_residue((-1)**x / (x**2 + 1), (x, -oo, oo)) == \
        pi/sinh(pi)
    assert eval_sum_residue((-1)**x / (x**2 + 1), (x, S(0), oo)) == \
        pi/(2*sinh(pi)) + S(1)/2
    assert eval_sum_residue((-1)**x / (x**2 + 1), (x, S(1), oo)) == \
        -S(1)/2 + pi/(2*sinh(pi))
    assert eval_sum_residue((-1)**x / (x**2 + 1), (x, S(-1), oo)) == \
        pi/(2*sinh(pi))

    # Some examples made from shifting of 1 / (x**2 + 1)
    assert eval_sum_residue(1 / (x**2 + 2*x + 2), (x, S(-1), oo)) == S(1)/2 + pi/(2*tanh(pi))
    assert eval_sum_residue(1 / (x**2 + 4*x + 5), (x, S(-2), oo)) == S(1)/2 + pi/(2*tanh(pi))
    assert eval_sum_residue(1 / (x**2 - 2*x + 2), (x, S(1), oo)) == S(1)/2 + pi/(2*tanh(pi))
    assert eval_sum_residue(1 / (x**2 - 4*x + 5), (x, S(2), oo)) == S(1)/2 + pi/(2*tanh(pi))
    assert eval_sum_residue((-1)**x * -1 / (x**2 + 2*x + 2), (x, S(-1), oo)) ==  S(1)/2 + pi/(2*sinh(pi))
    assert eval_sum_residue((-1)**x * -1 / (x**2 -2*x + 2), (x, S(1), oo)) == S(1)/2 + pi/(2*sinh(pi))

    # Some examples made from 1 / x**2
    assert eval_sum_residue(1 / x**2, (x, S(2), oo)) == -1 + pi**2/6
    assert eval_sum_residue(1 / x**2, (x, S(3), oo)) == -S(5)/4 + pi**2/6
    assert eval_sum_residue((-1)**x / x**2, (x, S(1), oo)) == -pi**2/12
    assert eval_sum_residue((-1)**x / x**2, (x, S(2), oo)) == 1 - pi**2/12


@slow
def test_summation_by_residues_failing():
    x = Symbol('x')

    # Failing because of the bug in residue computation
    assert eval_sum_residue(x**2 / (x**4 + 1), (x, S(1), oo))
    assert eval_sum_residue(1 / ((x - 1)*(x - 2) + 1), (x, -oo, oo)) != 0


def test_process_limits():
    from sympy.concrete.expr_with_limits import _process_limits

    # these should be (x, Range(3)) not Range(3)
    raises(ValueError, lambda: _process_limits(
        Range(3), discrete=True))
    raises(ValueError, lambda: _process_limits(
        Range(3), discrete=False))
    # these should be (x, union) not union
    # (but then we would get a TypeError because we don't
    # handle non-contiguous sets: see below use of `union`)
    union = Or(x < 1, x > 3).as_set()
    raises(ValueError, lambda: _process_limits(
        union, discrete=True))
    raises(ValueError, lambda: _process_limits(
        union, discrete=False))

    # error not triggered if not needed
    assert _process_limits((x, 1, 2)) == ([(x, 1, 2)], 1)

    # this equivalence is used to detect Reals in _process_limits
    assert isinstance(S.Reals, Interval)

    C = Integral  # continuous limits
    assert C(x, x >= 5) == C(x, (x, 5, oo))
    assert C(x, x < 3) == C(x, (x, -oo, 3))
    ans = C(x, (x, 0, 3))
    assert C(x, And(x >= 0, x < 3)) == ans
    assert C(x, (x, Interval.Ropen(0, 3))) == ans
    raises(TypeError, lambda: C(x, (x, Range(3))))

    # discrete limits
    for D in (Sum, Product):
        r, ans = Range(3, 10, 2), D(2*x + 3, (x, 0, 3))
        assert D(x, (x, r)) == ans
        assert D(x, (x, r.reversed)) == ans
        r, ans = Range(3, oo, 2), D(2*x + 3, (x, 0, oo))
        assert D(x, (x, r)) == ans
        assert D(x, (x, r.reversed)) == ans
        r, ans = Range(-oo, 5, 2), D(3 - 2*x, (x, 0, oo))
        assert D(x, (x, r)) == ans
        assert D(x, (x, r.reversed)) == ans
        raises(TypeError, lambda: D(x, x > 0))
        raises(ValueError, lambda: D(x, Interval(1, 3)))
        raises(NotImplementedError, lambda: D(x, (x, union)))


def test_pr_22677():
    b = Symbol('b', integer=True, positive=True)
    assert Sum(1/x**2,(x, 0, b)).doit() == Sum(x**(-2), (x, 0, b))
    assert Sum(1/(x - b)**2,(x, 0, b-1)).doit() == Sum(
        (-b + x)**(-2), (x, 0, b - 1))


def test_issue_23952():
    p, q = symbols("p q", real=True, nonnegative=True)
    k1, k2 = symbols("k1 k2", integer=True, nonnegative=True)
    n = Symbol("n", integer=True, positive=True)
    expr = Sum(abs(k1 - k2)*p**k1 *(1 - q)**(n - k2),
        (k1, 0, n), (k2, 0, n))
    assert expr.subs(p,0).subs(q,1).subs(n, 3).doit() == 3
