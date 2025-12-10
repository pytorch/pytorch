from sympy.core.expr import unchanged
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational as R, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.series.order import O
from sympy.simplify.radsimp import expand_numer
from sympy.core.function import (expand, expand_multinomial,
    expand_power_base, expand_log)

from sympy.testing.pytest import raises
from sympy.core.random import verify_numerically

from sympy.abc import x, y, z


def test_expand_no_log():
    assert (
        (1 + log(x**4))**2).expand(log=False) == 1 + 2*log(x**4) + log(x**4)**2
    assert ((1 + log(x**4))*(1 + log(x**3))).expand(
        log=False) == 1 + log(x**4) + log(x**3) + log(x**4)*log(x**3)


def test_expand_no_multinomial():
    assert ((1 + x)*(1 + (1 + x)**4)).expand(multinomial=False) == \
        1 + x + (1 + x)**4 + x*(1 + x)**4


def test_expand_negative_integer_powers():
    expr = (x + y)**(-2)
    assert expr.expand() == 1 / (2*x*y + x**2 + y**2)
    assert expr.expand(multinomial=False) == (x + y)**(-2)
    expr = (x + y)**(-3)
    assert expr.expand() == 1 / (3*x*x*y + 3*x*y*y + x**3 + y**3)
    assert expr.expand(multinomial=False) == (x + y)**(-3)
    expr = (x + y)**(2) * (x + y)**(-4)
    assert expr.expand() == 1 / (2*x*y + x**2 + y**2)
    assert expr.expand(multinomial=False) == (x + y)**(-2)


def test_expand_non_commutative():
    A = Symbol('A', commutative=False)
    B = Symbol('B', commutative=False)
    C = Symbol('C', commutative=False)
    a = Symbol('a')
    b = Symbol('b')
    i = Symbol('i', integer=True)
    n = Symbol('n', negative=True)
    m = Symbol('m', negative=True)
    p = Symbol('p', polar=True)
    np = Symbol('p', polar=False)

    assert (C*(A + B)).expand() == C*A + C*B
    assert (C*(A + B)).expand() != A*C + B*C
    assert ((A + B)**2).expand() == A**2 + A*B + B*A + B**2
    assert ((A + B)**3).expand() == (A**2*B + B**2*A + A*B**2 + B*A**2 +
                                     A**3 + B**3 + A*B*A + B*A*B)
    # issue 6219
    assert ((a*A*B*A**-1)**2).expand() == a**2*A*B**2/A
    # Note that (a*A*B*A**-1)**2 is automatically converted to a**2*(A*B*A**-1)**2
    assert ((a*A*B*A**-1)**2).expand(deep=False) == a**2*(A*B*A**-1)**2
    assert ((a*A*B*A**-1)**2).expand() == a**2*(A*B**2*A**-1)
    assert ((a*A*B*A**-1)**2).expand(force=True) == a**2*A*B**2*A**(-1)
    assert ((a*A*B)**2).expand() == a**2*A*B*A*B
    assert ((a*A)**2).expand() == a**2*A**2
    assert ((a*A*B)**i).expand() == a**i*(A*B)**i
    assert ((a*A*(B*(A*B/A)**2))**i).expand() == a**i*(A*B*A*B**2/A)**i
    # issue 6558
    assert (A*B*(A*B)**-1).expand() == 1
    assert ((a*A)**i).expand() == a**i*A**i
    assert ((a*A*B*A**-1)**3).expand() == a**3*A*B**3/A
    assert ((a*A*B*A*B/A)**3).expand() == \
        a**3*A*B*(A*B**2)*(A*B**2)*A*B*A**(-1)
    assert ((a*A*B*A*B/A)**-2).expand() == \
        A*B**-1*A**-1*B**-2*A**-1*B**-1*A**-1/a**2
    assert ((a*b*A*B*A**-1)**i).expand() == a**i*b**i*(A*B/A)**i
    assert ((a*(a*b)**i)**i).expand() == a**i*a**(i**2)*b**(i**2)
    e = Pow(Mul(a, 1/a, A, B, evaluate=False), S(2), evaluate=False)
    assert e.expand() == A*B*A*B
    assert sqrt(a*(A*b)**i).expand() == sqrt(a*b**i*A**i)
    assert (sqrt(-a)**a).expand() == sqrt(-a)**a
    assert expand((-2*n)**(i/3)) == 2**(i/3)*(-n)**(i/3)
    assert expand((-2*n*m)**(i/a)) == (-2)**(i/a)*(-n)**(i/a)*(-m)**(i/a)
    assert expand((-2*a*p)**b) == 2**b*p**b*(-a)**b
    assert expand((-2*a*np)**b) == 2**b*(-a*np)**b
    assert expand(sqrt(A*B)) == sqrt(A*B)
    assert expand(sqrt(-2*a*b)) == sqrt(2)*sqrt(-a*b)


def test_expand_radicals():
    a = (x + y)**R(1, 2)

    assert (a**1).expand() == a
    assert (a**3).expand() == x*a + y*a
    assert (a**5).expand() == x**2*a + 2*x*y*a + y**2*a

    assert (1/a**1).expand() == 1/a
    assert (1/a**3).expand() == 1/(x*a + y*a)
    assert (1/a**5).expand() == 1/(x**2*a + 2*x*y*a + y**2*a)

    a = (x + y)**R(1, 3)

    assert (a**1).expand() == a
    assert (a**2).expand() == a**2
    assert (a**4).expand() == x*a + y*a
    assert (a**5).expand() == x*a**2 + y*a**2
    assert (a**7).expand() == x**2*a + 2*x*y*a + y**2*a


def test_expand_modulus():
    assert ((x + y)**11).expand(modulus=11) == x**11 + y**11
    assert ((x + sqrt(2)*y)**11).expand(modulus=11) == x**11 + 10*sqrt(2)*y**11
    assert (x + y/2).expand(modulus=1) == y/2

    raises(ValueError, lambda: ((x + y)**11).expand(modulus=0))
    raises(ValueError, lambda: ((x + y)**11).expand(modulus=x))


def test_issue_5743():
    assert (x*sqrt(
        x + y)*(1 + sqrt(x + y))).expand() == x**2 + x*y + x*sqrt(x + y)
    assert (x*sqrt(
        x + y)*(1 + x*sqrt(x + y))).expand() == x**3 + x**2*y + x*sqrt(x + y)


def test_expand_frac():
    assert expand((x + y)*y/x/(x + 1), frac=True) == \
        (x*y + y**2)/(x**2 + x)
    assert expand((x + y)*y/x/(x + 1), numer=True) == \
        (x*y + y**2)/(x*(x + 1))
    assert expand((x + y)*y/x/(x + 1), denom=True) == \
        y*(x + y)/(x**2 + x)
    eq = (x + 1)**2/y
    assert expand_numer(eq, multinomial=False) == eq
    # issue 26329
    eq = (exp(x*z) - exp(y*z))/exp(z*(x + y))
    ans = exp(-y*z) - exp(-x*z)
    assert eq.expand(numer=True) != ans
    assert eq.expand(numer=True, exact=True) == ans
    assert expand_numer(eq) != ans
    assert expand_numer(eq, exact=True) == ans


def test_issue_6121():
    eq = -I*exp(-3*I*pi/4)/(4*pi**(S(3)/2)*sqrt(x))
    assert eq.expand(complex=True)  # does not give oo recursion
    eq = -I*exp(-3*I*pi/4)/(4*pi**(R(3, 2))*sqrt(x))
    assert eq.expand(complex=True)  # does not give oo recursion


def test_expand_power_base():
    assert expand_power_base((x*y*z)**4) == x**4*y**4*z**4
    assert expand_power_base((x*y*z)**x).is_Pow
    assert expand_power_base((x*y*z)**x, force=True) == x**x*y**x*z**x
    assert expand_power_base((x*(y*z)**2)**3) == x**3*y**6*z**6

    assert expand_power_base((sin((x*y)**2)*y)**z).is_Pow
    assert expand_power_base(
        (sin((x*y)**2)*y)**z, force=True) == sin((x*y)**2)**z*y**z
    assert expand_power_base(
        (sin((x*y)**2)*y)**z, deep=True) == (sin(x**2*y**2)*y)**z

    assert expand_power_base(exp(x)**2) == exp(2*x)
    assert expand_power_base((exp(x)*exp(y))**2) == exp(2*x)*exp(2*y)

    assert expand_power_base(
        (exp((x*y)**z)*exp(y))**2) == exp(2*(x*y)**z)*exp(2*y)
    assert expand_power_base((exp((x*y)**z)*exp(
        y))**2, deep=True, force=True) == exp(2*x**z*y**z)*exp(2*y)

    assert expand_power_base((exp(x)*exp(y))**z).is_Pow
    assert expand_power_base(
        (exp(x)*exp(y))**z, force=True) == exp(x)**z*exp(y)**z


def test_expand_arit():
    a = Symbol("a")
    b = Symbol("b", positive=True)
    c = Symbol("c")

    p = R(5)
    e = (a + b)*c
    assert e == c*(a + b)
    assert (e.expand() - a*c - b*c) == R(0)
    e = (a + b)*(a + b)
    assert e == (a + b)**2
    assert e.expand() == 2*a*b + a**2 + b**2
    e = (a + b)*(a + b)**R(2)
    assert e == (a + b)**3
    assert e.expand() == 3*b*a**2 + 3*a*b**2 + a**3 + b**3
    assert e.expand() == 3*b*a**2 + 3*a*b**2 + a**3 + b**3
    e = (a + b)*(a + c)*(b + c)
    assert e == (a + c)*(a + b)*(b + c)
    assert e.expand() == 2*a*b*c + b*a**2 + c*a**2 + b*c**2 + a*c**2 + c*b**2 + a*b**2
    e = (a + R(1))**p
    assert e == (1 + a)**5
    assert e.expand() == 1 + 5*a + 10*a**2 + 10*a**3 + 5*a**4 + a**5
    e = (a + b + c)*(a + c + p)
    assert e == (5 + a + c)*(a + b + c)
    assert e.expand() == 5*a + 5*b + 5*c + 2*a*c + b*c + a*b + a**2 + c**2
    x = Symbol("x")
    s = exp(x*x) - 1
    e = s.nseries(x, 0, 6)/x**2
    assert e.expand() == 1 + x**2/2 + O(x**4)

    e = (x*(y + z))**(x*(y + z))*(x + y)
    assert e.expand(power_exp=False, power_base=False) == x*(x*y + x*
                    z)**(x*y + x*z) + y*(x*y + x*z)**(x*y + x*z)
    assert e.expand(power_exp=False, power_base=False, deep=False) == x* \
        (x*(y + z))**(x*(y + z)) + y*(x*(y + z))**(x*(y + z))
    e = x * (x + (y + 1)**2)
    assert e.expand(deep=False) == x**2 + x*(y + 1)**2
    e = (x*(y + z))**z
    assert e.expand(power_base=True, mul=True, deep=True) in [x**z*(y +
                    z)**z, (x*y + x*z)**z]
    assert ((2*y)**z).expand() == 2**z*y**z
    p = Symbol('p', positive=True)
    assert sqrt(-x).expand().is_Pow
    assert sqrt(-x).expand(force=True) == I*sqrt(x)
    assert ((2*y*p)**z).expand() == 2**z*p**z*y**z
    assert ((2*y*p*x)**z).expand() == 2**z*p**z*(x*y)**z
    assert ((2*y*p*x)**z).expand(force=True) == 2**z*p**z*x**z*y**z
    assert ((2*y*p*-pi)**z).expand() == 2**z*pi**z*p**z*(-y)**z
    assert ((2*y*p*-pi*x)**z).expand() == 2**z*pi**z*p**z*(-x*y)**z
    n = Symbol('n', negative=True)
    m = Symbol('m', negative=True)
    assert ((-2*x*y*n)**z).expand() == 2**z*(-n)**z*(x*y)**z
    assert ((-2*x*y*n*m)**z).expand() == 2**z*(-m)**z*(-n)**z*(-x*y)**z
    # issue 5482
    assert sqrt(-2*x*n) == sqrt(2)*sqrt(-n)*sqrt(x)
    # issue 5605 (2)
    assert (cos(x + y)**2).expand(trig=True) in [
        (-sin(x)*sin(y) + cos(x)*cos(y))**2,
        sin(x)**2*sin(y)**2 - 2*sin(x)*sin(y)*cos(x)*cos(y) + cos(x)**2*cos(y)**2
    ]

    # Check that this isn't too slow
    x = Symbol('x')
    W = 1
    for i in range(1, 21):
        W = W * (x - i)
    W = W.expand()
    assert W.has(-1672280820*x**15)

def test_expand_mul():
    # part of issue 20597
    e = Mul(2, 3, evaluate=False)
    assert e.expand() == 6

    e = Mul(2, 3, 1/x, evaluate=False)
    assert e.expand() == 6/x
    e = Mul(2, R(1, 3), evaluate=False)
    assert e.expand() == R(2, 3)

def test_power_expand():
    """Test for Pow.expand()"""
    a = Symbol('a')
    b = Symbol('b')
    p = (a + b)**2
    assert p.expand() == a**2 + b**2 + 2*a*b

    p = (1 + 2*(1 + a))**2
    assert p.expand() == 9 + 4*(a**2) + 12*a

    p = 2**(a + b)
    assert p.expand() == 2**a*2**b

    A = Symbol('A', commutative=False)
    B = Symbol('B', commutative=False)
    assert (2**(A + B)).expand() == 2**(A + B)
    assert (A**(a + b)).expand() != A**(a + b)


def test_issues_5919_6830():
    # issue 5919
    n = -1 + 1/x
    z = n/x/(-n)**2 - 1/n/x
    assert expand(z) == 1/(x**2 - 2*x + 1) - 1/(x - 2 + 1/x) - 1/(-x + 1)

    # issue 6830
    p = (1 + x)**2
    assert expand_multinomial((1 + x*p)**2) == (
        x**2*(x**4 + 4*x**3 + 6*x**2 + 4*x + 1) + 2*x*(x**2 + 2*x + 1) + 1)
    assert expand_multinomial((1 + (y + x)*p)**2) == (
        2*((x + y)*(x**2 + 2*x + 1)) + (x**2 + 2*x*y + y**2)*
        (x**4 + 4*x**3 + 6*x**2 + 4*x + 1) + 1)
    A = Symbol('A', commutative=False)
    p = (1 + A)**2
    assert expand_multinomial((1 + x*p)**2) == (
        x**2*(1 + 4*A + 6*A**2 + 4*A**3 + A**4) + 2*x*(1 + 2*A + A**2) + 1)
    assert expand_multinomial((1 + (y + x)*p)**2) == (
        (x + y)*(1 + 2*A + A**2)*2 + (x**2 + 2*x*y + y**2)*
        (1 + 4*A + 6*A**2 + 4*A**3 + A**4) + 1)
    assert expand_multinomial((1 + (y + x)*p)**3) == (
        (x + y)*(1 + 2*A + A**2)*3 + (x**2 + 2*x*y + y**2)*(1 + 4*A +
        6*A**2 + 4*A**3 + A**4)*3 + (x**3 + 3*x**2*y + 3*x*y**2 + y**3)*(1 + 6*A
        + 15*A**2 + 20*A**3 + 15*A**4 + 6*A**5 + A**6) + 1)
    # unevaluate powers
    eq = (Pow((x + 1)*((A + 1)**2), 2, evaluate=False))
    # - in this case the base is not an Add so no further
    #   expansion is done
    assert expand_multinomial(eq) == \
        (x**2 + 2*x + 1)*(1 + 4*A + 6*A**2 + 4*A**3 + A**4)
    # - but here, the expanded base *is* an Add so it gets expanded
    eq = (Pow(((A + 1)**2), 2, evaluate=False))
    assert expand_multinomial(eq) == 1 + 4*A + 6*A**2 + 4*A**3 + A**4

    # coverage
    def ok(a, b, n):
        e = (a + I*b)**n
        return verify_numerically(e, expand_multinomial(e))

    for a in [2, S.Half]:
        for b in [3, R(1, 3)]:
            for n in range(2, 6):
                assert ok(a, b, n)

    assert expand_multinomial((x + 1 + O(z))**2) == \
        1 + 2*x + x**2 + O(z)
    assert expand_multinomial((x + 1 + O(z))**3) == \
        1 + 3*x + 3*x**2 + x**3 + O(z)

    assert expand_multinomial(3**(x + y + 3)) == 27*3**(x + y)

def test_expand_log():
    t = Symbol('t', positive=True)
    # after first expansion, -2*log(2) + log(4); then 0 after second
    assert expand(log(t**2) - log(t**2/4) - 2*log(2)) == 0
    assert expand_log(log(7*6)/log(6)) == 1 + log(7)/log(6)
    b = factorial(10)
    assert expand_log(log(7*b**4)/log(b)
        ) == 4 + log(7)/log(b)


def test_issue_23952():
    assert (x**(y + z)).expand(force=True) == x**y*x**z
    one = Symbol('1', integer=True, prime=True, odd=True, positive=True)
    two = Symbol('2', integer=True, prime=True, even=True)
    e = two - one
    for b in (0, x):
        # 0**e = 0, 0**-e = zoo; but if expanded then nan
        assert unchanged(Pow, b, e)  # power_exp
        assert unchanged(Pow, b, -e)  # power_exp
        assert unchanged(Pow, b, y - x)  # power_exp
        assert unchanged(Pow, b, 3 - x)  # multinomial
        assert (b**e).expand().is_Pow  # power_exp
        assert (b**-e).expand().is_Pow  # power_exp
        assert (b**(y - x)).expand().is_Pow  # power_exp
        assert (b**(3 - x)).expand().is_Pow  # multinomial
    nn1 = Symbol('nn1', nonnegative=True)
    nn2 = Symbol('nn2', nonnegative=True)
    nn3 = Symbol('nn3', nonnegative=True)
    assert (x**(nn1 + nn2)).expand() == x**nn1*x**nn2
    assert (x**(-nn1 - nn2)).expand() == x**-nn1*x**-nn2
    assert unchanged(Pow, x, nn1 + nn2 - nn3)
    assert unchanged(Pow, x, 1 + nn2 - nn3)
    assert unchanged(Pow, x, nn1 - nn2)
    assert unchanged(Pow, x, 1 - nn2)
    assert unchanged(Pow, x, -1 + nn2)
