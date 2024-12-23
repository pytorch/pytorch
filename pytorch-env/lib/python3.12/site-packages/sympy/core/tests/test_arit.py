from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, comp, nan,
    oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (im, re, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import (Max, sqrt)
from sympy.functions.elementary.trigonometric import (atan, cos, sin)
from sympy.polys.polytools import Poly
from sympy.sets.sets import FiniteSet

from sympy.core.parameters import distribute, evaluate
from sympy.core.expr import unchanged
from sympy.utilities.iterables import permutations
from sympy.testing.pytest import XFAIL, raises, warns
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.random import verify_numerically
from sympy.functions.elementary.trigonometric import asin

from itertools import product

a, c, x, y, z = symbols('a,c,x,y,z')
b = Symbol("b", positive=True)


def same_and_same_prec(a, b):
    # stricter matching for Floats
    return a == b and a._prec == b._prec


def test_bug1():
    assert re(x) != x
    x.series(x, 0, 1)
    assert re(x) != x


def test_Symbol():
    e = a*b
    assert e == a*b
    assert a*b*b == a*b**2
    assert a*b*b + c == c + a*b**2
    assert a*b*b - c == -c + a*b**2

    x = Symbol('x', complex=True, real=False)
    assert x.is_imaginary is None  # could be I or 1 + I
    x = Symbol('x', complex=True, imaginary=False)
    assert x.is_real is None  # could be 1 or 1 + I
    x = Symbol('x', real=True)
    assert x.is_complex
    x = Symbol('x', imaginary=True)
    assert x.is_complex
    x = Symbol('x', real=False, imaginary=False)
    assert x.is_complex is None  # might be a non-number


def test_arit0():
    p = Rational(5)
    e = a*b
    assert e == a*b
    e = a*b + b*a
    assert e == 2*a*b
    e = a*b + b*a + a*b + p*b*a
    assert e == 8*a*b
    e = a*b + b*a + a*b + p*b*a + a
    assert e == a + 8*a*b
    e = a + a
    assert e == 2*a
    e = a + b + a
    assert e == b + 2*a
    e = a + b*b + a + b*b
    assert e == 2*a + 2*b**2
    e = a + Rational(2) + b*b + a + b*b + p
    assert e == 7 + 2*a + 2*b**2
    e = (a + b*b + a + b*b)*p
    assert e == 5*(2*a + 2*b**2)
    e = (a*b*c + c*b*a + b*a*c)*p
    assert e == 15*a*b*c
    e = (a*b*c + c*b*a + b*a*c)*p - Rational(15)*a*b*c
    assert e == Rational(0)
    e = Rational(50)*(a - a)
    assert e == Rational(0)
    e = b*a - b - a*b + b
    assert e == Rational(0)
    e = a*b + c**p
    assert e == a*b + c**5
    e = a/b
    assert e == a*b**(-1)
    e = a*2*2
    assert e == 4*a
    e = 2 + a*2/2
    assert e == 2 + a
    e = 2 - a - 2
    assert e == -a
    e = 2*a*2
    assert e == 4*a
    e = 2/a/2
    assert e == a**(-1)
    e = 2**a**2
    assert e == 2**(a**2)
    e = -(1 + a)
    assert e == -1 - a
    e = S.Half*(1 + a)
    assert e == S.Half + a/2


def test_div():
    e = a/b
    assert e == a*b**(-1)
    e = a/b + c/2
    assert e == a*b**(-1) + Rational(1)/2*c
    e = (1 - b)/(b - 1)
    assert e == (1 + -b)*((-1) + b)**(-1)


def test_pow_arit():
    n1 = Rational(1)
    n2 = Rational(2)
    n5 = Rational(5)
    e = a*a
    assert e == a**2
    e = a*a*a
    assert e == a**3
    e = a*a*a*a**Rational(6)
    assert e == a**9
    e = a*a*a*a**Rational(6) - a**Rational(9)
    assert e == Rational(0)
    e = a**(b - b)
    assert e == Rational(1)
    e = (a + Rational(1) - a)**b
    assert e == Rational(1)

    e = (a + b + c)**n2
    assert e == (a + b + c)**2
    assert e.expand() == 2*b*c + 2*a*c + 2*a*b + a**2 + c**2 + b**2

    e = (a + b)**n2
    assert e == (a + b)**2
    assert e.expand() == 2*a*b + a**2 + b**2

    e = (a + b)**(n1/n2)
    assert e == sqrt(a + b)
    assert e.expand() == sqrt(a + b)

    n = n5**(n1/n2)
    assert n == sqrt(5)
    e = n*a*b - n*b*a
    assert e == Rational(0)
    e = n*a*b + n*b*a
    assert e == 2*a*b*sqrt(5)
    assert e.diff(a) == 2*b*sqrt(5)
    assert e.diff(a) == 2*b*sqrt(5)
    e = a/b**2
    assert e == a*b**(-2)

    assert sqrt(2*(1 + sqrt(2))) == (2*(1 + 2**S.Half))**S.Half

    x = Symbol('x')
    y = Symbol('y')

    assert ((x*y)**3).expand() == y**3 * x**3
    assert ((x*y)**-3).expand() == y**-3 * x**-3

    assert (x**5*(3*x)**(3)).expand() == 27 * x**8
    assert (x**5*(-3*x)**(3)).expand() == -27 * x**8
    assert (x**5*(3*x)**(-3)).expand() == x**2 * Rational(1, 27)
    assert (x**5*(-3*x)**(-3)).expand() == x**2 * Rational(-1, 27)

    # expand_power_exp
    _x = Symbol('x', zero=False)
    _y = Symbol('y', zero=False)
    assert (_x**(y**(x + exp(x + y)) + z)).expand(deep=False) == \
        _x**z*_x**(y**(x + exp(x + y)))
    assert (_x**(_y**(x + exp(x + y)) + z)).expand() == \
        _x**z*_x**(_y**x*_y**(exp(x)*exp(y)))

    n = Symbol('n', even=False)
    k = Symbol('k', even=True)
    o = Symbol('o', odd=True)

    assert unchanged(Pow, -1, x)
    assert unchanged(Pow, -1, n)
    assert (-2)**k == 2**k
    assert (-1)**k == 1
    assert (-1)**o == -1


def test_pow2():
    # x**(2*y) is always (x**y)**2 but is only (x**2)**y if
    #                                  x.is_positive or y.is_integer
    # let x = 1 to see why the following are not true.
    assert (-x)**Rational(2, 3) != x**Rational(2, 3)
    assert (-x)**Rational(5, 7) != -x**Rational(5, 7)
    assert ((-x)**2)**Rational(1, 3) != ((-x)**Rational(1, 3))**2
    assert sqrt(x**2) != x


def test_pow3():
    assert sqrt(2)**3 == 2 * sqrt(2)
    assert sqrt(2)**3 == sqrt(8)


def test_mod_pow():
    for s, t, u, v in [(4, 13, 497, 445), (4, -3, 497, 365),
            (3.2, 2.1, 1.9, 0.1031015682350942), (S(3)/2, 5, S(5)/6, S(3)/32)]:
        assert pow(S(s), t, u) == v
        assert pow(S(s), S(t), u) == v
        assert pow(S(s), t, S(u)) == v
        assert pow(S(s), S(t), S(u)) == v
    assert pow(S(2), S(10000000000), S(3)) == 1
    assert pow(x, y, z) == x**y%z
    raises(TypeError, lambda: pow(S(4), "13", 497))
    raises(TypeError, lambda: pow(S(4), 13, "497"))


def test_pow_E():
    assert 2**(y/log(2)) == S.Exp1**y
    assert 2**(y/log(2)/3) == S.Exp1**(y/3)
    assert 3**(1/log(-3)) != S.Exp1
    assert (3 + 2*I)**(1/(log(-3 - 2*I) + I*pi)) == S.Exp1
    assert (4 + 2*I)**(1/(log(-4 - 2*I) + I*pi)) == S.Exp1
    assert (3 + 2*I)**(1/(log(-3 - 2*I, 3)/2 + I*pi/log(3)/2)) == 9
    assert (3 + 2*I)**(1/(log(3 + 2*I, 3)/2)) == 9
    # every time tests are run they will affirm with a different random
    # value that this identity holds
    while 1:
        b = x._random()
        r, i = b.as_real_imag()
        if i:
            break
    assert verify_numerically(b**(1/(log(-b) + sign(i)*I*pi).n()), S.Exp1)


def test_pow_issue_3516():
    assert 4**Rational(1, 4) == sqrt(2)


def test_pow_im():
    for m in (-2, -1, 2):
        for d in (3, 4, 5):
            b = m*I
            for i in range(1, 4*d + 1):
                e = Rational(i, d)
                assert (b**e - b.n()**e.n()).n(2, chop=1e-10) == 0

    e = Rational(7, 3)
    assert (2*x*I)**e == 4*2**Rational(1, 3)*(I*x)**e  # same as Wolfram Alpha
    im = symbols('im', imaginary=True)
    assert (2*im*I)**e == 4*2**Rational(1, 3)*(I*im)**e

    args = [I, I, I, I, 2]
    e = Rational(1, 3)
    ans = 2**e
    assert Mul(*args, evaluate=False)**e == ans
    assert Mul(*args)**e == ans
    args = [I, I, I, 2]
    e = Rational(1, 3)
    ans = 2**e*(-I)**e
    assert Mul(*args, evaluate=False)**e == ans
    assert Mul(*args)**e == ans
    args.append(-3)
    ans = (6*I)**e
    assert Mul(*args, evaluate=False)**e == ans
    assert Mul(*args)**e == ans
    args.append(-1)
    ans = (-6*I)**e
    assert Mul(*args, evaluate=False)**e == ans
    assert Mul(*args)**e == ans

    args = [I, I, 2]
    e = Rational(1, 3)
    ans = (-2)**e
    assert Mul(*args, evaluate=False)**e == ans
    assert Mul(*args)**e == ans
    args.append(-3)
    ans = (6)**e
    assert Mul(*args, evaluate=False)**e == ans
    assert Mul(*args)**e == ans
    args.append(-1)
    ans = (-6)**e
    assert Mul(*args, evaluate=False)**e == ans
    assert Mul(*args)**e == ans
    assert Mul(Pow(-1, Rational(3, 2), evaluate=False), I, I) == I
    assert Mul(I*Pow(I, S.Half, evaluate=False)) == sqrt(I)*I


def test_real_mul():
    assert Float(0) * pi * x == 0
    assert set((Float(1) * pi * x).args) == {Float(1), pi, x}


def test_ncmul():
    A = Symbol("A", commutative=False)
    B = Symbol("B", commutative=False)
    C = Symbol("C", commutative=False)
    assert A*B != B*A
    assert A*B*C != C*B*A
    assert A*b*B*3*C == 3*b*A*B*C
    assert A*b*B*3*C != 3*b*B*A*C
    assert A*b*B*3*C == 3*A*B*C*b

    assert A + B == B + A
    assert (A + B)*C != C*(A + B)

    assert C*(A + B)*C != C*C*(A + B)

    assert A*A == A**2
    assert (A + B)*(A + B) == (A + B)**2

    assert A**-1 * A == 1
    assert A/A == 1
    assert A/(A**2) == 1/A

    assert A/(1 + A) == A/(1 + A)

    assert set((A + B + 2*(A + B)).args) == \
        {A, B, 2*(A + B)}


def test_mul_add_identity():
    m = Mul(1, 2)
    assert isinstance(m, Rational) and m.p == 2 and m.q == 1
    m = Mul(1, 2, evaluate=False)
    assert isinstance(m, Mul) and m.args == (1, 2)
    m = Mul(0, 1)
    assert m is S.Zero
    m = Mul(0, 1, evaluate=False)
    assert isinstance(m, Mul) and m.args == (0, 1)
    m = Add(0, 1)
    assert m is S.One
    m = Add(0, 1, evaluate=False)
    assert isinstance(m, Add) and m.args == (0, 1)


def test_ncpow():
    x = Symbol('x', commutative=False)
    y = Symbol('y', commutative=False)
    z = Symbol('z', commutative=False)
    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')

    assert (x**2)*(y**2) != (y**2)*(x**2)
    assert (x**-2)*y != y*(x**2)
    assert 2**x*2**y != 2**(x + y)
    assert 2**x*2**y*2**z != 2**(x + y + z)
    assert 2**x*2**(2*x) == 2**(3*x)
    assert 2**x*2**(2*x)*2**x == 2**(4*x)
    assert exp(x)*exp(y) != exp(y)*exp(x)
    assert exp(x)*exp(y)*exp(z) != exp(y)*exp(x)*exp(z)
    assert exp(x)*exp(y)*exp(z) != exp(x + y + z)
    assert x**a*x**b != x**(a + b)
    assert x**a*x**b*x**c != x**(a + b + c)
    assert x**3*x**4 == x**7
    assert x**3*x**4*x**2 == x**9
    assert x**a*x**(4*a) == x**(5*a)
    assert x**a*x**(4*a)*x**a == x**(6*a)


def test_powerbug():
    x = Symbol("x")
    assert x**1 != (-x)**1
    assert x**2 == (-x)**2
    assert x**3 != (-x)**3
    assert x**4 == (-x)**4
    assert x**5 != (-x)**5
    assert x**6 == (-x)**6

    assert x**128 == (-x)**128
    assert x**129 != (-x)**129

    assert (2*x)**2 == (-2*x)**2


def test_Mul_doesnt_expand_exp():
    x = Symbol('x')
    y = Symbol('y')
    assert unchanged(Mul, exp(x), exp(y))
    assert unchanged(Mul, 2**x, 2**y)
    assert x**2*x**3 == x**5
    assert 2**x*3**x == 6**x
    assert x**(y)*x**(2*y) == x**(3*y)
    assert sqrt(2)*sqrt(2) == 2
    assert 2**x*2**(2*x) == 2**(3*x)
    assert sqrt(2)*2**Rational(1, 4)*5**Rational(3, 4) == 10**Rational(3, 4)
    assert (x**(-log(5)/log(3))*x)/(x*x**( - log(5)/log(3))) == sympify(1)


def test_Mul_is_integer():
    k = Symbol('k', integer=True)
    n = Symbol('n', integer=True)
    nr = Symbol('nr', rational=False)
    ir = Symbol('ir', irrational=True)
    nz = Symbol('nz', integer=True, zero=False)
    e = Symbol('e', even=True)
    o = Symbol('o', odd=True)
    i2 = Symbol('2', prime=True, even=True)

    assert (k/3).is_integer is None
    assert (nz/3).is_integer is None
    assert (nr/3).is_integer is False
    assert (ir/3).is_integer is False
    assert (x*k*n).is_integer is None
    assert (e/2).is_integer is True
    assert (e**2/2).is_integer is True
    assert (2/k).is_integer is None
    assert (2/k**2).is_integer is None
    assert ((-1)**k*n).is_integer is True
    assert (3*k*e/2).is_integer is True
    assert (2*k*e/3).is_integer is None
    assert (e/o).is_integer is None
    assert (o/e).is_integer is False
    assert (o/i2).is_integer is False
    assert Mul(k, 1/k, evaluate=False).is_integer is None
    assert Mul(2., S.Half, evaluate=False).is_integer is None
    assert (2*sqrt(k)).is_integer is None
    assert (2*k**n).is_integer is None

    s = 2**2**2**Pow(2, 1000, evaluate=False)
    m = Mul(s, s, evaluate=False)
    assert m.is_integer

    # broken in 1.6 and before, see #20161
    xq = Symbol('xq', rational=True)
    yq = Symbol('yq', rational=True)
    assert (xq*yq).is_integer is None
    e_20161 = Mul(-1,Mul(1,Pow(2,-1,evaluate=False),evaluate=False),evaluate=False)
    assert e_20161.is_integer is not True # expand(e_20161) -> -1/2, but no need to see that in the assumption without evaluation


def test_Add_Mul_is_integer():
    x = Symbol('x')

    k = Symbol('k', integer=True)
    n = Symbol('n', integer=True)
    nk = Symbol('nk', integer=False)
    nr = Symbol('nr', rational=False)
    nz = Symbol('nz', integer=True, zero=False)

    assert (-nk).is_integer is None
    assert (-nr).is_integer is False
    assert (2*k).is_integer is True
    assert (-k).is_integer is True

    assert (k + nk).is_integer is False
    assert (k + n).is_integer is True
    assert (k + x).is_integer is None
    assert (k + n*x).is_integer is None
    assert (k + n/3).is_integer is None
    assert (k + nz/3).is_integer is None
    assert (k + nr/3).is_integer is False

    assert ((1 + sqrt(3))*(-sqrt(3) + 1)).is_integer is not False
    assert (1 + (1 + sqrt(3))*(-sqrt(3) + 1)).is_integer is not False


def test_Add_Mul_is_finite():
    x = Symbol('x', extended_real=True, finite=False)

    assert sin(x).is_finite is True
    assert (x*sin(x)).is_finite is None
    assert (x*atan(x)).is_finite is False
    assert (1024*sin(x)).is_finite is True
    assert (sin(x)*exp(x)).is_finite is None
    assert (sin(x)*cos(x)).is_finite is True
    assert (x*sin(x)*exp(x)).is_finite is None

    assert (sin(x) - 67).is_finite is True
    assert (sin(x) + exp(x)).is_finite is not True
    assert (1 + x).is_finite is False
    assert (1 + x**2 + (1 + x)*(1 - x)).is_finite is None
    assert (sqrt(2)*(1 + x)).is_finite is False
    assert (sqrt(2)*(1 + x)*(1 - x)).is_finite is False


def test_Mul_is_even_odd():
    x = Symbol('x', integer=True)
    y = Symbol('y', integer=True)

    k = Symbol('k', odd=True)
    n = Symbol('n', odd=True)
    m = Symbol('m', even=True)

    assert (2*x).is_even is True
    assert (2*x).is_odd is False

    assert (3*x).is_even is None
    assert (3*x).is_odd is None

    assert (k/3).is_integer is None
    assert (k/3).is_even is None
    assert (k/3).is_odd is None

    assert (2*n).is_even is True
    assert (2*n).is_odd is False

    assert (2*m).is_even is True
    assert (2*m).is_odd is False

    assert (-n).is_even is False
    assert (-n).is_odd is True

    assert (k*n).is_even is False
    assert (k*n).is_odd is True

    assert (k*m).is_even is True
    assert (k*m).is_odd is False

    assert (k*n*m).is_even is True
    assert (k*n*m).is_odd is False

    assert (k*m*x).is_even is True
    assert (k*m*x).is_odd is False

    # issue 6791:
    assert (x/2).is_integer is None
    assert (k/2).is_integer is False
    assert (m/2).is_integer is True

    assert (x*y).is_even is None
    assert (x*x).is_even is None
    assert (x*(x + k)).is_even is True
    assert (x*(x + m)).is_even is None

    assert (x*y).is_odd is None
    assert (x*x).is_odd is None
    assert (x*(x + k)).is_odd is False
    assert (x*(x + m)).is_odd is None

    # issue 8648
    assert (m**2/2).is_even
    assert (m**2/3).is_even is False
    assert (2/m**2).is_odd is False
    assert (2/m).is_odd is None


@XFAIL
def test_evenness_in_ternary_integer_product_with_odd():
    # Tests that oddness inference is independent of term ordering.
    # Term ordering at the point of testing depends on SymPy's symbol order, so
    # we try to force a different order by modifying symbol names.
    x = Symbol('x', integer=True)
    y = Symbol('y', integer=True)
    k = Symbol('k', odd=True)
    assert (x*y*(y + k)).is_even is True
    assert (y*x*(x + k)).is_even is True


def test_evenness_in_ternary_integer_product_with_even():
    x = Symbol('x', integer=True)
    y = Symbol('y', integer=True)
    m = Symbol('m', even=True)
    assert (x*y*(y + m)).is_even is None


@XFAIL
def test_oddness_in_ternary_integer_product_with_odd():
    # Tests that oddness inference is independent of term ordering.
    # Term ordering at the point of testing depends on SymPy's symbol order, so
    # we try to force a different order by modifying symbol names.
    x = Symbol('x', integer=True)
    y = Symbol('y', integer=True)
    k = Symbol('k', odd=True)
    assert (x*y*(y + k)).is_odd is False
    assert (y*x*(x + k)).is_odd is False


def test_oddness_in_ternary_integer_product_with_even():
    x = Symbol('x', integer=True)
    y = Symbol('y', integer=True)
    m = Symbol('m', even=True)
    assert (x*y*(y + m)).is_odd is None


def test_Mul_is_rational():
    x = Symbol('x')
    n = Symbol('n', integer=True)
    m = Symbol('m', integer=True, nonzero=True)

    assert (n/m).is_rational is True
    assert (x/pi).is_rational is None
    assert (x/n).is_rational is None
    assert (m/pi).is_rational is False

    r = Symbol('r', rational=True)
    assert (pi*r).is_rational is None

    # issue 8008
    z = Symbol('z', zero=True)
    i = Symbol('i', imaginary=True)
    assert (z*i).is_rational is True
    bi = Symbol('i', imaginary=True, finite=True)
    assert (z*bi).is_zero is True


def test_Add_is_rational():
    x = Symbol('x')
    n = Symbol('n', rational=True)
    m = Symbol('m', rational=True)

    assert (n + m).is_rational is True
    assert (x + pi).is_rational is None
    assert (x + n).is_rational is None
    assert (n + pi).is_rational is False


def test_Add_is_even_odd():
    x = Symbol('x', integer=True)

    k = Symbol('k', odd=True)
    n = Symbol('n', odd=True)
    m = Symbol('m', even=True)

    assert (k + 7).is_even is True
    assert (k + 7).is_odd is False

    assert (-k + 7).is_even is True
    assert (-k + 7).is_odd is False

    assert (k - 12).is_even is False
    assert (k - 12).is_odd is True

    assert (-k - 12).is_even is False
    assert (-k - 12).is_odd is True

    assert (k + n).is_even is True
    assert (k + n).is_odd is False

    assert (k + m).is_even is False
    assert (k + m).is_odd is True

    assert (k + n + m).is_even is True
    assert (k + n + m).is_odd is False

    assert (k + n + x + m).is_even is None
    assert (k + n + x + m).is_odd is None


def test_Mul_is_negative_positive():
    x = Symbol('x', real=True)
    y = Symbol('y', extended_real=False, complex=True)
    z = Symbol('z', zero=True)

    e = 2*z
    assert e.is_Mul and e.is_positive is False and e.is_negative is False

    neg = Symbol('neg', negative=True)
    pos = Symbol('pos', positive=True)
    nneg = Symbol('nneg', nonnegative=True)
    npos = Symbol('npos', nonpositive=True)

    assert neg.is_negative is True
    assert (-neg).is_negative is False
    assert (2*neg).is_negative is True

    assert (2*pos)._eval_is_extended_negative() is False
    assert (2*pos).is_negative is False

    assert pos.is_negative is False
    assert (-pos).is_negative is True
    assert (2*pos).is_negative is False

    assert (pos*neg).is_negative is True
    assert (2*pos*neg).is_negative is True
    assert (-pos*neg).is_negative is False
    assert (pos*neg*y).is_negative is False     # y.is_real=F;  !real -> !neg

    assert nneg.is_negative is False
    assert (-nneg).is_negative is None
    assert (2*nneg).is_negative is False

    assert npos.is_negative is None
    assert (-npos).is_negative is False
    assert (2*npos).is_negative is None

    assert (nneg*npos).is_negative is None

    assert (neg*nneg).is_negative is None
    assert (neg*npos).is_negative is False

    assert (pos*nneg).is_negative is False
    assert (pos*npos).is_negative is None

    assert (npos*neg*nneg).is_negative is False
    assert (npos*pos*nneg).is_negative is None

    assert (-npos*neg*nneg).is_negative is None
    assert (-npos*pos*nneg).is_negative is False

    assert (17*npos*neg*nneg).is_negative is False
    assert (17*npos*pos*nneg).is_negative is None

    assert (neg*npos*pos*nneg).is_negative is False

    assert (x*neg).is_negative is None
    assert (nneg*npos*pos*x*neg).is_negative is None

    assert neg.is_positive is False
    assert (-neg).is_positive is True
    assert (2*neg).is_positive is False

    assert pos.is_positive is True
    assert (-pos).is_positive is False
    assert (2*pos).is_positive is True

    assert (pos*neg).is_positive is False
    assert (2*pos*neg).is_positive is False
    assert (-pos*neg).is_positive is True
    assert (-pos*neg*y).is_positive is False    # y.is_real=F;  !real -> !neg

    assert nneg.is_positive is None
    assert (-nneg).is_positive is False
    assert (2*nneg).is_positive is None

    assert npos.is_positive is False
    assert (-npos).is_positive is None
    assert (2*npos).is_positive is False

    assert (nneg*npos).is_positive is False

    assert (neg*nneg).is_positive is False
    assert (neg*npos).is_positive is None

    assert (pos*nneg).is_positive is None
    assert (pos*npos).is_positive is False

    assert (npos*neg*nneg).is_positive is None
    assert (npos*pos*nneg).is_positive is False

    assert (-npos*neg*nneg).is_positive is False
    assert (-npos*pos*nneg).is_positive is None

    assert (17*npos*neg*nneg).is_positive is None
    assert (17*npos*pos*nneg).is_positive is False

    assert (neg*npos*pos*nneg).is_positive is None

    assert (x*neg).is_positive is None
    assert (nneg*npos*pos*x*neg).is_positive is None


def test_Mul_is_negative_positive_2():
    a = Symbol('a', nonnegative=True)
    b = Symbol('b', nonnegative=True)
    c = Symbol('c', nonpositive=True)
    d = Symbol('d', nonpositive=True)

    assert (a*b).is_nonnegative is True
    assert (a*b).is_negative is False
    assert (a*b).is_zero is None
    assert (a*b).is_positive is None

    assert (c*d).is_nonnegative is True
    assert (c*d).is_negative is False
    assert (c*d).is_zero is None
    assert (c*d).is_positive is None

    assert (a*c).is_nonpositive is True
    assert (a*c).is_positive is False
    assert (a*c).is_zero is None
    assert (a*c).is_negative is None


def test_Mul_is_nonpositive_nonnegative():
    x = Symbol('x', real=True)

    k = Symbol('k', negative=True)
    n = Symbol('n', positive=True)
    u = Symbol('u', nonnegative=True)
    v = Symbol('v', nonpositive=True)

    assert k.is_nonpositive is True
    assert (-k).is_nonpositive is False
    assert (2*k).is_nonpositive is True

    assert n.is_nonpositive is False
    assert (-n).is_nonpositive is True
    assert (2*n).is_nonpositive is False

    assert (n*k).is_nonpositive is True
    assert (2*n*k).is_nonpositive is True
    assert (-n*k).is_nonpositive is False

    assert u.is_nonpositive is None
    assert (-u).is_nonpositive is True
    assert (2*u).is_nonpositive is None

    assert v.is_nonpositive is True
    assert (-v).is_nonpositive is None
    assert (2*v).is_nonpositive is True

    assert (u*v).is_nonpositive is True

    assert (k*u).is_nonpositive is True
    assert (k*v).is_nonpositive is None

    assert (n*u).is_nonpositive is None
    assert (n*v).is_nonpositive is True

    assert (v*k*u).is_nonpositive is None
    assert (v*n*u).is_nonpositive is True

    assert (-v*k*u).is_nonpositive is True
    assert (-v*n*u).is_nonpositive is None

    assert (17*v*k*u).is_nonpositive is None
    assert (17*v*n*u).is_nonpositive is True

    assert (k*v*n*u).is_nonpositive is None

    assert (x*k).is_nonpositive is None
    assert (u*v*n*x*k).is_nonpositive is None

    assert k.is_nonnegative is False
    assert (-k).is_nonnegative is True
    assert (2*k).is_nonnegative is False

    assert n.is_nonnegative is True
    assert (-n).is_nonnegative is False
    assert (2*n).is_nonnegative is True

    assert (n*k).is_nonnegative is False
    assert (2*n*k).is_nonnegative is False
    assert (-n*k).is_nonnegative is True

    assert u.is_nonnegative is True
    assert (-u).is_nonnegative is None
    assert (2*u).is_nonnegative is True

    assert v.is_nonnegative is None
    assert (-v).is_nonnegative is True
    assert (2*v).is_nonnegative is None

    assert (u*v).is_nonnegative is None

    assert (k*u).is_nonnegative is None
    assert (k*v).is_nonnegative is True

    assert (n*u).is_nonnegative is True
    assert (n*v).is_nonnegative is None

    assert (v*k*u).is_nonnegative is True
    assert (v*n*u).is_nonnegative is None

    assert (-v*k*u).is_nonnegative is None
    assert (-v*n*u).is_nonnegative is True

    assert (17*v*k*u).is_nonnegative is True
    assert (17*v*n*u).is_nonnegative is None

    assert (k*v*n*u).is_nonnegative is True

    assert (x*k).is_nonnegative is None
    assert (u*v*n*x*k).is_nonnegative is None


def test_Add_is_negative_positive():
    x = Symbol('x', real=True)

    k = Symbol('k', negative=True)
    n = Symbol('n', positive=True)
    u = Symbol('u', nonnegative=True)
    v = Symbol('v', nonpositive=True)

    assert (k - 2).is_negative is True
    assert (k + 17).is_negative is None
    assert (-k - 5).is_negative is None
    assert (-k + 123).is_negative is False

    assert (k - n).is_negative is True
    assert (k + n).is_negative is None
    assert (-k - n).is_negative is None
    assert (-k + n).is_negative is False

    assert (k - n - 2).is_negative is True
    assert (k + n + 17).is_negative is None
    assert (-k - n - 5).is_negative is None
    assert (-k + n + 123).is_negative is False

    assert (-2*k + 123*n + 17).is_negative is False

    assert (k + u).is_negative is None
    assert (k + v).is_negative is True
    assert (n + u).is_negative is False
    assert (n + v).is_negative is None

    assert (u - v).is_negative is False
    assert (u + v).is_negative is None
    assert (-u - v).is_negative is None
    assert (-u + v).is_negative is None

    assert (u - v + n + 2).is_negative is False
    assert (u + v + n + 2).is_negative is None
    assert (-u - v + n + 2).is_negative is None
    assert (-u + v + n + 2).is_negative is None

    assert (k + x).is_negative is None
    assert (k + x - n).is_negative is None

    assert (k - 2).is_positive is False
    assert (k + 17).is_positive is None
    assert (-k - 5).is_positive is None
    assert (-k + 123).is_positive is True

    assert (k - n).is_positive is False
    assert (k + n).is_positive is None
    assert (-k - n).is_positive is None
    assert (-k + n).is_positive is True

    assert (k - n - 2).is_positive is False
    assert (k + n + 17).is_positive is None
    assert (-k - n - 5).is_positive is None
    assert (-k + n + 123).is_positive is True

    assert (-2*k + 123*n + 17).is_positive is True

    assert (k + u).is_positive is None
    assert (k + v).is_positive is False
    assert (n + u).is_positive is True
    assert (n + v).is_positive is None

    assert (u - v).is_positive is None
    assert (u + v).is_positive is None
    assert (-u - v).is_positive is None
    assert (-u + v).is_positive is False

    assert (u - v - n - 2).is_positive is None
    assert (u + v - n - 2).is_positive is None
    assert (-u - v - n - 2).is_positive is None
    assert (-u + v - n - 2).is_positive is False

    assert (n + x).is_positive is None
    assert (n + x - k).is_positive is None

    z = (-3 - sqrt(5) + (-sqrt(10)/2 - sqrt(2)/2)**2)
    assert z.is_zero
    z = sqrt(1 + sqrt(3)) + sqrt(3 + 3*sqrt(3)) - sqrt(10 + 6*sqrt(3))
    assert z.is_zero


def test_Add_is_nonpositive_nonnegative():
    x = Symbol('x', real=True)

    k = Symbol('k', negative=True)
    n = Symbol('n', positive=True)
    u = Symbol('u', nonnegative=True)
    v = Symbol('v', nonpositive=True)

    assert (u - 2).is_nonpositive is None
    assert (u + 17).is_nonpositive is False
    assert (-u - 5).is_nonpositive is True
    assert (-u + 123).is_nonpositive is None

    assert (u - v).is_nonpositive is None
    assert (u + v).is_nonpositive is None
    assert (-u - v).is_nonpositive is None
    assert (-u + v).is_nonpositive is True

    assert (u - v - 2).is_nonpositive is None
    assert (u + v + 17).is_nonpositive is None
    assert (-u - v - 5).is_nonpositive is None
    assert (-u + v - 123).is_nonpositive is True

    assert (-2*u + 123*v - 17).is_nonpositive is True

    assert (k + u).is_nonpositive is None
    assert (k + v).is_nonpositive is True
    assert (n + u).is_nonpositive is False
    assert (n + v).is_nonpositive is None

    assert (k - n).is_nonpositive is True
    assert (k + n).is_nonpositive is None
    assert (-k - n).is_nonpositive is None
    assert (-k + n).is_nonpositive is False

    assert (k - n + u + 2).is_nonpositive is None
    assert (k + n + u + 2).is_nonpositive is None
    assert (-k - n + u + 2).is_nonpositive is None
    assert (-k + n + u + 2).is_nonpositive is False

    assert (u + x).is_nonpositive is None
    assert (v - x - n).is_nonpositive is None

    assert (u - 2).is_nonnegative is None
    assert (u + 17).is_nonnegative is True
    assert (-u - 5).is_nonnegative is False
    assert (-u + 123).is_nonnegative is None

    assert (u - v).is_nonnegative is True
    assert (u + v).is_nonnegative is None
    assert (-u - v).is_nonnegative is None
    assert (-u + v).is_nonnegative is None

    assert (u - v + 2).is_nonnegative is True
    assert (u + v + 17).is_nonnegative is None
    assert (-u - v - 5).is_nonnegative is None
    assert (-u + v - 123).is_nonnegative is False

    assert (2*u - 123*v + 17).is_nonnegative is True

    assert (k + u).is_nonnegative is None
    assert (k + v).is_nonnegative is False
    assert (n + u).is_nonnegative is True
    assert (n + v).is_nonnegative is None

    assert (k - n).is_nonnegative is False
    assert (k + n).is_nonnegative is None
    assert (-k - n).is_nonnegative is None
    assert (-k + n).is_nonnegative is True

    assert (k - n - u - 2).is_nonnegative is False
    assert (k + n - u - 2).is_nonnegative is None
    assert (-k - n - u - 2).is_nonnegative is None
    assert (-k + n - u - 2).is_nonnegative is None

    assert (u - x).is_nonnegative is None
    assert (v + x + n).is_nonnegative is None


def test_Pow_is_integer():
    x = Symbol('x')

    k = Symbol('k', integer=True)
    n = Symbol('n', integer=True, nonnegative=True)
    m = Symbol('m', integer=True, positive=True)

    assert (k**2).is_integer is True
    assert (k**(-2)).is_integer is None
    assert ((m + 1)**(-2)).is_integer is False
    assert (m**(-1)).is_integer is None  # issue 8580

    assert (2**k).is_integer is None
    assert (2**(-k)).is_integer is None

    assert (2**n).is_integer is True
    assert (2**(-n)).is_integer is None

    assert (2**m).is_integer is True
    assert (2**(-m)).is_integer is False

    assert (x**2).is_integer is None
    assert (2**x).is_integer is None

    assert (k**n).is_integer is True
    assert (k**(-n)).is_integer is None

    assert (k**x).is_integer is None
    assert (x**k).is_integer is None

    assert (k**(n*m)).is_integer is True
    assert (k**(-n*m)).is_integer is None

    assert sqrt(3).is_integer is False
    assert sqrt(.3).is_integer is False
    assert Pow(3, 2, evaluate=False).is_integer is True
    assert Pow(3, 0, evaluate=False).is_integer is True
    assert Pow(3, -2, evaluate=False).is_integer is False
    assert Pow(S.Half, 3, evaluate=False).is_integer is False
    # decided by re-evaluating
    assert Pow(3, S.Half, evaluate=False).is_integer is False
    assert Pow(3, S.Half, evaluate=False).is_integer is False
    assert Pow(4, S.Half, evaluate=False).is_integer is True
    assert Pow(S.Half, -2, evaluate=False).is_integer is True

    assert ((-1)**k).is_integer

    # issue 8641
    x = Symbol('x', real=True, integer=False)
    assert (x**2).is_integer is None

    # issue 10458
    x = Symbol('x', positive=True)
    assert (1/(x + 1)).is_integer is False
    assert (1/(-x - 1)).is_integer is False
    assert (-1/(x + 1)).is_integer is False
    # issue 23287
    assert (x**2/2).is_integer is None

    # issue 8648-like
    k = Symbol('k', even=True)
    assert (k**3/2).is_integer
    assert (k**3/8).is_integer
    assert (k**3/16).is_integer is None
    assert (2/k).is_integer is None
    assert (2/k**2).is_integer is False
    o = Symbol('o', odd=True)
    assert (k/o).is_integer is None
    o = Symbol('o', odd=True, prime=True)
    assert (k/o).is_integer is False


def test_Pow_is_real():
    x = Symbol('x', real=True)
    y = Symbol('y', positive=True)

    assert (x**2).is_real is True
    assert (x**3).is_real is True
    assert (x**x).is_real is None
    assert (y**x).is_real is True

    assert (x**Rational(1, 3)).is_real is None
    assert (y**Rational(1, 3)).is_real is True

    assert sqrt(-1 - sqrt(2)).is_real is False

    i = Symbol('i', imaginary=True)
    assert (i**i).is_real is None
    assert (I**i).is_extended_real is True
    assert ((-I)**i).is_extended_real is True
    assert (2**i).is_real is None  # (2**(pi/log(2) * I)) is real, 2**I is not
    assert (2**I).is_real is False
    assert (2**-I).is_real is False
    assert (i**2).is_extended_real is True
    assert (i**3).is_extended_real is False
    assert (i**x).is_real is None  # could be (-I)**(2/3)
    e = Symbol('e', even=True)
    o = Symbol('o', odd=True)
    k = Symbol('k', integer=True)
    assert (i**e).is_extended_real is True
    assert (i**o).is_extended_real is False
    assert (i**k).is_real is None
    assert (i**(4*k)).is_extended_real is True

    x = Symbol("x", nonnegative=True)
    y = Symbol("y", nonnegative=True)
    assert im(x**y).expand(complex=True) is S.Zero
    assert (x**y).is_real is True
    i = Symbol('i', imaginary=True)
    assert (exp(i)**I).is_extended_real is True
    assert log(exp(i)).is_imaginary is None  # i could be 2*pi*I
    c = Symbol('c', complex=True)
    assert log(c).is_real is None  # c could be 0 or 2, too
    assert log(exp(c)).is_real is None  # log(0), log(E), ...
    n = Symbol('n', negative=False)
    assert log(n).is_real is None
    n = Symbol('n', nonnegative=True)
    assert log(n).is_real is None

    assert sqrt(-I).is_real is False  # issue 7843

    i = Symbol('i', integer=True)
    assert (1/(i-1)).is_real is None
    assert (1/(i-1)).is_extended_real is None

    # test issue 20715
    from sympy.core.parameters import evaluate
    x = S(-1)
    with evaluate(False):
        assert x.is_negative is True

    f = Pow(x, -1)
    with evaluate(False):
        assert f.is_imaginary is False


def test_real_Pow():
    k = Symbol('k', integer=True, nonzero=True)
    assert (k**(I*pi/log(k))).is_real


def test_Pow_is_finite():
    xe = Symbol('xe', extended_real=True)
    xr = Symbol('xr', real=True)
    p = Symbol('p', positive=True)
    n = Symbol('n', negative=True)
    i = Symbol('i', integer=True)

    assert (xe**2).is_finite is None  # xe could be oo
    assert (xr**2).is_finite is True

    assert (xe**xe).is_finite is None
    assert (xr**xe).is_finite is None
    assert (xe**xr).is_finite is None
    # FIXME: The line below should be True rather than None
    # assert (xr**xr).is_finite is True
    assert (xr**xr).is_finite is None

    assert (p**xe).is_finite is None
    assert (p**xr).is_finite is True

    assert (n**xe).is_finite is None
    assert (n**xr).is_finite is True

    assert (sin(xe)**2).is_finite is True
    assert (sin(xr)**2).is_finite is True

    assert (sin(xe)**xe).is_finite is None  # xe, xr could be -pi
    assert (sin(xr)**xr).is_finite is None

    # FIXME: Should the line below be True rather than None?
    assert (sin(xe)**exp(xe)).is_finite is None
    assert (sin(xr)**exp(xr)).is_finite is True

    assert (1/sin(xe)).is_finite is None  # if zero, no, otherwise yes
    assert (1/sin(xr)).is_finite is None

    assert (1/exp(xe)).is_finite is None  # xe could be -oo
    assert (1/exp(xr)).is_finite is True

    assert (1/S.Pi).is_finite is True

    assert (1/(i-1)).is_finite is None


def test_Pow_is_even_odd():
    x = Symbol('x')

    k = Symbol('k', even=True)
    n = Symbol('n', odd=True)
    m = Symbol('m', integer=True, nonnegative=True)
    p = Symbol('p', integer=True, positive=True)

    assert ((-1)**n).is_odd
    assert ((-1)**k).is_odd
    assert ((-1)**(m - p)).is_odd

    assert (k**2).is_even is True
    assert (n**2).is_even is False
    assert (2**k).is_even is None
    assert (x**2).is_even is None

    assert (k**m).is_even is None
    assert (n**m).is_even is False

    assert (k**p).is_even is True
    assert (n**p).is_even is False

    assert (m**k).is_even is None
    assert (p**k).is_even is None

    assert (m**n).is_even is None
    assert (p**n).is_even is None

    assert (k**x).is_even is None
    assert (n**x).is_even is None

    assert (k**2).is_odd is False
    assert (n**2).is_odd is True
    assert (3**k).is_odd is None

    assert (k**m).is_odd is None
    assert (n**m).is_odd is True

    assert (k**p).is_odd is False
    assert (n**p).is_odd is True

    assert (m**k).is_odd is None
    assert (p**k).is_odd is None

    assert (m**n).is_odd is None
    assert (p**n).is_odd is None

    assert (k**x).is_odd is None
    assert (n**x).is_odd is None


def test_Pow_is_negative_positive():
    r = Symbol('r', real=True)

    k = Symbol('k', integer=True, positive=True)
    n = Symbol('n', even=True)
    m = Symbol('m', odd=True)

    x = Symbol('x')

    assert (2**r).is_positive is True
    assert ((-2)**r).is_positive is None
    assert ((-2)**n).is_positive is True
    assert ((-2)**m).is_positive is False

    assert (k**2).is_positive is True
    assert (k**(-2)).is_positive is True

    assert (k**r).is_positive is True
    assert ((-k)**r).is_positive is None
    assert ((-k)**n).is_positive is True
    assert ((-k)**m).is_positive is False

    assert (2**r).is_negative is False
    assert ((-2)**r).is_negative is None
    assert ((-2)**n).is_negative is False
    assert ((-2)**m).is_negative is True

    assert (k**2).is_negative is False
    assert (k**(-2)).is_negative is False

    assert (k**r).is_negative is False
    assert ((-k)**r).is_negative is None
    assert ((-k)**n).is_negative is False
    assert ((-k)**m).is_negative is True

    assert (2**x).is_positive is None
    assert (2**x).is_negative is None


def test_Pow_is_zero():
    z = Symbol('z', zero=True)
    e = z**2
    assert e.is_zero
    assert e.is_positive is False
    assert e.is_negative is False

    assert Pow(0, 0, evaluate=False).is_zero is False
    assert Pow(0, 3, evaluate=False).is_zero
    assert Pow(0, oo, evaluate=False).is_zero
    assert Pow(0, -3, evaluate=False).is_zero is False
    assert Pow(0, -oo, evaluate=False).is_zero is False
    assert Pow(2, 2, evaluate=False).is_zero is False

    a = Symbol('a', zero=False)
    assert Pow(a, 3).is_zero is False  # issue 7965

    assert Pow(2, oo, evaluate=False).is_zero is False
    assert Pow(2, -oo, evaluate=False).is_zero
    assert Pow(S.Half, oo, evaluate=False).is_zero
    assert Pow(S.Half, -oo, evaluate=False).is_zero is False

    # All combinations of real/complex base/exponent
    h = S.Half
    T = True
    F = False
    N = None

    pow_iszero = [
        ['**',  0,  h,  1,  2, -h, -1,-2,-2*I,-I/2,I/2,1+I,oo,-oo,zoo],
        [   0,  F,  T,  T,  T,  F,  F,  F,  F,  F,  F,  N,  T,  F,  N],
        [   h,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  F,  N],
        [   1,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  N],
        [   2,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  N],
        [  -h,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  F,  N],
        [  -1,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  N],
        [  -2,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  N],
        [-2*I,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  N],
        [-I/2,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  F,  N],
        [ I/2,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  F,  N],
        [ 1+I,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  F,  T,  N],
        [  oo,  F,  F,  F,  F,  T,  T,  T,  F,  F,  F,  F,  F,  T,  N],
        [ -oo,  F,  F,  F,  F,  T,  T,  T,  F,  F,  F,  F,  F,  T,  N],
        [ zoo,  F,  F,  F,  F,  T,  T,  T,  N,  N,  N,  N,  F,  T,  N]
    ]

    def test_table(table):
        n = len(table[0])
        for row in range(1, n):
            base = table[row][0]
            for col in range(1, n):
                exp = table[0][col]
                is_zero = table[row][col]
                # The actual test here:
                assert Pow(base, exp, evaluate=False).is_zero is is_zero

    test_table(pow_iszero)

    # A zero symbol...
    zo, zo2 = symbols('zo, zo2', zero=True)

    # All combinations of finite symbols
    zf, zf2 = symbols('zf, zf2', finite=True)
    wf, wf2 = symbols('wf, wf2', nonzero=True)
    xf, xf2 = symbols('xf, xf2', real=True)
    yf, yf2 = symbols('yf, yf2', nonzero=True)
    af, af2 = symbols('af, af2', positive=True)
    bf, bf2 = symbols('bf, bf2', nonnegative=True)
    cf, cf2 = symbols('cf, cf2', negative=True)
    df, df2 = symbols('df, df2', nonpositive=True)

    # Without finiteness:
    zi, zi2 = symbols('zi, zi2')
    wi, wi2 = symbols('wi, wi2', zero=False)
    xi, xi2 = symbols('xi, xi2', extended_real=True)
    yi, yi2 = symbols('yi, yi2', zero=False, extended_real=True)
    ai, ai2 = symbols('ai, ai2', extended_positive=True)
    bi, bi2 = symbols('bi, bi2', extended_nonnegative=True)
    ci, ci2 = symbols('ci, ci2', extended_negative=True)
    di, di2 = symbols('di, di2', extended_nonpositive=True)

    pow_iszero_sym = [
        ['**',zo,wf,yf,af,cf,zf,xf,bf,df,zi,wi,xi,yi,ai,bi,ci,di],
        [ zo2, F, N, N, T, F, N, N, N, F, N, N, N, N, T, N, F, F],
        [ wf2, F, F, F, F, F, F, F, F, F, N, N, N, N, N, N, N, N],
        [ yf2, F, F, F, F, F, F, F, F, F, N, N, N, N, N, N, N, N],
        [ af2, F, F, F, F, F, F, F, F, F, N, N, N, N, N, N, N, N],
        [ cf2, F, F, F, F, F, F, F, F, F, N, N, N, N, N, N, N, N],
        [ zf2, N, N, N, N, F, N, N, N, N, N, N, N, N, N, N, N, N],
        [ xf2, N, N, N, N, F, N, N, N, N, N, N, N, N, N, N, N, N],
        [ bf2, N, N, N, N, F, N, N, N, N, N, N, N, N, N, N, N, N],
        [ df2, N, N, N, N, F, N, N, N, N, N, N, N, N, N, N, N, N],
        [ zi2, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N],
        [ wi2, F, N, N, F, N, N, N, F, N, N, N, N, N, N, N, N, N],
        [ xi2, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N],
        [ yi2, F, N, N, F, N, N, N, F, N, N, N, N, N, N, N, N, N],
        [ ai2, F, N, N, F, N, N, N, F, N, N, N, N, N, N, N, N, N],
        [ bi2, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N],
        [ ci2, F, N, N, F, N, N, N, F, N, N, N, N, N, N, N, N, N],
        [ di2, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N, N]
    ]

    test_table(pow_iszero_sym)

    # In some cases (x**x).is_zero is different from (x**y).is_zero even if y
    # has the same assumptions as x.
    assert (zo ** zo).is_zero is False
    assert (wf ** wf).is_zero is False
    assert (yf ** yf).is_zero is False
    assert (af ** af).is_zero is False
    assert (cf ** cf).is_zero is False
    assert (zf ** zf).is_zero is None
    assert (xf ** xf).is_zero is None
    assert (bf ** bf).is_zero is False # None in table
    assert (df ** df).is_zero is None
    assert (zi ** zi).is_zero is None
    assert (wi ** wi).is_zero is None
    assert (xi ** xi).is_zero is None
    assert (yi ** yi).is_zero is None
    assert (ai ** ai).is_zero is False # None in table
    assert (bi ** bi).is_zero is False # None in table
    assert (ci ** ci).is_zero is None
    assert (di ** di).is_zero is None


def test_Pow_is_nonpositive_nonnegative():
    x = Symbol('x', real=True)

    k = Symbol('k', integer=True, nonnegative=True)
    l = Symbol('l', integer=True, positive=True)
    n = Symbol('n', even=True)
    m = Symbol('m', odd=True)

    assert (x**(4*k)).is_nonnegative is True
    assert (2**x).is_nonnegative is True
    assert ((-2)**x).is_nonnegative is None
    assert ((-2)**n).is_nonnegative is True
    assert ((-2)**m).is_nonnegative is False

    assert (k**2).is_nonnegative is True
    assert (k**(-2)).is_nonnegative is None
    assert (k**k).is_nonnegative is True

    assert (k**x).is_nonnegative is None    # NOTE (0**x).is_real = U
    assert (l**x).is_nonnegative is True
    assert (l**x).is_positive is True
    assert ((-k)**x).is_nonnegative is None

    assert ((-k)**m).is_nonnegative is None

    assert (2**x).is_nonpositive is False
    assert ((-2)**x).is_nonpositive is None
    assert ((-2)**n).is_nonpositive is False
    assert ((-2)**m).is_nonpositive is True

    assert (k**2).is_nonpositive is None
    assert (k**(-2)).is_nonpositive is None

    assert (k**x).is_nonpositive is None
    assert ((-k)**x).is_nonpositive is None
    assert ((-k)**n).is_nonpositive is None


    assert (x**2).is_nonnegative is True
    i = symbols('i', imaginary=True)
    assert (i**2).is_nonpositive is True
    assert (i**4).is_nonpositive is False
    assert (i**3).is_nonpositive is False
    assert (I**i).is_nonnegative is True
    assert (exp(I)**i).is_nonnegative is True

    assert ((-l)**n).is_nonnegative is True
    assert ((-l)**m).is_nonpositive is True
    assert ((-k)**n).is_nonnegative is None
    assert ((-k)**m).is_nonpositive is None


def test_Mul_is_imaginary_real():
    r = Symbol('r', real=True)
    p = Symbol('p', positive=True)
    i1 = Symbol('i1', imaginary=True)
    i2 = Symbol('i2', imaginary=True)
    x = Symbol('x')

    assert I.is_imaginary is True
    assert I.is_real is False
    assert (-I).is_imaginary is True
    assert (-I).is_real is False
    assert (3*I).is_imaginary is True
    assert (3*I).is_real is False
    assert (I*I).is_imaginary is False
    assert (I*I).is_real is True

    e = (p + p*I)
    j = Symbol('j', integer=True, zero=False)
    assert (e**j).is_real is None
    assert (e**(2*j)).is_real is None
    assert (e**j).is_imaginary is None
    assert (e**(2*j)).is_imaginary is None

    assert (e**-1).is_imaginary is False
    assert (e**2).is_imaginary
    assert (e**3).is_imaginary is False
    assert (e**4).is_imaginary is False
    assert (e**5).is_imaginary is False
    assert (e**-1).is_real is False
    assert (e**2).is_real is False
    assert (e**3).is_real is False
    assert (e**4).is_real is True
    assert (e**5).is_real is False
    assert (e**3).is_complex

    assert (r*i1).is_imaginary is None
    assert (r*i1).is_real is None

    assert (x*i1).is_imaginary is None
    assert (x*i1).is_real is None

    assert (i1*i2).is_imaginary is False
    assert (i1*i2).is_real is True

    assert (r*i1*i2).is_imaginary is False
    assert (r*i1*i2).is_real is True

    # Github's issue 5874:
    nr = Symbol('nr', real=False, complex=True)  # e.g. I or 1 + I
    a = Symbol('a', real=True, nonzero=True)
    b = Symbol('b', real=True)
    assert (i1*nr).is_real is None
    assert (a*nr).is_real is False
    assert (b*nr).is_real is None

    ni = Symbol('ni', imaginary=False, complex=True)  # e.g. 2 or 1 + I
    a = Symbol('a', real=True, nonzero=True)
    b = Symbol('b', real=True)
    assert (i1*ni).is_real is False
    assert (a*ni).is_real is None
    assert (b*ni).is_real is None


def test_Mul_hermitian_antihermitian():
    xz, yz = symbols('xz, yz', zero=True, antihermitian=True)
    xf, yf = symbols('xf, yf', hermitian=False, antihermitian=False, finite=True)
    xh, yh = symbols('xh, yh', hermitian=True, antihermitian=False, nonzero=True)
    xa, ya = symbols('xa, ya', hermitian=False, antihermitian=True, zero=False, finite=True)
    assert (xz*xh).is_hermitian is True
    assert (xz*xh).is_antihermitian is True
    assert (xz*xa).is_hermitian is True
    assert (xz*xa).is_antihermitian is True
    assert (xf*yf).is_hermitian is None
    assert (xf*yf).is_antihermitian is None
    assert (xh*yh).is_hermitian is True
    assert (xh*yh).is_antihermitian is False
    assert (xh*ya).is_hermitian is False
    assert (xh*ya).is_antihermitian is True
    assert (xa*ya).is_hermitian is True
    assert (xa*ya).is_antihermitian is False

    a = Symbol('a', hermitian=True, zero=False)
    b = Symbol('b', hermitian=True)
    c = Symbol('c', hermitian=False)
    d = Symbol('d', antihermitian=True)
    e1 = Mul(a, b, c, evaluate=False)
    e2 = Mul(b, a, c, evaluate=False)
    e3 = Mul(a, b, c, d, evaluate=False)
    e4 = Mul(b, a, c, d, evaluate=False)
    e5 = Mul(a, c, evaluate=False)
    e6 = Mul(a, c, d, evaluate=False)
    assert e1.is_hermitian is None
    assert e2.is_hermitian is None
    assert e1.is_antihermitian is None
    assert e2.is_antihermitian is None
    assert e3.is_antihermitian is None
    assert e4.is_antihermitian is None
    assert e5.is_antihermitian is None
    assert e6.is_antihermitian is None


def test_Add_is_comparable():
    assert (x + y).is_comparable is False
    assert (x + 1).is_comparable is False
    assert (Rational(1, 3) - sqrt(8)).is_comparable is True


def test_Mul_is_comparable():
    assert (x*y).is_comparable is False
    assert (x*2).is_comparable is False
    assert (sqrt(2)*Rational(1, 3)).is_comparable is True


def test_Pow_is_comparable():
    assert (x**y).is_comparable is False
    assert (x**2).is_comparable is False
    assert (sqrt(Rational(1, 3))).is_comparable is True


def test_Add_is_positive_2():
    e = Rational(1, 3) - sqrt(8)
    assert e.is_positive is False
    assert e.is_negative is True

    e = pi - 1
    assert e.is_positive is True
    assert e.is_negative is False


def test_Add_is_irrational():
    i = Symbol('i', irrational=True)

    assert i.is_irrational is True
    assert i.is_rational is False

    assert (i + 1).is_irrational is True
    assert (i + 1).is_rational is False


def test_Mul_is_irrational():
    expr = Mul(1, 2, 3, evaluate=False)
    assert expr.is_irrational is False
    expr = Mul(1, I, I, evaluate=False)
    assert expr.is_rational is None # I * I = -1 but *no evaluation allowed*
    # sqrt(2) * I * I = -sqrt(2) is irrational but
    # this can't be determined without evaluating the
    # expression and the eval_is routines shouldn't do that
    expr = Mul(sqrt(2), I, I, evaluate=False)
    assert expr.is_irrational is None


def test_issue_3531():
    # https://github.com/sympy/sympy/issues/3531
    # https://github.com/sympy/sympy/pull/18116
    class MightyNumeric(tuple):
        def __rtruediv__(self, other):
            return "something"

    assert sympify(1)/MightyNumeric((1, 2)) == "something"


def test_issue_3531b():
    class Foo:
        def __init__(self):
            self.field = 1.0

        def __mul__(self, other):
            self.field = self.field * other

        def __rmul__(self, other):
            self.field = other * self.field
    f = Foo()
    x = Symbol("x")
    assert f*x == x*f


def test_bug3():
    a = Symbol("a")
    b = Symbol("b", positive=True)
    e = 2*a + b
    f = b + 2*a
    assert e == f


def test_suppressed_evaluation():
    a = Add(0, 3, 2, evaluate=False)
    b = Mul(1, 3, 2, evaluate=False)
    c = Pow(3, 2, evaluate=False)
    assert a != 6
    assert a.func is Add
    assert a.args == (0, 3, 2)
    assert b != 6
    assert b.func is Mul
    assert b.args == (1, 3, 2)
    assert c != 9
    assert c.func is Pow
    assert c.args == (3, 2)


def test_AssocOp_doit():
    a = Add(x,x, evaluate=False)
    b = Mul(y,y, evaluate=False)
    c = Add(b,b, evaluate=False)
    d = Mul(a,a, evaluate=False)
    assert c.doit(deep=False).func == Mul
    assert c.doit(deep=False).args == (2,y,y)
    assert c.doit().func == Mul
    assert c.doit().args == (2, Pow(y,2))
    assert d.doit(deep=False).func == Pow
    assert d.doit(deep=False).args == (a, 2*S.One)
    assert d.doit().func == Mul
    assert d.doit().args == (4*S.One, Pow(x,2))


def test_Add_Mul_Expr_args():
    nonexpr = [Basic(), Poly(x, x), FiniteSet(x)]
    for typ in [Add, Mul]:
        for obj in nonexpr:
            # The cache can mess with the stacklevel check
            with warns(SymPyDeprecationWarning, test_stacklevel=False):
                typ(obj, 1)


def test_Add_as_coeff_mul():
    # issue 5524.  These should all be (1, self)
    assert (x + 1).as_coeff_mul() == (1, (x + 1,))
    assert (x + 2).as_coeff_mul() == (1, (x + 2,))
    assert (x + 3).as_coeff_mul() == (1, (x + 3,))

    assert (x - 1).as_coeff_mul() == (1, (x - 1,))
    assert (x - 2).as_coeff_mul() == (1, (x - 2,))
    assert (x - 3).as_coeff_mul() == (1, (x - 3,))

    n = Symbol('n', integer=True)
    assert (n + 1).as_coeff_mul() == (1, (n + 1,))
    assert (n + 2).as_coeff_mul() == (1, (n + 2,))
    assert (n + 3).as_coeff_mul() == (1, (n + 3,))

    assert (n - 1).as_coeff_mul() == (1, (n - 1,))
    assert (n - 2).as_coeff_mul() == (1, (n - 2,))
    assert (n - 3).as_coeff_mul() == (1, (n - 3,))


def test_Pow_as_coeff_mul_doesnt_expand():
    assert exp(x + y).as_coeff_mul() == (1, (exp(x + y),))
    assert exp(x + exp(x + y)) != exp(x + exp(x)*exp(y))

def test_issue_24751():
    expr = Add(-2, -3, evaluate=False)
    expr1 = Add(-1, expr, evaluate=False)
    assert int(expr1) == int((-3 - 2) - 1)


def test_issue_3514_18626():
    assert sqrt(S.Half) * sqrt(6) == 2 * sqrt(3)/2
    assert S.Half*sqrt(6)*sqrt(2) == sqrt(3)
    assert sqrt(6)/2*sqrt(2) == sqrt(3)
    assert sqrt(6)*sqrt(2)/2 == sqrt(3)
    assert sqrt(8)**Rational(2, 3) == 2


def test_make_args():
    assert Add.make_args(x) == (x,)
    assert Mul.make_args(x) == (x,)

    assert Add.make_args(x*y*z) == (x*y*z,)
    assert Mul.make_args(x*y*z) == (x*y*z).args

    assert Add.make_args(x + y + z) == (x + y + z).args
    assert Mul.make_args(x + y + z) == (x + y + z,)

    assert Add.make_args((x + y)**z) == ((x + y)**z,)
    assert Mul.make_args((x + y)**z) == ((x + y)**z,)


def test_issue_5126():
    assert (-2)**x*(-3)**x != 6**x
    i = Symbol('i', integer=1)
    assert (-2)**i*(-3)**i == 6**i


def test_Rational_as_content_primitive():
    c, p = S.One, S.Zero
    assert (c*p).as_content_primitive() == (c, p)
    c, p = S.Half, S.One
    assert (c*p).as_content_primitive() == (c, p)


def test_Add_as_content_primitive():
    assert (x + 2).as_content_primitive() == (1, x + 2)

    assert (3*x + 2).as_content_primitive() == (1, 3*x + 2)
    assert (3*x + 3).as_content_primitive() == (3, x + 1)
    assert (3*x + 6).as_content_primitive() == (3, x + 2)

    assert (3*x + 2*y).as_content_primitive() == (1, 3*x + 2*y)
    assert (3*x + 3*y).as_content_primitive() == (3, x + y)
    assert (3*x + 6*y).as_content_primitive() == (3, x + 2*y)

    assert (3/x + 2*x*y*z**2).as_content_primitive() == (1, 3/x + 2*x*y*z**2)
    assert (3/x + 3*x*y*z**2).as_content_primitive() == (3, 1/x + x*y*z**2)
    assert (3/x + 6*x*y*z**2).as_content_primitive() == (3, 1/x + 2*x*y*z**2)

    assert (2*x/3 + 4*y/9).as_content_primitive() == \
        (Rational(2, 9), 3*x + 2*y)
    assert (2*x/3 + 2.5*y).as_content_primitive() == \
        (Rational(1, 3), 2*x + 7.5*y)

    # the coefficient may sort to a position other than 0
    p = 3 + x + y
    assert (2*p).expand().as_content_primitive() == (2, p)
    assert (2.0*p).expand().as_content_primitive() == (1, 2.*p)
    p *= -1
    assert (2*p).expand().as_content_primitive() == (2, p)


def test_Mul_as_content_primitive():
    assert (2*x).as_content_primitive() == (2, x)
    assert (x*(2 + 2*x)).as_content_primitive() == (2, x*(1 + x))
    assert (x*(2 + 2*y)*(3*x + 3)**2).as_content_primitive() == \
        (18, x*(1 + y)*(x + 1)**2)
    assert ((2 + 2*x)**2*(3 + 6*x) + S.Half).as_content_primitive() == \
        (S.Half, 24*(x + 1)**2*(2*x + 1) + 1)


def test_Pow_as_content_primitive():
    assert (x**y).as_content_primitive() == (1, x**y)
    assert ((2*x + 2)**y).as_content_primitive() == \
        (1, (Mul(2, (x + 1), evaluate=False))**y)
    assert ((2*x + 2)**3).as_content_primitive() == (8, (x + 1)**3)


def test_issue_5460():
    u = Mul(2, (1 + x), evaluate=False)
    assert (2 + u).args == (2, u)


def test_product_irrational():
    assert (I*pi).is_irrational is False
    # The following used to be deduced from the above bug:
    assert (I*pi).is_positive is False


def test_issue_5919():
    assert (x/(y*(1 + y))).expand() == x/(y**2 + y)


def test_Mod():
    assert Mod(x, 1).func is Mod
    assert pi % pi is S.Zero
    assert Mod(5, 3) == 2
    assert Mod(-5, 3) == 1
    assert Mod(5, -3) == -1
    assert Mod(-5, -3) == -2
    assert type(Mod(3.2, 2, evaluate=False)) == Mod
    assert 5 % x == Mod(5, x)
    assert x % 5 == Mod(x, 5)
    assert x % y == Mod(x, y)
    assert (x % y).subs({x: 5, y: 3}) == 2
    assert Mod(nan, 1) is nan
    assert Mod(1, nan) is nan
    assert Mod(nan, nan) is nan

    assert Mod(0, x) == 0
    with raises(ZeroDivisionError):
        Mod(x, 0)

    k = Symbol('k', integer=True)
    m = Symbol('m', integer=True, positive=True)
    assert (x**m % x).func is Mod
    assert (k**(-m) % k).func is Mod
    assert k**m % k == 0
    assert (-2*k)**m % k == 0

    # Float handling
    point3 = Float(3.3) % 1
    assert (x - 3.3) % 1 == Mod(1.*x + 1 - point3, 1)
    assert Mod(-3.3, 1) == 1 - point3
    assert Mod(0.7, 1) == Float(0.7)
    e = Mod(1.3, 1)
    assert comp(e, .3) and e.is_Float
    e = Mod(1.3, .7)
    assert comp(e, .6) and e.is_Float
    e = Mod(1.3, Rational(7, 10))
    assert comp(e, .6) and e.is_Float
    e = Mod(Rational(13, 10), 0.7)
    assert comp(e, .6) and e.is_Float
    e = Mod(Rational(13, 10), Rational(7, 10))
    assert comp(e, .6) and e.is_Rational

    # check that sign is right
    r2 = sqrt(2)
    r3 = sqrt(3)
    for i in [-r3, -r2, r2, r3]:
        for j in [-r3, -r2, r2, r3]:
            assert verify_numerically(i % j, i.n() % j.n())
    for _x in range(4):
        for _y in range(9):
            reps = [(x, _x), (y, _y)]
            assert Mod(3*x + y, 9).subs(reps) == (3*_x + _y) % 9

    # denesting
    t = Symbol('t', real=True)
    assert Mod(Mod(x, t), t) == Mod(x, t)
    assert Mod(-Mod(x, t), t) == Mod(-x, t)
    assert Mod(Mod(x, 2*t), t) == Mod(x, t)
    assert Mod(-Mod(x, 2*t), t) == Mod(-x, t)
    assert Mod(Mod(x, t), 2*t) == Mod(x, t)
    assert Mod(-Mod(x, t), -2*t) == -Mod(x, t)
    for i in [-4, -2, 2, 4]:
        for j in [-4, -2, 2, 4]:
            for k in range(4):
                assert Mod(Mod(x, i), j).subs({x: k}) == (k % i) % j
                assert Mod(-Mod(x, i), j).subs({x: k}) == -(k % i) % j

    # known difference
    assert Mod(5*sqrt(2), sqrt(5)) == 5*sqrt(2) - 3*sqrt(5)
    p = symbols('p', positive=True)
    assert Mod(2, p + 3) == 2
    assert Mod(-2, p + 3) == p + 1
    assert Mod(2, -p - 3) == -p - 1
    assert Mod(-2, -p - 3) == -2
    assert Mod(p + 5, p + 3) == 2
    assert Mod(-p - 5, p + 3) == p + 1
    assert Mod(p + 5, -p - 3) == -p - 1
    assert Mod(-p - 5, -p - 3) == -2
    assert Mod(p + 1, p - 1).func is Mod

    # handling sums
    assert (x + 3) % 1 == Mod(x, 1)
    assert (x + 3.0) % 1 == Mod(1.*x, 1)
    assert (x - S(33)/10) % 1 == Mod(x + S(7)/10, 1)

    a = Mod(.6*x + y, .3*y)
    b = Mod(0.1*y + 0.6*x, 0.3*y)
    # Test that a, b are equal, with 1e-14 accuracy in coefficients
    eps = 1e-14
    assert abs((a.args[0] - b.args[0]).subs({x: 1, y: 1})) < eps
    assert abs((a.args[1] - b.args[1]).subs({x: 1, y: 1})) < eps

    assert (x + 1) % x == 1 % x
    assert (x + y) % x == y % x
    assert (x + y + 2) % x == (y + 2) % x
    assert (a + 3*x + 1) % (2*x) == Mod(a + x + 1, 2*x)
    assert (12*x + 18*y) % (3*x) == 3*Mod(6*y, x)

    # gcd extraction
    assert (-3*x) % (-2*y) == -Mod(3*x, 2*y)
    assert (.6*pi) % (.3*x*pi) == 0.3*pi*Mod(2, x)
    assert (.6*pi) % (.31*x*pi) == pi*Mod(0.6, 0.31*x)
    assert (6*pi) % (.3*x*pi) == 0.3*pi*Mod(20, x)
    assert (6*pi) % (.31*x*pi) == pi*Mod(6, 0.31*x)
    assert (6*pi) % (.42*x*pi) == pi*Mod(6, 0.42*x)
    assert (12*x) % (2*y) == 2*Mod(6*x, y)
    assert (12*x) % (3*5*y) == 3*Mod(4*x, 5*y)
    assert (12*x) % (15*x*y) == 3*x*Mod(4, 5*y)
    assert (-2*pi) % (3*pi) == pi
    assert (2*x + 2) % (x + 1) == 0
    assert (x*(x + 1)) % (x + 1) == (x + 1)*Mod(x, 1)
    assert Mod(5.0*x, 0.1*y) == 0.1*Mod(50*x, y)
    i = Symbol('i', integer=True)
    assert (3*i*x) % (2*i*y) == i*Mod(3*x, 2*y)
    assert Mod(4*i, 4) == 0

    # issue 8677
    n = Symbol('n', integer=True, positive=True)
    assert factorial(n) % n == 0
    assert factorial(n + 2) % n == 0
    assert (factorial(n + 4) % (n + 5)).func is Mod

    # Wilson's theorem
    assert factorial(18042, evaluate=False) % 18043 == 18042
    p = Symbol('n', prime=True)
    assert factorial(p - 1) % p == p - 1
    assert factorial(p - 1) % -p == -1
    assert (factorial(3, evaluate=False) % 4).doit() == 2
    n = Symbol('n', composite=True, odd=True)
    assert factorial(n - 1) % n == 0

    # symbolic with known parity
    n = Symbol('n', even=True)
    assert Mod(n, 2) == 0
    n = Symbol('n', odd=True)
    assert Mod(n, 2) == 1

    # issue 10963
    assert (x**6000%400).args[1] == 400

    #issue 13543
    assert Mod(Mod(x + 1, 2) + 1, 2) == Mod(x, 2)

    x1 = Symbol('x1', integer=True)
    assert Mod(Mod(x1 + 2, 4)*(x1 + 4), 4) == Mod(x1*(x1 + 2), 4)
    assert Mod(Mod(x1 + 2, 4)*4, 4) == 0

    # issue 15493
    i, j = symbols('i j', integer=True, positive=True)
    assert Mod(3*i, 2) == Mod(i, 2)
    assert Mod(8*i/j, 4) == 4*Mod(2*i/j, 1)
    assert Mod(8*i, 4) == 0

    # rewrite
    assert Mod(x, y).rewrite(floor) == x - y*floor(x/y)
    assert ((x - Mod(x, y))/y).rewrite(floor) == floor(x/y)

    # issue 21373
    from sympy.functions.elementary.hyperbolic import sinh
    from sympy.functions.elementary.piecewise import Piecewise

    x_r, y_r = symbols('x_r y_r', real=True)
    assert (Piecewise((x_r, y_r > x_r), (y_r, True)) / z) % 1
    expr = exp(sinh(Piecewise((x_r, y_r > x_r), (y_r, True)) / z))
    expr.subs({1: 1.0})
    sinh(Piecewise((x_r, y_r > x_r), (y_r, True)) * z ** -1.0).is_zero

    # issue 24215
    from sympy.abc import phi
    assert Mod(4.0*Mod(phi, 1) , 2) == 2.0*(Mod(2*(Mod(phi, 1)), 1))

    xi = symbols('x', integer=True)
    assert unchanged(Mod, xi, 2)
    assert Mod(3*xi, 2) == Mod(xi, 2)
    assert unchanged(Mod, 3*x, 2)


def test_Mod_Pow():
    # modular exponentiation
    assert isinstance(Mod(Pow(2, 2, evaluate=False), 3), Integer)

    assert Mod(Pow(4, 13, evaluate=False), 497) == Mod(Pow(4, 13), 497)
    assert Mod(Pow(2, 10000000000, evaluate=False), 3) == 1
    assert Mod(Pow(32131231232, 9**10**6, evaluate=False),10**12) == \
        pow(32131231232,9**10**6,10**12)
    assert Mod(Pow(33284959323, 123**999, evaluate=False),11**13) == \
        pow(33284959323,123**999,11**13)
    assert Mod(Pow(78789849597, 333**555, evaluate=False),12**9) == \
        pow(78789849597,333**555,12**9)

    # modular nested exponentiation
    expr = Pow(2, 2, evaluate=False)
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3**10) == 16
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3**10) == 6487
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3**10) == 32191
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3**10) == 18016
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3**10) == 5137

    expr = Pow(2, 2, evaluate=False)
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3**10) == 16
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3**10) == 256
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3**10) == 6487
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3**10) == 38281
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3**10) == 15928

    expr = Pow(2, 2, evaluate=False)
    expr = Pow(expr, expr, evaluate=False)
    assert Mod(expr, 3**10) == 256
    expr = Pow(expr, expr, evaluate=False)
    assert Mod(expr, 3**10) == 9229
    expr = Pow(expr, expr, evaluate=False)
    assert Mod(expr, 3**10) == 25708
    expr = Pow(expr, expr, evaluate=False)
    assert Mod(expr, 3**10) == 26608
    expr = Pow(expr, expr, evaluate=False)
    # XXX This used to fail in a nondeterministic way because of overflow
    # error.
    assert Mod(expr, 3**10) == 1966


def test_Mod_is_integer():
    p = Symbol('p', integer=True)
    q1 = Symbol('q1', integer=True)
    q2 = Symbol('q2', integer=True, nonzero=True)
    assert Mod(x, y).is_integer is None
    assert Mod(p, q1).is_integer is None
    assert Mod(x, q2).is_integer is None
    assert Mod(p, q2).is_integer


def test_Mod_is_nonposneg():
    n = Symbol('n', integer=True)
    k = Symbol('k', integer=True, positive=True)
    assert (n%3).is_nonnegative
    assert Mod(n, -3).is_nonpositive
    assert Mod(n, k).is_nonnegative
    assert Mod(n, -k).is_nonpositive
    assert Mod(k, n).is_nonnegative is None


def test_issue_6001():
    A = Symbol("A", commutative=False)
    eq = A + A**2
    # it doesn't matter whether it's True or False; they should
    # just all be the same
    assert (
        eq.is_commutative ==
        (eq + 1).is_commutative ==
        (A + 1).is_commutative)

    B = Symbol("B", commutative=False)
    # Although commutative terms could cancel we return True
    # meaning "there are non-commutative symbols; aftersubstitution
    # that definition can change, e.g. (A*B).subs(B,A**-1) -> 1
    assert (sqrt(2)*A).is_commutative is False
    assert (sqrt(2)*A*B).is_commutative is False


def test_polar():
    from sympy.functions.elementary.complexes import polar_lift
    p = Symbol('p', polar=True)
    x = Symbol('x')
    assert p.is_polar
    assert x.is_polar is None
    assert S.One.is_polar is None
    assert (p**x).is_polar is True
    assert (x**p).is_polar is None
    assert ((2*p)**x).is_polar is True
    assert (2*p).is_polar is True
    assert (-2*p).is_polar is not True
    assert (polar_lift(-2)*p).is_polar is True

    q = Symbol('q', polar=True)
    assert (p*q)**2 == p**2 * q**2
    assert (2*q)**2 == 4 * q**2
    assert ((p*q)**x).expand() == p**x * q**x


def test_issue_6040():
    a, b = Pow(1, 2, evaluate=False), S.One
    assert a != b
    assert b != a
    assert not (a == b)
    assert not (b == a)


def test_issue_6082():
    # Comparison is symmetric
    assert Basic.compare(Max(x, 1), Max(x, 2)) == \
      - Basic.compare(Max(x, 2), Max(x, 1))
    # Equal expressions compare equal
    assert Basic.compare(Max(x, 1), Max(x, 1)) == 0
    # Basic subtypes (such as Max) compare different than standard types
    assert Basic.compare(Max(1, x), frozenset((1, x))) != 0


def test_issue_6077():
    assert x**2.0/x == x**1.0
    assert x/x**2.0 == x**-1.0
    assert x*x**2.0 == x**3.0
    assert x**1.5*x**2.5 == x**4.0

    assert 2**(2.0*x)/2**x == 2**(1.0*x)
    assert 2**x/2**(2.0*x) == 2**(-1.0*x)
    assert 2**x*2**(2.0*x) == 2**(3.0*x)
    assert 2**(1.5*x)*2**(2.5*x) == 2**(4.0*x)


def test_mul_flatten_oo():
    p = symbols('p', positive=True)
    n, m = symbols('n,m', negative=True)
    x_im = symbols('x_im', imaginary=True)
    assert n*oo is -oo
    assert n*m*oo is oo
    assert p*oo is oo
    assert x_im*oo != I*oo  # i could be +/- 3*I -> +/-oo


def test_add_flatten():
    # see https://github.com/sympy/sympy/issues/2633#issuecomment-29545524
    a = oo + I*oo
    b = oo - I*oo
    assert a + b is nan
    assert a - b is nan
    # FIXME: This evaluates as:
    #   >>> 1/a
    #   0*(oo + oo*I)
    # which should not simplify to 0. Should be fixed in Pow.eval
    #assert (1/a).simplify() == (1/b).simplify() == 0

    a = Pow(2, 3, evaluate=False)
    assert a + a == 16


def test_issue_5160_6087_6089_6090():
    # issue 6087
    assert ((-2*x*y**y)**3.2).n(2) == (2**3.2*(-x*y**y)**3.2).n(2)
    # issue 6089
    A, B, C = symbols('A,B,C', commutative=False)
    assert (2.*B*C)**3 == 8.0*(B*C)**3
    assert (-2.*B*C)**3 == -8.0*(B*C)**3
    assert (-2*B*C)**2 == 4*(B*C)**2
    # issue 5160
    assert sqrt(-1.0*x) == 1.0*sqrt(-x)
    assert sqrt(1.0*x) == 1.0*sqrt(x)
    # issue 6090
    assert (-2*x*y*A*B)**2 == 4*x**2*y**2*(A*B)**2


def test_float_int_round():
    assert int(float(sqrt(10))) == int(sqrt(10))
    assert int(pi**1000) % 10 == 2
    assert int(Float('1.123456789012345678901234567890e20', '')) == \
        int(112345678901234567890)
    assert int(Float('1.123456789012345678901234567890e25', '')) == \
        int(11234567890123456789012345)
    # decimal forces float so it's not an exact integer ending in 000000
    assert int(Float('1.123456789012345678901234567890e35', '')) == \
        112345678901234567890123456789000192
    assert int(Float('123456789012345678901234567890e5', '')) == \
        12345678901234567890123456789000000
    assert Integer(Float('1.123456789012345678901234567890e20', '')) == \
        112345678901234567890
    assert Integer(Float('1.123456789012345678901234567890e25', '')) == \
        11234567890123456789012345
    # decimal forces float so it's not an exact integer ending in 000000
    assert Integer(Float('1.123456789012345678901234567890e35', '')) == \
        112345678901234567890123456789000192
    assert Integer(Float('123456789012345678901234567890e5', '')) == \
        12345678901234567890123456789000000
    assert same_and_same_prec(Float('123000e-2',''), Float('1230.00', ''))
    assert same_and_same_prec(Float('123000e2',''), Float('12300000', ''))

    assert int(1 + Rational('.9999999999999999999999999')) == 1
    assert int(pi/1e20) == 0
    assert int(1 + pi/1e20) == 1
    assert int(Add(1.2, -2, evaluate=False)) == int(1.2 - 2)
    assert int(Add(1.2, +2, evaluate=False)) == int(1.2 + 2)
    assert int(Add(1 + Float('.99999999999999999', ''), evaluate=False)) == 1
    raises(TypeError, lambda: float(x))
    raises(TypeError, lambda: float(sqrt(-1)))

    assert int(12345678901234567890 + cos(1)**2 + sin(1)**2) == \
        12345678901234567891


def test_issue_6611a():
    assert Mul.flatten([3**Rational(1, 3),
        Pow(-Rational(1, 9), Rational(2, 3), evaluate=False)]) == \
        ([Rational(1, 3), (-1)**Rational(2, 3)], [], None)


def test_denest_add_mul():
    # when working with evaluated expressions make sure they denest
    eq = x + 1
    eq = Add(eq, 2, evaluate=False)
    eq = Add(eq, 2, evaluate=False)
    assert Add(*eq.args) == x + 5
    eq = x*2
    eq = Mul(eq, 2, evaluate=False)
    eq = Mul(eq, 2, evaluate=False)
    assert Mul(*eq.args) == 8*x
    # but don't let them denest unnecessarily
    eq = Mul(-2, x - 2, evaluate=False)
    assert 2*eq == Mul(-4, x - 2, evaluate=False)
    assert -eq == Mul(2, x - 2, evaluate=False)


def test_mul_coeff():
    # It is important that all Numbers be removed from the seq;
    # This can be tricky when powers combine to produce those numbers
    p = exp(I*pi/3)
    assert p**2*x*p*y*p*x*p**2 == x**2*y


def test_mul_zero_detection():
    nz = Dummy(real=True, zero=False)
    r = Dummy(extended_real=True)
    c = Dummy(real=False, complex=True)
    c2 = Dummy(real=False, complex=True)
    i = Dummy(imaginary=True)
    e = nz*r*c
    assert e.is_imaginary is None
    assert e.is_extended_real is None
    e = nz*c
    assert e.is_imaginary is None
    assert e.is_extended_real is False
    e = nz*i*c
    assert e.is_imaginary is False
    assert e.is_extended_real is None
    # check for more than one complex; it is important to use
    # uniquely named Symbols to ensure that two factors appear
    # e.g. if the symbols have the same name they just become
    # a single factor, a power.
    e = nz*i*c*c2
    assert e.is_imaginary is None
    assert e.is_extended_real is None

    # _eval_is_extended_real and _eval_is_zero both employ trapping of the
    # zero value so args should be tested in both directions and
    # TO AVOID GETTING THE CACHED RESULT, Dummy MUST BE USED

    # real is unknown
    def test(z, b, e):
        if z.is_zero and b.is_finite:
            assert e.is_extended_real and e.is_zero
        else:
            assert e.is_extended_real is None
            if b.is_finite:
                if z.is_zero:
                    assert e.is_zero
                else:
                    assert e.is_zero is None
            elif b.is_finite is False:
                if z.is_zero is None:
                    assert e.is_zero is None
                else:
                    assert e.is_zero is False


    for iz, ib in product(*[[True, False, None]]*2):
        z = Dummy('z', nonzero=iz)
        b = Dummy('f', finite=ib)
        e = Mul(z, b, evaluate=False)
        test(z, b, e)
        z = Dummy('nz', nonzero=iz)
        b = Dummy('f', finite=ib)
        e = Mul(b, z, evaluate=False)
        test(z, b, e)

    # real is True
    def test(z, b, e):
        if z.is_zero and not b.is_finite:
            assert e.is_extended_real is None
        else:
            assert e.is_extended_real is True

    for iz, ib in product(*[[True, False, None]]*2):
        z = Dummy('z', nonzero=iz, extended_real=True)
        b = Dummy('b', finite=ib, extended_real=True)
        e = Mul(z, b, evaluate=False)
        test(z, b, e)
        z = Dummy('z', nonzero=iz, extended_real=True)
        b = Dummy('b', finite=ib, extended_real=True)
        e = Mul(b, z, evaluate=False)
        test(z, b, e)


def test_Mul_with_zero_infinite():
    zer = Dummy(zero=True)
    inf = Dummy(finite=False)

    e = Mul(zer, inf, evaluate=False)
    assert e.is_extended_positive is None
    assert e.is_hermitian is None

    e = Mul(inf, zer, evaluate=False)
    assert e.is_extended_positive is None
    assert e.is_hermitian is None


def test_Mul_does_not_cancel_infinities():
    a, b = symbols('a b')
    assert ((zoo + 3*a)/(3*a + zoo)) is nan
    assert ((b - oo)/(b - oo)) is nan
    # issue 13904
    expr = (1/(a+b) + 1/(a-b))/(1/(a+b) - 1/(a-b))
    assert expr.subs(b, a) is nan


def test_Mul_does_not_distribute_infinity():
    a, b = symbols('a b')
    assert ((1 + I)*oo).is_Mul
    assert ((a + b)*(-oo)).is_Mul
    assert ((a + 1)*zoo).is_Mul
    assert ((1 + I)*oo).is_finite is False
    z = (1 + I)*oo
    assert ((1 - I)*z).expand() is oo


def test_issue_8247_8354():
    from sympy.functions.elementary.trigonometric import tan
    z = sqrt(1 + sqrt(3)) + sqrt(3 + 3*sqrt(3)) - sqrt(10 + 6*sqrt(3))
    assert z.is_positive is False  # it's 0
    z = S('''-2**(1/3)*(3*sqrt(93) + 29)**2 - 4*(3*sqrt(93) + 29)**(4/3) +
        12*sqrt(93)*(3*sqrt(93) + 29)**(1/3) + 116*(3*sqrt(93) + 29)**(1/3) +
        174*2**(1/3)*sqrt(93) + 1678*2**(1/3)''')
    assert z.is_positive is False  # it's 0
    z = 2*(-3*tan(19*pi/90) + sqrt(3))*cos(11*pi/90)*cos(19*pi/90) - \
        sqrt(3)*(-3 + 4*cos(19*pi/90)**2)
    assert z.is_positive is not True  # it's zero and it shouldn't hang
    z = S('''9*(3*sqrt(93) + 29)**(2/3)*((3*sqrt(93) +
        29)**(1/3)*(-2**(2/3)*(3*sqrt(93) + 29)**(1/3) - 2) - 2*2**(1/3))**3 +
        72*(3*sqrt(93) + 29)**(2/3)*(81*sqrt(93) + 783) + (162*sqrt(93) +
        1566)*((3*sqrt(93) + 29)**(1/3)*(-2**(2/3)*(3*sqrt(93) + 29)**(1/3) -
        2) - 2*2**(1/3))**2''')
    assert z.is_positive is False  # it's 0 (and a single _mexpand isn't enough)


def test_Add_is_zero():
    x, y = symbols('x y', zero=True)
    assert (x + y).is_zero

    # Issue 15873
    e = -2*I + (1 + I)**2
    assert e.is_zero is None


def test_issue_14392():
    assert (sin(zoo)**2).as_real_imag() == (nan, nan)


def test_divmod():
    assert divmod(x, y) == (x//y, x % y)
    assert divmod(x, 3) == (x//3, x % 3)
    assert divmod(3, x) == (3//x, 3 % x)


def test__neg__():
    assert -(x*y) == -x*y
    assert -(-x*y) == x*y
    assert -(1.*x) == -1.*x
    assert -(-1.*x) == 1.*x
    assert -(2.*x) == -2.*x
    assert -(-2.*x) == 2.*x
    with distribute(False):
        eq = -(x + y)
        assert eq.is_Mul and eq.args == (-1, x + y)
    with evaluate(False):
        eq = -(x + y)
        assert eq.is_Mul and eq.args == (-1, x + y)


def test_issue_18507():
    assert Mul(zoo, zoo, 0) is nan


def test_issue_17130():
    e = Add(b, -b, I, -I, evaluate=False)
    assert e.is_zero is None # ideally this would be True


def test_issue_21034():
    e = -I*log((re(asin(5)) + I*im(asin(5)))/sqrt(re(asin(5))**2 + im(asin(5))**2))/pi
    assert e.round(2)


def test_issue_22021():
    from sympy.calculus.accumulationbounds import AccumBounds
    # these objects are special cases in Mul
    from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
    L = TensorIndexType("L")
    i = tensor_indices("i", L)
    A, B = tensor_heads("A B", [L])
    e = A(i) + B(i)
    assert -e == -1*e
    e = zoo + x
    assert -e == -1*e
    a = AccumBounds(1, 2)
    e = a + x
    assert -e == -1*e
    for args in permutations((zoo, a, x)):
        e = Add(*args, evaluate=False)
        assert -e == -1*e
    assert 2*Add(1, x, x, evaluate=False) == 4*x + 2


def test_issue_22244():
    assert -(zoo*x) == zoo*x


def test_issue_22453():
    from sympy.utilities.iterables import cartes
    e = Symbol('e', extended_positive=True)
    for a, b in cartes(*[[oo, -oo, 3]]*2):
        if a == b == 3:
            continue
        i = a + I*b
        assert i**(1 + e) is S.ComplexInfinity
        assert i**-e is S.Zero
        assert unchanged(Pow, i, e)
    assert 1/(oo + I*oo) is S.Zero
    r, i = [Dummy(infinite=True, extended_real=True) for _ in range(2)]
    assert 1/(r + I*i) is S.Zero
    assert 1/(3 + I*i) is S.Zero
    assert 1/(r + I*3) is S.Zero


def test_issue_22613():
    assert (0**(x - 2)).as_content_primitive() == (1, 0**(x - 2))
    assert (0**(x + 2)).as_content_primitive() == (1, 0**(x + 2))


def test_issue_25176():
    assert sqrt(-4*3**(S(3)/4)*I/3) == 2*3**(S(7)/8)*sqrt(-I)/3
