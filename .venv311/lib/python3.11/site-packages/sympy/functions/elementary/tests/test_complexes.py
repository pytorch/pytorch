from sympy.core.function import (Derivative, Function, Lambda, expand, PoleError)
from sympy.core.numbers import (E, I, Rational, comp, nan, oo, pi, zoo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, adjoint, arg, conjugate, im, re, sign, transpose)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, atan, atan2, cos, sin)
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.special.delta_functions import (DiracDelta, Heaviside)
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.funcmatrix import FunctionMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.immutable import (ImmutableMatrix, ImmutableSparseMatrix)
from sympy.matrices import SparseMatrix
from sympy.sets.sets import Interval
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.series.order import Order
from sympy.testing.pytest import XFAIL, raises, _both_exp_pow


def N_equals(a, b):
    """Check whether two complex numbers are numerically close"""
    return comp(a.n(), b.n(), 1.e-6)


def test_re():
    x, y = symbols('x,y')
    a, b = symbols('a,b', real=True)

    r = Symbol('r', real=True)
    i = Symbol('i', imaginary=True)

    assert re(nan) is nan

    assert re(oo) is oo
    assert re(-oo) is -oo

    assert re(0) == 0

    assert re(1) == 1
    assert re(-1) == -1

    assert re(E) == E
    assert re(-E) == -E

    assert unchanged(re, x)
    assert re(x*I) == -im(x)
    assert re(r*I) == 0
    assert re(r) == r
    assert re(i*I) == I * i
    assert re(i) == 0

    assert re(x + y) == re(x) + re(y)
    assert re(x + r) == re(x) + r

    assert re(re(x)) == re(x)

    assert re(2 + I) == 2
    assert re(x + I) == re(x)

    assert re(x + y*I) == re(x) - im(y)
    assert re(x + r*I) == re(x)

    assert re(log(2*I)) == log(2)

    assert re((2 + I)**2).expand(complex=True) == 3

    assert re(conjugate(x)) == re(x)
    assert conjugate(re(x)) == re(x)

    assert re(x).as_real_imag() == (re(x), 0)

    assert re(i*r*x).diff(r) == re(i*x)
    assert re(i*r*x).diff(i) == I*r*im(x)

    assert re(
        sqrt(a + b*I)) == (a**2 + b**2)**Rational(1, 4)*cos(atan2(b, a)/2)
    assert re(a * (2 + b*I)) == 2*a

    assert re((1 + sqrt(a + b*I))/2) == \
        (a**2 + b**2)**Rational(1, 4)*cos(atan2(b, a)/2)/2 + S.Half

    assert re(x).rewrite(im) == x - S.ImaginaryUnit*im(x)
    assert (x + re(y)).rewrite(re, im) == x + y - S.ImaginaryUnit*im(y)

    a = Symbol('a', algebraic=True)
    t = Symbol('t', transcendental=True)
    x = Symbol('x')
    assert re(a).is_algebraic
    assert re(x).is_algebraic is None
    assert re(t).is_algebraic is False

    assert re(S.ComplexInfinity) is S.NaN

    n, m, l = symbols('n m l')
    A = MatrixSymbol('A',n,m)
    assert re(A) == (S.Half) * (A + conjugate(A))

    A = Matrix([[1 + 4*I,2],[0, -3*I]])
    assert re(A) == Matrix([[1, 2],[0, 0]])

    A = ImmutableMatrix([[1 + 3*I, 3-2*I],[0, 2*I]])
    assert re(A) == ImmutableMatrix([[1, 3],[0, 0]])

    X = SparseMatrix([[2*j + i*I for i in range(5)] for j in range(5)])
    assert re(X) - Matrix([[0, 0, 0, 0, 0],
                           [2, 2, 2, 2, 2],
                           [4, 4, 4, 4, 4],
                           [6, 6, 6, 6, 6],
                           [8, 8, 8, 8, 8]]) == Matrix.zeros(5)

    assert im(X) - Matrix([[0, 1, 2, 3, 4],
                           [0, 1, 2, 3, 4],
                           [0, 1, 2, 3, 4],
                           [0, 1, 2, 3, 4],
                           [0, 1, 2, 3, 4]]) == Matrix.zeros(5)

    X = FunctionMatrix(3, 3, Lambda((n, m), n + m*I))
    assert re(X) == Matrix([[0, 0, 0], [1, 1, 1], [2, 2, 2]])


def test_im():
    x, y = symbols('x,y')
    a, b = symbols('a,b', real=True)

    r = Symbol('r', real=True)
    i = Symbol('i', imaginary=True)

    assert im(nan) is nan

    assert im(oo*I) is oo
    assert im(-oo*I) is -oo

    assert im(0) == 0

    assert im(1) == 0
    assert im(-1) == 0

    assert im(E*I) == E
    assert im(-E*I) == -E

    assert unchanged(im, x)
    assert im(x*I) == re(x)
    assert im(r*I) == r
    assert im(r) == 0
    assert im(i*I) == 0
    assert im(i) == -I * i

    assert im(x + y) == im(x) + im(y)
    assert im(x + r) == im(x)
    assert im(x + r*I) == im(x) + r

    assert im(im(x)*I) == im(x)

    assert im(2 + I) == 1
    assert im(x + I) == im(x) + 1

    assert im(x + y*I) == im(x) + re(y)
    assert im(x + r*I) == im(x) + r

    assert im(log(2*I)) == pi/2

    assert im((2 + I)**2).expand(complex=True) == 4

    assert im(conjugate(x)) == -im(x)
    assert conjugate(im(x)) == im(x)

    assert im(x).as_real_imag() == (im(x), 0)

    assert im(i*r*x).diff(r) == im(i*x)
    assert im(i*r*x).diff(i) == -I * re(r*x)

    assert im(
        sqrt(a + b*I)) == (a**2 + b**2)**Rational(1, 4)*sin(atan2(b, a)/2)
    assert im(a * (2 + b*I)) == a*b

    assert im((1 + sqrt(a + b*I))/2) == \
        (a**2 + b**2)**Rational(1, 4)*sin(atan2(b, a)/2)/2

    assert im(x).rewrite(re) == -S.ImaginaryUnit * (x - re(x))
    assert (x + im(y)).rewrite(im, re) == x - S.ImaginaryUnit * (y - re(y))

    a = Symbol('a', algebraic=True)
    t = Symbol('t', transcendental=True)
    x = Symbol('x')
    assert re(a).is_algebraic
    assert re(x).is_algebraic is None
    assert re(t).is_algebraic is False

    assert im(S.ComplexInfinity) is S.NaN

    n, m, l = symbols('n m l')
    A = MatrixSymbol('A',n,m)

    assert im(A) == (S.One/(2*I)) * (A - conjugate(A))

    A = Matrix([[1 + 4*I, 2],[0, -3*I]])
    assert im(A) == Matrix([[4, 0],[0, -3]])

    A = ImmutableMatrix([[1 + 3*I, 3-2*I],[0, 2*I]])
    assert im(A) == ImmutableMatrix([[3, -2],[0, 2]])

    X = ImmutableSparseMatrix(
            [[i*I + i for i in range(5)] for i in range(5)])
    Y = SparseMatrix([list(range(5)) for i in range(5)])
    assert im(X).as_immutable() == Y

    X = FunctionMatrix(3, 3, Lambda((n, m), n + m*I))
    assert im(X) == Matrix([[0, 1, 2], [0, 1, 2], [0, 1, 2]])

def test_sign():
    assert sign(1.2) == 1
    assert sign(-1.2) == -1
    assert sign(3*I) == I
    assert sign(-3*I) == -I
    assert sign(0) == 0
    assert sign(0, evaluate=False).doit() == 0
    assert sign(oo, evaluate=False).doit() == 1
    assert sign(nan) is nan
    assert sign(2 + 2*I).doit() == sqrt(2)*(2 + 2*I)/4
    assert sign(2 + 3*I).simplify() == sign(2 + 3*I)
    assert sign(2 + 2*I).simplify() == sign(1 + I)
    assert sign(im(sqrt(1 - sqrt(3)))) == 1
    assert sign(sqrt(1 - sqrt(3))) == I

    x = Symbol('x')
    assert sign(x).is_finite is True
    assert sign(x).is_complex is True
    assert sign(x).is_imaginary is None
    assert sign(x).is_integer is None
    assert sign(x).is_real is None
    assert sign(x).is_zero is None
    assert sign(x).doit() == sign(x)
    assert sign(1.2*x) == sign(x)
    assert sign(2*x) == sign(x)
    assert sign(I*x) == I*sign(x)
    assert sign(-2*I*x) == -I*sign(x)
    assert sign(conjugate(x)) == conjugate(sign(x))

    p = Symbol('p', positive=True)
    n = Symbol('n', negative=True)
    m = Symbol('m', negative=True)
    assert sign(2*p*x) == sign(x)
    assert sign(n*x) == -sign(x)
    assert sign(n*m*x) == sign(x)

    x = Symbol('x', imaginary=True)
    assert sign(x).is_imaginary is True
    assert sign(x).is_integer is False
    assert sign(x).is_real is False
    assert sign(x).is_zero is False
    assert sign(x).diff(x) == 2*DiracDelta(-I*x)
    assert sign(x).doit() == x / Abs(x)
    assert conjugate(sign(x)) == -sign(x)

    x = Symbol('x', real=True)
    assert sign(x).is_imaginary is False
    assert sign(x).is_integer is True
    assert sign(x).is_real is True
    assert sign(x).is_zero is None
    assert sign(x).diff(x) == 2*DiracDelta(x)
    assert sign(x).doit() == sign(x)
    assert conjugate(sign(x)) == sign(x)

    x = Symbol('x', nonzero=True)
    assert sign(x).is_imaginary is False
    assert sign(x).is_integer is True
    assert sign(x).is_real is True
    assert sign(x).is_zero is False
    assert sign(x).doit() == x / Abs(x)
    assert sign(Abs(x)) == 1
    assert Abs(sign(x)) == 1

    x = Symbol('x', positive=True)
    assert sign(x).is_imaginary is False
    assert sign(x).is_integer is True
    assert sign(x).is_real is True
    assert sign(x).is_zero is False
    assert sign(x).doit() == x / Abs(x)
    assert sign(Abs(x)) == 1
    assert Abs(sign(x)) == 1

    x = 0
    assert sign(x).is_imaginary is False
    assert sign(x).is_integer is True
    assert sign(x).is_real is True
    assert sign(x).is_zero is True
    assert sign(x).doit() == 0
    assert sign(Abs(x)) == 0
    assert Abs(sign(x)) == 0

    nz = Symbol('nz', nonzero=True, integer=True)
    assert sign(nz).is_imaginary is False
    assert sign(nz).is_integer is True
    assert sign(nz).is_real is True
    assert sign(nz).is_zero is False
    assert sign(nz)**2 == 1
    assert (sign(nz)**3).args == (sign(nz), 3)

    assert sign(Symbol('x', nonnegative=True)).is_nonnegative
    assert sign(Symbol('x', nonnegative=True)).is_nonpositive is None
    assert sign(Symbol('x', nonpositive=True)).is_nonnegative is None
    assert sign(Symbol('x', nonpositive=True)).is_nonpositive
    assert sign(Symbol('x', real=True)).is_nonnegative is None
    assert sign(Symbol('x', real=True)).is_nonpositive is None
    assert sign(Symbol('x', real=True, zero=False)).is_nonpositive is None

    x, y = Symbol('x', real=True), Symbol('y')
    f = Function('f')
    assert sign(x).rewrite(Piecewise) == \
        Piecewise((1, x > 0), (-1, x < 0), (0, True))
    assert sign(y).rewrite(Piecewise) == sign(y)
    assert sign(x).rewrite(Heaviside) == 2*Heaviside(x, H0=S(1)/2) - 1
    assert sign(y).rewrite(Heaviside) == sign(y)
    assert sign(y).rewrite(Abs) == Piecewise((0, Eq(y, 0)), (y/Abs(y), True))
    assert sign(f(y)).rewrite(Abs) == Piecewise((0, Eq(f(y), 0)), (f(y)/Abs(f(y)), True))

    # evaluate what can be evaluated
    assert sign(exp_polar(I*pi)*pi) is S.NegativeOne

    eq = -sqrt(10 + 6*sqrt(3)) + sqrt(1 + sqrt(3)) + sqrt(3 + 3*sqrt(3))
    # if there is a fast way to know when and when you cannot prove an
    # expression like this is zero then the equality to zero is ok
    assert sign(eq).func is sign or sign(eq) == 0
    # but sometimes it's hard to do this so it's better not to load
    # abs down with tests that will be very slow
    q = 1 + sqrt(2) - 2*sqrt(3) + 1331*sqrt(6)
    p = expand(q**3)**Rational(1, 3)
    d = p - q
    assert sign(d).func is sign or sign(d) == 0


def test_as_real_imag():
    n = pi**1000
    # the special code for working out the real
    # and complex parts of a power with Integer exponent
    # should not run if there is no imaginary part, hence
    # this should not hang
    assert n.as_real_imag() == (n, 0)

    # issue 6261
    x = Symbol('x')
    assert sqrt(x).as_real_imag() == \
        ((re(x)**2 + im(x)**2)**Rational(1, 4)*cos(atan2(im(x), re(x))/2),
     (re(x)**2 + im(x)**2)**Rational(1, 4)*sin(atan2(im(x), re(x))/2))

    # issue 3853
    a, b = symbols('a,b', real=True)
    assert ((1 + sqrt(a + b*I))/2).as_real_imag() == \
           (
               (a**2 + b**2)**Rational(
                   1, 4)*cos(atan2(b, a)/2)/2 + S.Half,
               (a**2 + b**2)**Rational(1, 4)*sin(atan2(b, a)/2)/2)

    assert sqrt(a**2).as_real_imag() == (sqrt(a**2), 0)
    i = symbols('i', imaginary=True)
    assert sqrt(i**2).as_real_imag() == (0, abs(i))

    assert ((1 + I)/(1 - I)).as_real_imag() == (0, 1)
    assert ((1 + I)**3/(1 - I)).as_real_imag() == (-2, 0)


@XFAIL
def test_sign_issue_3068():
    n = pi**1000
    i = int(n)
    x = Symbol('x')
    assert (n - i).round() == 1  # doesn't hang
    assert sign(n - i) == 1
    # perhaps it's not possible to get the sign right when
    # only 1 digit is being requested for this situation;
    # 2 digits works
    assert (n - x).n(1, subs={x: i}) > 0
    assert (n - x).n(2, subs={x: i}) > 0


def test_Abs():
    raises(TypeError, lambda: Abs(Interval(2, 3)))  # issue 8717

    x, y = symbols('x,y')
    assert sign(sign(x)) == sign(x)
    assert sign(x*y).func is sign
    assert Abs(0) == 0
    assert Abs(1) == 1
    assert Abs(-1) == 1
    assert Abs(I) == 1
    assert Abs(-I) == 1
    assert Abs(nan) is nan
    assert Abs(zoo) is oo
    assert Abs(I * pi) == pi
    assert Abs(-I * pi) == pi
    assert Abs(I * x) == Abs(x)
    assert Abs(-I * x) == Abs(x)
    assert Abs(-2*x) == 2*Abs(x)
    assert Abs(-2.0*x) == 2.0*Abs(x)
    assert Abs(2*pi*x*y) == 2*pi*Abs(x*y)
    assert Abs(conjugate(x)) == Abs(x)
    assert conjugate(Abs(x)) == Abs(x)
    assert Abs(x).expand(complex=True) == sqrt(re(x)**2 + im(x)**2)

    a = Symbol('a', positive=True)
    assert Abs(2*pi*x*a) == 2*pi*a*Abs(x)
    assert Abs(2*pi*I*x*a) == 2*pi*a*Abs(x)

    x = Symbol('x', real=True)
    n = Symbol('n', integer=True)
    assert Abs((-1)**n) == 1
    assert x**(2*n) == Abs(x)**(2*n)
    assert Abs(x).diff(x) == sign(x)
    assert abs(x) == Abs(x)  # Python built-in
    assert Abs(x)**3 == x**2*Abs(x)
    assert Abs(x)**4 == x**4
    assert (
        Abs(x)**(3*n)).args == (Abs(x), 3*n)  # leave symbolic odd unchanged
    assert (1/Abs(x)).args == (Abs(x), -1)
    assert 1/Abs(x)**3 == 1/(x**2*Abs(x))
    assert Abs(x)**-3 == Abs(x)/(x**4)
    assert Abs(x**3) == x**2*Abs(x)
    assert Abs(I**I) == exp(-pi/2)
    assert Abs((4 + 5*I)**(6 + 7*I)) == 68921*exp(-7*atan(Rational(5, 4)))
    y = Symbol('y', real=True)
    assert Abs(I**y) == 1
    y = Symbol('y')
    assert Abs(I**y) == exp(-pi*im(y)/2)

    x = Symbol('x', imaginary=True)
    assert Abs(x).diff(x) == -sign(x)

    eq = -sqrt(10 + 6*sqrt(3)) + sqrt(1 + sqrt(3)) + sqrt(3 + 3*sqrt(3))
    # if there is a fast way to know when you can and when you cannot prove an
    # expression like this is zero then the equality to zero is ok
    assert abs(eq).func is Abs or abs(eq) == 0
    # but sometimes it's hard to do this so it's better not to load
    # abs down with tests that will be very slow
    q = 1 + sqrt(2) - 2*sqrt(3) + 1331*sqrt(6)
    p = expand(q**3)**Rational(1, 3)
    d = p - q
    assert abs(d).func is Abs or abs(d) == 0

    assert Abs(4*exp(pi*I/4)) == 4
    assert Abs(3**(2 + I)) == 9
    assert Abs((-3)**(1 - I)) == 3*exp(pi)

    assert Abs(oo) is oo
    assert Abs(-oo) is oo
    assert Abs(oo + I) is oo
    assert Abs(oo + I*oo) is oo

    a = Symbol('a', algebraic=True)
    t = Symbol('t', transcendental=True)
    x = Symbol('x')
    assert re(a).is_algebraic
    assert re(x).is_algebraic is None
    assert re(t).is_algebraic is False
    assert Abs(x).fdiff() == sign(x)
    raises(ArgumentIndexError, lambda: Abs(x).fdiff(2))

    # doesn't have recursion error
    arg = sqrt(acos(1 - I)*acos(1 + I))
    assert abs(arg) == arg

    # special handling to put Abs in denom
    assert abs(1/x) == 1/Abs(x)
    e = abs(2/x**2)
    assert e.is_Mul and e == 2/Abs(x**2)
    assert unchanged(Abs, y/x)
    assert unchanged(Abs, x/(x + 1))
    assert unchanged(Abs, x*y)
    p = Symbol('p', positive=True)
    assert abs(x/p) == abs(x)/p

    # coverage
    assert unchanged(Abs, Symbol('x', real=True)**y)
    # issue 19627
    f = Function('f', positive=True)
    assert sqrt(f(x)**2) == f(x)
    # issue 21625
    assert unchanged(Abs, S("im(acos(-i + acosh(-g + i)))"))


def test_Abs_rewrite():
    x = Symbol('x', real=True)
    a = Abs(x).rewrite(Heaviside).expand()
    assert a == x*Heaviside(x) - x*Heaviside(-x)
    for i in [-2, -1, 0, 1, 2]:
        assert a.subs(x, i) == abs(i)
    y = Symbol('y')
    assert Abs(y).rewrite(Heaviside) == Abs(y)

    x, y = Symbol('x', real=True), Symbol('y')
    assert Abs(x).rewrite(Piecewise) == Piecewise((x, x >= 0), (-x, True))
    assert Abs(y).rewrite(Piecewise) == Abs(y)
    assert Abs(y).rewrite(sign) == y/sign(y)

    i = Symbol('i', imaginary=True)
    assert abs(i).rewrite(Piecewise) == Piecewise((I*i, I*i >= 0), (-I*i, True))


    assert Abs(y).rewrite(conjugate) == sqrt(y*conjugate(y))
    assert Abs(i).rewrite(conjugate) == sqrt(-i**2) #  == -I*i

    y = Symbol('y', extended_real=True)
    assert  (Abs(exp(-I*x)-exp(-I*y))**2).rewrite(conjugate) == \
        -exp(I*x)*exp(-I*y) + 2 - exp(-I*x)*exp(I*y)


def test_Abs_real():
    # test some properties of abs that only apply
    # to real numbers
    x = Symbol('x', complex=True)
    assert sqrt(x**2) != Abs(x)
    assert Abs(x**2) != x**2

    x = Symbol('x', real=True)
    assert sqrt(x**2) == Abs(x)
    assert Abs(x**2) == x**2

    # if the symbol is zero, the following will still apply
    nn = Symbol('nn', nonnegative=True, real=True)
    np = Symbol('np', nonpositive=True, real=True)
    assert Abs(nn) == nn
    assert Abs(np) == -np


def test_Abs_properties():
    x = Symbol('x')
    assert Abs(x).is_real is None
    assert Abs(x).is_extended_real is True
    assert Abs(x).is_rational is None
    assert Abs(x).is_positive is None
    assert Abs(x).is_nonnegative is None
    assert Abs(x).is_extended_positive is None
    assert Abs(x).is_extended_nonnegative is True

    f = Symbol('x', finite=True)
    assert Abs(f).is_real is True
    assert Abs(f).is_extended_real is True
    assert Abs(f).is_rational is None
    assert Abs(f).is_positive is None
    assert Abs(f).is_nonnegative is True
    assert Abs(f).is_extended_positive is None
    assert Abs(f).is_extended_nonnegative is True

    z = Symbol('z', complex=True, zero=False)
    assert Abs(z).is_real is True # since complex implies finite
    assert Abs(z).is_extended_real is True
    assert Abs(z).is_rational is None
    assert Abs(z).is_positive is True
    assert Abs(z).is_extended_positive is True
    assert Abs(z).is_zero is False

    p = Symbol('p', positive=True)
    assert Abs(p).is_real is True
    assert Abs(p).is_extended_real is True
    assert Abs(p).is_rational is None
    assert Abs(p).is_positive is True
    assert Abs(p).is_zero is False

    q = Symbol('q', rational=True)
    assert Abs(q).is_real is True
    assert Abs(q).is_rational is True
    assert Abs(q).is_integer is None
    assert Abs(q).is_positive is None
    assert Abs(q).is_nonnegative is True

    i = Symbol('i', integer=True)
    assert Abs(i).is_real is True
    assert Abs(i).is_integer is True
    assert Abs(i).is_positive is None
    assert Abs(i).is_nonnegative is True

    e = Symbol('n', even=True)
    ne = Symbol('ne', real=True, even=False)
    assert Abs(e).is_even is True
    assert Abs(ne).is_even is False
    assert Abs(i).is_even is None

    o = Symbol('n', odd=True)
    no = Symbol('no', real=True, odd=False)
    assert Abs(o).is_odd is True
    assert Abs(no).is_odd is False
    assert Abs(i).is_odd is None


def test_abs():
    # this tests that abs calls Abs; don't rename to
    # test_Abs since that test is already above
    a = Symbol('a', positive=True)
    assert abs(I*(1 + a)**2) == (1 + a)**2


def test_arg():
    assert arg(0) is nan
    assert arg(1) == 0
    assert arg(-1) == pi
    assert arg(I) == pi/2
    assert arg(-I) == -pi/2
    assert arg(1 + I) == pi/4
    assert arg(-1 + I) == pi*Rational(3, 4)
    assert arg(1 - I) == -pi/4
    assert arg(exp_polar(4*pi*I)) == 4*pi
    assert arg(exp_polar(-7*pi*I)) == -7*pi
    assert arg(exp_polar(5 - 3*pi*I/4)) == pi*Rational(-3, 4)

    assert arg(exp(I*pi/7)) == pi/7     # issue 17300
    assert arg(exp(16*I)) == 16 - 6*pi
    assert arg(exp(13*I*pi/12)) == -11*pi/12
    assert arg(exp(123 - 5*I)) == -5 + 2*pi
    assert arg(exp(sin(1 + 3*I))) == -2*pi + cos(1)*sinh(3)
    r = Symbol('r', real=True)
    assert arg(exp(r - 2*I)) == -2

    f = Function('f')
    assert not arg(f(0) + I*f(1)).atoms(re)

    # check nesting
    x = Symbol('x')
    assert arg(arg(arg(x))) is not S.NaN
    assert arg(arg(arg(arg(x)))) is S.NaN
    r = Symbol('r', extended_real=True)
    assert arg(arg(r)) is not S.NaN
    assert arg(arg(arg(r))) is S.NaN

    p = Function('p', extended_positive=True)
    assert arg(p(x)) == 0
    assert arg((3 + I)*p(x)) == arg(3  + I)

    p = Symbol('p', positive=True)
    assert arg(p) == 0
    assert arg(p*I) == pi/2

    n = Symbol('n', negative=True)
    assert arg(n) == pi
    assert arg(n*I) == -pi/2

    x = Symbol('x')
    assert conjugate(arg(x)) == arg(x)

    e = p + I*p**2
    assert arg(e) == arg(1 + p*I)
    # make sure sign doesn't swap
    e = -2*p + 4*I*p**2
    assert arg(e) == arg(-1 + 2*p*I)
    # make sure sign isn't lost
    x = symbols('x', real=True)  # could be zero
    e = x + I*x
    assert arg(e) == arg(x*(1 + I))
    assert arg(e/p) == arg(x*(1 + I))
    e = p*cos(p) + I*log(p)*exp(p)
    assert arg(e).args[0] == e
    # keep it simple -- let the user do more advanced cancellation
    e = (p + 1) + I*(p**2 - 1)
    assert arg(e).args[0] == e

    f = Function('f')
    e = 2*x*(f(0) - 1) - 2*x*f(0)
    assert arg(e) == arg(-2*x)
    assert arg(f(0)).func == arg and arg(f(0)).args == (f(0),)


def test_arg_rewrite():
    assert arg(1 + I) == atan2(1, 1)

    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    assert arg(x + I*y).rewrite(atan2) == atan2(y, x)


def test_arg_leading_term_and_series():
    x = Symbol('x')
    assert arg(x).as_leading_term(x, cdir = 1) == 0
    assert arg(x).as_leading_term(x, cdir = -1) == pi
    raises(PoleError, lambda: arg(x + I).as_leading_term(x, cdir = 1))
    raises(PoleError, lambda: arg(2*x).as_leading_term(x, cdir = I))

    assert arg(x).nseries(x) == 0
    assert arg(x).nseries(x, n=0) == Order(1)


def test_adjoint():
    a = Symbol('a', antihermitian=True)
    b = Symbol('b', hermitian=True)
    assert adjoint(a) == -a
    assert adjoint(I*a) == I*a
    assert adjoint(b) == b
    assert adjoint(I*b) == -I*b
    assert adjoint(a*b) == -b*a
    assert adjoint(I*a*b) == I*b*a

    x, y = symbols('x y')
    assert adjoint(adjoint(x)) == x
    assert adjoint(x + y) == conjugate(x) + conjugate(y)
    assert adjoint(x - y) == conjugate(x) - conjugate(y)
    assert adjoint(x * y) == conjugate(x) * conjugate(y)
    assert adjoint(x / y) == conjugate(x) / conjugate(y)
    assert adjoint(-x) == -conjugate(x)

    x, y = symbols('x y', commutative=False)
    assert adjoint(adjoint(x)) == x
    assert adjoint(x + y) == adjoint(x) + adjoint(y)
    assert adjoint(x - y) == adjoint(x) - adjoint(y)
    assert adjoint(x * y) == adjoint(y) * adjoint(x)
    assert adjoint(x / y) == 1 / adjoint(y) * adjoint(x)
    assert adjoint(-x) == -adjoint(x)


def test_conjugate():
    a = Symbol('a', real=True)
    b = Symbol('b', imaginary=True)
    assert conjugate(a) == a
    assert conjugate(I*a) == -I*a
    assert conjugate(b) == -b
    assert conjugate(I*b) == I*b
    assert conjugate(a*b) == -a*b
    assert conjugate(I*a*b) == I*a*b

    x, y = symbols('x y')
    assert conjugate(conjugate(x)) == x
    assert conjugate(x).inverse() == conjugate
    assert conjugate(x + y) == conjugate(x) + conjugate(y)
    assert conjugate(x - y) == conjugate(x) - conjugate(y)
    assert conjugate(x * y) == conjugate(x) * conjugate(y)
    assert conjugate(x / y) == conjugate(x) / conjugate(y)
    assert conjugate(-x) == -conjugate(x)

    a = Symbol('a', algebraic=True)
    t = Symbol('t', transcendental=True)
    assert re(a).is_algebraic
    assert re(x).is_algebraic is None
    assert re(t).is_algebraic is False


def test_conjugate_transpose():
    x = Symbol('x', commutative=False)
    assert conjugate(transpose(x)) == adjoint(x)
    assert transpose(conjugate(x)) == adjoint(x)
    assert adjoint(transpose(x)) == conjugate(x)
    assert transpose(adjoint(x)) == conjugate(x)
    assert adjoint(conjugate(x)) == transpose(x)
    assert conjugate(adjoint(x)) == transpose(x)

    x = Symbol('x')
    assert conjugate(x) == adjoint(x)
    assert transpose(x) == x


def test_transpose():
    a = Symbol('a', complex=True)
    assert transpose(a) == a
    assert transpose(I*a) == I*a

    x, y = symbols('x y')
    assert transpose(transpose(x)) == x
    assert transpose(x + y) == x + y
    assert transpose(x - y) == x - y
    assert transpose(x * y) == x * y
    assert transpose(x / y) == x / y
    assert transpose(-x) == -x

    x, y = symbols('x y', commutative=False)
    assert transpose(transpose(x)) == x
    assert transpose(x + y) == transpose(x) + transpose(y)
    assert transpose(x - y) == transpose(x) - transpose(y)
    assert transpose(x * y) == transpose(y) * transpose(x)
    assert transpose(x / y) == 1 / transpose(y) * transpose(x)
    assert transpose(-x) == -transpose(x)


@_both_exp_pow
def test_polarify():
    from sympy.functions.elementary.complexes import (polar_lift, polarify)
    x = Symbol('x')
    z = Symbol('z', polar=True)
    f = Function('f')
    ES = {}

    assert polarify(-1) == (polar_lift(-1), ES)
    assert polarify(1 + I) == (polar_lift(1 + I), ES)

    assert polarify(exp(x), subs=False) == exp(x)
    assert polarify(1 + x, subs=False) == 1 + x
    assert polarify(f(I) + x, subs=False) == f(polar_lift(I)) + x

    assert polarify(x, lift=True) == polar_lift(x)
    assert polarify(z, lift=True) == z
    assert polarify(f(x), lift=True) == f(polar_lift(x))
    assert polarify(1 + x, lift=True) == polar_lift(1 + x)
    assert polarify(1 + f(x), lift=True) == polar_lift(1 + f(polar_lift(x)))

    newex, subs = polarify(f(x) + z)
    assert newex.subs(subs) == f(x) + z

    mu = Symbol("mu")
    sigma = Symbol("sigma", positive=True)

    # Make sure polarify(lift=True) doesn't try to lift the integration
    # variable
    assert polarify(
        Integral(sqrt(2)*x*exp(-(-mu + x)**2/(2*sigma**2))/(2*sqrt(pi)*sigma),
        (x, -oo, oo)), lift=True) == Integral(sqrt(2)*(sigma*exp_polar(0))**exp_polar(I*pi)*
        exp((sigma*exp_polar(0))**(2*exp_polar(I*pi))*exp_polar(I*pi)*polar_lift(-mu + x)**
        (2*exp_polar(0))/2)*exp_polar(0)*polar_lift(x)/(2*sqrt(pi)), (x, -oo, oo))


def test_unpolarify():
    from sympy.functions.elementary.complexes import (polar_lift, principal_branch, unpolarify)
    from sympy.core.relational import Ne
    from sympy.functions.elementary.hyperbolic import tanh
    from sympy.functions.special.error_functions import erf
    from sympy.functions.special.gamma_functions import (gamma, uppergamma)
    from sympy.abc import x
    p = exp_polar(7*I) + 1
    u = exp(7*I) + 1

    assert unpolarify(1) == 1
    assert unpolarify(p) == u
    assert unpolarify(p**2) == u**2
    assert unpolarify(p**x) == p**x
    assert unpolarify(p*x) == u*x
    assert unpolarify(p + x) == u + x
    assert unpolarify(sqrt(sin(p))) == sqrt(sin(u))

    # Test reduction to principal branch 2*pi.
    t = principal_branch(x, 2*pi)
    assert unpolarify(t) == x
    assert unpolarify(sqrt(t)) == sqrt(t)

    # Test exponents_only.
    assert unpolarify(p**p, exponents_only=True) == p**u
    assert unpolarify(uppergamma(x, p**p)) == uppergamma(x, p**u)

    # Test functions.
    assert unpolarify(sin(p)) == sin(u)
    assert unpolarify(tanh(p)) == tanh(u)
    assert unpolarify(gamma(p)) == gamma(u)
    assert unpolarify(erf(p)) == erf(u)
    assert unpolarify(uppergamma(x, p)) == uppergamma(x, p)

    assert unpolarify(uppergamma(sin(p), sin(p + exp_polar(0)))) == \
        uppergamma(sin(u), sin(u + 1))
    assert unpolarify(uppergamma(polar_lift(0), 2*exp_polar(0))) == \
        uppergamma(0, 2)

    assert unpolarify(Eq(p, 0)) == Eq(u, 0)
    assert unpolarify(Ne(p, 0)) == Ne(u, 0)
    assert unpolarify(polar_lift(x) > 0) == (x > 0)

    # Test bools
    assert unpolarify(True) is True


def test_issue_4035():
    x = Symbol('x')
    assert Abs(x).expand(trig=True) == Abs(x)
    assert sign(x).expand(trig=True) == sign(x)
    assert arg(x).expand(trig=True) == arg(x)


def test_issue_3206():
    x = Symbol('x')
    assert Abs(Abs(x)) == Abs(x)


def test_issue_4754_derivative_conjugate():
    x = Symbol('x', real=True)
    y = Symbol('y', imaginary=True)
    f = Function('f')
    assert (f(x).conjugate()).diff(x) == (f(x).diff(x)).conjugate()
    assert (f(y).conjugate()).diff(y) == -(f(y).diff(y)).conjugate()


def test_derivatives_issue_4757():
    x = Symbol('x', real=True)
    y = Symbol('y', imaginary=True)
    f = Function('f')
    assert re(f(x)).diff(x) == re(f(x).diff(x))
    assert im(f(x)).diff(x) == im(f(x).diff(x))
    assert re(f(y)).diff(y) == -I*im(f(y).diff(y))
    assert im(f(y)).diff(y) == -I*re(f(y).diff(y))
    assert Abs(f(x)).diff(x).subs(f(x), 1 + I*x).doit() == x/sqrt(1 + x**2)
    assert arg(f(x)).diff(x).subs(f(x), 1 + I*x**2).doit() == 2*x/(1 + x**4)
    assert Abs(f(y)).diff(y).subs(f(y), 1 + y).doit() == -y/sqrt(1 - y**2)
    assert arg(f(y)).diff(y).subs(f(y), I + y**2).doit() == 2*y/(1 + y**4)


def test_issue_11413():
    from sympy.simplify.simplify import simplify
    v0 = Symbol('v0')
    v1 = Symbol('v1')
    v2 = Symbol('v2')
    V = Matrix([[v0],[v1],[v2]])
    U = V.normalized()
    assert U == Matrix([
    [v0/sqrt(Abs(v0)**2 + Abs(v1)**2 + Abs(v2)**2)],
    [v1/sqrt(Abs(v0)**2 + Abs(v1)**2 + Abs(v2)**2)],
    [v2/sqrt(Abs(v0)**2 + Abs(v1)**2 + Abs(v2)**2)]])
    U.norm = sqrt(v0**2/(v0**2 + v1**2 + v2**2) + v1**2/(v0**2 + v1**2 + v2**2) + v2**2/(v0**2 + v1**2 + v2**2))
    assert simplify(U.norm) == 1


def test_periodic_argument():
    from sympy.functions.elementary.complexes import (periodic_argument, polar_lift, principal_branch, unbranched_argument)
    x = Symbol('x')
    p = Symbol('p', positive=True)

    assert unbranched_argument(2 + I) == periodic_argument(2 + I, oo)
    assert unbranched_argument(1 + x) == periodic_argument(1 + x, oo)
    assert N_equals(unbranched_argument((1 + I)**2), pi/2)
    assert N_equals(unbranched_argument((1 - I)**2), -pi/2)
    assert N_equals(periodic_argument((1 + I)**2, 3*pi), pi/2)
    assert N_equals(periodic_argument((1 - I)**2, 3*pi), -pi/2)

    assert unbranched_argument(principal_branch(x, pi)) == \
        periodic_argument(x, pi)

    assert unbranched_argument(polar_lift(2 + I)) == unbranched_argument(2 + I)
    assert periodic_argument(polar_lift(2 + I), 2*pi) == \
        periodic_argument(2 + I, 2*pi)
    assert periodic_argument(polar_lift(2 + I), 3*pi) == \
        periodic_argument(2 + I, 3*pi)
    assert periodic_argument(polar_lift(2 + I), pi) == \
        periodic_argument(polar_lift(2 + I), pi)

    assert unbranched_argument(polar_lift(1 + I)) == pi/4
    assert periodic_argument(2*p, p) == periodic_argument(p, p)
    assert periodic_argument(pi*p, p) == periodic_argument(p, p)

    assert Abs(polar_lift(1 + I)) == Abs(1 + I)


@XFAIL
def test_principal_branch_fail():
    # TODO XXX why does abs(x)._eval_evalf() not fall back to global evalf?
    from sympy.functions.elementary.complexes import principal_branch
    assert N_equals(principal_branch((1 + I)**2, pi/2), 0)


def test_principal_branch():
    from sympy.functions.elementary.complexes import (polar_lift, principal_branch)
    p = Symbol('p', positive=True)
    x = Symbol('x')
    neg = Symbol('x', negative=True)

    assert principal_branch(polar_lift(x), p) == principal_branch(x, p)
    assert principal_branch(polar_lift(2 + I), p) == principal_branch(2 + I, p)
    assert principal_branch(2*x, p) == 2*principal_branch(x, p)
    assert principal_branch(1, pi) == exp_polar(0)
    assert principal_branch(-1, 2*pi) == exp_polar(I*pi)
    assert principal_branch(-1, pi) == exp_polar(0)
    assert principal_branch(exp_polar(3*pi*I)*x, 2*pi) == \
        principal_branch(exp_polar(I*pi)*x, 2*pi)
    assert principal_branch(neg*exp_polar(pi*I), 2*pi) == neg*exp_polar(-I*pi)
    # related to issue #14692
    assert principal_branch(exp_polar(-I*pi/2)/polar_lift(neg), 2*pi) == \
        exp_polar(-I*pi/2)/neg

    assert N_equals(principal_branch((1 + I)**2, 2*pi), 2*I)
    assert N_equals(principal_branch((1 + I)**2, 3*pi), 2*I)
    assert N_equals(principal_branch((1 + I)**2, 1*pi), 2*I)

    # test argument sanitization
    assert principal_branch(x, I).func is principal_branch
    assert principal_branch(x, -4).func is principal_branch
    assert principal_branch(x, -oo).func is principal_branch
    assert principal_branch(x, zoo).func is principal_branch


@XFAIL
def test_issue_6167_6151():
    n = pi**1000
    i = int(n)
    assert sign(n - i) == 1
    assert abs(n - i) == n - i
    x = Symbol('x')
    eps = pi**-1500
    big = pi**1000
    one = cos(x)**2 + sin(x)**2
    e = big*one - big + eps
    from sympy.simplify.simplify import simplify
    assert sign(simplify(e)) == 1
    for xi in (111, 11, 1, Rational(1, 10)):
        assert sign(e.subs(x, xi)) == 1


def test_issue_14216():
    from sympy.functions.elementary.complexes import unpolarify
    A = MatrixSymbol("A", 2, 2)
    assert unpolarify(A[0, 0]) == A[0, 0]
    assert unpolarify(A[0, 0]*A[1, 0]) == A[0, 0]*A[1, 0]


def test_issue_14238():
    # doesn't cause recursion error
    r = Symbol('r', real=True)
    assert Abs(r + Piecewise((0, r > 0), (1 - r, True)))


def test_issue_22189():
    x = Symbol('x')
    for a in (sqrt(7 - 2*x) - 2, 1 - x):
        assert Abs(a) - Abs(-a) == 0, a


def test_zero_assumptions():
    nr = Symbol('nonreal', real=False, finite=True)
    ni = Symbol('nonimaginary', imaginary=False)
    # imaginary implies not zero
    nzni = Symbol('nonzerononimaginary', zero=False, imaginary=False)

    assert re(nr).is_zero is None
    assert im(nr).is_zero is False

    assert re(ni).is_zero is None
    assert im(ni).is_zero is None

    assert re(nzni).is_zero is False
    assert im(nzni).is_zero is None


@_both_exp_pow
def test_issue_15893():
    f = Function('f', real=True)
    x = Symbol('x', real=True)
    eq = Derivative(Abs(f(x)), f(x))
    assert eq.doit() == sign(f(x))
