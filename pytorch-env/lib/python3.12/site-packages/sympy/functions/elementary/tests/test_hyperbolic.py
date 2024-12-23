from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.function import (expand_mul, expand_trig)
from sympy.core.numbers import (E, I, Integer, Rational, nan, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, acoth, acsch, asech, asinh, atanh, cosh, coth, csch, sech, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, cos, cot, sec, sin, tan)
from sympy.series.order import O

from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError, PoleError
from sympy.testing.pytest import raises


def test_sinh():
    x, y = symbols('x,y')

    k = Symbol('k', integer=True)

    assert sinh(nan) is nan
    assert sinh(zoo) is nan

    assert sinh(oo) is oo
    assert sinh(-oo) is -oo

    assert sinh(0) == 0

    assert unchanged(sinh, 1)
    assert sinh(-1) == -sinh(1)

    assert unchanged(sinh, x)
    assert sinh(-x) == -sinh(x)

    assert unchanged(sinh, pi)
    assert sinh(-pi) == -sinh(pi)

    assert unchanged(sinh, 2**1024 * E)
    assert sinh(-2**1024 * E) == -sinh(2**1024 * E)

    assert sinh(pi*I) == 0
    assert sinh(-pi*I) == 0
    assert sinh(2*pi*I) == 0
    assert sinh(-2*pi*I) == 0
    assert sinh(-3*10**73*pi*I) == 0
    assert sinh(7*10**103*pi*I) == 0

    assert sinh(pi*I/2) == I
    assert sinh(-pi*I/2) == -I
    assert sinh(pi*I*Rational(5, 2)) == I
    assert sinh(pi*I*Rational(7, 2)) == -I

    assert sinh(pi*I/3) == S.Half*sqrt(3)*I
    assert sinh(pi*I*Rational(-2, 3)) == Rational(-1, 2)*sqrt(3)*I

    assert sinh(pi*I/4) == S.Half*sqrt(2)*I
    assert sinh(-pi*I/4) == Rational(-1, 2)*sqrt(2)*I
    assert sinh(pi*I*Rational(17, 4)) == S.Half*sqrt(2)*I
    assert sinh(pi*I*Rational(-3, 4)) == Rational(-1, 2)*sqrt(2)*I

    assert sinh(pi*I/6) == S.Half*I
    assert sinh(-pi*I/6) == Rational(-1, 2)*I
    assert sinh(pi*I*Rational(7, 6)) == Rational(-1, 2)*I
    assert sinh(pi*I*Rational(-5, 6)) == Rational(-1, 2)*I

    assert sinh(pi*I/105) == sin(pi/105)*I
    assert sinh(-pi*I/105) == -sin(pi/105)*I

    assert unchanged(sinh, 2 + 3*I)

    assert sinh(x*I) == sin(x)*I

    assert sinh(k*pi*I) == 0
    assert sinh(17*k*pi*I) == 0

    assert sinh(k*pi*I/2) == sin(k*pi/2)*I

    assert sinh(x).as_real_imag(deep=False) == (cos(im(x))*sinh(re(x)),
                sin(im(x))*cosh(re(x)))
    x = Symbol('x', extended_real=True)
    assert sinh(x).as_real_imag(deep=False) == (sinh(x), 0)

    x = Symbol('x', real=True)
    assert sinh(I*x).is_finite is True
    assert sinh(x).is_real is True
    assert sinh(I).is_real is False
    p = Symbol('p', positive=True)
    assert sinh(p).is_zero is False
    assert sinh(0, evaluate=False).is_zero is True
    assert sinh(2*pi*I, evaluate=False).is_zero is True


def test_sinh_series():
    x = Symbol('x')
    assert sinh(x).series(x, 0, 10) == \
        x + x**3/6 + x**5/120 + x**7/5040 + x**9/362880 + O(x**10)


def test_sinh_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: sinh(x).fdiff(2))


def test_cosh():
    x, y = symbols('x,y')

    k = Symbol('k', integer=True)

    assert cosh(nan) is nan
    assert cosh(zoo) is nan

    assert cosh(oo) is oo
    assert cosh(-oo) is oo

    assert cosh(0) == 1

    assert unchanged(cosh, 1)
    assert cosh(-1) == cosh(1)

    assert unchanged(cosh, x)
    assert cosh(-x) == cosh(x)

    assert cosh(pi*I) == cos(pi)
    assert cosh(-pi*I) == cos(pi)

    assert unchanged(cosh, 2**1024 * E)
    assert cosh(-2**1024 * E) == cosh(2**1024 * E)

    assert cosh(pi*I/2) == 0
    assert cosh(-pi*I/2) == 0
    assert cosh((-3*10**73 + 1)*pi*I/2) == 0
    assert cosh((7*10**103 + 1)*pi*I/2) == 0

    assert cosh(pi*I) == -1
    assert cosh(-pi*I) == -1
    assert cosh(5*pi*I) == -1
    assert cosh(8*pi*I) == 1

    assert cosh(pi*I/3) == S.Half
    assert cosh(pi*I*Rational(-2, 3)) == Rational(-1, 2)

    assert cosh(pi*I/4) == S.Half*sqrt(2)
    assert cosh(-pi*I/4) == S.Half*sqrt(2)
    assert cosh(pi*I*Rational(11, 4)) == Rational(-1, 2)*sqrt(2)
    assert cosh(pi*I*Rational(-3, 4)) == Rational(-1, 2)*sqrt(2)

    assert cosh(pi*I/6) == S.Half*sqrt(3)
    assert cosh(-pi*I/6) == S.Half*sqrt(3)
    assert cosh(pi*I*Rational(7, 6)) == Rational(-1, 2)*sqrt(3)
    assert cosh(pi*I*Rational(-5, 6)) == Rational(-1, 2)*sqrt(3)

    assert cosh(pi*I/105) == cos(pi/105)
    assert cosh(-pi*I/105) == cos(pi/105)

    assert unchanged(cosh, 2 + 3*I)

    assert cosh(x*I) == cos(x)

    assert cosh(k*pi*I) == cos(k*pi)
    assert cosh(17*k*pi*I) == cos(17*k*pi)

    assert unchanged(cosh, k*pi)

    assert cosh(x).as_real_imag(deep=False) == (cos(im(x))*cosh(re(x)),
                sin(im(x))*sinh(re(x)))
    x = Symbol('x', extended_real=True)
    assert cosh(x).as_real_imag(deep=False) == (cosh(x), 0)

    x = Symbol('x', real=True)
    assert cosh(I*x).is_finite is True
    assert cosh(I*x).is_real is True
    assert cosh(I*2 + 1).is_real is False
    assert cosh(5*I*S.Pi/2, evaluate=False).is_zero is True
    assert cosh(x).is_zero is False


def test_cosh_series():
    x = Symbol('x')
    assert cosh(x).series(x, 0, 10) == \
        1 + x**2/2 + x**4/24 + x**6/720 + x**8/40320 + O(x**10)


def test_cosh_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: cosh(x).fdiff(2))


def test_tanh():
    x, y = symbols('x,y')

    k = Symbol('k', integer=True)

    assert tanh(nan) is nan
    assert tanh(zoo) is nan

    assert tanh(oo) == 1
    assert tanh(-oo) == -1

    assert tanh(0) == 0

    assert unchanged(tanh, 1)
    assert tanh(-1) == -tanh(1)

    assert unchanged(tanh, x)
    assert tanh(-x) == -tanh(x)

    assert unchanged(tanh, pi)
    assert tanh(-pi) == -tanh(pi)

    assert unchanged(tanh, 2**1024 * E)
    assert tanh(-2**1024 * E) == -tanh(2**1024 * E)

    assert tanh(pi*I) == 0
    assert tanh(-pi*I) == 0
    assert tanh(2*pi*I) == 0
    assert tanh(-2*pi*I) == 0
    assert tanh(-3*10**73*pi*I) == 0
    assert tanh(7*10**103*pi*I) == 0

    assert tanh(pi*I/2) is zoo
    assert tanh(-pi*I/2) is zoo
    assert tanh(pi*I*Rational(5, 2)) is zoo
    assert tanh(pi*I*Rational(7, 2)) is zoo

    assert tanh(pi*I/3) == sqrt(3)*I
    assert tanh(pi*I*Rational(-2, 3)) == sqrt(3)*I

    assert tanh(pi*I/4) == I
    assert tanh(-pi*I/4) == -I
    assert tanh(pi*I*Rational(17, 4)) == I
    assert tanh(pi*I*Rational(-3, 4)) == I

    assert tanh(pi*I/6) == I/sqrt(3)
    assert tanh(-pi*I/6) == -I/sqrt(3)
    assert tanh(pi*I*Rational(7, 6)) == I/sqrt(3)
    assert tanh(pi*I*Rational(-5, 6)) == I/sqrt(3)

    assert tanh(pi*I/105) == tan(pi/105)*I
    assert tanh(-pi*I/105) == -tan(pi/105)*I

    assert unchanged(tanh, 2 + 3*I)

    assert tanh(x*I) == tan(x)*I

    assert tanh(k*pi*I) == 0
    assert tanh(17*k*pi*I) == 0

    assert tanh(k*pi*I/2) == tan(k*pi/2)*I

    assert tanh(x).as_real_imag(deep=False) == (sinh(re(x))*cosh(re(x))/(cos(im(x))**2
                                + sinh(re(x))**2),
                                sin(im(x))*cos(im(x))/(cos(im(x))**2 + sinh(re(x))**2))
    x = Symbol('x', extended_real=True)
    assert tanh(x).as_real_imag(deep=False) == (tanh(x), 0)
    assert tanh(I*pi/3 + 1).is_real is False
    assert tanh(x).is_real is True
    assert tanh(I*pi*x/2).is_real is None


def test_tanh_series():
    x = Symbol('x')
    assert tanh(x).series(x, 0, 10) == \
        x - x**3/3 + 2*x**5/15 - 17*x**7/315 + 62*x**9/2835 + O(x**10)


def test_tanh_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: tanh(x).fdiff(2))


def test_coth():
    x, y = symbols('x,y')

    k = Symbol('k', integer=True)

    assert coth(nan) is nan
    assert coth(zoo) is nan

    assert coth(oo) == 1
    assert coth(-oo) == -1

    assert coth(0) is zoo
    assert unchanged(coth, 1)
    assert coth(-1) == -coth(1)

    assert unchanged(coth, x)
    assert coth(-x) == -coth(x)

    assert coth(pi*I) == -I*cot(pi)
    assert coth(-pi*I) == cot(pi)*I

    assert unchanged(coth, 2**1024 * E)
    assert coth(-2**1024 * E) == -coth(2**1024 * E)

    assert coth(pi*I) == -I*cot(pi)
    assert coth(-pi*I) == I*cot(pi)
    assert coth(2*pi*I) == -I*cot(2*pi)
    assert coth(-2*pi*I) == I*cot(2*pi)
    assert coth(-3*10**73*pi*I) == I*cot(3*10**73*pi)
    assert coth(7*10**103*pi*I) == -I*cot(7*10**103*pi)

    assert coth(pi*I/2) == 0
    assert coth(-pi*I/2) == 0
    assert coth(pi*I*Rational(5, 2)) == 0
    assert coth(pi*I*Rational(7, 2)) == 0

    assert coth(pi*I/3) == -I/sqrt(3)
    assert coth(pi*I*Rational(-2, 3)) == -I/sqrt(3)

    assert coth(pi*I/4) == -I
    assert coth(-pi*I/4) == I
    assert coth(pi*I*Rational(17, 4)) == -I
    assert coth(pi*I*Rational(-3, 4)) == -I

    assert coth(pi*I/6) == -sqrt(3)*I
    assert coth(-pi*I/6) == sqrt(3)*I
    assert coth(pi*I*Rational(7, 6)) == -sqrt(3)*I
    assert coth(pi*I*Rational(-5, 6)) == -sqrt(3)*I

    assert coth(pi*I/105) == -cot(pi/105)*I
    assert coth(-pi*I/105) == cot(pi/105)*I

    assert unchanged(coth, 2 + 3*I)

    assert coth(x*I) == -cot(x)*I

    assert coth(k*pi*I) == -cot(k*pi)*I
    assert coth(17*k*pi*I) == -cot(17*k*pi)*I

    assert coth(k*pi*I) == -cot(k*pi)*I

    assert coth(log(tan(2))) == coth(log(-tan(2)))
    assert coth(1 + I*pi/2) == tanh(1)

    assert coth(x).as_real_imag(deep=False) == (sinh(re(x))*cosh(re(x))/(sin(im(x))**2
                                + sinh(re(x))**2),
                                -sin(im(x))*cos(im(x))/(sin(im(x))**2 + sinh(re(x))**2))
    x = Symbol('x', extended_real=True)
    assert coth(x).as_real_imag(deep=False) == (coth(x), 0)

    assert expand_trig(coth(2*x)) == (coth(x)**2 + 1)/(2*coth(x))
    assert expand_trig(coth(3*x)) == (coth(x)**3 + 3*coth(x))/(1 + 3*coth(x)**2)

    assert expand_trig(coth(x + y)) == (1 + coth(x)*coth(y))/(coth(x) + coth(y))


def test_coth_series():
    x = Symbol('x')
    assert coth(x).series(x, 0, 8) == \
        1/x + x/3 - x**3/45 + 2*x**5/945 - x**7/4725 + O(x**8)


def test_coth_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: coth(x).fdiff(2))


def test_csch():
    x, y = symbols('x,y')

    k = Symbol('k', integer=True)
    n = Symbol('n', positive=True)

    assert csch(nan) is nan
    assert csch(zoo) is nan

    assert csch(oo) == 0
    assert csch(-oo) == 0

    assert csch(0) is zoo

    assert csch(-1) == -csch(1)

    assert csch(-x) == -csch(x)
    assert csch(-pi) == -csch(pi)
    assert csch(-2**1024 * E) == -csch(2**1024 * E)

    assert csch(pi*I) is zoo
    assert csch(-pi*I) is zoo
    assert csch(2*pi*I) is zoo
    assert csch(-2*pi*I) is zoo
    assert csch(-3*10**73*pi*I) is zoo
    assert csch(7*10**103*pi*I) is zoo

    assert csch(pi*I/2) == -I
    assert csch(-pi*I/2) == I
    assert csch(pi*I*Rational(5, 2)) == -I
    assert csch(pi*I*Rational(7, 2)) == I

    assert csch(pi*I/3) == -2/sqrt(3)*I
    assert csch(pi*I*Rational(-2, 3)) == 2/sqrt(3)*I

    assert csch(pi*I/4) == -sqrt(2)*I
    assert csch(-pi*I/4) == sqrt(2)*I
    assert csch(pi*I*Rational(7, 4)) == sqrt(2)*I
    assert csch(pi*I*Rational(-3, 4)) == sqrt(2)*I

    assert csch(pi*I/6) == -2*I
    assert csch(-pi*I/6) == 2*I
    assert csch(pi*I*Rational(7, 6)) == 2*I
    assert csch(pi*I*Rational(-7, 6)) == -2*I
    assert csch(pi*I*Rational(-5, 6)) == 2*I

    assert csch(pi*I/105) == -1/sin(pi/105)*I
    assert csch(-pi*I/105) == 1/sin(pi/105)*I

    assert csch(x*I) == -1/sin(x)*I

    assert csch(k*pi*I) is zoo
    assert csch(17*k*pi*I) is zoo

    assert csch(k*pi*I/2) == -1/sin(k*pi/2)*I

    assert csch(n).is_real is True

    assert expand_trig(csch(x + y)) == 1/(sinh(x)*cosh(y) + cosh(x)*sinh(y))


def test_csch_series():
    x = Symbol('x')
    assert csch(x).series(x, 0, 10) == \
       1/ x - x/6 + 7*x**3/360 - 31*x**5/15120 + 127*x**7/604800 \
          - 73*x**9/3421440 + O(x**10)


def test_csch_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: csch(x).fdiff(2))


def test_sech():
    x, y = symbols('x, y')

    k = Symbol('k', integer=True)
    n = Symbol('n', positive=True)

    assert sech(nan) is nan
    assert sech(zoo) is nan

    assert sech(oo) == 0
    assert sech(-oo) == 0

    assert sech(0) == 1

    assert sech(-1) == sech(1)
    assert sech(-x) == sech(x)

    assert sech(pi*I) == sec(pi)

    assert sech(-pi*I) == sec(pi)
    assert sech(-2**1024 * E) == sech(2**1024 * E)

    assert sech(pi*I/2) is zoo
    assert sech(-pi*I/2) is zoo
    assert sech((-3*10**73 + 1)*pi*I/2) is zoo
    assert sech((7*10**103 + 1)*pi*I/2) is zoo

    assert sech(pi*I) == -1
    assert sech(-pi*I) == -1
    assert sech(5*pi*I) == -1
    assert sech(8*pi*I) == 1

    assert sech(pi*I/3) == 2
    assert sech(pi*I*Rational(-2, 3)) == -2

    assert sech(pi*I/4) == sqrt(2)
    assert sech(-pi*I/4) == sqrt(2)
    assert sech(pi*I*Rational(5, 4)) == -sqrt(2)
    assert sech(pi*I*Rational(-5, 4)) == -sqrt(2)

    assert sech(pi*I/6) == 2/sqrt(3)
    assert sech(-pi*I/6) == 2/sqrt(3)
    assert sech(pi*I*Rational(7, 6)) == -2/sqrt(3)
    assert sech(pi*I*Rational(-5, 6)) == -2/sqrt(3)

    assert sech(pi*I/105) == 1/cos(pi/105)
    assert sech(-pi*I/105) == 1/cos(pi/105)

    assert sech(x*I) == 1/cos(x)

    assert sech(k*pi*I) == 1/cos(k*pi)
    assert sech(17*k*pi*I) == 1/cos(17*k*pi)

    assert sech(n).is_real is True

    assert expand_trig(sech(x + y)) == 1/(cosh(x)*cosh(y) + sinh(x)*sinh(y))


def test_sech_series():
    x = Symbol('x')
    assert sech(x).series(x, 0, 10) == \
        1 - x**2/2 + 5*x**4/24 - 61*x**6/720 + 277*x**8/8064 + O(x**10)


def test_sech_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: sech(x).fdiff(2))


def test_asinh():
    x, y = symbols('x,y')
    assert unchanged(asinh, x)
    assert asinh(-x) == -asinh(x)

    # at specific points
    assert asinh(nan) is nan
    assert asinh( 0) == 0
    assert asinh(+1) == log(sqrt(2) + 1)

    assert asinh(-1) == log(sqrt(2) - 1)
    assert asinh(I) == pi*I/2
    assert asinh(-I) == -pi*I/2
    assert asinh(I/2) == pi*I/6
    assert asinh(-I/2) == -pi*I/6

    # at infinites
    assert asinh(oo) is oo
    assert asinh(-oo) is -oo

    assert asinh(I*oo) is oo
    assert asinh(-I *oo) is -oo

    assert asinh(zoo) is zoo

    # properties
    assert asinh(I *(sqrt(3) - 1)/(2**Rational(3, 2))) == pi*I/12
    assert asinh(-I *(sqrt(3) - 1)/(2**Rational(3, 2))) == -pi*I/12

    assert asinh(I*(sqrt(5) - 1)/4) == pi*I/10
    assert asinh(-I*(sqrt(5) - 1)/4) == -pi*I/10

    assert asinh(I*(sqrt(5) + 1)/4) == pi*I*Rational(3, 10)
    assert asinh(-I*(sqrt(5) + 1)/4) == pi*I*Rational(-3, 10)

    # reality
    assert asinh(S(2)).is_real is True
    assert asinh(S(2)).is_finite is True
    assert asinh(S(-2)).is_real is True
    assert asinh(S(oo)).is_extended_real is True
    assert asinh(-S(oo)).is_real is False
    assert (asinh(2) - oo) == -oo
    assert asinh(symbols('y', real=True)).is_real is True

    # Symmetry
    assert asinh(Rational(-1, 2)) == -asinh(S.Half)

    # inverse composition
    assert unchanged(asinh, sinh(Symbol('v1')))

    assert asinh(sinh(0, evaluate=False)) == 0
    assert asinh(sinh(-3, evaluate=False)) == -3
    assert asinh(sinh(2, evaluate=False)) == 2
    assert asinh(sinh(I, evaluate=False)) == I
    assert asinh(sinh(-I, evaluate=False)) == -I
    assert asinh(sinh(5*I, evaluate=False)) == -2*I*pi + 5*I
    assert asinh(sinh(15 + 11*I)) == 15 - 4*I*pi + 11*I
    assert asinh(sinh(-73 + 97*I)) == 73 - 97*I + 31*I*pi
    assert asinh(sinh(-7 - 23*I)) == 7 - 7*I*pi + 23*I
    assert asinh(sinh(13 - 3*I)) == -13 - I*pi + 3*I
    p = Symbol('p', positive=True)
    assert asinh(p).is_zero is False
    assert asinh(sinh(0, evaluate=False), evaluate=False).is_zero is True


def test_asinh_rewrite():
    x = Symbol('x')
    assert asinh(x).rewrite(log) == log(x + sqrt(x**2 + 1))
    assert asinh(x).rewrite(atanh) == atanh(x/sqrt(1 + x**2))
    assert asinh(x).rewrite(asin) == -I*asin(I*x, evaluate=False)
    assert asinh(x*(1 + I)).rewrite(asin) == -I*asin(I*x*(1+I))
    assert asinh(x).rewrite(acos) == I*acos(I*x, evaluate=False) - I*pi/2


def test_asinh_leading_term():
    x = Symbol('x')
    assert asinh(x).as_leading_term(x, cdir=1) == x
    # Tests concerning branch points
    assert asinh(x + I).as_leading_term(x, cdir=1) == I*pi/2
    assert asinh(x - I).as_leading_term(x, cdir=1) == -I*pi/2
    assert asinh(1/x).as_leading_term(x, cdir=1) == -log(x) + log(2)
    assert asinh(1/x).as_leading_term(x, cdir=-1) == log(x) - log(2) - I*pi
    # Tests concerning points lying on branch cuts
    assert asinh(x + 2*I).as_leading_term(x, cdir=1) == I*asin(2)
    assert asinh(x + 2*I).as_leading_term(x, cdir=-1) == -I*asin(2) + I*pi
    assert asinh(x - 2*I).as_leading_term(x, cdir=1) == -I*pi + I*asin(2)
    assert asinh(x - 2*I).as_leading_term(x, cdir=-1) == -I*asin(2)
    # Tests concerning re(ndir) == 0
    assert asinh(2*I + I*x - x**2).as_leading_term(x, cdir=1) == log(2 - sqrt(3)) + I*pi/2
    assert asinh(2*I + I*x - x**2).as_leading_term(x, cdir=-1) == log(2 - sqrt(3)) + I*pi/2


def test_asinh_series():
    x = Symbol('x')
    assert asinh(x).series(x, 0, 8) == \
        x - x**3/6 + 3*x**5/40 - 5*x**7/112 + O(x**8)
    t5 = asinh(x).taylor_term(5, x)
    assert t5 == 3*x**5/40
    assert asinh(x).taylor_term(7, x, t5, 0) == -5*x**7/112


def test_asinh_nseries():
    x = Symbol('x')
    # Tests concerning branch points
    assert asinh(x + I)._eval_nseries(x, 4, None) == I*pi/2 + \
    sqrt(x)*(1 - I) + x**(S(3)/2)*(S(1)/12 + I/12) + x**(S(5)/2)*(-S(3)/160 + 3*I/160) + \
    x**(S(7)/2)*(-S(5)/896 - 5*I/896) + O(x**4)
    assert asinh(x - I)._eval_nseries(x, 4, None) == -I*pi/2 + \
    sqrt(x)*(1 + I) + x**(S(3)/2)*(S(1)/12 - I/12) + x**(S(5)/2)*(-S(3)/160 - 3*I/160) + \
    x**(S(7)/2)*(-S(5)/896 + 5*I/896) + O(x**4)
    # Tests concerning points lying on branch cuts
    assert asinh(x + 2*I)._eval_nseries(x, 4, None, cdir=1) == I*asin(2) - \
    sqrt(3)*I*x/3 + sqrt(3)*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    assert asinh(x + 2*I)._eval_nseries(x, 4, None, cdir=-1) == I*pi - I*asin(2) + \
    sqrt(3)*I*x/3 - sqrt(3)*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    assert asinh(x - 2*I)._eval_nseries(x, 4, None, cdir=1) == I*asin(2) - I*pi + \
    sqrt(3)*I*x/3 + sqrt(3)*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    assert asinh(x - 2*I)._eval_nseries(x, 4, None, cdir=-1) == -I*asin(2) - \
    sqrt(3)*I*x/3 - sqrt(3)*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    # Tests concerning re(ndir) == 0
    assert asinh(2*I + I*x - x**2)._eval_nseries(x, 4, None) == I*pi/2 + log(2 - sqrt(3)) - \
    sqrt(3)*x/3 + x**2*(sqrt(3)/9 - sqrt(3)*I/3) + x**3*(-sqrt(3)/18 + 2*sqrt(3)*I/9) + O(x**4)


def test_asinh_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: asinh(x).fdiff(2))


def test_acosh():
    x = Symbol('x')

    assert unchanged(acosh, -x)

    #at specific points
    assert acosh(1) == 0
    assert acosh(-1) == pi*I
    assert acosh(0) == I*pi/2
    assert acosh(S.Half) == I*pi/3
    assert acosh(Rational(-1, 2)) == pi*I*Rational(2, 3)
    assert acosh(nan) is nan

    # at infinites
    assert acosh(oo) is oo
    assert acosh(-oo) is oo

    assert acosh(I*oo) == oo + I*pi/2
    assert acosh(-I*oo) == oo - I*pi/2

    assert acosh(zoo) is zoo

    assert acosh(I) == log(I*(1 + sqrt(2)))
    assert acosh(-I) == log(-I*(1 + sqrt(2)))
    assert acosh((sqrt(3) - 1)/(2*sqrt(2))) == pi*I*Rational(5, 12)
    assert acosh(-(sqrt(3) - 1)/(2*sqrt(2))) == pi*I*Rational(7, 12)
    assert acosh(sqrt(2)/2) == I*pi/4
    assert acosh(-sqrt(2)/2) == I*pi*Rational(3, 4)
    assert acosh(sqrt(3)/2) == I*pi/6
    assert acosh(-sqrt(3)/2) == I*pi*Rational(5, 6)
    assert acosh(sqrt(2 + sqrt(2))/2) == I*pi/8
    assert acosh(-sqrt(2 + sqrt(2))/2) == I*pi*Rational(7, 8)
    assert acosh(sqrt(2 - sqrt(2))/2) == I*pi*Rational(3, 8)
    assert acosh(-sqrt(2 - sqrt(2))/2) == I*pi*Rational(5, 8)
    assert acosh((1 + sqrt(3))/(2*sqrt(2))) == I*pi/12
    assert acosh(-(1 + sqrt(3))/(2*sqrt(2))) == I*pi*Rational(11, 12)
    assert acosh((sqrt(5) + 1)/4) == I*pi/5
    assert acosh(-(sqrt(5) + 1)/4) == I*pi*Rational(4, 5)

    assert str(acosh(5*I).n(6)) == '2.31244 + 1.5708*I'
    assert str(acosh(-5*I).n(6)) == '2.31244 - 1.5708*I'

    # inverse composition
    assert unchanged(acosh, Symbol('v1'))

    assert acosh(cosh(-3, evaluate=False)) == 3
    assert acosh(cosh(3, evaluate=False)) == 3
    assert acosh(cosh(0, evaluate=False)) == 0
    assert acosh(cosh(I, evaluate=False)) == I
    assert acosh(cosh(-I, evaluate=False)) == I
    assert acosh(cosh(7*I, evaluate=False)) == -2*I*pi + 7*I
    assert acosh(cosh(1 + I)) == 1 + I
    assert acosh(cosh(3 - 3*I)) == 3 - 3*I
    assert acosh(cosh(-3 + 2*I)) == 3 - 2*I
    assert acosh(cosh(-5 - 17*I)) == 5 - 6*I*pi + 17*I
    assert acosh(cosh(-21 + 11*I)) == 21 - 11*I + 4*I*pi
    assert acosh(cosh(cosh(1) + I)) == cosh(1) + I
    assert acosh(1, evaluate=False).is_zero is True

    # Reality
    assert acosh(S(2)).is_real is True
    assert acosh(S(2)).is_extended_real is True
    assert acosh(oo).is_extended_real is True
    assert acosh(S(2)).is_finite is True
    assert acosh(S(1) / 5).is_real is False
    assert (acosh(2) - oo) == -oo
    assert acosh(symbols('y', real=True)).is_real is None


def test_acosh_rewrite():
    x = Symbol('x')
    assert acosh(x).rewrite(log) == log(x + sqrt(x - 1)*sqrt(x + 1))
    assert acosh(x).rewrite(asin) == sqrt(x - 1)*(-asin(x) + pi/2)/sqrt(1 - x)
    assert acosh(x).rewrite(asinh) == sqrt(x - 1)*(I*asinh(I*x, evaluate=False) + pi/2)/sqrt(1 - x)
    assert acosh(x).rewrite(atanh) == \
        (sqrt(x - 1)*sqrt(x + 1)*atanh(sqrt(x**2 - 1)/x)/sqrt(x**2 - 1) +
         pi*sqrt(x - 1)*(-x*sqrt(x**(-2)) + 1)/(2*sqrt(1 - x)))
    x = Symbol('x', positive=True)
    assert acosh(x).rewrite(atanh) == \
        sqrt(x - 1)*sqrt(x + 1)*atanh(sqrt(x**2 - 1)/x)/sqrt(x**2 - 1)


def test_acosh_leading_term():
    x = Symbol('x')
    # Tests concerning branch points
    assert acosh(x).as_leading_term(x) == I*pi/2
    assert acosh(x + 1).as_leading_term(x) == sqrt(2)*sqrt(x)
    assert acosh(x - 1).as_leading_term(x) == I*pi
    assert acosh(1/x).as_leading_term(x, cdir=1) == -log(x) + log(2)
    assert acosh(1/x).as_leading_term(x, cdir=-1) == -log(x) + log(2) + 2*I*pi
    # Tests concerning points lying on branch cuts
    assert acosh(I*x - 2).as_leading_term(x, cdir=1) == acosh(-2)
    assert acosh(-I*x - 2).as_leading_term(x, cdir=1) == -2*I*pi + acosh(-2)
    assert acosh(x**2 - I*x + S(1)/3).as_leading_term(x, cdir=1) == -acosh(S(1)/3)
    assert acosh(x**2 - I*x + S(1)/3).as_leading_term(x, cdir=-1) == acosh(S(1)/3)
    assert acosh(1/(I*x - 3)).as_leading_term(x, cdir=1) == -acosh(-S(1)/3)
    assert acosh(1/(I*x - 3)).as_leading_term(x, cdir=-1) == acosh(-S(1)/3)
    # Tests concerning im(ndir) == 0
    assert acosh(-I*x**2 + x - 2).as_leading_term(x, cdir=1) == log(sqrt(3) + 2) - I*pi
    assert acosh(-I*x**2 + x - 2).as_leading_term(x, cdir=-1) == log(sqrt(3) + 2) - I*pi


def test_acosh_series():
    x = Symbol('x')
    assert acosh(x).series(x, 0, 8) == \
        -I*x + pi*I/2 - I*x**3/6 - 3*I*x**5/40 - 5*I*x**7/112 + O(x**8)
    t5 = acosh(x).taylor_term(5, x)
    assert t5 == - 3*I*x**5/40
    assert acosh(x).taylor_term(7, x, t5, 0) == - 5*I*x**7/112


def test_acosh_nseries():
    x = Symbol('x')
    # Tests concerning branch points
    assert acosh(x + 1)._eval_nseries(x, 4, None) == sqrt(2)*sqrt(x) - \
    sqrt(2)*x**(S(3)/2)/12 + 3*sqrt(2)*x**(S(5)/2)/160 - 5*sqrt(2)*x**(S(7)/2)/896 + O(x**4)
    # Tests concerning points lying on branch cuts
    assert acosh(x - 1)._eval_nseries(x, 4, None) == I*pi - \
    sqrt(2)*I*sqrt(x) - sqrt(2)*I*x**(S(3)/2)/12 - 3*sqrt(2)*I*x**(S(5)/2)/160 - \
    5*sqrt(2)*I*x**(S(7)/2)/896 + O(x**4)
    assert acosh(I*x - 2)._eval_nseries(x, 4, None, cdir=1) == acosh(-2) - \
    sqrt(3)*I*x/3 + sqrt(3)*x**2/9 + sqrt(3)*I*x**3/18 + O(x**4)
    assert acosh(-I*x - 2)._eval_nseries(x, 4, None, cdir=1) == acosh(-2) - \
    2*I*pi + sqrt(3)*I*x/3 + sqrt(3)*x**2/9 - sqrt(3)*I*x**3/18 + O(x**4)
    assert acosh(1/(I*x - 3))._eval_nseries(x, 4, None, cdir=1) == -acosh(-S(1)/3) + \
    sqrt(2)*x/12 + 17*sqrt(2)*I*x**2/576 - 443*sqrt(2)*x**3/41472 + O(x**4)
    assert acosh(1/(I*x - 3))._eval_nseries(x, 4, None, cdir=-1) == acosh(-S(1)/3) - \
    sqrt(2)*x/12 - 17*sqrt(2)*I*x**2/576 + 443*sqrt(2)*x**3/41472 + O(x**4)
    # Tests concerning im(ndir) == 0
    assert acosh(-I*x**2 + x - 2)._eval_nseries(x, 4, None) == -I*pi + log(sqrt(3) + 2) - \
    sqrt(3)*x/3 + x**2*(-sqrt(3)/9 + sqrt(3)*I/3) + x**3*(-sqrt(3)/18 + 2*sqrt(3)*I/9) + O(x**4)


def test_acosh_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: acosh(x).fdiff(2))


def test_asech():
    x = Symbol('x')

    assert unchanged(asech, -x)

    # values at fixed points
    assert asech(1) == 0
    assert asech(-1) == pi*I
    assert asech(0) is oo
    assert asech(2) == I*pi/3
    assert asech(-2) == 2*I*pi / 3
    assert asech(nan) is nan

    # at infinites
    assert asech(oo) == I*pi/2
    assert asech(-oo) == I*pi/2
    assert asech(zoo) == I*AccumBounds(-pi/2, pi/2)

    assert asech(I) == log(1 + sqrt(2)) - I*pi/2
    assert asech(-I) == log(1 + sqrt(2)) + I*pi/2
    assert asech(sqrt(2) - sqrt(6)) == 11*I*pi / 12
    assert asech(sqrt(2 - 2/sqrt(5))) == I*pi / 10
    assert asech(-sqrt(2 - 2/sqrt(5))) == 9*I*pi / 10
    assert asech(2 / sqrt(2 + sqrt(2))) == I*pi / 8
    assert asech(-2 / sqrt(2 + sqrt(2))) == 7*I*pi / 8
    assert asech(sqrt(5) - 1) == I*pi / 5
    assert asech(1 - sqrt(5)) == 4*I*pi / 5
    assert asech(-sqrt(2*(2 + sqrt(2)))) == 5*I*pi / 8

    # properties
    # asech(x) == acosh(1/x)
    assert asech(sqrt(2)) == acosh(1/sqrt(2))
    assert asech(2/sqrt(3)) == acosh(sqrt(3)/2)
    assert asech(2/sqrt(2 + sqrt(2))) == acosh(sqrt(2 + sqrt(2))/2)
    assert asech(2) == acosh(S.Half)

    # reality
    assert asech(S(2)).is_real is False
    assert asech(-S(1) / 3).is_real is False
    assert asech(S(2) / 3).is_finite is True
    assert asech(S(0)).is_real is False
    assert asech(S(0)).is_extended_real is True
    assert asech(symbols('y', real=True)).is_real is None

    # asech(x) == I*acos(1/x)
    # (Note: the exact formula is asech(x) == +/- I*acos(1/x))
    assert asech(-sqrt(2)) == I*acos(-1/sqrt(2))
    assert asech(-2/sqrt(3)) == I*acos(-sqrt(3)/2)
    assert asech(-S(2)) == I*acos(Rational(-1, 2))
    assert asech(-2/sqrt(2)) == I*acos(-sqrt(2)/2)

    # sech(asech(x)) / x == 1
    assert expand_mul(sech(asech(sqrt(6) - sqrt(2))) / (sqrt(6) - sqrt(2))) == 1
    assert expand_mul(sech(asech(sqrt(6) + sqrt(2))) / (sqrt(6) + sqrt(2))) == 1
    assert (sech(asech(sqrt(2 + 2/sqrt(5)))) / (sqrt(2 + 2/sqrt(5)))).simplify() == 1
    assert (sech(asech(-sqrt(2 + 2/sqrt(5)))) / (-sqrt(2 + 2/sqrt(5)))).simplify() == 1
    assert (sech(asech(sqrt(2*(2 + sqrt(2))))) / (sqrt(2*(2 + sqrt(2))))).simplify() == 1
    assert expand_mul(sech(asech(1 + sqrt(5))) / (1 + sqrt(5))) == 1
    assert expand_mul(sech(asech(-1 - sqrt(5))) / (-1 - sqrt(5))) == 1
    assert expand_mul(sech(asech(-sqrt(6) - sqrt(2))) / (-sqrt(6) - sqrt(2))) == 1

    # numerical evaluation
    assert str(asech(5*I).n(6)) == '0.19869 - 1.5708*I'
    assert str(asech(-5*I).n(6)) == '0.19869 + 1.5708*I'


def test_asech_leading_term():
    x = Symbol('x')
    # Tests concerning branch points
    assert asech(x).as_leading_term(x, cdir=1) == -log(x) + log(2)
    assert asech(x).as_leading_term(x, cdir=-1) == -log(x) + log(2) + 2*I*pi
    assert asech(x + 1).as_leading_term(x, cdir=1) == sqrt(2)*I*sqrt(x)
    assert asech(1/x).as_leading_term(x, cdir=1) == I*pi/2
    # Tests concerning points lying on branch cuts
    assert asech(x - 1).as_leading_term(x, cdir=1) == I*pi
    assert asech(I*x + 3).as_leading_term(x, cdir=1) == -asech(3)
    assert asech(-I*x + 3).as_leading_term(x, cdir=1) == asech(3)
    assert asech(I*x - 3).as_leading_term(x, cdir=1) == -asech(-3)
    assert asech(-I*x - 3).as_leading_term(x, cdir=1) == asech(-3)
    assert asech(I*x - S(1)/3).as_leading_term(x, cdir=1) == -2*I*pi + asech(-S(1)/3)
    assert asech(I*x - S(1)/3).as_leading_term(x, cdir=-1) == asech(-S(1)/3)
    # Tests concerning im(ndir) == 0
    assert asech(-I*x**2 + x - 3).as_leading_term(x, cdir=1) == log(-S(1)/3 + 2*sqrt(2)*I/3)
    assert asech(-I*x**2 + x - 3).as_leading_term(x, cdir=-1) == log(-S(1)/3 + 2*sqrt(2)*I/3)


def test_asech_series():
    x = Symbol('x')
    assert asech(x).series(x, 0, 9, cdir=1) == log(2) - log(x) - x**2/4 - 3*x**4/32 \
    - 5*x**6/96 - 35*x**8/1024 + O(x**9)
    assert asech(x).series(x, 0, 9, cdir=-1) == I*pi + log(2) - log(-x) - x**2/4 - \
    3*x**4/32 - 5*x**6/96 - 35*x**8/1024 + O(x**9)
    t6 = asech(x).taylor_term(6, x)
    assert t6 == -5*x**6/96
    assert asech(x).taylor_term(8, x, t6, 0) == -35*x**8/1024


def test_asech_nseries():
    x = Symbol('x')
    # Tests concerning branch points
    assert asech(x + 1)._eval_nseries(x, 4, None) == sqrt(2)*sqrt(-x) + 5*sqrt(2)*(-x)**(S(3)/2)/12 + \
    43*sqrt(2)*(-x)**(S(5)/2)/160 + 177*sqrt(2)*(-x)**(S(7)/2)/896 + O(x**4)
    # Tests concerning points lying on branch cuts
    assert asech(x - 1)._eval_nseries(x, 4, None) == I*pi + sqrt(2)*sqrt(x) + \
    5*sqrt(2)*x**(S(3)/2)/12 + 43*sqrt(2)*x**(S(5)/2)/160 + 177*sqrt(2)*x**(S(7)/2)/896 + O(x**4)
    assert asech(I*x + 3)._eval_nseries(x, 4, None) == -asech(3) + sqrt(2)*x/12 - \
    17*sqrt(2)*I*x**2/576 - 443*sqrt(2)*x**3/41472 + O(x**4)
    assert asech(-I*x + 3)._eval_nseries(x, 4, None) == asech(3) + sqrt(2)*x/12 + \
    17*sqrt(2)*I*x**2/576 - 443*sqrt(2)*x**3/41472 + O(x**4)
    assert asech(I*x - 3)._eval_nseries(x, 4, None) == -asech(-3) - sqrt(2)*x/12 - \
    17*sqrt(2)*I*x**2/576 + 443*sqrt(2)*x**3/41472 + O(x**4)
    assert asech(-I*x - 3)._eval_nseries(x, 4, None) == asech(-3) - sqrt(2)*x/12 + \
    17*sqrt(2)*I*x**2/576 + 443*sqrt(2)*x**3/41472 + O(x**4)
    # Tests concerning im(ndir) == 0
    assert asech(-I*x**2 + x - 2)._eval_nseries(x, 3, None) == 2*I*pi/3 + sqrt(3)*I*x/6 + \
    x**2*(sqrt(3)/6 + 7*sqrt(3)*I/72) + O(x**3)


def test_asech_rewrite():
    x = Symbol('x')
    assert asech(x).rewrite(log) == log(1/x + sqrt(1/x - 1) * sqrt(1/x + 1))
    assert asech(x).rewrite(acosh) == acosh(1/x)
    assert asech(x).rewrite(asinh) == sqrt(-1 + 1/x)*(I*asinh(I/x, evaluate=False) + pi/2)/sqrt(1 - 1/x)
    assert asech(x).rewrite(atanh) == \
        sqrt(x + 1)*sqrt(1/(x + 1))*atanh(sqrt(1 - x**2)) + I*pi*(-sqrt(x)*sqrt(1/x) + 1 - I*sqrt(x**2)/(2*sqrt(-x**2)) - I*sqrt(-x)/(2*sqrt(x)))


def test_asech_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: asech(x).fdiff(2))


def test_acsch():
    x = Symbol('x')

    assert unchanged(acsch, x)
    assert acsch(-x) == -acsch(x)

    # values at fixed points
    assert acsch(1) == log(1 + sqrt(2))
    assert acsch(-1) == - log(1 + sqrt(2))
    assert acsch(0) is zoo
    assert acsch(2) == log((1+sqrt(5))/2)
    assert acsch(-2) == - log((1+sqrt(5))/2)

    assert acsch(I) == - I*pi/2
    assert acsch(-I) == I*pi/2
    assert acsch(-I*(sqrt(6) + sqrt(2))) == I*pi / 12
    assert acsch(I*(sqrt(2) + sqrt(6))) == -I*pi / 12
    assert acsch(-I*(1 + sqrt(5))) == I*pi / 10
    assert acsch(I*(1 + sqrt(5))) == -I*pi / 10
    assert acsch(-I*2 / sqrt(2 - sqrt(2))) == I*pi / 8
    assert acsch(I*2 / sqrt(2 - sqrt(2))) == -I*pi / 8
    assert acsch(-I*2) == I*pi / 6
    assert acsch(I*2) == -I*pi / 6
    assert acsch(-I*sqrt(2 + 2/sqrt(5))) == I*pi / 5
    assert acsch(I*sqrt(2 + 2/sqrt(5))) == -I*pi / 5
    assert acsch(-I*sqrt(2)) == I*pi / 4
    assert acsch(I*sqrt(2)) == -I*pi / 4
    assert acsch(-I*(sqrt(5)-1)) == 3*I*pi / 10
    assert acsch(I*(sqrt(5)-1)) == -3*I*pi / 10
    assert acsch(-I*2 / sqrt(3)) == I*pi / 3
    assert acsch(I*2 / sqrt(3)) == -I*pi / 3
    assert acsch(-I*2 / sqrt(2 + sqrt(2))) == 3*I*pi / 8
    assert acsch(I*2 / sqrt(2 + sqrt(2))) == -3*I*pi / 8
    assert acsch(-I*sqrt(2 - 2/sqrt(5))) == 2*I*pi / 5
    assert acsch(I*sqrt(2 - 2/sqrt(5))) == -2*I*pi / 5
    assert acsch(-I*(sqrt(6) - sqrt(2))) == 5*I*pi / 12
    assert acsch(I*(sqrt(6) - sqrt(2))) == -5*I*pi / 12
    assert acsch(nan) is nan

    # properties
    # acsch(x) == asinh(1/x)
    assert acsch(-I*sqrt(2)) == asinh(I/sqrt(2))
    assert acsch(-I*2 / sqrt(3)) == asinh(I*sqrt(3) / 2)

    # reality
    assert acsch(S(2)).is_real is True
    assert acsch(S(2)).is_finite is True
    assert acsch(S(-2)).is_real is True
    assert acsch(S(oo)).is_extended_real is True
    assert acsch(-S(oo)).is_real is True
    assert (acsch(2) - oo) == -oo
    assert acsch(symbols('y', extended_real=True)).is_extended_real is True

    # acsch(x) == -I*asin(I/x)
    assert acsch(-I*sqrt(2)) == -I*asin(-1/sqrt(2))
    assert acsch(-I*2 / sqrt(3)) == -I*asin(-sqrt(3)/2)

    # csch(acsch(x)) / x == 1
    assert expand_mul(csch(acsch(-I*(sqrt(6) + sqrt(2)))) / (-I*(sqrt(6) + sqrt(2)))) == 1
    assert expand_mul(csch(acsch(I*(1 + sqrt(5)))) / (I*(1 + sqrt(5)))) == 1
    assert (csch(acsch(I*sqrt(2 - 2/sqrt(5)))) / (I*sqrt(2 - 2/sqrt(5)))).simplify() == 1
    assert (csch(acsch(-I*sqrt(2 - 2/sqrt(5)))) / (-I*sqrt(2 - 2/sqrt(5)))).simplify() == 1

    # numerical evaluation
    assert str(acsch(5*I+1).n(6)) == '0.0391819 - 0.193363*I'
    assert str(acsch(-5*I+1).n(6)) == '0.0391819 + 0.193363*I'


def test_acsch_infinities():
    assert acsch(oo) == 0
    assert acsch(-oo) == 0
    assert acsch(zoo) == 0


def test_acsch_leading_term():
    x = Symbol('x')
    assert acsch(1/x).as_leading_term(x) == x
    # Tests concerning branch points
    assert acsch(x + I).as_leading_term(x) == -I*pi/2
    assert acsch(x - I).as_leading_term(x) == I*pi/2
    # Tests concerning points lying on branch cuts
    assert acsch(x).as_leading_term(x, cdir=1) == -log(x) + log(2)
    assert acsch(x).as_leading_term(x, cdir=-1) == log(x) - log(2) - I*pi
    assert acsch(x + I/2).as_leading_term(x, cdir=1) == -I*pi - acsch(I/2)
    assert acsch(x + I/2).as_leading_term(x, cdir=-1) == acsch(I/2)
    assert acsch(x - I/2).as_leading_term(x, cdir=1) == -acsch(I/2)
    assert acsch(x - I/2).as_leading_term(x, cdir=-1) == acsch(I/2) + I*pi
    # Tests concerning re(ndir) == 0
    assert acsch(I/2 + I*x - x**2).as_leading_term(x, cdir=1) == log(2 - sqrt(3)) - I*pi/2
    assert acsch(I/2 + I*x - x**2).as_leading_term(x, cdir=-1) == log(2 - sqrt(3)) - I*pi/2


def test_acsch_series():
    x = Symbol('x')
    assert acsch(x).series(x, 0, 9) == log(2) - log(x) + x**2/4 - 3*x**4/32 \
    + 5*x**6/96 - 35*x**8/1024 + O(x**9)
    t4 = acsch(x).taylor_term(4, x)
    assert t4 == -3*x**4/32
    assert acsch(x).taylor_term(6, x, t4, 0) == 5*x**6/96


def test_acsch_nseries():
    x = Symbol('x')
    # Tests concerning branch points
    assert acsch(x + I)._eval_nseries(x, 4, None) == -I*pi/2 + I*sqrt(x) + \
    sqrt(x) + 5*I*x**(S(3)/2)/12 - 5*x**(S(3)/2)/12 - 43*I*x**(S(5)/2)/160 - \
    43*x**(S(5)/2)/160 - 177*I*x**(S(7)/2)/896 + 177*x**(S(7)/2)/896 + O(x**4)
    assert acsch(x - I)._eval_nseries(x, 4, None) == I*pi/2 - I*sqrt(x) + \
    sqrt(x) - 5*I*x**(S(3)/2)/12 - 5*x**(S(3)/2)/12 + 43*I*x**(S(5)/2)/160 - \
    43*x**(S(5)/2)/160 + 177*I*x**(S(7)/2)/896 + 177*x**(S(7)/2)/896 + O(x**4)
    # Tests concerning points lying on branch cuts
    assert acsch(x + I/2)._eval_nseries(x, 4, None, cdir=1) == -acsch(I/2) - \
    I*pi + 4*sqrt(3)*I*x/3 - 8*sqrt(3)*x**2/9 - 16*sqrt(3)*I*x**3/9 + O(x**4)
    assert acsch(x + I/2)._eval_nseries(x, 4, None, cdir=-1) == acsch(I/2) - \
    4*sqrt(3)*I*x/3 + 8*sqrt(3)*x**2/9 + 16*sqrt(3)*I*x**3/9 + O(x**4)
    assert acsch(x - I/2)._eval_nseries(x, 4, None, cdir=1) == -acsch(I/2) - \
    4*sqrt(3)*I*x/3 - 8*sqrt(3)*x**2/9 + 16*sqrt(3)*I*x**3/9 + O(x**4)
    assert acsch(x - I/2)._eval_nseries(x, 4, None, cdir=-1) == I*pi + \
    acsch(I/2) + 4*sqrt(3)*I*x/3 + 8*sqrt(3)*x**2/9 - 16*sqrt(3)*I*x**3/9 + O(x**4)
    # TODO: Tests concerning re(ndir) == 0
    assert acsch(I/2 + I*x - x**2)._eval_nseries(x, 4, None) == -I*pi/2 + \
    log(2 - sqrt(3)) + 4*sqrt(3)*x/3 + x**2*(-8*sqrt(3)/9 + 4*sqrt(3)*I/3) + \
    x**3*(16*sqrt(3)/9 - 16*sqrt(3)*I/9) + O(x**4)


def test_acsch_rewrite():
    x = Symbol('x')
    assert acsch(x).rewrite(log) == log(1/x + sqrt(1/x**2 + 1))
    assert acsch(x).rewrite(asinh) == asinh(1/x)
    assert acsch(x).rewrite(atanh) == (sqrt(-x**2)*(-sqrt(-(x**2 + 1)**2)
                                                    *atanh(sqrt(x**2 + 1))/(x**2 + 1)
                                                    + pi/2)/x)


def test_acsch_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: acsch(x).fdiff(2))


def test_atanh():
    x = Symbol('x')

    # at specific points
    assert atanh(0) == 0
    assert atanh(I) == I*pi/4
    assert atanh(-I) == -I*pi/4
    assert atanh(1) is oo
    assert atanh(-1) is -oo
    assert atanh(nan) is nan

    # at infinites
    assert atanh(oo) == -I*pi/2
    assert atanh(-oo) == I*pi/2

    assert atanh(I*oo) == I*pi/2
    assert atanh(-I*oo) == -I*pi/2

    assert atanh(zoo) == I*AccumBounds(-pi/2, pi/2)

    # properties
    assert atanh(-x) == -atanh(x)

    # reality
    assert atanh(S(2)).is_real is False
    assert atanh(S(-1)/5).is_real is True
    assert atanh(symbols('y', extended_real=True)).is_real is None
    assert atanh(S(1)).is_real is False
    assert atanh(S(1)).is_extended_real is True
    assert atanh(S(-1)).is_real is False

    # special values
    assert atanh(I/sqrt(3)) == I*pi/6
    assert atanh(-I/sqrt(3)) == -I*pi/6
    assert atanh(I*sqrt(3)) == I*pi/3
    assert atanh(-I*sqrt(3)) == -I*pi/3
    assert atanh(I*(1 + sqrt(2))) == pi*I*Rational(3, 8)
    assert atanh(I*(sqrt(2) - 1)) == pi*I/8
    assert atanh(I*(1 - sqrt(2))) == -pi*I/8
    assert atanh(-I*(1 + sqrt(2))) == pi*I*Rational(-3, 8)
    assert atanh(I*sqrt(5 + 2*sqrt(5))) == I*pi*Rational(2, 5)
    assert atanh(-I*sqrt(5 + 2*sqrt(5))) == I*pi*Rational(-2, 5)
    assert atanh(I*(2 - sqrt(3))) == pi*I/12
    assert atanh(I*(sqrt(3) - 2)) == -pi*I/12
    assert atanh(oo) == -I*pi/2

    # Symmetry
    assert atanh(Rational(-1, 2)) == -atanh(S.Half)

    # inverse composition
    assert unchanged(atanh, tanh(Symbol('v1')))

    assert atanh(tanh(-5, evaluate=False)) == -5
    assert atanh(tanh(0, evaluate=False)) == 0
    assert atanh(tanh(7, evaluate=False)) == 7
    assert atanh(tanh(I, evaluate=False)) == I
    assert atanh(tanh(-I, evaluate=False)) == -I
    assert atanh(tanh(-11*I, evaluate=False)) == -11*I + 4*I*pi
    assert atanh(tanh(3 + I)) == 3 + I
    assert atanh(tanh(4 + 5*I)) == 4 - 2*I*pi + 5*I
    assert atanh(tanh(pi/2)) == pi/2
    assert atanh(tanh(pi)) == pi
    assert atanh(tanh(-3 + 7*I)) == -3 - 2*I*pi + 7*I
    assert atanh(tanh(9 - I*2/3)) == 9 - I*2/3
    assert atanh(tanh(-32 - 123*I)) == -32 - 123*I + 39*I*pi


def test_atanh_rewrite():
    x = Symbol('x')
    assert atanh(x).rewrite(log) == (log(1 + x) - log(1 - x)) / 2
    assert atanh(x).rewrite(asinh) == \
        pi*x/(2*sqrt(-x**2)) - sqrt(-x)*sqrt(1 - x**2)*sqrt(1/(x**2 - 1))*asinh(sqrt(1/(x**2 - 1)))/sqrt(x)


def test_atanh_leading_term():
    x = Symbol('x')
    assert atanh(x).as_leading_term(x) == x
    # Tests concerning branch points
    assert atanh(x + 1).as_leading_term(x, cdir=1) == -log(x)/2 + log(2)/2 - I*pi/2
    assert atanh(x + 1).as_leading_term(x, cdir=-1) == -log(x)/2 + log(2)/2 + I*pi/2
    assert atanh(x - 1).as_leading_term(x, cdir=1) == log(x)/2 - log(2)/2
    assert atanh(x - 1).as_leading_term(x, cdir=-1) == log(x)/2 - log(2)/2
    assert atanh(1/x).as_leading_term(x, cdir=1) == -I*pi/2
    assert atanh(1/x).as_leading_term(x, cdir=-1) == I*pi/2
    # Tests concerning points lying on branch cuts
    assert atanh(I*x + 2).as_leading_term(x, cdir=1) == atanh(2) + I*pi
    assert atanh(-I*x + 2).as_leading_term(x, cdir=1) == atanh(2)
    assert atanh(I*x - 2).as_leading_term(x, cdir=1) == -atanh(2)
    assert atanh(-I*x - 2).as_leading_term(x, cdir=1) == -I*pi - atanh(2)
    # Tests concerning im(ndir) == 0
    assert atanh(-I*x**2 + x - 2).as_leading_term(x, cdir=1) == -log(3)/2 - I*pi/2
    assert atanh(-I*x**2 + x - 2).as_leading_term(x, cdir=-1) == -log(3)/2 - I*pi/2


def test_atanh_series():
    x = Symbol('x')
    assert atanh(x).series(x, 0, 10) == \
        x + x**3/3 + x**5/5 + x**7/7 + x**9/9 + O(x**10)


def test_atanh_nseries():
    x = Symbol('x')
    # Tests concerning branch points
    assert atanh(x + 1)._eval_nseries(x, 4, None, cdir=1) == -I*pi/2 + log(2)/2 - \
    log(x)/2 + x/4 - x**2/16 + x**3/48 + O(x**4)
    assert atanh(x + 1)._eval_nseries(x, 4, None, cdir=-1) == I*pi/2 + log(2)/2 - \
    log(x)/2 + x/4 - x**2/16 + x**3/48 + O(x**4)
    assert atanh(x - 1)._eval_nseries(x, 4, None, cdir=1) == -log(2)/2 + log(x)/2 + \
    x/4 + x**2/16 + x**3/48 + O(x**4)
    assert atanh(x - 1)._eval_nseries(x, 4, None, cdir=-1) == -log(2)/2 + log(x)/2 + \
    x/4 + x**2/16 + x**3/48 + O(x**4)
    # Tests concerning points lying on branch cuts
    assert atanh(I*x + 2)._eval_nseries(x, 4, None, cdir=1) == I*pi + atanh(2) - \
    I*x/3 - 2*x**2/9 + 13*I*x**3/81 + O(x**4)
    assert atanh(I*x + 2)._eval_nseries(x, 4, None, cdir=-1) == atanh(2) - I*x/3 - \
    2*x**2/9 + 13*I*x**3/81 + O(x**4)
    assert atanh(I*x - 2)._eval_nseries(x, 4, None, cdir=1) == -atanh(2) - I*x/3 + \
    2*x**2/9 + 13*I*x**3/81 + O(x**4)
    assert atanh(I*x - 2)._eval_nseries(x, 4, None, cdir=-1) == -atanh(2) - I*pi - \
    I*x/3 + 2*x**2/9 + 13*I*x**3/81 + O(x**4)
    # Tests concerning im(ndir) == 0
    assert atanh(-I*x**2 + x - 2)._eval_nseries(x, 4, None) == -I*pi/2 - log(3)/2 - x/3 + \
    x**2*(-S(1)/4 + I/2) + x**2*(S(1)/36 - I/6) + x**3*(-S(1)/6 + I/2) + x**3*(S(1)/162 - I/18) + O(x**4)


def test_atanh_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: atanh(x).fdiff(2))


def test_acoth():
    x = Symbol('x')

    #at specific points
    assert acoth(0) == I*pi/2
    assert acoth(I) == -I*pi/4
    assert acoth(-I) == I*pi/4
    assert acoth(1) is oo
    assert acoth(-1) is -oo
    assert acoth(nan) is nan

    # at infinites
    assert acoth(oo) == 0
    assert acoth(-oo) == 0
    assert acoth(I*oo) == 0
    assert acoth(-I*oo) == 0
    assert acoth(zoo) == 0

    #properties
    assert acoth(-x) == -acoth(x)

    assert acoth(I/sqrt(3)) == -I*pi/3
    assert acoth(-I/sqrt(3)) == I*pi/3
    assert acoth(I*sqrt(3)) == -I*pi/6
    assert acoth(-I*sqrt(3)) == I*pi/6
    assert acoth(I*(1 + sqrt(2))) == -pi*I/8
    assert acoth(-I*(sqrt(2) + 1)) == pi*I/8
    assert acoth(I*(1 - sqrt(2))) == pi*I*Rational(3, 8)
    assert acoth(I*(sqrt(2) - 1)) == pi*I*Rational(-3, 8)
    assert acoth(I*sqrt(5 + 2*sqrt(5))) == -I*pi/10
    assert acoth(-I*sqrt(5 + 2*sqrt(5))) == I*pi/10
    assert acoth(I*(2 + sqrt(3))) == -pi*I/12
    assert acoth(-I*(2 + sqrt(3))) == pi*I/12
    assert acoth(I*(2 - sqrt(3))) == pi*I*Rational(-5, 12)
    assert acoth(I*(sqrt(3) - 2)) == pi*I*Rational(5, 12)

    # reality
    assert acoth(S(2)).is_real is True
    assert acoth(S(2)).is_finite is True
    assert acoth(S(2)).is_extended_real is True
    assert acoth(S(-2)).is_real is True
    assert acoth(S(1)).is_real is False
    assert acoth(S(1)).is_extended_real is True
    assert acoth(S(-1)).is_real is False
    assert acoth(symbols('y', real=True)).is_real is None

    # Symmetry
    assert acoth(Rational(-1, 2)) == -acoth(S.Half)


def test_acoth_rewrite():
    x = Symbol('x')
    assert acoth(x).rewrite(log) == (log(1 + 1/x) - log(1 - 1/x)) / 2
    assert acoth(x).rewrite(atanh) == atanh(1/x)
    assert acoth(x).rewrite(asinh) == \
        x*sqrt(x**(-2))*asinh(sqrt(1/(x**2 - 1))) + I*pi*(sqrt((x - 1)/x)*sqrt(x/(x - 1)) - sqrt(x/(x + 1))*sqrt(1 + 1/x))/2


def test_acoth_leading_term():
    x = Symbol('x')
    # Tests concerning branch points
    assert acoth(x + 1).as_leading_term(x, cdir=1) == -log(x)/2 + log(2)/2
    assert acoth(x + 1).as_leading_term(x, cdir=-1) == -log(x)/2 + log(2)/2
    assert acoth(x - 1).as_leading_term(x, cdir=1) == log(x)/2 - log(2)/2 + I*pi/2
    assert acoth(x - 1).as_leading_term(x, cdir=-1) == log(x)/2 - log(2)/2 - I*pi/2
    # Tests concerning points lying on branch cuts
    assert acoth(x).as_leading_term(x, cdir=-1) == I*pi/2
    assert acoth(x).as_leading_term(x, cdir=1) == -I*pi/2
    assert acoth(I*x + 1/2).as_leading_term(x, cdir=1) == acoth(1/2)
    assert acoth(-I*x + 1/2).as_leading_term(x, cdir=1) == acoth(1/2) + I*pi
    assert acoth(I*x - 1/2).as_leading_term(x, cdir=1) == -I*pi - acoth(1/2)
    assert acoth(-I*x - 1/2).as_leading_term(x, cdir=1) == -acoth(1/2)
    # Tests concerning im(ndir) == 0
    assert acoth(-I*x**2 - x - S(1)/2).as_leading_term(x, cdir=1) == -log(3)/2 + I*pi/2
    assert acoth(-I*x**2 - x - S(1)/2).as_leading_term(x, cdir=-1) == -log(3)/2 + I*pi/2


def test_acoth_series():
    x = Symbol('x')
    assert acoth(x).series(x, 0, 10) == \
        -I*pi/2 + x + x**3/3 + x**5/5 + x**7/7 + x**9/9 + O(x**10)


def test_acoth_nseries():
    x = Symbol('x')
    # Tests concerning branch points
    assert acoth(x + 1)._eval_nseries(x, 4, None) == log(2)/2 - log(x)/2 + x/4 - \
    x**2/16 + x**3/48 + O(x**4)
    assert acoth(x - 1)._eval_nseries(x, 4, None, cdir=1) == I*pi/2 - log(2)/2 + \
    log(x)/2 + x/4 + x**2/16 + x**3/48 + O(x**4)
    assert acoth(x - 1)._eval_nseries(x, 4, None, cdir=-1) == -I*pi/2 - log(2)/2 + \
    log(x)/2 + x/4 + x**2/16 + x**3/48 + O(x**4)
    # Tests concerning points lying on branch cuts
    assert acoth(I*x + S(1)/2)._eval_nseries(x, 4, None, cdir=1) == acoth(S(1)/2) + \
    4*I*x/3 - 8*x**2/9 - 112*I*x**3/81 + O(x**4)
    assert acoth(I*x + S(1)/2)._eval_nseries(x, 4, None, cdir=-1) == I*pi + \
    acoth(S(1)/2) + 4*I*x/3 - 8*x**2/9 - 112*I*x**3/81 + O(x**4)
    assert acoth(I*x - S(1)/2)._eval_nseries(x, 4, None, cdir=1) == -acoth(S(1)/2) - \
    I*pi + 4*I*x/3 + 8*x**2/9 - 112*I*x**3/81 + O(x**4)
    assert acoth(I*x - S(1)/2)._eval_nseries(x, 4, None, cdir=-1) == -acoth(S(1)/2) + \
    4*I*x/3 + 8*x**2/9 - 112*I*x**3/81 + O(x**4)
    # Tests concerning im(ndir) == 0
    assert acoth(-I*x**2 - x - S(1)/2)._eval_nseries(x, 4, None) == I*pi/2 - log(3)/2 - \
    4*x/3 + x**2*(-S(8)/9 + 2*I/3) - 2*I*x**2 + x**3*(S(104)/81 - 16*I/9) - 8*x**3/3 + O(x**4)


def test_acoth_fdiff():
    x = Symbol('x')
    raises(ArgumentIndexError, lambda: acoth(x).fdiff(2))


def test_inverses():
    x = Symbol('x')
    assert sinh(x).inverse() == asinh
    raises(AttributeError, lambda: cosh(x).inverse())
    assert tanh(x).inverse() == atanh
    assert coth(x).inverse() == acoth
    assert asinh(x).inverse() == sinh
    assert acosh(x).inverse() == cosh
    assert atanh(x).inverse() == tanh
    assert acoth(x).inverse() == coth
    assert asech(x).inverse() == sech
    assert acsch(x).inverse() == csch


def test_leading_term():
    x = Symbol('x')
    assert cosh(x).as_leading_term(x) == 1
    assert coth(x).as_leading_term(x) == 1/x
    for func in [sinh, tanh]:
        assert func(x).as_leading_term(x) == x
    for func in [sinh, cosh, tanh, coth]:
        for ar in (1/x, S.Half):
            eq = func(ar)
            assert eq.as_leading_term(x) == eq
    for func in [csch, sech]:
        eq = func(S.Half)
        assert eq.as_leading_term(x) == eq


def test_complex():
    a, b = symbols('a,b', real=True)
    z = a + b*I
    for func in [sinh, cosh, tanh, coth, sech, csch]:
        assert func(z).conjugate() == func(a - b*I)
    for deep in [True, False]:
        assert sinh(z).expand(
            complex=True, deep=deep) == sinh(a)*cos(b) + I*cosh(a)*sin(b)
        assert cosh(z).expand(
            complex=True, deep=deep) == cosh(a)*cos(b) + I*sinh(a)*sin(b)
        assert tanh(z).expand(complex=True, deep=deep) == sinh(a)*cosh(
            a)/(cos(b)**2 + sinh(a)**2) + I*sin(b)*cos(b)/(cos(b)**2 + sinh(a)**2)
        assert coth(z).expand(complex=True, deep=deep) == sinh(a)*cosh(
            a)/(sin(b)**2 + sinh(a)**2) - I*sin(b)*cos(b)/(sin(b)**2 + sinh(a)**2)
        assert csch(z).expand(complex=True, deep=deep) == cos(b) * sinh(a) / (sin(b)**2\
            *cosh(a)**2 + cos(b)**2 * sinh(a)**2) - I*sin(b) * cosh(a) / (sin(b)**2\
            *cosh(a)**2 + cos(b)**2 * sinh(a)**2)
        assert sech(z).expand(complex=True, deep=deep) == cos(b) * cosh(a) / (sin(b)**2\
            *sinh(a)**2 + cos(b)**2 * cosh(a)**2) - I*sin(b) * sinh(a) / (sin(b)**2\
            *sinh(a)**2 + cos(b)**2 * cosh(a)**2)


def test_complex_2899():
    a, b = symbols('a,b', real=True)
    for deep in [True, False]:
        for func in [sinh, cosh, tanh, coth]:
            assert func(a).expand(complex=True, deep=deep) == func(a)


def test_simplifications():
    x = Symbol('x')
    assert sinh(asinh(x)) == x
    assert sinh(acosh(x)) == sqrt(x - 1) * sqrt(x + 1)
    assert sinh(atanh(x)) == x/sqrt(1 - x**2)
    assert sinh(acoth(x)) == 1/(sqrt(x - 1) * sqrt(x + 1))

    assert cosh(asinh(x)) == sqrt(1 + x**2)
    assert cosh(acosh(x)) == x
    assert cosh(atanh(x)) == 1/sqrt(1 - x**2)
    assert cosh(acoth(x)) == x/(sqrt(x - 1) * sqrt(x + 1))

    assert tanh(asinh(x)) == x/sqrt(1 + x**2)
    assert tanh(acosh(x)) == sqrt(x - 1) * sqrt(x + 1) / x
    assert tanh(atanh(x)) == x
    assert tanh(acoth(x)) == 1/x

    assert coth(asinh(x)) == sqrt(1 + x**2)/x
    assert coth(acosh(x)) == x/(sqrt(x - 1) * sqrt(x + 1))
    assert coth(atanh(x)) == 1/x
    assert coth(acoth(x)) == x

    assert csch(asinh(x)) == 1/x
    assert csch(acosh(x)) == 1/(sqrt(x - 1) * sqrt(x + 1))
    assert csch(atanh(x)) == sqrt(1 - x**2)/x
    assert csch(acoth(x)) == sqrt(x - 1) * sqrt(x + 1)

    assert sech(asinh(x)) == 1/sqrt(1 + x**2)
    assert sech(acosh(x)) == 1/x
    assert sech(atanh(x)) == sqrt(1 - x**2)
    assert sech(acoth(x)) == sqrt(x - 1) * sqrt(x + 1)/x


def test_issue_4136():
    assert cosh(asinh(Integer(3)/2)) == sqrt(Integer(13)/4)


def test_sinh_rewrite():
    x = Symbol('x')
    assert sinh(x).rewrite(exp) == (exp(x) - exp(-x))/2 \
        == sinh(x).rewrite('tractable')
    assert sinh(x).rewrite(cosh) == -I*cosh(x + I*pi/2)
    tanh_half = tanh(S.Half*x)
    assert sinh(x).rewrite(tanh) == 2*tanh_half/(1 - tanh_half**2)
    coth_half = coth(S.Half*x)
    assert sinh(x).rewrite(coth) == 2*coth_half/(coth_half**2 - 1)


def test_cosh_rewrite():
    x = Symbol('x')
    assert cosh(x).rewrite(exp) == (exp(x) + exp(-x))/2 \
        == cosh(x).rewrite('tractable')
    assert cosh(x).rewrite(sinh) == -I*sinh(x + I*pi/2, evaluate=False)
    tanh_half = tanh(S.Half*x)**2
    assert cosh(x).rewrite(tanh) == (1 + tanh_half)/(1 - tanh_half)
    coth_half = coth(S.Half*x)**2
    assert cosh(x).rewrite(coth) == (coth_half + 1)/(coth_half - 1)


def test_tanh_rewrite():
    x = Symbol('x')
    assert tanh(x).rewrite(exp) == (exp(x) - exp(-x))/(exp(x) + exp(-x)) \
        == tanh(x).rewrite('tractable')
    assert tanh(x).rewrite(sinh) == I*sinh(x)/sinh(I*pi/2 - x, evaluate=False)
    assert tanh(x).rewrite(cosh) == I*cosh(I*pi/2 - x, evaluate=False)/cosh(x)
    assert tanh(x).rewrite(coth) == 1/coth(x)


def test_coth_rewrite():
    x = Symbol('x')
    assert coth(x).rewrite(exp) == (exp(x) + exp(-x))/(exp(x) - exp(-x)) \
        == coth(x).rewrite('tractable')
    assert coth(x).rewrite(sinh) == -I*sinh(I*pi/2 - x, evaluate=False)/sinh(x)
    assert coth(x).rewrite(cosh) == -I*cosh(x)/cosh(I*pi/2 - x, evaluate=False)
    assert coth(x).rewrite(tanh) == 1/tanh(x)


def test_csch_rewrite():
    x = Symbol('x')
    assert csch(x).rewrite(exp) == 1 / (exp(x)/2 - exp(-x)/2) \
        == csch(x).rewrite('tractable')
    assert csch(x).rewrite(cosh) == I/cosh(x + I*pi/2, evaluate=False)
    tanh_half = tanh(S.Half*x)
    assert csch(x).rewrite(tanh) == (1 - tanh_half**2)/(2*tanh_half)
    coth_half = coth(S.Half*x)
    assert csch(x).rewrite(coth) == (coth_half**2 - 1)/(2*coth_half)


def test_sech_rewrite():
    x = Symbol('x')
    assert sech(x).rewrite(exp) == 1 / (exp(x)/2 + exp(-x)/2) \
        == sech(x).rewrite('tractable')
    assert sech(x).rewrite(sinh) == I/sinh(x + I*pi/2, evaluate=False)
    tanh_half = tanh(S.Half*x)**2
    assert sech(x).rewrite(tanh) == (1 - tanh_half)/(1 + tanh_half)
    coth_half = coth(S.Half*x)**2
    assert sech(x).rewrite(coth) == (coth_half - 1)/(coth_half + 1)


def test_derivs():
    x = Symbol('x')
    assert coth(x).diff(x) == -sinh(x)**(-2)
    assert sinh(x).diff(x) == cosh(x)
    assert cosh(x).diff(x) == sinh(x)
    assert tanh(x).diff(x) == -tanh(x)**2 + 1
    assert csch(x).diff(x) == -coth(x)*csch(x)
    assert sech(x).diff(x) == -tanh(x)*sech(x)
    assert acoth(x).diff(x) == 1/(-x**2 + 1)
    assert asinh(x).diff(x) == 1/sqrt(x**2 + 1)
    assert acosh(x).diff(x) == 1/(sqrt(x - 1)*sqrt(x + 1))
    assert acosh(x).diff(x) == acosh(x).rewrite(log).diff(x).together()
    assert atanh(x).diff(x) == 1/(-x**2 + 1)
    assert asech(x).diff(x) == -1/(x*sqrt(1 - x**2))
    assert acsch(x).diff(x) == -1/(x**2*sqrt(1 + x**(-2)))


def test_sinh_expansion():
    x, y = symbols('x,y')
    assert sinh(x+y).expand(trig=True) == sinh(x)*cosh(y) + cosh(x)*sinh(y)
    assert sinh(2*x).expand(trig=True) == 2*sinh(x)*cosh(x)
    assert sinh(3*x).expand(trig=True).expand() == \
        sinh(x)**3 + 3*sinh(x)*cosh(x)**2


def test_cosh_expansion():
    x, y = symbols('x,y')
    assert cosh(x+y).expand(trig=True) == cosh(x)*cosh(y) + sinh(x)*sinh(y)
    assert cosh(2*x).expand(trig=True) == cosh(x)**2 + sinh(x)**2
    assert cosh(3*x).expand(trig=True).expand() == \
        3*sinh(x)**2*cosh(x) + cosh(x)**3

def test_cosh_positive():
    # See issue 11721
    # cosh(x) is positive for real values of x
    k = symbols('k', real=True)
    n = symbols('n', integer=True)

    assert cosh(k, evaluate=False).is_positive is True
    assert cosh(k + 2*n*pi*I, evaluate=False).is_positive is True
    assert cosh(I*pi/4, evaluate=False).is_positive is True
    assert cosh(3*I*pi/4, evaluate=False).is_positive is False

def test_cosh_nonnegative():
    k = symbols('k', real=True)
    n = symbols('n', integer=True)

    assert cosh(k, evaluate=False).is_nonnegative is True
    assert cosh(k + 2*n*pi*I, evaluate=False).is_nonnegative is True
    assert cosh(I*pi/4, evaluate=False).is_nonnegative is True
    assert cosh(3*I*pi/4, evaluate=False).is_nonnegative is False
    assert cosh(S.Zero, evaluate=False).is_nonnegative is True

def test_real_assumptions():
    z = Symbol('z', real=False)
    assert sinh(z).is_real is None
    assert cosh(z).is_real is None
    assert tanh(z).is_real is None
    assert sech(z).is_real is None
    assert csch(z).is_real is None
    assert coth(z).is_real is None

def test_sign_assumptions():
    p = Symbol('p', positive=True)
    n = Symbol('n', negative=True)
    assert sinh(n).is_negative is True
    assert sinh(p).is_positive is True
    assert cosh(n).is_positive is True
    assert cosh(p).is_positive is True
    assert tanh(n).is_negative is True
    assert tanh(p).is_positive is True
    assert csch(n).is_negative is True
    assert csch(p).is_positive is True
    assert sech(n).is_positive is True
    assert sech(p).is_positive is True
    assert coth(n).is_negative is True
    assert coth(p).is_positive is True


def test_issue_25847():
    x = Symbol('x')

    #atanh
    assert atanh(sin(x)/x).as_leading_term(x) == atanh(sin(x)/x)
    raises(PoleError, lambda: atanh(exp(1/x)).as_leading_term(x))

    #asinh
    assert asinh(sin(x)/x).as_leading_term(x) == log(1 + sqrt(2))
    raises(PoleError, lambda: asinh(exp(1/x)).as_leading_term(x))

    #acosh
    assert acosh(sin(x)/x).as_leading_term(x) == 0
    raises(PoleError, lambda: acosh(exp(1/x)).as_leading_term(x))

    #acoth
    assert acoth(sin(x)/x).as_leading_term(x) == acoth(sin(x)/x)
    raises(PoleError, lambda: acoth(exp(1/x)).as_leading_term(x))

    #asech
    assert asech(sinh(x)/x).as_leading_term(x) == 0
    raises(PoleError, lambda: asech(exp(1/x)).as_leading_term(x))

    #acsch
    assert acsch(sin(x)/x).as_leading_term(x) == log(1 + sqrt(2))
    raises(PoleError, lambda: acsch(exp(1/x)).as_leading_term(x))
