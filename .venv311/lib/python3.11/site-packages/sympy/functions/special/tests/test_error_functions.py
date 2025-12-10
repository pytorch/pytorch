from sympy.core.function import (diff, expand, expand_func)
from sympy.core import EulerGamma
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols, Dummy)
from sympy.functions.elementary.complexes import (conjugate, im, polar_lift, re)
from sympy.functions.elementary.exponential import (exp, exp_polar, log)
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin, sinc)
from sympy.functions.special.error_functions import (Chi, Ci, E1, Ei, Li, Shi, Si, erf, erf2, erf2inv, erfc, erfcinv, erfi, erfinv, expint, fresnelc, fresnels, li)
from sympy.functions.special.gamma_functions import (gamma, uppergamma)
from sympy.functions.special.hyper import (hyper, meijerg)
from sympy.integrals.integrals import (Integral, integrate)
from sympy.series.gruntz import gruntz
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.functions.special.error_functions import _erfs, _eis
from sympy.testing.pytest import raises

x, y, z = symbols('x,y,z')
w = Symbol("w", real=True)
n = Symbol("n", integer=True)
t = Dummy('t')


def test_erf():
    assert erf(nan) is nan

    assert erf(oo) == 1
    assert erf(-oo) == -1

    assert erf(0) is S.Zero

    assert erf(I*oo) == oo*I
    assert erf(-I*oo) == -oo*I

    assert erf(-2) == -erf(2)
    assert erf(-x*y) == -erf(x*y)
    assert erf(-x - y) == -erf(x + y)

    assert erf(erfinv(x)) == x
    assert erf(erfcinv(x)) == 1 - x
    assert erf(erf2inv(0, x)) == x
    assert erf(erf2inv(0, x, evaluate=False)) == x # To cover code in erf
    assert erf(erf2inv(0, erf(erfcinv(1 - erf(erfinv(x)))))) == x

    alpha = symbols('alpha', extended_real=True)
    assert erf(alpha).is_real is True
    assert erf(alpha).is_finite is True
    alpha = symbols('alpha', extended_real=False)
    assert erf(alpha).is_real is None
    assert erf(alpha).is_finite is None
    assert erf(alpha).is_zero is None
    assert erf(alpha).is_positive is None
    assert erf(alpha).is_negative is None
    alpha = symbols('alpha', extended_positive=True)
    assert erf(alpha).is_positive is True
    alpha = symbols('alpha', extended_negative=True)
    assert erf(alpha).is_negative is True
    assert erf(I).is_real is False
    assert erf(0, evaluate=False).is_real
    assert erf(0, evaluate=False).is_zero

    assert conjugate(erf(z)) == erf(conjugate(z))

    assert erf(x).as_leading_term(x) == 2*x/sqrt(pi)
    assert erf(x*y).as_leading_term(y) == 2*x*y/sqrt(pi)
    assert (erf(x*y)/erf(y)).as_leading_term(y) == x
    assert erf(1/x).as_leading_term(x) == S.One

    assert erf(z).rewrite('uppergamma') == sqrt(z**2)*(1 - erfc(sqrt(z**2)))/z
    assert erf(z).rewrite('erfc') == S.One - erfc(z)
    assert erf(z).rewrite('erfi') == -I*erfi(I*z)
    assert erf(z).rewrite('fresnels') == (1 + I)*(fresnelc(z*(1 - I)/sqrt(pi)) -
        I*fresnels(z*(1 - I)/sqrt(pi)))
    assert erf(z).rewrite('fresnelc') == (1 + I)*(fresnelc(z*(1 - I)/sqrt(pi)) -
        I*fresnels(z*(1 - I)/sqrt(pi)))
    assert erf(z).rewrite('hyper') == 2*z*hyper([S.Half], [3*S.Half], -z**2)/sqrt(pi)
    assert erf(z).rewrite('meijerg') == z*meijerg([S.Half], [], [0], [Rational(-1, 2)], z**2)/sqrt(pi)
    assert erf(z).rewrite('expint') == sqrt(z**2)/z - z*expint(S.Half, z**2)/sqrt(S.Pi)

    assert limit(exp(x)*exp(x**2)*(erf(x + 1/exp(x)) - erf(x)), x, oo) == \
        2/sqrt(pi)
    assert limit((1 - erf(z))*exp(z**2)*z, z, oo) == 1/sqrt(pi)
    assert limit((1 - erf(x))*exp(x**2)*sqrt(pi)*x, x, oo) == 1
    assert limit(((1 - erf(x))*exp(x**2)*sqrt(pi)*x - 1)*2*x**2, x, oo) == -1
    assert limit(erf(x)/x, x, 0) == 2/sqrt(pi)
    assert limit(x**(-4) - sqrt(pi)*erf(x**2) / (2*x**6), x, 0) == S(1)/3

    assert erf(x).as_real_imag() == \
        (erf(re(x) - I*im(x))/2 + erf(re(x) + I*im(x))/2,
         -I*(-erf(re(x) - I*im(x)) + erf(re(x) + I*im(x)))/2)

    assert erf(x).as_real_imag(deep=False) == \
        (erf(re(x) - I*im(x))/2 + erf(re(x) + I*im(x))/2,
         -I*(-erf(re(x) - I*im(x)) + erf(re(x) + I*im(x)))/2)

    assert erf(w).as_real_imag() == (erf(w), 0)
    assert erf(w).as_real_imag(deep=False) == (erf(w), 0)
    # issue 13575
    assert erf(I).as_real_imag() == (0, -I*erf(I))

    raises(ArgumentIndexError, lambda: erf(x).fdiff(2))

    assert erf(x).inverse() == erfinv


def test_erf_series():
    assert erf(x).series(x, 0, 7) == 2*x/sqrt(pi) - \
        2*x**3/3/sqrt(pi) + x**5/5/sqrt(pi) + O(x**7)

    assert erf(x).series(x, oo) == \
        -exp(-x**2)*(3/(4*x**5) - 1/(2*x**3) + 1/x + O(x**(-6), (x, oo)))/sqrt(pi) + 1
    assert erf(x**2).series(x, oo, n=8) == \
        (-1/(2*x**6) + x**(-2) + O(x**(-8), (x, oo)))*exp(-x**4)/sqrt(pi)*-1 + 1
    assert erf(sqrt(x)).series(x, oo, n=3) == (sqrt(1/x) - (1/x)**(S(3)/2)/2\
        + 3*(1/x)**(S(5)/2)/4 + O(x**(-3), (x, oo)))*exp(-x)/sqrt(pi)*-1 + 1


def test_erf_evalf():
    assert abs( erf(Float(2.0)) - 0.995322265 ) < 1E-8 # XXX


def test__erfs():
    assert _erfs(z).diff(z) == -2/sqrt(S.Pi) + 2*z*_erfs(z)

    assert _erfs(1/z).series(z) == \
        z/sqrt(pi) - z**3/(2*sqrt(pi)) + 3*z**5/(4*sqrt(pi)) + O(z**6)

    assert expand(erf(z).rewrite('tractable').diff(z).rewrite('intractable')) \
        == erf(z).diff(z)
    assert _erfs(z).rewrite("intractable") == (-erf(z) + 1)*exp(z**2)
    raises(ArgumentIndexError, lambda: _erfs(z).fdiff(2))


def test_erfc():
    assert erfc(nan) is nan

    assert erfc(oo) is S.Zero
    assert erfc(-oo) == 2

    assert erfc(0) == 1

    assert erfc(I*oo) == -oo*I
    assert erfc(-I*oo) == oo*I

    assert erfc(-x) == S(2) - erfc(x)
    assert erfc(erfcinv(x)) == x

    alpha = symbols('alpha', extended_real=True)
    assert erfc(alpha).is_real is True
    alpha = symbols('alpha', extended_real=False)
    assert erfc(alpha).is_real is None
    assert erfc(I).is_real is False
    assert erfc(0, evaluate=False).is_real
    assert erfc(0, evaluate=False).is_zero is False

    assert erfc(erfinv(x)) == 1 - x

    assert conjugate(erfc(z)) == erfc(conjugate(z))

    assert erfc(x).as_leading_term(x) is S.One
    assert erfc(1/x).as_leading_term(x) == S.Zero

    assert erfc(z).rewrite('erf') == 1 - erf(z)
    assert erfc(z).rewrite('erfi') == 1 + I*erfi(I*z)
    assert erfc(z).rewrite('fresnels') == 1 - (1 + I)*(fresnelc(z*(1 - I)/sqrt(pi)) -
        I*fresnels(z*(1 - I)/sqrt(pi)))
    assert erfc(z).rewrite('fresnelc') == 1 - (1 + I)*(fresnelc(z*(1 - I)/sqrt(pi)) -
        I*fresnels(z*(1 - I)/sqrt(pi)))
    assert erfc(z).rewrite('hyper') == 1 - 2*z*hyper([S.Half], [3*S.Half], -z**2)/sqrt(pi)
    assert erfc(z).rewrite('meijerg') == 1 - z*meijerg([S.Half], [], [0], [Rational(-1, 2)], z**2)/sqrt(pi)
    assert erfc(z).rewrite('uppergamma') == 1 - sqrt(z**2)*(1 - erfc(sqrt(z**2)))/z
    assert erfc(z).rewrite('expint') == S.One - sqrt(z**2)/z + z*expint(S.Half, z**2)/sqrt(S.Pi)
    assert erfc(z).rewrite('tractable') == _erfs(z)*exp(-z**2)
    assert expand_func(erf(x) + erfc(x)) is S.One

    assert erfc(x).as_real_imag() == \
        (erfc(re(x) - I*im(x))/2 + erfc(re(x) + I*im(x))/2,
         -I*(-erfc(re(x) - I*im(x)) + erfc(re(x) + I*im(x)))/2)

    assert erfc(x).as_real_imag(deep=False) == \
        (erfc(re(x) - I*im(x))/2 + erfc(re(x) + I*im(x))/2,
         -I*(-erfc(re(x) - I*im(x)) + erfc(re(x) + I*im(x)))/2)

    assert erfc(w).as_real_imag() == (erfc(w), 0)
    assert erfc(w).as_real_imag(deep=False) == (erfc(w), 0)
    raises(ArgumentIndexError, lambda: erfc(x).fdiff(2))

    assert erfc(x).inverse() == erfcinv


def test_erfc_series():
    assert erfc(x).series(x, 0, 7) == 1 - 2*x/sqrt(pi) + \
        2*x**3/3/sqrt(pi) - x**5/5/sqrt(pi) + O(x**7)

    assert erfc(x).series(x, oo) == \
            (3/(4*x**5) - 1/(2*x**3) + 1/x + O(x**(-6), (x, oo)))*exp(-x**2)/sqrt(pi)


def test_erfc_evalf():
    assert abs( erfc(Float(2.0)) - 0.00467773 ) < 1E-8 # XXX


def test_erfi():
    assert erfi(nan) is nan

    assert erfi(oo) is S.Infinity
    assert erfi(-oo) is S.NegativeInfinity

    assert erfi(0) is S.Zero

    assert erfi(I*oo) == I
    assert erfi(-I*oo) == -I

    assert erfi(-x) == -erfi(x)

    assert erfi(I*erfinv(x)) == I*x
    assert erfi(I*erfcinv(x)) == I*(1 - x)
    assert erfi(I*erf2inv(0, x)) == I*x
    assert erfi(I*erf2inv(0, x, evaluate=False)) == I*x # To cover code in erfi

    assert erfi(I).is_real is False
    assert erfi(0, evaluate=False).is_real
    assert erfi(0, evaluate=False).is_zero

    assert conjugate(erfi(z)) == erfi(conjugate(z))

    assert erfi(x).as_leading_term(x) == 2*x/sqrt(pi)
    assert erfi(x*y).as_leading_term(y) == 2*x*y/sqrt(pi)
    assert (erfi(x*y)/erfi(y)).as_leading_term(y) == x
    assert erfi(1/x).as_leading_term(x) == erfi(1/x)

    assert erfi(z).rewrite('erf') == -I*erf(I*z)
    assert erfi(z).rewrite('erfc') == I*erfc(I*z) - I
    assert erfi(z).rewrite('fresnels') == (1 - I)*(fresnelc(z*(1 + I)/sqrt(pi)) -
        I*fresnels(z*(1 + I)/sqrt(pi)))
    assert erfi(z).rewrite('fresnelc') == (1 - I)*(fresnelc(z*(1 + I)/sqrt(pi)) -
        I*fresnels(z*(1 + I)/sqrt(pi)))
    assert erfi(z).rewrite('hyper') == 2*z*hyper([S.Half], [3*S.Half], z**2)/sqrt(pi)
    assert erfi(z).rewrite('meijerg') == z*meijerg([S.Half], [], [0], [Rational(-1, 2)], -z**2)/sqrt(pi)
    assert erfi(z).rewrite('uppergamma') == (sqrt(-z**2)/z*(uppergamma(S.Half,
        -z**2)/sqrt(S.Pi) - S.One))
    assert erfi(z).rewrite('expint') == sqrt(-z**2)/z - z*expint(S.Half, -z**2)/sqrt(S.Pi)
    assert erfi(z).rewrite('tractable') == -I*(-_erfs(I*z)*exp(z**2) + 1)
    assert expand_func(erfi(I*z)) == I*erf(z)

    assert erfi(x).as_real_imag() == \
        (erfi(re(x) - I*im(x))/2 + erfi(re(x) + I*im(x))/2,
         -I*(-erfi(re(x) - I*im(x)) + erfi(re(x) + I*im(x)))/2)
    assert erfi(x).as_real_imag(deep=False) == \
        (erfi(re(x) - I*im(x))/2 + erfi(re(x) + I*im(x))/2,
         -I*(-erfi(re(x) - I*im(x)) + erfi(re(x) + I*im(x)))/2)

    assert erfi(w).as_real_imag() == (erfi(w), 0)
    assert erfi(w).as_real_imag(deep=False) == (erfi(w), 0)

    raises(ArgumentIndexError, lambda: erfi(x).fdiff(2))


def test_erfi_series():
    assert erfi(x).series(x, 0, 7) == 2*x/sqrt(pi) + \
        2*x**3/3/sqrt(pi) + x**5/5/sqrt(pi) + O(x**7)

    assert erfi(x).series(x, oo) == \
        (3/(4*x**5) + 1/(2*x**3) + 1/x + O(x**(-6), (x, oo)))*exp(x**2)/sqrt(pi) - I


def test_erfi_evalf():
    assert abs( erfi(Float(2.0)) - 18.5648024145756 ) < 1E-13  # XXX


def test_erf2():

    assert erf2(0, 0) is S.Zero
    assert erf2(x, x) is S.Zero
    assert erf2(nan, 0) is nan

    assert erf2(-oo,  y) ==  erf(y) + 1
    assert erf2( oo,  y) ==  erf(y) - 1
    assert erf2(  x, oo) ==  1 - erf(x)
    assert erf2(  x,-oo) == -1 - erf(x)
    assert erf2(x, erf2inv(x, y)) == y

    assert erf2(-x, -y) == -erf2(x,y)
    assert erf2(-x,  y) == erf(y) + erf(x)
    assert erf2( x, -y) == -erf(y) - erf(x)
    assert erf2(x, y).rewrite('fresnels') == erf(y).rewrite(fresnels)-erf(x).rewrite(fresnels)
    assert erf2(x, y).rewrite('fresnelc') == erf(y).rewrite(fresnelc)-erf(x).rewrite(fresnelc)
    assert erf2(x, y).rewrite('hyper') == erf(y).rewrite(hyper)-erf(x).rewrite(hyper)
    assert erf2(x, y).rewrite('meijerg') == erf(y).rewrite(meijerg)-erf(x).rewrite(meijerg)
    assert erf2(x, y).rewrite('uppergamma') == erf(y).rewrite(uppergamma) - erf(x).rewrite(uppergamma)
    assert erf2(x, y).rewrite('expint') == erf(y).rewrite(expint)-erf(x).rewrite(expint)

    assert erf2(I, 0).is_real is False
    assert erf2(0, 0, evaluate=False).is_real
    assert erf2(0, 0, evaluate=False).is_zero
    assert erf2(x, x, evaluate=False).is_zero
    assert erf2(x, y).is_zero is None

    assert expand_func(erf(x) + erf2(x, y)) == erf(y)

    assert conjugate(erf2(x, y)) == erf2(conjugate(x), conjugate(y))

    assert erf2(x, y).rewrite('erf')  == erf(y) - erf(x)
    assert erf2(x, y).rewrite('erfc') == erfc(x) - erfc(y)
    assert erf2(x, y).rewrite('erfi') == I*(erfi(I*x) - erfi(I*y))

    assert erf2(x, y).diff(x) == erf2(x, y).fdiff(1)
    assert erf2(x, y).diff(y) == erf2(x, y).fdiff(2)
    assert erf2(x, y).diff(x) == -2*exp(-x**2)/sqrt(pi)
    assert erf2(x, y).diff(y) == 2*exp(-y**2)/sqrt(pi)
    raises(ArgumentIndexError, lambda: erf2(x, y).fdiff(3))

    assert erf2(x, y).is_extended_real is None
    xr, yr = symbols('xr yr', extended_real=True)
    assert erf2(xr, yr).is_extended_real is True


def test_erfinv():
    assert erfinv(0) is S.Zero
    assert erfinv(1) is S.Infinity
    assert erfinv(nan) is S.NaN
    assert erfinv(-1) is S.NegativeInfinity

    assert erfinv(erf(w)) == w
    assert erfinv(erf(-w)) == -w

    assert erfinv(x).diff() == sqrt(pi)*exp(erfinv(x)**2)/2
    raises(ArgumentIndexError, lambda: erfinv(x).fdiff(2))

    assert erfinv(z).rewrite('erfcinv') == erfcinv(1-z)
    assert erfinv(z).inverse() == erf


def test_erfinv_evalf():
    assert abs( erfinv(Float(0.2)) - 0.179143454621292 ) < 1E-13


def test_erfcinv():
    assert erfcinv(1) is S.Zero
    assert erfcinv(0) is S.Infinity
    assert erfcinv(0, evaluate=False).is_infinite is True
    assert erfcinv(2, evaluate=False).is_infinite is True
    assert erfcinv(nan) is S.NaN

    assert erfcinv(x).diff() == -sqrt(pi)*exp(erfcinv(x)**2)/2
    raises(ArgumentIndexError, lambda: erfcinv(x).fdiff(2))

    assert erfcinv(z).rewrite('erfinv') == erfinv(1-z)
    assert erfcinv(z).inverse() == erfc


def test_erf2inv():
    assert erf2inv(0, 0) is S.Zero
    assert erf2inv(0, 1) is S.Infinity
    assert erf2inv(1, 0) is S.One
    assert erf2inv(0, y) == erfinv(y)
    assert erf2inv(oo, y) == erfcinv(-y)
    assert erf2inv(x, 0) == x
    assert erf2inv(x, oo) == erfinv(x)
    assert erf2inv(nan, 0) is nan
    assert erf2inv(0, nan) is nan

    assert erf2inv(x, y).diff(x) == exp(-x**2 + erf2inv(x, y)**2)
    assert erf2inv(x, y).diff(y) == sqrt(pi)*exp(erf2inv(x, y)**2)/2
    raises(ArgumentIndexError, lambda: erf2inv(x, y).fdiff(3))


# NOTE we multiply by exp_polar(I*pi) and need this to be on the principal
# branch, hence take x in the lower half plane (d=0).


def mytn(expr1, expr2, expr3, x, d=0):
    from sympy.core.random import verify_numerically, random_complex_number
    subs = {}
    for a in expr1.free_symbols:
        if a != x:
            subs[a] = random_complex_number()
    return expr2 == expr3 and verify_numerically(expr1.subs(subs),
                                               expr2.subs(subs), x, d=d)


def mytd(expr1, expr2, x):
    from sympy.core.random import test_derivative_numerically, \
        random_complex_number
    subs = {}
    for a in expr1.free_symbols:
        if a != x:
            subs[a] = random_complex_number()
    return expr1.diff(x) == expr2 and test_derivative_numerically(expr1.subs(subs), x)


def tn_branch(func, s=None):
    from sympy.core.random import uniform

    def fn(x):
        if s is None:
            return func(x)
        return func(s, x)
    c = uniform(1, 5)
    expr = fn(c*exp_polar(I*pi)) - fn(c*exp_polar(-I*pi))
    eps = 1e-15
    expr2 = fn(-c + eps*I) - fn(-c - eps*I)
    return abs(expr.n() - expr2.n()).n() < 1e-10


def test_ei():
    assert Ei(0) is S.NegativeInfinity
    assert Ei(oo) is S.Infinity
    assert Ei(-oo) is S.Zero

    assert tn_branch(Ei)
    assert mytd(Ei(x), exp(x)/x, x)
    assert mytn(Ei(x), Ei(x).rewrite(uppergamma),
                -uppergamma(0, x*polar_lift(-1)) - I*pi, x)
    assert mytn(Ei(x), Ei(x).rewrite(expint),
                -expint(1, x*polar_lift(-1)) - I*pi, x)
    assert Ei(x).rewrite(expint).rewrite(Ei) == Ei(x)
    assert Ei(x*exp_polar(2*I*pi)) == Ei(x) + 2*I*pi
    assert Ei(x*exp_polar(-2*I*pi)) == Ei(x) - 2*I*pi

    assert mytn(Ei(x), Ei(x).rewrite(Shi), Chi(x) + Shi(x), x)
    assert mytn(Ei(x*polar_lift(I)), Ei(x*polar_lift(I)).rewrite(Si),
                Ci(x) + I*Si(x) + I*pi/2, x)

    assert Ei(log(x)).rewrite(li) == li(x)
    assert Ei(2*log(x)).rewrite(li) == li(x**2)

    assert gruntz(Ei(x+exp(-x))*exp(-x)*x, x, oo) == 1

    assert Ei(x).series(x) == EulerGamma + log(x) + x + x**2/4 + \
        x**3/18 + x**4/96 + x**5/600 + O(x**6)
    assert Ei(x).series(x, 1, 3) == Ei(1) + E*(x - 1) + O((x - 1)**3, (x, 1))
    assert Ei(x).series(x, oo) == \
        (120/x**5 + 24/x**4 + 6/x**3 + 2/x**2 + 1/x + 1 + O(x**(-6), (x, oo)))*exp(x)/x
    assert Ei(x).series(x, -oo) == \
        (120/x**5 + 24/x**4 + 6/x**3 + 2/x**2 + 1/x + 1 + O(x**(-6), (x, -oo)))*exp(x)/x
    assert Ei(-x).series(x, oo) == \
        -((-120/x**5 + 24/x**4 - 6/x**3 + 2/x**2 - 1/x + 1 + O(x**(-6), (x, oo)))*exp(-x)/x)

    assert str(Ei(cos(2)).evalf(n=10)) == '-0.6760647401'
    raises(ArgumentIndexError, lambda: Ei(x).fdiff(2))


def test_expint():
    assert mytn(expint(x, y), expint(x, y).rewrite(uppergamma),
                y**(x - 1)*uppergamma(1 - x, y), x)
    assert mytd(
        expint(x, y), -y**(x - 1)*meijerg([], [1, 1], [0, 0, 1 - x], [], y), x)
    assert mytd(expint(x, y), -expint(x - 1, y), y)
    assert mytn(expint(1, x), expint(1, x).rewrite(Ei),
                -Ei(x*polar_lift(-1)) + I*pi, x)

    assert expint(-4, x) == exp(-x)/x + 4*exp(-x)/x**2 + 12*exp(-x)/x**3 \
        + 24*exp(-x)/x**4 + 24*exp(-x)/x**5
    assert expint(Rational(-3, 2), x) == \
        exp(-x)/x + 3*exp(-x)/(2*x**2) + 3*sqrt(pi)*erfc(sqrt(x))/(4*x**S('5/2'))

    assert tn_branch(expint, 1)
    assert tn_branch(expint, 2)
    assert tn_branch(expint, 3)
    assert tn_branch(expint, 1.7)
    assert tn_branch(expint, pi)

    assert expint(y, x*exp_polar(2*I*pi)) == \
        x**(y - 1)*(exp(2*I*pi*y) - 1)*gamma(-y + 1) + expint(y, x)
    assert expint(y, x*exp_polar(-2*I*pi)) == \
        x**(y - 1)*(exp(-2*I*pi*y) - 1)*gamma(-y + 1) + expint(y, x)
    assert expint(2, x*exp_polar(2*I*pi)) == 2*I*pi*x + expint(2, x)
    assert expint(2, x*exp_polar(-2*I*pi)) == -2*I*pi*x + expint(2, x)
    assert expint(1, x).rewrite(Ei).rewrite(expint) == expint(1, x)
    assert expint(x, y).rewrite(Ei) == expint(x, y)
    assert expint(x, y).rewrite(Ci) == expint(x, y)

    assert mytn(E1(x), E1(x).rewrite(Shi), Shi(x) - Chi(x), x)
    assert mytn(E1(polar_lift(I)*x), E1(polar_lift(I)*x).rewrite(Si),
                -Ci(x) + I*Si(x) - I*pi/2, x)

    assert mytn(expint(2, x), expint(2, x).rewrite(Ei).rewrite(expint),
                -x*E1(x) + exp(-x), x)
    assert mytn(expint(3, x), expint(3, x).rewrite(Ei).rewrite(expint),
                x**2*E1(x)/2 + (1 - x)*exp(-x)/2, x)

    assert expint(Rational(3, 2), z).nseries(z) == \
        2 + 2*z - z**2/3 + z**3/15 - z**4/84 + z**5/540 - \
        2*sqrt(pi)*sqrt(z) + O(z**6)

    assert E1(z).series(z) == -EulerGamma - log(z) + z - \
        z**2/4 + z**3/18 - z**4/96 + z**5/600 + O(z**6)

    assert expint(4, z).series(z) == Rational(1, 3) - z/2 + z**2/2 + \
        z**3*(log(z)/6 - Rational(11, 36) + EulerGamma/6 - I*pi/6) - z**4/24 + \
        z**5/240 + O(z**6)

    assert expint(n, x).series(x, oo, n=3) == \
        (n*(n + 1)/x**2 - n/x + 1 + O(x**(-3), (x, oo)))*exp(-x)/x

    assert expint(z, y).series(z, 0, 2) == exp(-y)/y - z*meijerg(((), (1, 1)),
                                  ((0, 0, 1), ()), y)/y + O(z**2)
    raises(ArgumentIndexError, lambda: expint(x, y).fdiff(3))

    neg = Symbol('neg', negative=True)
    assert Ei(neg).rewrite(Si) == Shi(neg) + Chi(neg) - I*pi


def test__eis():
    assert _eis(z).diff(z) == -_eis(z) + 1/z

    assert _eis(1/z).series(z) == \
        z + z**2 + 2*z**3 + 6*z**4 + 24*z**5 + O(z**6)

    assert Ei(z).rewrite('tractable') == exp(z)*_eis(z)
    assert li(z).rewrite('tractable') == z*_eis(log(z))

    assert _eis(z).rewrite('intractable') == exp(-z)*Ei(z)

    assert expand(li(z).rewrite('tractable').diff(z).rewrite('intractable')) \
        == li(z).diff(z)

    assert expand(Ei(z).rewrite('tractable').diff(z).rewrite('intractable')) \
        == Ei(z).diff(z)

    assert _eis(z).series(z, n=3) == EulerGamma + log(z) + z*(-log(z) - \
        EulerGamma + 1) + z**2*(log(z)/2 - Rational(3, 4) + EulerGamma/2)\
        + O(z**3*log(z))
    raises(ArgumentIndexError, lambda: _eis(z).fdiff(2))


def tn_arg(func):
    def test(arg, e1, e2):
        from sympy.core.random import uniform
        v = uniform(1, 5)
        v1 = func(arg*x).subs(x, v).n()
        v2 = func(e1*v + e2*1e-15).n()
        return abs(v1 - v2).n() < 1e-10
    return test(exp_polar(I*pi/2), I, 1) and \
        test(exp_polar(-I*pi/2), -I, 1) and \
        test(exp_polar(I*pi), -1, I) and \
        test(exp_polar(-I*pi), -1, -I)


def test_li():
    z = Symbol("z")
    zr = Symbol("z", real=True)
    zp = Symbol("z", positive=True)
    zn = Symbol("z", negative=True)

    assert li(0) is S.Zero
    assert li(1) is -oo
    assert li(oo) is oo

    assert isinstance(li(z), li)
    assert unchanged(li, -zp)
    assert unchanged(li, zn)

    assert diff(li(z), z) == 1/log(z)

    assert conjugate(li(z)) == li(conjugate(z))
    assert conjugate(li(-zr)) == li(-zr)
    assert unchanged(conjugate, li(-zp))
    assert unchanged(conjugate, li(zn))

    assert li(z).rewrite(Li) == Li(z) + li(2)
    assert li(z).rewrite(Ei) == Ei(log(z))
    assert li(z).rewrite(uppergamma) == (-log(1/log(z))/2 - log(-log(z)) +
                                         log(log(z))/2 - expint(1, -log(z)))
    assert li(z).rewrite(Si) == (-log(I*log(z)) - log(1/log(z))/2 +
                                 log(log(z))/2 + Ci(I*log(z)) + Shi(log(z)))
    assert li(z).rewrite(Ci) == (-log(I*log(z)) - log(1/log(z))/2 +
                                 log(log(z))/2 + Ci(I*log(z)) + Shi(log(z)))
    assert li(z).rewrite(Shi) == (-log(1/log(z))/2 + log(log(z))/2 +
                                  Chi(log(z)) - Shi(log(z)))
    assert li(z).rewrite(Chi) == (-log(1/log(z))/2 + log(log(z))/2 +
                                  Chi(log(z)) - Shi(log(z)))
    assert li(z).rewrite(hyper) ==(log(z)*hyper((1, 1), (2, 2), log(z)) -
                                   log(1/log(z))/2 + log(log(z))/2 + EulerGamma)
    assert li(z).rewrite(meijerg) == (-log(1/log(z))/2 - log(-log(z)) + log(log(z))/2 -
                                      meijerg(((), (1,)), ((0, 0), ()), -log(z)))

    assert gruntz(1/li(z), z, oo) is S.Zero
    assert li(z).series(z) == log(z)**5/600 + log(z)**4/96 + log(z)**3/18 + log(z)**2/4 + \
            log(z) + log(log(z)) + EulerGamma
    raises(ArgumentIndexError, lambda: li(z).fdiff(2))


def test_Li():
    assert Li(2) is S.Zero
    assert Li(oo) is oo

    assert isinstance(Li(z), Li)

    assert diff(Li(z), z) == 1/log(z)

    assert gruntz(1/Li(z), z, oo) is S.Zero
    assert Li(z).rewrite(li) == li(z) - li(2)
    assert Li(z).series(z) == \
        log(z)**5/600 + log(z)**4/96 + log(z)**3/18 + log(z)**2/4 + log(z) + log(log(z)) - li(2) + EulerGamma
    raises(ArgumentIndexError, lambda: Li(z).fdiff(2))


def test_si():
    assert Si(I*x) == I*Shi(x)
    assert Shi(I*x) == I*Si(x)
    assert Si(-I*x) == -I*Shi(x)
    assert Shi(-I*x) == -I*Si(x)
    assert Si(-x) == -Si(x)
    assert Shi(-x) == -Shi(x)
    assert Si(exp_polar(2*pi*I)*x) == Si(x)
    assert Si(exp_polar(-2*pi*I)*x) == Si(x)
    assert Shi(exp_polar(2*pi*I)*x) == Shi(x)
    assert Shi(exp_polar(-2*pi*I)*x) == Shi(x)

    assert Si(oo) == pi/2
    assert Si(-oo) == -pi/2
    assert Shi(oo) is oo
    assert Shi(-oo) is -oo

    assert mytd(Si(x), sin(x)/x, x)
    assert mytd(Shi(x), sinh(x)/x, x)

    assert mytn(Si(x), Si(x).rewrite(Ei),
                -I*(-Ei(x*exp_polar(-I*pi/2))/2
               + Ei(x*exp_polar(I*pi/2))/2 - I*pi) + pi/2, x)
    assert mytn(Si(x), Si(x).rewrite(expint),
                -I*(-expint(1, x*exp_polar(-I*pi/2))/2 +
                    expint(1, x*exp_polar(I*pi/2))/2) + pi/2, x)
    assert mytn(Shi(x), Shi(x).rewrite(Ei),
                Ei(x)/2 - Ei(x*exp_polar(I*pi))/2 + I*pi/2, x)
    assert mytn(Shi(x), Shi(x).rewrite(expint),
                expint(1, x)/2 - expint(1, x*exp_polar(I*pi))/2 - I*pi/2, x)

    assert tn_arg(Si)
    assert tn_arg(Shi)

    assert Si(x)._eval_as_leading_term(x, None, 1) == x
    assert Si(2*x)._eval_as_leading_term(x, None, 1) == 2*x
    assert Si(sin(x))._eval_as_leading_term(x, None, 1) == x
    assert Si(x + 1)._eval_as_leading_term(x, None, 1) == Si(1)
    assert Si(1/x)._eval_as_leading_term(x, None, 1) == \
        Si(1/x)._eval_as_leading_term(x, None, -1) == Si(1/x)

    assert Si(x).nseries(x, n=8) == \
        x - x**3/18 + x**5/600 - x**7/35280 + O(x**8)
    assert Shi(x).nseries(x, n=8) == \
        x + x**3/18 + x**5/600 + x**7/35280 + O(x**8)
    assert Si(sin(x)).nseries(x, n=5) == x - 2*x**3/9 + O(x**5)
    assert Si(x).nseries(x, 1, n=3) == \
        Si(1) + (x - 1)*sin(1) + (x - 1)**2*(-sin(1)/2 + cos(1)/2) + O((x - 1)**3, (x, 1))

    assert Si(x).series(x, oo) == -sin(x)*(-6/x**4 + x**(-2) + O(x**(-6), (x, oo))) - \
        cos(x)*(24/x**5 - 2/x**3 + 1/x + O(x**(-6), (x, oo))) + pi/2

    t = Symbol('t', Dummy=True)
    assert Si(x).rewrite(sinc).dummy_eq(Integral(sinc(t), (t, 0, x)))

    assert limit(Shi(x), x, S.Infinity) == S.Infinity
    assert limit(Shi(x), x, S.NegativeInfinity) == S.NegativeInfinity


def test_ci():
    m1 = exp_polar(I*pi)
    m1_ = exp_polar(-I*pi)
    pI = exp_polar(I*pi/2)
    mI = exp_polar(-I*pi/2)

    assert Ci(m1*x) == Ci(x) + I*pi
    assert Ci(m1_*x) == Ci(x) - I*pi
    assert Ci(pI*x) == Chi(x) + I*pi/2
    assert Ci(mI*x) == Chi(x) - I*pi/2
    assert Chi(m1*x) == Chi(x) + I*pi
    assert Chi(m1_*x) == Chi(x) - I*pi
    assert Chi(pI*x) == Ci(x) + I*pi/2
    assert Chi(mI*x) == Ci(x) - I*pi/2
    assert Ci(exp_polar(2*I*pi)*x) == Ci(x) + 2*I*pi
    assert Chi(exp_polar(-2*I*pi)*x) == Chi(x) - 2*I*pi
    assert Chi(exp_polar(2*I*pi)*x) == Chi(x) + 2*I*pi
    assert Ci(exp_polar(-2*I*pi)*x) == Ci(x) - 2*I*pi

    assert Ci(oo) is S.Zero
    assert Ci(-oo) == I*pi
    assert Chi(oo) is oo
    assert Chi(-oo) is oo

    assert mytd(Ci(x), cos(x)/x, x)
    assert mytd(Chi(x), cosh(x)/x, x)

    assert mytn(Ci(x), Ci(x).rewrite(Ei),
                Ei(x*exp_polar(-I*pi/2))/2 + Ei(x*exp_polar(I*pi/2))/2, x)
    assert mytn(Chi(x), Chi(x).rewrite(Ei),
                Ei(x)/2 + Ei(x*exp_polar(I*pi))/2 - I*pi/2, x)

    assert tn_arg(Ci)
    assert tn_arg(Chi)

    assert Ci(x).nseries(x, n=4) == \
        EulerGamma + log(x) - x**2/4 + O(x**4)
    assert Chi(x).nseries(x, n=4) == \
        EulerGamma + log(x) + x**2/4 + O(x**4)

    assert Ci(x).series(x, oo) == -cos(x)*(-6/x**4 + x**(-2) + O(x**(-6), (x, oo))) + \
        sin(x)*(24/x**5 - 2/x**3 + 1/x + O(x**(-6), (x, oo)))

    assert Ci(x).series(x, -oo) == -cos(x)*(-6/x**4 + x**(-2) + O(x**(-6), (x, -oo))) + \
        sin(x)*(24/x**5 - 2/x**3 + 1/x + O(x**(-6), (x, -oo))) + I*pi

    assert limit(log(x) - Ci(2*x), x, 0) == -log(2) - EulerGamma
    assert Ci(x).rewrite(uppergamma) == -expint(1, x*exp_polar(-I*pi/2))/2 -\
                                        expint(1, x*exp_polar(I*pi/2))/2
    assert Ci(x).rewrite(expint) == -expint(1, x*exp_polar(-I*pi/2))/2 -\
                                        expint(1, x*exp_polar(I*pi/2))/2
    raises(ArgumentIndexError, lambda: Ci(x).fdiff(2))


def test_fresnel():
    assert fresnels(0) is S.Zero
    assert fresnels(oo) is S.Half
    assert fresnels(-oo) == Rational(-1, 2)
    assert fresnels(I*oo) == -I*S.Half

    assert unchanged(fresnels, z)
    assert fresnels(-z) == -fresnels(z)
    assert fresnels(I*z) == -I*fresnels(z)
    assert fresnels(-I*z) == I*fresnels(z)

    assert conjugate(fresnels(z)) == fresnels(conjugate(z))

    assert fresnels(z).diff(z) == sin(pi*z**2/2)

    assert fresnels(z).rewrite(erf) == (S.One + I)/4 * (
        erf((S.One + I)/2*sqrt(pi)*z) - I*erf((S.One - I)/2*sqrt(pi)*z))

    assert fresnels(z).rewrite(hyper) == \
        pi*z**3/6 * hyper([Rational(3, 4)], [Rational(3, 2), Rational(7, 4)], -pi**2*z**4/16)

    assert fresnels(z).series(z, n=15) == \
        pi*z**3/6 - pi**3*z**7/336 + pi**5*z**11/42240 + O(z**15)

    assert fresnels(w).is_extended_real is True
    assert fresnels(w).is_finite is True

    assert fresnels(z).is_extended_real is None
    assert fresnels(z).is_finite is None

    assert fresnels(z).as_real_imag() == (fresnels(re(z) - I*im(z))/2 +
                    fresnels(re(z) + I*im(z))/2,
                    -I*(-fresnels(re(z) - I*im(z)) + fresnels(re(z) + I*im(z)))/2)

    assert fresnels(z).as_real_imag(deep=False) == (fresnels(re(z) - I*im(z))/2 +
                    fresnels(re(z) + I*im(z))/2,
                    -I*(-fresnels(re(z) - I*im(z)) + fresnels(re(z) + I*im(z)))/2)

    assert fresnels(w).as_real_imag() == (fresnels(w), 0)
    assert fresnels(w).as_real_imag(deep=True) == (fresnels(w), 0)

    assert fresnels(2 + 3*I).as_real_imag() == (
            fresnels(2 + 3*I)/2 + fresnels(2 - 3*I)/2,
            -I*(fresnels(2 + 3*I) - fresnels(2 - 3*I))/2
            )

    assert expand_func(integrate(fresnels(z), z)) == \
        z*fresnels(z) + cos(pi*z**2/2)/pi

    assert fresnels(z).rewrite(meijerg) == sqrt(2)*pi*z**Rational(9, 4) * \
        meijerg(((), (1,)), ((Rational(3, 4),),
        (Rational(1, 4), 0)), -pi**2*z**4/16)/(2*(-z)**Rational(3, 4)*(z**2)**Rational(3, 4))

    assert fresnelc(0) is S.Zero
    assert fresnelc(oo) == S.Half
    assert fresnelc(-oo) == Rational(-1, 2)
    assert fresnelc(I*oo) == I*S.Half

    assert unchanged(fresnelc, z)
    assert fresnelc(-z) == -fresnelc(z)
    assert fresnelc(I*z) == I*fresnelc(z)
    assert fresnelc(-I*z) == -I*fresnelc(z)

    assert conjugate(fresnelc(z)) == fresnelc(conjugate(z))

    assert fresnelc(z).diff(z) == cos(pi*z**2/2)

    assert fresnelc(z).rewrite(erf) == (S.One - I)/4 * (
        erf((S.One + I)/2*sqrt(pi)*z) + I*erf((S.One - I)/2*sqrt(pi)*z))

    assert fresnelc(z).rewrite(hyper) == \
        z * hyper([Rational(1, 4)], [S.Half, Rational(5, 4)], -pi**2*z**4/16)

    assert fresnelc(w).is_extended_real is True

    assert fresnelc(z).as_real_imag() == \
        (fresnelc(re(z) - I*im(z))/2 + fresnelc(re(z) + I*im(z))/2,
         -I*(-fresnelc(re(z) - I*im(z)) + fresnelc(re(z) + I*im(z)))/2)

    assert fresnelc(z).as_real_imag(deep=False) == \
        (fresnelc(re(z) - I*im(z))/2 + fresnelc(re(z) + I*im(z))/2,
         -I*(-fresnelc(re(z) - I*im(z)) + fresnelc(re(z) + I*im(z)))/2)

    assert fresnelc(2 + 3*I).as_real_imag() == (
        fresnelc(2 - 3*I)/2 + fresnelc(2 + 3*I)/2,
         -I*(fresnelc(2 + 3*I) - fresnelc(2 - 3*I))/2
    )

    assert expand_func(integrate(fresnelc(z), z)) == \
        z*fresnelc(z) - sin(pi*z**2/2)/pi

    assert fresnelc(z).rewrite(meijerg) == sqrt(2)*pi*z**Rational(3, 4) * \
        meijerg(((), (1,)), ((Rational(1, 4),),
        (Rational(3, 4), 0)), -pi**2*z**4/16)/(2*(-z)**Rational(1, 4)*(z**2)**Rational(1, 4))

    from sympy.core.random import verify_numerically

    verify_numerically(re(fresnels(z)), fresnels(z).as_real_imag()[0], z)
    verify_numerically(im(fresnels(z)), fresnels(z).as_real_imag()[1], z)
    verify_numerically(fresnels(z), fresnels(z).rewrite(hyper), z)
    verify_numerically(fresnels(z), fresnels(z).rewrite(meijerg), z)

    verify_numerically(re(fresnelc(z)), fresnelc(z).as_real_imag()[0], z)
    verify_numerically(im(fresnelc(z)), fresnelc(z).as_real_imag()[1], z)
    verify_numerically(fresnelc(z), fresnelc(z).rewrite(hyper), z)
    verify_numerically(fresnelc(z), fresnelc(z).rewrite(meijerg), z)

    raises(ArgumentIndexError, lambda: fresnels(z).fdiff(2))
    raises(ArgumentIndexError, lambda: fresnelc(z).fdiff(2))

    assert fresnels(x).taylor_term(-1, x) is S.Zero
    assert fresnelc(x).taylor_term(-1, x) is S.Zero
    assert fresnelc(x).taylor_term(1, x) == -pi**2*x**5/40


def test_fresnel_series():
    assert fresnelc(z).series(z, n=15) == \
        z - pi**2*z**5/40 + pi**4*z**9/3456 - pi**6*z**13/599040 + O(z**15)

    # issues 6510, 10102
    fs = (S.Half - sin(pi*z**2/2)/(pi**2*z**3)
        + (-1/(pi*z) + 3/(pi**3*z**5))*cos(pi*z**2/2))
    fc = (S.Half - cos(pi*z**2/2)/(pi**2*z**3)
        + (1/(pi*z) - 3/(pi**3*z**5))*sin(pi*z**2/2))
    assert fresnels(z).series(z, oo) == fs + O(z**(-6), (z, oo))
    assert fresnelc(z).series(z, oo) == fc + O(z**(-6), (z, oo))
    assert (fresnels(z).series(z, -oo) + fs.subs(z, -z)).expand().is_Order
    assert (fresnelc(z).series(z, -oo) + fc.subs(z, -z)).expand().is_Order
    assert (fresnels(1/z).series(z) - fs.subs(z, 1/z)).expand().is_Order
    assert (fresnelc(1/z).series(z) - fc.subs(z, 1/z)).expand().is_Order
    assert ((2*fresnels(3*z)).series(z, oo) - 2*fs.subs(z, 3*z)).expand().is_Order
    assert ((3*fresnelc(2*z)).series(z, oo) - 3*fc.subs(z, 2*z)).expand().is_Order


def test_integral_rewrites(): #issues 26134, 26144, 26306
    assert expint(n, x).rewrite(Integral).dummy_eq(Integral(t**-n * exp(-t*x), (t, 1, oo)))
    assert Si(x).rewrite(Integral).dummy_eq(Integral(sinc(t), (t, 0, x)))
    assert Ci(x).rewrite(Integral).dummy_eq(log(x) - Integral((1 - cos(t))/t, (t, 0, x)) + EulerGamma)
    assert fresnels(x).rewrite(Integral).dummy_eq(Integral(sin(pi*t**2/2), (t, 0, x)))
    assert fresnelc(x).rewrite(Integral).dummy_eq(Integral(cos(pi*t**2/2), (t, 0, x)))
    assert Ei(x).rewrite(Integral).dummy_eq(Integral(exp(t)/t, (t, -oo, x)))
    assert fresnels(x).diff(x) == fresnels(x).rewrite(Integral).diff(x)
    assert fresnelc(x).diff(x) == fresnelc(x).rewrite(Integral).diff(x)
