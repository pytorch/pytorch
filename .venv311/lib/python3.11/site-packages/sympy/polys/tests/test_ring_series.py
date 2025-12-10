from sympy.polys.domains import ZZ, QQ, EX, RR
from sympy.polys.rings import ring
from sympy.polys.puiseux import puiseux_ring
from sympy.polys.ring_series import (_invert_monoms, rs_integrate,
    rs_trunc, rs_mul, rs_square, rs_pow, _has_constant_term, rs_hadamard_exp,
    rs_series_from_list, rs_exp, rs_log, rs_newton, rs_series_inversion,
    rs_compose_add, rs_asin, _atan, rs_atan, _atanh, rs_atanh, rs_asinh, rs_tan,
    rs_cot, rs_sin, rs_cos, rs_cos_sin, rs_sinh, rs_cosh, rs_cosh_sinh, rs_tanh,
    _tan1, rs_fun, rs_nth_root, rs_LambertW, rs_series_reversion, rs_is_puiseux,
    rs_series)
from sympy.testing.pytest import raises, slow
from sympy.core.symbol import symbols
from sympy.functions import (sin, cos, exp, tan, cot, sinh, cosh, atan, atanh,
    asinh, tanh, log, sqrt)
from sympy.core.numbers import Rational, pi
from sympy.core import expand, S

def is_close(a, b):
    tol = 10**(-10)
    assert abs(a - b) < tol


def test_ring_series1():
    R, x = ring('x', QQ)
    p = x**4 + 2*x**3 + 3*x + 4
    assert _invert_monoms(p) == 4*x**4 + 3*x**3 + 2*x + 1
    assert rs_hadamard_exp(p) == x**4/24 + x**3/3 + 3*x + 4
    R, x = ring('x', QQ)
    p = x**4 + 2*x**3 + 3*x + 4
    assert rs_integrate(p, x) == x**5/5 + x**4/2 + 3*x**2/2 + 4*x
    R, x, y = ring('x, y', QQ)
    p = x**2*y**2 + x + 1
    assert rs_integrate(p, x) == x**3*y**2/3 + x**2/2 + x
    assert rs_integrate(p, y) == x**2*y**3/3 + x*y + y


def test_trunc():
    R, x, y, t = ring('x, y, t', QQ)
    p = (y + t*x)**4
    p1 = rs_trunc(p, x, 3)
    assert p1 == y**4 + 4*y**3*t*x + 6*y**2*t**2*x**2


def test_mul_trunc():
    R, x, y, t = ring('x, y, t', QQ)
    p = 1 + t*x + t*y
    for i in range(2):
        p = rs_mul(p, p, t, 3)

    assert p == 6*x**2*t**2 + 12*x*y*t**2 + 6*y**2*t**2 + 4*x*t + 4*y*t + 1
    p = 1 + t*x + t*y + t**2*x*y
    p1 = rs_mul(p, p, t, 2)
    assert p1 == 1 + 2*t*x + 2*t*y
    R1, z = ring('z', QQ)
    raises(ValueError, lambda: rs_mul(p, z, x, 2))

    p1 = 2 + 2*x + 3*x**2
    p2 = 3 + x**2
    assert rs_mul(p1, p2, x, 4) == 2*x**3 + 11*x**2 + 6*x + 6


def test_square_trunc():
    R, x, y, t = ring('x, y, t', QQ)
    p = (1 + t*x + t*y)*2
    p1 = rs_mul(p, p, x, 3)
    p2 = rs_square(p, x, 3)
    assert p1 == p2
    p = 1 + x + x**2 + x**3
    assert rs_square(p, x, 4) == 4*x**3 + 3*x**2 + 2*x + 1


def test_pow_trunc():
    R, x, y, z = ring('x, y, z', QQ)
    p0 = y + x*z
    p = p0**16
    for xx in (x, y, z):
        p1 = rs_trunc(p, xx, 8)
        p2 = rs_pow(p0, 16, xx, 8)
        assert p1 == p2

    p = 1 + x
    p1 = rs_pow(p, 3, x, 2)
    assert p1 == 1 + 3*x
    assert rs_pow(p, 0, x, 2) == 1
    assert rs_pow(p, -2, x, 2) == 1 - 2*x
    p = x + y
    assert rs_pow(p, 3, y, 3) == x**3 + 3*x**2*y + 3*x*y**2
    assert rs_pow(1 + x, Rational(2, 3), x, 4) == 4*x**3/81 - x**2/9 + x*Rational(2, 3) + 1


def test_has_constant_term():
    R, x, y, z = ring('x, y, z', QQ)
    p = y + x*z
    assert _has_constant_term(p, x)
    p = x + x**4
    assert not _has_constant_term(p, x)
    p = 1 + x + x**4
    assert _has_constant_term(p, x)
    p = x + y + x*z


def test_inversion():
    R, x = ring('x', QQ)
    p = 2 + x + 2*x**2
    n = 5
    p1 = rs_series_inversion(p, x, n)
    assert rs_trunc(p*p1, x, n) == 1
    R, x, y = ring('x, y', QQ)
    p = 2 + x + 2*x**2 + y*x + x**2*y
    p1 = rs_series_inversion(p, x, n)
    assert rs_trunc(p*p1, x, n) == 1

    R, x, y = ring('x, y', QQ)
    p = 1 + x + y
    raises(NotImplementedError, lambda: rs_series_inversion(p, x, 4))
    p = R.zero
    raises(ZeroDivisionError, lambda: rs_series_inversion(p, x, 3))

    R, x = ring('x', ZZ)
    p = 2 + x
    raises(ValueError, lambda: rs_series_inversion(p, x, 3))


def test_series_reversion():
    R, x, y = ring('x, y', QQ)

    p = rs_tan(x, x, 10)
    assert rs_series_reversion(p, x, 8, y) == rs_atan(y, y, 8)

    p = rs_sin(x, x, 10)
    assert rs_series_reversion(p, x, 8, y) == 5*y**7/112 + 3*y**5/40 + \
        y**3/6 + y


def test_series_from_list():
    R, x = ring('x', QQ)
    p = 1 + 2*x + x**2 + 3*x**3
    c = [1, 2, 0, 4, 4]
    r = rs_series_from_list(p, c, x, 5)
    pc = R.from_list(list(reversed(c)))
    r1 = rs_trunc(pc.compose(x, p), x, 5)
    assert r == r1
    R, x, y = ring('x, y', QQ)
    c = [1, 3, 5, 7]
    p1 = rs_series_from_list(x + y, c, x, 3, concur=0)
    p2 = rs_trunc((1 + 3*(x+y) + 5*(x+y)**2 + 7*(x+y)**3), x, 3)
    assert p1 == p2

    R, x = ring('x', QQ)
    h = 25
    p = rs_exp(x, x, h) - 1
    p1 = rs_series_from_list(p, c, x, h)
    p2 = 0
    for i, cx in enumerate(c):
        p2 += cx*rs_pow(p, i, x, h)
    assert p1 == p2


def test_log():
    R, x = ring('x', QQ)
    p = 1 + x
    assert rs_log(p, x, 4) == x - x**2/2 + x**3/3
    p = 1 + x +2*x**2/3
    p1 = rs_log(p, x, 9)
    assert p1 == -17*x**8/648 + 13*x**7/189 - 11*x**6/162 - x**5/45 + \
      7*x**4/36 - x**3/3 + x**2/6 + x
    p2 = rs_series_inversion(p, x, 9)
    p3 = rs_log(p2, x, 9)
    assert p3 == -p1

    R, x, y = ring('x, y', QQ)
    p = 1 + x + 2*y*x**2
    p1 = rs_log(p, x, 6)
    assert p1 == (4*x**5*y**2 - 2*x**5*y - 2*x**4*y**2 + x**5/5 + 2*x**4*y -
                  x**4/4 - 2*x**3*y + x**3/3 + 2*x**2*y - x**2/2 + x)

    # Constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', EX)
    assert rs_log(x + a, x, 5) == -EX(1/(4*a**4))*x**4 + EX(1/(3*a**3))*x**3 \
        - EX(1/(2*a**2))*x**2 + EX(1/a)*x + EX(log(a))
    assert rs_log(x + x**2*y + a, x, 4) == -EX(a**(-2))*x**3*y + \
        EX(1/(3*a**3))*x**3 + EX(1/a)*x**2*y - EX(1/(2*a**2))*x**2 + \
        EX(1/a)*x + EX(log(a))

    p = x + x**2 + 3
    assert rs_log(p, x, 10).compose(x, 5) == EX(log(3) + Rational(19281291595, 9920232))


def test_exp():
    R, x = ring('x', QQ)
    p = x + x**4
    for h in [10, 30]:
        q = rs_series_inversion(1 + p, x, h) - 1
        p1 = rs_exp(q, x, h)
        q1 = rs_log(p1, x, h)
        assert q1 == q
    p1 = rs_exp(p, x, 30)
    assert p1.coeff(x**29) == QQ(74274246775059676726972369, 353670479749588078181744640000)
    prec = 21
    p = rs_log(1 + x, x, prec)
    p1 = rs_exp(p, x, prec)
    assert p1 == x + 1

    # Constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', QQ[exp(a), a])
    assert rs_exp(x + a, x, 5) == exp(a)*x**4/24 + exp(a)*x**3/6 + \
        exp(a)*x**2/2 + exp(a)*x + exp(a)
    assert rs_exp(x + x**2*y + a, x, 5) == exp(a)*x**4*y**2/2 + \
            exp(a)*x**4*y/2 + exp(a)*x**4/24 + exp(a)*x**3*y + \
            exp(a)*x**3/6 + exp(a)*x**2*y + exp(a)*x**2/2 + exp(a)*x + exp(a)

    R, x, y = ring('x, y', EX)
    assert rs_exp(x + a, x, 5) ==  EX(exp(a)/24)*x**4 + EX(exp(a)/6)*x**3 + \
        EX(exp(a)/2)*x**2 + EX(exp(a))*x + EX(exp(a))
    assert rs_exp(x + x**2*y + a, x, 5) == EX(exp(a)/2)*x**4*y**2 + \
        EX(exp(a)/2)*x**4*y + EX(exp(a)/24)*x**4 + EX(exp(a))*x**3*y + \
        EX(exp(a)/6)*x**3 + EX(exp(a))*x**2*y + EX(exp(a)/2)*x**2 + \
        EX(exp(a))*x + EX(exp(a))


def test_newton():
    R, x = ring('x', QQ)
    p = x**2 - 2
    r = rs_newton(p, x, 4)
    assert r == 8*x**4 + 4*x**2 + 2


def test_compose_add():
    R, x = ring('x', QQ)
    p1 = x**3 - 1
    p2 = x**2 - 2
    assert rs_compose_add(p1, p2) == x**6 - 6*x**4 - 2*x**3 + 12*x**2 - 12*x - 7


def test_fun():
    R, x, y = ring('x, y', QQ)
    p = x*y + x**2*y**3 + x**5*y
    assert rs_fun(p, rs_tan, x, 10) == rs_tan(p, x, 10)
    assert rs_fun(p, _tan1, x, 10) == _tan1(p, x, 10)


def test_nth_root():
    R, x, y = puiseux_ring('x, y', QQ)
    assert rs_nth_root(1 + x**2*y, 4, x, 10) == -77*x**8*y**4/2048 + \
        7*x**6*y**3/128 - 3*x**4*y**2/32 + x**2*y/4 + 1
    assert rs_nth_root(1 + x*y + x**2*y**3, 3, x, 5) == -x**4*y**6/9 + \
        5*x**4*y**5/27 - 10*x**4*y**4/243 - 2*x**3*y**4/9 + 5*x**3*y**3/81 + \
        x**2*y**3/3 - x**2*y**2/9 + x*y/3 + 1
    assert rs_nth_root(8*x, 3, x, 3) == 2*x**QQ(1, 3)
    assert rs_nth_root(8*x + x**2 + x**3, 3, x, 3) == x**QQ(4,3)/12 + 2*x**QQ(1,3)
    r = rs_nth_root(8*x + x**2*y + x**3, 3, x, 4)
    assert r == -x**QQ(7,3)*y**2/288 + x**QQ(7,3)/12 + x**QQ(4,3)*y/12 + 2*x**QQ(1,3)

    # Constant term in series
    a = symbols('a')
    R, x, y = puiseux_ring('x, y', EX)
    assert rs_nth_root(x + EX(a), 3, x, 4) == EX(5/(81*a**QQ(8, 3)))*x**3 - \
        EX(1/(9*a**QQ(5, 3)))*x**2 + EX(1/(3*a**QQ(2, 3)))*x + EX(a**QQ(1, 3))
    assert rs_nth_root(x**QQ(2, 3) + x**2*y + 5, 2, x, 3) == -EX(sqrt(5)/100)*\
        x**QQ(8, 3)*y - EX(sqrt(5)/16000)*x**QQ(8, 3) + EX(sqrt(5)/10)*x**2*y + \
        EX(sqrt(5)/2000)*x**2 - EX(sqrt(5)/200)*x**QQ(4, 3) + \
        EX(sqrt(5)/10)*x**QQ(2, 3) + EX(sqrt(5))


def test_atan():
    R, x, y = ring('x, y', QQ)
    assert rs_atan(x, x, 9) == -x**7/7 + x**5/5 - x**3/3 + x
    assert rs_atan(x*y + x**2*y**3, x, 9) == 2*x**8*y**11 - x**8*y**9 + \
        2*x**7*y**9 - x**7*y**7/7 - x**6*y**9/3 + x**6*y**7 - x**5*y**7 + \
        x**5*y**5/5 - x**4*y**5 - x**3*y**3/3 + x**2*y**3 + x*y

    # Constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', EX)
    assert rs_atan(x + a, x, 5) == -EX((a**3 - a)/(a**8 + 4*a**6 + 6*a**4 + \
        4*a**2 + 1))*x**4 + EX((3*a**2 - 1)/(3*a**6 + 9*a**4 + \
        9*a**2 + 3))*x**3 - EX(a/(a**4 + 2*a**2 + 1))*x**2 + \
        EX(1/(a**2 + 1))*x + EX(atan(a))
    assert rs_atan(x + x**2*y + a, x, 4) == -EX(2*a/(a**4 + 2*a**2 + 1)) \
        *x**3*y + EX((3*a**2 - 1)/(3*a**6 + 9*a**4 + 9*a**2 + 3))*x**3 + \
        EX(1/(a**2 + 1))*x**2*y - EX(a/(a**4 + 2*a**2 + 1))*x**2 + EX(1/(a**2 \
        + 1))*x + EX(atan(a))

    # Test for _atan faster for small and univariate series
    R, x = ring('x', QQ)
    p = x**2 + 2*x
    assert _atan(p, x, 5) == rs_atan(p, x, 5)

    R, x = ring('x', EX)
    p = x**2 + 2*x
    assert _atan(p, x, 9) == rs_atan(p, x, 9)


def test_asin():
    R, x, y = ring('x, y', QQ)
    assert rs_asin(x + x*y, x, 5) == x**3*y**3/6 + x**3*y**2/2 + x**3*y/2 + \
        x**3/6 + x*y + x
    assert rs_asin(x*y + x**2*y**3, x, 6) == x**5*y**7/2 + 3*x**5*y**5/40 + \
        x**4*y**5/2 + x**3*y**3/6 + x**2*y**3 + x*y


def test_tan():
    R, x, y = ring('x, y', QQ)
    assert rs_tan(x, x, 9) == x + x**3/3 + QQ(2,15)*x**5 + QQ(17,315)*x**7
    assert rs_tan(x*y + x**2*y**3, x, 9) == 4*x**8*y**11/3 + 17*x**8*y**9/45 + \
        4*x**7*y**9/3 + 17*x**7*y**7/315 + x**6*y**9/3 + 2*x**6*y**7/3 + \
        x**5*y**7 + 2*x**5*y**5/15 + x**4*y**5 + x**3*y**3/3 + x**2*y**3 + x*y

    # Constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', QQ[tan(a), a])
    assert rs_tan(x + a, x, 5) == (tan(a)**5 + 5*tan(a)**3/3 +
        2*tan(a)/3)*x**4 + (tan(a)**4 + 4*tan(a)**2/3 + Rational(1, 3))*x**3 + \
        (tan(a)**3 + tan(a))*x**2 + (tan(a)**2 + 1)*x + tan(a)
    assert rs_tan(x + x**2*y + a, x, 4) == (2*tan(a)**3 + 2*tan(a))*x**3*y + \
        (tan(a)**4 + Rational(4, 3)*tan(a)**2 + Rational(1, 3))*x**3 + (tan(a)**2 + 1)*x**2*y + \
        (tan(a)**3 + tan(a))*x**2 + (tan(a)**2 + 1)*x + tan(a)

    R, x, y = ring('x, y', EX)
    assert rs_tan(x + a, x, 5) == EX(tan(a)**5 + 5*tan(a)**3/3 +
        2*tan(a)/3)*x**4 + EX(tan(a)**4 + 4*tan(a)**2/3 + EX(1)/3)*x**3 + \
        EX(tan(a)**3 + tan(a))*x**2 + EX(tan(a)**2 + 1)*x + EX(tan(a))
    assert rs_tan(x + x**2*y + a, x, 4) == EX(2*tan(a)**3 +
        2*tan(a))*x**3*y + EX(tan(a)**4 + 4*tan(a)**2/3 + EX(1)/3)*x**3 + \
        EX(tan(a)**2 + 1)*x**2*y + EX(tan(a)**3 + tan(a))*x**2 + \
        EX(tan(a)**2 + 1)*x + EX(tan(a))

    p = x + x**2 + 5
    assert rs_atan(p, x, 10).compose(x, 10) == EX(atan(5) + S(67701870330562640) / \
        668083460499)


def test_cot():
    R, x, y = puiseux_ring('x, y', QQ)
    assert rs_cot(x**6 + x**7, x, 8) == x**(-6) - x**(-5) + x**(-4) - \
        x**(-3) + x**(-2) - x**(-1) + 1 - x + x**2 - x**3 + x**4 - x**5 + \
        2*x**6/3 - 4*x**7/3
    assert rs_cot(x + x**2*y, x, 5) == -x**4*y**5 - x**4*y/15 + x**3*y**4 - \
        x**3/45 - x**2*y**3 - x**2*y/3 + x*y**2 - x/3 - y + x**(-1)


def test_sin():
    R, x, y = ring('x, y', QQ)
    assert rs_sin(x, x, 9) == x - x**3/6 + x**5/120 - x**7/5040
    assert rs_sin(x*y + x**2*y**3, x, 9) == x**8*y**11/12 - \
        x**8*y**9/720 + x**7*y**9/12 - x**7*y**7/5040 - x**6*y**9/6 + \
        x**6*y**7/24 - x**5*y**7/2 + x**5*y**5/120 - x**4*y**5/2 - \
        x**3*y**3/6 + x**2*y**3 + x*y

    # Constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', QQ[sin(a), cos(a), a])
    assert rs_sin(x + a, x, 5) == sin(a)*x**4/24 - cos(a)*x**3/6 - \
        sin(a)*x**2/2 + cos(a)*x + sin(a)
    assert rs_sin(x + x**2*y + a, x, 5) == -sin(a)*x**4*y**2/2 - \
        cos(a)*x**4*y/2 + sin(a)*x**4/24 - sin(a)*x**3*y - cos(a)*x**3/6 + \
        cos(a)*x**2*y - sin(a)*x**2/2 + cos(a)*x + sin(a)

    R, x, y = ring('x, y', EX)
    assert rs_sin(x + a, x, 5) == EX(sin(a)/24)*x**4 - EX(cos(a)/6)*x**3 - \
        EX(sin(a)/2)*x**2 + EX(cos(a))*x + EX(sin(a))
    assert rs_sin(x + x**2*y + a, x, 5) == -EX(sin(a)/2)*x**4*y**2 - \
        EX(cos(a)/2)*x**4*y + EX(sin(a)/24)*x**4 - EX(sin(a))*x**3*y - \
        EX(cos(a)/6)*x**3 + EX(cos(a))*x**2*y - EX(sin(a)/2)*x**2 + \
        EX(cos(a))*x + EX(sin(a))


def test_cos():
    R, x, y = ring('x, y', QQ)
    assert rs_cos(x, x, 9) == 1 - x**2/2 + x**4/24 - x**6/720 + x**8/40320
    assert rs_cos(x*y + x**2*y**3, x, 9) == x**8*y**12/24 - \
        x**8*y**10/48 + x**8*y**8/40320 + x**7*y**10/6 - \
        x**7*y**8/120 + x**6*y**8/4 - x**6*y**6/720 + x**5*y**6/6 - \
        x**4*y**6/2 + x**4*y**4/24 - x**3*y**4 - x**2*y**2/2 + 1

    # Constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', QQ[sin(a), cos(a), a])
    assert rs_cos(x + a, x, 5) == cos(a)*x**4/24 + sin(a)*x**3/6 - \
        cos(a)*x**2/2 - sin(a)*x + cos(a)
    assert rs_cos(x + x**2*y + a, x, 5) == -cos(a)*x**4*y**2/2 + \
        sin(a)*x**4*y/2 + cos(a)*x**4/24 - cos(a)*x**3*y + sin(a)*x**3/6 - \
        sin(a)*x**2*y - cos(a)*x**2/2 - sin(a)*x + cos(a)

    R, x, y = ring('x, y', EX)
    assert rs_cos(x + a, x, 5) == EX(cos(a)/24)*x**4 + EX(sin(a)/6)*x**3 - \
        EX(cos(a)/2)*x**2 - EX(sin(a))*x + EX(cos(a))
    assert rs_cos(x + x**2*y + a, x, 5) == -EX(cos(a)/2)*x**4*y**2 + \
        EX(sin(a)/2)*x**4*y + EX(cos(a)/24)*x**4 - EX(cos(a))*x**3*y + \
        EX(sin(a)/6)*x**3 - EX(sin(a))*x**2*y - EX(cos(a)/2)*x**2 - \
        EX(sin(a))*x + EX(cos(a))


def test_cos_sin():
    R, x, y = ring('x, y', QQ)
    c, s = rs_cos_sin(x, x, 9)
    assert c == rs_cos(x, x, 9)
    assert s == rs_sin(x, x, 9)
    c, s = rs_cos_sin(x + x*y, x, 5)
    assert c == rs_cos(x + x*y, x, 5)
    assert s == rs_sin(x + x*y, x, 5)

    # constant term in series
    c, s = rs_cos_sin(1 + x + x**2, x, 5)
    assert c == rs_cos(1 + x + x**2, x, 5)
    assert s == rs_sin(1 + x + x**2, x, 5)

    a = symbols('a')
    R, x, y = ring('x, y', QQ[sin(a), cos(a), a])
    c, s = rs_cos_sin(x + a, x, 5)
    assert c == rs_cos(x + a, x, 5)
    assert s == rs_sin(x + a, x, 5)

    R, x, y = ring('x, y', EX)
    c, s = rs_cos_sin(x + a, x, 5)
    assert c == rs_cos(x + a, x, 5)
    assert s == rs_sin(x + a, x, 5)


def test_atanh():
    R, x, y = ring('x, y', QQ)
    assert rs_atanh(x, x, 9) == x + x**3/3 + x**5/5 + x**7/7
    assert rs_atanh(x*y + x**2*y**3, x, 9) == 2*x**8*y**11 + x**8*y**9 + \
        2*x**7*y**9 + x**7*y**7/7 + x**6*y**9/3 + x**6*y**7 + x**5*y**7 + \
        x**5*y**5/5 + x**4*y**5 + x**3*y**3/3 + x**2*y**3 + x*y

    # Constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', EX)
    assert rs_atanh(x + a, x, 5) == EX((a**3 + a)/(a**8 - 4*a**6 + 6*a**4 - \
        4*a**2 + 1))*x**4 - EX((3*a**2 + 1)/(3*a**6 - 9*a**4 + \
        9*a**2 - 3))*x**3 + EX(a/(a**4 - 2*a**2 + 1))*x**2 - EX(1/(a**2 - \
        1))*x + EX(atanh(a))
    assert rs_atanh(x + x**2*y + a, x, 4) == EX(2*a/(a**4 - 2*a**2 + \
        1))*x**3*y - EX((3*a**2 + 1)/(3*a**6 - 9*a**4 + 9*a**2 - 3))*x**3 - \
        EX(1/(a**2 - 1))*x**2*y + EX(a/(a**4 - 2*a**2 + 1))*x**2 - \
        EX(1/(a**2 - 1))*x + EX(atanh(a))

    p = x + x**2 + 5
    assert rs_atanh(p, x, 10).compose(x, 10) == EX(Rational(-733442653682135, 5079158784) \
        + atanh(5))

    # Test for _atanh faster for small and univariate series
    R,x  = ring('x', QQ)
    p = x**2 + 2*x
    assert _atanh(p, x, 5) == rs_atanh(p, x, 5)

    R,x = ring('x', EX)
    p = x**2 + 2*x
    assert _atanh(p, x, 9) == rs_atanh(p, x, 9)


def test_asinh():
    R, x, y = ring('x, y', QQ)
    assert rs_asinh(x, x, 9) == -5/112*x**7 + 3/40*x**5 - 1/6*x**3 + x
    assert rs_asinh(x*y + x**2*y**3, x, 9) == 3/4*x**8*y**11 - 5/16*x**8*y**9 + \
        3/4*x**7*y**9 - 5/112*x**7*y**7 - 1/6*x**6*y**9 + 3/8*x**6*y**7 - 1/2*x \
        **5*y**7 + 3/40*x**5*y**5 - 1/2*x**4*y**5 - 1/6*x**3*y**3 + x**2*y**3 + x*y

    # Constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', EX)
    assert rs_asinh(x + a, x, 3) == -EX(a/(2*a**2*sqrt(a**2 + 1) + 2*sqrt(a**2 + 1))) \
        *x**2 + EX(1/sqrt(a**2 + 1))*x + EX(asinh(a))
    assert rs_asinh(x + x**2*y + a, x, 3) == EX(1/sqrt(a**2 + 1))*x**2*y - EX(a/(2*a**2 \
        *sqrt(a**2 + 1) + 2*sqrt(a**2 + 1)))*x**2 + EX(1/sqrt(a**2 + 1))*x + EX(asinh(a))

    p = x + x ** 2 + 5
    assert rs_asinh(p, x, 10).compose(x, 10) == EX(asinh(5) + 4643789843094995*sqrt(26)/\
        205564141692)


def test_sinh():
    R, x, y = ring('x, y', QQ)
    assert rs_sinh(x, x, 9) == x + x**3/6 + x**5/120 + x**7/5040
    assert rs_sinh(x*y + x**2*y**3, x, 9) == x**8*y**11/12 + \
        x**8*y**9/720 + x**7*y**9/12 + x**7*y**7/5040 + x**6*y**9/6 + \
        x**6*y**7/24 + x**5*y**7/2 + x**5*y**5/120 + x**4*y**5/2 + \
        x**3*y**3/6 + x**2*y**3 + x*y

    # constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', QQ[sinh(a), cosh(a), a])
    assert rs_sinh(x + a, x, 5) == 1/24*x**4*(sinh(a)) + 1/6*x**3*(cosh(a)) + 1/\
        2*x**2*(sinh(a)) + x*(cosh(a)) + (sinh(a))
    assert rs_sinh(x + x**2*y + a, x, 5) == 1/2*(sinh(a))*x**4*y**2 + 1/2*(cosh(a))\
        *x**4*y + 1/24*(sinh(a))*x**4 + (sinh(a))*x**3*y + 1/6*(cosh(a))*x**3 + \
        (cosh(a))*x**2*y + 1/2*(sinh(a))*x**2 + (cosh(a))*x + (sinh(a))

    R, x, y = ring('x, y', EX)
    assert rs_sinh(x + a, x, 5) == EX(sinh(a)/24)*x**4 + EX(cosh(a)/6)*x**3 + \
        EX(sinh(a)/2)*x**2 + EX(cosh(a))*x + EX(sinh(a))
    assert rs_sinh(x + x**2*y + a, x, 5) == EX(sinh(a)/2)*x**4*y**2 + EX(cosh(a)/\
        2)*x**4*y + EX(sinh(a)/24)*x**4 + EX(sinh(a))*x**3*y + EX(cosh(a)/6)*x**3 \
        + EX(cosh(a))*x**2*y + EX(sinh(a)/2)*x**2 + EX(cosh(a))*x + EX(sinh(a))


def test_cosh():
    R, x, y = ring('x, y', QQ)
    assert rs_cosh(x, x, 9) == 1 + x**2/2 + x**4/24 + x**6/720 + x**8/40320
    assert rs_cosh(x*y + x**2*y**3, x, 9) == x**8*y**12/24 + \
        x**8*y**10/48 + x**8*y**8/40320 + x**7*y**10/6 + \
        x**7*y**8/120 + x**6*y**8/4 + x**6*y**6/720 + x**5*y**6/6 + \
        x**4*y**6/2 + x**4*y**4/24 + x**3*y**4 + x**2*y**2/2 + 1

    # constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', QQ[sinh(a), cosh(a), a])
    assert rs_cosh(x + a, x, 5) == 1/24*(cosh(a))*x**4 + 1/6*(sinh(a))*x**3 + \
        1/2*(cosh(a))*x**2 + (sinh(a))*x + (cosh(a))
    assert rs_cosh(x + x**2*y + a, x, 5) == 1/2*(cosh(a))*x**4*y**2 + 1/2*(sinh(a))\
        *x**4*y + 1/24*(cosh(a))*x**4 + (cosh(a))*x**3*y + 1/6*(sinh(a))*x**3 + \
        (sinh(a))*x**2*y + 1/2*(cosh(a))*x**2 + (sinh(a))*x + (cosh(a))
    R, x, y = ring('x, y', EX)
    assert rs_cosh(x + a, x, 5) == EX(cosh(a)/24)*x**4 + EX(sinh(a)/6)*x**3 + \
        EX(cosh(a)/2)*x**2 + EX(sinh(a))*x + EX(cosh(a))
    assert rs_cosh(x + x**2*y + a, x, 5) == EX(cosh(a)/2)*x**4*y**2 + EX(sinh(a)/\
        2)*x**4*y + EX(cosh(a)/24)*x**4 + EX(cosh(a))*x**3*y + EX(sinh(a)/6)*x**3 \
        + EX(sinh(a))*x**2*y + EX(cosh(a)/2)*x**2 + EX(sinh(a))*x + EX(cosh(a))


def test_cosh_sinh():
    R, x, y = ring('x, y', QQ)
    ch, sh = rs_cosh_sinh(x, x, 9)
    assert ch == rs_cosh(x, x, 9)
    assert sh == rs_sinh(x, x, 9)
    ch, sh = rs_cosh_sinh(x + x*y, x, 5)
    assert ch == rs_cosh(x + x*y, x, 5)
    assert sh == rs_sinh(x + x*y, x, 5)

    # constant term in series
    c, s = rs_cosh_sinh(1 + x + x**2, x, 5)
    assert c == rs_cosh(1 + x + x**2, x, 5)
    assert s == rs_sinh(1 + x + x**2, x, 5)

    a = symbols('a')
    R, x, y = ring('x, y', QQ[sinh(a), cosh(a), a])
    ch, sh = rs_cosh_sinh(x + a, x, 5)
    assert ch == rs_cosh(x + a, x, 5)
    assert sh == rs_sinh(x + a, x, 5)
    R, x, y = ring('x, y', EX)
    ch, sh = rs_cosh_sinh(x + a, x, 5)
    assert ch == rs_cosh(x + a, x, 5)
    assert sh == rs_sinh(x + a, x, 5)


def test_tanh():
    R, x, y = ring('x, y', QQ)
    assert rs_tanh(x, x, 9) == x - QQ(1,3)*x**3 + QQ(2,15)*x**5 - QQ(17,315)*x**7
    assert rs_tanh(x*y + x**2*y**3, x, 9) == 4*x**8*y**11/3 - \
        17*x**8*y**9/45 + 4*x**7*y**9/3 - 17*x**7*y**7/315 - x**6*y**9/3 + \
        2*x**6*y**7/3 - x**5*y**7 + 2*x**5*y**5/15 - x**4*y**5 - \
        x**3*y**3/3 + x**2*y**3 + x*y

    # Constant term in series
    a = symbols('a')
    R, x, y = ring('x, y', EX)
    assert rs_tanh(x + a, x, 5) == EX(tanh(a)**5 - 5*tanh(a)**3/3 +
        2*tanh(a)/3)*x**4 + EX(-tanh(a)**4 + 4*tanh(a)**2/3 - QQ(1, 3))*x**3 + \
        EX(tanh(a)**3 - tanh(a))*x**2 + EX(-tanh(a)**2 + 1)*x + EX(tanh(a))

    p = rs_tanh(x + x**2*y + a, x, 4)
    assert (p.compose(x, 10)).compose(y, 5) == EX(-1000*tanh(a)**4 + \
        10100*tanh(a)**3 + 2470*tanh(a)**2/3 - 10099*tanh(a) + QQ(530, 3))


def test_RR():
    rs_funcs = [rs_sin, rs_cos, rs_tan, rs_cot, rs_atan, rs_tanh]
    sympy_funcs = [sin, cos, tan, cot, atan, tanh]
    R, x, y = ring('x, y', RR)
    a = symbols('a')
    for rs_func, sympy_func in zip(rs_funcs, sympy_funcs):
        p = rs_func(2 + x, x, 5).compose(x, 5)
        q = sympy_func(2 + a).series(a, 0, 5).removeO()
        is_close(p.as_expr(), q.subs(a, 5).n())

    p = rs_nth_root(2 + x, 5, x, 5).compose(x, 5)
    q = ((2 + a)**QQ(1, 5)).series(a, 0, 5).removeO()
    is_close(p.as_expr(), q.subs(a, 5).n())


def test_is_regular():
    R, x, y = puiseux_ring('x, y', QQ)
    p = 1 + 2*x + x**2 + 3*x**3
    assert not rs_is_puiseux(p, x)

    p = x + x**QQ(1,5)*y
    assert rs_is_puiseux(p, x)
    assert not rs_is_puiseux(p, y)

    p = x + x**2*y**QQ(1,5)*y
    assert not rs_is_puiseux(p, x)


def test_puiseux():
    R, x, y = puiseux_ring('x, y', QQ)
    p = x**QQ(2,5) + x**QQ(2,3) + x

    r = rs_series_inversion(p, x, 1)
    r1 = -x**QQ(14,15) + x**QQ(4,5) - 3*x**QQ(11,15) + x**QQ(2,3) + \
        2*x**QQ(7,15) - x**QQ(2,5) - x**QQ(1,5) + x**QQ(2,15) - x**QQ(-2,15) \
        + x**QQ(-2,5)
    assert r == r1

    r = rs_nth_root(1 + p, 3, x, 1)
    assert r == -x**QQ(4,5)/9 + x**QQ(2,3)/3 + x**QQ(2,5)/3 + 1

    r = rs_log(1 + p, x, 1)
    assert r == -x**QQ(4,5)/2 + x**QQ(2,3) + x**QQ(2,5)

    r = rs_LambertW(p, x, 1)
    assert r == -x**QQ(4,5) + x**QQ(2,3) + x**QQ(2,5)

    p1 = x + x**QQ(1,5)*y
    r = rs_exp(p1, x, 1)
    assert r == x**QQ(4,5)*y**4/24 + x**QQ(3,5)*y**3/6 + x**QQ(2,5)*y**2/2 + \
        x**QQ(1,5)*y + 1

    r = rs_atan(p, x, 2)
    assert r ==  -x**QQ(9,5) - x**QQ(26,15) - x**QQ(22,15) - x**QQ(6,5)/3 + \
        x + x**QQ(2,3) + x**QQ(2,5)

    r = rs_atan(p1, x, 2)
    assert r ==  x**QQ(9,5)*y**9/9 + x**QQ(9,5)*y**4 - x**QQ(7,5)*y**7/7 - \
        x**QQ(7,5)*y**2 + x*y**5/5 + x - x**QQ(3,5)*y**3/3 + x**QQ(1,5)*y

    r = rs_tan(p, x, 2)
    assert r == x**QQ(2,5) + x**QQ(2,3) + x + QQ(1,3)*x**QQ(6,5) + x**QQ(22,15)\
        + x**QQ(26,15) + x**QQ(9,5)

    r = rs_sin(p, x, 2)
    assert r == x**QQ(2,5) + x**QQ(2,3) + x - QQ(1,6)*x**QQ(6,5) - QQ(1,2)*x**\
        QQ(22,15) - QQ(1,2)*x**QQ(26,15) - QQ(1,2)*x**QQ(9,5)

    r = rs_cos(p, x, 2)
    assert r == 1 - QQ(1,2)*x**QQ(4,5) - x**QQ(16,15) - QQ(1,2)*x**QQ(4,3) - \
        x**QQ(7,5) + QQ(1,24)*x**QQ(8,5) - x**QQ(5,3) + QQ(1,6)*x**QQ(28,15)

    r = rs_asin(p, x, 2)
    assert r == x**QQ(9,5)/2 + x**QQ(26,15)/2 + x**QQ(22,15)/2 + \
        x**QQ(6,5)/6 + x + x**QQ(2,3) + x**QQ(2,5)

    r = rs_cot(p, x, 1)
    assert r == -x**QQ(14,15) + x**QQ(4,5) - 3*x**QQ(11,15) + \
        2*x**QQ(2,3)/3 + 2*x**QQ(7,15) - 4*x**QQ(2,5)/3 - x**QQ(1,5) + \
        x**QQ(2,15) - x**QQ(-2,15) + x**QQ(-2,5)

    r = rs_cos_sin(p, x, 2)
    assert r[0] == x**QQ(28,15)/6 - x**QQ(5,3) + x**QQ(8,5)/24 - x**QQ(7,5) - \
        x**QQ(4,3)/2 - x**QQ(16,15) - x**QQ(4,5)/2 + 1
    assert r[1] == -x**QQ(9,5)/2 - x**QQ(26,15)/2 - x**QQ(22,15)/2 - \
        x**QQ(6,5)/6 + x + x**QQ(2,3) + x**QQ(2,5)

    r = rs_atanh(p, x, 2)
    assert r == x**QQ(9,5) + x**QQ(26,15) + x**QQ(22,15) + x**QQ(6,5)/3 + x + \
        x**QQ(2,3) + x**QQ(2,5)

    r = rs_asinh(p, x, 2)
    assert r == x**QQ(2,5) + x**QQ(2,3) + x - QQ(1,6)*x**QQ(6,5) - QQ(1,2)*x**\
        QQ(22,15) - QQ(1,2)*x**QQ(26,15) - QQ(1,2)*x**QQ(9,5)

    r = rs_cosh(p, x, 2)
    assert r == x**QQ(28,15)/6 + x**QQ(5,3) + x**QQ(8,5)/24 + x**QQ(7,5) + \
        x**QQ(4,3)/2 + x**QQ(16,15) + x**QQ(4,5)/2 + 1

    r = rs_sinh(p, x, 2)
    assert r == x**QQ(9,5)/2 + x**QQ(26,15)/2 + x**QQ(22,15)/2 + \
        x**QQ(6,5)/6 + x + x**QQ(2,3) + x**QQ(2,5)

    r = rs_cosh_sinh(p, x, 2)
    assert r[0] == x**QQ(28,15)/6 + x**QQ(5,3) + x**QQ(8,5)/24 + x**QQ(7,5) + \
        x**QQ(4,3)/2 + x**QQ(16,15) + x**QQ(4,5)/2 + 1
    assert r[1] == x**QQ(9,5)/2 + x**QQ(26,15)/2 + x**QQ(22,15)/2 + \
        x**QQ(6,5)/6 + x + x**QQ(2,3) + x**QQ(2,5)

    r = rs_tanh(p, x, 2)
    assert r == -x**QQ(9,5) - x**QQ(26,15) - x**QQ(22,15) - x**QQ(6,5)/3 + \
        x + x**QQ(2,3) + x**QQ(2,5)


def test_puiseux_algebraic(): # https://github.com/sympy/sympy/issues/24395

    K = QQ.algebraic_field(sqrt(2))
    sqrt2 = K.from_sympy(sqrt(2))
    x, y = symbols('x, y')
    R, xr, yr = puiseux_ring([x, y], K)
    p = (1+sqrt2)*xr**QQ(1,2) + (1-sqrt2)*yr**QQ(2,3)

    assert p.to_dict() == {(QQ(1,2),QQ(0)):1+sqrt2, (QQ(0),QQ(2,3)):1-sqrt2}
    assert p.as_expr() == (1 + sqrt(2))*x**(S(1)/2) + (1 - sqrt(2))*y**(S(2)/3)


def test1():
    R, x = puiseux_ring('x', QQ)
    r = rs_sin(x, x, 15)*x**(-5)
    assert r == x**8/6227020800 - x**6/39916800 + x**4/362880 - x**2/5040 + \
        QQ(1,120) - x**-2/6 + x**-4

    p = rs_sin(x, x, 10)
    r = rs_nth_root(p, 2, x, 10)
    assert  r == -67*x**QQ(17,2)/29030400 - x**QQ(13,2)/24192 + \
        x**QQ(9,2)/1440 - x**QQ(5,2)/12 + x**QQ(1,2)

    p = rs_sin(x, x, 10)
    r = rs_nth_root(p, 7, x, 10)
    r = rs_pow(r, 5, x, 10)
    assert r == -97*x**QQ(61,7)/124467840 - x**QQ(47,7)/16464 + \
        11*x**QQ(33,7)/3528 - 5*x**QQ(19,7)/42 + x**QQ(5,7)

    r = rs_exp(x**QQ(1,2), x, 10)
    assert r == x**QQ(19,2)/121645100408832000 + x**9/6402373705728000 + \
        x**QQ(17,2)/355687428096000 + x**8/20922789888000 + \
        x**QQ(15,2)/1307674368000 + x**7/87178291200 + \
        x**QQ(13,2)/6227020800 + x**6/479001600 + x**QQ(11,2)/39916800 + \
        x**5/3628800 + x**QQ(9,2)/362880 + x**4/40320 + x**QQ(7,2)/5040 + \
        x**3/720 + x**QQ(5,2)/120 + x**2/24 + x**QQ(3,2)/6 + x/2 + \
        x**QQ(1,2) + 1


def test_puiseux2():
    R, y = ring('y', QQ)
    S, x = puiseux_ring('x', R.to_domain())

    p = x + x**QQ(1,5)*y
    r = rs_atan(p, x, 3)
    assert r == (y**13/13 + y**8 + 2*y**3)*x**QQ(13,5) - (y**11/11 + y**6 +
        y)*x**QQ(11,5) + (y**9/9 + y**4)*x**QQ(9,5) - (y**7/7 +
        y**2)*x**QQ(7,5) + (y**5/5 + 1)*x - y**3*x**QQ(3,5)/3 + y*x**QQ(1,5)


@slow
def test_rs_series():
    x, a, b, c = symbols('x, a, b, c')

    assert rs_series(a, a, 5).as_expr() == a
    assert rs_series(sin(a), a, 5).as_expr() == (sin(a).series(a, 0,
        5)).removeO()
    assert rs_series(sin(a) + cos(a), a, 5).as_expr() == ((sin(a) +
        cos(a)).series(a, 0, 5)).removeO()
    assert rs_series(sin(a)*cos(a), a, 5).as_expr() == ((sin(a)*
        cos(a)).series(a, 0, 5)).removeO()

    p = (sin(a) - a)*(cos(a**2) + a**4/2)
    assert expand(rs_series(p, a, 10).as_expr()) == expand(p.series(a, 0,
        10).removeO())

    p = sin(a**2/2 + a/3) + cos(a/5)*sin(a/2)**3
    assert expand(rs_series(p, a, 5).as_expr()) == expand(p.series(a, 0,
        5).removeO())

    p = sin(x**2 + a)*(cos(x**3 - 1) - a - a**2)
    assert expand(rs_series(p, a, 5).as_expr()) == expand(p.series(a, 0,
        5).removeO())

    p = sin(a**2 - a/3 + 2)**5*exp(a**3 - a/2)
    assert expand(rs_series(p, a, 10).as_expr()) == expand(p.series(a, 0,
        10).removeO())

    p = sin(a + b + c)
    assert expand(rs_series(p, a, 5).as_expr()) == expand(p.series(a, 0,
        5).removeO())

    p = tan(sin(a**2 + 4) + b + c)
    assert expand(rs_series(p, a, 6).as_expr()) == expand(p.series(a, 0,
        6).removeO())

    p = a**QQ(2,5) + a**QQ(2,3) + a

    r = rs_series(tan(p), a, 2)
    assert r.as_expr() == a**QQ(9,5) + a**QQ(26,15) + a**QQ(22,15) + a**QQ(6,5)/3 + \
        a + a**QQ(2,3) + a**QQ(2,5)

    r = rs_series(exp(p), a, 1)
    assert r.as_expr() == a**QQ(4,5)/2 + a**QQ(2,3) + a**QQ(2,5) + 1

    r = rs_series(sin(p), a, 2)
    assert r.as_expr() == -a**QQ(9,5)/2 - a**QQ(26,15)/2 - a**QQ(22,15)/2 - \
        a**QQ(6,5)/6 + a + a**QQ(2,3) + a**QQ(2,5)

    r = rs_series(cos(p), a, 2)
    assert r.as_expr() == a**QQ(28,15)/6 - a**QQ(5,3) + a**QQ(8,5)/24 - a**QQ(7,5) - \
        a**QQ(4,3)/2 - a**QQ(16,15) - a**QQ(4,5)/2 + 1

    assert rs_series(sin(a)/7, a, 5).as_expr() == (sin(a)/7).series(a, 0,
            5).removeO()


def test_rs_series_ConstantInExpr():
    x, a = symbols('x a')
    assert rs_series(log(1 + x), x, 5).as_expr() == -x**4/4 + x**3/3 - \
            x**2/2 + x
    assert rs_series(log(1 + 4*x), x, 5).as_expr() == -64*x**4 + 64*x**3/3 - \
            8*x**2 + 4*x
    assert rs_series(log(1 + x + x**2), x, 10).as_expr() == -2*x**9/9 + \
            x**8/8 + x**7/7 - x**6/3 + x**5/5 + x**4/4 - 2*x**3/3 + x**2/2 + x
    assert rs_series(log(1 + x*a**2), x, 7).as_expr() == -x**6*a**12/6 + \
            x**5*a**10/5 - x**4*a**8/4 + x**3*a**6/3 - x**2*a**4/2 + x*a**2

    assert rs_series(atan(1 + x), x, 9).as_expr() == -x**7/112 + x**6/48 - x**5/40 \
            + x**3/12 - x**2/4 + x/2 + pi/4
    assert rs_series(atan(1 + x + x**2),x, 9).as_expr() == -15*x**7/112 - x**6/48 + \
            9*x**5/40 - 5*x**3/12 + x**2/4 + x/2 + pi/4
    assert rs_series(atan(1 + x * a), x, 9).as_expr() == -a**7*x**7/112 + a**6*x**6/48 \
            - a**5*x**5/40 + a**3*x**3/12 - a**2*x**2/4 + a*x/2 + pi/4

    assert rs_series(tanh(1 + x), x, 5).as_expr() == -5*x**4*tanh(1)**3/3 + x**4* \
            tanh(1)**5 + 2*x**4*tanh(1)/3 - x**3*tanh(1)**4 - x**3/3 + 4*x**3*tanh(1) \
           **2/3 - x**2*tanh(1) + x**2*tanh(1)**3 - x*tanh(1)**2 + x + tanh(1)
    assert rs_series(tanh(1 + x * a), x, 3).as_expr() == -a**2*x**2*tanh(1) + a**2*x** \
            2*tanh(1)**3 - a*x*tanh(1)**2 + a*x + tanh(1)

    assert rs_series(sinh(1 + x), x, 5).as_expr() == x**4*sinh(1)/24 + x**3*cosh(1)/6 + \
            x**2*sinh(1)/2 + x*cosh(1) + sinh(1)
    assert rs_series(sinh(1 + x * a), x, 5).as_expr() == a**4*x**4*sinh(1)/24 + \
            a**3*x**3*cosh(1)/6 + a**2*x**2*sinh(1)/2 + a*x*cosh(1) + sinh(1)

    assert rs_series(cosh(1 + x), x, 5).as_expr() == x**4*cosh(1)/24 + x**3*sinh(1)/6 + \
            x**2*cosh(1)/2 + x*sinh(1) + cosh(1)
    assert rs_series(cosh(1 + x * a), x, 5).as_expr() == a**4*x**4*cosh(1)/24 + \
            a**3*x**3*sinh(1)/6 + a**2*x**2*cosh(1)/2 + a*x*sinh(1) + cosh(1)


def test_issue():
    # https://github.com/sympy/sympy/issues/10191
    # https://github.com/sympy/sympy/issues/19543

    a, b = symbols('a b')
    assert rs_series(sin(a**QQ(3,7))*exp(a + b**QQ(6,7)), a,2).as_expr() == \
        a**QQ(10,7)*exp(b**QQ(6,7)) - a**QQ(9,7)*exp(b**QQ(6,7))/6 + a**QQ(3,7)*exp(b**QQ(6,7))
