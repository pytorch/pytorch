"""Tests for high-level polynomials manipulation functions. """

from sympy.polys.polyfuncs import (
    symmetrize, horner, interpolate, rational_interpolate, viete,
)

from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
)

from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.testing.pytest import raises

from sympy.abc import a, b, c, d, e, x, y, z


def test_symmetrize():
    assert symmetrize(0, x, y, z) == (0, 0)
    assert symmetrize(1, x, y, z) == (1, 0)

    s1 = x + y + z
    s2 = x*y + x*z + y*z

    assert symmetrize(1) == (1, 0)
    assert symmetrize(1, formal=True) == (1, 0, [])

    assert symmetrize(x) == (x, 0)
    assert symmetrize(x + 1) == (x + 1, 0)

    assert symmetrize(x, x, y) == (x + y, -y)
    assert symmetrize(x + 1, x, y) == (x + y + 1, -y)

    assert symmetrize(x, x, y, z) == (s1, -y - z)
    assert symmetrize(x + 1, x, y, z) == (s1 + 1, -y - z)

    assert symmetrize(x**2, x, y, z) == (s1**2 - 2*s2, -y**2 - z**2)

    assert symmetrize(x**2 + y**2) == (-2*x*y + (x + y)**2, 0)
    assert symmetrize(x**2 - y**2) == (-2*x*y + (x + y)**2, -2*y**2)

    assert symmetrize(x**3 + y**2 + a*x**2 + b*y**3, x, y) == \
        (-3*x*y*(x + y) - 2*a*x*y + a*(x + y)**2 + (x + y)**3,
         y**2*(1 - a) + y**3*(b - 1))

    U = [u0, u1, u2] = symbols('u:3')

    assert symmetrize(x + 1, x, y, z, formal=True, symbols=U) == \
        (u0 + 1, -y - z, [(u0, x + y + z), (u1, x*y + x*z + y*z), (u2, x*y*z)])

    assert symmetrize([1, 2, 3]) == [(1, 0), (2, 0), (3, 0)]
    assert symmetrize([1, 2, 3], formal=True) == ([(1, 0), (2, 0), (3, 0)], [])

    assert symmetrize([x + y, x - y]) == [(x + y, 0), (x + y, -2*y)]


def test_horner():
    assert horner(0) == 0
    assert horner(1) == 1
    assert horner(x) == x

    assert horner(x + 1) == x + 1
    assert horner(x**2 + 1) == x**2 + 1
    assert horner(x**2 + x) == (x + 1)*x
    assert horner(x**2 + x + 1) == (x + 1)*x + 1

    assert horner(
        9*x**4 + 8*x**3 + 7*x**2 + 6*x + 5) == (((9*x + 8)*x + 7)*x + 6)*x + 5
    assert horner(
        a*x**4 + b*x**3 + c*x**2 + d*x + e) == (((a*x + b)*x + c)*x + d)*x + e

    assert horner(4*x**2*y**2 + 2*x**2*y + 2*x*y**2 + x*y, wrt=x) == ((
        4*y + 2)*x*y + (2*y + 1)*y)*x
    assert horner(4*x**2*y**2 + 2*x**2*y + 2*x*y**2 + x*y, wrt=y) == ((
        4*x + 2)*y*x + (2*x + 1)*x)*y


def test_interpolate():
    assert interpolate([1, 4, 9, 16], x) == x**2
    assert interpolate([1, 4, 9, 25], x) == S(3)*x**3/2 - S(8)*x**2 + S(33)*x/2 - 9
    assert interpolate([(1, 1), (2, 4), (3, 9)], x) == x**2
    assert interpolate([(1, 2), (2, 5), (3, 10)], x) == 1 + x**2
    assert interpolate({1: 2, 2: 5, 3: 10}, x) == 1 + x**2
    assert interpolate({5: 2, 7: 5, 8: 10, 9: 13}, x) == \
        -S(13)*x**3/24 + S(12)*x**2 - S(2003)*x/24 + 187
    assert interpolate([(1, 3), (0, 6), (2, 5), (5, 7), (-2, 4)], x) == \
        S(-61)*x**4/280 + S(247)*x**3/210 + S(139)*x**2/280 - S(1871)*x/420 + 6
    assert interpolate((9, 4, 9), 3) == 9
    assert interpolate((1, 9, 16), 1) is S.One
    assert interpolate(((x, 1), (2, 3)), x) is S.One
    assert interpolate({x: 1, 2: 3}, x) is S.One
    assert interpolate(((2, x), (1, 3)), x) == x**2 - 4*x + 6


def test_rational_interpolate():
    x, y = symbols('x,y')
    xdata = [1, 2, 3, 4, 5, 6]
    ydata1 = [120, 150, 200, 255, 312, 370]
    ydata2 = [-210, -35, 105, 231, 350, 465]
    assert rational_interpolate(list(zip(xdata, ydata1)), 2) == (
      (60*x**2 + 60)/x )
    assert rational_interpolate(list(zip(xdata, ydata1)), 3) == (
      (60*x**2 + 60)/x )
    assert rational_interpolate(list(zip(xdata, ydata2)), 2, X=y) == (
      (105*y**2 - 525)/(y + 1) )
    xdata = list(range(1,11))
    ydata = [-1923885361858460, -5212158811973685, -9838050145867125,
      -15662936261217245, -22469424125057910, -30073793365223685,
      -38332297297028735, -47132954289530109, -56387719094026320,
      -66026548943876885]
    assert rational_interpolate(list(zip(xdata, ydata)), 5) == (
      (-12986226192544605*x**4 +
      8657484128363070*x**3 - 30301194449270745*x**2 + 4328742064181535*x
      - 4328742064181535)/(x**3 + 9*x**2 - 3*x + 11))


def test_viete():
    r1, r2 = symbols('r1, r2')

    assert viete(
        a*x**2 + b*x + c, [r1, r2], x) == [(r1 + r2, -b/a), (r1*r2, c/a)]

    raises(ValueError, lambda: viete(1, [], x))
    raises(ValueError, lambda: viete(x**2 + 1, [r1]))

    raises(MultivariatePolynomialError, lambda: viete(x + y, [r1]))
