from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.abc import x, y, z, s, t
from sympy.sets import FiniteSet, EmptySet
from sympy.geometry import Point
from sympy.vector import ImplicitRegion
from sympy.testing.pytest import raises


def test_ImplicitRegion():
    ellipse = ImplicitRegion((x, y), (x**2/4 + y**2/16 - 1))
    assert ellipse.equation == x**2/4 + y**2/16 - 1
    assert ellipse.variables == (x, y)
    assert ellipse.degree == 2
    r = ImplicitRegion((x, y, z), Eq(x**4 + y**2 - x*y, 6))
    assert r.equation == x**4 + y**2 - x*y - 6
    assert r.variables == (x, y, z)
    assert r.degree == 4


def test_regular_point():
    r1 = ImplicitRegion((x,), x**2 - 16)
    assert r1.regular_point() == (-4,)
    c1 = ImplicitRegion((x, y), x**2 + y**2 - 4)
    assert c1.regular_point() == (0, -2)
    c2 = ImplicitRegion((x, y), (x - S(5)/2)**2 + y**2 - (S(1)/4)**2)
    assert c2.regular_point() == (S(5)/2, -S(1)/4)
    c3 = ImplicitRegion((x, y), (y - 5)**2  - 16*(x - 5))
    assert c3.regular_point() == (5, 5)
    r2 = ImplicitRegion((x, y), x**2 - 4*x*y - 3*y**2 + 4*x + 8*y - 5)
    assert r2.regular_point() == (S(4)/7, S(9)/7)
    r3 = ImplicitRegion((x, y), x**2 - 2*x*y + 3*y**2 - 2*x - 5*y + 3/2)
    raises(ValueError, lambda: r3.regular_point())


def test_singular_points_and_multiplicty():
    r1 = ImplicitRegion((x, y, z), Eq(x + y + z, 0))
    assert r1.singular_points() == EmptySet
    r2 = ImplicitRegion((x, y, z), x*y*z + y**4 -x**2*z**2)
    assert r2.singular_points() == FiniteSet((0, 0, z), (x, 0, 0))
    assert r2.multiplicity((0, 0, 0)) == 3
    assert r2.multiplicity((0, 0, 6)) == 2
    r3 = ImplicitRegion((x, y, z), z**2 - x**2 - y**2)
    assert r3.singular_points() == FiniteSet((0, 0, 0))
    assert r3.multiplicity((0, 0, 0)) == 2
    r4 = ImplicitRegion((x, y), x**2 + y**2 - 2*x)
    assert r4.singular_points() == EmptySet
    assert r4.multiplicity(Point(1, 3)) == 0


def test_rational_parametrization():
    p = ImplicitRegion((x,), x - 2)
    assert p.rational_parametrization() == (x - 2,)

    line = ImplicitRegion((x, y), Eq(y, 3*x + 2))
    assert line.rational_parametrization() == (x, 3*x + 2)

    circle1 = ImplicitRegion((x, y), (x-2)**2 + (y+3)**2 - 4)
    assert circle1.rational_parametrization(parameters=t) == (4*t/(t**2 + 1) + 2, 4*t**2/(t**2 + 1) - 5)
    circle2 = ImplicitRegion((x, y), (x - S.Half)**2 + y**2 - (S(1)/2)**2)

    assert circle2.rational_parametrization(parameters=t) == (t/(t**2 + 1) + S(1)/2, t**2/(t**2 + 1) - S(1)/2)
    circle3 = ImplicitRegion((x, y), Eq(x**2 + y**2, 2*x))
    assert circle3.rational_parametrization(parameters=(t,)) == (2*t/(t**2 + 1) + 1, 2*t**2/(t**2 + 1) - 1)

    parabola = ImplicitRegion((x, y), (y - 3)**2 - 4*(x + 6))
    assert parabola.rational_parametrization(t) == (-6 + 4/t**2, 3 + 4/t)

    rect_hyperbola = ImplicitRegion((x, y), x*y - 1)
    assert rect_hyperbola.rational_parametrization(t) == (-1 + (t + 1)/t, t)

    cubic_curve = ImplicitRegion((x, y), x**3 + x**2 - y**2)
    assert cubic_curve.rational_parametrization(parameters=(t)) == (t**2 - 1, t*(t**2 - 1))
    cuspidal = ImplicitRegion((x, y), (x**3 - y**2))
    assert cuspidal.rational_parametrization(t) == (t**2, t**3)

    I = ImplicitRegion((x, y), x**3 + x**2 - y**2)
    assert I.rational_parametrization(t) == (t**2 - 1, t*(t**2 - 1))

    sphere = ImplicitRegion((x, y, z), Eq(x**2 + y**2 + z**2, 2*x))
    assert sphere.rational_parametrization(parameters=(s, t)) == (2/(s**2 + t**2 + 1), 2*t/(s**2 + t**2 + 1), 2*s/(s**2 + t**2 + 1))

    conic = ImplicitRegion((x, y), Eq(x**2 + 4*x*y + 3*y**2 + x - y + 10, 0))
    assert conic.rational_parametrization(t) == (
        S(17)/2 + 4/(3*t**2 + 4*t + 1), 4*t/(3*t**2 + 4*t + 1) - S(11)/2)

    r1 = ImplicitRegion((x, y), y**2 - x**3 + x)
    raises(NotImplementedError, lambda: r1.rational_parametrization())
    r2 = ImplicitRegion((x, y), y**2 - x**3 - x**2 + 1)
    raises(NotImplementedError, lambda: r2.rational_parametrization())
