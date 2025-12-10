import pytest
from sympy.core.numbers import Float
from sympy.core.function import (Derivative, Function)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions import exp, cos, sin, tan, cosh, sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.geometry import Point, Point2D, Line, Polygon, Segment, convex_hull,\
    intersection, centroid, Point3D, Line3D, Ray, Ellipse
from sympy.geometry.util import idiff, closest_points, farthest_points, _ordered_points, are_coplanar
from sympy.solvers.solvers import solve
from sympy.testing.pytest import raises


def test_idiff():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    t = Symbol('t', real=True)
    f = Function('f')
    g = Function('g')
    # the use of idiff in ellipse also provides coverage
    circ = x**2 + y**2 - 4
    ans = -3*x*(x**2/y**2 + 1)/y**3
    assert ans == idiff(circ, y, x, 3), idiff(circ, y, x, 3)
    assert ans == idiff(circ, [y], x, 3)
    assert idiff(circ, y, x, 3) == ans
    explicit  = 12*x/sqrt(-x**2 + 4)**5
    assert ans.subs(y, solve(circ, y)[0]).equals(explicit)
    assert True in [sol.diff(x, 3).equals(explicit) for sol in solve(circ, y)]
    assert idiff(x + t + y, [y, t], x) == -Derivative(t, x) - 1
    assert idiff(f(x) * exp(f(x)) - x * exp(x), f(x), x) == (x + 1)*exp(x)*exp(-f(x))/(f(x) + 1)
    assert idiff(f(x) - y * exp(x), [f(x), y], x) == (y + Derivative(y, x))*exp(x)
    assert idiff(f(x) - y * exp(x), [y, f(x)], x) == -y + Derivative(f(x), x)*exp(-x)
    assert idiff(f(x) - g(x), [f(x), g(x)], x) == Derivative(g(x), x)
    # this should be fast
    fxy = y - (-10*(-sin(x) + 1/x)**2 + tan(x)**2 + 2*cosh(x/10))
    assert idiff(fxy, y, x) == -20*sin(x)*cos(x) + 2*tan(x)**3 + \
        2*tan(x) + sinh(x/10)/5 + 20*cos(x)/x - 20*sin(x)/x**2 + 20/x**3


def test_intersection():
    assert intersection(Point(0, 0)) == []
    raises(TypeError, lambda: intersection(Point(0, 0), 3))
    assert intersection(
            Segment((0, 0), (2, 0)),
            Segment((-1, 0), (1, 0)),
            Line((0, 0), (0, 1)), pairwise=True) == [
        Point(0, 0), Segment((0, 0), (1, 0))]
    assert intersection(
            Line((0, 0), (0, 1)),
            Segment((0, 0), (2, 0)),
            Segment((-1, 0), (1, 0)), pairwise=True) == [
        Point(0, 0), Segment((0, 0), (1, 0))]
    assert intersection(
            Line((0, 0), (0, 1)),
            Segment((0, 0), (2, 0)),
            Segment((-1, 0), (1, 0)),
            Line((0, 0), slope=1), pairwise=True) == [
        Point(0, 0), Segment((0, 0), (1, 0))]
    R = 4.0
    c = intersection(
            Ray(Point2D(0.001, -1),
            Point2D(0.0008, -1.7)),
            Ellipse(center=Point2D(0, 0), hradius=R, vradius=2.0), pairwise=True)[0].coordinates
    assert c == pytest.approx(
            Point2D(0.000714285723396502, -1.99999996811224, evaluate=False).coordinates)
    # check this is responds to a lower precision parameter
    R = Float(4, 5)
    c2 = intersection(
            Ray(Point2D(0.001, -1),
            Point2D(0.0008, -1.7)),
            Ellipse(center=Point2D(0, 0), hradius=R, vradius=2.0), pairwise=True)[0].coordinates
    assert c2 == pytest.approx(
            Point2D(0.000714285723396502, -1.99999996811224, evaluate=False).coordinates)
    assert c[0]._prec == 53
    assert c2[0]._prec == 20


def test_convex_hull():
    raises(TypeError, lambda: convex_hull(Point(0, 0), 3))
    points = [(1, -1), (1, -2), (3, -1), (-5, -2), (15, -4)]
    assert convex_hull(*points, **{"polygon": False}) == (
        [Point2D(-5, -2), Point2D(1, -1), Point2D(3, -1), Point2D(15, -4)],
        [Point2D(-5, -2), Point2D(15, -4)])


def test_centroid():
    p = Polygon((0, 0), (10, 0), (10, 10))
    q = p.translate(0, 20)
    assert centroid(p, q) == Point(20, 40)/3
    p = Segment((0, 0), (2, 0))
    q = Segment((0, 0), (2, 2))
    assert centroid(p, q) == Point(1, -sqrt(2) + 2)
    assert centroid(Point(0, 0), Point(2, 0)) == Point(2, 0)/2
    assert centroid(Point(0, 0), Point(0, 0), Point(2, 0)) == Point(2, 0)/3


def test_farthest_points_closest_points():
    from sympy.core.random import randint
    from sympy.utilities.iterables import subsets

    for how in (min, max):
        if how == min:
            func = closest_points
        else:
            func = farthest_points

        raises(ValueError, lambda: func(Point2D(0, 0), Point2D(0, 0)))

        # 3rd pt dx is close and pt is closer to 1st pt
        p1 = [Point2D(0, 0), Point2D(3, 0), Point2D(1, 1)]
        # 3rd pt dx is close and pt is closer to 2nd pt
        p2 = [Point2D(0, 0), Point2D(3, 0), Point2D(2, 1)]
        # 3rd pt dx is close and but pt is not closer
        p3 = [Point2D(0, 0), Point2D(3, 0), Point2D(1, 10)]
        # 3rd pt dx is not closer and it's closer to 2nd pt
        p4 = [Point2D(0, 0), Point2D(3, 0), Point2D(4, 0)]
        # 3rd pt dx is not closer and it's closer to 1st pt
        p5 = [Point2D(0, 0), Point2D(3, 0), Point2D(-1, 0)]
        # duplicate point doesn't affect outcome
        dup = [Point2D(0, 0), Point2D(3, 0), Point2D(3, 0), Point2D(-1, 0)]
        # symbolic
        x = Symbol('x', positive=True)
        s = [Point2D(a) for a in ((x, 1), (x + 3, 2), (x + 2, 2))]

        for points in (p1, p2, p3, p4, p5, dup, s):
            d = how(i.distance(j) for i, j in subsets(set(points), 2))
            ans = a, b = list(func(*points))[0]
            assert a.distance(b) == d
            assert ans == _ordered_points(ans)

        # if the following ever fails, the above tests were not sufficient
        # and the logical error in the routine should be fixed
        points = set()
        while len(points) != 7:
            points.add(Point2D(randint(1, 100), randint(1, 100)))
        points = list(points)
        d = how(i.distance(j) for i, j in subsets(points, 2))
        ans = a, b = list(func(*points))[0]
        assert a.distance(b) == d
        assert ans == _ordered_points(ans)

    # equidistant points
    a, b, c = (
        Point2D(0, 0), Point2D(1, 0), Point2D(S.Half, sqrt(3)/2))
    ans = {_ordered_points((i, j))
        for i, j in subsets((a, b, c), 2)}
    assert closest_points(b, c, a) == ans
    assert farthest_points(b, c, a) == ans

    # unique to farthest
    points = [(1, 1), (1, 2), (3, 1), (-5, 2), (15, 4)]
    assert farthest_points(*points) == {
        (Point2D(-5, 2), Point2D(15, 4))}
    points = [(1, -1), (1, -2), (3, -1), (-5, -2), (15, -4)]
    assert farthest_points(*points) == {
        (Point2D(-5, -2), Point2D(15, -4))}
    assert farthest_points((1, 1), (0, 0)) == {
        (Point2D(0, 0), Point2D(1, 1))}
    raises(ValueError, lambda: farthest_points((1, 1)))


def test_are_coplanar():
    a = Line3D(Point3D(5, 0, 0), Point3D(1, -1, 1))
    b = Line3D(Point3D(0, -2, 0), Point3D(3, 1, 1))
    c = Line3D(Point3D(0, -1, 0), Point3D(5, -1, 9))
    d = Line(Point2D(0, 3), Point2D(1, 5))

    assert are_coplanar(a, b, c) == False
    assert are_coplanar(a, d) == False
