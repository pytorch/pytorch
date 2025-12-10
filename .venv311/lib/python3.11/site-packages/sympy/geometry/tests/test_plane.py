from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (asin, cos, sin)
from sympy.geometry import Line, Point, Ray, Segment, Point3D, Line3D, Ray3D, Segment3D, Plane, Circle
from sympy.geometry.util import are_coplanar
from sympy.testing.pytest import raises


def test_plane():
    x, y, z, u, v = symbols('x y z u v', real=True)
    p1 = Point3D(0, 0, 0)
    p2 = Point3D(1, 1, 1)
    p3 = Point3D(1, 2, 3)
    pl3 = Plane(p1, p2, p3)
    pl4 = Plane(p1, normal_vector=(1, 1, 1))
    pl4b = Plane(p1, p2)
    pl5 = Plane(p3, normal_vector=(1, 2, 3))
    pl6 = Plane(Point3D(2, 3, 7), normal_vector=(2, 2, 2))
    pl7 = Plane(Point3D(1, -5, -6), normal_vector=(1, -2, 1))
    pl8 = Plane(p1, normal_vector=(0, 0, 1))
    pl9 = Plane(p1, normal_vector=(0, 12, 0))
    pl10 = Plane(p1, normal_vector=(-2, 0, 0))
    pl11 = Plane(p2, normal_vector=(0, 0, 1))
    l1 = Line3D(Point3D(5, 0, 0), Point3D(1, -1, 1))
    l2 = Line3D(Point3D(0, -2, 0), Point3D(3, 1, 1))
    l3 = Line3D(Point3D(0, -1, 0), Point3D(5, -1, 9))

    raises(ValueError, lambda: Plane(p1, p1, p1))

    assert Plane(p1, p2, p3) != Plane(p1, p3, p2)
    assert Plane(p1, p2, p3).is_coplanar(Plane(p1, p3, p2))
    assert Plane(p1, p2, p3).is_coplanar(p1)
    assert Plane(p1, p2, p3).is_coplanar(Circle(p1, 1)) is False
    assert Plane(p1, normal_vector=(0, 0, 1)).is_coplanar(Circle(p1, 1))

    assert pl3 == Plane(Point3D(0, 0, 0), normal_vector=(1, -2, 1))
    assert pl3 != pl4
    assert pl4 == pl4b
    assert pl5 == Plane(Point3D(1, 2, 3), normal_vector=(1, 2, 3))

    assert pl5.equation(x, y, z) == x + 2*y + 3*z - 14
    assert pl3.equation(x, y, z) == x - 2*y + z

    assert pl3.p1 == p1
    assert pl4.p1 == p1
    assert pl5.p1 == p3

    assert pl4.normal_vector == (1, 1, 1)
    assert pl5.normal_vector == (1, 2, 3)

    assert p1 in pl3
    assert p1 in pl4
    assert p3 in pl5

    assert pl3.projection(Point(0, 0)) == p1
    p = pl3.projection(Point3D(1, 1, 0))
    assert p == Point3D(Rational(7, 6), Rational(2, 3), Rational(1, 6))
    assert p in pl3

    l = pl3.projection_line(Line(Point(0, 0), Point(1, 1)))
    assert l == Line3D(Point3D(0, 0, 0), Point3D(Rational(7, 6), Rational(2, 3), Rational(1, 6)))
    assert l in pl3
    # get a segment that does not intersect the plane which is also
    # parallel to pl3's normal veector
    t = Dummy()
    r = pl3.random_point()
    a = pl3.perpendicular_line(r).arbitrary_point(t)
    s = Segment3D(a.subs(t, 1), a.subs(t, 2))
    assert s.p1 not in pl3 and s.p2 not in pl3
    assert pl3.projection_line(s).equals(r)
    assert pl3.projection_line(Segment(Point(1, 0), Point(1, 1))) == \
               Segment3D(Point3D(Rational(5, 6), Rational(1, 3), Rational(-1, 6)), Point3D(Rational(7, 6), Rational(2, 3), Rational(1, 6)))
    assert pl6.projection_line(Ray(Point(1, 0), Point(1, 1))) == \
               Ray3D(Point3D(Rational(14, 3), Rational(11, 3), Rational(11, 3)), Point3D(Rational(13, 3), Rational(13, 3), Rational(10, 3)))
    assert pl3.perpendicular_line(r.args) == pl3.perpendicular_line(r)

    assert pl3.is_parallel(pl6) is False
    assert pl4.is_parallel(pl6)
    assert pl3.is_parallel(Line(p1, p2))
    assert pl6.is_parallel(l1) is False

    assert pl3.is_perpendicular(pl6)
    assert pl4.is_perpendicular(pl7)
    assert pl6.is_perpendicular(pl7)
    assert pl6.is_perpendicular(pl4) is False
    assert pl6.is_perpendicular(l1) is False
    assert pl6.is_perpendicular(Line((0, 0, 0), (1, 1, 1)))
    assert pl6.is_perpendicular((1, 1)) is False

    assert pl6.distance(pl6.arbitrary_point(u, v)) == 0
    assert pl7.distance(pl7.arbitrary_point(u, v)) == 0
    assert pl6.distance(pl6.arbitrary_point(t)) == 0
    assert pl7.distance(pl7.arbitrary_point(t)) == 0
    assert pl6.p1.distance(pl6.arbitrary_point(t)).simplify() == 1
    assert pl7.p1.distance(pl7.arbitrary_point(t)).simplify() == 1
    assert pl3.arbitrary_point(t) == Point3D(-sqrt(30)*sin(t)/30 + \
        2*sqrt(5)*cos(t)/5, sqrt(30)*sin(t)/15 + sqrt(5)*cos(t)/5, sqrt(30)*sin(t)/6)
    assert pl3.arbitrary_point(u, v) == Point3D(2*u - v, u + 2*v, 5*v)

    assert pl7.distance(Point3D(1, 3, 5)) == 5*sqrt(6)/6
    assert pl6.distance(Point3D(0, 0, 0)) == 4*sqrt(3)
    assert pl6.distance(pl6.p1) == 0
    assert pl7.distance(pl6) == 0
    assert pl7.distance(l1) == 0
    assert pl6.distance(Segment3D(Point3D(2, 3, 1), Point3D(1, 3, 4))) == \
        pl6.distance(Point3D(1, 3, 4)) == 4*sqrt(3)/3
    assert pl6.distance(Segment3D(Point3D(1, 3, 4), Point3D(0, 3, 7))) == \
        pl6.distance(Point3D(0, 3, 7)) == 2*sqrt(3)/3
    assert pl6.distance(Segment3D(Point3D(0, 3, 7), Point3D(-1, 3, 10))) == 0
    assert pl6.distance(Segment3D(Point3D(-1, 3, 10), Point3D(-2, 3, 13))) == 0
    assert pl6.distance(Segment3D(Point3D(-2, 3, 13), Point3D(-3, 3, 16))) == \
        pl6.distance(Point3D(-2, 3, 13)) == 2*sqrt(3)/3
    assert pl6.distance(Plane(Point3D(5, 5, 5), normal_vector=(8, 8, 8))) == sqrt(3)
    assert pl6.distance(Ray3D(Point3D(1, 3, 4), direction_ratio=[1, 0, -3])) == 4*sqrt(3)/3
    assert pl6.distance(Ray3D(Point3D(2, 3, 1), direction_ratio=[-1, 0, 3])) == 0


    assert pl6.angle_between(pl3) == pi/2
    assert pl6.angle_between(pl6) == 0
    assert pl6.angle_between(pl4) == 0
    assert pl7.angle_between(Line3D(Point3D(2, 3, 5), Point3D(2, 4, 6))) == \
        -asin(sqrt(3)/6)
    assert pl6.angle_between(Ray3D(Point3D(2, 4, 1), Point3D(6, 5, 3))) == \
        asin(sqrt(7)/3)
    assert pl7.angle_between(Segment3D(Point3D(5, 6, 1), Point3D(1, 2, 4))) == \
        asin(7*sqrt(246)/246)

    assert are_coplanar(l1, l2, l3) is False
    assert are_coplanar(l1) is False
    assert are_coplanar(Point3D(2, 7, 2), Point3D(0, 0, 2),
        Point3D(1, 1, 2), Point3D(1, 2, 2))
    assert are_coplanar(Plane(p1, p2, p3), Plane(p1, p3, p2))
    assert Plane.are_concurrent(pl3, pl4, pl5) is False
    assert Plane.are_concurrent(pl6) is False
    raises(ValueError, lambda: Plane.are_concurrent(Point3D(0, 0, 0)))
    raises(ValueError, lambda: Plane((1, 2, 3), normal_vector=(0, 0, 0)))

    assert pl3.parallel_plane(Point3D(1, 2, 5)) == Plane(Point3D(1, 2, 5), \
                                                      normal_vector=(1, -2, 1))

    # perpendicular_plane
    p = Plane((0, 0, 0), (1, 0, 0))
    # default
    assert p.perpendicular_plane() == Plane(Point3D(0, 0, 0), (0, 1, 0))
    # 1 pt
    assert p.perpendicular_plane(Point3D(1, 0, 1)) == \
        Plane(Point3D(1, 0, 1), (0, 1, 0))
    # pts as tuples
    assert p.perpendicular_plane((1, 0, 1), (1, 1, 1)) == \
        Plane(Point3D(1, 0, 1), (0, 0, -1))
    # more than two planes
    raises(ValueError, lambda: p.perpendicular_plane((1, 0, 1), (1, 1, 1), (1, 1, 0)))

    a, b = Point3D(0, 0, 0), Point3D(0, 1, 0)
    Z = (0, 0, 1)
    p = Plane(a, normal_vector=Z)
    # case 4
    assert p.perpendicular_plane(a, b) == Plane(a, (1, 0, 0))
    n = Point3D(*Z)
    # case 1
    assert p.perpendicular_plane(a, n) == Plane(a, (-1, 0, 0))
    # case 2
    assert Plane(a, normal_vector=b.args).perpendicular_plane(a, a + b) == \
        Plane(Point3D(0, 0, 0), (1, 0, 0))
    # case 1&3
    assert Plane(b, normal_vector=Z).perpendicular_plane(b, b + n) == \
        Plane(Point3D(0, 1, 0), (-1, 0, 0))
    # case 2&3
    assert Plane(b, normal_vector=b.args).perpendicular_plane(n, n + b) == \
        Plane(Point3D(0, 0, 1), (1, 0, 0))

    p = Plane(a, normal_vector=(0, 0, 1))
    assert p.perpendicular_plane() == Plane(a, normal_vector=(1, 0, 0))

    assert pl6.intersection(pl6) == [pl6]
    assert pl4.intersection(pl4.p1) == [pl4.p1]
    assert pl3.intersection(pl6) == [
        Line3D(Point3D(8, 4, 0), Point3D(2, 4, 6))]
    assert pl3.intersection(Line3D(Point3D(1,2,4), Point3D(4,4,2))) == [
        Point3D(2, Rational(8, 3), Rational(10, 3))]
    assert pl3.intersection(Plane(Point3D(6, 0, 0), normal_vector=(2, -5, 3))
        ) == [Line3D(Point3D(-24, -12, 0), Point3D(-25, -13, -1))]
    assert pl6.intersection(Ray3D(Point3D(2, 3, 1), Point3D(1, 3, 4))) == [
        Point3D(-1, 3, 10)]
    assert pl6.intersection(Segment3D(Point3D(2, 3, 1), Point3D(1, 3, 4))) == []
    assert pl7.intersection(Line(Point(2, 3), Point(4, 2))) == [
        Point3D(Rational(13, 2), Rational(3, 4), 0)]
    r = Ray(Point(2, 3), Point(4, 2))
    assert Plane((1,2,0), normal_vector=(0,0,1)).intersection(r) == [
        Ray3D(Point(2, 3), Point(4, 2))]
    assert pl9.intersection(pl8) == [Line3D(Point3D(0, 0, 0), Point3D(12, 0, 0))]
    assert pl10.intersection(pl11) == [Line3D(Point3D(0, 0, 1), Point3D(0, 2, 1))]
    assert pl4.intersection(pl8) == [Line3D(Point3D(0, 0, 0), Point3D(1, -1, 0))]
    assert pl11.intersection(pl8) == []
    assert pl9.intersection(pl11) == [Line3D(Point3D(0, 0, 1), Point3D(12, 0, 1))]
    assert pl9.intersection(pl4) == [Line3D(Point3D(0, 0, 0), Point3D(12, 0, -12))]
    assert pl3.random_point() in pl3
    assert pl3.random_point(seed=1) in pl3

    # test geometrical entity using equals
    assert pl4.intersection(pl4.p1)[0].equals(pl4.p1)
    assert pl3.intersection(pl6)[0].equals(Line3D(Point3D(8, 4, 0), Point3D(2, 4, 6)))
    pl8 = Plane((1, 2, 0), normal_vector=(0, 0, 1))
    assert pl8.intersection(Line3D(p1, (1, 12, 0)))[0].equals(Line((0, 0, 0), (0.1, 1.2, 0)))
    assert pl8.intersection(Ray3D(p1, (1, 12, 0)))[0].equals(Ray((0, 0, 0), (1, 12, 0)))
    assert pl8.intersection(Segment3D(p1, (21, 1, 0)))[0].equals(Segment3D(p1, (21, 1, 0)))
    assert pl8.intersection(Plane(p1, normal_vector=(0, 0, 112)))[0].equals(pl8)
    assert pl8.intersection(Plane(p1, normal_vector=(0, 12, 0)))[0].equals(
        Line3D(p1, direction_ratio=(112 * pi, 0, 0)))
    assert pl8.intersection(Plane(p1, normal_vector=(11, 0, 1)))[0].equals(
        Line3D(p1, direction_ratio=(0, -11, 0)))
    assert pl8.intersection(Plane(p1, normal_vector=(1, 0, 11)))[0].equals(
        Line3D(p1, direction_ratio=(0, 11, 0)))
    assert pl8.intersection(Plane(p1, normal_vector=(-1, -1, -11)))[0].equals(
        Line3D(p1, direction_ratio=(1, -1, 0)))
    assert pl3.random_point() in pl3
    assert len(pl8.intersection(Ray3D(Point3D(0, 2, 3), Point3D(1, 0, 3)))) == 0
    # check if two plane are equals
    assert pl6.intersection(pl6)[0].equals(pl6)
    assert pl8.equals(Plane(p1, normal_vector=(0, 12, 0))) is False
    assert pl8.equals(pl8)
    assert pl8.equals(Plane(p1, normal_vector=(0, 0, -12)))
    assert pl8.equals(Plane(p1, normal_vector=(0, 0, -12*sqrt(3))))
    assert pl8.equals(p1) is False

    # issue 8570
    l2 = Line3D(Point3D(Rational(50000004459633, 5000000000000),
                        Rational(-891926590718643, 1000000000000000),
                        Rational(231800966893633, 100000000000000)),
                Point3D(Rational(50000004459633, 50000000000000),
                        Rational(-222981647679771, 250000000000000),
                        Rational(231800966893633, 100000000000000)))

    p2 = Plane(Point3D(Rational(402775636372767, 100000000000000),
                       Rational(-97224357654973, 100000000000000),
                       Rational(216793600814789, 100000000000000)),
               (-S('9.00000087501922'), -S('4.81170658872543e-13'),
                S('0.0')))

    assert str([i.n(2) for i in p2.intersection(l2)]) == \
           '[Point3D(4.0, -0.89, 2.3)]'


def test_dimension_normalization():
    A = Plane(Point3D(1, 1, 2), normal_vector=(1, 1, 1))
    b = Point(1, 1)
    assert A.projection(b) == Point(Rational(5, 3), Rational(5, 3), Rational(2, 3))

    a, b = Point(0, 0), Point3D(0, 1)
    Z = (0, 0, 1)
    p = Plane(a, normal_vector=Z)
    assert p.perpendicular_plane(a, b) == Plane(Point3D(0, 0, 0), (1, 0, 0))
    assert Plane((1, 2, 1), (2, 1, 0), (3, 1, 2)
        ).intersection((2, 1)) == [Point(2, 1, 0)]


def test_parameter_value():
    t, u, v = symbols("t, u v")
    p1, p2, p3 = Point(0, 0, 0), Point(0, 0, 1), Point(0, 1, 0)
    p = Plane(p1, p2, p3)
    assert p.parameter_value((0, -3, 2), t) == {t: asin(2*sqrt(13)/13)}
    assert p.parameter_value((0, -3, 2), u, v) == {u: 3, v: 2}
    assert p.parameter_value(p1, t) == p1
    raises(ValueError, lambda: p.parameter_value((1, 0, 0), t))
    raises(ValueError, lambda: p.parameter_value(Line(Point(0, 0), Point(1, 1)), t))
    raises(ValueError, lambda: p.parameter_value((0, -3, 2), t, 1))
