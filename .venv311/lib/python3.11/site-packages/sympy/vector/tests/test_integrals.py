from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.testing.pytest import raises
from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.integrals import ParametricIntegral, vector_integrate
from sympy.vector.parametricregion import ParametricRegion
from sympy.vector.implicitregion import ImplicitRegion
from sympy.abc import x, y, z, u, v, r, t, theta, phi
from sympy.geometry import Point, Segment, Curve, Circle, Polygon, Plane

C = CoordSys3D('C')

def test_parametric_lineintegrals():
    halfcircle = ParametricRegion((4*cos(theta), 4*sin(theta)), (theta, -pi/2, pi/2))
    assert ParametricIntegral(C.x*C.y**4, halfcircle) == S(8192)/5

    curve = ParametricRegion((t, t**2, t**3), (t, 0, 1))
    field1 = 8*C.x**2*C.y*C.z*C.i + 5*C.z*C.j - 4*C.x*C.y*C.k
    assert ParametricIntegral(field1, curve) == 1
    line = ParametricRegion((4*t - 1, 2 - 2*t, t), (t, 0, 1))
    assert ParametricIntegral(C.x*C.z*C.i - C.y*C.z*C.k, line) == 3

    assert ParametricIntegral(4*C.x**3, ParametricRegion((1, t), (t, 0, 2))) == 8

    helix = ParametricRegion((cos(t), sin(t), 3*t), (t, 0, 4*pi))
    assert ParametricIntegral(C.x*C.y*C.z, helix) == -3*sqrt(10)*pi

    field2 = C.y*C.i + C.z*C.j + C.z*C.k
    assert ParametricIntegral(field2, ParametricRegion((cos(t), sin(t), t**2), (t, 0, pi))) == -5*pi/2 + pi**4/2

def test_parametric_surfaceintegrals():

    semisphere = ParametricRegion((2*sin(phi)*cos(theta), 2*sin(phi)*sin(theta), 2*cos(phi)),\
                            (theta, 0, 2*pi), (phi, 0, pi/2))
    assert ParametricIntegral(C.z, semisphere) == 8*pi

    cylinder = ParametricRegion((sqrt(3)*cos(theta), sqrt(3)*sin(theta), z), (z, 0, 6), (theta, 0, 2*pi))
    assert ParametricIntegral(C.y, cylinder) == 0

    cone = ParametricRegion((v*cos(u), v*sin(u), v), (u, 0, 2*pi), (v, 0, 1))
    assert ParametricIntegral(C.x*C.i + C.y*C.j + C.z**4*C.k, cone) == pi/3

    triangle1 = ParametricRegion((x, y), (x, 0, 2), (y, 0, 10 - 5*x))
    triangle2 = ParametricRegion((x, y), (y, 0, 10 - 5*x), (x, 0, 2))
    assert ParametricIntegral(-15.6*C.y*C.k, triangle1) == ParametricIntegral(-15.6*C.y*C.k, triangle2)
    assert ParametricIntegral(C.z, triangle1) == 10*C.z

def test_parametric_volumeintegrals():

    cube = ParametricRegion((x, y, z), (x, 0, 1), (y, 0, 1), (z, 0, 1))
    assert ParametricIntegral(1, cube) == 1

    solidsphere1 = ParametricRegion((r*sin(phi)*cos(theta), r*sin(phi)*sin(theta), r*cos(phi)),\
                            (r, 0, 2), (theta, 0, 2*pi), (phi, 0, pi))
    solidsphere2 = ParametricRegion((r*sin(phi)*cos(theta), r*sin(phi)*sin(theta), r*cos(phi)),\
                            (r, 0, 2), (phi, 0, pi), (theta, 0, 2*pi))
    assert ParametricIntegral(C.x**2 + C.y**2, solidsphere1) == -256*pi/15
    assert ParametricIntegral(C.x**2 + C.y**2, solidsphere2) == 256*pi/15

    region_under_plane1 = ParametricRegion((x, y, z), (x, 0, 3), (y, 0, -2*x/3 + 2),\
                                    (z, 0, 6 - 2*x - 3*y))
    region_under_plane2 = ParametricRegion((x, y, z), (x, 0, 3), (z, 0, 6 - 2*x - 3*y),\
                                    (y, 0, -2*x/3 + 2))

    assert ParametricIntegral(C.x*C.i + C.j - 100*C.k, region_under_plane1) == \
        ParametricIntegral(C.x*C.i + C.j - 100*C.k, region_under_plane2)
    assert ParametricIntegral(2*C.x, region_under_plane2) == -9

def test_vector_integrate():
    halfdisc = ParametricRegion((r*cos(theta), r* sin(theta)), (r, -2, 2), (theta, 0, pi))
    assert vector_integrate(C.x**2, halfdisc) == 4*pi
    assert vector_integrate(C.x, ParametricRegion((t, t**2), (t, 2, 3))) == -17*sqrt(17)/12 + 37*sqrt(37)/12

    assert  vector_integrate(C.y**3*C.z, (C.x, 0, 3), (C.y, -1, 4)) == 765*C.z/4

    s1 = Segment(Point(0, 0), Point(0, 1))
    assert vector_integrate(-15*C.y, s1) == S(-15)/2
    s2 = Segment(Point(4, 3, 9), Point(1, 1, 7))
    assert vector_integrate(C.y*C.i, s2) == -6

    curve = Curve((sin(t), cos(t)), (t, 0, 2))
    assert vector_integrate(5*C.z, curve) == 10*C.z

    c1 = Circle(Point(2, 3), 6)
    assert vector_integrate(C.x*C.y, c1) == 72*pi
    c2 = Circle(Point(0, 0), Point(1, 1), Point(1, 0))
    assert vector_integrate(1, c2) == c2.circumference

    triangle = Polygon((0, 0), (1, 0), (1, 1))
    assert vector_integrate(C.x*C.i - 14*C.y*C.j, triangle) == 0
    p1, p2, p3, p4 = [(0, 0), (1, 0), (5, 1), (0, 1)]
    poly = Polygon(p1, p2, p3, p4)
    assert vector_integrate(-23*C.z, poly) == -161*C.z - 23*sqrt(17)*C.z

    point = Point(2, 3)
    assert vector_integrate(C.i*C.y, point) == ParametricIntegral(C.y*C.i, ParametricRegion((2, 3)))

    c3 = ImplicitRegion((x, y), x**2 + y**2 - 4)
    assert vector_integrate(45, c3) == 180*pi
    c4 = ImplicitRegion((x, y), (x - 3)**2 + (y - 4)**2 - 9)
    assert vector_integrate(1, c4) == 6*pi

    pl = Plane(Point(1, 1, 1), Point(2, 3, 4), Point(2, 2, 2))
    raises(ValueError, lambda: vector_integrate(C.x*C.z*C.i + C.k, pl))
