from sympy.core.numbers import (Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.geometry.ellipse import (Circle, Ellipse)
from sympy.geometry.line import (Line, Ray2D, Segment2D)
from sympy.geometry.parabola import Parabola
from sympy.geometry.point import (Point, Point2D)
from sympy.testing.pytest import raises

from sympy.abc import x, y

def test_parabola_geom():
    a, b = symbols('a b')
    p1 = Point(0, 0)
    p2 = Point(3, 7)
    p3 = Point(0, 4)
    p4 = Point(6, 0)
    p5 = Point(a, a)
    d1 = Line(Point(4, 0), Point(4, 9))
    d2 = Line(Point(7, 6), Point(3, 6))
    d3 = Line(Point(4, 0), slope=oo)
    d4 = Line(Point(7, 6), slope=0)
    d5 = Line(Point(b, a), slope=oo)
    d6 = Line(Point(a, b), slope=0)

    half = S.Half

    pa1 = Parabola(None, d2)
    pa2 = Parabola(directrix=d1)
    pa3 = Parabola(p1, d1)
    pa4 = Parabola(p2, d2)
    pa5 = Parabola(p2, d4)
    pa6 = Parabola(p3, d2)
    pa7 = Parabola(p2, d1)
    pa8 = Parabola(p4, d1)
    pa9 = Parabola(p4, d3)
    pa10 = Parabola(p5, d5)
    pa11 = Parabola(p5, d6)
    d = Line(Point(3, 7), Point(2, 9))
    pa12 = Parabola(Point(7, 8), d)
    pa12r = Parabola(Point(7, 8).reflect(d), d)

    raises(ValueError, lambda:
           Parabola(Point(7, 8, 9), Line(Point(6, 7), Point(7, 7))))
    raises(ValueError, lambda:
           Parabola(Point(0, 2), Line(Point(7, 2), Point(6, 2))))
    raises(ValueError, lambda: Parabola(Point(7, 8), Point(3, 8)))

    # Basic Stuff
    assert pa1.focus == Point(0, 0)
    assert pa1.ambient_dimension == S(2)
    assert pa2 == pa3
    assert pa4 != pa7
    assert pa6 != pa7
    assert pa6.focus == Point2D(0, 4)
    assert pa6.focal_length == 1
    assert pa6.p_parameter == -1
    assert pa6.vertex == Point2D(0, 5)
    assert pa6.eccentricity == 1
    assert pa7.focus == Point2D(3, 7)
    assert pa7.focal_length == half
    assert pa7.p_parameter == -half
    assert pa7.vertex == Point2D(7*half, 7)
    assert pa4.focal_length == half
    assert pa4.p_parameter == half
    assert pa4.vertex == Point2D(3, 13*half)
    assert pa8.focal_length == 1
    assert pa8.p_parameter == 1
    assert pa8.vertex == Point2D(5, 0)
    assert pa4.focal_length == pa5.focal_length
    assert pa4.p_parameter == pa5.p_parameter
    assert pa4.vertex == pa5.vertex
    assert pa4.equation() == pa5.equation()
    assert pa8.focal_length == pa9.focal_length
    assert pa8.p_parameter == pa9.p_parameter
    assert pa8.vertex == pa9.vertex
    assert pa8.equation() == pa9.equation()
    assert pa10.focal_length == pa11.focal_length == sqrt((a - b) ** 2) / 2 # if a, b real == abs(a - b)/2
    assert pa11.vertex == Point(*pa10.vertex[::-1]) == Point(a,
                            a - sqrt((a - b)**2)*sign(a - b)/2) # change axis x->y, y->x on pa10
    aos = pa12.axis_of_symmetry
    assert aos == Line(Point(7, 8), Point(5, 7))
    assert pa12.directrix == Line(Point(3, 7), Point(2, 9))
    assert pa12.directrix.angle_between(aos) == S.Pi/2
    assert pa12.eccentricity == 1
    assert pa12.equation(x, y) == (x - 7)**2 + (y - 8)**2 - (-2*x - y + 13)**2/5
    assert pa12.focal_length == 9*sqrt(5)/10
    assert pa12.focus == Point(7, 8)
    assert pa12.p_parameter == 9*sqrt(5)/10
    assert pa12.vertex == Point2D(S(26)/5, S(71)/10)
    assert pa12r.focal_length == 9*sqrt(5)/10
    assert pa12r.focus == Point(-S(1)/5, S(22)/5)
    assert pa12r.p_parameter == -9*sqrt(5)/10
    assert pa12r.vertex == Point(S(8)/5, S(53)/10)


def test_parabola_intersection():
    l1 = Line(Point(1, -2), Point(-1,-2))
    l2 = Line(Point(1, 2), Point(-1,2))
    l3 = Line(Point(1, 0), Point(-1,0))

    p1 = Point(0,0)
    p2 = Point(0, -2)
    p3 = Point(120, -12)
    parabola1 = Parabola(p1, l1)

    # parabola with parabola
    assert parabola1.intersection(parabola1) == [parabola1]
    assert parabola1.intersection(Parabola(p1, l2)) == [Point2D(-2, 0), Point2D(2, 0)]
    assert parabola1.intersection(Parabola(p2, l3)) == [Point2D(0, -1)]
    assert parabola1.intersection(Parabola(Point(16, 0), l1)) == [Point2D(8, 15)]
    assert parabola1.intersection(Parabola(Point(0, 16), l1)) == [Point2D(-6, 8), Point2D(6, 8)]
    assert parabola1.intersection(Parabola(p3, l3)) == []
    # parabola with point
    assert parabola1.intersection(p1) == []
    assert parabola1.intersection(Point2D(0, -1)) == [Point2D(0, -1)]
    assert parabola1.intersection(Point2D(4, 3)) == [Point2D(4, 3)]
    # parabola with line
    assert parabola1.intersection(Line(Point2D(-7, 3), Point(12, 3))) == [Point2D(-4, 3), Point2D(4, 3)]
    assert parabola1.intersection(Line(Point(-4, -1), Point(4, -1))) == [Point(0, -1)]
    assert parabola1.intersection(Line(Point(2, 0), Point(0, -2))) == [Point2D(2, 0)]
    raises(TypeError, lambda: parabola1.intersection(Line(Point(0, 0, 0), Point(1, 1, 1))))
    # parabola with segment
    assert parabola1.intersection(Segment2D((-4, -5), (4, 3))) == [Point2D(0, -1), Point2D(4, 3)]
    assert parabola1.intersection(Segment2D((0, -5), (0, 6))) == [Point2D(0, -1)]
    assert parabola1.intersection(Segment2D((-12, -65), (14, -68))) == []
    # parabola with ray
    assert parabola1.intersection(Ray2D((-4, -5), (4, 3))) == [Point2D(0, -1), Point2D(4, 3)]
    assert parabola1.intersection(Ray2D((0, 7), (1, 14))) == [Point2D(14 + 2*sqrt(57), 105 + 14*sqrt(57))]
    assert parabola1.intersection(Ray2D((0, 7), (0, 14))) == []
    # parabola with ellipse/circle
    assert parabola1.intersection(Circle(p1, 2)) == [Point2D(-2, 0), Point2D(2, 0)]
    assert parabola1.intersection(Circle(p2, 1)) == [Point2D(0, -1)]
    assert parabola1.intersection(Ellipse(p2, 2, 1)) == [Point2D(0, -1)]
    assert parabola1.intersection(Ellipse(Point(0, 19), 5, 7)) == []
    assert parabola1.intersection(Ellipse((0, 3), 12, 4)) == [
           Point2D(0, -1),
           Point2D(-4*sqrt(17)/3, Rational(59, 9)),
           Point2D(4*sqrt(17)/3, Rational(59, 9))]
    # parabola with unsupported type
    raises(TypeError, lambda: parabola1.intersection(2))
