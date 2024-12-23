from sympy.core import expand
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sec
from sympy.geometry.line import Segment2D
from sympy.geometry.point import Point2D
from sympy.geometry import (Circle, Ellipse, GeometryError, Line, Point,
                            Polygon, Ray, RegularPolygon, Segment,
                            Triangle, intersection)
from sympy.testing.pytest import raises, slow
from sympy.integrals.integrals import integrate
from sympy.functions.special.elliptic_integrals import elliptic_e
from sympy.functions.elementary.miscellaneous import Max


def test_ellipse_equation_using_slope():
    from sympy.abc import x, y

    e1 = Ellipse(Point(1, 0), 3, 2)
    assert str(e1.equation(_slope=1)) == str((-x + y + 1)**2/8 + (x + y - 1)**2/18 - 1)

    e2 = Ellipse(Point(0, 0), 4, 1)
    assert str(e2.equation(_slope=1)) == str((-x + y)**2/2 + (x + y)**2/32 - 1)

    e3 = Ellipse(Point(1, 5), 6, 2)
    assert str(e3.equation(_slope=2)) == str((-2*x + y - 3)**2/20 + (x + 2*y - 11)**2/180 - 1)


def test_object_from_equation():
    from sympy.abc import x, y, a, b, c, d, e
    assert Circle(x**2 + y**2 + 3*x + 4*y - 8) == Circle(Point2D(S(-3) / 2, -2), sqrt(57) / 2)
    assert Circle(x**2 + y**2 + 6*x + 8*y + 25) == Circle(Point2D(-3, -4), 0)
    assert Circle(a**2 + b**2 + 6*a + 8*b + 25, x='a', y='b') == Circle(Point2D(-3, -4), 0)
    assert Circle(x**2 + y**2 - 25) == Circle(Point2D(0, 0), 5)
    assert Circle(x**2 + y**2) == Circle(Point2D(0, 0), 0)
    assert Circle(a**2 + b**2, x='a', y='b') == Circle(Point2D(0, 0), 0)
    assert Circle(x**2 + y**2 + 6*x + 8) == Circle(Point2D(-3, 0), 1)
    assert Circle(x**2 + y**2 + 6*y + 8) == Circle(Point2D(0, -3), 1)
    assert Circle((x - 1)**2 + y**2 - 9) == Circle(Point2D(1, 0), 3)
    assert Circle(6*(x**2) + 6*(y**2) + 6*x + 8*y - 25) == Circle(Point2D(Rational(-1, 2), Rational(-2, 3)), 5*sqrt(7)/6)
    assert Circle(Eq(a**2 + b**2, 25), x='a', y=b) == Circle(Point2D(0, 0), 5)
    raises(GeometryError, lambda: Circle(x**2 + y**2 + 3*x + 4*y + 26))
    raises(GeometryError, lambda: Circle(x**2 + y**2 + 25))
    raises(GeometryError, lambda: Circle(a**2 + b**2 + 25, x='a', y='b'))
    raises(GeometryError, lambda: Circle(x**2 + 6*y + 8))
    raises(GeometryError, lambda: Circle(6*(x ** 2) + 4*(y**2) + 6*x + 8*y + 25))
    raises(ValueError, lambda: Circle(a**2 + b**2 + 3*a + 4*b - 8))
    # .equation() adds 'real=True' assumption; '==' would fail if assumptions differed
    x, y = symbols('x y', real=True)
    eq = a*x**2 + a*y**2 + c*x + d*y + e
    assert expand(Circle(eq).equation()*a) == eq


@slow
def test_ellipse_geom():
    x = Symbol('x', real=True)
    y = Symbol('y', real=True)
    t = Symbol('t', real=True)
    y1 = Symbol('y1', real=True)
    half = S.Half
    p1 = Point(0, 0)
    p2 = Point(1, 1)
    p4 = Point(0, 1)

    e1 = Ellipse(p1, 1, 1)
    e2 = Ellipse(p2, half, 1)
    e3 = Ellipse(p1, y1, y1)
    c1 = Circle(p1, 1)
    c2 = Circle(p2, 1)
    c3 = Circle(Point(sqrt(2), sqrt(2)), 1)
    l1 = Line(p1, p2)

    # Test creation with three points
    cen, rad = Point(3*half, 2), 5*half
    assert Circle(Point(0, 0), Point(3, 0), Point(0, 4)) == Circle(cen, rad)
    assert Circle(Point(0, 0), Point(1, 1), Point(2, 2)) == Segment2D(Point2D(0, 0), Point2D(2, 2))

    raises(ValueError, lambda: Ellipse(None, None, None, 1))
    raises(ValueError, lambda: Ellipse())
    raises(GeometryError, lambda: Circle(Point(0, 0)))
    raises(GeometryError, lambda: Circle(Symbol('x')*Symbol('y')))

    # Basic Stuff
    assert Ellipse(None, 1, 1).center == Point(0, 0)
    assert e1 == c1
    assert e1 != e2
    assert e1 != l1
    assert p4 in e1
    assert e1 in e1
    assert e2 in e2
    assert 1 not in e2
    assert p2 not in e2
    assert e1.area == pi
    assert e2.area == pi/2
    assert e3.area == pi*y1*abs(y1)
    assert c1.area == e1.area
    assert c1.circumference == e1.circumference
    assert e3.circumference == 2*pi*y1
    assert e1.plot_interval() == e2.plot_interval() == [t, -pi, pi]
    assert e1.plot_interval(x) == e2.plot_interval(x) == [x, -pi, pi]

    assert c1.minor == 1
    assert c1.major == 1
    assert c1.hradius == 1
    assert c1.vradius == 1

    assert Ellipse((1, 1), 0, 0) == Point(1, 1)
    assert Ellipse((1, 1), 1, 0) == Segment(Point(0, 1), Point(2, 1))
    assert Ellipse((1, 1), 0, 1) == Segment(Point(1, 0), Point(1, 2))

    # Private Functions
    assert hash(c1) == hash(Circle(Point(1, 0), Point(0, 1), Point(0, -1)))
    assert c1 in e1
    assert (Line(p1, p2) in e1) is False
    assert e1.__cmp__(e1) == 0
    assert e1.__cmp__(Point(0, 0)) > 0

    # Encloses
    assert e1.encloses(Segment(Point(-0.5, -0.5), Point(0.5, 0.5))) is True
    assert e1.encloses(Line(p1, p2)) is False
    assert e1.encloses(Ray(p1, p2)) is False
    assert e1.encloses(e1) is False
    assert e1.encloses(
        Polygon(Point(-0.5, -0.5), Point(-0.5, 0.5), Point(0.5, 0.5))) is True
    assert e1.encloses(RegularPolygon(p1, 0.5, 3)) is True
    assert e1.encloses(RegularPolygon(p1, 5, 3)) is False
    assert e1.encloses(RegularPolygon(p2, 5, 3)) is False

    assert e2.arbitrary_point() in e2
    raises(ValueError, lambda: Ellipse(Point(x, y), 1, 1).arbitrary_point(parameter='x'))

    # Foci
    f1, f2 = Point(sqrt(12), 0), Point(-sqrt(12), 0)
    ef = Ellipse(Point(0, 0), 4, 2)
    assert ef.foci in [(f1, f2), (f2, f1)]

    # Tangents
    v = sqrt(2) / 2
    p1_1 = Point(v, v)
    p1_2 = p2 + Point(half, 0)
    p1_3 = p2 + Point(0, 1)
    assert e1.tangent_lines(p4) == c1.tangent_lines(p4)
    assert e2.tangent_lines(p1_2) == [Line(Point(Rational(3, 2), 1), Point(Rational(3, 2), S.Half))]
    assert e2.tangent_lines(p1_3) == [Line(Point(1, 2), Point(Rational(5, 4), 2))]
    assert c1.tangent_lines(p1_1) != [Line(p1_1, Point(0, sqrt(2)))]
    assert c1.tangent_lines(p1) == []
    assert e2.is_tangent(Line(p1_2, p2 + Point(half, 1)))
    assert e2.is_tangent(Line(p1_3, p2 + Point(half, 1)))
    assert c1.is_tangent(Line(p1_1, Point(0, sqrt(2))))
    assert e1.is_tangent(Line(Point(0, 0), Point(1, 1))) is False
    assert c1.is_tangent(e1) is True
    assert c1.is_tangent(Ellipse(Point(2, 0), 1, 1)) is True
    assert c1.is_tangent(
        Polygon(Point(1, 1), Point(1, -1), Point(2, 0))) is False
    assert c1.is_tangent(
        Polygon(Point(1, 1), Point(1, 0), Point(2, 0))) is False
    assert Circle(Point(5, 5), 3).is_tangent(Circle(Point(0, 5), 1)) is False

    assert Ellipse(Point(5, 5), 2, 1).tangent_lines(Point(0, 0)) == \
        [Line(Point(0, 0), Point(Rational(77, 25), Rational(132, 25))),
     Line(Point(0, 0), Point(Rational(33, 5), Rational(22, 5)))]
    assert Ellipse(Point(5, 5), 2, 1).tangent_lines(Point(3, 4)) == \
        [Line(Point(3, 4), Point(4, 4)), Line(Point(3, 4), Point(3, 5))]
    assert Circle(Point(5, 5), 2).tangent_lines(Point(3, 3)) == \
        [Line(Point(3, 3), Point(4, 3)), Line(Point(3, 3), Point(3, 4))]
    assert Circle(Point(5, 5), 2).tangent_lines(Point(5 - 2*sqrt(2), 5)) == \
        [Line(Point(5 - 2*sqrt(2), 5), Point(5 - sqrt(2), 5 - sqrt(2))),
     Line(Point(5 - 2*sqrt(2), 5), Point(5 - sqrt(2), 5 + sqrt(2))), ]
    assert Circle(Point(5, 5), 5).tangent_lines(Point(4, 0)) == \
        [Line(Point(4, 0), Point(Rational(40, 13), Rational(5, 13))),
     Line(Point(4, 0), Point(5, 0))]
    assert Circle(Point(5, 5), 5).tangent_lines(Point(0, 6)) == \
        [Line(Point(0, 6), Point(0, 7)),
        Line(Point(0, 6), Point(Rational(5, 13), Rational(90, 13)))]

    # for numerical calculations, we shouldn't demand exact equality,
    # so only test up to the desired precision
    def lines_close(l1, l2, prec):
        """ tests whether l1 and 12 are within 10**(-prec)
        of each other """
        return abs(l1.p1 - l2.p1) < 10**(-prec) and abs(l1.p2 - l2.p2) < 10**(-prec)
    def line_list_close(ll1, ll2, prec):
        return all(lines_close(l1, l2, prec) for l1, l2 in zip(ll1, ll2))

    e = Ellipse(Point(0, 0), 2, 1)
    assert e.normal_lines(Point(0, 0)) == \
        [Line(Point(0, 0), Point(0, 1)), Line(Point(0, 0), Point(1, 0))]
    assert e.normal_lines(Point(1, 0)) == \
        [Line(Point(0, 0), Point(1, 0))]
    assert e.normal_lines((0, 1)) == \
        [Line(Point(0, 0), Point(0, 1))]
    assert line_list_close(e.normal_lines(Point(1, 1), 2), [
        Line(Point(Rational(-51, 26), Rational(-1, 5)), Point(Rational(-25, 26), Rational(17, 83))),
        Line(Point(Rational(28, 29), Rational(-7, 8)), Point(Rational(57, 29), Rational(-9, 2)))], 2)
    # test the failure of Poly.intervals and checks a point on the boundary
    p = Point(sqrt(3), S.Half)
    assert p in e
    assert line_list_close(e.normal_lines(p, 2), [
        Line(Point(Rational(-341, 171), Rational(-1, 13)), Point(Rational(-170, 171), Rational(5, 64))),
        Line(Point(Rational(26, 15), Rational(-1, 2)), Point(Rational(41, 15), Rational(-43, 26)))], 2)
    # be sure to use the slope that isn't undefined on boundary
    e = Ellipse((0, 0), 2, 2*sqrt(3)/3)
    assert line_list_close(e.normal_lines((1, 1), 2), [
        Line(Point(Rational(-64, 33), Rational(-20, 71)), Point(Rational(-31, 33), Rational(2, 13))),
        Line(Point(1, -1), Point(2, -4))], 2)
    # general ellipse fails except under certain conditions
    e = Ellipse((0, 0), x, 1)
    assert e.normal_lines((x + 1, 0)) == [Line(Point(0, 0), Point(1, 0))]
    raises(NotImplementedError, lambda: e.normal_lines((x + 1, 1)))
    # Properties
    major = 3
    minor = 1
    e4 = Ellipse(p2, minor, major)
    assert e4.focus_distance == sqrt(major**2 - minor**2)
    ecc = e4.focus_distance / major
    assert e4.eccentricity == ecc
    assert e4.periapsis == major*(1 - ecc)
    assert e4.apoapsis == major*(1 + ecc)
    assert e4.semilatus_rectum == major*(1 - ecc ** 2)
    # independent of orientation
    e4 = Ellipse(p2, major, minor)
    assert e4.focus_distance == sqrt(major**2 - minor**2)
    ecc = e4.focus_distance / major
    assert e4.eccentricity == ecc
    assert e4.periapsis == major*(1 - ecc)
    assert e4.apoapsis == major*(1 + ecc)

    # Intersection
    l1 = Line(Point(1, -5), Point(1, 5))
    l2 = Line(Point(-5, -1), Point(5, -1))
    l3 = Line(Point(-1, -1), Point(1, 1))
    l4 = Line(Point(-10, 0), Point(0, 10))
    pts_c1_l3 = [Point(sqrt(2)/2, sqrt(2)/2), Point(-sqrt(2)/2, -sqrt(2)/2)]

    assert intersection(e2, l4) == []
    assert intersection(c1, Point(1, 0)) == [Point(1, 0)]
    assert intersection(c1, l1) == [Point(1, 0)]
    assert intersection(c1, l2) == [Point(0, -1)]
    assert intersection(c1, l3) in [pts_c1_l3, [pts_c1_l3[1], pts_c1_l3[0]]]
    assert intersection(c1, c2) == [Point(0, 1), Point(1, 0)]
    assert intersection(c1, c3) == [Point(sqrt(2)/2, sqrt(2)/2)]
    assert e1.intersection(l1) == [Point(1, 0)]
    assert e2.intersection(l4) == []
    assert e1.intersection(Circle(Point(0, 2), 1)) == [Point(0, 1)]
    assert e1.intersection(Circle(Point(5, 0), 1)) == []
    assert e1.intersection(Ellipse(Point(2, 0), 1, 1)) == [Point(1, 0)]
    assert e1.intersection(Ellipse(Point(5, 0), 1, 1)) == []
    assert e1.intersection(Point(2, 0)) == []
    assert e1.intersection(e1) == e1
    assert intersection(Ellipse(Point(0, 0), 2, 1), Ellipse(Point(3, 0), 1, 2)) == [Point(2, 0)]
    assert intersection(Circle(Point(0, 0), 2), Circle(Point(3, 0), 1)) == [Point(2, 0)]
    assert intersection(Circle(Point(0, 0), 2), Circle(Point(7, 0), 1)) == []
    assert intersection(Ellipse(Point(0, 0), 5, 17), Ellipse(Point(4, 0), 1, 0.2)
        ) == [Point(5.0, 0, evaluate=False)]
    assert intersection(Ellipse(Point(0, 0), 5, 17), Ellipse(Point(4, 0), 0.999, 0.2)) == []
    assert Circle((0, 0), S.Half).intersection(
        Triangle((-1, 0), (1, 0), (0, 1))) == [
        Point(Rational(-1, 2), 0), Point(S.Half, 0)]
    raises(TypeError, lambda: intersection(e2, Line((0, 0, 0), (0, 0, 1))))
    raises(TypeError, lambda: intersection(e2, Rational(12)))
    raises(TypeError, lambda: Ellipse.intersection(e2, 1))
    # some special case intersections
    csmall = Circle(p1, 3)
    cbig = Circle(p1, 5)
    cout = Circle(Point(5, 5), 1)
    # one circle inside of another
    assert csmall.intersection(cbig) == []
    # separate circles
    assert csmall.intersection(cout) == []
    # coincident circles
    assert csmall.intersection(csmall) == csmall

    v = sqrt(2)
    t1 = Triangle(Point(0, v), Point(0, -v), Point(v, 0))
    points = intersection(t1, c1)
    assert len(points) == 4
    assert Point(0, 1) in points
    assert Point(0, -1) in points
    assert Point(v/2, v/2) in points
    assert Point(v/2, -v/2) in points

    circ = Circle(Point(0, 0), 5)
    elip = Ellipse(Point(0, 0), 5, 20)
    assert intersection(circ, elip) in \
        [[Point(5, 0), Point(-5, 0)], [Point(-5, 0), Point(5, 0)]]
    assert elip.tangent_lines(Point(0, 0)) == []
    elip = Ellipse(Point(0, 0), 3, 2)
    assert elip.tangent_lines(Point(3, 0)) == \
        [Line(Point(3, 0), Point(3, -12))]

    e1 = Ellipse(Point(0, 0), 5, 10)
    e2 = Ellipse(Point(2, 1), 4, 8)
    a = Rational(53, 17)
    c = 2*sqrt(3991)/17
    ans = [Point(a - c/8, a/2 + c), Point(a + c/8, a/2 - c)]
    assert e1.intersection(e2) == ans
    e2 = Ellipse(Point(x, y), 4, 8)
    c = sqrt(3991)
    ans = [Point(-c/68 + a, c*Rational(2, 17) + a/2), Point(c/68 + a, c*Rational(-2, 17) + a/2)]
    assert [p.subs({x: 2, y:1}) for p in e1.intersection(e2)] == ans

    # Combinations of above
    assert e3.is_tangent(e3.tangent_lines(p1 + Point(y1, 0))[0])

    e = Ellipse((1, 2), 3, 2)
    assert e.tangent_lines(Point(10, 0)) == \
        [Line(Point(10, 0), Point(1, 0)),
        Line(Point(10, 0), Point(Rational(14, 5), Rational(18, 5)))]

    # encloses_point
    e = Ellipse((0, 0), 1, 2)
    assert e.encloses_point(e.center)
    assert e.encloses_point(e.center + Point(0, e.vradius - Rational(1, 10)))
    assert e.encloses_point(e.center + Point(e.hradius - Rational(1, 10), 0))
    assert e.encloses_point(e.center + Point(e.hradius, 0)) is False
    assert e.encloses_point(
        e.center + Point(e.hradius + Rational(1, 10), 0)) is False
    e = Ellipse((0, 0), 2, 1)
    assert e.encloses_point(e.center)
    assert e.encloses_point(e.center + Point(0, e.vradius - Rational(1, 10)))
    assert e.encloses_point(e.center + Point(e.hradius - Rational(1, 10), 0))
    assert e.encloses_point(e.center + Point(e.hradius, 0)) is False
    assert e.encloses_point(
        e.center + Point(e.hradius + Rational(1, 10), 0)) is False
    assert c1.encloses_point(Point(1, 0)) is False
    assert c1.encloses_point(Point(0.3, 0.4)) is True

    assert e.scale(2, 3) == Ellipse((0, 0), 4, 3)
    assert e.scale(3, 6) == Ellipse((0, 0), 6, 6)
    assert e.rotate(pi) == e
    assert e.rotate(pi, (1, 2)) == Ellipse(Point(2, 4), 2, 1)
    raises(NotImplementedError, lambda: e.rotate(pi/3))

    # Circle rotation tests (Issue #11743)
    # Link - https://github.com/sympy/sympy/issues/11743
    cir = Circle(Point(1, 0), 1)
    assert cir.rotate(pi/2) == Circle(Point(0, 1), 1)
    assert cir.rotate(pi/3) == Circle(Point(S.Half, sqrt(3)/2), 1)
    assert cir.rotate(pi/3, Point(1, 0)) == Circle(Point(1, 0), 1)
    assert cir.rotate(pi/3, Point(0, 1)) == Circle(Point(S.Half + sqrt(3)/2, S.Half + sqrt(3)/2), 1)


def test_construction():
    e1 = Ellipse(hradius=2, vradius=1, eccentricity=None)
    assert e1.eccentricity == sqrt(3)/2

    e2 = Ellipse(hradius=2, vradius=None, eccentricity=sqrt(3)/2)
    assert e2.vradius == 1

    e3 = Ellipse(hradius=None, vradius=1, eccentricity=sqrt(3)/2)
    assert e3.hradius == 2

    # filter(None, iterator) filters out anything falsey, including 0
    # eccentricity would be filtered out in this case and the constructor would throw an error
    e4 = Ellipse(Point(0, 0), hradius=1, eccentricity=0)
    assert e4.vradius == 1

    #tests for eccentricity > 1
    raises(GeometryError, lambda: Ellipse(Point(3, 1), hradius=3, eccentricity = S(3)/2))
    raises(GeometryError, lambda: Ellipse(Point(3, 1), hradius=3, eccentricity=sec(5)))
    raises(GeometryError, lambda: Ellipse(Point(3, 1), hradius=3, eccentricity=S.Pi-S(2)))

    #tests for eccentricity = 1
    #if vradius is not defined
    assert Ellipse(None, 1, None, 1).length == 2
    #if hradius is not defined
    raises(GeometryError, lambda: Ellipse(None, None, 1, eccentricity = 1))

    #tests for eccentricity < 0
    raises(GeometryError, lambda: Ellipse(Point(3, 1), hradius=3, eccentricity = -3))
    raises(GeometryError, lambda: Ellipse(Point(3, 1), hradius=3, eccentricity = -0.5))

def test_ellipse_random_point():
    y1 = Symbol('y1', real=True)
    e3 = Ellipse(Point(0, 0), y1, y1)
    rx, ry = Symbol('rx'), Symbol('ry')
    for ind in range(0, 5):
        r = e3.random_point()
        # substitution should give zero*y1**2
        assert e3.equation(rx, ry).subs(zip((rx, ry), r.args)).equals(0)
    # test for the case with seed
    r = e3.random_point(seed=1)
    assert e3.equation(rx, ry).subs(zip((rx, ry), r.args)).equals(0)


def test_repr():
    assert repr(Circle((0, 1), 2)) == 'Circle(Point2D(0, 1), 2)'


def test_transform():
    c = Circle((1, 1), 2)
    assert c.scale(-1) == Circle((-1, 1), 2)
    assert c.scale(y=-1) == Circle((1, -1), 2)
    assert c.scale(2) == Ellipse((2, 1), 4, 2)

    assert Ellipse((0, 0), 2, 3).scale(2, 3, (4, 5)) == \
        Ellipse(Point(-4, -10), 4, 9)
    assert Circle((0, 0), 2).scale(2, 3, (4, 5)) == \
        Ellipse(Point(-4, -10), 4, 6)
    assert Ellipse((0, 0), 2, 3).scale(3, 3, (4, 5)) == \
        Ellipse(Point(-8, -10), 6, 9)
    assert Circle((0, 0), 2).scale(3, 3, (4, 5)) == \
        Circle(Point(-8, -10), 6)
    assert Circle(Point(-8, -10), 6).scale(Rational(1, 3), Rational(1, 3), (4, 5)) == \
        Circle((0, 0), 2)
    assert Circle((0, 0), 2).translate(4, 5) == \
        Circle((4, 5), 2)
    assert Circle((0, 0), 2).scale(3, 3) == \
        Circle((0, 0), 6)


def test_bounds():
    e1 = Ellipse(Point(0, 0), 3, 5)
    e2 = Ellipse(Point(2, -2), 7, 7)
    c1 = Circle(Point(2, -2), 7)
    c2 = Circle(Point(-2, 0), Point(0, 2), Point(2, 0))
    assert e1.bounds == (-3, -5, 3, 5)
    assert e2.bounds == (-5, -9, 9, 5)
    assert c1.bounds == (-5, -9, 9, 5)
    assert c2.bounds == (-2, -2, 2, 2)


def test_reflect():
    b = Symbol('b')
    m = Symbol('m')
    l = Line((0, b), slope=m)
    t1 = Triangle((0, 0), (1, 0), (2, 3))
    assert t1.area == -t1.reflect(l).area
    e = Ellipse((1, 0), 1, 2)
    assert e.area == -e.reflect(Line((1, 0), slope=0)).area
    assert e.area == -e.reflect(Line((1, 0), slope=oo)).area
    raises(NotImplementedError, lambda: e.reflect(Line((1, 0), slope=m)))
    assert Circle((0, 1), 1).reflect(Line((0, 0), (1, 1))) == Circle(Point2D(1, 0), -1)


def test_is_tangent():
    e1 = Ellipse(Point(0, 0), 3, 5)
    c1 = Circle(Point(2, -2), 7)
    assert e1.is_tangent(Point(0, 0)) is False
    assert e1.is_tangent(Point(3, 0)) is False
    assert e1.is_tangent(e1) is True
    assert e1.is_tangent(Ellipse((0, 0), 1, 2)) is False
    assert e1.is_tangent(Ellipse((0, 0), 3, 2)) is True
    assert c1.is_tangent(Ellipse((2, -2), 7, 1)) is True
    assert c1.is_tangent(Circle((11, -2), 2)) is True
    assert c1.is_tangent(Circle((7, -2), 2)) is True
    assert c1.is_tangent(Ray((-5, -2), (-15, -20))) is False
    assert c1.is_tangent(Ray((-3, -2), (-15, -20))) is False
    assert c1.is_tangent(Ray((-3, -22), (15, 20))) is False
    assert c1.is_tangent(Ray((9, 20), (9, -20))) is True
    assert c1.is_tangent(Ray((2, 5), (9, 5))) is True
    assert c1.is_tangent(Segment((2, 5), (9, 5))) is True
    assert e1.is_tangent(Segment((2, 2), (-7, 7))) is False
    assert e1.is_tangent(Segment((0, 0), (1, 2))) is False
    assert c1.is_tangent(Segment((0, 0), (-5, -2))) is False
    assert e1.is_tangent(Segment((3, 0), (12, 12))) is False
    assert e1.is_tangent(Segment((12, 12), (3, 0))) is False
    assert e1.is_tangent(Segment((-3, 0), (3, 0))) is False
    assert e1.is_tangent(Segment((-3, 5), (3, 5))) is True
    assert e1.is_tangent(Line((10, 0), (10, 10))) is False
    assert e1.is_tangent(Line((0, 0), (1, 1))) is False
    assert e1.is_tangent(Line((-3, 0), (-2.99, -0.001))) is False
    assert e1.is_tangent(Line((-3, 0), (-3, 1))) is True
    assert e1.is_tangent(Polygon((0, 0), (5, 5), (5, -5))) is False
    assert e1.is_tangent(Polygon((-100, -50), (-40, -334), (-70, -52))) is False
    assert e1.is_tangent(Polygon((-3, 0), (3, 0), (0, 1))) is False
    assert e1.is_tangent(Polygon((-3, 0), (3, 0), (0, 5))) is False
    assert e1.is_tangent(Polygon((-3, 0), (0, -5), (3, 0), (0, 5))) is False
    assert e1.is_tangent(Polygon((-3, -5), (-3, 5), (3, 5), (3, -5))) is True
    assert c1.is_tangent(Polygon((-3, -5), (-3, 5), (3, 5), (3, -5))) is False
    assert e1.is_tangent(Polygon((0, 0), (3, 0), (7, 7), (0, 5))) is False
    assert e1.is_tangent(Polygon((3, 12), (3, -12), (6, 5))) is False
    assert e1.is_tangent(Polygon((3, 12), (3, -12), (0, -5), (0, 5))) is False
    assert e1.is_tangent(Polygon((3, 0), (5, 7), (6, -5))) is False
    assert c1.is_tangent(Segment((0, 0), (-5, -2))) is False
    assert e1.is_tangent(Segment((-3, 0), (3, 0))) is False
    assert e1.is_tangent(Segment((-3, 5), (3, 5))) is True
    assert e1.is_tangent(Polygon((0, 0), (5, 5), (5, -5))) is False
    assert e1.is_tangent(Polygon((-100, -50), (-40, -334), (-70, -52))) is False
    assert e1.is_tangent(Polygon((-3, -5), (-3, 5), (3, 5), (3, -5))) is True
    assert c1.is_tangent(Polygon((-3, -5), (-3, 5), (3, 5), (3, -5))) is False
    assert e1.is_tangent(Polygon((3, 12), (3, -12), (0, -5), (0, 5))) is False
    assert e1.is_tangent(Polygon((3, 0), (5, 7), (6, -5))) is False
    raises(TypeError, lambda: e1.is_tangent(Point(0, 0, 0)))
    raises(TypeError, lambda: e1.is_tangent(Rational(5)))


def test_parameter_value():
    t = Symbol('t')
    e = Ellipse(Point(0, 0), 3, 5)
    assert e.parameter_value((3, 0), t) == {t: 0}
    raises(ValueError, lambda: e.parameter_value((4, 0), t))


@slow
def test_second_moment_of_area():
    x, y = symbols('x, y')
    e = Ellipse(Point(0, 0), 5, 4)
    I_yy = 2*4*integrate(sqrt(25 - x**2)*x**2, (x, -5, 5))/5
    I_xx = 2*5*integrate(sqrt(16 - y**2)*y**2, (y, -4, 4))/4
    Y = 3*sqrt(1 - x**2/5**2)
    I_xy = integrate(integrate(y, (y, -Y, Y))*x, (x, -5, 5))
    assert I_yy == e.second_moment_of_area()[1]
    assert I_xx == e.second_moment_of_area()[0]
    assert I_xy == e.second_moment_of_area()[2]
    #checking for other point
    t1 = e.second_moment_of_area(Point(6,5))
    t2 = (580*pi, 845*pi, 600*pi)
    assert t1==t2


def test_section_modulus_and_polar_second_moment_of_area():
    d = Symbol('d', positive=True)
    c = Circle((3, 7), 8)
    assert c.polar_second_moment_of_area() == 2048*pi
    assert c.section_modulus() == (128*pi, 128*pi)
    c = Circle((2, 9), d/2)
    assert c.polar_second_moment_of_area() == pi*d**3*Abs(d)/64 + pi*d*Abs(d)**3/64
    assert c.section_modulus() == (pi*d**3/S(32), pi*d**3/S(32))

    a, b = symbols('a, b', positive=True)
    e = Ellipse((4, 6), a, b)
    assert e.section_modulus() == (pi*a*b**2/S(4), pi*a**2*b/S(4))
    assert e.polar_second_moment_of_area() == pi*a**3*b/S(4) + pi*a*b**3/S(4)
    e = e.rotate(pi/2) # no change in polar and section modulus
    assert e.section_modulus() == (pi*a**2*b/S(4), pi*a*b**2/S(4))
    assert e.polar_second_moment_of_area() == pi*a**3*b/S(4) + pi*a*b**3/S(4)

    e = Ellipse((a, b), 2, 6)
    assert e.section_modulus() == (18*pi, 6*pi)
    assert e.polar_second_moment_of_area() == 120*pi

    e = Ellipse(Point(0, 0), 2, 2)
    assert e.section_modulus() == (2*pi, 2*pi)
    assert e.section_modulus(Point(2, 2)) == (2*pi, 2*pi)
    assert e.section_modulus((2, 2)) == (2*pi, 2*pi)


def test_circumference():
    M = Symbol('M')
    m = Symbol('m')
    assert Ellipse(Point(0, 0), M, m).circumference == 4 * M * elliptic_e((M ** 2 - m ** 2) / M**2)

    assert Ellipse(Point(0, 0), 5, 4).circumference == 20 * elliptic_e(S(9) / 25)

    # circle
    assert Ellipse(None, 1, None, 0).circumference == 2*pi

    # test numerically
    assert abs(Ellipse(None, hradius=5, vradius=3).circumference.evalf(16) - 25.52699886339813) < 1e-10


def test_issue_15259():
    assert Circle((1, 2), 0) == Point(1, 2)


def test_issue_15797_equals():
    Ri = 0.024127189424130748
    Ci = (0.0864931002830291, 0.0819863295239654)
    A = Point(0, 0.0578591400998346)
    c = Circle(Ci, Ri)  # evaluated
    assert c.is_tangent(c.tangent_lines(A)[0]) == True
    assert c.center.x.is_Rational
    assert c.center.y.is_Rational
    assert c.radius.is_Rational
    u = Circle(Ci, Ri, evaluate=False)  # unevaluated
    assert u.center.x.is_Float
    assert u.center.y.is_Float
    assert u.radius.is_Float


def test_auxiliary_circle():
    x, y, a, b = symbols('x y a b')
    e = Ellipse((x, y), a, b)
    # the general result
    assert e.auxiliary_circle() == Circle((x, y), Max(a, b))
    # a special case where Ellipse is a Circle
    assert Circle((3, 4), 8).auxiliary_circle() == Circle((3, 4), 8)


def test_director_circle():
    x, y, a, b = symbols('x y a b')
    e = Ellipse((x, y), a, b)
    # the general result
    assert e.director_circle() == Circle((x, y), sqrt(a**2 + b**2))
    # a special case where Ellipse is a Circle
    assert Circle((3, 4), 8).director_circle() == Circle((3, 4), 8*sqrt(2))


def test_evolute():
    #ellipse centered at h,k
    x, y, h, k = symbols('x y h k',real = True)
    a, b = symbols('a b')
    e = Ellipse(Point(h, k), a, b)
    t1 = (e.hradius*(x - e.center.x))**Rational(2, 3)
    t2 = (e.vradius*(y - e.center.y))**Rational(2, 3)
    E = t1 + t2 - (e.hradius**2 - e.vradius**2)**Rational(2, 3)
    assert e.evolute() == E
    #Numerical Example
    e = Ellipse(Point(1, 1), 6, 3)
    t1 = (6*(x - 1))**Rational(2, 3)
    t2 = (3*(y - 1))**Rational(2, 3)
    E = t1 + t2 - (27)**Rational(2, 3)
    assert e.evolute() == E


def test_svg():
    e1 = Ellipse(Point(1, 0), 3, 2)
    assert e1._svg(2, "#FFAAFF") == '<ellipse fill="#FFAAFF" stroke="#555555" stroke-width="4.0" opacity="0.6" cx="1.00000000000000" cy="0" rx="3.00000000000000" ry="2.00000000000000"/>'
