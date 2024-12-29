from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.geometry import Circle, Line, Point, Polygon, Segment
from sympy.sets import FiniteSet, Union, Intersection, EmptySet


def test_booleans():
    """ test basic unions and intersections """
    half = S.Half

    p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
    p5, p6, p7 = map(Point, [(3, 2), (1, -1), (0, 2)])
    l1 = Line(Point(0,0), Point(1,1))
    l2 = Line(Point(half, half), Point(5,5))
    l3 = Line(p2, p3)
    l4 = Line(p3, p4)
    poly1 = Polygon(p1, p2, p3, p4)
    poly2 = Polygon(p5, p6, p7)
    poly3 = Polygon(p1, p2, p5)
    assert Union(l1, l2).equals(l1)
    assert Intersection(l1, l2).equals(l1)
    assert Intersection(l1, l4) == FiniteSet(Point(1,1))
    assert Intersection(Union(l1, l4), l3) == FiniteSet(Point(Rational(-1, 3), Rational(-1, 3)), Point(5, 1))
    assert Intersection(l1, FiniteSet(Point(7,-7))) == EmptySet
    assert Intersection(Circle(Point(0,0), 3), Line(p1,p2)) == FiniteSet(Point(-3,0), Point(3,0))
    assert Intersection(l1, FiniteSet(p1)) == FiniteSet(p1)
    assert Union(l1, FiniteSet(p1)) == l1

    fs = FiniteSet(Point(Rational(1, 3), 1), Point(Rational(2, 3), 0), Point(Rational(9, 5), Rational(1, 5)), Point(Rational(7, 3), 1))
    # test the intersection of polygons
    assert Intersection(poly1, poly2) == fs
    # make sure if we union polygons with subsets, the subsets go away
    assert Union(poly1, poly2, fs) == Union(poly1, poly2)
    # make sure that if we union with a FiniteSet that isn't a subset,
    # that the points in the intersection stop being listed
    assert Union(poly1, FiniteSet(Point(0,0), Point(3,5))) == Union(poly1, FiniteSet(Point(3,5)))
    # intersect two polygons that share an edge
    assert Intersection(poly1, poly3) == Union(FiniteSet(Point(Rational(3, 2), 1), Point(2, 1)), Segment(Point(0, 0), Point(1, 0)))
