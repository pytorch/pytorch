"""Utility functions for geometrical entities.

Contains
========
intersection
convex_hull
closest_points
farthest_points
are_coplanar
are_similar

"""

from collections import deque
from math import sqrt as _sqrt

from sympy import nsimplify
from .entity import GeometryEntity
from .exceptions import GeometryError
from .point import Point, Point2D, Point3D
from sympy.core.containers import OrderedSet
from sympy.core.exprtools import factor_terms
from sympy.core.function import Function, expand_mul
from sympy.core.numbers import Float
from sympy.core.sorting import ordered
from sympy.core.symbol import Symbol
from sympy.core.singleton import S
from sympy.polys.polytools import cancel
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.utilities.iterables import is_sequence

from mpmath.libmp.libmpf import prec_to_dps


def find(x, equation):
    """
    Checks whether a Symbol matching ``x`` is present in ``equation``
    or not. If present, the matching symbol is returned, else a
    ValueError is raised. If ``x`` is a string the matching symbol
    will have the same name; if ``x`` is a Symbol then it will be
    returned if found.

    Examples
    ========

    >>> from sympy.geometry.util import find
    >>> from sympy import Dummy
    >>> from sympy.abc import x
    >>> find('x', x)
    x
    >>> find('x', Dummy('x'))
    _x

    The dummy symbol is returned since it has a matching name:

    >>> _.name == 'x'
    True
    >>> find(x, Dummy('x'))
    Traceback (most recent call last):
    ...
    ValueError: could not find x
    """

    free = equation.free_symbols
    xs = [i for i in free if (i.name if isinstance(x, str) else i) == x]
    if not xs:
        raise ValueError('could not find %s' % x)
    if len(xs) != 1:
        raise ValueError('ambiguous %s' % x)
    return xs[0]


def _ordered_points(p):
    """Return the tuple of points sorted numerically according to args"""
    return tuple(sorted(p, key=lambda x: x.args))


def are_coplanar(*e):
    """ Returns True if the given entities are coplanar otherwise False

    Parameters
    ==========

    e: entities to be checked for being coplanar

    Returns
    =======

    Boolean

    Examples
    ========

    >>> from sympy import Point3D, Line3D
    >>> from sympy.geometry.util import are_coplanar
    >>> a = Line3D(Point3D(5, 0, 0), Point3D(1, -1, 1))
    >>> b = Line3D(Point3D(0, -2, 0), Point3D(3, 1, 1))
    >>> c = Line3D(Point3D(0, -1, 0), Point3D(5, -1, 9))
    >>> are_coplanar(a, b, c)
    False

    """
    from .line import LinearEntity3D
    from .plane import Plane
    # XXX update tests for coverage

    e = set(e)
    # first work with a Plane if present
    for i in list(e):
        if isinstance(i, Plane):
            e.remove(i)
            return all(p.is_coplanar(i) for p in e)

    if all(isinstance(i, Point3D) for i in e):
        if len(e) < 3:
            return False

        # remove pts that are collinear with 2 pts
        a, b = e.pop(), e.pop()
        for i in list(e):
            if Point3D.are_collinear(a, b, i):
                e.remove(i)

        if not e:
            return False
        else:
            # define a plane
            p = Plane(a, b, e.pop())
            for i in e:
                if i not in p:
                    return False
            return True
    else:
        pt3d = []
        for i in e:
            if isinstance(i, Point3D):
                pt3d.append(i)
            elif isinstance(i, LinearEntity3D):
                pt3d.extend(i.args)
            elif isinstance(i, GeometryEntity):  # XXX we should have a GeometryEntity3D class so we can tell the difference between 2D and 3D -- here we just want to deal with 2D objects; if new 3D objects are encountered that we didn't handle above, an error should be raised
                # all 2D objects have some Point that defines them; so convert those points to 3D pts by making z=0
                for p in i.args:
                    if isinstance(p, Point):
                        pt3d.append(Point3D(*(p.args + (0,))))
        return are_coplanar(*pt3d)


def are_similar(e1, e2):
    """Are two geometrical entities similar.

    Can one geometrical entity be uniformly scaled to the other?

    Parameters
    ==========

    e1 : GeometryEntity
    e2 : GeometryEntity

    Returns
    =======

    are_similar : boolean

    Raises
    ======

    GeometryError
        When `e1` and `e2` cannot be compared.

    Notes
    =====

    If the two objects are equal then they are similar.

    See Also
    ========

    sympy.geometry.entity.GeometryEntity.is_similar

    Examples
    ========

    >>> from sympy import Point, Circle, Triangle, are_similar
    >>> c1, c2 = Circle(Point(0, 0), 4), Circle(Point(1, 4), 3)
    >>> t1 = Triangle(Point(0, 0), Point(1, 0), Point(0, 1))
    >>> t2 = Triangle(Point(0, 0), Point(2, 0), Point(0, 2))
    >>> t3 = Triangle(Point(0, 0), Point(3, 0), Point(0, 1))
    >>> are_similar(t1, t2)
    True
    >>> are_similar(t1, t3)
    False

    """
    if e1 == e2:
        return True
    is_similar1 = getattr(e1, 'is_similar', None)
    if is_similar1:
        return is_similar1(e2)
    is_similar2 = getattr(e2, 'is_similar', None)
    if is_similar2:
        return is_similar2(e1)
    n1 = e1.__class__.__name__
    n2 = e2.__class__.__name__
    raise GeometryError(
        "Cannot test similarity between %s and %s" % (n1, n2))


def centroid(*args):
    """Find the centroid (center of mass) of the collection containing only Points,
    Segments or Polygons. The centroid is the weighted average of the individual centroid
    where the weights are the lengths (of segments) or areas (of polygons).
    Overlapping regions will add to the weight of that region.

    If there are no objects (or a mixture of objects) then None is returned.

    See Also
    ========

    sympy.geometry.point.Point, sympy.geometry.line.Segment,
    sympy.geometry.polygon.Polygon

    Examples
    ========

    >>> from sympy import Point, Segment, Polygon
    >>> from sympy.geometry.util import centroid
    >>> p = Polygon((0, 0), (10, 0), (10, 10))
    >>> q = p.translate(0, 20)
    >>> p.centroid, q.centroid
    (Point2D(20/3, 10/3), Point2D(20/3, 70/3))
    >>> centroid(p, q)
    Point2D(20/3, 40/3)
    >>> p, q = Segment((0, 0), (2, 0)), Segment((0, 0), (2, 2))
    >>> centroid(p, q)
    Point2D(1, 2 - sqrt(2))
    >>> centroid(Point(0, 0), Point(2, 0))
    Point2D(1, 0)

    Stacking 3 polygons on top of each other effectively triples the
    weight of that polygon:

    >>> p = Polygon((0, 0), (1, 0), (1, 1), (0, 1))
    >>> q = Polygon((1, 0), (3, 0), (3, 1), (1, 1))
    >>> centroid(p, q)
    Point2D(3/2, 1/2)
    >>> centroid(p, p, p, q) # centroid x-coord shifts left
    Point2D(11/10, 1/2)

    Stacking the squares vertically above and below p has the same
    effect:

    >>> centroid(p, p.translate(0, 1), p.translate(0, -1), q)
    Point2D(11/10, 1/2)

    """
    from .line import Segment
    from .polygon import Polygon
    if args:
        if all(isinstance(g, Point) for g in args):
            c = Point(0, 0)
            for g in args:
                c += g
            den = len(args)
        elif all(isinstance(g, Segment) for g in args):
            c = Point(0, 0)
            L = 0
            for g in args:
                l = g.length
                c += g.midpoint*l
                L += l
            den = L
        elif all(isinstance(g, Polygon) for g in args):
            c = Point(0, 0)
            A = 0
            for g in args:
                a = g.area
                c += g.centroid*a
                A += a
            den = A
        c /= den
        return c.func(*[i.simplify() for i in c.args])


def closest_points(*args):
    """Return the subset of points from a set of points that were
    the closest to each other in the 2D plane.

    Parameters
    ==========

    args
        A collection of Points on 2D plane.

    Notes
    =====

    This can only be performed on a set of points whose coordinates can
    be ordered on the number line. If there are no ties then a single
    pair of Points will be in the set.

    Examples
    ========

    >>> from sympy import closest_points, Triangle
    >>> Triangle(sss=(3, 4, 5)).args
    (Point2D(0, 0), Point2D(3, 0), Point2D(3, 4))
    >>> closest_points(*_)
    {(Point2D(0, 0), Point2D(3, 0))}

    References
    ==========

    .. [1] https://www.cs.mcgill.ca/~cs251/ClosestPair/ClosestPairPS.html

    .. [2] Sweep line algorithm
        https://en.wikipedia.org/wiki/Sweep_line_algorithm

    """
    p = [Point2D(i) for i in set(args)]
    if len(p) < 2:
        raise ValueError('At least 2 distinct points must be given.')

    try:
        p.sort(key=lambda x: x.args)
    except TypeError:
        raise ValueError("The points could not be sorted.")

    if not all(i.is_Rational for j in p for i in j.args):
        def hypot(x, y):
            arg = x*x + y*y
            if arg.is_Rational:
                return _sqrt(arg)
            return sqrt(arg)
    else:
        from math import hypot

    rv = [(0, 1)]
    best_dist = hypot(p[1].x - p[0].x, p[1].y - p[0].y)
    left = 0
    box = deque([0, 1])
    for i in range(2, len(p)):
        while left < i and p[i][0] - p[left][0] > best_dist:
            box.popleft()
            left += 1

        for j in box:
            d = hypot(p[i].x - p[j].x, p[i].y - p[j].y)
            if d < best_dist:
                rv = [(j, i)]
            elif d == best_dist:
                rv.append((j, i))
            else:
                continue
            best_dist = d
        box.append(i)

    return {tuple([p[i] for i in pair]) for pair in rv}


def convex_hull(*args, polygon=True):
    """The convex hull surrounding the Points contained in the list of entities.

    Parameters
    ==========

    args : a collection of Points, Segments and/or Polygons

    Optional parameters
    ===================

    polygon : Boolean. If True, returns a Polygon, if false a tuple, see below.
              Default is True.

    Returns
    =======

    convex_hull : Polygon if ``polygon`` is True else as a tuple `(U, L)` where
                  ``L`` and ``U`` are the lower and upper hulls, respectively.

    Notes
    =====

    This can only be performed on a set of points whose coordinates can
    be ordered on the number line.

    See Also
    ========

    sympy.geometry.point.Point, sympy.geometry.polygon.Polygon

    Examples
    ========

    >>> from sympy import convex_hull
    >>> points = [(1, 1), (1, 2), (3, 1), (-5, 2), (15, 4)]
    >>> convex_hull(*points)
    Polygon(Point2D(-5, 2), Point2D(1, 1), Point2D(3, 1), Point2D(15, 4))
    >>> convex_hull(*points, **dict(polygon=False))
    ([Point2D(-5, 2), Point2D(15, 4)],
     [Point2D(-5, 2), Point2D(1, 1), Point2D(3, 1), Point2D(15, 4)])

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Graham_scan

    .. [2] Andrew's Monotone Chain Algorithm
      (A.M. Andrew,
      "Another Efficient Algorithm for Convex Hulls in Two Dimensions", 1979)
      https://web.archive.org/web/20210511015444/http://geomalgorithms.com/a10-_hull-1.html

    """
    from .line import Segment
    from .polygon import Polygon
    p = OrderedSet()
    for e in args:
        if not isinstance(e, GeometryEntity):
            try:
                e = Point(e)
            except NotImplementedError:
                raise ValueError('%s is not a GeometryEntity and cannot be made into Point' % str(e))
        if isinstance(e, Point):
            p.add(e)
        elif isinstance(e, Segment):
            p.update(e.points)
        elif isinstance(e, Polygon):
            p.update(e.vertices)
        else:
            raise NotImplementedError(
                'Convex hull for %s not implemented.' % type(e))

    # make sure all our points are of the same dimension
    if any(len(x) != 2 for x in p):
        raise ValueError('Can only compute the convex hull in two dimensions')

    p = list(p)
    if len(p) == 1:
        return p[0] if polygon else (p[0], None)
    elif len(p) == 2:
        s = Segment(p[0], p[1])
        return s if polygon else (s, None)

    def _orientation(p, q, r):
        '''Return positive if p-q-r are clockwise, neg if ccw, zero if
        collinear.'''
        return (q.y - p.y)*(r.x - p.x) - (q.x - p.x)*(r.y - p.y)

    # scan to find upper and lower convex hulls of a set of 2d points.
    U = []
    L = []
    try:
        p.sort(key=lambda x: x.args)
    except TypeError:
        raise ValueError("The points could not be sorted.")
    for p_i in p:
        while len(U) > 1 and _orientation(U[-2], U[-1], p_i) <= 0:
            U.pop()
        while len(L) > 1 and _orientation(L[-2], L[-1], p_i) >= 0:
            L.pop()
        U.append(p_i)
        L.append(p_i)
    U.reverse()
    convexHull = tuple(L + U[1:-1])

    if len(convexHull) == 2:
        s = Segment(convexHull[0], convexHull[1])
        return s if polygon else (s, None)
    if polygon:
        return Polygon(*convexHull)
    else:
        U.reverse()
        return (U, L)

def farthest_points(*args):
    """Return the subset of points from a set of points that were
    the furthest apart from each other in the 2D plane.

    Parameters
    ==========

    args
        A collection of Points on 2D plane.

    Notes
    =====

    This can only be performed on a set of points whose coordinates can
    be ordered on the number line. If there are no ties then a single
    pair of Points will be in the set.

    Examples
    ========

    >>> from sympy.geometry import farthest_points, Triangle
    >>> Triangle(sss=(3, 4, 5)).args
    (Point2D(0, 0), Point2D(3, 0), Point2D(3, 4))
    >>> farthest_points(*_)
    {(Point2D(0, 0), Point2D(3, 4))}

    References
    ==========

    .. [1] https://code.activestate.com/recipes/117225-convex-hull-and-diameter-of-2d-point-sets/

    .. [2] Rotating Callipers Technique
        https://en.wikipedia.org/wiki/Rotating_calipers

    """

    def rotatingCalipers(Points):
        U, L = convex_hull(*Points, **{"polygon": False})

        if L is None:
            if isinstance(U, Point):
                raise ValueError('At least two distinct points must be given.')
            yield U.args
        else:
            i = 0
            j = len(L) - 1
            while i < len(U) - 1 or j > 0:
                yield U[i], L[j]
                # if all the way through one side of hull, advance the other side
                if i == len(U) - 1:
                    j -= 1
                elif j == 0:
                    i += 1
                # still points left on both lists, compare slopes of next hull edges
                # being careful to avoid divide-by-zero in slope calculation
                elif (U[i+1].y - U[i].y) * (L[j].x - L[j-1].x) > \
                        (L[j].y - L[j-1].y) * (U[i+1].x - U[i].x):
                    i += 1
                else:
                    j -= 1

    p = [Point2D(i) for i in set(args)]

    if not all(i.is_Rational for j in p for i in j.args):
        def hypot(x, y):
            arg = x*x + y*y
            if arg.is_Rational:
                return _sqrt(arg)
            return sqrt(arg)
    else:
        from math import hypot

    rv = []
    diam = 0
    for pair in rotatingCalipers(args):
        h, q = _ordered_points(pair)
        d = hypot(h.x - q.x, h.y - q.y)
        if d > diam:
            rv = [(h, q)]
        elif d == diam:
            rv.append((h, q))
        else:
            continue
        diam = d

    return set(rv)


def idiff(eq, y, x, n=1):
    """Return ``dy/dx`` assuming that ``eq == 0``.

    Parameters
    ==========

    y : the dependent variable or a list of dependent variables (with y first)
    x : the variable that the derivative is being taken with respect to
    n : the order of the derivative (default is 1)

    Examples
    ========

    >>> from sympy.abc import x, y, a
    >>> from sympy.geometry.util import idiff

    >>> circ = x**2 + y**2 - 4
    >>> idiff(circ, y, x)
    -x/y
    >>> idiff(circ, y, x, 2).simplify()
    (-x**2 - y**2)/y**3

    Here, ``a`` is assumed to be independent of ``x``:

    >>> idiff(x + a + y, y, x)
    -1

    Now the x-dependence of ``a`` is made explicit by listing ``a`` after
    ``y`` in a list.

    >>> idiff(x + a + y, [y, a], x)
    -Derivative(a, x) - 1

    See Also
    ========

    sympy.core.function.Derivative: represents unevaluated derivatives
    sympy.core.function.diff: explicitly differentiates wrt symbols

    """
    if is_sequence(y):
        dep = set(y)
        y = y[0]
    elif isinstance(y, Symbol):
        dep = {y}
    elif isinstance(y, Function):
        pass
    else:
        raise ValueError("expecting x-dependent symbol(s) or function(s) but got: %s" % y)

    f = {s: Function(s.name)(x) for s in eq.free_symbols
        if s != x and s in dep}

    if isinstance(y, Symbol):
        dydx = Function(y.name)(x).diff(x)
    else:
        dydx = y.diff(x)

    eq = eq.subs(f)
    derivs = {}
    for i in range(n):
        # equation will be linear in dydx, a*dydx + b, so dydx = -b/a
        deq = eq.diff(x)
        b = deq.xreplace({dydx: S.Zero})
        a = (deq - b).xreplace({dydx: S.One})
        yp = factor_terms(expand_mul(cancel((-b/a).subs(derivs)), deep=False))
        if i == n - 1:
            return yp.subs([(v, k) for k, v in f.items()])
        derivs[dydx] = yp
        eq = dydx - yp
        dydx = dydx.diff(x)


def intersection(*entities, pairwise=False, **kwargs):
    """The intersection of a collection of GeometryEntity instances.

    Parameters
    ==========
    entities : sequence of GeometryEntity
    pairwise (keyword argument) : Can be either True or False

    Returns
    =======
    intersection : list of GeometryEntity

    Raises
    ======
    NotImplementedError
        When unable to calculate intersection.

    Notes
    =====
    The intersection of any geometrical entity with itself should return
    a list with one item: the entity in question.
    An intersection requires two or more entities. If only a single
    entity is given then the function will return an empty list.
    It is possible for `intersection` to miss intersections that one
    knows exists because the required quantities were not fully
    simplified internally.
    Reals should be converted to Rationals, e.g. Rational(str(real_num))
    or else failures due to floating point issues may result.

    Case 1: When the keyword argument 'pairwise' is False (default value):
    In this case, the function returns a list of intersections common to
    all entities.

    Case 2: When the keyword argument 'pairwise' is True:
    In this case, the functions returns a list intersections that occur
    between any pair of entities.

    See Also
    ========

    sympy.geometry.entity.GeometryEntity.intersection

    Examples
    ========

    >>> from sympy import Ray, Circle, intersection
    >>> c = Circle((0, 1), 1)
    >>> intersection(c, c.center)
    []
    >>> right = Ray((0, 0), (1, 0))
    >>> up = Ray((0, 0), (0, 1))
    >>> intersection(c, right, up)
    [Point2D(0, 0)]
    >>> intersection(c, right, up, pairwise=True)
    [Point2D(0, 0), Point2D(0, 2)]
    >>> left = Ray((1, 0), (0, 0))
    >>> intersection(right, left)
    [Segment2D(Point2D(0, 0), Point2D(1, 0))]

    """
    if len(entities) <= 1:
        return []

    entities = list(entities)
    prec = None
    for i, e in enumerate(entities):
        if not isinstance(e, GeometryEntity):
            # entities may be an immutable tuple
            e = Point(e)
        # convert to exact Rationals
        d = {}
        for f in e.atoms(Float):
            prec = f._prec if prec is None else min(f._prec, prec)
            d.setdefault(f, nsimplify(f, rational=True))
        entities[i] = e.xreplace(d)

    if not pairwise:
        # find the intersection common to all objects
        res = entities[0].intersection(entities[1])
        for entity in entities[2:]:
            newres = []
            for x in res:
                newres.extend(x.intersection(entity))
            res = newres
    else:
        # find all pairwise intersections
        ans = []
        for j in range(len(entities)):
            for k in range(j + 1, len(entities)):
                ans.extend(intersection(entities[j], entities[k]))
        res = list(ordered(set(ans)))

    # convert back to Floats
    if prec is not None:
        p = prec_to_dps(prec)
        res = [i.n(p) for i in res]
    return res
