from sympy.core import Expr, S, oo, pi, sympify
from sympy.core.evalf import N
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import _symbol, Dummy, Symbol
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cos, sin, tan
from .ellipse import Circle
from .entity import GeometryEntity, GeometrySet
from .exceptions import GeometryError
from .line import Line, Segment, Ray
from .point import Point
from sympy.logic import And
from sympy.matrices import Matrix
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import has_dups, has_variety, uniq, rotate_left, least_rotation
from sympy.utilities.misc import as_int, func_name

from mpmath.libmp.libmpf import prec_to_dps

import warnings


x, y, T = [Dummy('polygon_dummy', real=True) for i in range(3)]


class Polygon(GeometrySet):
    """A two-dimensional polygon.

    A simple polygon in space. Can be constructed from a sequence of points
    or from a center, radius, number of sides and rotation angle.

    Parameters
    ==========

    vertices
        A sequence of points.

    n : int, optional
        If $> 0$, an n-sided RegularPolygon is created.
        Default value is $0$.

    Attributes
    ==========

    area
    angles
    perimeter
    vertices
    centroid
    sides

    Raises
    ======

    GeometryError
        If all parameters are not Points.

    See Also
    ========

    sympy.geometry.point.Point, sympy.geometry.line.Segment, Triangle

    Notes
    =====

    Polygons are treated as closed paths rather than 2D areas so
    some calculations can be be negative or positive (e.g., area)
    based on the orientation of the points.

    Any consecutive identical points are reduced to a single point
    and any points collinear and between two points will be removed
    unless they are needed to define an explicit intersection (see examples).

    A Triangle, Segment or Point will be returned when there are 3 or
    fewer points provided.

    Examples
    ========

    >>> from sympy import Polygon, pi
    >>> p1, p2, p3, p4, p5 = [(0, 0), (1, 0), (5, 1), (0, 1), (3, 0)]
    >>> Polygon(p1, p2, p3, p4)
    Polygon(Point2D(0, 0), Point2D(1, 0), Point2D(5, 1), Point2D(0, 1))
    >>> Polygon(p1, p2)
    Segment2D(Point2D(0, 0), Point2D(1, 0))
    >>> Polygon(p1, p2, p5)
    Segment2D(Point2D(0, 0), Point2D(3, 0))

    The area of a polygon is calculated as positive when vertices are
    traversed in a ccw direction. When the sides of a polygon cross the
    area will have positive and negative contributions. The following
    defines a Z shape where the bottom right connects back to the top
    left.

    >>> Polygon((0, 2), (2, 2), (0, 0), (2, 0)).area
    0

    When the keyword `n` is used to define the number of sides of the
    Polygon then a RegularPolygon is created and the other arguments are
    interpreted as center, radius and rotation. The unrotated RegularPolygon
    will always have a vertex at Point(r, 0) where `r` is the radius of the
    circle that circumscribes the RegularPolygon. Its method `spin` can be
    used to increment that angle.

    >>> p = Polygon((0,0), 1, n=3)
    >>> p
    RegularPolygon(Point2D(0, 0), 1, 3, 0)
    >>> p.vertices[0]
    Point2D(1, 0)
    >>> p.args[0]
    Point2D(0, 0)
    >>> p.spin(pi/2)
    >>> p.vertices[0]
    Point2D(0, 1)

    """

    __slots__ = ()

    def __new__(cls, *args, n = 0, **kwargs):
        if n:
            args = list(args)
            # return a virtual polygon with n sides
            if len(args) == 2:  # center, radius
                args.append(n)
            elif len(args) == 3:  # center, radius, rotation
                args.insert(2, n)
            return RegularPolygon(*args, **kwargs)

        vertices = [Point(a, dim=2, **kwargs) for a in args]

        # remove consecutive duplicates
        nodup = []
        for p in vertices:
            if nodup and p == nodup[-1]:
                continue
            nodup.append(p)
        if len(nodup) > 1 and nodup[-1] == nodup[0]:
            nodup.pop()  # last point was same as first

        # remove collinear points
        i = -3
        while i < len(nodup) - 3 and len(nodup) > 2:
            a, b, c = nodup[i], nodup[i + 1], nodup[i + 2]
            if Point.is_collinear(a, b, c):
                nodup.pop(i + 1)
                if a == c:
                    nodup.pop(i)
            else:
                i += 1

        vertices = list(nodup)

        if len(vertices) > 3:
            return GeometryEntity.__new__(cls, *vertices, **kwargs)
        elif len(vertices) == 3:
            return Triangle(*vertices, **kwargs)
        elif len(vertices) == 2:
            return Segment(*vertices, **kwargs)
        else:
            return Point(*vertices, **kwargs)

    @property
    def area(self):
        """
        The area of the polygon.

        Notes
        =====

        The area calculation can be positive or negative based on the
        orientation of the points. If any side of the polygon crosses
        any other side, there will be areas having opposite signs.

        See Also
        ========

        sympy.geometry.ellipse.Ellipse.area

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.area
        3

        In the Z shaped polygon (with the lower right connecting back
        to the upper left) the areas cancel out:

        >>> Z = Polygon((0, 1), (1, 1), (0, 0), (1, 0))
        >>> Z.area
        0

        In the M shaped polygon, areas do not cancel because no side
        crosses any other (though there is a point of contact).

        >>> M = Polygon((0, 0), (0, 1), (2, 0), (3, 1), (3, 0))
        >>> M.area
        -3/2

        """
        area = 0
        args = self.args
        for i in range(len(args)):
            x1, y1 = args[i - 1].args
            x2, y2 = args[i].args
            area += x1*y2 - x2*y1
        return simplify(area) / 2

    @staticmethod
    def _is_clockwise(a, b, c):
        """Return True/False for cw/ccw orientation.

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> a, b, c = [Point(i) for i in [(0, 0), (1, 1), (1, 0)]]
        >>> Polygon._is_clockwise(a, b, c)
        True
        >>> Polygon._is_clockwise(a, c, b)
        False
        """
        ba = b - a
        ca = c - a
        t_area = simplify(ba.x*ca.y - ca.x*ba.y)
        res = t_area.is_nonpositive
        if res is None:
            raise ValueError("Can't determine orientation")
        return res

    @property
    def angles(self):
        """The internal angle at each vertex.

        Returns
        =======

        angles : dict
            A dictionary where each key is a vertex and each value is the
            internal angle at that vertex. The vertices are represented as
            Points.

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.LinearEntity.angle_between

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.angles[p1]
        pi/2
        >>> poly.angles[p2]
        acos(-4*sqrt(17)/17)

        """

        args = self.vertices
        n = len(args)
        ret = {}
        for i in range(n):
            a, b, c = args[i - 2], args[i - 1], args[i]
            reflex_ang = Ray(b, a).angle_between(Ray(b, c))
            if self._is_clockwise(a, b, c):
                ret[b] = 2*S.Pi - reflex_ang
            else:
                ret[b] = reflex_ang

        # internal sum should be pi*(n - 2), not pi*(n+2)
        # so if ratio is (n+2)/(n-2) > 1 it is wrong
        wrong = ((sum(ret.values())/S.Pi-1)/(n - 2) - 1).is_positive
        if wrong:
            two_pi = 2*S.Pi
            for b in ret:
                ret[b] = two_pi - ret[b]
        elif wrong is None:
            raise ValueError("could not determine Polygon orientation.")
        return ret

    @property
    def ambient_dimension(self):
        return self.vertices[0].ambient_dimension

    @property
    def perimeter(self):
        """The perimeter of the polygon.

        Returns
        =======

        perimeter : number or Basic instance

        See Also
        ========

        sympy.geometry.line.Segment.length

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.perimeter
        sqrt(17) + 7
        """
        p = 0
        args = self.vertices
        for i in range(len(args)):
            p += args[i - 1].distance(args[i])
        return simplify(p)

    @property
    def vertices(self):
        """The vertices of the polygon.

        Returns
        =======

        vertices : list of Points

        Notes
        =====

        When iterating over the vertices, it is more efficient to index self
        rather than to request the vertices and index them. Only use the
        vertices when you want to process all of them at once. This is even
        more important with RegularPolygons that calculate each vertex.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.vertices
        [Point2D(0, 0), Point2D(1, 0), Point2D(5, 1), Point2D(0, 1)]
        >>> poly.vertices[0]
        Point2D(0, 0)

        """
        return list(self.args)

    @property
    def centroid(self):
        """The centroid of the polygon.

        Returns
        =======

        centroid : Point

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.util.centroid

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.centroid
        Point2D(31/18, 11/18)

        """
        A = 1/(6*self.area)
        cx, cy = 0, 0
        args = self.args
        for i in range(len(args)):
            x1, y1 = args[i - 1].args
            x2, y2 = args[i].args
            v = x1*y2 - x2*y1
            cx += v*(x1 + x2)
            cy += v*(y1 + y2)
        return Point(simplify(A*cx), simplify(A*cy))


    def second_moment_of_area(self, point=None):
        """Returns the second moment and product moment of area of a two dimensional polygon.

        Parameters
        ==========

        point : Point, two-tuple of sympifyable objects, or None(default=None)
            point is the point about which second moment of area is to be found.
            If "point=None" it will be calculated about the axis passing through the
            centroid of the polygon.

        Returns
        =======

        I_xx, I_yy, I_xy : number or SymPy expression
                           I_xx, I_yy are second moment of area of a two dimensional polygon.
                           I_xy is product moment of area of a two dimensional polygon.

        Examples
        ========

        >>> from sympy import Polygon, symbols
        >>> a, b = symbols('a, b')
        >>> p1, p2, p3, p4, p5 = [(0, 0), (a, 0), (a, b), (0, b), (a/3, b/3)]
        >>> rectangle = Polygon(p1, p2, p3, p4)
        >>> rectangle.second_moment_of_area()
        (a*b**3/12, a**3*b/12, 0)
        >>> rectangle.second_moment_of_area(p5)
        (a*b**3/9, a**3*b/9, a**2*b**2/36)

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Second_moment_of_area

        """

        I_xx, I_yy, I_xy = 0, 0, 0
        args = self.vertices
        for i in range(len(args)):
            x1, y1 = args[i-1].args
            x2, y2 = args[i].args
            v = x1*y2 - x2*y1
            I_xx += (y1**2 + y1*y2 + y2**2)*v
            I_yy += (x1**2 + x1*x2 + x2**2)*v
            I_xy += (x1*y2 + 2*x1*y1 + 2*x2*y2 + x2*y1)*v
        A = self.area
        c_x = self.centroid[0]
        c_y = self.centroid[1]
        # parallel axis theorem
        I_xx_c = (I_xx/12) - (A*(c_y**2))
        I_yy_c = (I_yy/12) - (A*(c_x**2))
        I_xy_c = (I_xy/24) - (A*(c_x*c_y))
        if point is None:
            return I_xx_c, I_yy_c, I_xy_c

        I_xx = (I_xx_c + A*((point[1]-c_y)**2))
        I_yy = (I_yy_c + A*((point[0]-c_x)**2))
        I_xy = (I_xy_c + A*((point[0]-c_x)*(point[1]-c_y)))

        return I_xx, I_yy, I_xy


    def first_moment_of_area(self, point=None):
        """
        Returns the first moment of area of a two-dimensional polygon with
        respect to a certain point of interest.

        First moment of area is a measure of the distribution of the area
        of a polygon in relation to an axis. The first moment of area of
        the entire polygon about its own centroid is always zero. Therefore,
        here it is calculated for an area, above or below a certain point
        of interest, that makes up a smaller portion of the polygon. This
        area is bounded by the point of interest and the extreme end
        (top or bottom) of the polygon. The first moment for this area is
        is then determined about the centroidal axis of the initial polygon.

        References
        ==========

        .. [1] https://skyciv.com/docs/tutorials/section-tutorials/calculating-the-statical-or-first-moment-of-area-of-beam-sections/?cc=BMD
        .. [2] https://mechanicalc.com/reference/cross-sections

        Parameters
        ==========

        point: Point, two-tuple of sympifyable objects, or None (default=None)
            point is the point above or below which the area of interest lies
            If ``point=None`` then the centroid acts as the point of interest.

        Returns
        =======

        Q_x, Q_y: number or SymPy expressions
            Q_x is the first moment of area about the x-axis
            Q_y is the first moment of area about the y-axis
            A negative sign indicates that the section modulus is
            determined for a section below (or left of) the centroidal axis

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> a, b = 50, 10
        >>> p1, p2, p3, p4 = [(0, b), (0, 0), (a, 0), (a, b)]
        >>> p = Polygon(p1, p2, p3, p4)
        >>> p.first_moment_of_area()
        (625, 3125)
        >>> p.first_moment_of_area(point=Point(30, 7))
        (525, 3000)
        """
        if point:
            xc, yc = self.centroid
        else:
            point = self.centroid
            xc, yc = point

        h_line = Line(point, slope=0)
        v_line = Line(point, slope=S.Infinity)

        h_poly = self.cut_section(h_line)
        v_poly = self.cut_section(v_line)

        poly_1 = h_poly[0] if h_poly[0].area <= h_poly[1].area else h_poly[1]
        poly_2 = v_poly[0] if v_poly[0].area <= v_poly[1].area else v_poly[1]

        Q_x = (poly_1.centroid.y - yc)*poly_1.area
        Q_y = (poly_2.centroid.x - xc)*poly_2.area

        return Q_x, Q_y


    def polar_second_moment_of_area(self):
        """Returns the polar modulus of a two-dimensional polygon

        It is a constituent of the second moment of area, linked through
        the perpendicular axis theorem. While the planar second moment of
        area describes an object's resistance to deflection (bending) when
        subjected to a force applied to a plane parallel to the central
        axis, the polar second moment of area describes an object's
        resistance to deflection when subjected to a moment applied in a
        plane perpendicular to the object's central axis (i.e. parallel to
        the cross-section)

        Examples
        ========

        >>> from sympy import Polygon, symbols
        >>> a, b = symbols('a, b')
        >>> rectangle = Polygon((0, 0), (a, 0), (a, b), (0, b))
        >>> rectangle.polar_second_moment_of_area()
        a**3*b/12 + a*b**3/12

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Polar_moment_of_inertia

        """
        second_moment = self.second_moment_of_area()
        return second_moment[0] + second_moment[1]


    def section_modulus(self, point=None):
        """Returns a tuple with the section modulus of a two-dimensional
        polygon.

        Section modulus is a geometric property of a polygon defined as the
        ratio of second moment of area to the distance of the extreme end of
        the polygon from the centroidal axis.

        Parameters
        ==========

        point : Point, two-tuple of sympifyable objects, or None(default=None)
            point is the point at which section modulus is to be found.
            If "point=None" it will be calculated for the point farthest from the
            centroidal axis of the polygon.

        Returns
        =======

        S_x, S_y: numbers or SymPy expressions
                  S_x is the section modulus with respect to the x-axis
                  S_y is the section modulus with respect to the y-axis
                  A negative sign indicates that the section modulus is
                  determined for a point below the centroidal axis

        Examples
        ========

        >>> from sympy import symbols, Polygon, Point
        >>> a, b = symbols('a, b', positive=True)
        >>> rectangle = Polygon((0, 0), (a, 0), (a, b), (0, b))
        >>> rectangle.section_modulus()
        (a*b**2/6, a**2*b/6)
        >>> rectangle.section_modulus(Point(a/4, b/4))
        (-a*b**2/3, -a**2*b/3)

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Section_modulus

        """
        x_c, y_c = self.centroid
        if point is None:
            # taking x and y as maximum distances from centroid
            x_min, y_min, x_max, y_max = self.bounds
            y = max(y_c - y_min, y_max - y_c)
            x = max(x_c - x_min, x_max - x_c)
        else:
            # taking x and y as distances of the given point from the centroid
            y = point.y - y_c
            x = point.x - x_c

        second_moment= self.second_moment_of_area()
        S_x = second_moment[0]/y
        S_y = second_moment[1]/x

        return S_x, S_y


    @property
    def sides(self):
        """The directed line segments that form the sides of the polygon.

        Returns
        =======

        sides : list of sides
            Each side is a directed Segment.

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.sides
        [Segment2D(Point2D(0, 0), Point2D(1, 0)),
        Segment2D(Point2D(1, 0), Point2D(5, 1)),
        Segment2D(Point2D(5, 1), Point2D(0, 1)), Segment2D(Point2D(0, 1), Point2D(0, 0))]

        """
        res = []
        args = self.vertices
        for i in range(-len(args), 0):
            res.append(Segment(args[i], args[i + 1]))
        return res

    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.

        """

        verts = self.vertices
        xs = [p.x for p in verts]
        ys = [p.y for p in verts]
        return (min(xs), min(ys), max(xs), max(ys))

    def is_convex(self):
        """Is the polygon convex?

        A polygon is convex if all its interior angles are less than 180
        degrees and there are no intersections between sides.

        Returns
        =======

        is_convex : boolean
            True if this polygon is convex, False otherwise.

        See Also
        ========

        sympy.geometry.util.convex_hull

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly = Polygon(p1, p2, p3, p4)
        >>> poly.is_convex()
        True

        """
        # Determine orientation of points
        args = self.vertices
        cw = self._is_clockwise(args[-2], args[-1], args[0])
        for i in range(1, len(args)):
            if cw ^ self._is_clockwise(args[i - 2], args[i - 1], args[i]):
                return False
        # check for intersecting sides
        sides = self.sides
        for i, si in enumerate(sides):
            pts = si.args
            # exclude the sides connected to si
            for j in range(1 if i == len(sides) - 1 else 0, i - 1):
                sj = sides[j]
                if sj.p1 not in pts and sj.p2 not in pts:
                    hit = si.intersection(sj)
                    if hit:
                        return False
        return True

    def encloses_point(self, p):
        """
        Return True if p is enclosed by (is inside of) self.

        Notes
        =====

        Being on the border of self is considered False.

        Parameters
        ==========

        p : Point

        Returns
        =======

        encloses_point : True, False or None

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.ellipse.Ellipse.encloses_point

        Examples
        ========

        >>> from sympy import Polygon, Point
        >>> p = Polygon((0, 0), (4, 0), (4, 4))
        >>> p.encloses_point(Point(2, 1))
        True
        >>> p.encloses_point(Point(2, 2))
        False
        >>> p.encloses_point(Point(5, 5))
        False

        References
        ==========

        .. [1] https://paulbourke.net/geometry/polygonmesh/#insidepoly

        """
        p = Point(p, dim=2)
        if p in self.vertices or any(p in s for s in self.sides):
            return False

        # move to p, checking that the result is numeric
        lit = []
        for v in self.vertices:
            lit.append(v - p)  # the difference is simplified
            if lit[-1].free_symbols:
                return None

        poly = Polygon(*lit)

        # polygon closure is assumed in the following test but Polygon removes duplicate pts so
        # the last point has to be added so all sides are computed. Using Polygon.sides is
        # not good since Segments are unordered.
        args = poly.args
        indices = list(range(-len(args), 1))

        if poly.is_convex():
            orientation = None
            for i in indices:
                a = args[i]
                b = args[i + 1]
                test = ((-a.y)*(b.x - a.x) - (-a.x)*(b.y - a.y)).is_negative
                if orientation is None:
                    orientation = test
                elif test is not orientation:
                    return False
            return True

        hit_odd = False
        p1x, p1y = args[0].args
        for i in indices[1:]:
            p2x, p2y = args[i].args
            if 0 > min(p1y, p2y):
                if 0 <= max(p1y, p2y):
                    if 0 <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (-p1y)*(p2x - p1x)/(p2y - p1y) + p1x
                            if p1x == p2x or 0 <= xinters:
                                hit_odd = not hit_odd
            p1x, p1y = p2x, p2y
        return hit_odd

    def arbitrary_point(self, parameter='t'):
        """A parameterized point on the polygon.

        The parameter, varying from 0 to 1, assigns points to the position on
        the perimeter that is that fraction of the total perimeter. So the
        point evaluated at t=1/2 would return the point from the first vertex
        that is 1/2 way around the polygon.

        Parameters
        ==========

        parameter : str, optional
            Default value is 't'.

        Returns
        =======

        arbitrary_point : Point

        Raises
        ======

        ValueError
            When `parameter` already appears in the Polygon's definition.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Polygon, Symbol
        >>> t = Symbol('t', real=True)
        >>> tri = Polygon((0, 0), (1, 0), (1, 1))
        >>> p = tri.arbitrary_point('t')
        >>> perimeter = tri.perimeter
        >>> s1, s2 = [s.length for s in tri.sides[:2]]
        >>> p.subs(t, (s1 + s2/2)/perimeter)
        Point2D(1, 1/2)

        """
        t = _symbol(parameter, real=True)
        if t.name in (f.name for f in self.free_symbols):
            raise ValueError('Symbol %s already appears in object and cannot be used as a parameter.' % t.name)
        sides = []
        perimeter = self.perimeter
        perim_fraction_start = 0
        for s in self.sides:
            side_perim_fraction = s.length/perimeter
            perim_fraction_end = perim_fraction_start + side_perim_fraction
            pt = s.arbitrary_point(parameter).subs(
                t, (t - perim_fraction_start)/side_perim_fraction)
            sides.append(
                (pt, (And(perim_fraction_start <= t, t < perim_fraction_end))))
            perim_fraction_start = perim_fraction_end
        return Piecewise(*sides)

    def parameter_value(self, other, t):
        if not isinstance(other,GeometryEntity):
            other = Point(other, dim=self.ambient_dimension)
        if not isinstance(other,Point):
            raise ValueError("other must be a point")
        if other.free_symbols:
            raise NotImplementedError('non-numeric coordinates')
        unknown = False
        p = self.arbitrary_point(T)
        for pt, cond in p.args:
            sol = solve(pt - other, T, dict=True)
            if not sol:
                continue
            value = sol[0][T]
            if simplify(cond.subs(T, value)) == True:
                return {t: value}
            unknown = True
        if unknown:
            raise ValueError("Given point may not be on %s" % func_name(self))
        raise ValueError("Given point is not on %s" % func_name(self))

    def plot_interval(self, parameter='t'):
        """The plot interval for the default geometric plot of the polygon.

        Parameters
        ==========

        parameter : str, optional
            Default value is 't'.

        Returns
        =======

        plot_interval : list (plot interval)
            [parameter, lower_bound, upper_bound]

        Examples
        ========

        >>> from sympy import Polygon
        >>> p = Polygon((0, 0), (1, 0), (1, 1))
        >>> p.plot_interval()
        [t, 0, 1]

        """
        t = Symbol(parameter, real=True)
        return [t, 0, 1]

    def intersection(self, o):
        """The intersection of polygon and geometry entity.

        The intersection may be empty and can contain individual Points and
        complete Line Segments.

        Parameters
        ==========

        other: GeometryEntity

        Returns
        =======

        intersection : list
            The list of Segments and Points

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment

        Examples
        ========

        >>> from sympy import Point, Polygon, Line
        >>> p1, p2, p3, p4 = map(Point, [(0, 0), (1, 0), (5, 1), (0, 1)])
        >>> poly1 = Polygon(p1, p2, p3, p4)
        >>> p5, p6, p7 = map(Point, [(3, 2), (1, -1), (0, 2)])
        >>> poly2 = Polygon(p5, p6, p7)
        >>> poly1.intersection(poly2)
        [Point2D(1/3, 1), Point2D(2/3, 0), Point2D(9/5, 1/5), Point2D(7/3, 1)]
        >>> poly1.intersection(Line(p1, p2))
        [Segment2D(Point2D(0, 0), Point2D(1, 0))]
        >>> poly1.intersection(p1)
        [Point2D(0, 0)]
        """
        intersection_result = []
        k = o.sides if isinstance(o, Polygon) else [o]
        for side in self.sides:
            for side1 in k:
                intersection_result.extend(side.intersection(side1))

        intersection_result = list(uniq(intersection_result))
        points = [entity for entity in intersection_result if isinstance(entity, Point)]
        segments = [entity for entity in intersection_result if isinstance(entity, Segment)]

        if points and segments:
            points_in_segments = list(uniq([point for point in points for segment in segments if point in segment]))
            if points_in_segments:
                for i in points_in_segments:
                    points.remove(i)
            return list(ordered(segments + points))
        else:
            return list(ordered(intersection_result))


    def cut_section(self, line):
        """
        Returns a tuple of two polygon segments that lie above and below
        the intersecting line respectively.

        Parameters
        ==========

        line: Line object of geometry module
            line which cuts the Polygon. The part of the Polygon that lies
            above and below this line is returned.

        Returns
        =======

        upper_polygon, lower_polygon: Polygon objects or None
            upper_polygon is the polygon that lies above the given line.
            lower_polygon is the polygon that lies below the given line.
            upper_polygon and lower polygon are ``None`` when no polygon
            exists above the line or below the line.

        Raises
        ======

        ValueError: When the line does not intersect the polygon

        Examples
        ========

        >>> from sympy import Polygon, Line
        >>> a, b = 20, 10
        >>> p1, p2, p3, p4 = [(0, b), (0, 0), (a, 0), (a, b)]
        >>> rectangle = Polygon(p1, p2, p3, p4)
        >>> t = rectangle.cut_section(Line((0, 5), slope=0))
        >>> t
        (Polygon(Point2D(0, 10), Point2D(0, 5), Point2D(20, 5), Point2D(20, 10)),
        Polygon(Point2D(0, 5), Point2D(0, 0), Point2D(20, 0), Point2D(20, 5)))
        >>> upper_segment, lower_segment = t
        >>> upper_segment.area
        100
        >>> upper_segment.centroid
        Point2D(10, 15/2)
        >>> lower_segment.centroid
        Point2D(10, 5/2)

        References
        ==========

        .. [1] https://github.com/sympy/sympy/wiki/A-method-to-return-a-cut-section-of-any-polygon-geometry

        """
        intersection_points = self.intersection(line)
        if not intersection_points:
            raise ValueError("This line does not intersect the polygon")

        points = list(self.vertices)
        points.append(points[0])

        eq = line.equation(x, y)

        # considering equation of line to be `ax +by + c`
        a = eq.coeff(x)
        b = eq.coeff(y)

        upper_vertices = []
        lower_vertices = []
        # prev is true when previous point is above the line
        prev = True
        prev_point = None
        for point in points:
            # when coefficient of y is 0, right side of the line is
            # considered
            compare = eq.subs({x: point.x, y: point.y})/b if b \
                    else eq.subs(x, point.x)/a

            # if point lies above line
            if compare > 0:
                if not prev:
                    # if previous point lies below the line, the intersection
                    # point of the polygon edge and the line has to be included
                    edge = Line(point, prev_point)
                    new_point = edge.intersection(line)
                    upper_vertices.append(new_point[0])
                    lower_vertices.append(new_point[0])

                upper_vertices.append(point)
                prev = True
            else:
                if prev and prev_point:
                    edge = Line(point, prev_point)
                    new_point = edge.intersection(line)
                    upper_vertices.append(new_point[0])
                    lower_vertices.append(new_point[0])
                lower_vertices.append(point)
                prev = False
            prev_point = point

        upper_polygon, lower_polygon = None, None
        if upper_vertices and isinstance(Polygon(*upper_vertices), Polygon):
            upper_polygon = Polygon(*upper_vertices)
        if lower_vertices and isinstance(Polygon(*lower_vertices), Polygon):
            lower_polygon = Polygon(*lower_vertices)

        return upper_polygon, lower_polygon


    def distance(self, o):
        """
        Returns the shortest distance between self and o.

        If o is a point, then self does not need to be convex.
        If o is another polygon self and o must be convex.

        Examples
        ========

        >>> from sympy import Point, Polygon, RegularPolygon
        >>> p1, p2 = map(Point, [(0, 0), (7, 5)])
        >>> poly = Polygon(*RegularPolygon(p1, 1, 3).vertices)
        >>> poly.distance(p2)
        sqrt(61)
        """
        if isinstance(o, Point):
            dist = oo
            for side in self.sides:
                current = side.distance(o)
                if current == 0:
                    return S.Zero
                elif current < dist:
                    dist = current
            return dist
        elif isinstance(o, Polygon) and self.is_convex() and o.is_convex():
            return self._do_poly_distance(o)
        raise NotImplementedError()

    def _do_poly_distance(self, e2):
        """
        Calculates the least distance between the exteriors of two
        convex polygons e1 and e2. Does not check for the convexity
        of the polygons as this is checked by Polygon.distance.

        Notes
        =====

            - Prints a warning if the two polygons possibly intersect as the return
              value will not be valid in such a case. For a more through test of
              intersection use intersection().

        See Also
        ========

        sympy.geometry.point.Point.distance

        Examples
        ========

        >>> from sympy import Point, Polygon
        >>> square = Polygon(Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0))
        >>> triangle = Polygon(Point(1, 2), Point(2, 2), Point(2, 1))
        >>> square._do_poly_distance(triangle)
        sqrt(2)/2

        Description of method used
        ==========================

        Method:
        [1] https://web.archive.org/web/20150509035744/http://cgm.cs.mcgill.ca/~orm/mind2p.html
        Uses rotating calipers:
        [2] https://en.wikipedia.org/wiki/Rotating_calipers
        and antipodal points:
        [3] https://en.wikipedia.org/wiki/Antipodal_point
        """
        e1 = self

        '''Tests for a possible intersection between the polygons and outputs a warning'''
        e1_center = e1.centroid
        e2_center = e2.centroid
        e1_max_radius = S.Zero
        e2_max_radius = S.Zero
        for vertex in e1.vertices:
            r = Point.distance(e1_center, vertex)
            if e1_max_radius < r:
                e1_max_radius = r
        for vertex in e2.vertices:
            r = Point.distance(e2_center, vertex)
            if e2_max_radius < r:
                e2_max_radius = r
        center_dist = Point.distance(e1_center, e2_center)
        if center_dist <= e1_max_radius + e2_max_radius:
            warnings.warn("Polygons may intersect producing erroneous output",
                          stacklevel=3)

        '''
        Find the upper rightmost vertex of e1 and the lowest leftmost vertex of e2
        '''
        e1_ymax = Point(0, -oo)
        e2_ymin = Point(0, oo)

        for vertex in e1.vertices:
            if vertex.y > e1_ymax.y or (vertex.y == e1_ymax.y and vertex.x > e1_ymax.x):
                e1_ymax = vertex
        for vertex in e2.vertices:
            if vertex.y < e2_ymin.y or (vertex.y == e2_ymin.y and vertex.x < e2_ymin.x):
                e2_ymin = vertex
        min_dist = Point.distance(e1_ymax, e2_ymin)

        '''
        Produce a dictionary with vertices of e1 as the keys and, for each vertex, the points
        to which the vertex is connected as its value. The same is then done for e2.
        '''
        e1_connections = {}
        e2_connections = {}

        for side in e1.sides:
            if side.p1 in e1_connections:
                e1_connections[side.p1].append(side.p2)
            else:
                e1_connections[side.p1] = [side.p2]

            if side.p2 in e1_connections:
                e1_connections[side.p2].append(side.p1)
            else:
                e1_connections[side.p2] = [side.p1]

        for side in e2.sides:
            if side.p1 in e2_connections:
                e2_connections[side.p1].append(side.p2)
            else:
                e2_connections[side.p1] = [side.p2]

            if side.p2 in e2_connections:
                e2_connections[side.p2].append(side.p1)
            else:
                e2_connections[side.p2] = [side.p1]

        e1_current = e1_ymax
        e2_current = e2_ymin
        support_line = Line(Point(S.Zero, S.Zero), Point(S.One, S.Zero))

        '''
        Determine which point in e1 and e2 will be selected after e2_ymin and e1_ymax,
        this information combined with the above produced dictionaries determines the
        path that will be taken around the polygons
        '''
        point1 = e1_connections[e1_ymax][0]
        point2 = e1_connections[e1_ymax][1]
        angle1 = support_line.angle_between(Line(e1_ymax, point1))
        angle2 = support_line.angle_between(Line(e1_ymax, point2))
        if angle1 < angle2:
            e1_next = point1
        elif angle2 < angle1:
            e1_next = point2
        elif Point.distance(e1_ymax, point1) > Point.distance(e1_ymax, point2):
            e1_next = point2
        else:
            e1_next = point1

        point1 = e2_connections[e2_ymin][0]
        point2 = e2_connections[e2_ymin][1]
        angle1 = support_line.angle_between(Line(e2_ymin, point1))
        angle2 = support_line.angle_between(Line(e2_ymin, point2))
        if angle1 > angle2:
            e2_next = point1
        elif angle2 > angle1:
            e2_next = point2
        elif Point.distance(e2_ymin, point1) > Point.distance(e2_ymin, point2):
            e2_next = point2
        else:
            e2_next = point1

        '''
        Loop which determines the distance between anti-podal pairs and updates the
        minimum distance accordingly. It repeats until it reaches the starting position.
        '''
        while True:
            e1_angle = support_line.angle_between(Line(e1_current, e1_next))
            e2_angle = pi - support_line.angle_between(Line(
                e2_current, e2_next))

            if (e1_angle < e2_angle) is True:
                support_line = Line(e1_current, e1_next)
                e1_segment = Segment(e1_current, e1_next)
                min_dist_current = e1_segment.distance(e2_current)

                if min_dist_current.evalf() < min_dist.evalf():
                    min_dist = min_dist_current

                if e1_connections[e1_next][0] != e1_current:
                    e1_current = e1_next
                    e1_next = e1_connections[e1_next][0]
                else:
                    e1_current = e1_next
                    e1_next = e1_connections[e1_next][1]
            elif (e1_angle > e2_angle) is True:
                support_line = Line(e2_next, e2_current)
                e2_segment = Segment(e2_current, e2_next)
                min_dist_current = e2_segment.distance(e1_current)

                if min_dist_current.evalf() < min_dist.evalf():
                    min_dist = min_dist_current

                if e2_connections[e2_next][0] != e2_current:
                    e2_current = e2_next
                    e2_next = e2_connections[e2_next][0]
                else:
                    e2_current = e2_next
                    e2_next = e2_connections[e2_next][1]
            else:
                support_line = Line(e1_current, e1_next)
                e1_segment = Segment(e1_current, e1_next)
                e2_segment = Segment(e2_current, e2_next)
                min1 = e1_segment.distance(e2_next)
                min2 = e2_segment.distance(e1_next)

                min_dist_current = min(min1, min2)
                if min_dist_current.evalf() < min_dist.evalf():
                    min_dist = min_dist_current

                if e1_connections[e1_next][0] != e1_current:
                    e1_current = e1_next
                    e1_next = e1_connections[e1_next][0]
                else:
                    e1_current = e1_next
                    e1_next = e1_connections[e1_next][1]

                if e2_connections[e2_next][0] != e2_current:
                    e2_current = e2_next
                    e2_next = e2_connections[e2_next][0]
                else:
                    e2_current = e2_next
                    e2_next = e2_connections[e2_next][1]
            if e1_current == e1_ymax and e2_current == e2_ymin:
                break
        return min_dist

    def _svg(self, scale_factor=1., fill_color="#66cc99"):
        """Returns SVG path element for the Polygon.

        Parameters
        ==========

        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is "#66cc99".
        """
        verts = map(N, self.vertices)
        coords = ["{},{}".format(p.x, p.y) for p in verts]
        path = "M {} L {} z".format(coords[0], " L ".join(coords[1:]))
        return (
            '<path fill-rule="evenodd" fill="{2}" stroke="#555555" '
            'stroke-width="{0}" opacity="0.6" d="{1}" />'
            ).format(2. * scale_factor, path, fill_color)

    def _hashable_content(self):

        D = {}
        def ref_list(point_list):
            kee = {}
            for i, p in enumerate(ordered(set(point_list))):
                kee[p] = i
                D[i] = p
            return [kee[p] for p in point_list]

        S1 = ref_list(self.args)
        r_nor = rotate_left(S1, least_rotation(S1))
        S2 = ref_list(list(reversed(self.args)))
        r_rev = rotate_left(S2, least_rotation(S2))
        if r_nor < r_rev:
            r = r_nor
        else:
            r = r_rev
        canonical_args = [ D[order] for order in r ]
        return tuple(canonical_args)

    def __contains__(self, o):
        """
        Return True if o is contained within the boundary lines of self.altitudes

        Parameters
        ==========

        other : GeometryEntity

        Returns
        =======

        contained in : bool
            The points (and sides, if applicable) are contained in self.

        See Also
        ========

        sympy.geometry.entity.GeometryEntity.encloses

        Examples
        ========

        >>> from sympy import Line, Segment, Point
        >>> p = Point(0, 0)
        >>> q = Point(1, 1)
        >>> s = Segment(p, q*2)
        >>> l = Line(p, q)
        >>> p in q
        False
        >>> p in s
        True
        >>> q*3 in s
        False
        >>> s in l
        True

        """

        if isinstance(o, Polygon):
            return self == o
        elif isinstance(o, Segment):
            return any(o in s for s in self.sides)
        elif isinstance(o, Point):
            if o in self.vertices:
                return True
            for side in self.sides:
                if o in side:
                    return True

        return False

    def bisectors(p, prec=None):
        """Returns angle bisectors of a polygon. If prec is given
        then approximate the point defining the ray to that precision.

        The distance between the points defining the bisector ray is 1.

        Examples
        ========

        >>> from sympy import Polygon, Point
        >>> p = Polygon(Point(0, 0), Point(2, 0), Point(1, 1), Point(0, 3))
        >>> p.bisectors(2)
        {Point2D(0, 0): Ray2D(Point2D(0, 0), Point2D(0.71, 0.71)),
         Point2D(0, 3): Ray2D(Point2D(0, 3), Point2D(0.23, 2.0)),
         Point2D(1, 1): Ray2D(Point2D(1, 1), Point2D(0.19, 0.42)),
         Point2D(2, 0): Ray2D(Point2D(2, 0), Point2D(1.1, 0.38))}
        """
        b = {}
        pts = list(p.args)
        pts.append(pts[0])  # close it
        cw = Polygon._is_clockwise(*pts[:3])
        if cw:
            pts = list(reversed(pts))
        for v, a in p.angles.items():
            i = pts.index(v)
            p1, p2 = Point._normalize_dimension(pts[i], pts[i + 1])
            ray = Ray(p1, p2).rotate(a/2, v)
            dir = ray.direction
            ray = Ray(ray.p1, ray.p1 + dir/dir.distance((0, 0)))
            if prec is not None:
                ray = Ray(ray.p1, ray.p2.n(prec))
            b[v] = ray
        return b


class RegularPolygon(Polygon):
    """
    A regular polygon.

    Such a polygon has all internal angles equal and all sides the same length.

    Parameters
    ==========

    center : Point
    radius : number or Basic instance
        The distance from the center to a vertex
    n : int
        The number of sides

    Attributes
    ==========

    vertices
    center
    radius
    rotation
    apothem
    interior_angle
    exterior_angle
    circumcircle
    incircle
    angles

    Raises
    ======

    GeometryError
        If the `center` is not a Point, or the `radius` is not a number or Basic
        instance, or the number of sides, `n`, is less than three.

    Notes
    =====

    A RegularPolygon can be instantiated with Polygon with the kwarg n.

    Regular polygons are instantiated with a center, radius, number of sides
    and a rotation angle. Whereas the arguments of a Polygon are vertices, the
    vertices of the RegularPolygon must be obtained with the vertices method.

    See Also
    ========

    sympy.geometry.point.Point, Polygon

    Examples
    ========

    >>> from sympy import RegularPolygon, Point
    >>> r = RegularPolygon(Point(0, 0), 5, 3)
    >>> r
    RegularPolygon(Point2D(0, 0), 5, 3, 0)
    >>> r.vertices[0]
    Point2D(5, 0)

    """

    __slots__ = ('_n', '_center', '_radius', '_rot')

    def __new__(self, c, r, n, rot=0, **kwargs):
        r, n, rot = map(sympify, (r, n, rot))
        c = Point(c, dim=2, **kwargs)
        if not isinstance(r, Expr):
            raise GeometryError("r must be an Expr object, not %s" % r)
        if n.is_Number:
            as_int(n)  # let an error raise if necessary
            if n < 3:
                raise GeometryError("n must be a >= 3, not %s" % n)

        obj = GeometryEntity.__new__(self, c, r, n, **kwargs)
        obj._n = n
        obj._center = c
        obj._radius = r
        obj._rot = rot % (2*S.Pi/n) if rot.is_number else rot
        return obj

    def _eval_evalf(self, prec=15, **options):
        c, r, n, a = self.args
        dps = prec_to_dps(prec)
        c, r, a = [i.evalf(n=dps, **options) for i in (c, r, a)]
        return self.func(c, r, n, a)

    @property
    def args(self):
        """
        Returns the center point, the radius,
        the number of sides, and the orientation angle.

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> r = RegularPolygon(Point(0, 0), 5, 3)
        >>> r.args
        (Point2D(0, 0), 5, 3, 0)
        """
        return self._center, self._radius, self._n, self._rot

    def __str__(self):
        return 'RegularPolygon(%s, %s, %s, %s)' % tuple(self.args)

    def __repr__(self):
        return 'RegularPolygon(%s, %s, %s, %s)' % tuple(self.args)

    @property
    def area(self):
        """Returns the area.

        Examples
        ========

        >>> from sympy import RegularPolygon
        >>> square = RegularPolygon((0, 0), 1, 4)
        >>> square.area
        2
        >>> _ == square.length**2
        True
        """
        c, r, n, rot = self.args
        return sign(r)*n*self.length**2/(4*tan(pi/n))

    @property
    def length(self):
        """Returns the length of the sides.

        The half-length of the side and the apothem form two legs
        of a right triangle whose hypotenuse is the radius of the
        regular polygon.

        Examples
        ========

        >>> from sympy import RegularPolygon
        >>> from sympy import sqrt
        >>> s = square_in_unit_circle = RegularPolygon((0, 0), 1, 4)
        >>> s.length
        sqrt(2)
        >>> sqrt((_/2)**2 + s.apothem**2) == s.radius
        True

        """
        return self.radius*2*sin(pi/self._n)

    @property
    def center(self):
        """The center of the RegularPolygon

        This is also the center of the circumscribing circle.

        Returns
        =======

        center : Point

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.ellipse.Ellipse.center

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 5, 4)
        >>> rp.center
        Point2D(0, 0)
        """
        return self._center

    centroid = center

    @property
    def circumcenter(self):
        """
        Alias for center.

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 5, 4)
        >>> rp.circumcenter
        Point2D(0, 0)
        """
        return self.center

    @property
    def radius(self):
        """Radius of the RegularPolygon

        This is also the radius of the circumscribing circle.

        Returns
        =======

        radius : number or instance of Basic

        See Also
        ========

        sympy.geometry.line.Segment.length, sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.radius
        r

        """
        return self._radius

    @property
    def circumradius(self):
        """
        Alias for radius.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.circumradius
        r
        """
        return self.radius

    @property
    def rotation(self):
        """CCW angle by which the RegularPolygon is rotated

        Returns
        =======

        rotation : number or instance of Basic

        Examples
        ========

        >>> from sympy import pi
        >>> from sympy.abc import a
        >>> from sympy import RegularPolygon, Point
        >>> RegularPolygon(Point(0, 0), 3, 4, pi/4).rotation
        pi/4

        Numerical rotation angles are made canonical:

        >>> RegularPolygon(Point(0, 0), 3, 4, a).rotation
        a
        >>> RegularPolygon(Point(0, 0), 3, 4, pi).rotation
        0

        """
        return self._rot

    @property
    def apothem(self):
        """The inradius of the RegularPolygon.

        The apothem/inradius is the radius of the inscribed circle.

        Returns
        =======

        apothem : number or instance of Basic

        See Also
        ========

        sympy.geometry.line.Segment.length, sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.apothem
        sqrt(2)*r/2

        """
        return self.radius * cos(S.Pi/self._n)

    @property
    def inradius(self):
        """
        Alias for apothem.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import RegularPolygon, Point
        >>> radius = Symbol('r')
        >>> rp = RegularPolygon(Point(0, 0), radius, 4)
        >>> rp.inradius
        sqrt(2)*r/2
        """
        return self.apothem

    @property
    def interior_angle(self):
        """Measure of the interior angles.

        Returns
        =======

        interior_angle : number

        See Also
        ========

        sympy.geometry.line.LinearEntity.angle_between

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 8)
        >>> rp.interior_angle
        3*pi/4

        """
        return (self._n - 2)*S.Pi/self._n

    @property
    def exterior_angle(self):
        """Measure of the exterior angles.

        Returns
        =======

        exterior_angle : number

        See Also
        ========

        sympy.geometry.line.LinearEntity.angle_between

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 8)
        >>> rp.exterior_angle
        pi/4

        """
        return 2*S.Pi/self._n

    @property
    def circumcircle(self):
        """The circumcircle of the RegularPolygon.

        Returns
        =======

        circumcircle : Circle

        See Also
        ========

        circumcenter, sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 8)
        >>> rp.circumcircle
        Circle(Point2D(0, 0), 4)

        """
        return Circle(self.center, self.radius)

    @property
    def incircle(self):
        """The incircle of the RegularPolygon.

        Returns
        =======

        incircle : Circle

        See Also
        ========

        inradius, sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 4, 7)
        >>> rp.incircle
        Circle(Point2D(0, 0), 4*cos(pi/7))

        """
        return Circle(self.center, self.apothem)

    @property
    def angles(self):
        """
        Returns a dictionary with keys, the vertices of the Polygon,
        and values, the interior angle at each vertex.

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> r = RegularPolygon(Point(0, 0), 5, 3)
        >>> r.angles
        {Point2D(-5/2, -5*sqrt(3)/2): pi/3,
         Point2D(-5/2, 5*sqrt(3)/2): pi/3,
         Point2D(5, 0): pi/3}
        """
        ret = {}
        ang = self.interior_angle
        for v in self.vertices:
            ret[v] = ang
        return ret

    def encloses_point(self, p):
        """
        Return True if p is enclosed by (is inside of) self.

        Notes
        =====

        Being on the border of self is considered False.

        The general Polygon.encloses_point method is called only if
        a point is not within or beyond the incircle or circumcircle,
        respectively.

        Parameters
        ==========

        p : Point

        Returns
        =======

        encloses_point : True, False or None

        See Also
        ========

        sympy.geometry.ellipse.Ellipse.encloses_point

        Examples
        ========

        >>> from sympy import RegularPolygon, S, Point, Symbol
        >>> p = RegularPolygon((0, 0), 3, 4)
        >>> p.encloses_point(Point(0, 0))
        True
        >>> r, R = p.inradius, p.circumradius
        >>> p.encloses_point(Point((r + R)/2, 0))
        True
        >>> p.encloses_point(Point(R/2, R/2 + (R - r)/10))
        False
        >>> t = Symbol('t', real=True)
        >>> p.encloses_point(p.arbitrary_point().subs(t, S.Half))
        False
        >>> p.encloses_point(Point(5, 5))
        False

        """

        c = self.center
        d = Segment(c, p).length
        if d >= self.radius:
            return False
        elif d < self.inradius:
            return True
        else:
            # now enumerate the RegularPolygon like a general polygon.
            return Polygon.encloses_point(self, p)

    def spin(self, angle):
        """Increment *in place* the virtual Polygon's rotation by ccw angle.

        See also: rotate method which moves the center.

        >>> from sympy import Polygon, Point, pi
        >>> r = Polygon(Point(0,0), 1, n=3)
        >>> r.vertices[0]
        Point2D(1, 0)
        >>> r.spin(pi/6)
        >>> r.vertices[0]
        Point2D(sqrt(3)/2, 1/2)

        See Also
        ========

        rotation
        rotate : Creates a copy of the RegularPolygon rotated about a Point

        """
        self._rot += angle

    def rotate(self, angle, pt=None):
        """Override GeometryEntity.rotate to first rotate the RegularPolygon
        about its center.

        >>> from sympy import Point, RegularPolygon, pi
        >>> t = RegularPolygon(Point(1, 0), 1, 3)
        >>> t.vertices[0] # vertex on x-axis
        Point2D(2, 0)
        >>> t.rotate(pi/2).vertices[0] # vertex on y axis now
        Point2D(0, 2)

        See Also
        ========

        rotation
        spin : Rotates a RegularPolygon in place

        """

        r = type(self)(*self.args)  # need a copy or else changes are in-place
        r._rot += angle
        return GeometryEntity.rotate(r, angle, pt)

    def scale(self, x=1, y=1, pt=None):
        """Override GeometryEntity.scale since it is the radius that must be
        scaled (if x == y) or else a new Polygon must be returned.

        >>> from sympy import RegularPolygon

        Symmetric scaling returns a RegularPolygon:

        >>> RegularPolygon((0, 0), 1, 4).scale(2, 2)
        RegularPolygon(Point2D(0, 0), 2, 4, 0)

        Asymmetric scaling returns a kite as a Polygon:

        >>> RegularPolygon((0, 0), 1, 4).scale(2, 1)
        Polygon(Point2D(2, 0), Point2D(0, 1), Point2D(-2, 0), Point2D(0, -1))

        """
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        if x != y:
            return Polygon(*self.vertices).scale(x, y)
        c, r, n, rot = self.args
        r *= x
        return self.func(c, r, n, rot)

    def reflect(self, line):
        """Override GeometryEntity.reflect since this is not made of only
        points.

        Examples
        ========

        >>> from sympy import RegularPolygon, Line

        >>> RegularPolygon((0, 0), 1, 4).reflect(Line((0, 1), slope=-2))
        RegularPolygon(Point2D(4/5, 2/5), -1, 4, atan(4/3))

        """
        c, r, n, rot = self.args
        v = self.vertices[0]
        d = v - c
        cc = c.reflect(line)
        vv = v.reflect(line)
        dd = vv - cc
        # calculate rotation about the new center
        # which will align the vertices
        l1 = Ray((0, 0), dd)
        l2 = Ray((0, 0), d)
        ang = l1.closing_angle(l2)
        rot += ang
        # change sign of radius as point traversal is reversed
        return self.func(cc, -r, n, rot)

    @property
    def vertices(self):
        """The vertices of the RegularPolygon.

        Returns
        =======

        vertices : list
            Each vertex is a Point.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import RegularPolygon, Point
        >>> rp = RegularPolygon(Point(0, 0), 5, 4)
        >>> rp.vertices
        [Point2D(5, 0), Point2D(0, 5), Point2D(-5, 0), Point2D(0, -5)]

        """
        c = self._center
        r = abs(self._radius)
        rot = self._rot
        v = 2*S.Pi/self._n

        return [Point(c.x + r*cos(k*v + rot), c.y + r*sin(k*v + rot))
                for k in range(self._n)]

    def __eq__(self, o):
        if not isinstance(o, Polygon):
            return False
        elif not isinstance(o, RegularPolygon):
            return Polygon.__eq__(o, self)
        return self.args == o.args

    def __hash__(self):
        return super().__hash__()


class Triangle(Polygon):
    """
    A polygon with three vertices and three sides.

    Parameters
    ==========

    points : sequence of Points
    keyword: asa, sas, or sss to specify sides/angles of the triangle

    Attributes
    ==========

    vertices
    altitudes
    orthocenter
    circumcenter
    circumradius
    circumcircle
    inradius
    incircle
    exradii
    medians
    medial
    nine_point_circle

    Raises
    ======

    GeometryError
        If the number of vertices is not equal to three, or one of the vertices
        is not a Point, or a valid keyword is not given.

    See Also
    ========

    sympy.geometry.point.Point, Polygon

    Examples
    ========

    >>> from sympy import Triangle, Point
    >>> Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
    Triangle(Point2D(0, 0), Point2D(4, 0), Point2D(4, 3))

    Keywords sss, sas, or asa can be used to give the desired
    side lengths (in order) and interior angles (in degrees) that
    define the triangle:

    >>> Triangle(sss=(3, 4, 5))
    Triangle(Point2D(0, 0), Point2D(3, 0), Point2D(3, 4))
    >>> Triangle(asa=(30, 1, 30))
    Triangle(Point2D(0, 0), Point2D(1, 0), Point2D(1/2, sqrt(3)/6))
    >>> Triangle(sas=(1, 45, 2))
    Triangle(Point2D(0, 0), Point2D(2, 0), Point2D(sqrt(2)/2, sqrt(2)/2))

    """

    def __new__(cls, *args, **kwargs):
        if len(args) != 3:
            if 'sss' in kwargs:
                return _sss(*[simplify(a) for a in kwargs['sss']])
            if 'asa' in kwargs:
                return _asa(*[simplify(a) for a in kwargs['asa']])
            if 'sas' in kwargs:
                return _sas(*[simplify(a) for a in kwargs['sas']])
            msg = "Triangle instantiates with three points or a valid keyword."
            raise GeometryError(msg)

        vertices = [Point(a, dim=2, **kwargs) for a in args]

        # remove consecutive duplicates
        nodup = []
        for p in vertices:
            if nodup and p == nodup[-1]:
                continue
            nodup.append(p)
        if len(nodup) > 1 and nodup[-1] == nodup[0]:
            nodup.pop()  # last point was same as first

        # remove collinear points
        i = -3
        while i < len(nodup) - 3 and len(nodup) > 2:
            a, b, c = sorted(
                [nodup[i], nodup[i + 1], nodup[i + 2]], key=default_sort_key)
            if Point.is_collinear(a, b, c):
                nodup[i] = a
                nodup[i + 1] = None
                nodup.pop(i + 1)
            i += 1

        vertices = list(filter(lambda x: x is not None, nodup))

        if len(vertices) == 3:
            return GeometryEntity.__new__(cls, *vertices, **kwargs)
        elif len(vertices) == 2:
            return Segment(*vertices, **kwargs)
        else:
            return Point(*vertices, **kwargs)

    @property
    def vertices(self):
        """The triangle's vertices

        Returns
        =======

        vertices : tuple
            Each element in the tuple is a Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t.vertices
        (Point2D(0, 0), Point2D(4, 0), Point2D(4, 3))

        """
        return self.args

    def is_similar(t1, t2):
        """Is another triangle similar to this one.

        Two triangles are similar if one can be uniformly scaled to the other.

        Parameters
        ==========

        other: Triangle

        Returns
        =======

        is_similar : boolean

        See Also
        ========

        sympy.geometry.entity.GeometryEntity.is_similar

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t2 = Triangle(Point(0, 0), Point(-4, 0), Point(-4, -3))
        >>> t1.is_similar(t2)
        True

        >>> t2 = Triangle(Point(0, 0), Point(-4, 0), Point(-4, -4))
        >>> t1.is_similar(t2)
        False

        """
        if not isinstance(t2, Polygon):
            return False

        s1_1, s1_2, s1_3 = [side.length for side in t1.sides]
        s2 = [side.length for side in t2.sides]

        def _are_similar(u1, u2, u3, v1, v2, v3):
            e1 = simplify(u1/v1)
            e2 = simplify(u2/v2)
            e3 = simplify(u3/v3)
            return bool(e1 == e2) and bool(e2 == e3)

        # There's only 6 permutations, so write them out
        return _are_similar(s1_1, s1_2, s1_3, *s2) or \
            _are_similar(s1_1, s1_3, s1_2, *s2) or \
            _are_similar(s1_2, s1_1, s1_3, *s2) or \
            _are_similar(s1_2, s1_3, s1_1, *s2) or \
            _are_similar(s1_3, s1_1, s1_2, *s2) or \
            _are_similar(s1_3, s1_2, s1_1, *s2)

    def is_equilateral(self):
        """Are all the sides the same length?

        Returns
        =======

        is_equilateral : boolean

        See Also
        ========

        sympy.geometry.entity.GeometryEntity.is_similar, RegularPolygon
        is_isosceles, is_right, is_scalene

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t1.is_equilateral()
        False

        >>> from sympy import sqrt
        >>> t2 = Triangle(Point(0, 0), Point(10, 0), Point(5, 5*sqrt(3)))
        >>> t2.is_equilateral()
        True

        """
        return not has_variety(s.length for s in self.sides)

    def is_isosceles(self):
        """Are two or more of the sides the same length?

        Returns
        =======

        is_isosceles : boolean

        See Also
        ========

        is_equilateral, is_right, is_scalene

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(2, 4))
        >>> t1.is_isosceles()
        True

        """
        return has_dups(s.length for s in self.sides)

    def is_scalene(self):
        """Are all the sides of the triangle of different lengths?

        Returns
        =======

        is_scalene : boolean

        See Also
        ========

        is_equilateral, is_isosceles, is_right

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(1, 4))
        >>> t1.is_scalene()
        True

        """
        return not has_dups(s.length for s in self.sides)

    def is_right(self):
        """Is the triangle right-angled.

        Returns
        =======

        is_right : boolean

        See Also
        ========

        sympy.geometry.line.LinearEntity.is_perpendicular
        is_equilateral, is_isosceles, is_scalene

        Examples
        ========

        >>> from sympy import Triangle, Point
        >>> t1 = Triangle(Point(0, 0), Point(4, 0), Point(4, 3))
        >>> t1.is_right()
        True

        """
        s = self.sides
        return Segment.is_perpendicular(s[0], s[1]) or \
            Segment.is_perpendicular(s[1], s[2]) or \
            Segment.is_perpendicular(s[0], s[2])

    @property
    def altitudes(self):
        """The altitudes of the triangle.

        An altitude of a triangle is a segment through a vertex,
        perpendicular to the opposite side, with length being the
        height of the vertex measured from the line containing the side.

        Returns
        =======

        altitudes : dict
            The dictionary consists of keys which are vertices and values
            which are Segments.

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment.length

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.altitudes[p1]
        Segment2D(Point2D(0, 0), Point2D(1/2, 1/2))

        """
        s = self.sides
        v = self.vertices
        return {v[0]: s[1].perpendicular_segment(v[0]),
                v[1]: s[2].perpendicular_segment(v[1]),
                v[2]: s[0].perpendicular_segment(v[2])}

    @property
    def orthocenter(self):
        """The orthocenter of the triangle.

        The orthocenter is the intersection of the altitudes of a triangle.
        It may lie inside, outside or on the triangle.

        Returns
        =======

        orthocenter : Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.orthocenter
        Point2D(0, 0)

        """
        a = self.altitudes
        v = self.vertices
        return Line(a[v[0]]).intersection(Line(a[v[1]]))[0]

    @property
    def circumcenter(self):
        """The circumcenter of the triangle

        The circumcenter is the center of the circumcircle.

        Returns
        =======

        circumcenter : Point

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.circumcenter
        Point2D(1/2, 1/2)
        """
        a, b, c = [x.perpendicular_bisector() for x in self.sides]
        return a.intersection(b)[0]

    @property
    def circumradius(self):
        """The radius of the circumcircle of the triangle.

        Returns
        =======

        circumradius : number of Basic instance

        See Also
        ========

        sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy import Point, Triangle
        >>> a = Symbol('a')
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, a)
        >>> t = Triangle(p1, p2, p3)
        >>> t.circumradius
        sqrt(a**2/4 + 1/4)
        """
        return Point.distance(self.circumcenter, self.vertices[0])

    @property
    def circumcircle(self):
        """The circle which passes through the three vertices of the triangle.

        Returns
        =======

        circumcircle : Circle

        See Also
        ========

        sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.circumcircle
        Circle(Point2D(1/2, 1/2), sqrt(2)/2)

        """
        return Circle(self.circumcenter, self.circumradius)

    def bisectors(self):
        """The angle bisectors of the triangle.

        An angle bisector of a triangle is a straight line through a vertex
        which cuts the corresponding angle in half.

        Returns
        =======

        bisectors : dict
            Each key is a vertex (Point) and each value is the corresponding
            bisector (Segment).

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Segment

        Examples
        ========

        >>> from sympy import Point, Triangle, Segment
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> from sympy import sqrt
        >>> t.bisectors()[p2] == Segment(Point(1, 0), Point(0, sqrt(2) - 1))
        True

        """
        # use lines containing sides so containment check during
        # intersection calculation can be avoided, thus reducing
        # the processing time for calculating the bisectors
        s = [Line(l) for l in self.sides]
        v = self.vertices
        c = self.incenter
        l1 = Segment(v[0], Line(v[0], c).intersection(s[1])[0])
        l2 = Segment(v[1], Line(v[1], c).intersection(s[2])[0])
        l3 = Segment(v[2], Line(v[2], c).intersection(s[0])[0])
        return {v[0]: l1, v[1]: l2, v[2]: l3}

    @property
    def incenter(self):
        """The center of the incircle.

        The incircle is the circle which lies inside the triangle and touches
        all three sides.

        Returns
        =======

        incenter : Point

        See Also
        ========

        incircle, sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.incenter
        Point2D(1 - sqrt(2)/2, 1 - sqrt(2)/2)

        """
        s = self.sides
        l = Matrix([s[i].length for i in [1, 2, 0]])
        p = sum(l)
        v = self.vertices
        x = simplify(l.dot(Matrix([vi.x for vi in v]))/p)
        y = simplify(l.dot(Matrix([vi.y for vi in v]))/p)
        return Point(x, y)

    @property
    def inradius(self):
        """The radius of the incircle.

        Returns
        =======

        inradius : number of Basic instance

        See Also
        ========

        incircle, sympy.geometry.ellipse.Circle.radius

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(4, 0), Point(0, 3)
        >>> t = Triangle(p1, p2, p3)
        >>> t.inradius
        1

        """
        return simplify(2 * self.area / self.perimeter)

    @property
    def incircle(self):
        """The incircle of the triangle.

        The incircle is the circle which lies inside the triangle and touches
        all three sides.

        Returns
        =======

        incircle : Circle

        See Also
        ========

        sympy.geometry.ellipse.Circle

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(2, 0), Point(0, 2)
        >>> t = Triangle(p1, p2, p3)
        >>> t.incircle
        Circle(Point2D(2 - sqrt(2), 2 - sqrt(2)), 2 - sqrt(2))

        """
        return Circle(self.incenter, self.inradius)

    @property
    def exradii(self):
        """The radius of excircles of a triangle.

        An excircle of the triangle is a circle lying outside the triangle,
        tangent to one of its sides and tangent to the extensions of the
        other two.

        Returns
        =======

        exradii : dict

        See Also
        ========

        sympy.geometry.polygon.Triangle.inradius

        Examples
        ========

        The exradius touches the side of the triangle to which it is keyed, e.g.
        the exradius touching side 2 is:

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(6, 0), Point(0, 2)
        >>> t = Triangle(p1, p2, p3)
        >>> t.exradii[t.sides[2]]
        -2 + sqrt(10)

        References
        ==========

        .. [1] https://mathworld.wolfram.com/Exradius.html
        .. [2] https://mathworld.wolfram.com/Excircles.html

        """

        side = self.sides
        a = side[0].length
        b = side[1].length
        c = side[2].length
        s = (a+b+c)/2
        area = self.area
        exradii = {self.sides[0]: simplify(area/(s-a)),
                   self.sides[1]: simplify(area/(s-b)),
                   self.sides[2]: simplify(area/(s-c))}

        return exradii

    @property
    def excenters(self):
        """Excenters of the triangle.

        An excenter is the center of a circle that is tangent to a side of the
        triangle and the extensions of the other two sides.

        Returns
        =======

        excenters : dict


        Examples
        ========

        The excenters are keyed to the side of the triangle to which their corresponding
        excircle is tangent: The center is keyed, e.g. the excenter of a circle touching
        side 0 is:

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(6, 0), Point(0, 2)
        >>> t = Triangle(p1, p2, p3)
        >>> t.excenters[t.sides[0]]
        Point2D(12*sqrt(10), 2/3 + sqrt(10)/3)

        See Also
        ========

        sympy.geometry.polygon.Triangle.exradii

        References
        ==========

        .. [1] https://mathworld.wolfram.com/Excircles.html

        """

        s = self.sides
        v = self.vertices
        a = s[0].length
        b = s[1].length
        c = s[2].length
        x = [v[0].x, v[1].x, v[2].x]
        y = [v[0].y, v[1].y, v[2].y]

        exc_coords = {
            "x1": simplify(-a*x[0]+b*x[1]+c*x[2]/(-a+b+c)),
            "x2": simplify(a*x[0]-b*x[1]+c*x[2]/(a-b+c)),
            "x3": simplify(a*x[0]+b*x[1]-c*x[2]/(a+b-c)),
            "y1": simplify(-a*y[0]+b*y[1]+c*y[2]/(-a+b+c)),
            "y2": simplify(a*y[0]-b*y[1]+c*y[2]/(a-b+c)),
            "y3": simplify(a*y[0]+b*y[1]-c*y[2]/(a+b-c))
        }

        excenters = {
            s[0]: Point(exc_coords["x1"], exc_coords["y1"]),
            s[1]: Point(exc_coords["x2"], exc_coords["y2"]),
            s[2]: Point(exc_coords["x3"], exc_coords["y3"])
        }

        return excenters

    @property
    def medians(self):
        """The medians of the triangle.

        A median of a triangle is a straight line through a vertex and the
        midpoint of the opposite side, and divides the triangle into two
        equal areas.

        Returns
        =======

        medians : dict
            Each key is a vertex (Point) and each value is the median (Segment)
            at that point.

        See Also
        ========

        sympy.geometry.point.Point.midpoint, sympy.geometry.line.Segment.midpoint

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.medians[p1]
        Segment2D(Point2D(0, 0), Point2D(1/2, 1/2))

        """
        s = self.sides
        v = self.vertices
        return {v[0]: Segment(v[0], s[1].midpoint),
                v[1]: Segment(v[1], s[2].midpoint),
                v[2]: Segment(v[2], s[0].midpoint)}

    @property
    def medial(self):
        """The medial triangle of the triangle.

        The triangle which is formed from the midpoints of the three sides.

        Returns
        =======

        medial : Triangle

        See Also
        ========

        sympy.geometry.line.Segment.midpoint

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.medial
        Triangle(Point2D(1/2, 0), Point2D(1/2, 1/2), Point2D(0, 1/2))

        """
        s = self.sides
        return Triangle(s[0].midpoint, s[1].midpoint, s[2].midpoint)

    @property
    def nine_point_circle(self):
        """The nine-point circle of the triangle.

        Nine-point circle is the circumcircle of the medial triangle, which
        passes through the feet of altitudes and the middle points of segments
        connecting the vertices and the orthocenter.

        Returns
        =======

        nine_point_circle : Circle

        See also
        ========

        sympy.geometry.line.Segment.midpoint
        sympy.geometry.polygon.Triangle.medial
        sympy.geometry.polygon.Triangle.orthocenter

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.nine_point_circle
        Circle(Point2D(1/4, 1/4), sqrt(2)/4)

        """
        return Circle(*self.medial.vertices)

    @property
    def eulerline(self):
        """The Euler line of the triangle.

        The line which passes through circumcenter, centroid and orthocenter.

        Returns
        =======

        eulerline : Line (or Point for equilateral triangles in which case all
                    centers coincide)

        Examples
        ========

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(1, 0), Point(0, 1)
        >>> t = Triangle(p1, p2, p3)
        >>> t.eulerline
        Line2D(Point2D(0, 0), Point2D(1/2, 1/2))

        """
        if self.is_equilateral():
            return self.orthocenter
        return Line(self.orthocenter, self.circumcenter)

def rad(d):
    """Return the radian value for the given degrees (pi = 180 degrees)."""
    return d*pi/180


def deg(r):
    """Return the degree value for the given radians (pi = 180 degrees)."""
    return r/pi*180


def _slope(d):
    rv = tan(rad(d))
    return rv


def _asa(d1, l, d2):
    """Return triangle having side with length l on the x-axis."""
    xy = Line((0, 0), slope=_slope(d1)).intersection(
        Line((l, 0), slope=_slope(180 - d2)))[0]
    return Triangle((0, 0), (l, 0), xy)


def _sss(l1, l2, l3):
    """Return triangle having side of length l1 on the x-axis."""
    c1 = Circle((0, 0), l3)
    c2 = Circle((l1, 0), l2)
    inter = [a for a in c1.intersection(c2) if a.y.is_nonnegative]
    if not inter:
        return None
    pt = inter[0]
    return Triangle((0, 0), (l1, 0), pt)


def _sas(l1, d, l2):
    """Return triangle having side with length l2 on the x-axis."""
    p1 = Point(0, 0)
    p2 = Point(l2, 0)
    p3 = Point(cos(rad(d))*l1, sin(rad(d))*l1)
    return Triangle(p1, p2, p3)
