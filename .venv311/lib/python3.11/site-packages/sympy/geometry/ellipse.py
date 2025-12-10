"""Elliptical geometrical entities.

Contains
* Ellipse
* Circle

"""

from sympy.core.expr import Expr
from sympy.core.relational import Eq
from sympy.core import S, pi, sympify
from sympy.core.evalf import N
from sympy.core.parameters import global_parameters
from sympy.core.logic import fuzzy_bool
from sympy.core.numbers import Rational, oo
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, uniquely_named_symbol, _symbol
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.elliptic_integrals import elliptic_e
from .entity import GeometryEntity, GeometrySet
from .exceptions import GeometryError
from .line import Line, Segment, Ray2D, Segment2D, Line2D, LinearEntity3D
from .point import Point, Point2D, Point3D
from .util import idiff, find
from sympy.polys import DomainError, Poly, PolynomialError
from sympy.polys.polyutils import _not_a_coeff, _nsort
from sympy.solvers import solve
from sympy.solvers.solveset import linear_coeffs
from sympy.utilities.misc import filldedent, func_name

from mpmath.libmp.libmpf import prec_to_dps

import random

x, y = [Dummy('ellipse_dummy', real=True) for i in range(2)]


class Ellipse(GeometrySet):
    """An elliptical GeometryEntity.

    Parameters
    ==========

    center : Point, optional
        Default value is Point(0, 0)
    hradius : number or SymPy expression, optional
    vradius : number or SymPy expression, optional
    eccentricity : number or SymPy expression, optional
        Two of `hradius`, `vradius` and `eccentricity` must be supplied to
        create an Ellipse. The third is derived from the two supplied.

    Attributes
    ==========

    center
    hradius
    vradius
    area
    circumference
    eccentricity
    periapsis
    apoapsis
    focus_distance
    foci

    Raises
    ======

    GeometryError
        When `hradius`, `vradius` and `eccentricity` are incorrectly supplied
        as parameters.
    TypeError
        When `center` is not a Point.

    See Also
    ========

    Circle

    Notes
    -----
    Constructed from a center and two radii, the first being the horizontal
    radius (along the x-axis) and the second being the vertical radius (along
    the y-axis).

    When symbolic value for hradius and vradius are used, any calculation that
    refers to the foci or the major or minor axis will assume that the ellipse
    has its major radius on the x-axis. If this is not true then a manual
    rotation is necessary.

    Examples
    ========

    >>> from sympy import Ellipse, Point, Rational
    >>> e1 = Ellipse(Point(0, 0), 5, 1)
    >>> e1.hradius, e1.vradius
    (5, 1)
    >>> e2 = Ellipse(Point(3, 1), hradius=3, eccentricity=Rational(4, 5))
    >>> e2
    Ellipse(Point2D(3, 1), 3, 9/5)

    """

    def __contains__(self, o):
        if isinstance(o, Point):
            res = self.equation(x, y).subs({x: o.x, y: o.y})
            return trigsimp(simplify(res)) is S.Zero
        elif isinstance(o, Ellipse):
            return self == o
        return False

    def __eq__(self, o):
        """Is the other GeometryEntity the same as this ellipse?"""
        return isinstance(o, Ellipse) and (self.center == o.center and
                                           self.hradius == o.hradius and
                                           self.vradius == o.vradius)

    def __hash__(self):
        return super().__hash__()

    def __new__(
        cls, center=None, hradius=None, vradius=None, eccentricity=None, **kwargs):

        hradius = sympify(hradius)
        vradius = sympify(vradius)

        if center is None:
            center = Point(0, 0)
        else:
            if len(center) != 2:
                raise ValueError('The center of "{}" must be a two dimensional point'.format(cls))
            center = Point(center, dim=2)

        if len(list(filter(lambda x: x is not None, (hradius, vradius, eccentricity)))) != 2:
            raise ValueError(filldedent('''
                Exactly two arguments of "hradius", "vradius", and
                "eccentricity" must not be None.'''))

        if eccentricity is not None:
            eccentricity = sympify(eccentricity)
            if eccentricity.is_negative:
                raise GeometryError("Eccentricity of ellipse/circle should lie between [0, 1)")
            elif hradius is None:
                hradius = vradius / sqrt(1 - eccentricity**2)
            elif vradius is None:
                vradius = hradius * sqrt(1 - eccentricity**2)

        if hradius == vradius:
            return Circle(center, hradius, **kwargs)

        if S.Zero in (hradius, vradius):
            return Segment(Point(center[0] - hradius, center[1] - vradius), Point(center[0] + hradius, center[1] + vradius))

        if hradius.is_real is False or vradius.is_real is False:
            raise GeometryError("Invalid value encountered when computing hradius / vradius.")

        return GeometryEntity.__new__(cls, center, hradius, vradius, **kwargs)

    def _svg(self, scale_factor=1., fill_color="#66cc99"):
        """Returns SVG ellipse element for the Ellipse.

        Parameters
        ==========

        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is "#66cc99".
        """

        c = N(self.center)
        h, v = N(self.hradius), N(self.vradius)
        return (
            '<ellipse fill="{1}" stroke="#555555" '
            'stroke-width="{0}" opacity="0.6" cx="{2}" cy="{3}" rx="{4}" ry="{5}"/>'
        ).format(2. * scale_factor, fill_color, c.x, c.y, h, v)

    @property
    def ambient_dimension(self):
        return 2

    @property
    def apoapsis(self):
        """The apoapsis of the ellipse.

        The greatest distance between the focus and the contour.

        Returns
        =======

        apoapsis : number

        See Also
        ========

        periapsis : Returns shortest distance between foci and contour

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.apoapsis
        2*sqrt(2) + 3

        """
        return self.major * (1 + self.eccentricity)

    def arbitrary_point(self, parameter='t'):
        """A parameterized point on the ellipse.

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
            When `parameter` already appears in the functions.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(0, 0), 3, 2)
        >>> e1.arbitrary_point()
        Point2D(3*cos(t), 2*sin(t))

        """
        t = _symbol(parameter, real=True)
        if t.name in (f.name for f in self.free_symbols):
            raise ValueError(filldedent('Symbol %s already appears in object '
                                        'and cannot be used as a parameter.' % t.name))
        return Point(self.center.x + self.hradius*cos(t),
                     self.center.y + self.vradius*sin(t))

    @property
    def area(self):
        """The area of the ellipse.

        Returns
        =======

        area : number

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.area
        3*pi

        """
        return simplify(S.Pi * self.hradius * self.vradius)

    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.

        """

        h, v = self.hradius, self.vradius
        return (self.center.x - h, self.center.y - v, self.center.x + h, self.center.y + v)

    @property
    def center(self):
        """The center of the ellipse.

        Returns
        =======

        center : number

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.center
        Point2D(0, 0)

        """
        return self.args[0]

    @property
    def circumference(self):
        """The circumference of the ellipse.

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.circumference
        12*elliptic_e(8/9)

        """
        if self.eccentricity == 1:
            # degenerate
            return 4*self.major
        elif self.eccentricity == 0:
            # circle
            return 2*pi*self.hradius
        else:
            return 4*self.major*elliptic_e(self.eccentricity**2)

    @property
    def eccentricity(self):
        """The eccentricity of the ellipse.

        Returns
        =======

        eccentricity : number

        Examples
        ========

        >>> from sympy import Point, Ellipse, sqrt
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, sqrt(2))
        >>> e1.eccentricity
        sqrt(7)/3

        """
        return self.focus_distance / self.major

    def encloses_point(self, p):
        """
        Return True if p is enclosed by (is inside of) self.

        Notes
        -----
        Being on the border of self is considered False.

        Parameters
        ==========

        p : Point

        Returns
        =======

        encloses_point : True, False or None

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Ellipse, S
        >>> from sympy.abc import t
        >>> e = Ellipse((0, 0), 3, 2)
        >>> e.encloses_point((0, 0))
        True
        >>> e.encloses_point(e.arbitrary_point(t).subs(t, S.Half))
        False
        >>> e.encloses_point((4, 0))
        False

        """
        p = Point(p, dim=2)
        if p in self:
            return False

        if len(self.foci) == 2:
            # if the combined distance from the foci to p (h1 + h2) is less
            # than the combined distance from the foci to the minor axis
            # (which is the same as the major axis length) then p is inside
            # the ellipse
            h1, h2 = [f.distance(p) for f in self.foci]
            test = 2*self.major - (h1 + h2)
        else:
            test = self.radius - self.center.distance(p)

        return fuzzy_bool(test.is_positive)

    def equation(self, x='x', y='y', _slope=None):
        """
        Returns the equation of an ellipse aligned with the x and y axes;
        when slope is given, the equation returned corresponds to an ellipse
        with a major axis having that slope.

        Parameters
        ==========

        x : str, optional
            Label for the x-axis. Default value is 'x'.
        y : str, optional
            Label for the y-axis. Default value is 'y'.
        _slope : Expr, optional
                The slope of the major axis. Ignored when 'None'.

        Returns
        =======

        equation : SymPy expression

        See Also
        ========

        arbitrary_point : Returns parameterized point on ellipse

        Examples
        ========

        >>> from sympy import Point, Ellipse, pi
        >>> from sympy.abc import x, y
        >>> e1 = Ellipse(Point(1, 0), 3, 2)
        >>> eq1 = e1.equation(x, y); eq1
        y**2/4 + (x/3 - 1/3)**2 - 1
        >>> eq2 = e1.equation(x, y, _slope=1); eq2
        (-x + y + 1)**2/8 + (x + y - 1)**2/18 - 1

        A point on e1 satisfies eq1. Let's use one on the x-axis:

        >>> p1 = e1.center + Point(e1.major, 0)
        >>> assert eq1.subs(x, p1.x).subs(y, p1.y) == 0

        When rotated the same as the rotated ellipse, about the center
        point of the ellipse, it will satisfy the rotated ellipse's
        equation, too:

        >>> r1 = p1.rotate(pi/4, e1.center)
        >>> assert eq2.subs(x, r1.x).subs(y, r1.y) == 0

        References
        ==========

        .. [1] https://math.stackexchange.com/questions/108270/what-is-the-equation-of-an-ellipse-that-is-not-aligned-with-the-axis
        .. [2] https://en.wikipedia.org/wiki/Ellipse#Shifted_ellipse

        """

        x = _symbol(x, real=True)
        y = _symbol(y, real=True)

        dx = x - self.center.x
        dy = y - self.center.y

        if _slope is not None:
            L = (dy - _slope*dx)**2
            l = (_slope*dy + dx)**2
            h = 1 + _slope**2
            b = h*self.major**2
            a = h*self.minor**2
            return l/b + L/a - 1

        else:
            t1 = (dx/self.hradius)**2
            t2 = (dy/self.vradius)**2
            return t1 + t2 - 1

    def evolute(self, x='x', y='y'):
        """The equation of evolute of the ellipse.

        Parameters
        ==========

        x : str, optional
            Label for the x-axis. Default value is 'x'.
        y : str, optional
            Label for the y-axis. Default value is 'y'.

        Returns
        =======

        equation : SymPy expression

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(1, 0), 3, 2)
        >>> e1.evolute()
        2**(2/3)*y**(2/3) + (3*x - 3)**(2/3) - 5**(2/3)
        """
        if len(self.args) != 3:
            raise NotImplementedError('Evolute of arbitrary Ellipse is not supported.')
        x = _symbol(x, real=True)
        y = _symbol(y, real=True)
        t1 = (self.hradius*(x - self.center.x))**Rational(2, 3)
        t2 = (self.vradius*(y - self.center.y))**Rational(2, 3)
        return t1 + t2 - (self.hradius**2 - self.vradius**2)**Rational(2, 3)

    @property
    def foci(self):
        """The foci of the ellipse.

        Notes
        -----
        The foci can only be calculated if the major/minor axes are known.

        Raises
        ======

        ValueError
            When the major and minor axis cannot be determined.

        See Also
        ========

        sympy.geometry.point.Point
        focus_distance : Returns the distance between focus and center

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.foci
        (Point2D(-2*sqrt(2), 0), Point2D(2*sqrt(2), 0))

        """
        c = self.center
        hr, vr = self.hradius, self.vradius
        if hr == vr:
            return (c, c)

        # calculate focus distance manually, since focus_distance calls this
        # routine
        fd = sqrt(self.major**2 - self.minor**2)
        if hr == self.minor:
            # foci on the y-axis
            return (c + Point(0, -fd), c + Point(0, fd))
        elif hr == self.major:
            # foci on the x-axis
            return (c + Point(-fd, 0), c + Point(fd, 0))

    @property
    def focus_distance(self):
        """The focal distance of the ellipse.

        The distance between the center and one focus.

        Returns
        =======

        focus_distance : number

        See Also
        ========

        foci

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.focus_distance
        2*sqrt(2)

        """
        return Point.distance(self.center, self.foci[0])

    @property
    def hradius(self):
        """The horizontal radius of the ellipse.

        Returns
        =======

        hradius : number

        See Also
        ========

        vradius, major, minor

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.hradius
        3

        """
        return self.args[1]

    def intersection(self, o):
        """The intersection of this ellipse and another geometrical entity
        `o`.

        Parameters
        ==========

        o : GeometryEntity

        Returns
        =======

        intersection : list of GeometryEntity objects

        Notes
        -----
        Currently supports intersections with Point, Line, Segment, Ray,
        Circle and Ellipse types.

        See Also
        ========

        sympy.geometry.entity.GeometryEntity

        Examples
        ========

        >>> from sympy import Ellipse, Point, Line
        >>> e = Ellipse(Point(0, 0), 5, 7)
        >>> e.intersection(Point(0, 0))
        []
        >>> e.intersection(Point(5, 0))
        [Point2D(5, 0)]
        >>> e.intersection(Line(Point(0,0), Point(0, 1)))
        [Point2D(0, -7), Point2D(0, 7)]
        >>> e.intersection(Line(Point(5,0), Point(5, 1)))
        [Point2D(5, 0)]
        >>> e.intersection(Line(Point(6,0), Point(6, 1)))
        []
        >>> e = Ellipse(Point(-1, 0), 4, 3)
        >>> e.intersection(Ellipse(Point(1, 0), 4, 3))
        [Point2D(0, -3*sqrt(15)/4), Point2D(0, 3*sqrt(15)/4)]
        >>> e.intersection(Ellipse(Point(5, 0), 4, 3))
        [Point2D(2, -3*sqrt(7)/4), Point2D(2, 3*sqrt(7)/4)]
        >>> e.intersection(Ellipse(Point(100500, 0), 4, 3))
        []
        >>> e.intersection(Ellipse(Point(0, 0), 3, 4))
        [Point2D(3, 0), Point2D(-363/175, -48*sqrt(111)/175), Point2D(-363/175, 48*sqrt(111)/175)]
        >>> e.intersection(Ellipse(Point(-1, 0), 3, 4))
        [Point2D(-17/5, -12/5), Point2D(-17/5, 12/5), Point2D(7/5, -12/5), Point2D(7/5, 12/5)]
        """
        # TODO: Replace solve with nonlinsolve, when nonlinsolve will be able to solve in real domain

        if isinstance(o, Point):
            if o in self:
                return [o]
            else:
                return []

        elif isinstance(o, (Segment2D, Ray2D)):
            ellipse_equation = self.equation(x, y)
            result = solve([ellipse_equation, Line(
                o.points[0], o.points[1]).equation(x, y)], [x, y],
                set=True)[1]
            return list(ordered([Point(i) for i in result if i in o]))

        elif isinstance(o, Polygon):
            return o.intersection(self)

        elif isinstance(o, (Ellipse, Line2D)):
            if o == self:
                return self
            else:
                ellipse_equation = self.equation(x, y)
                return list(ordered([Point(i) for i in solve(
                    [ellipse_equation, o.equation(x, y)], [x, y],
                    set=True)[1]]))
        elif isinstance(o, LinearEntity3D):
            raise TypeError('Entity must be two dimensional, not three dimensional')
        else:
            raise TypeError('Intersection not handled for %s' % func_name(o))

    def is_tangent(self, o):
        """Is `o` tangent to the ellipse?

        Parameters
        ==========

        o : GeometryEntity
            An Ellipse, LinearEntity or Polygon

        Raises
        ======

        NotImplementedError
            When the wrong type of argument is supplied.

        Returns
        =======

        is_tangent: boolean
            True if o is tangent to the ellipse, False otherwise.

        See Also
        ========

        tangent_lines

        Examples
        ========

        >>> from sympy import Point, Ellipse, Line
        >>> p0, p1, p2 = Point(0, 0), Point(3, 0), Point(3, 3)
        >>> e1 = Ellipse(p0, 3, 2)
        >>> l1 = Line(p1, p2)
        >>> e1.is_tangent(l1)
        True

        """
        if isinstance(o, Point2D):
            return False
        elif isinstance(o, Ellipse):
            intersect = self.intersection(o)
            if isinstance(intersect, Ellipse):
                return True
            elif intersect:
                return all((self.tangent_lines(i)[0]).equals(o.tangent_lines(i)[0]) for i in intersect)
            else:
                return False
        elif isinstance(o, Line2D):
            hit = self.intersection(o)
            if not hit:
                return False
            if len(hit) == 1:
                return True
            # might return None if it can't decide
            return hit[0].equals(hit[1])
        elif isinstance(o, (Segment2D, Ray2D)):
            intersect = self.intersection(o)
            if len(intersect) == 1:
                return o in self.tangent_lines(intersect[0])[0]
            else:
                return False
        elif isinstance(o, Polygon):
            return all(self.is_tangent(s) for s in o.sides)
        elif isinstance(o, (LinearEntity3D, Point3D)):
            raise TypeError('Entity must be two dimensional, not three dimensional')
        else:
            raise TypeError('Is_tangent not handled for %s' % func_name(o))

    @property
    def major(self):
        """Longer axis of the ellipse (if it can be determined) else hradius.

        Returns
        =======

        major : number or expression

        See Also
        ========

        hradius, vradius, minor

        Examples
        ========

        >>> from sympy import Point, Ellipse, Symbol
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.major
        3

        >>> a = Symbol('a')
        >>> b = Symbol('b')
        >>> Ellipse(p1, a, b).major
        a
        >>> Ellipse(p1, b, a).major
        b

        >>> m = Symbol('m')
        >>> M = m + 1
        >>> Ellipse(p1, m, M).major
        m + 1

        """
        ab = self.args[1:3]
        if len(ab) == 1:
            return ab[0]
        a, b = ab
        o = b - a < 0
        if o == True:
            return a
        elif o == False:
            return b
        return self.hradius

    @property
    def minor(self):
        """Shorter axis of the ellipse (if it can be determined) else vradius.

        Returns
        =======

        minor : number or expression

        See Also
        ========

        hradius, vradius, major

        Examples
        ========

        >>> from sympy import Point, Ellipse, Symbol
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.minor
        1

        >>> a = Symbol('a')
        >>> b = Symbol('b')
        >>> Ellipse(p1, a, b).minor
        b
        >>> Ellipse(p1, b, a).minor
        a

        >>> m = Symbol('m')
        >>> M = m + 1
        >>> Ellipse(p1, m, M).minor
        m

        """
        ab = self.args[1:3]
        if len(ab) == 1:
            return ab[0]
        a, b = ab
        o = a - b < 0
        if o == True:
            return a
        elif o == False:
            return b
        return self.vradius

    def normal_lines(self, p, prec=None):
        """Normal lines between `p` and the ellipse.

        Parameters
        ==========

        p : Point

        Returns
        =======

        normal_lines : list with 1, 2 or 4 Lines

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e = Ellipse((0, 0), 2, 3)
        >>> c = e.center
        >>> e.normal_lines(c + Point(1, 0))
        [Line2D(Point2D(0, 0), Point2D(1, 0))]
        >>> e.normal_lines(c)
        [Line2D(Point2D(0, 0), Point2D(0, 1)), Line2D(Point2D(0, 0), Point2D(1, 0))]

        Off-axis points require the solution of a quartic equation. This
        often leads to very large expressions that may be of little practical
        use. An approximate solution of `prec` digits can be obtained by
        passing in the desired value:

        >>> e.normal_lines((3, 3), prec=2)
        [Line2D(Point2D(-0.81, -2.7), Point2D(0.19, -1.2)),
        Line2D(Point2D(1.5, -2.0), Point2D(2.5, -2.7))]

        Whereas the above solution has an operation count of 12, the exact
        solution has an operation count of 2020.
        """
        p = Point(p, dim=2)

        # XXX change True to something like self.angle == 0 if the arbitrarily
        # rotated ellipse is introduced.
        # https://github.com/sympy/sympy/issues/2815)
        if True:
            rv = []
            if p.x == self.center.x:
                rv.append(Line(self.center, slope=oo))
            if p.y == self.center.y:
                rv.append(Line(self.center, slope=0))
            if rv:
                # at these special orientations of p either 1 or 2 normals
                # exist and we are done
                return rv

        # find the 4 normal points and construct lines through them with
        # the corresponding slope
        eq = self.equation(x, y)
        dydx = idiff(eq, y, x)
        norm = -1/dydx
        slope = Line(p, (x, y)).slope
        seq = slope - norm

        # TODO: Replace solve with solveset, when this line is tested
        yis = solve(seq, y)[0]
        xeq = eq.subs(y, yis).as_numer_denom()[0].expand()
        if len(xeq.free_symbols) == 1:
            try:
                # this is so much faster, it's worth a try
                xsol = Poly(xeq, x).real_roots()
            except (DomainError, PolynomialError, NotImplementedError):
                # TODO: Replace solve with solveset, when these lines are tested
                xsol = _nsort(solve(xeq, x), separated=True)[0]
            points = [Point(i, solve(eq.subs(x, i), y)[0]) for i in xsol]
        else:
            raise NotImplementedError(
                'intersections for the general ellipse are not supported')
        slopes = [norm.subs(zip((x, y), pt.args)) for pt in points]
        if prec is not None:
            points = [pt.n(prec) for pt in points]
            slopes = [i if _not_a_coeff(i) else i.n(prec) for i in slopes]
        return [Line(pt, slope=s) for pt, s in zip(points, slopes)]

    @property
    def periapsis(self):
        """The periapsis of the ellipse.

        The shortest distance between the focus and the contour.

        Returns
        =======

        periapsis : number

        See Also
        ========

        apoapsis : Returns greatest distance between focus and contour

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.periapsis
        3 - 2*sqrt(2)

        """
        return self.major * (1 - self.eccentricity)

    @property
    def semilatus_rectum(self):
        """
        Calculates the semi-latus rectum of the Ellipse.

        Semi-latus rectum is defined as one half of the chord through a
        focus parallel to the conic section directrix of a conic section.

        Returns
        =======

        semilatus_rectum : number

        See Also
        ========

        apoapsis : Returns greatest distance between focus and contour

        periapsis : The shortest distance between the focus and the contour

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.semilatus_rectum
        1/3

        References
        ==========

        .. [1] https://mathworld.wolfram.com/SemilatusRectum.html
        .. [2] https://en.wikipedia.org/wiki/Ellipse#Semi-latus_rectum

        """
        return self.major * (1 - self.eccentricity ** 2)

    def auxiliary_circle(self):
        """Returns a Circle whose diameter is the major axis of the ellipse.

        Examples
        ========

        >>> from sympy import Ellipse, Point, symbols
        >>> c = Point(1, 2)
        >>> Ellipse(c, 8, 7).auxiliary_circle()
        Circle(Point2D(1, 2), 8)
        >>> a, b = symbols('a b')
        >>> Ellipse(c, a, b).auxiliary_circle()
        Circle(Point2D(1, 2), Max(a, b))
        """
        return Circle(self.center, Max(self.hradius, self.vradius))

    def director_circle(self):
        """
        Returns a Circle consisting of all points where two perpendicular
        tangent lines to the ellipse cross each other.

        Returns
        =======

        Circle
            A director circle returned as a geometric object.

        Examples
        ========

        >>> from sympy import Ellipse, Point, symbols
        >>> c = Point(3,8)
        >>> Ellipse(c, 7, 9).director_circle()
        Circle(Point2D(3, 8), sqrt(130))
        >>> a, b = symbols('a b')
        >>> Ellipse(c, a, b).director_circle()
        Circle(Point2D(3, 8), sqrt(a**2 + b**2))

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Director_circle

        """
        return Circle(self.center, sqrt(self.hradius**2 + self.vradius**2))

    def plot_interval(self, parameter='t'):
        """The plot interval for the default geometric plot of the Ellipse.

        Parameters
        ==========

        parameter : str, optional
            Default value is 't'.

        Returns
        =======

        plot_interval : list
            [parameter, lower_bound, upper_bound]

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(0, 0), 3, 2)
        >>> e1.plot_interval()
        [t, -pi, pi]

        """
        t = _symbol(parameter, real=True)
        return [t, -S.Pi, S.Pi]

    def random_point(self, seed=None):
        """A random point on the ellipse.

        Returns
        =======

        point : Point

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(0, 0), 3, 2)
        >>> e1.random_point() # gives some random point
        Point2D(...)
        >>> p1 = e1.random_point(seed=0); p1.n(2)
        Point2D(2.1, 1.4)

        Notes
        =====

        When creating a random point, one may simply replace the
        parameter with a random number. When doing so, however, the
        random number should be made a Rational or else the point
        may not test as being in the ellipse:

        >>> from sympy.abc import t
        >>> from sympy import Rational
        >>> arb = e1.arbitrary_point(t); arb
        Point2D(3*cos(t), 2*sin(t))
        >>> arb.subs(t, .1) in e1
        False
        >>> arb.subs(t, Rational(.1)) in e1
        True
        >>> arb.subs(t, Rational('.1')) in e1
        True

        See Also
        ========
        sympy.geometry.point.Point
        arbitrary_point : Returns parameterized point on ellipse
        """
        t = _symbol('t', real=True)
        x, y = self.arbitrary_point(t).args
        # get a random value in [-1, 1) corresponding to cos(t)
        # and confirm that it will test as being in the ellipse
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        # simplify this now or else the Float will turn s into a Float
        r = Rational(rng.random())
        c = 2*r - 1
        s = sqrt(1 - c**2)
        return Point(x.subs(cos(t), c), y.subs(sin(t), s))

    def reflect(self, line):
        """Override GeometryEntity.reflect since the radius
        is not a GeometryEntity.

        Examples
        ========

        >>> from sympy import Circle, Line
        >>> Circle((0, 1), 1).reflect(Line((0, 0), (1, 1)))
        Circle(Point2D(1, 0), -1)
        >>> from sympy import Ellipse, Line, Point
        >>> Ellipse(Point(3, 4), 1, 3).reflect(Line(Point(0, -4), Point(5, 0)))
        Traceback (most recent call last):
        ...
        NotImplementedError:
        General Ellipse is not supported but the equation of the reflected
        Ellipse is given by the zeros of: f(x, y) = (9*x/41 + 40*y/41 +
        37/41)**2 + (40*x/123 - 3*y/41 - 364/123)**2 - 1

        Notes
        =====

        Until the general ellipse (with no axis parallel to the x-axis) is
        supported a NotImplemented error is raised and the equation whose
        zeros define the rotated ellipse is given.

        """

        if line.slope in (0, oo):
            c = self.center
            c = c.reflect(line)
            return self.func(c, -self.hradius, self.vradius)
        else:
            x, y = [uniquely_named_symbol(
                name, (self, line), modify=lambda s: '_' + s, real=True)
                for name in 'xy']
            expr = self.equation(x, y)
            p = Point(x, y).reflect(line)
            result = expr.subs(zip((x, y), p.args
                                   ), simultaneous=True)
            raise NotImplementedError(filldedent(
                'General Ellipse is not supported but the equation '
                'of the reflected Ellipse is given by the zeros of: ' +
                "f(%s, %s) = %s" % (str(x), str(y), str(result))))

    def rotate(self, angle=0, pt=None):
        """Rotate ``angle`` radians counterclockwise about Point ``pt``.

        Note: since the general ellipse is not supported, only rotations that
        are integer multiples of pi/2 are allowed.

        Examples
        ========

        >>> from sympy import Ellipse, pi
        >>> Ellipse((1, 0), 2, 1).rotate(pi/2)
        Ellipse(Point2D(0, 1), 1, 2)
        >>> Ellipse((1, 0), 2, 1).rotate(pi)
        Ellipse(Point2D(-1, 0), 2, 1)
        """
        if self.hradius == self.vradius:
            return self.func(self.center.rotate(angle, pt), self.hradius)
        if (angle/S.Pi).is_integer:
            return super().rotate(angle, pt)
        if (2*angle/S.Pi).is_integer:
            return self.func(self.center.rotate(angle, pt), self.vradius, self.hradius)
        # XXX see https://github.com/sympy/sympy/issues/2815 for general ellipes
        raise NotImplementedError('Only rotations of pi/2 are currently supported for Ellipse.')

    def scale(self, x=1, y=1, pt=None):
        """Override GeometryEntity.scale since it is the major and minor
        axes which must be scaled and they are not GeometryEntities.

        Examples
        ========

        >>> from sympy import Ellipse
        >>> Ellipse((0, 0), 2, 1).scale(2, 4)
        Circle(Point2D(0, 0), 4)
        >>> Ellipse((0, 0), 2, 1).scale(2)
        Ellipse(Point2D(0, 0), 4, 1)
        """
        c = self.center
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        h = self.hradius
        v = self.vradius
        return self.func(c.scale(x, y), hradius=h*x, vradius=v*y)

    def tangent_lines(self, p):
        """Tangent lines between `p` and the ellipse.

        If `p` is on the ellipse, returns the tangent line through point `p`.
        Otherwise, returns the tangent line(s) from `p` to the ellipse, or
        None if no tangent line is possible (e.g., `p` inside ellipse).

        Parameters
        ==========

        p : Point

        Returns
        =======

        tangent_lines : list with 1 or 2 Lines

        Raises
        ======

        NotImplementedError
            Can only find tangent lines for a point, `p`, on the ellipse.

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Line

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(0, 0), 3, 2)
        >>> e1.tangent_lines(Point(3, 0))
        [Line2D(Point2D(3, 0), Point2D(3, -12))]

        """
        p = Point(p, dim=2)
        if self.encloses_point(p):
            return []

        if p in self:
            delta = self.center - p
            rise = (self.vradius**2)*delta.x
            run = -(self.hradius**2)*delta.y
            p2 = Point(simplify(p.x + run),
                       simplify(p.y + rise))
            return [Line(p, p2)]
        else:
            if len(self.foci) == 2:
                f1, f2 = self.foci
                maj = self.hradius
                test = (2*maj -
                        Point.distance(f1, p) -
                        Point.distance(f2, p))
            else:
                test = self.radius - Point.distance(self.center, p)
            if test.is_number and test.is_positive:
                return []
            # else p is outside the ellipse or we can't tell. In case of the
            # latter, the solutions returned will only be valid if
            # the point is not inside the ellipse; if it is, nan will result.
            eq = self.equation(x, y)
            dydx = idiff(eq, y, x)
            slope = Line(p, Point(x, y)).slope

            # TODO: Replace solve with solveset, when this line is tested
            tangent_points = solve([slope - dydx, eq], [x, y])

            # handle horizontal and vertical tangent lines
            if len(tangent_points) == 1:
                if tangent_points[0][
                           0] == p.x or tangent_points[0][1] == p.y:
                    return [Line(p, p + Point(1, 0)), Line(p, p + Point(0, 1))]
                else:
                    return [Line(p, p + Point(0, 1)), Line(p, tangent_points[0])]

            # others
            return [Line(p, tangent_points[0]), Line(p, tangent_points[1])]

    @property
    def vradius(self):
        """The vertical radius of the ellipse.

        Returns
        =======

        vradius : number

        See Also
        ========

        hradius, major, minor

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.vradius
        1

        """
        return self.args[2]


    def second_moment_of_area(self, point=None):
        """Returns the second moment and product moment area of an ellipse.

        Parameters
        ==========

        point : Point, two-tuple of sympifiable objects, or None(default=None)
            point is the point about which second moment of area is to be found.
            If "point=None" it will be calculated about the axis passing through the
            centroid of the ellipse.

        Returns
        =======

        I_xx, I_yy, I_xy : number or SymPy expression
            I_xx, I_yy are second moment of area of an ellise.
            I_xy is product moment of area of an ellipse.

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.second_moment_of_area()
        (3*pi/4, 27*pi/4, 0)

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/List_of_second_moments_of_area

        """

        I_xx = (S.Pi*(self.hradius)*(self.vradius**3))/4
        I_yy = (S.Pi*(self.hradius**3)*(self.vradius))/4
        I_xy = 0

        if point is None:
            return I_xx, I_yy, I_xy

        # parallel axis theorem
        I_xx = I_xx + self.area*((point[1] - self.center.y)**2)
        I_yy = I_yy + self.area*((point[0] - self.center.x)**2)
        I_xy = I_xy + self.area*(point[0] - self.center.x)*(point[1] - self.center.y)

        return I_xx, I_yy, I_xy


    def polar_second_moment_of_area(self):
        """Returns the polar second moment of area of an Ellipse

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

        >>> from sympy import symbols, Circle, Ellipse
        >>> c = Circle((5, 5), 4)
        >>> c.polar_second_moment_of_area()
        128*pi
        >>> a, b = symbols('a, b')
        >>> e = Ellipse((0, 0), a, b)
        >>> e.polar_second_moment_of_area()
        pi*a**3*b/4 + pi*a*b**3/4

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Polar_moment_of_inertia

        """
        second_moment = self.second_moment_of_area()
        return second_moment[0] + second_moment[1]


    def section_modulus(self, point=None):
        """Returns a tuple with the section modulus of an ellipse

        Section modulus is a geometric property of an ellipse defined as the
        ratio of second moment of area to the distance of the extreme end of
        the ellipse from the centroidal axis.

        Parameters
        ==========

        point : Point, two-tuple of sympifyable objects, or None(default=None)
            point is the point at which section modulus is to be found.
            If "point=None" section modulus will be calculated for the
            point farthest from the centroidal axis of the ellipse.

        Returns
        =======

        S_x, S_y: numbers or SymPy expressions
                  S_x is the section modulus with respect to the x-axis
                  S_y is the section modulus with respect to the y-axis
                  A negative sign indicates that the section modulus is
                  determined for a point below the centroidal axis.

        Examples
        ========

        >>> from sympy import Symbol, Ellipse, Circle, Point2D
        >>> d = Symbol('d', positive=True)
        >>> c = Circle((0, 0), d/2)
        >>> c.section_modulus()
        (pi*d**3/32, pi*d**3/32)
        >>> e = Ellipse(Point2D(0, 0), 2, 4)
        >>> e.section_modulus()
        (8*pi, 4*pi)
        >>> e.section_modulus((2, 2))
        (16*pi, 4*pi)

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Section_modulus

        """
        x_c, y_c = self.center
        if point is None:
            # taking x and y as maximum distances from centroid
            x_min, y_min, x_max, y_max = self.bounds
            y = max(y_c - y_min, y_max - y_c)
            x = max(x_c - x_min, x_max - x_c)
        else:
            # taking x and y as distances of the given point from the center
            point = Point2D(point)
            y = point.y - y_c
            x = point.x - x_c

        second_moment = self.second_moment_of_area()
        S_x = second_moment[0]/y
        S_y = second_moment[1]/x

        return S_x, S_y


class Circle(Ellipse):
    r"""A circle in space.

    Constructed simply from a center and a radius, from three
    non-collinear points, or the equation of a circle.

    Parameters
    ==========

    center : Point
    radius : number or SymPy expression
    points : sequence of three Points
    equation : equation of a circle

    Attributes
    ==========

    radius (synonymous with hradius, vradius, major and minor)
    circumference
    equation

    Raises
    ======

    GeometryError
        When the given equation is not that of a circle.
        When trying to construct circle from incorrect parameters.

    See Also
    ========

    Ellipse, sympy.geometry.point.Point

    Examples
    ========

    >>> from sympy import Point, Circle, Eq
    >>> from sympy.abc import x, y, a, b

    A circle constructed from a center and radius:

    >>> c1 = Circle(Point(0, 0), 5)
    >>> c1.hradius, c1.vradius, c1.radius
    (5, 5, 5)

    A circle constructed from three points:

    >>> c2 = Circle(Point(0, 0), Point(1, 1), Point(1, 0))
    >>> c2.hradius, c2.vradius, c2.radius, c2.center
    (sqrt(2)/2, sqrt(2)/2, sqrt(2)/2, Point2D(1/2, 1/2))

    A circle can be constructed from an equation in the form
    `ax^2 + by^2 + gx + hy + c = 0`, too:

    >>> Circle(x**2 + y**2 - 25)
    Circle(Point2D(0, 0), 5)

    If the variables corresponding to x and y are named something
    else, their name or symbol can be supplied:

    >>> Circle(Eq(a**2 + b**2, 25), x='a', y=b)
    Circle(Point2D(0, 0), 5)
    """

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
        if len(args) == 1 and isinstance(args[0], (Expr, Eq)):
            x = kwargs.get('x', 'x')
            y = kwargs.get('y', 'y')
            equation = args[0].expand()
            if isinstance(equation, Eq):
                equation = equation.lhs - equation.rhs
            x = find(x, equation)
            y = find(y, equation)

            try:
                a, b, c, d, e = linear_coeffs(equation, x**2, y**2, x, y)
            except ValueError:
                raise GeometryError("The given equation is not that of a circle.")

            if S.Zero in (a, b) or a != b:
                raise GeometryError("The given equation is not that of a circle.")

            center_x = -c/a/2
            center_y = -d/b/2
            r2 = (center_x**2) + (center_y**2) - e/a

            return Circle((center_x, center_y), sqrt(r2), evaluate=evaluate)

        else:
            c, r = None, None
            if len(args) == 3:
                args = [Point(a, dim=2, evaluate=evaluate) for a in args]
                t = Triangle(*args)
                if not isinstance(t, Triangle):
                    return t
                c = t.circumcenter
                r = t.circumradius
            elif len(args) == 2:
                # Assume (center, radius) pair
                c = Point(args[0], dim=2, evaluate=evaluate)
                r = args[1]
                # this will prohibit imaginary radius
                try:
                    r = Point(r, 0, evaluate=evaluate).x
                except ValueError:
                    raise GeometryError("Circle with imaginary radius is not permitted")

            if not (c is None or r is None):
                if r == 0:
                    return c
                return GeometryEntity.__new__(cls, c, r, **kwargs)

            raise GeometryError("Circle.__new__ received unknown arguments")

    def _eval_evalf(self, prec=15, **options):
        pt, r = self.args
        dps = prec_to_dps(prec)
        pt = pt.evalf(n=dps, **options)
        r = r.evalf(n=dps, **options)
        return self.func(pt, r, evaluate=False)

    @property
    def circumference(self):
        """The circumference of the circle.

        Returns
        =======

        circumference : number or SymPy expression

        Examples
        ========

        >>> from sympy import Point, Circle
        >>> c1 = Circle(Point(3, 4), 6)
        >>> c1.circumference
        12*pi

        """
        return 2 * S.Pi * self.radius

    def equation(self, x='x', y='y'):
        """The equation of the circle.

        Parameters
        ==========

        x : str or Symbol, optional
            Default value is 'x'.
        y : str or Symbol, optional
            Default value is 'y'.

        Returns
        =======

        equation : SymPy expression

        Examples
        ========

        >>> from sympy import Point, Circle
        >>> c1 = Circle(Point(0, 0), 5)
        >>> c1.equation()
        x**2 + y**2 - 25

        """
        x = _symbol(x, real=True)
        y = _symbol(y, real=True)
        t1 = (x - self.center.x)**2
        t2 = (y - self.center.y)**2
        return t1 + t2 - self.major**2

    def intersection(self, o):
        """The intersection of this circle with another geometrical entity.

        Parameters
        ==========

        o : GeometryEntity

        Returns
        =======

        intersection : list of GeometryEntities

        Examples
        ========

        >>> from sympy import Point, Circle, Line, Ray
        >>> p1, p2, p3 = Point(0, 0), Point(5, 5), Point(6, 0)
        >>> p4 = Point(5, 0)
        >>> c1 = Circle(p1, 5)
        >>> c1.intersection(p2)
        []
        >>> c1.intersection(p4)
        [Point2D(5, 0)]
        >>> c1.intersection(Ray(p1, p2))
        [Point2D(5*sqrt(2)/2, 5*sqrt(2)/2)]
        >>> c1.intersection(Line(p2, p3))
        []

        """
        return Ellipse.intersection(self, o)

    @property
    def radius(self):
        """The radius of the circle.

        Returns
        =======

        radius : number or SymPy expression

        See Also
        ========

        Ellipse.major, Ellipse.minor, Ellipse.hradius, Ellipse.vradius

        Examples
        ========

        >>> from sympy import Point, Circle
        >>> c1 = Circle(Point(3, 4), 6)
        >>> c1.radius
        6

        """
        return self.args[1]

    def reflect(self, line):
        """Override GeometryEntity.reflect since the radius
        is not a GeometryEntity.

        Examples
        ========

        >>> from sympy import Circle, Line
        >>> Circle((0, 1), 1).reflect(Line((0, 0), (1, 1)))
        Circle(Point2D(1, 0), -1)
        """
        c = self.center
        c = c.reflect(line)
        return self.func(c, -self.radius)

    def scale(self, x=1, y=1, pt=None):
        """Override GeometryEntity.scale since the radius
        is not a GeometryEntity.

        Examples
        ========

        >>> from sympy import Circle
        >>> Circle((0, 0), 1).scale(2, 2)
        Circle(Point2D(0, 0), 2)
        >>> Circle((0, 0), 1).scale(2, 4)
        Ellipse(Point2D(0, 0), 2, 4)
        """
        c = self.center
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        c = c.scale(x, y)
        x, y = [abs(i) for i in (x, y)]
        if x == y:
            return self.func(c, x*self.radius)
        h = v = self.radius
        return Ellipse(c, hradius=h*x, vradius=v*y)

    @property
    def vradius(self):
        """
        This Ellipse property is an alias for the Circle's radius.

        Whereas hradius, major and minor can use Ellipse's conventions,
        the vradius does not exist for a circle. It is always a positive
        value in order that the Circle, like Polygons, will have an
        area that can be positive or negative as determined by the sign
        of the hradius.

        Examples
        ========

        >>> from sympy import Point, Circle
        >>> c1 = Circle(Point(3, 4), 6)
        >>> c1.vradius
        6
        """
        return abs(self.radius)


from .polygon import Polygon, Triangle
