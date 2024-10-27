"""Geometrical Points.

Contains
========
Point
Point2D
Point3D

When methods of Point require 1 or more points as arguments, they
can be passed as a sequence of coordinates or Points:

>>> from sympy import Point
>>> Point(1, 1).is_collinear((2, 2), (3, 4))
False
>>> Point(1, 1).is_collinear(Point(2, 2), Point(3, 4))
False

"""

import warnings

from sympy.core import S, sympify, Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.numbers import Float
from sympy.core.parameters import global_parameters
from sympy.simplify import nsimplify, simplify
from sympy.geometry.exceptions import GeometryError
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.complexes import im
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.matrices import Matrix
from sympy.matrices.expressions import Transpose
from sympy.utilities.iterables import uniq, is_sequence
from sympy.utilities.misc import filldedent, func_name, Undecidable

from .entity import GeometryEntity

from mpmath.libmp.libmpf import prec_to_dps


class Point(GeometryEntity):
    """A point in a n-dimensional Euclidean space.

    Parameters
    ==========

    coords : sequence of n-coordinate values. In the special
        case where n=2 or 3, a Point2D or Point3D will be created
        as appropriate.
    evaluate : if `True` (default), all floats are turn into
        exact types.
    dim : number of coordinates the point should have.  If coordinates
        are unspecified, they are padded with zeros.
    on_morph : indicates what should happen when the number of
        coordinates of a point need to be changed by adding or
        removing zeros.  Possible values are `'warn'`, `'error'`, or
        `ignore` (default).  No warning or error is given when `*args`
        is empty and `dim` is given. An error is always raised when
        trying to remove nonzero coordinates.


    Attributes
    ==========

    length
    origin: A `Point` representing the origin of the
        appropriately-dimensioned space.

    Raises
    ======

    TypeError : When instantiating with anything but a Point or sequence
    ValueError : when instantiating with a sequence with length < 2 or
        when trying to reduce dimensions if keyword `on_morph='error'` is
        set.

    See Also
    ========

    sympy.geometry.line.Segment : Connects two Points

    Examples
    ========

    >>> from sympy import Point
    >>> from sympy.abc import x
    >>> Point(1, 2, 3)
    Point3D(1, 2, 3)
    >>> Point([1, 2])
    Point2D(1, 2)
    >>> Point(0, x)
    Point2D(0, x)
    >>> Point(dim=4)
    Point(0, 0, 0, 0)

    Floats are automatically converted to Rational unless the
    evaluate flag is False:

    >>> Point(0.5, 0.25)
    Point2D(1/2, 1/4)
    >>> Point(0.5, 0.25, evaluate=False)
    Point2D(0.5, 0.25)

    """

    is_Point = True

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
        on_morph = kwargs.get('on_morph', 'ignore')

        # unpack into coords
        coords = args[0] if len(args) == 1 else args

        # check args and handle quickly handle Point instances
        if isinstance(coords, Point):
            # even if we're mutating the dimension of a point, we
            # don't reevaluate its coordinates
            evaluate = False
            if len(coords) == kwargs.get('dim', len(coords)):
                return coords

        if not is_sequence(coords):
            raise TypeError(filldedent('''
                Expecting sequence of coordinates, not `{}`'''
                                       .format(func_name(coords))))
        # A point where only `dim` is specified is initialized
        # to zeros.
        if len(coords) == 0 and kwargs.get('dim', None):
            coords = (S.Zero,)*kwargs.get('dim')

        coords = Tuple(*coords)
        dim = kwargs.get('dim', len(coords))

        if len(coords) < 2:
            raise ValueError(filldedent('''
                Point requires 2 or more coordinates or
                keyword `dim` > 1.'''))
        if len(coords) != dim:
            message = ("Dimension of {} needs to be changed "
                       "from {} to {}.").format(coords, len(coords), dim)
            if on_morph == 'ignore':
                pass
            elif on_morph == "error":
                raise ValueError(message)
            elif on_morph == 'warn':
                warnings.warn(message, stacklevel=2)
            else:
                raise ValueError(filldedent('''
                        on_morph value should be 'error',
                        'warn' or 'ignore'.'''))
        if any(coords[dim:]):
            raise ValueError('Nonzero coordinates cannot be removed.')
        if any(a.is_number and im(a).is_zero is False for a in coords):
            raise ValueError('Imaginary coordinates are not permitted.')
        if not all(isinstance(a, Expr) for a in coords):
            raise TypeError('Coordinates must be valid SymPy expressions.')

        # pad with zeros appropriately
        coords = coords[:dim] + (S.Zero,)*(dim - len(coords))

        # Turn any Floats into rationals and simplify
        # any expressions before we instantiate
        if evaluate:
            coords = coords.xreplace({
                f: simplify(nsimplify(f, rational=True))
                 for f in coords.atoms(Float)})

        # return 2D or 3D instances
        if len(coords) == 2:
            kwargs['_nocheck'] = True
            return Point2D(*coords, **kwargs)
        elif len(coords) == 3:
            kwargs['_nocheck'] = True
            return Point3D(*coords, **kwargs)

        # the general Point
        return GeometryEntity.__new__(cls, *coords)

    def __abs__(self):
        """Returns the distance between this point and the origin."""
        origin = Point([0]*len(self))
        return Point.distance(origin, self)

    def __add__(self, other):
        """Add other to self by incrementing self's coordinates by
        those of other.

        Notes
        =====

        >>> from sympy import Point

        When sequences of coordinates are passed to Point methods, they
        are converted to a Point internally. This __add__ method does
        not do that so if floating point values are used, a floating
        point result (in terms of SymPy Floats) will be returned.

        >>> Point(1, 2) + (.1, .2)
        Point2D(1.1, 2.2)

        If this is not desired, the `translate` method can be used or
        another Point can be added:

        >>> Point(1, 2).translate(.1, .2)
        Point2D(11/10, 11/5)
        >>> Point(1, 2) + Point(.1, .2)
        Point2D(11/10, 11/5)

        See Also
        ========

        sympy.geometry.point.Point.translate

        """
        try:
            s, o = Point._normalize_dimension(self, Point(other, evaluate=False))
        except TypeError:
            raise GeometryError("Don't know how to add {} and a Point object".format(other))

        coords = [simplify(a + b) for a, b in zip(s, o)]
        return Point(coords, evaluate=False)

    def __contains__(self, item):
        return item in self.args

    def __truediv__(self, divisor):
        """Divide point's coordinates by a factor."""
        divisor = sympify(divisor)
        coords = [simplify(x/divisor) for x in self.args]
        return Point(coords, evaluate=False)

    def __eq__(self, other):
        if not isinstance(other, Point) or len(self.args) != len(other.args):
            return False
        return self.args == other.args

    def __getitem__(self, key):
        return self.args[key]

    def __hash__(self):
        return hash(self.args)

    def __iter__(self):
        return self.args.__iter__()

    def __len__(self):
        return len(self.args)

    def __mul__(self, factor):
        """Multiply point's coordinates by a factor.

        Notes
        =====

        >>> from sympy import Point

        When multiplying a Point by a floating point number,
        the coordinates of the Point will be changed to Floats:

        >>> Point(1, 2)*0.1
        Point2D(0.1, 0.2)

        If this is not desired, the `scale` method can be used or
        else only multiply or divide by integers:

        >>> Point(1, 2).scale(1.1, 1.1)
        Point2D(11/10, 11/5)
        >>> Point(1, 2)*11/10
        Point2D(11/10, 11/5)

        See Also
        ========

        sympy.geometry.point.Point.scale
        """
        factor = sympify(factor)
        coords = [simplify(x*factor) for x in self.args]
        return Point(coords, evaluate=False)

    def __rmul__(self, factor):
        """Multiply a factor by point's coordinates."""
        return self.__mul__(factor)

    def __neg__(self):
        """Negate the point."""
        coords = [-x for x in self.args]
        return Point(coords, evaluate=False)

    def __sub__(self, other):
        """Subtract two points, or subtract a factor from this point's
        coordinates."""
        return self + [-x for x in other]

    @classmethod
    def _normalize_dimension(cls, *points, **kwargs):
        """Ensure that points have the same dimension.
        By default `on_morph='warn'` is passed to the
        `Point` constructor."""
        # if we have a built-in ambient dimension, use it
        dim = getattr(cls, '_ambient_dimension', None)
        # override if we specified it
        dim = kwargs.get('dim', dim)
        # if no dim was given, use the highest dimensional point
        if dim is None:
            dim = max(i.ambient_dimension for i in points)
        if all(i.ambient_dimension == dim for i in points):
            return list(points)
        kwargs['dim'] = dim
        kwargs['on_morph'] = kwargs.get('on_morph', 'warn')
        return [Point(i, **kwargs) for i in points]

    @staticmethod
    def affine_rank(*args):
        """The affine rank of a set of points is the dimension
        of the smallest affine space containing all the points.
        For example, if the points lie on a line (and are not all
        the same) their affine rank is 1.  If the points lie on a plane
        but not a line, their affine rank is 2.  By convention, the empty
        set has affine rank -1."""

        if len(args) == 0:
            return -1
        # make sure we're genuinely points
        # and translate every point to the origin
        points = Point._normalize_dimension(*[Point(i) for i in args])
        origin = points[0]
        points = [i - origin for i in points[1:]]

        m = Matrix([i.args for i in points])
        # XXX fragile -- what is a better way?
        return m.rank(iszerofunc = lambda x:
            abs(x.n(2)) < 1e-12 if x.is_number else x.is_zero)

    @property
    def ambient_dimension(self):
        """Number of components this point has."""
        return getattr(self, '_ambient_dimension', len(self))

    @classmethod
    def are_coplanar(cls, *points):
        """Return True if there exists a plane in which all the points
        lie.  A trivial True value is returned if `len(points) < 3` or
        all Points are 2-dimensional.

        Parameters
        ==========

        A set of points

        Raises
        ======

        ValueError : if less than 3 unique points are given

        Returns
        =======

        boolean

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1 = Point3D(1, 2, 2)
        >>> p2 = Point3D(2, 7, 2)
        >>> p3 = Point3D(0, 0, 2)
        >>> p4 = Point3D(1, 1, 2)
        >>> Point3D.are_coplanar(p1, p2, p3, p4)
        True
        >>> p5 = Point3D(0, 1, 3)
        >>> Point3D.are_coplanar(p1, p2, p3, p5)
        False

        """
        if len(points) <= 1:
            return True

        points = cls._normalize_dimension(*[Point(i) for i in points])
        # quick exit if we are in 2D
        if points[0].ambient_dimension == 2:
            return True
        points = list(uniq(points))
        return Point.affine_rank(*points) <= 2

    def distance(self, other):
        """The Euclidean distance between self and another GeometricEntity.

        Returns
        =======

        distance : number or symbolic expression.

        Raises
        ======

        TypeError : if other is not recognized as a GeometricEntity or is a
                    GeometricEntity for which distance is not defined.

        See Also
        ========

        sympy.geometry.line.Segment.length
        sympy.geometry.point.Point.taxicab_distance

        Examples
        ========

        >>> from sympy import Point, Line
        >>> p1, p2 = Point(1, 1), Point(4, 5)
        >>> l = Line((3, 1), (2, 2))
        >>> p1.distance(p2)
        5
        >>> p1.distance(l)
        sqrt(2)

        The computed distance may be symbolic, too:

        >>> from sympy.abc import x, y
        >>> p3 = Point(x, y)
        >>> p3.distance((0, 0))
        sqrt(x**2 + y**2)

        """
        if not isinstance(other, GeometryEntity):
            try:
                other = Point(other, dim=self.ambient_dimension)
            except TypeError:
                raise TypeError("not recognized as a GeometricEntity: %s" % type(other))
        if isinstance(other, Point):
            s, p = Point._normalize_dimension(self, Point(other))
            return sqrt(Add(*((a - b)**2 for a, b in zip(s, p))))
        distance = getattr(other, 'distance', None)
        if distance is None:
            raise TypeError("distance between Point and %s is not defined" % type(other))
        return distance(self)

    def dot(self, p):
        """Return dot product of self with another Point."""
        if not is_sequence(p):
            p = Point(p)  # raise the error via Point
        return Add(*(a*b for a, b in zip(self, p)))

    def equals(self, other):
        """Returns whether the coordinates of self and other agree."""
        # a point is equal to another point if all its components are equal
        if not isinstance(other, Point) or len(self) != len(other):
            return False
        return all(a.equals(b) for a, b in zip(self, other))

    def _eval_evalf(self, prec=15, **options):
        """Evaluate the coordinates of the point.

        This method will, where possible, create and return a new Point
        where the coordinates are evaluated as floating point numbers to
        the precision indicated (default=15).

        Parameters
        ==========

        prec : int

        Returns
        =======

        point : Point

        Examples
        ========

        >>> from sympy import Point, Rational
        >>> p1 = Point(Rational(1, 2), Rational(3, 2))
        >>> p1
        Point2D(1/2, 3/2)
        >>> p1.evalf()
        Point2D(0.5, 1.5)

        """
        dps = prec_to_dps(prec)
        coords = [x.evalf(n=dps, **options) for x in self.args]
        return Point(*coords, evaluate=False)

    def intersection(self, other):
        """The intersection between this point and another GeometryEntity.

        Parameters
        ==========

        other : GeometryEntity or sequence of coordinates

        Returns
        =======

        intersection : list of Points

        Notes
        =====

        The return value will either be an empty list if there is no
        intersection, otherwise it will contain this point.

        Examples
        ========

        >>> from sympy import Point
        >>> p1, p2, p3 = Point(0, 0), Point(1, 1), Point(0, 0)
        >>> p1.intersection(p2)
        []
        >>> p1.intersection(p3)
        [Point2D(0, 0)]

        """
        if not isinstance(other, GeometryEntity):
            other = Point(other)
        if isinstance(other, Point):
            if self == other:
                return [self]
            p1, p2 = Point._normalize_dimension(self, other)
            if p1 == self and p1 == p2:
                return [self]
            return []
        return other.intersection(self)

    def is_collinear(self, *args):
        """Returns `True` if there exists a line
        that contains `self` and `points`.  Returns `False` otherwise.
        A trivially True value is returned if no points are given.

        Parameters
        ==========

        args : sequence of Points

        Returns
        =======

        is_collinear : boolean

        See Also
        ========

        sympy.geometry.line.Line

        Examples
        ========

        >>> from sympy import Point
        >>> from sympy.abc import x
        >>> p1, p2 = Point(0, 0), Point(1, 1)
        >>> p3, p4, p5 = Point(2, 2), Point(x, x), Point(1, 2)
        >>> Point.is_collinear(p1, p2, p3, p4)
        True
        >>> Point.is_collinear(p1, p2, p3, p5)
        False

        """
        points = (self,) + args
        points = Point._normalize_dimension(*[Point(i) for i in points])
        points = list(uniq(points))
        return Point.affine_rank(*points) <= 1

    def is_concyclic(self, *args):
        """Do `self` and the given sequence of points lie in a circle?

        Returns True if the set of points are concyclic and
        False otherwise. A trivial value of True is returned
        if there are fewer than 2 other points.

        Parameters
        ==========

        args : sequence of Points

        Returns
        =======

        is_concyclic : boolean


        Examples
        ========

        >>> from sympy import Point

        Define 4 points that are on the unit circle:

        >>> p1, p2, p3, p4 = Point(1, 0), (0, 1), (-1, 0), (0, -1)

        >>> p1.is_concyclic() == p1.is_concyclic(p2, p3, p4) == True
        True

        Define a point not on that circle:

        >>> p = Point(1, 1)

        >>> p.is_concyclic(p1, p2, p3)
        False

        """
        points = (self,) + args
        points = Point._normalize_dimension(*[Point(i) for i in points])
        points = list(uniq(points))
        if not Point.affine_rank(*points) <= 2:
            return False
        origin = points[0]
        points = [p - origin for p in points]
        # points are concyclic if they are coplanar and
        # there is a point c so that ||p_i-c|| == ||p_j-c|| for all
        # i and j.  Rearranging this equation gives us the following
        # condition: the matrix `mat` must not a pivot in the last
        # column.
        mat = Matrix([list(i) + [i.dot(i)] for i in points])
        rref, pivots = mat.rref()
        if len(origin) not in pivots:
            return True
        return False

    @property
    def is_nonzero(self):
        """True if any coordinate is nonzero, False if every coordinate is zero,
        and None if it cannot be determined."""
        is_zero = self.is_zero
        if is_zero is None:
            return None
        return not is_zero

    def is_scalar_multiple(self, p):
        """Returns whether each coordinate of `self` is a scalar
        multiple of the corresponding coordinate in point p.
        """
        s, o = Point._normalize_dimension(self, Point(p))
        # 2d points happen a lot, so optimize this function call
        if s.ambient_dimension == 2:
            (x1, y1), (x2, y2) = s.args, o.args
            rv = (x1*y2 - x2*y1).equals(0)
            if rv is None:
                raise Undecidable(filldedent(
                    '''Cannot determine if %s is a scalar multiple of
                    %s''' % (s, o)))

        # if the vectors p1 and p2 are linearly dependent, then they must
        # be scalar multiples of each other
        m = Matrix([s.args, o.args])
        return m.rank() < 2

    @property
    def is_zero(self):
        """True if every coordinate is zero, False if any coordinate is not zero,
        and None if it cannot be determined."""
        nonzero = [x.is_nonzero for x in self.args]
        if any(nonzero):
            return False
        if any(x is None for x in nonzero):
            return None
        return True

    @property
    def length(self):
        """
        Treating a Point as a Line, this returns 0 for the length of a Point.

        Examples
        ========

        >>> from sympy import Point
        >>> p = Point(0, 1)
        >>> p.length
        0
        """
        return S.Zero

    def midpoint(self, p):
        """The midpoint between self and point p.

        Parameters
        ==========

        p : Point

        Returns
        =======

        midpoint : Point

        See Also
        ========

        sympy.geometry.line.Segment.midpoint

        Examples
        ========

        >>> from sympy import Point
        >>> p1, p2 = Point(1, 1), Point(13, 5)
        >>> p1.midpoint(p2)
        Point2D(7, 3)

        """
        s, p = Point._normalize_dimension(self, Point(p))
        return Point([simplify((a + b)*S.Half) for a, b in zip(s, p)])

    @property
    def origin(self):
        """A point of all zeros of the same ambient dimension
        as the current point"""
        return Point([0]*len(self), evaluate=False)

    @property
    def orthogonal_direction(self):
        """Returns a non-zero point that is orthogonal to the
        line containing `self` and the origin.

        Examples
        ========

        >>> from sympy import Line, Point
        >>> a = Point(1, 2, 3)
        >>> a.orthogonal_direction
        Point3D(-2, 1, 0)
        >>> b = _
        >>> Line(b, b.origin).is_perpendicular(Line(a, a.origin))
        True
        """
        dim = self.ambient_dimension
        # if a coordinate is zero, we can put a 1 there and zeros elsewhere
        if self[0].is_zero:
            return Point([1] + (dim - 1)*[0])
        if self[1].is_zero:
            return Point([0,1] + (dim - 2)*[0])
        # if the first two coordinates aren't zero, we can create a non-zero
        # orthogonal vector by swapping them, negating one, and padding with zeros
        return Point([-self[1], self[0]] + (dim - 2)*[0])

    @staticmethod
    def project(a, b):
        """Project the point `a` onto the line between the origin
        and point `b` along the normal direction.

        Parameters
        ==========

        a : Point
        b : Point

        Returns
        =======

        p : Point

        See Also
        ========

        sympy.geometry.line.LinearEntity.projection

        Examples
        ========

        >>> from sympy import Line, Point
        >>> a = Point(1, 2)
        >>> b = Point(2, 5)
        >>> z = a.origin
        >>> p = Point.project(a, b)
        >>> Line(p, a).is_perpendicular(Line(p, b))
        True
        >>> Point.is_collinear(z, p, b)
        True
        """
        a, b = Point._normalize_dimension(Point(a), Point(b))
        if b.is_zero:
            raise ValueError("Cannot project to the zero vector.")
        return b*(a.dot(b) / b.dot(b))

    def taxicab_distance(self, p):
        """The Taxicab Distance from self to point p.

        Returns the sum of the horizontal and vertical distances to point p.

        Parameters
        ==========

        p : Point

        Returns
        =======

        taxicab_distance : The sum of the horizontal
        and vertical distances to point p.

        See Also
        ========

        sympy.geometry.point.Point.distance

        Examples
        ========

        >>> from sympy import Point
        >>> p1, p2 = Point(1, 1), Point(4, 5)
        >>> p1.taxicab_distance(p2)
        7

        """
        s, p = Point._normalize_dimension(self, Point(p))
        return Add(*(abs(a - b) for a, b in zip(s, p)))

    def canberra_distance(self, p):
        """The Canberra Distance from self to point p.

        Returns the weighted sum of horizontal and vertical distances to
        point p.

        Parameters
        ==========

        p : Point

        Returns
        =======

        canberra_distance : The weighted sum of horizontal and vertical
        distances to point p. The weight used is the sum of absolute values
        of the coordinates.

        Examples
        ========

        >>> from sympy import Point
        >>> p1, p2 = Point(1, 1), Point(3, 3)
        >>> p1.canberra_distance(p2)
        1
        >>> p1, p2 = Point(0, 0), Point(3, 3)
        >>> p1.canberra_distance(p2)
        2

        Raises
        ======

        ValueError when both vectors are zero.

        See Also
        ========

        sympy.geometry.point.Point.distance

        """

        s, p = Point._normalize_dimension(self, Point(p))
        if self.is_zero and p.is_zero:
            raise ValueError("Cannot project to the zero vector.")
        return Add(*((abs(a - b)/(abs(a) + abs(b))) for a, b in zip(s, p)))

    @property
    def unit(self):
        """Return the Point that is in the same direction as `self`
        and a distance of 1 from the origin"""
        return self / abs(self)


class Point2D(Point):
    """A point in a 2-dimensional Euclidean space.

    Parameters
    ==========

    coords
        A sequence of 2 coordinate values.

    Attributes
    ==========

    x
    y
    length

    Raises
    ======

    TypeError
        When trying to add or subtract points with different dimensions.
        When trying to create a point with more than two dimensions.
        When `intersection` is called with object other than a Point.

    See Also
    ========

    sympy.geometry.line.Segment : Connects two Points

    Examples
    ========

    >>> from sympy import Point2D
    >>> from sympy.abc import x
    >>> Point2D(1, 2)
    Point2D(1, 2)
    >>> Point2D([1, 2])
    Point2D(1, 2)
    >>> Point2D(0, x)
    Point2D(0, x)

    Floats are automatically converted to Rational unless the
    evaluate flag is False:

    >>> Point2D(0.5, 0.25)
    Point2D(1/2, 1/4)
    >>> Point2D(0.5, 0.25, evaluate=False)
    Point2D(0.5, 0.25)

    """

    _ambient_dimension = 2

    def __new__(cls, *args, _nocheck=False, **kwargs):
        if not _nocheck:
            kwargs['dim'] = 2
            args = Point(*args, **kwargs)
        return GeometryEntity.__new__(cls, *args)

    def __contains__(self, item):
        return item == self

    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.

        """

        return (self.x, self.y, self.x, self.y)

    def rotate(self, angle, pt=None):
        """Rotate ``angle`` radians counterclockwise about Point ``pt``.

        See Also
        ========

        translate, scale

        Examples
        ========

        >>> from sympy import Point2D, pi
        >>> t = Point2D(1, 0)
        >>> t.rotate(pi/2)
        Point2D(0, 1)
        >>> t.rotate(pi/2, (2, 0))
        Point2D(2, -1)

        """
        c = cos(angle)
        s = sin(angle)

        rv = self
        if pt is not None:
            pt = Point(pt, dim=2)
            rv -= pt
        x, y = rv.args
        rv = Point(c*x - s*y, s*x + c*y)
        if pt is not None:
            rv += pt
        return rv

    def scale(self, x=1, y=1, pt=None):
        """Scale the coordinates of the Point by multiplying by
        ``x`` and ``y`` after subtracting ``pt`` -- default is (0, 0) --
        and then adding ``pt`` back again (i.e. ``pt`` is the point of
        reference for the scaling).

        See Also
        ========

        rotate, translate

        Examples
        ========

        >>> from sympy import Point2D
        >>> t = Point2D(1, 1)
        >>> t.scale(2)
        Point2D(2, 1)
        >>> t.scale(2, 2)
        Point2D(2, 2)

        """
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        return Point(self.x*x, self.y*y)

    def transform(self, matrix):
        """Return the point after applying the transformation described
        by the 3x3 Matrix, ``matrix``.

        See Also
        ========
        sympy.geometry.point.Point2D.rotate
        sympy.geometry.point.Point2D.scale
        sympy.geometry.point.Point2D.translate
        """
        if not (matrix.is_Matrix and matrix.shape == (3, 3)):
            raise ValueError("matrix must be a 3x3 matrix")
        x, y = self.args
        return Point(*(Matrix(1, 3, [x, y, 1])*matrix).tolist()[0][:2])

    def translate(self, x=0, y=0):
        """Shift the Point by adding x and y to the coordinates of the Point.

        See Also
        ========

        sympy.geometry.point.Point2D.rotate, scale

        Examples
        ========

        >>> from sympy import Point2D
        >>> t = Point2D(0, 1)
        >>> t.translate(2)
        Point2D(2, 1)
        >>> t.translate(2, 2)
        Point2D(2, 3)
        >>> t + Point2D(2, 2)
        Point2D(2, 3)

        """
        return Point(self.x + x, self.y + y)

    @property
    def coordinates(self):
        """
        Returns the two coordinates of the Point.

        Examples
        ========

        >>> from sympy import Point2D
        >>> p = Point2D(0, 1)
        >>> p.coordinates
        (0, 1)
        """
        return self.args

    @property
    def x(self):
        """
        Returns the X coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point2D
        >>> p = Point2D(0, 1)
        >>> p.x
        0
        """
        return self.args[0]

    @property
    def y(self):
        """
        Returns the Y coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point2D
        >>> p = Point2D(0, 1)
        >>> p.y
        1
        """
        return self.args[1]

class Point3D(Point):
    """A point in a 3-dimensional Euclidean space.

    Parameters
    ==========

    coords
        A sequence of 3 coordinate values.

    Attributes
    ==========

    x
    y
    z
    length

    Raises
    ======

    TypeError
        When trying to add or subtract points with different dimensions.
        When `intersection` is called with object other than a Point.

    Examples
    ========

    >>> from sympy import Point3D
    >>> from sympy.abc import x
    >>> Point3D(1, 2, 3)
    Point3D(1, 2, 3)
    >>> Point3D([1, 2, 3])
    Point3D(1, 2, 3)
    >>> Point3D(0, x, 3)
    Point3D(0, x, 3)

    Floats are automatically converted to Rational unless the
    evaluate flag is False:

    >>> Point3D(0.5, 0.25, 2)
    Point3D(1/2, 1/4, 2)
    >>> Point3D(0.5, 0.25, 3, evaluate=False)
    Point3D(0.5, 0.25, 3)

    """

    _ambient_dimension = 3

    def __new__(cls, *args, _nocheck=False, **kwargs):
        if not _nocheck:
            kwargs['dim'] = 3
            args = Point(*args, **kwargs)
        return GeometryEntity.__new__(cls, *args)

    def __contains__(self, item):
        return item == self

    @staticmethod
    def are_collinear(*points):
        """Is a sequence of points collinear?

        Test whether or not a set of points are collinear. Returns True if
        the set of points are collinear, or False otherwise.

        Parameters
        ==========

        points : sequence of Point

        Returns
        =======

        are_collinear : boolean

        See Also
        ========

        sympy.geometry.line.Line3D

        Examples
        ========

        >>> from sympy import Point3D
        >>> from sympy.abc import x
        >>> p1, p2 = Point3D(0, 0, 0), Point3D(1, 1, 1)
        >>> p3, p4, p5 = Point3D(2, 2, 2), Point3D(x, x, x), Point3D(1, 2, 6)
        >>> Point3D.are_collinear(p1, p2, p3, p4)
        True
        >>> Point3D.are_collinear(p1, p2, p3, p5)
        False
        """
        return Point.is_collinear(*points)

    def direction_cosine(self, point):
        """
        Gives the direction cosine between 2 points

        Parameters
        ==========

        p : Point3D

        Returns
        =======

        list

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1 = Point3D(1, 2, 3)
        >>> p1.direction_cosine(Point3D(2, 3, 5))
        [sqrt(6)/6, sqrt(6)/6, sqrt(6)/3]
        """
        a = self.direction_ratio(point)
        b = sqrt(Add(*(i**2 for i in a)))
        return [(point.x - self.x) / b,(point.y - self.y) / b,
                (point.z - self.z) / b]

    def direction_ratio(self, point):
        """
        Gives the direction ratio between 2 points

        Parameters
        ==========

        p : Point3D

        Returns
        =======

        list

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1 = Point3D(1, 2, 3)
        >>> p1.direction_ratio(Point3D(2, 3, 5))
        [1, 1, 2]
        """
        return [(point.x - self.x),(point.y - self.y),(point.z - self.z)]

    def intersection(self, other):
        """The intersection between this point and another GeometryEntity.

        Parameters
        ==========

        other : GeometryEntity or sequence of coordinates

        Returns
        =======

        intersection : list of Points

        Notes
        =====

        The return value will either be an empty list if there is no
        intersection, otherwise it will contain this point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p1, p2, p3 = Point3D(0, 0, 0), Point3D(1, 1, 1), Point3D(0, 0, 0)
        >>> p1.intersection(p2)
        []
        >>> p1.intersection(p3)
        [Point3D(0, 0, 0)]

        """
        if not isinstance(other, GeometryEntity):
            other = Point(other, dim=3)
        if isinstance(other, Point3D):
            if self == other:
                return [self]
            return []
        return other.intersection(self)

    def scale(self, x=1, y=1, z=1, pt=None):
        """Scale the coordinates of the Point by multiplying by
        ``x`` and ``y`` after subtracting ``pt`` -- default is (0, 0) --
        and then adding ``pt`` back again (i.e. ``pt`` is the point of
        reference for the scaling).

        See Also
        ========

        translate

        Examples
        ========

        >>> from sympy import Point3D
        >>> t = Point3D(1, 1, 1)
        >>> t.scale(2)
        Point3D(2, 1, 1)
        >>> t.scale(2, 2)
        Point3D(2, 2, 1)

        """
        if pt:
            pt = Point3D(pt)
            return self.translate(*(-pt).args).scale(x, y, z).translate(*pt.args)
        return Point3D(self.x*x, self.y*y, self.z*z)

    def transform(self, matrix):
        """Return the point after applying the transformation described
        by the 4x4 Matrix, ``matrix``.

        See Also
        ========
        sympy.geometry.point.Point3D.scale
        sympy.geometry.point.Point3D.translate
        """
        if not (matrix.is_Matrix and matrix.shape == (4, 4)):
            raise ValueError("matrix must be a 4x4 matrix")
        x, y, z = self.args
        m = Transpose(matrix)
        return Point3D(*(Matrix(1, 4, [x, y, z, 1])*m).tolist()[0][:3])

    def translate(self, x=0, y=0, z=0):
        """Shift the Point by adding x and y to the coordinates of the Point.

        See Also
        ========

        scale

        Examples
        ========

        >>> from sympy import Point3D
        >>> t = Point3D(0, 1, 1)
        >>> t.translate(2)
        Point3D(2, 1, 1)
        >>> t.translate(2, 2)
        Point3D(2, 3, 1)
        >>> t + Point3D(2, 2, 2)
        Point3D(2, 3, 3)

        """
        return Point3D(self.x + x, self.y + y, self.z + z)

    @property
    def coordinates(self):
        """
        Returns the three coordinates of the Point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 2)
        >>> p.coordinates
        (0, 1, 2)
        """
        return self.args

    @property
    def x(self):
        """
        Returns the X coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 3)
        >>> p.x
        0
        """
        return self.args[0]

    @property
    def y(self):
        """
        Returns the Y coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 2)
        >>> p.y
        1
        """
        return self.args[1]

    @property
    def z(self):
        """
        Returns the Z coordinate of the Point.

        Examples
        ========

        >>> from sympy import Point3D
        >>> p = Point3D(0, 1, 1)
        >>> p.z
        1
        """
        return self.args[2]
