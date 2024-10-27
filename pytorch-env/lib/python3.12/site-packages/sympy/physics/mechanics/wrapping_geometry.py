"""Geometry objects for use by wrapping pathways."""

from abc import ABC, abstractmethod

from sympy import Integer, acos, pi, sqrt, sympify, tan
from sympy.core.relational import Eq
from sympy.functions.elementary.trigonometric import atan2
from sympy.polys.polytools import cancel
from sympy.physics.vector import Vector, dot
from sympy.simplify.simplify import trigsimp


__all__ = [
    'WrappingGeometryBase',
    'WrappingCylinder',
    'WrappingSphere',
]


class WrappingGeometryBase(ABC):
    """Abstract base class for all geometry classes to inherit from.

    Notes
    =====

    Instances of this class cannot be directly instantiated by users. However,
    it can be used to created custom geometry types through subclassing.

    """

    @property
    @abstractmethod
    def point(cls):
        """The point with which the geometry is associated."""
        pass

    @abstractmethod
    def point_on_surface(self, point):
        """Returns ``True`` if a point is on the geometry's surface.

        Parameters
        ==========
        point : Point
            The point for which it's to be ascertained if it's on the
            geometry's surface or not.

        """
        pass

    @abstractmethod
    def geodesic_length(self, point_1, point_2):
        """Returns the shortest distance between two points on a geometry's
        surface.

        Parameters
        ==========

        point_1 : Point
            The point from which the geodesic length should be calculated.
        point_2 : Point
            The point to which the geodesic length should be calculated.

        """
        pass

    @abstractmethod
    def geodesic_end_vectors(self, point_1, point_2):
        """The vectors parallel to the geodesic at the two end points.

        Parameters
        ==========

        point_1 : Point
            The point from which the geodesic originates.
        point_2 : Point
            The point at which the geodesic terminates.

        """
        pass

    def __repr__(self):
        """Default representation of a geometry object."""
        return f'{self.__class__.__name__}()'


class WrappingSphere(WrappingGeometryBase):
    """A solid spherical object.

    Explanation
    ===========

    A wrapping geometry that allows for circular arcs to be defined between
    pairs of points. These paths are always geodetic (the shortest possible).

    Examples
    ========

    To create a ``WrappingSphere`` instance, a ``Symbol`` denoting its radius
    and ``Point`` at which its center will be located are needed:

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import Point, WrappingSphere
    >>> r = symbols('r')
    >>> pO = Point('pO')

    A sphere with radius ``r`` centered on ``pO`` can be instantiated with:

    >>> WrappingSphere(r, pO)
    WrappingSphere(radius=r, point=pO)

    Parameters
    ==========

    radius : Symbol
        Radius of the sphere. This symbol must represent a value that is
        positive and constant, i.e. it cannot be a dynamic symbol, nor can it
        be an expression.
    point : Point
        A point at which the sphere is centered.

    See Also
    ========

    WrappingCylinder: Cylindrical geometry where the wrapping direction can be
        defined.

    """

    def __init__(self, radius, point):
        """Initializer for ``WrappingSphere``.

        Parameters
        ==========

        radius : Symbol
            The radius of the sphere.
        point : Point
            A point on which the sphere is centered.

        """
        self.radius = radius
        self.point = point

    @property
    def radius(self):
        """Radius of the sphere."""
        return self._radius

    @radius.setter
    def radius(self, radius):
        self._radius = radius

    @property
    def point(self):
        """A point on which the sphere is centered."""
        return self._point

    @point.setter
    def point(self, point):
        self._point = point

    def point_on_surface(self, point):
        """Returns ``True`` if a point is on the sphere's surface.

        Parameters
        ==========

        point : Point
            The point for which it's to be ascertained if it's on the sphere's
            surface or not. This point's position relative to the sphere's
            center must be a simple expression involving the radius of the
            sphere, otherwise this check will likely not work.

        """
        point_vector = point.pos_from(self.point)
        if isinstance(point_vector, Vector):
            point_radius_squared = dot(point_vector, point_vector)
        else:
            point_radius_squared = point_vector**2
        return Eq(point_radius_squared, self.radius**2) == True

    def geodesic_length(self, point_1, point_2):
        r"""Returns the shortest distance between two points on the sphere's
        surface.

        Explanation
        ===========

        The geodesic length, i.e. the shortest arc along the surface of a
        sphere, connecting two points can be calculated using the formula:

        .. math::

           l = \arccos\left(\mathbf{v}_1 \cdot \mathbf{v}_2\right)

        where $\mathbf{v}_1$ and $\mathbf{v}_2$ are the unit vectors from the
        sphere's center to the first and second points on the sphere's surface
        respectively. Note that the actual path that the geodesic will take is
        undefined when the two points are directly opposite one another.

        Examples
        ========

        A geodesic length can only be calculated between two points on the
        sphere's surface. Firstly, a ``WrappingSphere`` instance must be
        created along with two points that will lie on its surface:

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import (Point, ReferenceFrame,
        ...     WrappingSphere)
        >>> N = ReferenceFrame('N')
        >>> r = symbols('r')
        >>> pO = Point('pO')
        >>> pO.set_vel(N, 0)
        >>> sphere = WrappingSphere(r, pO)
        >>> p1 = Point('p1')
        >>> p2 = Point('p2')

        Let's assume that ``p1`` lies at a distance of ``r`` in the ``N.x``
        direction from ``pO`` and that ``p2`` is located on the sphere's
        surface in the ``N.y + N.z`` direction from ``pO``. These positions can
        be set with:

        >>> p1.set_pos(pO, r*N.x)
        >>> p1.pos_from(pO)
        r*N.x
        >>> p2.set_pos(pO, r*(N.y + N.z).normalize())
        >>> p2.pos_from(pO)
        sqrt(2)*r/2*N.y + sqrt(2)*r/2*N.z

        The geodesic length, which is in this case is a quarter of the sphere's
        circumference, can be calculated using the ``geodesic_length`` method:

        >>> sphere.geodesic_length(p1, p2)
        pi*r/2

        If the ``geodesic_length`` method is passed an argument, the ``Point``
        that doesn't lie on the sphere's surface then a ``ValueError`` is
        raised because it's not possible to calculate a value in this case.

        Parameters
        ==========

        point_1 : Point
            Point from which the geodesic length should be calculated.
        point_2 : Point
            Point to which the geodesic length should be calculated.

        """
        for point in (point_1, point_2):
            if not self.point_on_surface(point):
                msg = (
                    f'Geodesic length cannot be calculated as point {point} '
                    f'with radius {point.pos_from(self.point).magnitude()} '
                    f'from the sphere\'s center {self.point} does not lie on '
                    f'the surface of {self} with radius {self.radius}.'
                )
                raise ValueError(msg)
        point_1_vector = point_1.pos_from(self.point).normalize()
        point_2_vector = point_2.pos_from(self.point).normalize()
        central_angle = acos(point_2_vector.dot(point_1_vector))
        geodesic_length = self.radius*central_angle
        return geodesic_length

    def geodesic_end_vectors(self, point_1, point_2):
        """The vectors parallel to the geodesic at the two end points.

        Parameters
        ==========

        point_1 : Point
            The point from which the geodesic originates.
        point_2 : Point
            The point at which the geodesic terminates.

        """
        pA, pB = point_1, point_2
        pO = self.point
        pA_vec = pA.pos_from(pO)
        pB_vec = pB.pos_from(pO)

        if pA_vec.cross(pB_vec) == 0:
            msg = (
                f'Can\'t compute geodesic end vectors for the pair of points '
                f'{pA} and {pB} on a sphere {self} as they are diametrically '
                f'opposed, thus the geodesic is not defined.'
            )
            raise ValueError(msg)

        return (
            pA_vec.cross(pB.pos_from(pA)).cross(pA_vec).normalize(),
            pB_vec.cross(pA.pos_from(pB)).cross(pB_vec).normalize(),
        )

    def __repr__(self):
        """Representation of a ``WrappingSphere``."""
        return (
            f'{self.__class__.__name__}(radius={self.radius}, '
            f'point={self.point})'
        )


class WrappingCylinder(WrappingGeometryBase):
    """A solid (infinite) cylindrical object.

    Explanation
    ===========

    A wrapping geometry that allows for circular arcs to be defined between
    pairs of points. These paths are always geodetic (the shortest possible) in
    the sense that they will be a straight line on the unwrapped cylinder's
    surface. However, it is also possible for a direction to be specified, i.e.
    paths can be influenced such that they either wrap along the shortest side
    or the longest side of the cylinder. To define these directions, rotations
    are in the positive direction following the right-hand rule.

    Examples
    ========

    To create a ``WrappingCylinder`` instance, a ``Symbol`` denoting its
    radius, a ``Vector`` defining its axis, and a ``Point`` through which its
    axis passes are needed:

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (Point, ReferenceFrame,
    ...     WrappingCylinder)
    >>> N = ReferenceFrame('N')
    >>> r = symbols('r')
    >>> pO = Point('pO')
    >>> ax = N.x

    A cylinder with radius ``r``, and axis parallel to ``N.x`` passing through
    ``pO`` can be instantiated with:

    >>> WrappingCylinder(r, pO, ax)
    WrappingCylinder(radius=r, point=pO, axis=N.x)

    Parameters
    ==========

    radius : Symbol
        The radius of the cylinder.
    point : Point
        A point through which the cylinder's axis passes.
    axis : Vector
        The axis along which the cylinder is aligned.

    See Also
    ========

    WrappingSphere: Spherical geometry where the wrapping direction is always
        geodetic.

    """

    def __init__(self, radius, point, axis):
        """Initializer for ``WrappingCylinder``.

        Parameters
        ==========

        radius : Symbol
            The radius of the cylinder. This symbol must represent a value that
            is positive and constant, i.e. it cannot be a dynamic symbol.
        point : Point
            A point through which the cylinder's axis passes.
        axis : Vector
            The axis along which the cylinder is aligned.

        """
        self.radius = radius
        self.point = point
        self.axis = axis

    @property
    def radius(self):
        """Radius of the cylinder."""
        return self._radius

    @radius.setter
    def radius(self, radius):
        self._radius = radius

    @property
    def point(self):
        """A point through which the cylinder's axis passes."""
        return self._point

    @point.setter
    def point(self, point):
        self._point = point

    @property
    def axis(self):
        """Axis along which the cylinder is aligned."""
        return self._axis

    @axis.setter
    def axis(self, axis):
        self._axis = axis.normalize()

    def point_on_surface(self, point):
        """Returns ``True`` if a point is on the cylinder's surface.

        Parameters
        ==========

        point : Point
            The point for which it's to be ascertained if it's on the
            cylinder's surface or not. This point's position relative to the
            cylinder's axis must be a simple expression involving the radius of
            the sphere, otherwise this check will likely not work.

        """
        relative_position = point.pos_from(self.point)
        parallel = relative_position.dot(self.axis) * self.axis
        point_vector = relative_position - parallel
        if isinstance(point_vector, Vector):
            point_radius_squared = dot(point_vector, point_vector)
        else:
            point_radius_squared = point_vector**2
        return Eq(trigsimp(point_radius_squared), self.radius**2) == True

    def geodesic_length(self, point_1, point_2):
        """The shortest distance between two points on a geometry's surface.

        Explanation
        ===========

        The geodesic length, i.e. the shortest arc along the surface of a
        cylinder, connecting two points. It can be calculated using Pythagoras'
        theorem. The first short side is the distance between the two points on
        the cylinder's surface parallel to the cylinder's axis. The second
        short side is the arc of a circle between the two points of the
        cylinder's surface perpendicular to the cylinder's axis. The resulting
        hypotenuse is the geodesic length.

        Examples
        ========

        A geodesic length can only be calculated between two points on the
        cylinder's surface. Firstly, a ``WrappingCylinder`` instance must be
        created along with two points that will lie on its surface:

        >>> from sympy import symbols, cos, sin
        >>> from sympy.physics.mechanics import (Point, ReferenceFrame,
        ...     WrappingCylinder, dynamicsymbols)
        >>> N = ReferenceFrame('N')
        >>> r = symbols('r')
        >>> pO = Point('pO')
        >>> pO.set_vel(N, 0)
        >>> cylinder = WrappingCylinder(r, pO, N.x)
        >>> p1 = Point('p1')
        >>> p2 = Point('p2')

        Let's assume that ``p1`` is located at ``N.x + r*N.y`` relative to
        ``pO`` and that ``p2`` is located at ``r*(cos(q)*N.y + sin(q)*N.z)``
        relative to ``pO``, where ``q(t)`` is a generalized coordinate
        specifying the angle rotated around the ``N.x`` axis according to the
        right-hand rule where ``N.y`` is zero. These positions can be set with:

        >>> q = dynamicsymbols('q')
        >>> p1.set_pos(pO, N.x + r*N.y)
        >>> p1.pos_from(pO)
        N.x + r*N.y
        >>> p2.set_pos(pO, r*(cos(q)*N.y + sin(q)*N.z).normalize())
        >>> p2.pos_from(pO).simplify()
        r*cos(q(t))*N.y + r*sin(q(t))*N.z

        The geodesic length, which is in this case a is the hypotenuse of a
        right triangle where the other two side lengths are ``1`` (parallel to
        the cylinder's axis) and ``r*q(t)`` (parallel to the cylinder's cross
        section), can be calculated using the ``geodesic_length`` method:

        >>> cylinder.geodesic_length(p1, p2).simplify()
        sqrt(r**2*q(t)**2 + 1)

        If the ``geodesic_length`` method is passed an argument ``Point`` that
        doesn't lie on the sphere's surface then a ``ValueError`` is raised
        because it's not possible to calculate a value in this case.

        Parameters
        ==========

        point_1 : Point
            Point from which the geodesic length should be calculated.
        point_2 : Point
            Point to which the geodesic length should be calculated.

        """
        for point in (point_1, point_2):
            if not self.point_on_surface(point):
                msg = (
                    f'Geodesic length cannot be calculated as point {point} '
                    f'with radius {point.pos_from(self.point).magnitude()} '
                    f'from the cylinder\'s center {self.point} does not lie on '
                    f'the surface of {self} with radius {self.radius} and axis '
                    f'{self.axis}.'
                )
                raise ValueError(msg)

        relative_position = point_2.pos_from(point_1)
        parallel_length = relative_position.dot(self.axis)

        point_1_relative_position = point_1.pos_from(self.point)
        point_1_perpendicular_vector = (
            point_1_relative_position
            - point_1_relative_position.dot(self.axis)*self.axis
        ).normalize()

        point_2_relative_position = point_2.pos_from(self.point)
        point_2_perpendicular_vector = (
            point_2_relative_position
            - point_2_relative_position.dot(self.axis)*self.axis
        ).normalize()

        central_angle = _directional_atan(
            cancel(point_1_perpendicular_vector
                .cross(point_2_perpendicular_vector)
                .dot(self.axis)),
            cancel(point_1_perpendicular_vector.dot(point_2_perpendicular_vector)),
        )

        planar_arc_length = self.radius*central_angle
        geodesic_length = sqrt(parallel_length**2 + planar_arc_length**2)
        return geodesic_length

    def geodesic_end_vectors(self, point_1, point_2):
        """The vectors parallel to the geodesic at the two end points.

        Parameters
        ==========

        point_1 : Point
            The point from which the geodesic originates.
        point_2 : Point
            The point at which the geodesic terminates.

        """
        point_1_from_origin_point = point_1.pos_from(self.point)
        point_2_from_origin_point = point_2.pos_from(self.point)

        if point_1_from_origin_point == point_2_from_origin_point:
            msg = (
                f'Cannot compute geodesic end vectors for coincident points '
                f'{point_1} and {point_2} as no geodesic exists.'
            )
            raise ValueError(msg)

        point_1_parallel = point_1_from_origin_point.dot(self.axis) * self.axis
        point_2_parallel = point_2_from_origin_point.dot(self.axis) * self.axis
        point_1_normal = (point_1_from_origin_point - point_1_parallel)
        point_2_normal = (point_2_from_origin_point - point_2_parallel)

        if point_1_normal == point_2_normal:
            point_1_perpendicular = Vector(0)
            point_2_perpendicular = Vector(0)
        else:
            point_1_perpendicular = self.axis.cross(point_1_normal).normalize()
            point_2_perpendicular = -self.axis.cross(point_2_normal).normalize()

        geodesic_length = self.geodesic_length(point_1, point_2)
        relative_position = point_2.pos_from(point_1)
        parallel_length = relative_position.dot(self.axis)
        planar_arc_length = sqrt(geodesic_length**2 - parallel_length**2)

        point_1_vector = (
            planar_arc_length * point_1_perpendicular
            + parallel_length * self.axis
        ).normalize()
        point_2_vector = (
            planar_arc_length * point_2_perpendicular
            - parallel_length * self.axis
        ).normalize()

        return (point_1_vector, point_2_vector)

    def __repr__(self):
        """Representation of a ``WrappingCylinder``."""
        return (
            f'{self.__class__.__name__}(radius={self.radius}, '
            f'point={self.point}, axis={self.axis})'
        )


def _directional_atan(numerator, denominator):
    """Compute atan in a directional sense as required for geodesics.

    Explanation
    ===========

    To be able to control the direction of the geodesic length along the
    surface of a cylinder a dedicated arctangent function is needed that
    properly handles the directionality of different case. This function
    ensures that the central angle is always positive but shifting the case
    where ``atan2`` would return a negative angle to be centered around
    ``2*pi``.

    Notes
    =====

    This function only handles very specific cases, i.e. the ones that are
    expected to be encountered when calculating symbolic geodesics on uniformly
    curved surfaces. As such, ``NotImplemented`` errors can be raised in many
    cases. This function is named with a leader underscore to indicate that it
    only aims to provide very specific functionality within the private scope
    of this module.

    """

    if numerator.is_number and denominator.is_number:
        angle = atan2(numerator, denominator)
        if angle < 0:
            angle += 2 * pi
    elif numerator.is_number:
        msg = (
            f'Cannot compute a directional atan when the numerator {numerator} '
            f'is numeric and the denominator {denominator} is symbolic.'
        )
        raise NotImplementedError(msg)
    elif denominator.is_number:
        msg = (
            f'Cannot compute a directional atan when the numerator {numerator} '
            f'is symbolic and the denominator {denominator} is numeric.'
        )
        raise NotImplementedError(msg)
    else:
        ratio = sympify(trigsimp(numerator / denominator))
        if isinstance(ratio, tan):
            angle = ratio.args[0]
        elif (
            ratio.is_Mul
            and ratio.args[0] == Integer(-1)
            and isinstance(ratio.args[1], tan)
        ):
            angle = 2 * pi - ratio.args[1].args[0]
        else:
            msg = f'Cannot compute a directional atan for the value {ratio}.'
            raise NotImplementedError(msg)

    return angle
