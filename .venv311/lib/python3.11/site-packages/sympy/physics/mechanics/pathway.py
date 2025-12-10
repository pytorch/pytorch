"""Implementations of pathways for use by actuators."""

from abc import ABC, abstractmethod

from sympy.core.singleton import S
from sympy.physics.mechanics.loads import Force
from sympy.physics.mechanics.wrapping_geometry import WrappingGeometryBase
from sympy.physics.vector import Point, dynamicsymbols


__all__ = ['PathwayBase', 'LinearPathway', 'ObstacleSetPathway',
           'WrappingPathway']


class PathwayBase(ABC):
    """Abstract base class for all pathway classes to inherit from.

    Notes
    =====

    Instances of this class cannot be directly instantiated by users. However,
    it can be used to created custom pathway types through subclassing.

    """

    def __init__(self, *attachments):
        """Initializer for ``PathwayBase``."""
        self.attachments = attachments

    @property
    def attachments(self):
        """The pair of points defining a pathway's ends."""
        return self._attachments

    @attachments.setter
    def attachments(self, attachments):
        if hasattr(self, '_attachments'):
            msg = (
                f'Can\'t set attribute `attachments` to {repr(attachments)} '
                f'as it is immutable.'
            )
            raise AttributeError(msg)
        if len(attachments) != 2:
            msg = (
                f'Value {repr(attachments)} passed to `attachments` was an '
                f'iterable of length {len(attachments)}, must be an iterable '
                f'of length 2.'
            )
            raise ValueError(msg)
        for i, point in enumerate(attachments):
            if not isinstance(point, Point):
                msg = (
                    f'Value {repr(point)} passed to `attachments` at index '
                    f'{i} was of type {type(point)}, must be {Point}.'
                )
                raise TypeError(msg)
        self._attachments = tuple(attachments)

    @property
    @abstractmethod
    def length(self):
        """An expression representing the pathway's length."""
        pass

    @property
    @abstractmethod
    def extension_velocity(self):
        """An expression representing the pathway's extension velocity."""
        pass

    @abstractmethod
    def to_loads(self, force):
        """Loads required by the equations of motion method classes.

        Explanation
        ===========

        ``KanesMethod`` requires a list of ``Point``-``Vector`` tuples to be
        passed to the ``loads`` parameters of its ``kanes_equations`` method
        when constructing the equations of motion. This method acts as a
        utility to produce the correctly-structred pairs of points and vectors
        required so that these can be easily concatenated with other items in
        the list of loads and passed to ``KanesMethod.kanes_equations``. These
        loads are also in the correct form to also be passed to the other
        equations of motion method classes, e.g. ``LagrangesMethod``.

        """
        pass

    def __repr__(self):
        """Default representation of a pathway."""
        attachments = ', '.join(str(a) for a in self.attachments)
        return f'{self.__class__.__name__}({attachments})'


class LinearPathway(PathwayBase):
    """Linear pathway between a pair of attachment points.

    Explanation
    ===========

    A linear pathway forms a straight-line segment between two points and is
    the simplest pathway that can be formed. It will not interact with any
    other objects in the system, i.e. a ``LinearPathway`` will intersect other
    objects to ensure that the path between its two ends (its attachments) is
    the shortest possible.

    A linear pathway is made up of two points that can move relative to each
    other, and a pair of equal and opposite forces acting on the points. If the
    positive time-varying Euclidean distance between the two points is defined,
    then the "extension velocity" is the time derivative of this distance. The
    extension velocity is positive when the two points are moving away from
    each other and negative when moving closer to each other. The direction for
    the force acting on either point is determined by constructing a unit
    vector directed from the other point to this point. This establishes a sign
    convention such that a positive force magnitude tends to push the points
    apart. The following diagram shows the positive force sense and the
    distance between the points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    >>> from sympy.physics.mechanics import LinearPathway

    To construct a pathway, two points are required to be passed to the
    ``attachments`` parameter as a ``tuple``.

    >>> from sympy.physics.mechanics import Point
    >>> pA, pB = Point('pA'), Point('pB')
    >>> linear_pathway = LinearPathway(pA, pB)
    >>> linear_pathway
    LinearPathway(pA, pB)

    The pathway created above isn't very interesting without the positions and
    velocities of its attachment points being described. Without this its not
    possible to describe how the pathway moves, i.e. its length or its
    extension velocity.

    >>> from sympy.physics.mechanics import ReferenceFrame
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> pB.set_pos(pA, q*N.x)
    >>> pB.pos_from(pA)
    q(t)*N.x

    A pathway's length can be accessed via its ``length`` attribute.

    >>> linear_pathway.length
    sqrt(q(t)**2)

    Note how what appears to be an overly-complex expression is returned. This
    is actually required as it ensures that a pathway's length is always
    positive.

    A pathway's extension velocity can be accessed similarly via its
    ``extension_velocity`` attribute.

    >>> linear_pathway.extension_velocity
    sqrt(q(t)**2)*Derivative(q(t), t)/q(t)

    Parameters
    ==========

    attachments : tuple[Point, Point]
        Pair of ``Point`` objects between which the linear pathway spans.
        Constructor expects two points to be passed, e.g.
        ``LinearPathway(Point('pA'), Point('pB'))``. More or fewer points will
        cause an error to be thrown.

    """

    def __init__(self, *attachments):
        """Initializer for ``LinearPathway``.

        Parameters
        ==========

        attachments : Point
            Pair of ``Point`` objects between which the linear pathway spans.
            Constructor expects two points to be passed, e.g.
            ``LinearPathway(Point('pA'), Point('pB'))``. More or fewer points
            will cause an error to be thrown.

        """
        super().__init__(*attachments)

    @property
    def length(self):
        """Exact analytical expression for the pathway's length."""
        return _point_pair_length(*self.attachments)

    @property
    def extension_velocity(self):
        """Exact analytical expression for the pathway's extension velocity."""
        return _point_pair_extension_velocity(*self.attachments)

    def to_loads(self, force):
        """Loads required by the equations of motion method classes.

        Explanation
        ===========

        ``KanesMethod`` requires a list of ``Point``-``Vector`` tuples to be
        passed to the ``loads`` parameters of its ``kanes_equations`` method
        when constructing the equations of motion. This method acts as a
        utility to produce the correctly-structred pairs of points and vectors
        required so that these can be easily concatenated with other items in
        the list of loads and passed to ``KanesMethod.kanes_equations``. These
        loads are also in the correct form to also be passed to the other
        equations of motion method classes, e.g. ``LagrangesMethod``.

        Examples
        ========

        The below example shows how to generate the loads produced in a linear
        actuator that produces an expansile force ``F``. First, create a linear
        actuator between two points separated by the coordinate ``q`` in the
        ``x`` direction of the global frame ``N``.

        >>> from sympy.physics.mechanics import (LinearPathway, Point,
        ...     ReferenceFrame)
        >>> from sympy.physics.vector import dynamicsymbols
        >>> q = dynamicsymbols('q')
        >>> N = ReferenceFrame('N')
        >>> pA, pB = Point('pA'), Point('pB')
        >>> pB.set_pos(pA, q*N.x)
        >>> linear_pathway = LinearPathway(pA, pB)

        Now create a symbol ``F`` to describe the magnitude of the (expansile)
        force that will be produced along the pathway. The list of loads that
        ``KanesMethod`` requires can be produced by calling the pathway's
        ``to_loads`` method with ``F`` passed as the only argument.

        >>> from sympy import symbols
        >>> F = symbols('F')
        >>> linear_pathway.to_loads(F)
        [(pA, - F*q(t)/sqrt(q(t)**2)*N.x), (pB, F*q(t)/sqrt(q(t)**2)*N.x)]

        Parameters
        ==========

        force : Expr
            Magnitude of the force acting along the length of the pathway. As
            per the sign conventions for the pathway length, pathway extension
            velocity, and pair of point forces, if this ``Expr`` is positive
            then the force will act to push the pair of points away from one
            another (it is expansile).

        """
        relative_position = _point_pair_relative_position(*self.attachments)
        loads = [
            Force(self.attachments[0], -force*relative_position/self.length),
            Force(self.attachments[-1], force*relative_position/self.length),
        ]
        return loads


class ObstacleSetPathway(PathwayBase):
    """Obstacle-set pathway between a set of attachment points.

    Explanation
    ===========

    An obstacle-set pathway forms a series of straight-line segment between
    pairs of consecutive points in a set of points. It is similar to multiple
    linear pathways joined end-to-end. It will not interact with any other
    objects in the system, i.e. an ``ObstacleSetPathway`` will intersect other
    objects to ensure that the path between its pairs of points (its
    attachments) is the shortest possible.

    Examples
    ========

    To construct an obstacle-set pathway, three or more points are required to
    be passed to the ``attachments`` parameter as a ``tuple``.

    >>> from sympy.physics.mechanics import ObstacleSetPathway, Point
    >>> pA, pB, pC, pD = Point('pA'), Point('pB'), Point('pC'), Point('pD')
    >>> obstacle_set_pathway = ObstacleSetPathway(pA, pB, pC, pD)
    >>> obstacle_set_pathway
    ObstacleSetPathway(pA, pB, pC, pD)

    The pathway created above isn't very interesting without the positions and
    velocities of its attachment points being described. Without this its not
    possible to describe how the pathway moves, i.e. its length or its
    extension velocity.

    >>> from sympy import cos, sin
    >>> from sympy.physics.mechanics import ReferenceFrame
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> pO = Point('pO')
    >>> pA.set_pos(pO, N.y)
    >>> pB.set_pos(pO, -N.x)
    >>> pC.set_pos(pA, cos(q) * N.x - (sin(q) + 1) * N.y)
    >>> pD.set_pos(pA, sin(q) * N.x + (cos(q) - 1) * N.y)
    >>> pB.pos_from(pA)
    - N.x - N.y
    >>> pC.pos_from(pA)
    cos(q(t))*N.x + (-sin(q(t)) - 1)*N.y
    >>> pD.pos_from(pA)
    sin(q(t))*N.x + (cos(q(t)) - 1)*N.y

    A pathway's length can be accessed via its ``length`` attribute.

    >>> obstacle_set_pathway.length.simplify()
    sqrt(2)*(sqrt(cos(q(t)) + 1) + 2)

    A pathway's extension velocity can be accessed similarly via its
    ``extension_velocity`` attribute.

    >>> obstacle_set_pathway.extension_velocity.simplify()
    -sqrt(2)*sin(q(t))*Derivative(q(t), t)/(2*sqrt(cos(q(t)) + 1))

    Parameters
    ==========

    attachments : tuple[Point, ...]
        The set of ``Point`` objects that define the segmented obstacle-set
        pathway.

    """

    def __init__(self, *attachments):
        """Initializer for ``ObstacleSetPathway``.

        Parameters
        ==========

        attachments : tuple[Point, ...]
            The set of ``Point`` objects that define the segmented obstacle-set
            pathway.

        """
        super().__init__(*attachments)

    @property
    def attachments(self):
        """The set of points defining a pathway's segmented path."""
        return self._attachments

    @attachments.setter
    def attachments(self, attachments):
        if hasattr(self, '_attachments'):
            msg = (
                f'Can\'t set attribute `attachments` to {repr(attachments)} '
                f'as it is immutable.'
            )
            raise AttributeError(msg)
        if len(attachments) <= 2:
            msg = (
                f'Value {repr(attachments)} passed to `attachments` was an '
                f'iterable of length {len(attachments)}, must be an iterable '
                f'of length 3 or greater.'
            )
            raise ValueError(msg)
        for i, point in enumerate(attachments):
            if not isinstance(point, Point):
                msg = (
                    f'Value {repr(point)} passed to `attachments` at index '
                    f'{i} was of type {type(point)}, must be {Point}.'
                )
                raise TypeError(msg)
        self._attachments = tuple(attachments)

    @property
    def length(self):
        """Exact analytical expression for the pathway's length."""
        length = S.Zero
        attachment_pairs = zip(self.attachments[:-1], self.attachments[1:])
        for attachment_pair in attachment_pairs:
            length += _point_pair_length(*attachment_pair)
        return length

    @property
    def extension_velocity(self):
        """Exact analytical expression for the pathway's extension velocity."""
        extension_velocity = S.Zero
        attachment_pairs = zip(self.attachments[:-1], self.attachments[1:])
        for attachment_pair in attachment_pairs:
            extension_velocity += _point_pair_extension_velocity(*attachment_pair)
        return extension_velocity

    def to_loads(self, force):
        """Loads required by the equations of motion method classes.

        Explanation
        ===========

        ``KanesMethod`` requires a list of ``Point``-``Vector`` tuples to be
        passed to the ``loads`` parameters of its ``kanes_equations`` method
        when constructing the equations of motion. This method acts as a
        utility to produce the correctly-structred pairs of points and vectors
        required so that these can be easily concatenated with other items in
        the list of loads and passed to ``KanesMethod.kanes_equations``. These
        loads are also in the correct form to also be passed to the other
        equations of motion method classes, e.g. ``LagrangesMethod``.

        Examples
        ========

        The below example shows how to generate the loads produced in an
        actuator that follows an obstacle-set pathway between four points and
        produces an expansile force ``F``. First, create a pair of reference
        frames, ``A`` and ``B``, in which the four points ``pA``, ``pB``,
        ``pC``, and ``pD`` will be located. The first two points in frame ``A``
        and the second two in frame ``B``. Frame ``B`` will also be oriented
        such that it relates to ``A`` via a rotation of ``q`` about an axis
        ``N.z`` in a global frame (``N.z``, ``A.z``, and ``B.z`` are parallel).

        >>> from sympy.physics.mechanics import (ObstacleSetPathway, Point,
        ...     ReferenceFrame)
        >>> from sympy.physics.vector import dynamicsymbols
        >>> q = dynamicsymbols('q')
        >>> N = ReferenceFrame('N')
        >>> N = ReferenceFrame('N')
        >>> A = N.orientnew('A', 'axis', (0, N.x))
        >>> B = A.orientnew('B', 'axis', (q, N.z))
        >>> pO = Point('pO')
        >>> pA, pB, pC, pD = Point('pA'), Point('pB'), Point('pC'), Point('pD')
        >>> pA.set_pos(pO, A.x)
        >>> pB.set_pos(pO, -A.y)
        >>> pC.set_pos(pO, B.y)
        >>> pD.set_pos(pO, B.x)
        >>> obstacle_set_pathway = ObstacleSetPathway(pA, pB, pC, pD)

        Now create a symbol ``F`` to describe the magnitude of the (expansile)
        force that will be produced along the pathway. The list of loads that
        ``KanesMethod`` requires can be produced by calling the pathway's
        ``to_loads`` method with ``F`` passed as the only argument.

        >>> from sympy import Symbol
        >>> F = Symbol('F')
        >>> obstacle_set_pathway.to_loads(F)
        [(pA, sqrt(2)*F/2*A.x + sqrt(2)*F/2*A.y),
         (pB, - sqrt(2)*F/2*A.x - sqrt(2)*F/2*A.y),
         (pB, - F/sqrt(2*cos(q(t)) + 2)*A.y - F/sqrt(2*cos(q(t)) + 2)*B.y),
         (pC, F/sqrt(2*cos(q(t)) + 2)*A.y + F/sqrt(2*cos(q(t)) + 2)*B.y),
         (pC, - sqrt(2)*F/2*B.x + sqrt(2)*F/2*B.y),
         (pD, sqrt(2)*F/2*B.x - sqrt(2)*F/2*B.y)]

        Parameters
        ==========

        force : Expr
            The force acting along the length of the pathway. It is assumed
            that this ``Expr`` represents an expansile force.

        """
        loads = []
        attachment_pairs = zip(self.attachments[:-1], self.attachments[1:])
        for attachment_pair in attachment_pairs:
            relative_position = _point_pair_relative_position(*attachment_pair)
            length = _point_pair_length(*attachment_pair)
            loads.extend([
                Force(attachment_pair[0], -force*relative_position/length),
                Force(attachment_pair[1], force*relative_position/length),
            ])
        return loads


class WrappingPathway(PathwayBase):
    """Pathway that wraps a geometry object.

    Explanation
    ===========

    A wrapping pathway interacts with a geometry object and forms a path that
    wraps smoothly along its surface. The wrapping pathway along the geometry
    object will be the geodesic that the geometry object defines based on the
    two points. It will not interact with any other objects in the system, i.e.
    a ``WrappingPathway`` will intersect other objects to ensure that the path
    between its two ends (its attachments) is the shortest possible.

    To explain the sign conventions used for pathway length, extension
    velocity, and direction of applied forces, we can ignore the geometry with
    which the wrapping pathway interacts. A wrapping pathway is made up of two
    points that can move relative to each other, and a pair of equal and
    opposite forces acting on the points. If the positive time-varying
    Euclidean distance between the two points is defined, then the "extension
    velocity" is the time derivative of this distance. The extension velocity
    is positive when the two points are moving away from each other and
    negative when moving closer to each other. The direction for the force
    acting on either point is determined by constructing a unit vector directed
    from the other point to this point. This establishes a sign convention such
    that a positive force magnitude tends to push the points apart. The
    following diagram shows the positive force sense and the distance between
    the points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    >>> from sympy.physics.mechanics import WrappingPathway

    To construct a wrapping pathway, like other pathways, a pair of points must
    be passed, followed by an instance of a wrapping geometry class as a
    keyword argument. We'll use a cylinder with radius ``r`` and its axis
    parallel to ``N.x`` passing through a point ``pO``.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import Point, ReferenceFrame, WrappingCylinder
    >>> r = symbols('r')
    >>> N = ReferenceFrame('N')
    >>> pA, pB, pO = Point('pA'), Point('pB'), Point('pO')
    >>> cylinder = WrappingCylinder(r, pO, N.x)
    >>> wrapping_pathway = WrappingPathway(pA, pB, cylinder)
    >>> wrapping_pathway
    WrappingPathway(pA, pB, geometry=WrappingCylinder(radius=r, point=pO,
        axis=N.x))

    Parameters
    ==========

    attachment_1 : Point
        First of the pair of ``Point`` objects between which the wrapping
        pathway spans.
    attachment_2 : Point
        Second of the pair of ``Point`` objects between which the wrapping
        pathway spans.
    geometry : WrappingGeometryBase
        Geometry about which the pathway wraps.

    """

    def __init__(self, attachment_1, attachment_2, geometry):
        """Initializer for ``WrappingPathway``.

        Parameters
        ==========

        attachment_1 : Point
            First of the pair of ``Point`` objects between which the wrapping
            pathway spans.
        attachment_2 : Point
            Second of the pair of ``Point`` objects between which the wrapping
            pathway spans.
        geometry : WrappingGeometryBase
            Geometry about which the pathway wraps.
            The geometry about which the pathway wraps.

        """
        super().__init__(attachment_1, attachment_2)
        self.geometry = geometry

    @property
    def geometry(self):
        """Geometry around which the pathway wraps."""
        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        if hasattr(self, '_geometry'):
            msg = (
                f'Can\'t set attribute `geometry` to {repr(geometry)} as it '
                f'is immutable.'
            )
            raise AttributeError(msg)
        if not isinstance(geometry, WrappingGeometryBase):
            msg = (
                f'Value {repr(geometry)} passed to `geometry` was of type '
                f'{type(geometry)}, must be {WrappingGeometryBase}.'
            )
            raise TypeError(msg)
        self._geometry = geometry

    @property
    def length(self):
        """Exact analytical expression for the pathway's length."""
        return self.geometry.geodesic_length(*self.attachments)

    @property
    def extension_velocity(self):
        """Exact analytical expression for the pathway's extension velocity."""
        return self.length.diff(dynamicsymbols._t)

    def to_loads(self, force):
        """Loads required by the equations of motion method classes.

        Explanation
        ===========

        ``KanesMethod`` requires a list of ``Point``-``Vector`` tuples to be
        passed to the ``loads`` parameters of its ``kanes_equations`` method
        when constructing the equations of motion. This method acts as a
        utility to produce the correctly-structred pairs of points and vectors
        required so that these can be easily concatenated with other items in
        the list of loads and passed to ``KanesMethod.kanes_equations``. These
        loads are also in the correct form to also be passed to the other
        equations of motion method classes, e.g. ``LagrangesMethod``.

        Examples
        ========

        The below example shows how to generate the loads produced in an
        actuator that produces an expansile force ``F`` while wrapping around a
        cylinder. First, create a cylinder with radius ``r`` and an axis
        parallel to the ``N.z`` direction of the global frame ``N`` that also
        passes through a point ``pO``.

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import (Point, ReferenceFrame,
        ...     WrappingCylinder)
        >>> N = ReferenceFrame('N')
        >>> r = symbols('r', positive=True)
        >>> pO = Point('pO')
        >>> cylinder = WrappingCylinder(r, pO, N.z)

        Create the pathway of the actuator using the ``WrappingPathway`` class,
        defined to span between two points ``pA`` and ``pB``. Both points lie
        on the surface of the cylinder and the location of ``pB`` is defined
        relative to ``pA`` by the dynamics symbol ``q``.

        >>> from sympy import cos, sin
        >>> from sympy.physics.mechanics import WrappingPathway, dynamicsymbols
        >>> q = dynamicsymbols('q')
        >>> pA = Point('pA')
        >>> pB = Point('pB')
        >>> pA.set_pos(pO, r*N.x)
        >>> pB.set_pos(pO, r*(cos(q)*N.x + sin(q)*N.y))
        >>> pB.pos_from(pA)
        (r*cos(q(t)) - r)*N.x + r*sin(q(t))*N.y
        >>> pathway = WrappingPathway(pA, pB, cylinder)

        Now create a symbol ``F`` to describe the magnitude of the (expansile)
        force that will be produced along the pathway. The list of loads that
        ``KanesMethod`` requires can be produced by calling the pathway's
        ``to_loads`` method with ``F`` passed as the only argument.

        >>> F = symbols('F')
        >>> loads = pathway.to_loads(F)
        >>> [load.__class__(load.location, load.vector.simplify()) for load in loads]
        [(pA, F*N.y), (pB, F*sin(q(t))*N.x - F*cos(q(t))*N.y),
         (pO, - F*sin(q(t))*N.x + F*(cos(q(t)) - 1)*N.y)]

        Parameters
        ==========

        force : Expr
            Magnitude of the force acting along the length of the pathway. It
            is assumed that this ``Expr`` represents an expansile force.

        """
        pA, pB = self.attachments
        pO = self.geometry.point
        pA_force, pB_force = self.geometry.geodesic_end_vectors(pA, pB)
        pO_force = -(pA_force + pB_force)

        loads = [
            Force(pA, force * pA_force),
            Force(pB, force * pB_force),
            Force(pO, force * pO_force),
        ]
        return loads

    def __repr__(self):
        """Representation of a ``WrappingPathway``."""
        attachments = ', '.join(str(a) for a in self.attachments)
        return (
            f'{self.__class__.__name__}({attachments}, '
            f'geometry={self.geometry})'
        )


def _point_pair_relative_position(point_1, point_2):
    """The relative position between a pair of points."""
    return point_2.pos_from(point_1)


def _point_pair_length(point_1, point_2):
    """The length of the direct linear path between two points."""
    return _point_pair_relative_position(point_1, point_2).magnitude()


def _point_pair_extension_velocity(point_1, point_2):
    """The extension velocity of the direct linear path between two points."""
    return _point_pair_length(point_1, point_2).diff(dynamicsymbols._t)
