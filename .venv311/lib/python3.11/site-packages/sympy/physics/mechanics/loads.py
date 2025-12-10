from abc import ABC
from collections import namedtuple
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.vector import Vector, ReferenceFrame, Point

__all__ = ['LoadBase', 'Force', 'Torque']


class LoadBase(ABC, namedtuple('LoadBase', ['location', 'vector'])):
    """Abstract base class for the various loading types."""

    def __add__(self, other):
        raise TypeError(f"unsupported operand type(s) for +: "
                        f"'{self.__class__.__name__}' and "
                        f"'{other.__class__.__name__}'")

    def __mul__(self, other):
        raise TypeError(f"unsupported operand type(s) for *: "
                        f"'{self.__class__.__name__}' and "
                        f"'{other.__class__.__name__}'")

    __radd__ = __add__
    __rmul__ = __mul__


class Force(LoadBase):
    """Force acting upon a point.

    Explanation
    ===========

    A force is a vector that is bound to a line of action. This class stores
    both a point, which lies on the line of action, and the vector. A tuple can
    also be used, with the location as the first entry and the vector as second
    entry.

    Examples
    ========

    A force of magnitude 2 along N.x acting on a point Po can be created as
    follows:

    >>> from sympy.physics.mechanics import Point, ReferenceFrame, Force
    >>> N = ReferenceFrame('N')
    >>> Po = Point('Po')
    >>> Force(Po, 2 * N.x)
    (Po, 2*N.x)

    If a body is supplied, then the center of mass of that body is used.

    >>> from sympy.physics.mechanics import Particle
    >>> P = Particle('P', point=Po)
    >>> Force(P, 2 * N.x)
    (Po, 2*N.x)

    """

    def __new__(cls, point, force):
        if isinstance(point, BodyBase):
            point = point.masscenter
        if not isinstance(point, Point):
            raise TypeError('Force location should be a Point.')
        if not isinstance(force, Vector):
            raise TypeError('Force vector should be a Vector.')
        return super().__new__(cls, point, force)

    def __repr__(self):
        return (f'{self.__class__.__name__}(point={self.point}, '
                f'force={self.force})')

    @property
    def point(self):
        return self.location

    @property
    def force(self):
        return self.vector


class Torque(LoadBase):
    """Torque acting upon a frame.

    Explanation
    ===========

    A torque is a free vector that is acting on a reference frame, which is
    associated with a rigid body. This class stores both the frame and the
    vector. A tuple can also be used, with the location as the first item and
    the vector as second item.

    Examples
    ========

    A torque of magnitude 2 about N.x acting on a frame N can be created as
    follows:

    >>> from sympy.physics.mechanics import ReferenceFrame, Torque
    >>> N = ReferenceFrame('N')
    >>> Torque(N, 2 * N.x)
    (N, 2*N.x)

    If a body is supplied, then the frame fixed to that body is used.

    >>> from sympy.physics.mechanics import RigidBody
    >>> rb = RigidBody('rb', frame=N)
    >>> Torque(rb, 2 * N.x)
    (N, 2*N.x)

    """

    def __new__(cls, frame, torque):
        if isinstance(frame, BodyBase):
            frame = frame.frame
        if not isinstance(frame, ReferenceFrame):
            raise TypeError('Torque location should be a ReferenceFrame.')
        if not isinstance(torque, Vector):
            raise TypeError('Torque vector should be a Vector.')
        return super().__new__(cls, frame, torque)

    def __repr__(self):
        return (f'{self.__class__.__name__}(frame={self.frame}, '
                f'torque={self.torque})')

    @property
    def frame(self):
        return self.location

    @property
    def torque(self):
        return self.vector


def gravity(acceleration, *bodies):
    """
    Returns a list of gravity forces given the acceleration
    due to gravity and any number of particles or rigidbodies.

    Example
    =======

    >>> from sympy.physics.mechanics import ReferenceFrame, Particle, RigidBody
    >>> from sympy.physics.mechanics.loads import gravity
    >>> from sympy import symbols
    >>> N = ReferenceFrame('N')
    >>> g = symbols('g')
    >>> P = Particle('P')
    >>> B = RigidBody('B')
    >>> gravity(g*N.y, P, B)
    [(P_masscenter, P_mass*g*N.y),
     (B_masscenter, B_mass*g*N.y)]

    """

    gravity_force = []
    for body in bodies:
        if not isinstance(body, BodyBase):
            raise TypeError(f'{type(body)} is not a body type')
        gravity_force.append(Force(body.masscenter, body.mass * acceleration))
    return gravity_force


def _parse_load(load):
    """Helper function to parse loads and convert tuples to load objects."""
    if isinstance(load, LoadBase):
        return load
    elif isinstance(load, tuple):
        if len(load) != 2:
            raise ValueError(f'Load {load} should have a length of 2.')
        if isinstance(load[0], Point):
            return Force(load[0], load[1])
        elif isinstance(load[0], ReferenceFrame):
            return Torque(load[0], load[1])
        else:
            raise ValueError(f'Load not recognized. The load location {load[0]}'
                             f' should either be a Point or a ReferenceFrame.')
    raise TypeError(f'Load type {type(load)} not recognized as a load. It '
                    f'should be a Force, Torque or tuple.')
