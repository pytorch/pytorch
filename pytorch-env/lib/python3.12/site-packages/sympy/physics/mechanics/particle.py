from sympy import S
from sympy.physics.vector import cross, dot
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.mechanics.inertia import inertia_of_point_mass
from sympy.utilities.exceptions import sympy_deprecation_warning

__all__ = ['Particle']


class Particle(BodyBase):
    """A particle.

    Explanation
    ===========

    Particles have a non-zero mass and lack spatial extension; they take up no
    space.

    Values need to be supplied on initialization, but can be changed later.

    Parameters
    ==========

    name : str
        Name of particle
    point : Point
        A physics/mechanics Point which represents the position, velocity, and
        acceleration of this Particle
    mass : Sympifyable
        A SymPy expression representing the Particle's mass
    potential_energy : Sympifyable
        The potential energy of the Particle.

    Examples
    ========

    >>> from sympy.physics.mechanics import Particle, Point
    >>> from sympy import Symbol
    >>> po = Point('po')
    >>> m = Symbol('m')
    >>> pa = Particle('pa', po, m)
    >>> # Or you could change these later
    >>> pa.mass = m
    >>> pa.point = po

    """
    point = BodyBase.masscenter

    def __init__(self, name, point=None, mass=None):
        super().__init__(name, point, mass)

    def linear_momentum(self, frame):
        """Linear momentum of the particle.

        Explanation
        ===========

        The linear momentum L, of a particle P, with respect to frame N is
        given by:

        L = m * v

        where m is the mass of the particle, and v is the velocity of the
        particle in the frame N.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which linear momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Particle, Point, ReferenceFrame
        >>> from sympy.physics.mechanics import dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> m, v = dynamicsymbols('m v')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> A = Particle('A', P, m)
        >>> P.set_vel(N, v * N.x)
        >>> A.linear_momentum(N)
        m*v*N.x

        """

        return self.mass * self.point.vel(frame)

    def angular_momentum(self, point, frame):
        """Angular momentum of the particle about the point.

        Explanation
        ===========

        The angular momentum H, about some point O of a particle, P, is given
        by:

        ``H = cross(r, m * v)``

        where r is the position vector from point O to the particle P, m is
        the mass of the particle, and v is the velocity of the particle in
        the inertial frame, N.

        Parameters
        ==========

        point : Point
            The point about which angular momentum of the particle is desired.

        frame : ReferenceFrame
            The frame in which angular momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Particle, Point, ReferenceFrame
        >>> from sympy.physics.mechanics import dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> m, v, r = dynamicsymbols('m v r')
        >>> N = ReferenceFrame('N')
        >>> O = Point('O')
        >>> A = O.locatenew('A', r * N.x)
        >>> P = Particle('P', A, m)
        >>> P.point.set_vel(N, v * N.y)
        >>> P.angular_momentum(O, N)
        m*r*v*N.z

        """

        return cross(self.point.pos_from(point),
                     self.mass * self.point.vel(frame))

    def kinetic_energy(self, frame):
        """Kinetic energy of the particle.

        Explanation
        ===========

        The kinetic energy, T, of a particle, P, is given by:

        ``T = 1/2 (dot(m * v, v))``

        where m is the mass of particle P, and v is the velocity of the
        particle in the supplied ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The Particle's velocity is typically defined with respect to
            an inertial frame but any relevant frame in which the velocity is
            known can be supplied.

        Examples
        ========

        >>> from sympy.physics.mechanics import Particle, Point, ReferenceFrame
        >>> from sympy import symbols
        >>> m, v, r = symbols('m v r')
        >>> N = ReferenceFrame('N')
        >>> O = Point('O')
        >>> P = Particle('P', O, m)
        >>> P.point.set_vel(N, v * N.y)
        >>> P.kinetic_energy(N)
        m*v**2/2

        """

        return S.Half * self.mass * dot(self.point.vel(frame),
                                        self.point.vel(frame))

    def set_potential_energy(self, scalar):
        sympy_deprecation_warning(
            """
The sympy.physics.mechanics.Particle.set_potential_energy()
method is deprecated. Instead use

    P.potential_energy = scalar
            """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-set-potential-energy",
        )
        self.potential_energy = scalar

    def parallel_axis(self, point, frame):
        """Returns an inertia dyadic of the particle with respect to another
        point and frame.

        Parameters
        ==========

        point : sympy.physics.vector.Point
            The point to express the inertia dyadic about.
        frame : sympy.physics.vector.ReferenceFrame
            The reference frame used to construct the dyadic.

        Returns
        =======

        inertia : sympy.physics.vector.Dyadic
            The inertia dyadic of the particle expressed about the provided
            point and frame.

        """
        return inertia_of_point_mass(self.mass, self.point.pos_from(point),
                                     frame)
