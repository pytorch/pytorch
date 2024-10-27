from sympy import Symbol, S
from sympy.physics.vector import ReferenceFrame, Dyadic, Point, dot
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.mechanics.inertia import inertia_of_point_mass, Inertia
from sympy.utilities.exceptions import sympy_deprecation_warning

__all__ = ['RigidBody']


class RigidBody(BodyBase):
    """An idealized rigid body.

    Explanation
    ===========

    This is essentially a container which holds the various components which
    describe a rigid body: a name, mass, center of mass, reference frame, and
    inertia.

    All of these need to be supplied on creation, but can be changed
    afterwards.

    Attributes
    ==========

    name : string
        The body's name.
    masscenter : Point
        The point which represents the center of mass of the rigid body.
    frame : ReferenceFrame
        The ReferenceFrame which the rigid body is fixed in.
    mass : Sympifyable
        The body's mass.
    inertia : (Dyadic, Point)
        The body's inertia about a point; stored in a tuple as shown above.
    potential_energy : Sympifyable
        The potential energy of the RigidBody.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.physics.mechanics import ReferenceFrame, Point, RigidBody
    >>> from sympy.physics.mechanics import outer
    >>> m = Symbol('m')
    >>> A = ReferenceFrame('A')
    >>> P = Point('P')
    >>> I = outer (A.x, A.x)
    >>> inertia_tuple = (I, P)
    >>> B = RigidBody('B', P, A, m, inertia_tuple)
    >>> # Or you could change them afterwards
    >>> m2 = Symbol('m2')
    >>> B.mass = m2

    """

    def __init__(self, name, masscenter=None, frame=None, mass=None,
                 inertia=None):
        super().__init__(name, masscenter, mass)
        if frame is None:
            frame = ReferenceFrame(f'{name}_frame')
        self.frame = frame
        if inertia is None:
            ixx = Symbol(f'{name}_ixx')
            iyy = Symbol(f'{name}_iyy')
            izz = Symbol(f'{name}_izz')
            izx = Symbol(f'{name}_izx')
            ixy = Symbol(f'{name}_ixy')
            iyz = Symbol(f'{name}_iyz')
            inertia = Inertia.from_inertia_scalars(self.masscenter, self.frame,
                                                   ixx, iyy, izz, ixy, iyz, izx)
        self.inertia = inertia

    def __repr__(self):
        return (f'{self.__class__.__name__}({repr(self.name)}, masscenter='
                f'{repr(self.masscenter)}, frame={repr(self.frame)}, mass='
                f'{repr(self.mass)}, inertia={repr(self.inertia)})')

    @property
    def frame(self):
        """The ReferenceFrame fixed to the body."""
        return self._frame

    @frame.setter
    def frame(self, F):
        if not isinstance(F, ReferenceFrame):
            raise TypeError("RigidBody frame must be a ReferenceFrame object.")
        self._frame = F

    @property
    def x(self):
        """The basis Vector for the body, in the x direction. """
        return self.frame.x

    @property
    def y(self):
        """The basis Vector for the body, in the y direction. """
        return self.frame.y

    @property
    def z(self):
        """The basis Vector for the body, in the z direction. """
        return self.frame.z

    @property
    def inertia(self):
        """The body's inertia about a point; stored as (Dyadic, Point)."""
        return self._inertia

    @inertia.setter
    def inertia(self, I):
        # check if I is of the form (Dyadic, Point)
        if len(I) != 2 or not isinstance(I[0], Dyadic) or not isinstance(I[1], Point):
            raise TypeError("RigidBody inertia must be a tuple of the form (Dyadic, Point).")

        self._inertia = Inertia(I[0], I[1])
        # have I S/O, want I S/S*
        # I S/O = I S/S* + I S*/O; I S/S* = I S/O - I S*/O
        # I_S/S* = I_S/O - I_S*/O
        I_Ss_O = inertia_of_point_mass(self.mass,
                                       self.masscenter.pos_from(I[1]),
                                       self.frame)
        self._central_inertia = I[0] - I_Ss_O

    @property
    def central_inertia(self):
        """The body's central inertia dyadic."""
        return self._central_inertia

    @central_inertia.setter
    def central_inertia(self, I):
        if not isinstance(I, Dyadic):
            raise TypeError("RigidBody inertia must be a Dyadic object.")
        self.inertia = Inertia(I, self.masscenter)

    def linear_momentum(self, frame):
        """ Linear momentum of the rigid body.

        Explanation
        ===========

        The linear momentum L, of a rigid body B, with respect to frame N is
        given by:

        ``L = m * v``

        where m is the mass of the rigid body, and v is the velocity of the mass
        center of B in the frame N.

        Parameters
        ==========

        frame : ReferenceFrame
            The frame in which linear momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer
        >>> from sympy.physics.mechanics import RigidBody, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> m, v = dynamicsymbols('m v')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> P.set_vel(N, v * N.x)
        >>> I = outer (N.x, N.x)
        >>> Inertia_tuple = (I, P)
        >>> B = RigidBody('B', P, N, m, Inertia_tuple)
        >>> B.linear_momentum(N)
        m*v*N.x

        """

        return self.mass * self.masscenter.vel(frame)

    def angular_momentum(self, point, frame):
        """Returns the angular momentum of the rigid body about a point in the
        given frame.

        Explanation
        ===========

        The angular momentum H of a rigid body B about some point O in a frame N
        is given by:

        ``H = dot(I, w) + cross(r, m * v)``

        where I and m are the central inertia dyadic and mass of rigid body B, w
        is the angular velocity of body B in the frame N, r is the position
        vector from point O to the mass center of B, and v is the velocity of
        the mass center in the frame N.

        Parameters
        ==========

        point : Point
            The point about which angular momentum is desired.
        frame : ReferenceFrame
            The frame in which angular momentum is desired.

        Examples
        ========

        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer
        >>> from sympy.physics.mechanics import RigidBody, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> m, v, r, omega = dynamicsymbols('m v r omega')
        >>> N = ReferenceFrame('N')
        >>> b = ReferenceFrame('b')
        >>> b.set_ang_vel(N, omega * b.x)
        >>> P = Point('P')
        >>> P.set_vel(N, 1 * N.x)
        >>> I = outer(b.x, b.x)
        >>> B = RigidBody('B', P, b, m, (I, P))
        >>> B.angular_momentum(P, N)
        omega*b.x

        """
        I = self.central_inertia
        w = self.frame.ang_vel_in(frame)
        m = self.mass
        r = self.masscenter.pos_from(point)
        v = self.masscenter.vel(frame)

        return I.dot(w) + r.cross(m * v)

    def kinetic_energy(self, frame):
        """Kinetic energy of the rigid body.

        Explanation
        ===========

        The kinetic energy, T, of a rigid body, B, is given by:

        ``T = 1/2 * (dot(dot(I, w), w) + dot(m * v, v))``

        where I and m are the central inertia dyadic and mass of rigid body B
        respectively, w is the body's angular velocity, and v is the velocity of
        the body's mass center in the supplied ReferenceFrame.

        Parameters
        ==========

        frame : ReferenceFrame
            The RigidBody's angular velocity and the velocity of it's mass
            center are typically defined with respect to an inertial frame but
            any relevant frame in which the velocities are known can be
            supplied.

        Examples
        ========

        >>> from sympy.physics.mechanics import Point, ReferenceFrame, outer
        >>> from sympy.physics.mechanics import RigidBody
        >>> from sympy import symbols
        >>> m, v, r, omega = symbols('m v r omega')
        >>> N = ReferenceFrame('N')
        >>> b = ReferenceFrame('b')
        >>> b.set_ang_vel(N, omega * b.x)
        >>> P = Point('P')
        >>> P.set_vel(N, v * N.x)
        >>> I = outer (b.x, b.x)
        >>> inertia_tuple = (I, P)
        >>> B = RigidBody('B', P, b, m, inertia_tuple)
        >>> B.kinetic_energy(N)
        m*v**2/2 + omega**2/2

        """

        rotational_KE = S.Half * dot(
            self.frame.ang_vel_in(frame),
            dot(self.central_inertia, self.frame.ang_vel_in(frame)))
        translational_KE = S.Half * self.mass * dot(self.masscenter.vel(frame),
                                                    self.masscenter.vel(frame))
        return rotational_KE + translational_KE

    def set_potential_energy(self, scalar):
        sympy_deprecation_warning(
            """
The sympy.physics.mechanics.RigidBody.set_potential_energy()
method is deprecated. Instead use

    B.potential_energy = scalar
            """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-set-potential-energy",
        )
        self.potential_energy = scalar

    def parallel_axis(self, point, frame=None):
        """Returns the inertia dyadic of the body with respect to another point.

        Parameters
        ==========

        point : sympy.physics.vector.Point
            The point to express the inertia dyadic about.
        frame : sympy.physics.vector.ReferenceFrame
            The reference frame used to construct the dyadic.

        Returns
        =======

        inertia : sympy.physics.vector.Dyadic
            The inertia dyadic of the rigid body expressed about the provided
            point.

        """
        if frame is None:
            frame = self.frame
        return self.central_inertia + inertia_of_point_mass(
            self.mass, self.masscenter.pos_from(point), frame)
