"""Implementations of actuators for linked force and torque application."""

from abc import ABC, abstractmethod

from sympy import S, sympify, exp, sign
from sympy.physics.mechanics.joint import PinJoint
from sympy.physics.mechanics.loads import Torque
from sympy.physics.mechanics.pathway import PathwayBase
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.physics.vector import ReferenceFrame, Vector


__all__ = [
    'ActuatorBase',
    'ForceActuator',
    'LinearDamper',
    'LinearSpring',
    'TorqueActuator',
    'DuffingSpring',
    'CoulombKineticFriction',
]


class ActuatorBase(ABC):
    """Abstract base class for all actuator classes to inherit from.

    Notes
    =====

    Instances of this class cannot be directly instantiated by users. However,
    it can be used to created custom actuator types through subclassing.

    """

    def __init__(self):
        """Initializer for ``ActuatorBase``."""
        pass

    @abstractmethod
    def to_loads(self):
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
        """Default representation of an actuator."""
        return f'{self.__class__.__name__}()'


class ForceActuator(ActuatorBase):
    """Force-producing actuator.

    Explanation
    ===========

    A ``ForceActuator`` is an actuator that produces a (expansile) force along
    its length.

    A force actuator uses a pathway instance to determine the direction and
    number of forces that it applies to a system. Consider the simplest case
    where a ``LinearPathway`` instance is used. This pathway is made up of two
    points that can move relative to each other, and results in a pair of equal
    and opposite forces acting on the endpoints. If the positive time-varying
    Euclidean distance between the two points is defined, then the "extension
    velocity" is the time derivative of this distance. The extension velocity
    is positive when the two points are moving away from each other and
    negative when moving closer to each other. The direction for the force
    acting on either point is determined by constructing a unit vector directed
    from the other point to this point. This establishes a sign convention such
    that a positive force magnitude tends to push the points apart, this is the
    meaning of "expansile" in this context. The following diagram shows the
    positive force sense and the distance between the points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    To construct an actuator, an expression (or symbol) must be supplied to
    represent the force it can produce, alongside a pathway specifying its line
    of action. Let's also create a global reference frame and spatially fix one
    of the points in it while setting the other to be positioned such that it
    can freely move in the frame's x direction specified by the coordinate
    ``q``.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (ForceActuator, LinearPathway,
    ...     Point, ReferenceFrame)
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> force = symbols('F')
    >>> pA, pB = Point('pA'), Point('pB')
    >>> pA.set_vel(N, 0)
    >>> pB.set_pos(pA, q*N.x)
    >>> pB.pos_from(pA)
    q(t)*N.x
    >>> linear_pathway = LinearPathway(pA, pB)
    >>> actuator = ForceActuator(force, linear_pathway)
    >>> actuator
    ForceActuator(F, LinearPathway(pA, pB))

    Parameters
    ==========

    force : Expr
        The scalar expression defining the (expansile) force that the actuator
        produces.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.

    """

    def __init__(self, force, pathway):
        """Initializer for ``ForceActuator``.

        Parameters
        ==========

        force : Expr
            The scalar expression defining the (expansile) force that the
            actuator produces.
        pathway : PathwayBase
            The pathway that the actuator follows. This must be an instance of
            a concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.

        """
        self.force = force
        self.pathway = pathway

    @property
    def force(self):
        """The magnitude of the force produced by the actuator."""
        return self._force

    @force.setter
    def force(self, force):
        if hasattr(self, '_force'):
            msg = (
                f'Can\'t set attribute `force` to {repr(force)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        self._force = sympify(force, strict=True)

    @property
    def pathway(self):
        """The ``Pathway`` defining the actuator's line of action."""
        return self._pathway

    @pathway.setter
    def pathway(self, pathway):
        if hasattr(self, '_pathway'):
            msg = (
                f'Can\'t set attribute `pathway` to {repr(pathway)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        if not isinstance(pathway, PathwayBase):
            msg = (
                f'Value {repr(pathway)} passed to `pathway` was of type '
                f'{type(pathway)}, must be {PathwayBase}.'
            )
            raise TypeError(msg)
        self._pathway = pathway

    def to_loads(self):
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

        The below example shows how to generate the loads produced by a force
        actuator that follows a linear pathway. In this example we'll assume
        that the force actuator is being used to model a simple linear spring.
        First, create a linear pathway between two points separated by the
        coordinate ``q`` in the ``x`` direction of the global frame ``N``.

        >>> from sympy.physics.mechanics import (LinearPathway, Point,
        ...     ReferenceFrame)
        >>> from sympy.physics.vector import dynamicsymbols
        >>> q = dynamicsymbols('q')
        >>> N = ReferenceFrame('N')
        >>> pA, pB = Point('pA'), Point('pB')
        >>> pB.set_pos(pA, q*N.x)
        >>> pathway = LinearPathway(pA, pB)

        Now create a symbol ``k`` to describe the spring's stiffness and
        instantiate a force actuator that produces a (contractile) force
        proportional to both the spring's stiffness and the pathway's length.
        Note that actuator classes use the sign convention that expansile
        forces are positive, so for a spring to produce a contractile force the
        spring force needs to be calculated as the negative for the stiffness
        multiplied by the length.

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import ForceActuator
        >>> stiffness = symbols('k')
        >>> spring_force = -stiffness*pathway.length
        >>> spring = ForceActuator(spring_force, pathway)

        The forces produced by the spring can be generated in the list of loads
        form that ``KanesMethod`` (and other equations of motion methods)
        requires by calling the ``to_loads`` method.

        >>> spring.to_loads()
        [(pA, k*q(t)*N.x), (pB, - k*q(t)*N.x)]

        A simple linear damper can be modeled in a similar way. Create another
        symbol ``c`` to describe the dampers damping coefficient. This time
        instantiate a force actuator that produces a force proportional to both
        the damper's damping coefficient and the pathway's extension velocity.
        Note that the damping force is negative as it acts in the opposite
        direction to which the damper is changing in length.

        >>> damping_coefficient = symbols('c')
        >>> damping_force = -damping_coefficient*pathway.extension_velocity
        >>> damper = ForceActuator(damping_force, pathway)

        Again, the forces produces by the damper can be generated by calling
        the ``to_loads`` method.

        >>> damper.to_loads()
        [(pA, c*Derivative(q(t), t)*N.x), (pB, - c*Derivative(q(t), t)*N.x)]

        """
        return self.pathway.to_loads(self.force)

    def __repr__(self):
        """Representation of a ``ForceActuator``."""
        return f'{self.__class__.__name__}({self.force}, {self.pathway})'


class LinearSpring(ForceActuator):
    """A spring with its spring force as a linear function of its length.

    Explanation
    ===========

    Note that the "linear" in the name ``LinearSpring`` refers to the fact that
    the spring force is a linear function of the springs length. I.e. for a
    linear spring with stiffness ``k``, distance between its ends of ``x``, and
    an equilibrium length of ``0``, the spring force will be ``-k*x``, which is
    a linear function in ``x``. To create a spring that follows a linear, or
    straight, pathway between its two ends, a ``LinearPathway`` instance needs
    to be passed to the ``pathway`` parameter.

    A ``LinearSpring`` is a subclass of ``ForceActuator`` and so follows the
    same sign conventions for length, extension velocity, and the direction of
    the forces it applies to its points of attachment on bodies. The sign
    convention for the direction of forces is such that, for the case where a
    linear spring is instantiated with a ``LinearPathway`` instance as its
    pathway, they act to push the two ends of the spring away from one another.
    Because springs produces a contractile force and acts to pull the two ends
    together towards the equilibrium length when stretched, the scalar portion
    of the forces on the endpoint are negative in order to flip the sign of the
    forces on the endpoints when converted into vector quantities. The
    following diagram shows the positive force sense and the distance between
    the points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    To construct a linear spring, an expression (or symbol) must be supplied to
    represent the stiffness (spring constant) of the spring, alongside a
    pathway specifying its line of action. Let's also create a global reference
    frame and spatially fix one of the points in it while setting the other to
    be positioned such that it can freely move in the frame's x direction
    specified by the coordinate ``q``.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (LinearPathway, LinearSpring,
    ...     Point, ReferenceFrame)
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> stiffness = symbols('k')
    >>> pA, pB = Point('pA'), Point('pB')
    >>> pA.set_vel(N, 0)
    >>> pB.set_pos(pA, q*N.x)
    >>> pB.pos_from(pA)
    q(t)*N.x
    >>> linear_pathway = LinearPathway(pA, pB)
    >>> spring = LinearSpring(stiffness, linear_pathway)
    >>> spring
    LinearSpring(k, LinearPathway(pA, pB))

    This spring will produce a force that is proportional to both its stiffness
    and the pathway's length. Note that this force is negative as SymPy's sign
    convention for actuators is that negative forces are contractile.

    >>> spring.force
    -k*sqrt(q(t)**2)

    To create a linear spring with a non-zero equilibrium length, an expression
    (or symbol) can be passed to the ``equilibrium_length`` parameter on
    construction on a ``LinearSpring`` instance. Let's create a symbol ``l``
    to denote a non-zero equilibrium length and create another linear spring.

    >>> l = symbols('l')
    >>> spring = LinearSpring(stiffness, linear_pathway, equilibrium_length=l)
    >>> spring
    LinearSpring(k, LinearPathway(pA, pB), equilibrium_length=l)

    The spring force of this new spring is again proportional to both its
    stiffness and the pathway's length. However, the spring will not produce
    any force when ``q(t)`` equals ``l``. Note that the force will become
    expansile when ``q(t)`` is less than ``l``, as expected.

    >>> spring.force
    -k*(-l + sqrt(q(t)**2))

    Parameters
    ==========

    stiffness : Expr
        The spring constant.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
    equilibrium_length : Expr, optional
        The length at which the spring is in equilibrium, i.e. it produces no
        force. The default value is 0, i.e. the spring force is a linear
        function of the pathway's length with no constant offset.

    See Also
    ========

    ForceActuator: force-producing actuator (superclass of ``LinearSpring``).
    LinearPathway: straight-line pathway between a pair of points.

    """

    def __init__(self, stiffness, pathway, equilibrium_length=S.Zero):
        """Initializer for ``LinearSpring``.

        Parameters
        ==========

        stiffness : Expr
            The spring constant.
        pathway : PathwayBase
            The pathway that the actuator follows. This must be an instance of
            a concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.
        equilibrium_length : Expr, optional
            The length at which the spring is in equilibrium, i.e. it produces
            no force. The default value is 0, i.e. the spring force is a linear
            function of the pathway's length with no constant offset.

        """
        self.stiffness = stiffness
        self.pathway = pathway
        self.equilibrium_length = equilibrium_length

    @property
    def force(self):
        """The spring force produced by the linear spring."""
        return -self.stiffness*(self.pathway.length - self.equilibrium_length)

    @force.setter
    def force(self, force):
        raise AttributeError('Can\'t set computed attribute `force`.')

    @property
    def stiffness(self):
        """The spring constant for the linear spring."""
        return self._stiffness

    @stiffness.setter
    def stiffness(self, stiffness):
        if hasattr(self, '_stiffness'):
            msg = (
                f'Can\'t set attribute `stiffness` to {repr(stiffness)} as it '
                f'is immutable.'
            )
            raise AttributeError(msg)
        self._stiffness = sympify(stiffness, strict=True)

    @property
    def equilibrium_length(self):
        """The length of the spring at which it produces no force."""
        return self._equilibrium_length

    @equilibrium_length.setter
    def equilibrium_length(self, equilibrium_length):
        if hasattr(self, '_equilibrium_length'):
            msg = (
                f'Can\'t set attribute `equilibrium_length` to '
                f'{repr(equilibrium_length)} as it is immutable.'
            )
            raise AttributeError(msg)
        self._equilibrium_length = sympify(equilibrium_length, strict=True)

    def __repr__(self):
        """Representation of a ``LinearSpring``."""
        string = f'{self.__class__.__name__}({self.stiffness}, {self.pathway}'
        if self.equilibrium_length == S.Zero:
            string += ')'
        else:
            string += f', equilibrium_length={self.equilibrium_length})'
        return string


class LinearDamper(ForceActuator):
    """A damper whose force is a linear function of its extension velocity.

    Explanation
    ===========

    Note that the "linear" in the name ``LinearDamper`` refers to the fact that
    the damping force is a linear function of the damper's rate of change in
    its length. I.e. for a linear damper with damping ``c`` and extension
    velocity ``v``, the damping force will be ``-c*v``, which is a linear
    function in ``v``. To create a damper that follows a linear, or straight,
    pathway between its two ends, a ``LinearPathway`` instance needs to be
    passed to the ``pathway`` parameter.

    A ``LinearDamper`` is a subclass of ``ForceActuator`` and so follows the
    same sign conventions for length, extension velocity, and the direction of
    the forces it applies to its points of attachment on bodies. The sign
    convention for the direction of forces is such that, for the case where a
    linear damper is instantiated with a ``LinearPathway`` instance as its
    pathway, they act to push the two ends of the damper away from one another.
    Because dampers produce a force that opposes the direction of change in
    length, when extension velocity is positive the scalar portions of the
    forces applied at the two endpoints are negative in order to flip the sign
    of the forces on the endpoints wen converted into vector quantities. When
    extension velocity is negative (i.e. when the damper is shortening), the
    scalar portions of the fofces applied are also negative so that the signs
    cancel producing forces on the endpoints that are in the same direction as
    the positive sign convention for the forces at the endpoints of the pathway
    (i.e. they act to push the endpoints away from one another). The following
    diagram shows the positive force sense and the distance between the
    points::

       P           Q
       o<--- F --->o
       |           |
       |<--l(t)--->|

    Examples
    ========

    To construct a linear damper, an expression (or symbol) must be supplied to
    represent the damping coefficient of the damper (we'll use the symbol
    ``c``), alongside a pathway specifying its line of action. Let's also
    create a global reference frame and spatially fix one of the points in it
    while setting the other to be positioned such that it can freely move in
    the frame's x direction specified by the coordinate ``q``. The velocity
    that the two points move away from one another can be specified by the
    coordinate ``u`` where ``u`` is the first time derivative of ``q``
    (i.e., ``u = Derivative(q(t), t)``).

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (LinearDamper, LinearPathway,
    ...     Point, ReferenceFrame)
    >>> from sympy.physics.vector import dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q = dynamicsymbols('q')
    >>> damping = symbols('c')
    >>> pA, pB = Point('pA'), Point('pB')
    >>> pA.set_vel(N, 0)
    >>> pB.set_pos(pA, q*N.x)
    >>> pB.pos_from(pA)
    q(t)*N.x
    >>> pB.vel(N)
    Derivative(q(t), t)*N.x
    >>> linear_pathway = LinearPathway(pA, pB)
    >>> damper = LinearDamper(damping, linear_pathway)
    >>> damper
    LinearDamper(c, LinearPathway(pA, pB))

    This damper will produce a force that is proportional to both its damping
    coefficient and the pathway's extension length. Note that this force is
    negative as SymPy's sign convention for actuators is that negative forces
    are contractile and the damping force of the damper will oppose the
    direction of length change.

    >>> damper.force
    -c*sqrt(q(t)**2)*Derivative(q(t), t)/q(t)

    Parameters
    ==========

    damping : Expr
        The damping constant.
    pathway : PathwayBase
        The pathway that the actuator follows. This must be an instance of a
        concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.

    See Also
    ========

    ForceActuator: force-producing actuator (superclass of ``LinearDamper``).
    LinearPathway: straight-line pathway between a pair of points.

    """

    def __init__(self, damping, pathway):
        """Initializer for ``LinearDamper``.

        Parameters
        ==========

        damping : Expr
            The damping constant.
        pathway : PathwayBase
            The pathway that the actuator follows. This must be an instance of
            a concrete subclass of ``PathwayBase``, e.g. ``LinearPathway``.

        """
        self.damping = damping
        self.pathway = pathway

    @property
    def force(self):
        """The damping force produced by the linear damper."""
        return -self.damping*self.pathway.extension_velocity

    @force.setter
    def force(self, force):
        raise AttributeError('Can\'t set computed attribute `force`.')

    @property
    def damping(self):
        """The damping constant for the linear damper."""
        return self._damping

    @damping.setter
    def damping(self, damping):
        if hasattr(self, '_damping'):
            msg = (
                f'Can\'t set attribute `damping` to {repr(damping)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        self._damping = sympify(damping, strict=True)

    def __repr__(self):
        """Representation of a ``LinearDamper``."""
        return f'{self.__class__.__name__}({self.damping}, {self.pathway})'


class TorqueActuator(ActuatorBase):
    """Torque-producing actuator.

    Explanation
    ===========

    A ``TorqueActuator`` is an actuator that produces a pair of equal and
    opposite torques on a pair of bodies.

    Examples
    ========

    To construct a torque actuator, an expression (or symbol) must be supplied
    to represent the torque it can produce, alongside a vector specifying the
    axis about which the torque will act, and a pair of frames on which the
    torque will act.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import (ReferenceFrame, RigidBody,
    ...     TorqueActuator)
    >>> N = ReferenceFrame('N')
    >>> A = ReferenceFrame('A')
    >>> torque = symbols('T')
    >>> axis = N.z
    >>> parent = RigidBody('parent', frame=N)
    >>> child = RigidBody('child', frame=A)
    >>> bodies = (child, parent)
    >>> actuator = TorqueActuator(torque, axis, *bodies)
    >>> actuator
    TorqueActuator(T, axis=N.z, target_frame=A, reaction_frame=N)

    Note that because torques actually act on frames, not bodies,
    ``TorqueActuator`` will extract the frame associated with a ``RigidBody``
    when one is passed instead of a ``ReferenceFrame``.

    Parameters
    ==========

    torque : Expr
        The scalar expression defining the torque that the actuator produces.
    axis : Vector
        The axis about which the actuator applies torques.
    target_frame : ReferenceFrame | RigidBody
        The primary frame on which the actuator will apply the torque.
    reaction_frame : ReferenceFrame | RigidBody | None
        The secondary frame on which the actuator will apply the torque. Note
        that the (equal and opposite) reaction torque is applied to this frame.

    """

    def __init__(self, torque, axis, target_frame, reaction_frame=None):
        """Initializer for ``TorqueActuator``.

        Parameters
        ==========

        torque : Expr
            The scalar expression defining the torque that the actuator
            produces.
        axis : Vector
            The axis about which the actuator applies torques.
        target_frame : ReferenceFrame | RigidBody
            The primary frame on which the actuator will apply the torque.
        reaction_frame : ReferenceFrame | RigidBody | None
           The secondary frame on which the actuator will apply the torque.
           Note that the (equal and opposite) reaction torque is applied to
           this frame.

        """
        self.torque = torque
        self.axis = axis
        self.target_frame = target_frame
        self.reaction_frame = reaction_frame

    @classmethod
    def at_pin_joint(cls, torque, pin_joint):
        """Alternate constructor to instantiate from a ``PinJoint`` instance.

        Examples
        ========

        To create a pin joint the ``PinJoint`` class requires a name, parent
        body, and child body to be passed to its constructor. It is also
        possible to control the joint axis using the ``joint_axis`` keyword
        argument. In this example let's use the parent body's reference frame's
        z-axis as the joint axis.

        >>> from sympy.physics.mechanics import (PinJoint, ReferenceFrame,
        ...     RigidBody, TorqueActuator)
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> parent = RigidBody('parent', frame=N)
        >>> child = RigidBody('child', frame=A)
        >>> pin_joint = PinJoint(
        ...     'pin',
        ...     parent,
        ...     child,
        ...     joint_axis=N.z,
        ... )

        Let's also create a symbol ``T`` that will represent the torque applied
        by the torque actuator.

        >>> from sympy import symbols
        >>> torque = symbols('T')

        To create the torque actuator from the ``torque`` and ``pin_joint``
        variables previously instantiated, these can be passed to the alternate
        constructor class method ``at_pin_joint`` of the ``TorqueActuator``
        class. It should be noted that a positive torque will cause a positive
        displacement of the joint coordinate or that the torque is applied on
        the child body with a reaction torque on the parent.

        >>> actuator = TorqueActuator.at_pin_joint(torque, pin_joint)
        >>> actuator
        TorqueActuator(T, axis=N.z, target_frame=A, reaction_frame=N)

        Parameters
        ==========

        torque : Expr
            The scalar expression defining the torque that the actuator
            produces.
        pin_joint : PinJoint
            The pin joint, and by association the parent and child bodies, on
            which the torque actuator will act. The pair of bodies acted upon
            by the torque actuator are the parent and child bodies of the pin
            joint, with the child acting as the reaction body. The pin joint's
            axis is used as the axis about which the torque actuator will apply
            its torque.

        """
        if not isinstance(pin_joint, PinJoint):
            msg = (
                f'Value {repr(pin_joint)} passed to `pin_joint` was of type '
                f'{type(pin_joint)}, must be {PinJoint}.'
            )
            raise TypeError(msg)
        return cls(
            torque,
            pin_joint.joint_axis,
            pin_joint.child_interframe,
            pin_joint.parent_interframe,
        )

    @property
    def torque(self):
        """The magnitude of the torque produced by the actuator."""
        return self._torque

    @torque.setter
    def torque(self, torque):
        if hasattr(self, '_torque'):
            msg = (
                f'Can\'t set attribute `torque` to {repr(torque)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        self._torque = sympify(torque, strict=True)

    @property
    def axis(self):
        """The axis about which the torque acts."""
        return self._axis

    @axis.setter
    def axis(self, axis):
        if hasattr(self, '_axis'):
            msg = (
                f'Can\'t set attribute `axis` to {repr(axis)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        if not isinstance(axis, Vector):
            msg = (
                f'Value {repr(axis)} passed to `axis` was of type '
                f'{type(axis)}, must be {Vector}.'
            )
            raise TypeError(msg)
        self._axis = axis

    @property
    def target_frame(self):
        """The primary reference frames on which the torque will act."""
        return self._target_frame

    @target_frame.setter
    def target_frame(self, target_frame):
        if hasattr(self, '_target_frame'):
            msg = (
                f'Can\'t set attribute `target_frame` to {repr(target_frame)} '
                f'as it is immutable.'
            )
            raise AttributeError(msg)
        if isinstance(target_frame, RigidBody):
            target_frame = target_frame.frame
        elif not isinstance(target_frame, ReferenceFrame):
            msg = (
                f'Value {repr(target_frame)} passed to `target_frame` was of '
                f'type {type(target_frame)}, must be {ReferenceFrame}.'
            )
            raise TypeError(msg)
        self._target_frame = target_frame

    @property
    def reaction_frame(self):
        """The primary reference frames on which the torque will act."""
        return self._reaction_frame

    @reaction_frame.setter
    def reaction_frame(self, reaction_frame):
        if hasattr(self, '_reaction_frame'):
            msg = (
                f'Can\'t set attribute `reaction_frame` to '
                f'{repr(reaction_frame)} as it is immutable.'
            )
            raise AttributeError(msg)
        if isinstance(reaction_frame, RigidBody):
            reaction_frame = reaction_frame.frame
        elif (
            not isinstance(reaction_frame, ReferenceFrame)
            and reaction_frame is not None
        ):
            msg = (
                f'Value {repr(reaction_frame)} passed to `reaction_frame` was '
                f'of type {type(reaction_frame)}, must be {ReferenceFrame}.'
            )
            raise TypeError(msg)
        self._reaction_frame = reaction_frame

    def to_loads(self):
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

        The below example shows how to generate the loads produced by a torque
        actuator that acts on a pair of bodies attached by a pin joint.

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import (PinJoint, ReferenceFrame,
        ...     RigidBody, TorqueActuator)
        >>> torque = symbols('T')
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> parent = RigidBody('parent', frame=N)
        >>> child = RigidBody('child', frame=A)
        >>> pin_joint = PinJoint(
        ...     'pin',
        ...     parent,
        ...     child,
        ...     joint_axis=N.z,
        ... )
        >>> actuator = TorqueActuator.at_pin_joint(torque, pin_joint)

        The forces produces by the damper can be generated by calling the
        ``to_loads`` method.

        >>> actuator.to_loads()
        [(A, T*N.z), (N, - T*N.z)]

        Alternatively, if a torque actuator is created without a reaction frame
        then the loads returned by the ``to_loads`` method will contain just
        the single load acting on the target frame.

        >>> actuator = TorqueActuator(torque, N.z, N)
        >>> actuator.to_loads()
        [(N, T*N.z)]

        """
        loads = [
            Torque(self.target_frame, self.torque*self.axis),
        ]
        if self.reaction_frame is not None:
            loads.append(Torque(self.reaction_frame, -self.torque*self.axis))
        return loads

    def __repr__(self):
        """Representation of a ``TorqueActuator``."""
        string = (
            f'{self.__class__.__name__}({self.torque}, axis={self.axis}, '
            f'target_frame={self.target_frame}'
        )
        if self.reaction_frame is not None:
            string += f', reaction_frame={self.reaction_frame})'
        else:
            string += ')'
        return string


class DuffingSpring(ForceActuator):
    """A nonlinear spring based on the Duffing equation.

    Explanation
    ===========

    Here, ``DuffingSpring`` represents the force exerted by a nonlinear spring based on the Duffing equation:
    F = -beta*x-alpha*x**3, where x is the displacement from the equilibrium position, beta is the linear spring constant,
    and alpha is the coefficient for the nonlinear cubic term.

    Parameters
    ==========

    linear_stiffness : Expr
        The linear stiffness coefficient (beta).
    nonlinear_stiffness : Expr
        The nonlinear stiffness coefficient (alpha).
    pathway : PathwayBase
        The pathway that the actuator follows.
    equilibrium_length : Expr, optional
        The length at which the spring is in equilibrium (x).
    """

    def __init__(self, linear_stiffness, nonlinear_stiffness, pathway, equilibrium_length=S.Zero):
        self.linear_stiffness = sympify(linear_stiffness, strict=True)
        self.nonlinear_stiffness = sympify(nonlinear_stiffness, strict=True)
        self.equilibrium_length = sympify(equilibrium_length, strict=True)

        if not isinstance(pathway, PathwayBase):
            raise TypeError("pathway must be an instance of PathwayBase.")
        self._pathway = pathway

    @property
    def linear_stiffness(self):
        return self._linear_stiffness

    @linear_stiffness.setter
    def linear_stiffness(self, linear_stiffness):
        if hasattr(self, '_linear_stiffness'):
            msg = (
                f'Can\'t set attribute `linear_stiffness` to '
                f'{repr(linear_stiffness)} as it is immutable.'
            )
            raise AttributeError(msg)
        self._linear_stiffness = sympify(linear_stiffness, strict=True)

    @property
    def nonlinear_stiffness(self):
        return self._nonlinear_stiffness

    @nonlinear_stiffness.setter
    def nonlinear_stiffness(self, nonlinear_stiffness):
        if hasattr(self, '_nonlinear_stiffness'):
            msg = (
                f'Can\'t set attribute `nonlinear_stiffness` to '
                f'{repr(nonlinear_stiffness)} as it is immutable.'
            )
            raise AttributeError(msg)
        self._nonlinear_stiffness = sympify(nonlinear_stiffness, strict=True)

    @property
    def pathway(self):
        return self._pathway

    @pathway.setter
    def pathway(self, pathway):
        if hasattr(self, '_pathway'):
            msg = (
                f'Can\'t set attribute `pathway` to {repr(pathway)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        if not isinstance(pathway, PathwayBase):
            msg = (
                f'Value {repr(pathway)} passed to `pathway` was of type '
                f'{type(pathway)}, must be {PathwayBase}.'
            )
            raise TypeError(msg)
        self._pathway = pathway

    @property
    def equilibrium_length(self):
        return self._equilibrium_length

    @equilibrium_length.setter
    def equilibrium_length(self, equilibrium_length):
        if hasattr(self, '_equilibrium_length'):
            msg = (
                f'Can\'t set attribute `equilibrium_length` to '
                f'{repr(equilibrium_length)} as it is immutable.'
            )
            raise AttributeError(msg)
        self._equilibrium_length = sympify(equilibrium_length, strict=True)

    @property
    def force(self):
        """The force produced by the Duffing spring."""
        displacement = self.pathway.length - self.equilibrium_length
        return -self.linear_stiffness * displacement - self.nonlinear_stiffness * displacement**3

    @force.setter
    def force(self, force):
        if hasattr(self, '_force'):
            msg = (
                f'Can\'t set attribute `force` to {repr(force)} as it is '
                f'immutable.'
            )
            raise AttributeError(msg)
        self._force = sympify(force, strict=True)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"{self.linear_stiffness}, {self.nonlinear_stiffness}, {self.pathway}, "
                f"equilibrium_length={self.equilibrium_length})")

class CoulombKineticFriction(ForceActuator):
    r"""Coulomb kinetic friction with Stribeck and viscous effects.

    Explanation
    ===========

    This represents a Coulomb kinetic friction with the Stribeck and viscous effect,
    described by the function:

    .. math::
        F = (\mu_k f_n + (\mu_s - \mu_k) f_n e^{-(\frac{v}{v_s})^2}) \text{sign}(v) + \sigma  v

    where :math:`\mu_k` is the coefficient of kinetic friction, :math:`\mu_s` is the
    coefficient of static friction, :math:`f_n` is the normal force, :math:`v` is the
    relative velocity, :math:`v_s` is the Stribeck friction coefficient, and
    :math:`\sigma` is the viscous friction constant.

    The default friction force is :math:`F = \mu_k f_n`.
    When specified, the actuator includes:

    - Stribeck effect: :math:`(\mu_s - \mu_k) f_n e^{-(\frac{v}{v_s})^2}`
    - Viscous effect: :math:`\sigma v`

    Notes
    =====

    The actuator makes the following assumptions:

    - The actuator assumes relative motion is non-zero.
    - The normal force is assumed to be a non-negative scalar.
    - The resultant friction force is opposite to the velocity direction.
    - Each point in the pathway is fixed within separate objects that are sliding relative to each other. In other words, these two points are fixed in the mutually sliding objects.

    This actuator has been tested for straightforward motions, like a block sliding
    on a surface.

    The friction force is defined to always oppose the direction of relative velocity :math:`v`.
    Specifically:

    - The default Coulomb friction force :math:`\mu_k f_n \text{sign}(v)` is opposite to :math:`v`.
    - The Stribeck effect :math:`(\mu_s - \mu_k) f_n e^{-(\frac{v}{v_s})^2} \text{sign}(v)` is also opposite to :math:`v`.
    - The viscous friction term :math:`\sigma v` is opposite to :math:`v`.

    Examples
    ========

    The below example shows how to generate the loads produced by a Coulomb kinetic
    friction actuator in a mass-spring system with friction.

    >>> import sympy as sm
    >>> from sympy.physics.mechanics import (dynamicsymbols, ReferenceFrame, Point,
    ...     LinearPathway, CoulombKineticFriction, LinearSpring, KanesMethod, Particle)

    >>> x, v = dynamicsymbols('x, v', real=True)
    >>> m, g, k, mu_k, mu_s, v_s, sigma = sm.symbols('m, g, k, mu_k, mu_s, v_s, sigma')

    >>> N = ReferenceFrame('N')
    >>> O, P = Point('O'), Point('P')
    >>> O.set_vel(N, 0)
    >>> P.set_pos(O, x*N.x)

    >>> pathway = LinearPathway(O, P)
    >>> friction = CoulombKineticFriction(mu_k, m*g, pathway, v_s=v_s, sigma=sigma, mu_s=mu_k)
    >>> spring = LinearSpring(k, pathway)
    >>> block = Particle('block', point=P, mass=m)

    >>> kane = KanesMethod(N, (x,), (v,), kd_eqs=(x.diff() - v,))
    >>> friction.to_loads()
        [(O, (g*m*mu_k*sign(sign(x(t))*Derivative(x(t), t)) + sigma*sign(x(t))*Derivative(x(t), t))*x(t)/Abs(x(t))*N.x), (P, (-g*m*mu_k*sign(sign(x(t))*Derivative(x(t), t)) - sigma*sign(x(t))*Derivative(x(t), t))*x(t)/Abs(x(t))*N.x)]
    >>> loads = friction.to_loads() + spring.to_loads()
    >>> fr, frstar = kane.kanes_equations([block], loads)
    >>> eom = fr + frstar
    >>> eom
        Matrix([[-k*x(t) - m*Derivative(v(t), t) + (-g*m*mu_k*sign(v(t)*sign(x(t))) - sigma*v(t)*sign(x(t)))*x(t)/Abs(x(t))]])

    Parameters
    ==========

    f_n : sympifiable
        The normal force between the surfaces. It should always be a non-negative scalar.
    mu_k : sympifiable
        The coefficient of kinetic friction.
    pathway : PathwayBase
        The pathway that the actuator follows.
    v_s : sympifiable, optional
        The Stribeck friction coefficient.
    sigma : sympifiable, optional
        The viscous friction coefficient.
    mu_s : sympifiable, optional
        The coefficient of static friction. Defaults to mu_k, meaning the Stribeck effect evaluates to 0 by default.

    References
    ==========

    .. [Moore2022] https://moorepants.github.io/learn-multibody-dynamics/loads.html#friction.
    .. [Flores2023] Paulo Flores, Jorge Ambrosio, Hamid M. Lankarani,
            "Contact-impact events with friction in multibody dynamics: Back to basics",
            Mechanism and Machine Theory, vol. 184, 2023. https://doi.org/10.1016/j.mechmachtheory.2023.105305.
    .. [Rogner2017] I. Rogner, "Friction modelling for robotic applications with planar motion",
            Chalmers University of Technology, Department of Electrical Engineering, 2017.

    """

    def __init__(self, mu_k, f_n, pathway, *, v_s=None, sigma=None, mu_s=None):
        self._mu_k = sympify(mu_k, strict=True) if mu_k is not None else 1
        self._mu_s = sympify(mu_s, strict=True) if mu_s is not None else self._mu_k
        self._f_n = sympify(f_n, strict=True)
        self._sigma = sympify(sigma, strict=True) if sigma is not None else 0
        self._v_s = sympify(v_s, strict=True) if v_s is not None or v_s == 0 else 0.01
        self.pathway = pathway

    @property
    def mu_k(self):
        """The coefficient of kinetic friction."""
        return self._mu_k

    @property
    def mu_s(self):
        """The coefficient of static friction."""
        return self._mu_s

    @property
    def f_n(self):
        """The normal force between the surfaces."""
        return self._f_n

    @property
    def sigma(self):
        """The viscous friction coefficient."""
        return self._sigma

    @property
    def v_s(self):
        """The Stribeck friction coefficient."""
        return self._v_s

    @property
    def force(self):
        v = self.pathway.extension_velocity
        f_c = self.mu_k * self.f_n
        f_max = self.mu_s * self.f_n
        stribeck_term = (f_max - f_c) * exp(-(v / self.v_s)**2) if self.v_s is not None else 0
        viscous_term = self.sigma * v if self.sigma is not None else 0
        return (f_c + stribeck_term) * -sign(v) - viscous_term

    @force.setter
    def force(self, force):
        raise AttributeError('Can\'t set computed attribute `force`.')

    def __repr__(self):
        return (f'{self.__class__.__name__}({self._mu_k}, {self._mu_s} '
                f'{self._f_n}, {self.pathway}, {self._v_s}, '
                f'{self._sigma})')
