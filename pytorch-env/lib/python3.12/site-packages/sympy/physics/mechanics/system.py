from functools import wraps

from sympy.core.basic import Basic
from sympy.matrices.immutable import ImmutableMatrix
from sympy.matrices.dense import Matrix, eye, zeros
from sympy.core.containers import OrderedSet
from sympy.physics.mechanics.actuator import ActuatorBase
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.mechanics.functions import (
    Lagrangian, _validate_coordinates, find_dynamicsymbols)
from sympy.physics.mechanics.joint import Joint
from sympy.physics.mechanics.kane import KanesMethod
from sympy.physics.mechanics.lagrange import LagrangesMethod
from sympy.physics.mechanics.loads import _parse_load, gravity
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.particle import Particle
from sympy.physics.vector import Point, ReferenceFrame, dynamicsymbols
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import filldedent

__all__ = ['SymbolicSystem', 'System']


def _reset_eom_method(method):
    """Decorator to reset the eom_method if a property is changed."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self._eom_method = None
        return method(self, *args, **kwargs)

    return wrapper


class System(_Methods):
    """Class to define a multibody system and form its equations of motion.

    Explanation
    ===========

    A ``System`` instance stores the different objects associated with a model,
    including bodies, joints, constraints, and other relevant information. With
    all the relationships between components defined, the ``System`` can be used
    to form the equations of motion using a backend, such as ``KanesMethod``.
    The ``System`` has been designed to be compatible with third-party
    libraries for greater flexibility and integration with other tools.

    Attributes
    ==========

    frame : ReferenceFrame
        Inertial reference frame of the system.
    fixed_point : Point
        A fixed point in the inertial reference frame.
    x : Vector
        Unit vector fixed in the inertial reference frame.
    y : Vector
        Unit vector fixed in the inertial reference frame.
    z : Vector
        Unit vector fixed in the inertial reference frame.
    q : ImmutableMatrix
        Matrix of all the generalized coordinates, i.e. the independent
        generalized coordinates stacked upon the dependent.
    u : ImmutableMatrix
        Matrix of all the generalized speeds, i.e. the independent generealized
        speeds stacked upon the dependent.
    q_ind : ImmutableMatrix
        Matrix of the independent generalized coordinates.
    q_dep : ImmutableMatrix
        Matrix of the dependent generalized coordinates.
    u_ind : ImmutableMatrix
        Matrix of the independent generalized speeds.
    u_dep : ImmutableMatrix
        Matrix of the dependent generalized speeds.
    u_aux : ImmutableMatrix
        Matrix of auxiliary generalized speeds.
    kdes : ImmutableMatrix
        Matrix of the kinematical differential equations as expressions equated
        to the zero matrix.
    bodies : tuple of BodyBase subclasses
        Tuple of all bodies that make up the system.
    joints : tuple of Joint
        Tuple of all joints that connect bodies in the system.
    loads : tuple of LoadBase subclasses
        Tuple of all loads that have been applied to the system.
    actuators : tuple of ActuatorBase subclasses
        Tuple of all actuators present in the system.
    holonomic_constraints : ImmutableMatrix
        Matrix with the holonomic constraints as expressions equated to the zero
        matrix.
    nonholonomic_constraints : ImmutableMatrix
        Matrix with the nonholonomic constraints as expressions equated to the
        zero matrix.
    velocity_constraints : ImmutableMatrix
        Matrix with the velocity constraints as expressions equated to the zero
        matrix. These are by default derived as the time derivatives of the
        holonomic constraints extended with the nonholonomic constraints.
    eom_method : subclass of KanesMethod or LagrangesMethod
        Backend for forming the equations of motion.

    Examples
    ========

    In the example below a cart with a pendulum is created. The cart moves along
    the x axis of the rail and the pendulum rotates about the z axis. The length
    of the pendulum is ``l`` with the pendulum represented as a particle. To
    move the cart a time dependent force ``F`` is applied to the cart.

    We first need to import some functions and create some of our variables.

    >>> from sympy import symbols, simplify
    >>> from sympy.physics.mechanics import (
    ...     mechanics_printing, dynamicsymbols, RigidBody, Particle,
    ...     ReferenceFrame, PrismaticJoint, PinJoint, System)
    >>> mechanics_printing(pretty_print=False)
    >>> g, l = symbols('g l')
    >>> F = dynamicsymbols('F')

    The next step is to create bodies. It is also useful to create a frame for
    locating the particle with respect to the pin joint later on, as a particle
    does not have a body-fixed frame.

    >>> rail = RigidBody('rail')
    >>> cart = RigidBody('cart')
    >>> bob = Particle('bob')
    >>> bob_frame = ReferenceFrame('bob_frame')

    Initialize the system, with the rail as the Newtonian reference. The body is
    also automatically added to the system.

    >>> system = System.from_newtonian(rail)
    >>> print(system.bodies[0])
    rail

    Create the joints, while immediately also adding them to the system.

    >>> system.add_joints(
    ...     PrismaticJoint('slider', rail, cart, joint_axis=rail.x),
    ...     PinJoint('pin', cart, bob, joint_axis=cart.z,
    ...              child_interframe=bob_frame,
    ...              child_point=l * bob_frame.y)
    ... )
    >>> system.joints
    (PrismaticJoint: slider  parent: rail  child: cart,
    PinJoint: pin  parent: cart  child: bob)

    While adding the joints, the associated generalized coordinates, generalized
    speeds, kinematic differential equations and bodies are also added to the
    system.

    >>> system.q
    Matrix([
    [q_slider],
    [   q_pin]])
    >>> system.u
    Matrix([
    [u_slider],
    [   u_pin]])
    >>> system.kdes
    Matrix([
    [u_slider - q_slider'],
    [      u_pin - q_pin']])
    >>> [body.name for body in system.bodies]
    ['rail', 'cart', 'bob']

    With the kinematics established, we can now apply gravity and the cart force
    ``F``.

    >>> system.apply_uniform_gravity(-g * system.y)
    >>> system.add_loads((cart.masscenter, F * rail.x))
    >>> system.loads
    ((rail_masscenter, - g*rail_mass*rail_frame.y),
     (cart_masscenter, - cart_mass*g*rail_frame.y),
     (bob_masscenter, - bob_mass*g*rail_frame.y),
     (cart_masscenter, F*rail_frame.x))

    With the entire system defined, we can now form the equations of motion.
    Before forming the equations of motion, one can also run some checks that
    will try to identify some common errors.

    >>> system.validate_system()
    >>> system.form_eoms()
    Matrix([
    [bob_mass*l*u_pin**2*sin(q_pin) - bob_mass*l*cos(q_pin)*u_pin'
     - (bob_mass + cart_mass)*u_slider' + F],
    [                   -bob_mass*g*l*sin(q_pin) - bob_mass*l**2*u_pin'
     - bob_mass*l*cos(q_pin)*u_slider']])
    >>> simplify(system.mass_matrix)
    Matrix([
    [ bob_mass + cart_mass, bob_mass*l*cos(q_pin)],
    [bob_mass*l*cos(q_pin),         bob_mass*l**2]])
    >>> system.forcing
    Matrix([
    [bob_mass*l*u_pin**2*sin(q_pin) + F],
    [          -bob_mass*g*l*sin(q_pin)]])

    The complexity of the above example can be increased if we add a constraint
    to prevent the particle from moving in the horizontal (x) direction. This
    can be done by adding a holonomic constraint. After which we should also
    redefine what our (in)dependent generalized coordinates and speeds are.

    >>> system.add_holonomic_constraints(
    ...     bob.masscenter.pos_from(rail.masscenter).dot(system.x)
    ... )
    >>> system.q_ind = system.get_joint('pin').coordinates
    >>> system.q_dep = system.get_joint('slider').coordinates
    >>> system.u_ind = system.get_joint('pin').speeds
    >>> system.u_dep = system.get_joint('slider').speeds

    With the updated system the equations of motion can be formed again.

    >>> system.validate_system()
    >>> system.form_eoms()
    Matrix([[-bob_mass*g*l*sin(q_pin)
             - bob_mass*l**2*u_pin'
             - bob_mass*l*cos(q_pin)*u_slider'
             - l*(bob_mass*l*u_pin**2*sin(q_pin)
             - bob_mass*l*cos(q_pin)*u_pin'
             - (bob_mass + cart_mass)*u_slider')*cos(q_pin)
             - l*F*cos(q_pin)]])
    >>> simplify(system.mass_matrix)
    Matrix([
    [bob_mass*l**2*sin(q_pin)**2, -cart_mass*l*cos(q_pin)],
    [               l*cos(q_pin),                       1]])
    >>> simplify(system.forcing)
    Matrix([
    [-l*(bob_mass*g*sin(q_pin) + bob_mass*l*u_pin**2*sin(2*q_pin)/2
     + F*cos(q_pin))],
    [
    l*u_pin**2*sin(q_pin)]])

    """

    def __init__(self, frame=None, fixed_point=None):
        """Initialize the system.

        Parameters
        ==========

        frame : ReferenceFrame, optional
            The inertial frame of the system. If none is supplied, a new frame
            will be created.
        fixed_point : Point, optional
            A fixed point in the inertial reference frame. If none is supplied,
            a new fixed_point will be created.

        """
        if frame is None:
            frame = ReferenceFrame('inertial_frame')
        elif not isinstance(frame, ReferenceFrame):
            raise TypeError('Frame must be an instance of ReferenceFrame.')
        self._frame = frame
        if fixed_point is None:
            fixed_point = Point('inertial_point')
        elif not isinstance(fixed_point, Point):
            raise TypeError('Fixed point must be an instance of Point.')
        self._fixed_point = fixed_point
        self._fixed_point.set_vel(self._frame, 0)
        self._q_ind = ImmutableMatrix(1, 0, []).T
        self._q_dep = ImmutableMatrix(1, 0, []).T
        self._u_ind = ImmutableMatrix(1, 0, []).T
        self._u_dep = ImmutableMatrix(1, 0, []).T
        self._u_aux = ImmutableMatrix(1, 0, []).T
        self._kdes = ImmutableMatrix(1, 0, []).T
        self._hol_coneqs = ImmutableMatrix(1, 0, []).T
        self._nonhol_coneqs = ImmutableMatrix(1, 0, []).T
        self._vel_constrs = None
        self._bodies = []
        self._joints = []
        self._loads = []
        self._actuators = []
        self._eom_method = None

    @classmethod
    def from_newtonian(cls, newtonian):
        """Constructs the system with respect to a Newtonian body."""
        if isinstance(newtonian, Particle):
            raise TypeError('A Particle has no frame so cannot act as '
                            'the Newtonian.')
        system = cls(frame=newtonian.frame, fixed_point=newtonian.masscenter)
        system.add_bodies(newtonian)
        return system

    @property
    def fixed_point(self):
        """Fixed point in the inertial reference frame."""
        return self._fixed_point

    @property
    def frame(self):
        """Inertial reference frame of the system."""
        return self._frame

    @property
    def x(self):
        """Unit vector fixed in the inertial reference frame."""
        return self._frame.x

    @property
    def y(self):
        """Unit vector fixed in the inertial reference frame."""
        return self._frame.y

    @property
    def z(self):
        """Unit vector fixed in the inertial reference frame."""
        return self._frame.z

    @property
    def bodies(self):
        """Tuple of all bodies that have been added to the system."""
        return tuple(self._bodies)

    @bodies.setter
    @_reset_eom_method
    def bodies(self, bodies):
        bodies = self._objects_to_list(bodies)
        self._check_objects(bodies, [], BodyBase, 'Bodies', 'bodies')
        self._bodies = bodies

    @property
    def joints(self):
        """Tuple of all joints that have been added to the system."""
        return tuple(self._joints)

    @joints.setter
    @_reset_eom_method
    def joints(self, joints):
        joints = self._objects_to_list(joints)
        self._check_objects(joints, [], Joint, 'Joints', 'joints')
        self._joints = []
        self.add_joints(*joints)

    @property
    def loads(self):
        """Tuple of loads that have been applied on the system."""
        return tuple(self._loads)

    @loads.setter
    @_reset_eom_method
    def loads(self, loads):
        loads = self._objects_to_list(loads)
        self._loads = [_parse_load(load) for load in loads]

    @property
    def actuators(self):
        """Tuple of actuators present in the system."""
        return tuple(self._actuators)

    @actuators.setter
    @_reset_eom_method
    def actuators(self, actuators):
        actuators = self._objects_to_list(actuators)
        self._check_objects(actuators, [], ActuatorBase, 'Actuators',
                            'actuators')
        self._actuators = actuators

    @property
    def q(self):
        """Matrix of all the generalized coordinates with the independent
        stacked upon the dependent."""
        return self._q_ind.col_join(self._q_dep)

    @property
    def u(self):
        """Matrix of all the generalized speeds with the independent stacked
        upon the dependent."""
        return self._u_ind.col_join(self._u_dep)

    @property
    def q_ind(self):
        """Matrix of the independent generalized coordinates."""
        return self._q_ind

    @q_ind.setter
    @_reset_eom_method
    def q_ind(self, q_ind):
        self._q_ind, self._q_dep = self._parse_coordinates(
            self._objects_to_list(q_ind), True, [], self.q_dep, 'coordinates')

    @property
    def q_dep(self):
        """Matrix of the dependent generalized coordinates."""
        return self._q_dep

    @q_dep.setter
    @_reset_eom_method
    def q_dep(self, q_dep):
        self._q_ind, self._q_dep = self._parse_coordinates(
            self._objects_to_list(q_dep), False, self.q_ind, [], 'coordinates')

    @property
    def u_ind(self):
        """Matrix of the independent generalized speeds."""
        return self._u_ind

    @u_ind.setter
    @_reset_eom_method
    def u_ind(self, u_ind):
        self._u_ind, self._u_dep = self._parse_coordinates(
            self._objects_to_list(u_ind), True, [], self.u_dep, 'speeds')

    @property
    def u_dep(self):
        """Matrix of the dependent generalized speeds."""
        return self._u_dep

    @u_dep.setter
    @_reset_eom_method
    def u_dep(self, u_dep):
        self._u_ind, self._u_dep = self._parse_coordinates(
            self._objects_to_list(u_dep), False, self.u_ind, [], 'speeds')

    @property
    def u_aux(self):
        """Matrix of auxiliary generalized speeds."""
        return self._u_aux

    @u_aux.setter
    @_reset_eom_method
    def u_aux(self, u_aux):
        self._u_aux = self._parse_coordinates(
            self._objects_to_list(u_aux), True, [], [], 'u_auxiliary')[0]

    @property
    def kdes(self):
        """Kinematical differential equations as expressions equated to the zero
        matrix. These equations describe the coupling between the generalized
        coordinates and the generalized speeds."""
        return self._kdes

    @kdes.setter
    @_reset_eom_method
    def kdes(self, kdes):
        kdes = self._objects_to_list(kdes)
        self._kdes = self._parse_expressions(
            kdes, [], 'kinematic differential equations')

    @property
    def holonomic_constraints(self):
        """Matrix with the holonomic constraints as expressions equated to the
        zero matrix."""
        return self._hol_coneqs

    @holonomic_constraints.setter
    @_reset_eom_method
    def holonomic_constraints(self, constraints):
        constraints = self._objects_to_list(constraints)
        self._hol_coneqs = self._parse_expressions(
            constraints, [], 'holonomic constraints')

    @property
    def nonholonomic_constraints(self):
        """Matrix with the nonholonomic constraints as expressions equated to
        the zero matrix."""
        return self._nonhol_coneqs

    @nonholonomic_constraints.setter
    @_reset_eom_method
    def nonholonomic_constraints(self, constraints):
        constraints = self._objects_to_list(constraints)
        self._nonhol_coneqs = self._parse_expressions(
            constraints, [], 'nonholonomic constraints')

    @property
    def velocity_constraints(self):
        """Matrix with the velocity constraints as expressions equated to the
        zero matrix. The velocity constraints are by default derived from the
        holonomic and nonholonomic constraints unless they are explicitly set.
        """
        if self._vel_constrs is None:
            return self.holonomic_constraints.diff(dynamicsymbols._t).col_join(
                self.nonholonomic_constraints)
        return self._vel_constrs

    @velocity_constraints.setter
    @_reset_eom_method
    def velocity_constraints(self, constraints):
        if constraints is None:
            self._vel_constrs = None
            return
        constraints = self._objects_to_list(constraints)
        self._vel_constrs = self._parse_expressions(
            constraints, [], 'velocity constraints')

    @property
    def eom_method(self):
        """Backend for forming the equations of motion."""
        return self._eom_method

    @staticmethod
    def _objects_to_list(lst):
        """Helper to convert passed objects to a list."""
        if not iterable(lst):  # Only one object
            return [lst]
        return list(lst[:])  # converts Matrix and tuple to flattened list

    @staticmethod
    def _check_objects(objects, obj_lst, expected_type, obj_name, type_name):
        """Helper to check the objects that are being added to the system.

        Explanation
        ===========
        This method checks that the objects that are being added to the system
        are of the correct type and have not already been added. If any of the
        objects are not of the correct type or have already been added, then
        an error is raised.

        Parameters
        ==========
        objects : iterable
            The objects that would be added to the system.
        obj_lst : list
            The list of objects that are already in the system.
        expected_type : type
            The type that the objects should be.
        obj_name : str
            The name of the category of objects. This string is used to
            formulate the error message for the user.
        type_name : str
            The name of the type that the objects should be. This string is used
            to formulate the error message for the user.

        """
        seen = set(obj_lst)
        duplicates = set()
        wrong_types = set()
        for obj in objects:
            if not isinstance(obj, expected_type):
                wrong_types.add(obj)
            if obj in seen:
                duplicates.add(obj)
            else:
                seen.add(obj)
        if wrong_types:
            raise TypeError(f'{obj_name} {wrong_types} are not {type_name}.')
        if duplicates:
            raise ValueError(f'{obj_name} {duplicates} have already been added '
                             f'to the system.')

    def _parse_coordinates(self, new_coords, independent, old_coords_ind,
                           old_coords_dep, coord_type='coordinates'):
        """Helper to parse coordinates and speeds."""
        # Construct lists of the independent and dependent coordinates
        coords_ind, coords_dep = old_coords_ind[:], old_coords_dep[:]
        if not iterable(independent):
            independent = [independent] * len(new_coords)
        for coord, indep in zip(new_coords, independent):
            if indep:
                coords_ind.append(coord)
            else:
                coords_dep.append(coord)
        # Check types and duplicates
        current = {'coordinates': self.q_ind[:] + self.q_dep[:],
                   'speeds': self.u_ind[:] + self.u_dep[:],
                   'u_auxiliary': self._u_aux[:],
                   coord_type: coords_ind + coords_dep}
        _validate_coordinates(**current)
        return (ImmutableMatrix(1, len(coords_ind), coords_ind).T,
                ImmutableMatrix(1, len(coords_dep), coords_dep).T)

    @staticmethod
    def _parse_expressions(new_expressions, old_expressions, name,
                           check_negatives=False):
        """Helper to parse expressions like constraints."""
        old_expressions = old_expressions[:]
        new_expressions = list(new_expressions)  # Converts a possible tuple
        if check_negatives:
            check_exprs = old_expressions + [-expr for expr in old_expressions]
        else:
            check_exprs = old_expressions
        System._check_objects(new_expressions, check_exprs, Basic, name,
                              'expressions')
        for expr in new_expressions:
            if expr == 0:
                raise ValueError(f'Parsed {name} are zero.')
        return ImmutableMatrix(1, len(old_expressions) + len(new_expressions),
                               old_expressions + new_expressions).T

    @_reset_eom_method
    def add_coordinates(self, *coordinates, independent=True):
        """Add generalized coordinate(s) to the system.

        Parameters
        ==========

        *coordinates : dynamicsymbols
            One or more generalized coordinates to be added to the system.
        independent : bool or list of bool, optional
            Boolean whether a coordinate is dependent or independent. The
            default is True, so the coordinates are added as independent by
            default.

        """
        self._q_ind, self._q_dep = self._parse_coordinates(
            coordinates, independent, self.q_ind, self.q_dep, 'coordinates')

    @_reset_eom_method
    def add_speeds(self, *speeds, independent=True):
        """Add generalized speed(s) to the system.

        Parameters
        ==========

        *speeds : dynamicsymbols
            One or more generalized speeds to be added to the system.
        independent : bool or list of bool, optional
            Boolean whether a speed is dependent or independent. The default is
            True, so the speeds are added as independent by default.

        """
        self._u_ind, self._u_dep = self._parse_coordinates(
            speeds, independent, self.u_ind, self.u_dep, 'speeds')

    @_reset_eom_method
    def add_auxiliary_speeds(self, *speeds):
        """Add auxiliary speed(s) to the system.

        Parameters
        ==========

        *speeds : dynamicsymbols
            One or more auxiliary speeds to be added to the system.

        """
        self._u_aux = self._parse_coordinates(
            speeds, True, self._u_aux, [], 'u_auxiliary')[0]

    @_reset_eom_method
    def add_kdes(self, *kdes):
        """Add kinematic differential equation(s) to the system.

        Parameters
        ==========

        *kdes : Expr
            One or more kinematic differential equations.

        """
        self._kdes = self._parse_expressions(
            kdes, self.kdes, 'kinematic differential equations',
            check_negatives=True)

    @_reset_eom_method
    def add_holonomic_constraints(self, *constraints):
        """Add holonomic constraint(s) to the system.

        Parameters
        ==========

        *constraints : Expr
            One or more holonomic constraints, which are expressions that should
            be zero.

        """
        self._hol_coneqs = self._parse_expressions(
            constraints, self._hol_coneqs, 'holonomic constraints',
            check_negatives=True)

    @_reset_eom_method
    def add_nonholonomic_constraints(self, *constraints):
        """Add nonholonomic constraint(s) to the system.

        Parameters
        ==========

        *constraints : Expr
            One or more nonholonomic constraints, which are expressions that
            should be zero.

        """
        self._nonhol_coneqs = self._parse_expressions(
            constraints, self._nonhol_coneqs, 'nonholonomic constraints',
            check_negatives=True)

    @_reset_eom_method
    def add_bodies(self, *bodies):
        """Add body(ies) to the system.

        Parameters
        ==========

        bodies : Particle or RigidBody
            One or more bodies.

        """
        self._check_objects(bodies, self.bodies, BodyBase, 'Bodies', 'bodies')
        self._bodies.extend(bodies)

    @_reset_eom_method
    def add_loads(self, *loads):
        """Add load(s) to the system.

        Parameters
        ==========

        *loads : Force or Torque
            One or more loads.

        """
        loads = [_parse_load(load) for load in loads]  # Checks the loads
        self._loads.extend(loads)

    @_reset_eom_method
    def apply_uniform_gravity(self, acceleration):
        """Apply uniform gravity to all bodies in the system by adding loads.

        Parameters
        ==========

        acceleration : Vector
            The acceleration due to gravity.

        """
        self.add_loads(*gravity(acceleration, *self.bodies))

    @_reset_eom_method
    def add_actuators(self, *actuators):
        """Add actuator(s) to the system.

        Parameters
        ==========

        *actuators : subclass of ActuatorBase
            One or more actuators.

        """
        self._check_objects(actuators, self.actuators, ActuatorBase,
                            'Actuators', 'actuators')
        self._actuators.extend(actuators)

    @_reset_eom_method
    def add_joints(self, *joints):
        """Add joint(s) to the system.

        Explanation
        ===========

        This methods adds one or more joints to the system including its
        associated objects, i.e. generalized coordinates, generalized speeds,
        kinematic differential equations and the bodies.

        Parameters
        ==========

        *joints : subclass of Joint
            One or more joints.

        Notes
        =====

        For the generalized coordinates, generalized speeds and bodies it is
        checked whether they are already known by the system instance. If they
        are, then they are not added. The kinematic differential equations are
        however always added to the system, so you should not also manually add
        those on beforehand.

        """
        self._check_objects(joints, self.joints, Joint, 'Joints', 'joints')
        self._joints.extend(joints)
        coordinates, speeds, kdes, bodies = (OrderedSet() for _ in range(4))
        for joint in joints:
            coordinates.update(joint.coordinates)
            speeds.update(joint.speeds)
            kdes.update(joint.kdes)
            bodies.update((joint.parent, joint.child))
        coordinates = coordinates.difference(self.q)
        speeds = speeds.difference(self.u)
        kdes = kdes.difference(self.kdes[:] + (-self.kdes)[:])
        bodies = bodies.difference(self.bodies)
        self.add_coordinates(*tuple(coordinates))
        self.add_speeds(*tuple(speeds))
        self.add_kdes(*(kde for kde in tuple(kdes) if not kde == 0))
        self.add_bodies(*tuple(bodies))

    def get_body(self, name):
        """Retrieve a body from the system by name.

        Parameters
        ==========

        name : str
            The name of the body to retrieve.

        Returns
        =======

        RigidBody or Particle
            The body with the given name, or None if no such body exists.

        """
        for body in self._bodies:
            if body.name == name:
                return body

    def get_joint(self, name):
        """Retrieve a joint from the system by name.

        Parameters
        ==========

        name : str
            The name of the joint to retrieve.

        Returns
        =======

        subclass of Joint
            The joint with the given name, or None if no such joint exists.

        """
        for joint in self._joints:
            if joint.name == name:
                return joint

    def _form_eoms(self):
        return self.form_eoms()

    def form_eoms(self, eom_method=KanesMethod, **kwargs):
        """Form the equations of motion of the system.

        Parameters
        ==========

        eom_method : subclass of KanesMethod or LagrangesMethod
            Backend class to be used for forming the equations of motion. The
            default is ``KanesMethod``.

        Returns
        ========

        ImmutableMatrix
            Vector of equations of motions.

        Examples
        ========

        This is a simple example for a one degree of freedom translational
        spring-mass-damper.

        >>> from sympy import S, symbols
        >>> from sympy.physics.mechanics import (
        ...     LagrangesMethod, dynamicsymbols, PrismaticJoint, Particle,
        ...     RigidBody, System)
        >>> q = dynamicsymbols('q')
        >>> qd = dynamicsymbols('q', 1)
        >>> m, k, b = symbols('m k b')
        >>> wall = RigidBody('W')
        >>> system = System.from_newtonian(wall)
        >>> bob = Particle('P', mass=m)
        >>> bob.potential_energy = S.Half * k * q**2
        >>> system.add_joints(PrismaticJoint('J', wall, bob, q, qd))
        >>> system.add_loads((bob.masscenter, b * qd * system.x))
        >>> system.form_eoms(LagrangesMethod)
        Matrix([[-b*Derivative(q(t), t) + k*q(t) + m*Derivative(q(t), (t, 2))]])

        We can also solve for the states using the 'rhs' method.

        >>> system.rhs()
        Matrix([
        [               Derivative(q(t), t)],
        [(b*Derivative(q(t), t) - k*q(t))/m]])

        """
        # KanesMethod does not accept empty iterables
        loads = self.loads + tuple(
            load for act in self.actuators for load in act.to_loads())
        loads = loads if loads else None
        if issubclass(eom_method, KanesMethod):
            disallowed_kwargs = {
                "frame", "q_ind", "u_ind", "kd_eqs", "q_dependent",
                "u_dependent", "u_auxiliary", "configuration_constraints",
                "velocity_constraints", "forcelist", "bodies"}
            wrong_kwargs = disallowed_kwargs.intersection(kwargs)
            if wrong_kwargs:
                raise ValueError(
                    f"The following keyword arguments are not allowed to be "
                    f"overwritten in {eom_method.__name__}: {wrong_kwargs}.")
            kwargs = {"frame": self.frame, "q_ind": self.q_ind,
                      "u_ind": self.u_ind, "kd_eqs": self.kdes,
                      "q_dependent": self.q_dep, "u_dependent": self.u_dep,
                      "configuration_constraints": self.holonomic_constraints,
                      "velocity_constraints": self.velocity_constraints,
                      "u_auxiliary": self.u_aux,
                      "forcelist": loads, "bodies": self.bodies,
                      "explicit_kinematics": False, **kwargs}
            self._eom_method = eom_method(**kwargs)
        elif issubclass(eom_method, LagrangesMethod):
            disallowed_kwargs = {
                "frame", "qs", "forcelist", "bodies", "hol_coneqs",
                "nonhol_coneqs", "Lagrangian"}
            wrong_kwargs = disallowed_kwargs.intersection(kwargs)
            if wrong_kwargs:
                raise ValueError(
                    f"The following keyword arguments are not allowed to be "
                    f"overwritten in {eom_method.__name__}: {wrong_kwargs}.")
            kwargs = {"frame": self.frame, "qs": self.q, "forcelist": loads,
                      "bodies": self.bodies,
                      "hol_coneqs": self.holonomic_constraints,
                      "nonhol_coneqs": self.nonholonomic_constraints, **kwargs}
            if "Lagrangian" not in kwargs:
                kwargs["Lagrangian"] = Lagrangian(kwargs["frame"],
                                                  *kwargs["bodies"])
            self._eom_method = eom_method(**kwargs)
        else:
            raise NotImplementedError(f'{eom_method} has not been implemented.')
        return self.eom_method._form_eoms()

    def rhs(self, inv_method=None):
        """Compute the equations of motion in the explicit form.

        Parameters
        ==========

        inv_method : str
            The specific sympy inverse matrix calculation method to use. For a
            list of valid methods, see
            :meth:`~sympy.matrices.matrixbase.MatrixBase.inv`

        Returns
        ========

        ImmutableMatrix
            Equations of motion in the explicit form.

        See Also
        ========

        sympy.physics.mechanics.kane.KanesMethod.rhs:
            KanesMethod's ``rhs`` function.
        sympy.physics.mechanics.lagrange.LagrangesMethod.rhs:
            LagrangesMethod's ``rhs`` function.

        """
        return self.eom_method.rhs(inv_method=inv_method)

    @property
    def mass_matrix(self):
        r"""The mass matrix of the system.

        Explanation
        ===========

        The mass matrix $M_d$ and the forcing vector $f_d$ of a system describe
        the system's dynamics according to the following equations:

        .. math::
            M_d \dot{u} = f_d

        where $\dot{u}$ is the time derivative of the generalized speeds.

        """
        return self.eom_method.mass_matrix

    @property
    def mass_matrix_full(self):
        r"""The mass matrix of the system, augmented by the kinematic
        differential equations in explicit or implicit form.

        Explanation
        ===========

        The full mass matrix $M_m$ and the full forcing vector $f_m$ of a system
        describe the dynamics and kinematics according to the following
        equation:

        .. math::
            M_m \dot{x} = f_m

        where $x$ is the state vector stacking $q$ and $u$.

        """
        return self.eom_method.mass_matrix_full

    @property
    def forcing(self):
        """The forcing vector of the system."""
        return self.eom_method.forcing

    @property
    def forcing_full(self):
        """The forcing vector of the system, augmented by the kinematic
        differential equations in explicit or implicit form."""
        return self.eom_method.forcing_full

    def validate_system(self, eom_method=KanesMethod, check_duplicates=False):
        """Validates the system using some basic checks.

        Explanation
        ===========

        This method validates the system based on the following checks:

        - The number of dependent generalized coordinates should equal the
          number of holonomic constraints.
        - All generalized coordinates defined by the joints should also be known
          to the system.
        - If ``KanesMethod`` is used as a ``eom_method``:
            - All generalized speeds and kinematic differential equations
              defined by the joints should also be known to the system.
            - The number of dependent generalized speeds should equal the number
              of velocity constraints.
            - The number of generalized coordinates should be less than or equal
              to the number of generalized speeds.
            - The number of generalized coordinates should equal the number of
              kinematic differential equations.
        - If ``LagrangesMethod`` is used as ``eom_method``:
            - There should not be any generalized speeds that are not
              derivatives of the generalized coordinates (this includes the
              generalized speeds defined by the joints).

        Parameters
        ==========

        eom_method : subclass of KanesMethod or LagrangesMethod
            Backend class that will be used for forming the equations of motion.
            There are different checks for the different backends. The default
            is ``KanesMethod``.
        check_duplicates : bool
            Boolean whether the system should be checked for duplicate
            definitions. The default is False, because duplicates are already
            checked when adding objects to the system.

        Notes
        =====

        This method is not guaranteed to be backwards compatible as it may
        improve over time. The method can become both more and less strict in
        certain areas. However a well-defined system should always pass all
        these tests.

        """
        msgs = []
        # Save some data in variables
        n_hc = self.holonomic_constraints.shape[0]
        n_vc = self.velocity_constraints.shape[0]
        n_q_dep, n_u_dep = self.q_dep.shape[0], self.u_dep.shape[0]
        q_set, u_set = set(self.q), set(self.u)
        n_q, n_u = len(q_set), len(u_set)
        # Check number of holonomic constraints
        if n_q_dep != n_hc:
            msgs.append(filldedent(f"""
            The number of dependent generalized coordinates {n_q_dep} should be
            equal to the number of holonomic constraints {n_hc}."""))
        # Check if all joint coordinates and speeds are present
        missing_q = set()
        for joint in self.joints:
            missing_q.update(set(joint.coordinates).difference(q_set))
        if missing_q:
            msgs.append(filldedent(f"""
            The generalized coordinates {missing_q} used in joints are not added
            to the system."""))
        # Method dependent checks
        if issubclass(eom_method, KanesMethod):
            n_kdes = len(self.kdes)
            missing_kdes, missing_u = set(), set()
            for joint in self.joints:
                missing_u.update(set(joint.speeds).difference(u_set))
                missing_kdes.update(set(joint.kdes).difference(
                    self.kdes[:] + (-self.kdes)[:]))
            if missing_u:
                msgs.append(filldedent(f"""
                The generalized speeds {missing_u} used in joints are not added
                to the system."""))
            if missing_kdes:
                msgs.append(filldedent(f"""
                The kinematic differential equations {missing_kdes} used in
                joints are not added to the system."""))
            if n_u_dep != n_vc:
                msgs.append(filldedent(f"""
                The number of dependent generalized speeds {n_u_dep} should be
                equal to the number of velocity constraints {n_vc}."""))
            if n_q > n_u:
                msgs.append(filldedent(f"""
                The number of generalized coordinates {n_q} should be less than
                or equal to the number of generalized speeds {n_u}."""))
            if n_u != n_kdes:
                msgs.append(filldedent(f"""
                The number of generalized speeds {n_u} should be equal to the
                number of kinematic differential equations {n_kdes}."""))
        elif issubclass(eom_method, LagrangesMethod):
            not_qdots = set(self.u).difference(self.q.diff(dynamicsymbols._t))
            for joint in self.joints:
                not_qdots.update(set(
                    joint.speeds).difference(self.q.diff(dynamicsymbols._t)))
            if not_qdots:
                msgs.append(filldedent(f"""
                The generalized speeds {not_qdots} are not supported by this
                method. Only derivatives of the generalized coordinates are
                supported. If these symbols are used in your expressions, then
                this will result in wrong equations of motion."""))
            if self.u_aux:
                msgs.append(filldedent(f"""
                This method does not support auxiliary speeds. If these symbols
                are used in your expressions, then this will result in wrong
                equations of motion. The auxiliary speeds are {self.u_aux}."""))
        else:
            raise NotImplementedError(f'{eom_method} has not been implemented.')
        if check_duplicates:  # Should be redundant
            duplicates_to_check = [('generalized coordinates', self.q),
                                   ('generalized speeds', self.u),
                                   ('auxiliary speeds', self.u_aux),
                                   ('bodies', self.bodies),
                                   ('joints', self.joints)]
            for name, lst in duplicates_to_check:
                seen = set()
                duplicates = {x for x in lst if x in seen or seen.add(x)}
                if duplicates:
                    msgs.append(filldedent(f"""
                    The {name} {duplicates} exist multiple times within the
                    system."""))
        if msgs:
            raise ValueError('\n'.join(msgs))


class SymbolicSystem:
    """SymbolicSystem is a class that contains all the information about a
    system in a symbolic format such as the equations of motions and the bodies
    and loads in the system.

    There are three ways that the equations of motion can be described for
    Symbolic System:


        [1] Explicit form where the kinematics and dynamics are combined
            x' = F_1(x, t, r, p)

        [2] Implicit form where the kinematics and dynamics are combined
            M_2(x, p) x' = F_2(x, t, r, p)

        [3] Implicit form where the kinematics and dynamics are separate
            M_3(q, p) u' = F_3(q, u, t, r, p)
            q' = G(q, u, t, r, p)

    where

    x : states, e.g. [q, u]
    t : time
    r : specified (exogenous) inputs
    p : constants
    q : generalized coordinates
    u : generalized speeds
    F_1 : right hand side of the combined equations in explicit form
    F_2 : right hand side of the combined equations in implicit form
    F_3 : right hand side of the dynamical equations in implicit form
    M_2 : mass matrix of the combined equations in implicit form
    M_3 : mass matrix of the dynamical equations in implicit form
    G : right hand side of the kinematical differential equations

        Parameters
        ==========

        coord_states : ordered iterable of functions of time
            This input will either be a collection of the coordinates or states
            of the system depending on whether or not the speeds are also
            given. If speeds are specified this input will be assumed to
            be the coordinates otherwise this input will be assumed to
            be the states.

        right_hand_side : Matrix
            This variable is the right hand side of the equations of motion in
            any of the forms. The specific form will be assumed depending on
            whether a mass matrix or coordinate derivatives are given.

        speeds : ordered iterable of functions of time, optional
            This is a collection of the generalized speeds of the system. If
            given it will be assumed that the first argument (coord_states)
            will represent the generalized coordinates of the system.

        mass_matrix : Matrix, optional
            The matrix of the implicit forms of the equations of motion (forms
            [2] and [3]). The distinction between the forms is determined by
            whether or not the coordinate derivatives are passed in. If
            they are given form [3] will be assumed otherwise form [2] is
            assumed.

        coordinate_derivatives : Matrix, optional
            The right hand side of the kinematical equations in explicit form.
            If given it will be assumed that the equations of motion are being
            entered in form [3].

        alg_con : Iterable, optional
            The indexes of the rows in the equations of motion that contain
            algebraic constraints instead of differential equations. If the
            equations are input in form [3], it will be assumed the indexes are
            referencing the mass_matrix/right_hand_side combination and not the
            coordinate_derivatives.

        output_eqns : Dictionary, optional
            Any output equations that are desired to be tracked are stored in a
            dictionary where the key corresponds to the name given for the
            specific equation and the value is the equation itself in symbolic
            form

        coord_idxs : Iterable, optional
            If coord_states corresponds to the states rather than the
            coordinates this variable will tell SymbolicSystem which indexes of
            the states correspond to generalized coordinates.

        speed_idxs : Iterable, optional
            If coord_states corresponds to the states rather than the
            coordinates this variable will tell SymbolicSystem which indexes of
            the states correspond to generalized speeds.

        bodies : iterable of Body/Rigidbody objects, optional
            Iterable containing the bodies of the system

        loads : iterable of load instances (described below), optional
            Iterable containing the loads of the system where forces are given
            by (point of application, force vector) and torques are given by
            (reference frame acting upon, torque vector). Ex [(point, force),
            (ref_frame, torque)]

    Attributes
    ==========

    coordinates : Matrix, shape(n, 1)
        This is a matrix containing the generalized coordinates of the system

    speeds : Matrix, shape(m, 1)
        This is a matrix containing the generalized speeds of the system

    states : Matrix, shape(o, 1)
        This is a matrix containing the state variables of the system

    alg_con : List
        This list contains the indices of the algebraic constraints in the
        combined equations of motion. The presence of these constraints
        requires that a DAE solver be used instead of an ODE solver.
        If the system is given in form [3] the alg_con variable will be
        adjusted such that it is a representation of the combined kinematics
        and dynamics thus make sure it always matches the mass matrix
        entered.

    dyn_implicit_mat : Matrix, shape(m, m)
        This is the M matrix in form [3] of the equations of motion (the mass
        matrix or generalized inertia matrix of the dynamical equations of
        motion in implicit form).

    dyn_implicit_rhs : Matrix, shape(m, 1)
        This is the F vector in form [3] of the equations of motion (the right
        hand side of the dynamical equations of motion in implicit form).

    comb_implicit_mat : Matrix, shape(o, o)
        This is the M matrix in form [2] of the equations of motion.
        This matrix contains a block diagonal structure where the top
        left block (the first rows) represent the matrix in the
        implicit form of the kinematical equations and the bottom right
        block (the last rows) represent the matrix in the implicit form
        of the dynamical equations.

    comb_implicit_rhs : Matrix, shape(o, 1)
        This is the F vector in form [2] of the equations of motion. The top
        part of the vector represents the right hand side of the implicit form
        of the kinemaical equations and the bottom of the vector represents the
        right hand side of the implicit form of the dynamical equations of
        motion.

    comb_explicit_rhs : Matrix, shape(o, 1)
        This vector represents the right hand side of the combined equations of
        motion in explicit form (form [1] from above).

    kin_explicit_rhs : Matrix, shape(m, 1)
        This is the right hand side of the explicit form of the kinematical
        equations of motion as can be seen in form [3] (the G matrix).

    output_eqns : Dictionary
        If output equations were given they are stored in a dictionary where
        the key corresponds to the name given for the specific equation and
        the value is the equation itself in symbolic form

    bodies : Tuple
        If the bodies in the system were given they are stored in a tuple for
        future access

    loads : Tuple
        If the loads in the system were given they are stored in a tuple for
        future access. This includes forces and torques where forces are given
        by (point of application, force vector) and torques are given by
        (reference frame acted upon, torque vector).

    Example
    =======

    As a simple example, the dynamics of a simple pendulum will be input into a
    SymbolicSystem object manually. First some imports will be needed and then
    symbols will be set up for the length of the pendulum (l), mass at the end
    of the pendulum (m), and a constant for gravity (g). ::

        >>> from sympy import Matrix, sin, symbols
        >>> from sympy.physics.mechanics import dynamicsymbols, SymbolicSystem
        >>> l, m, g = symbols('l m g')

    The system will be defined by an angle of theta from the vertical and a
    generalized speed of omega will be used where omega = theta_dot. ::

        >>> theta, omega = dynamicsymbols('theta omega')

    Now the equations of motion are ready to be formed and passed to the
    SymbolicSystem object. ::

        >>> kin_explicit_rhs = Matrix([omega])
        >>> dyn_implicit_mat = Matrix([l**2 * m])
        >>> dyn_implicit_rhs = Matrix([-g * l * m * sin(theta)])
        >>> symsystem = SymbolicSystem([theta], dyn_implicit_rhs, [omega],
        ...                            dyn_implicit_mat)

    Notes
    =====

    m : number of generalized speeds
    n : number of generalized coordinates
    o : number of states

    """

    def __init__(self, coord_states, right_hand_side, speeds=None,
                 mass_matrix=None, coordinate_derivatives=None, alg_con=None,
                 output_eqns={}, coord_idxs=None, speed_idxs=None, bodies=None,
                 loads=None):
        """Initializes a SymbolicSystem object"""

        # Extract information on speeds, coordinates and states
        if speeds is None:
            self._states = Matrix(coord_states)

            if coord_idxs is None:
                self._coordinates = None
            else:
                coords = [coord_states[i] for i in coord_idxs]
                self._coordinates = Matrix(coords)

            if speed_idxs is None:
                self._speeds = None
            else:
                speeds_inter = [coord_states[i] for i in speed_idxs]
                self._speeds = Matrix(speeds_inter)
        else:
            self._coordinates = Matrix(coord_states)
            self._speeds = Matrix(speeds)
            self._states = self._coordinates.col_join(self._speeds)

        # Extract equations of motion form
        if coordinate_derivatives is not None:
            self._kin_explicit_rhs = coordinate_derivatives
            self._dyn_implicit_rhs = right_hand_side
            self._dyn_implicit_mat = mass_matrix
            self._comb_implicit_rhs = None
            self._comb_implicit_mat = None
            self._comb_explicit_rhs = None
        elif mass_matrix is not None:
            self._kin_explicit_rhs = None
            self._dyn_implicit_rhs = None
            self._dyn_implicit_mat = None
            self._comb_implicit_rhs = right_hand_side
            self._comb_implicit_mat = mass_matrix
            self._comb_explicit_rhs = None
        else:
            self._kin_explicit_rhs = None
            self._dyn_implicit_rhs = None
            self._dyn_implicit_mat = None
            self._comb_implicit_rhs = None
            self._comb_implicit_mat = None
            self._comb_explicit_rhs = right_hand_side

        # Set the remainder of the inputs as instance attributes
        if alg_con is not None and coordinate_derivatives is not None:
            alg_con = [i + len(coordinate_derivatives) for i in alg_con]
        self._alg_con = alg_con
        self.output_eqns = output_eqns

        # Change the body and loads iterables to tuples if they are not tuples
        # already
        if not isinstance(bodies, tuple) and bodies is not None:
            bodies = tuple(bodies)
        if not isinstance(loads, tuple) and loads is not None:
            loads = tuple(loads)
        self._bodies = bodies
        self._loads = loads

    @property
    def coordinates(self):
        """Returns the column matrix of the generalized coordinates"""
        if self._coordinates is None:
            raise AttributeError("The coordinates were not specified.")
        else:
            return self._coordinates

    @property
    def speeds(self):
        """Returns the column matrix of generalized speeds"""
        if self._speeds is None:
            raise AttributeError("The speeds were not specified.")
        else:
            return self._speeds

    @property
    def states(self):
        """Returns the column matrix of the state variables"""
        return self._states

    @property
    def alg_con(self):
        """Returns a list with the indices of the rows containing algebraic
        constraints in the combined form of the equations of motion"""
        return self._alg_con

    @property
    def dyn_implicit_mat(self):
        """Returns the matrix, M, corresponding to the dynamic equations in
        implicit form, M x' = F, where the kinematical equations are not
        included"""
        if self._dyn_implicit_mat is None:
            raise AttributeError("dyn_implicit_mat is not specified for "
                                 "equations of motion form [1] or [2].")
        else:
            return self._dyn_implicit_mat

    @property
    def dyn_implicit_rhs(self):
        """Returns the column matrix, F, corresponding to the dynamic equations
        in implicit form, M x' = F, where the kinematical equations are not
        included"""
        if self._dyn_implicit_rhs is None:
            raise AttributeError("dyn_implicit_rhs is not specified for "
                                 "equations of motion form [1] or [2].")
        else:
            return self._dyn_implicit_rhs

    @property
    def comb_implicit_mat(self):
        """Returns the matrix, M, corresponding to the equations of motion in
        implicit form (form [2]), M x' = F, where the kinematical equations are
        included"""
        if self._comb_implicit_mat is None:
            if self._dyn_implicit_mat is not None:
                num_kin_eqns = len(self._kin_explicit_rhs)
                num_dyn_eqns = len(self._dyn_implicit_rhs)
                zeros1 = zeros(num_kin_eqns, num_dyn_eqns)
                zeros2 = zeros(num_dyn_eqns, num_kin_eqns)
                inter1 = eye(num_kin_eqns).row_join(zeros1)
                inter2 = zeros2.row_join(self._dyn_implicit_mat)
                self._comb_implicit_mat = inter1.col_join(inter2)
                return self._comb_implicit_mat
            else:
                raise AttributeError("comb_implicit_mat is not specified for "
                                     "equations of motion form [1].")
        else:
            return self._comb_implicit_mat

    @property
    def comb_implicit_rhs(self):
        """Returns the column matrix, F, corresponding to the equations of
        motion in implicit form (form [2]), M x' = F, where the kinematical
        equations are included"""
        if self._comb_implicit_rhs is None:
            if self._dyn_implicit_rhs is not None:
                kin_inter = self._kin_explicit_rhs
                dyn_inter = self._dyn_implicit_rhs
                self._comb_implicit_rhs = kin_inter.col_join(dyn_inter)
                return self._comb_implicit_rhs
            else:
                raise AttributeError("comb_implicit_mat is not specified for "
                                     "equations of motion in form [1].")
        else:
            return self._comb_implicit_rhs

    def compute_explicit_form(self):
        """If the explicit right hand side of the combined equations of motion
        is to provided upon initialization, this method will calculate it. This
        calculation can potentially take awhile to compute."""
        if self._comb_explicit_rhs is not None:
            raise AttributeError("comb_explicit_rhs is already formed.")

        inter1 = getattr(self, 'kin_explicit_rhs', None)
        if inter1 is not None:
            inter2 = self._dyn_implicit_mat.LUsolve(self._dyn_implicit_rhs)
            out = inter1.col_join(inter2)
        else:
            out = self._comb_implicit_mat.LUsolve(self._comb_implicit_rhs)

        self._comb_explicit_rhs = out

    @property
    def comb_explicit_rhs(self):
        """Returns the right hand side of the equations of motion in explicit
        form, x' = F, where the kinematical equations are included"""
        if self._comb_explicit_rhs is None:
            raise AttributeError("Please run .combute_explicit_form before "
                                 "attempting to access comb_explicit_rhs.")
        else:
            return self._comb_explicit_rhs

    @property
    def kin_explicit_rhs(self):
        """Returns the right hand side of the kinematical equations in explicit
        form, q' = G"""
        if self._kin_explicit_rhs is None:
            raise AttributeError("kin_explicit_rhs is not specified for "
                                 "equations of motion form [1] or [2].")
        else:
            return self._kin_explicit_rhs

    def dynamic_symbols(self):
        """Returns a column matrix containing all of the symbols in the system
        that depend on time"""
        # Create a list of all of the expressions in the equations of motion
        if self._comb_explicit_rhs is None:
            eom_expressions = (self.comb_implicit_mat[:] +
                               self.comb_implicit_rhs[:])
        else:
            eom_expressions = (self._comb_explicit_rhs[:])

        functions_of_time = set()
        for expr in eom_expressions:
            functions_of_time = functions_of_time.union(
                find_dynamicsymbols(expr))
        functions_of_time = functions_of_time.union(self._states)

        return tuple(functions_of_time)

    def constant_symbols(self):
        """Returns a column matrix containing all of the symbols in the system
        that do not depend on time"""
        # Create a list of all of the expressions in the equations of motion
        if self._comb_explicit_rhs is None:
            eom_expressions = (self.comb_implicit_mat[:] +
                               self.comb_implicit_rhs[:])
        else:
            eom_expressions = (self._comb_explicit_rhs[:])

        constants = set()
        for expr in eom_expressions:
            constants = constants.union(expr.free_symbols)
        constants.remove(dynamicsymbols._t)

        return tuple(constants)

    @property
    def bodies(self):
        """Returns the bodies in the system"""
        if self._bodies is None:
            raise AttributeError("bodies were not specified for the system.")
        else:
            return self._bodies

    @property
    def loads(self):
        """Returns the loads in the system"""
        if self._loads is None:
            raise AttributeError("loads were not specified for the system.")
        else:
            return self._loads
