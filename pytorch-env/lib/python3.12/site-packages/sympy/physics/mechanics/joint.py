# coding=utf-8

from abc import ABC, abstractmethod

from sympy import pi, Derivative, Matrix
from sympy.core.function import AppliedUndef
from sympy.physics.mechanics.body_base import BodyBase
from sympy.physics.mechanics.functions import _validate_coordinates
from sympy.physics.vector import (Vector, dynamicsymbols, cross, Point,
                                  ReferenceFrame)
from sympy.utilities.iterables import iterable
from sympy.utilities.exceptions import sympy_deprecation_warning

__all__ = ['Joint', 'PinJoint', 'PrismaticJoint', 'CylindricalJoint',
           'PlanarJoint', 'SphericalJoint', 'WeldJoint']


class Joint(ABC):
    """Abstract base class for all specific joints.

    Explanation
    ===========

    A joint subtracts degrees of freedom from a body. This is the base class
    for all specific joints and holds all common methods acting as an interface
    for all joints. Custom joint can be created by inheriting Joint class and
    defining all abstract functions.

    The abstract methods are:

    - ``_generate_coordinates``
    - ``_generate_speeds``
    - ``_orient_frames``
    - ``_set_angular_velocity``
    - ``_set_linear_velocity``

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody or Body
        The parent body of joint.
    child : Particle or RigidBody or Body
        The child body of joint.
    coordinates : iterable of dynamicsymbols, optional
        Generalized coordinates of the joint.
    speeds : iterable of dynamicsymbols, optional
        Generalized speeds of joint.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the parent body which aligns with an axis fixed in the
            child body. The default is the x axis of parent's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    child_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the child body which aligns with an axis fixed in the
            parent body. The default is the x axis of child's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.
    parent_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by parent_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.
    child_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by child_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody or Body
        The joint's parent body.
    child : Particle or RigidBody or Body
        The joint's child body.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates.
    speeds : Matrix
        Matrix of the joint's generalized speeds.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_axis : Vector
        The axis fixed in the parent frame that represents the joint.
    child_axis : Vector
        The axis fixed in the child frame that represents the joint.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    kdes : Matrix
        Kinematical differential equations of the joint.

    Notes
    =====

    When providing a vector as the intermediate frame, a new intermediate frame
    is created which aligns its X axis with the provided vector. This is done
    with a single fixed rotation about a rotation axis. This rotation axis is
    determined by taking the cross product of the ``body.x`` axis with the
    provided vector. In the case where the provided vector is in the ``-body.x``
    direction, the rotation is done about the ``body.y`` axis.

    """

    def __init__(self, name, parent, child, coordinates=None, speeds=None,
                 parent_point=None, child_point=None, parent_interframe=None,
                 child_interframe=None, parent_axis=None, child_axis=None,
                 parent_joint_pos=None, child_joint_pos=None):

        if not isinstance(name, str):
            raise TypeError('Supply a valid name.')
        self._name = name

        if not isinstance(parent, BodyBase):
            raise TypeError('Parent must be a body.')
        self._parent = parent

        if not isinstance(child, BodyBase):
            raise TypeError('Child must be a body.')
        self._child = child

        if parent_axis is not None or child_axis is not None:
            sympy_deprecation_warning(
                """
                The parent_axis and child_axis arguments for the Joint classes
                are deprecated. Instead use parent_interframe, child_interframe.
                """,
                deprecated_since_version="1.12",
                active_deprecations_target="deprecated-mechanics-joint-axis",
                stacklevel=4
            )
            if parent_interframe is None:
                parent_interframe = parent_axis
            if child_interframe is None:
                child_interframe = child_axis

        # Set parent and child frame attributes
        if hasattr(self._parent, 'frame'):
            self._parent_frame = self._parent.frame
        else:
            if isinstance(parent_interframe, ReferenceFrame):
                self._parent_frame = parent_interframe
            else:
                self._parent_frame = ReferenceFrame(
                    f'{self.name}_{self._parent.name}_frame')
        if hasattr(self._child, 'frame'):
            self._child_frame = self._child.frame
        else:
            if isinstance(child_interframe, ReferenceFrame):
                self._child_frame = child_interframe
            else:
                self._child_frame = ReferenceFrame(
                    f'{self.name}_{self._child.name}_frame')

        self._parent_interframe = self._locate_joint_frame(
            self._parent, parent_interframe, self._parent_frame)
        self._child_interframe = self._locate_joint_frame(
            self._child, child_interframe, self._child_frame)
        self._parent_axis = self._axis(parent_axis, self._parent_frame)
        self._child_axis = self._axis(child_axis, self._child_frame)

        if parent_joint_pos is not None or child_joint_pos is not None:
            sympy_deprecation_warning(
                """
                The parent_joint_pos and child_joint_pos arguments for the Joint
                classes are deprecated. Instead use parent_point and child_point.
                """,
                deprecated_since_version="1.12",
                active_deprecations_target="deprecated-mechanics-joint-pos",
                stacklevel=4
            )
            if parent_point is None:
                parent_point = parent_joint_pos
            if child_point is None:
                child_point = child_joint_pos
        self._parent_point = self._locate_joint_pos(
            self._parent, parent_point, self._parent_frame)
        self._child_point = self._locate_joint_pos(
            self._child, child_point, self._child_frame)

        self._coordinates = self._generate_coordinates(coordinates)
        self._speeds = self._generate_speeds(speeds)
        _validate_coordinates(self.coordinates, self.speeds)
        self._kdes = self._generate_kdes()

        self._orient_frames()
        self._set_angular_velocity()
        self._set_linear_velocity()

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    @property
    def name(self):
        """Name of the joint."""
        return self._name

    @property
    def parent(self):
        """Parent body of Joint."""
        return self._parent

    @property
    def child(self):
        """Child body of Joint."""
        return self._child

    @property
    def coordinates(self):
        """Matrix of the joint's generalized coordinates."""
        return self._coordinates

    @property
    def speeds(self):
        """Matrix of the joint's generalized speeds."""
        return self._speeds

    @property
    def kdes(self):
        """Kinematical differential equations of the joint."""
        return self._kdes

    @property
    def parent_axis(self):
        """The axis of parent frame."""
        # Will be removed with `deprecated-mechanics-joint-axis`
        return self._parent_axis

    @property
    def child_axis(self):
        """The axis of child frame."""
        # Will be removed with `deprecated-mechanics-joint-axis`
        return self._child_axis

    @property
    def parent_point(self):
        """Attachment point where the joint is fixed to the parent body."""
        return self._parent_point

    @property
    def child_point(self):
        """Attachment point where the joint is fixed to the child body."""
        return self._child_point

    @property
    def parent_interframe(self):
        return self._parent_interframe

    @property
    def child_interframe(self):
        return self._child_interframe

    @abstractmethod
    def _generate_coordinates(self, coordinates):
        """Generate Matrix of the joint's generalized coordinates."""
        pass

    @abstractmethod
    def _generate_speeds(self, speeds):
        """Generate Matrix of the joint's generalized speeds."""
        pass

    @abstractmethod
    def _orient_frames(self):
        """Orient frames as per the joint."""
        pass

    @abstractmethod
    def _set_angular_velocity(self):
        """Set angular velocity of the joint related frames."""
        pass

    @abstractmethod
    def _set_linear_velocity(self):
        """Set velocity of related points to the joint."""
        pass

    @staticmethod
    def _to_vector(matrix, frame):
        """Converts a matrix to a vector in the given frame."""
        return Vector([(matrix, frame)])

    @staticmethod
    def _axis(ax, *frames):
        """Check whether an axis is fixed in one of the frames."""
        if ax is None:
            ax = frames[0].x
            return ax
        if not isinstance(ax, Vector):
            raise TypeError("Axis must be a Vector.")
        ref_frame = None  # Find a body in which the axis can be expressed
        for frame in frames:
            try:
                ax.to_matrix(frame)
            except ValueError:
                pass
            else:
                ref_frame = frame
                break
        if ref_frame is None:
            raise ValueError("Axis cannot be expressed in one of the body's "
                             "frames.")
        if not ax.dt(ref_frame) == 0:
            raise ValueError('Axis cannot be time-varying when viewed from the '
                             'associated body.')
        return ax

    @staticmethod
    def _choose_rotation_axis(frame, axis):
        components = axis.to_matrix(frame)
        x, y, z = components[0], components[1], components[2]

        if x != 0:
            if y != 0:
                if z != 0:
                    return cross(axis, frame.x)
            if z != 0:
                return frame.y
            return frame.z
        else:
            if y != 0:
                return frame.x
            return frame.y

    @staticmethod
    def _create_aligned_interframe(frame, align_axis, frame_axis=None,
                                   frame_name=None):
        """
        Returns an intermediate frame, where the ``frame_axis`` defined in
        ``frame`` is aligned with ``axis``. By default this means that the X
        axis will be aligned with ``axis``.

        Parameters
        ==========

        frame : BodyBase or ReferenceFrame
            The body or reference frame with respect to which the intermediate
            frame is oriented.
        align_axis : Vector
            The vector with respect to which the intermediate frame will be
            aligned.
        frame_axis : Vector
            The vector of the frame which should get aligned with ``axis``. The
            default is the X axis of the frame.
        frame_name : string
            Name of the to be created intermediate frame. The default adds
            "_int_frame" to the name of ``frame``.

        Example
        =======

        An intermediate frame, where the X axis of the parent becomes aligned
        with ``parent.y + parent.z`` can be created as follows:

        >>> from sympy.physics.mechanics.joint import Joint
        >>> from sympy.physics.mechanics import RigidBody
        >>> parent = RigidBody('parent')
        >>> parent_interframe = Joint._create_aligned_interframe(
        ...     parent, parent.y + parent.z)
        >>> parent_interframe
        parent_int_frame
        >>> parent.frame.dcm(parent_interframe)
        Matrix([
        [        0, -sqrt(2)/2, -sqrt(2)/2],
        [sqrt(2)/2,        1/2,       -1/2],
        [sqrt(2)/2,       -1/2,        1/2]])
        >>> (parent.y + parent.z).express(parent_interframe)
        sqrt(2)*parent_int_frame.x

        Notes
        =====

        The direction cosine matrix between the given frame and intermediate
        frame is formed using a simple rotation about an axis that is normal to
        both ``align_axis`` and ``frame_axis``. In general, the normal axis is
        formed by crossing the ``frame_axis`` with the ``align_axis``. The
        exception is if the axes are parallel with opposite directions, in which
        case the rotation vector is chosen using the rules in the following
        table with the vectors expressed in the given frame:

        .. list-table::
           :header-rows: 1

           * - ``align_axis``
             - ``frame_axis``
             - ``rotation_axis``
           * - ``-x``
             - ``x``
             - ``z``
           * - ``-y``
             - ``y``
             - ``x``
           * - ``-z``
             - ``z``
             - ``y``
           * - ``-x-y``
             - ``x+y``
             - ``z``
           * - ``-y-z``
             - ``y+z``
             - ``x``
           * - ``-x-z``
             - ``x+z``
             - ``y``
           * - ``-x-y-z``
             - ``x+y+z``
             - ``(x+y+z) × x``

        """
        if isinstance(frame, BodyBase):
            frame = frame.frame
        if frame_axis is None:
            frame_axis = frame.x
        if frame_name is None:
            if frame.name[-6:] == '_frame':
                frame_name = f'{frame.name[:-6]}_int_frame'
            else:
                frame_name = f'{frame.name}_int_frame'
        angle = frame_axis.angle_between(align_axis)
        rotation_axis = cross(frame_axis, align_axis)
        if rotation_axis == Vector(0) and angle == 0:
            return frame
        if angle == pi:
            rotation_axis = Joint._choose_rotation_axis(frame, align_axis)

        int_frame = ReferenceFrame(frame_name)
        int_frame.orient_axis(frame, rotation_axis, angle)
        int_frame.set_ang_vel(frame, 0 * rotation_axis)
        return int_frame

    def _generate_kdes(self):
        """Generate kinematical differential equations."""
        kdes = []
        t = dynamicsymbols._t
        for i in range(len(self.coordinates)):
            kdes.append(-self.coordinates[i].diff(t) + self.speeds[i])
        return Matrix(kdes)

    def _locate_joint_pos(self, body, joint_pos, body_frame=None):
        """Returns the attachment point of a body."""
        if body_frame is None:
            body_frame = body.frame
        if joint_pos is None:
            return body.masscenter
        if not isinstance(joint_pos, (Point, Vector)):
            raise TypeError('Attachment point must be a Point or Vector.')
        if isinstance(joint_pos, Vector):
            point_name = f'{self.name}_{body.name}_joint'
            joint_pos = body.masscenter.locatenew(point_name, joint_pos)
        if not joint_pos.pos_from(body.masscenter).dt(body_frame) == 0:
            raise ValueError('Attachment point must be fixed to the associated '
                             'body.')
        return joint_pos

    def _locate_joint_frame(self, body, interframe, body_frame=None):
        """Returns the attachment frame of a body."""
        if body_frame is None:
            body_frame = body.frame
        if interframe is None:
            return body_frame
        if isinstance(interframe, Vector):
            interframe = Joint._create_aligned_interframe(
                body_frame, interframe,
                frame_name=f'{self.name}_{body.name}_int_frame')
        elif not isinstance(interframe, ReferenceFrame):
            raise TypeError('Interframe must be a ReferenceFrame.')
        if not interframe.ang_vel_in(body_frame) == 0:
            raise ValueError(f'Interframe {interframe} is not fixed to body '
                             f'{body}.')
        body.masscenter.set_vel(interframe, 0)  # Fixate interframe to body
        return interframe

    def _fill_coordinate_list(self, coordinates, n_coords, label='q', offset=0,
                              number_single=False):
        """Helper method for _generate_coordinates and _generate_speeds.

        Parameters
        ==========

        coordinates : iterable
            Iterable of coordinates or speeds that have been provided.
        n_coords : Integer
            Number of coordinates that should be returned.
        label : String, optional
            Coordinate type either 'q' (coordinates) or 'u' (speeds). The
            Default is 'q'.
        offset : Integer
            Count offset when creating new dynamicsymbols. The default is 0.
        number_single : Boolean
            Boolean whether if n_coords == 1, number should still be used. The
            default is False.

        """

        def create_symbol(number):
            if n_coords == 1 and not number_single:
                return dynamicsymbols(f'{label}_{self.name}')
            return dynamicsymbols(f'{label}{number}_{self.name}')

        name = 'generalized coordinate' if label == 'q' else 'generalized speed'
        generated_coordinates = []
        if coordinates is None:
            coordinates = []
        elif not iterable(coordinates):
            coordinates = [coordinates]
        if not (len(coordinates) == 0 or len(coordinates) == n_coords):
            raise ValueError(f'Expected {n_coords} {name}s, instead got '
                             f'{len(coordinates)} {name}s.')
        # Supports more iterables, also Matrix
        for i, coord in enumerate(coordinates):
            if coord is None:
                generated_coordinates.append(create_symbol(i + offset))
            elif isinstance(coord, (AppliedUndef, Derivative)):
                generated_coordinates.append(coord)
            else:
                raise TypeError(f'The {name} {coord} should have been a '
                                f'dynamicsymbol.')
        for i in range(len(coordinates) + offset, n_coords + offset):
            generated_coordinates.append(create_symbol(i))
        return Matrix(generated_coordinates)


class PinJoint(Joint):
    """Pin (Revolute) Joint.

    .. raw:: html
        :file: ../../../doc/src/modules/physics/mechanics/api/PinJoint.svg

    Explanation
    ===========

    A pin joint is defined such that the joint rotation axis is fixed in both
    the child and parent and the location of the joint is relative to the mass
    center of each body. The child rotates an angle, θ, from the parent about
    the rotation axis and has a simple angular speed, ω, relative to the
    parent. The direction cosine matrix between the child interframe and
    parent interframe is formed using a simple rotation about the joint axis.
    The page on the joints framework gives a more detailed explanation of the
    intermediate frames.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody or Body
        The parent body of joint.
    child : Particle or RigidBody or Body
        The child body of joint.
    coordinates : dynamicsymbol, optional
        Generalized coordinates of the joint.
    speeds : dynamicsymbol, optional
        Generalized speeds of joint.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the parent body which aligns with an axis fixed in the
            child body. The default is the x axis of parent's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    child_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the child body which aligns with an axis fixed in the
            parent body. The default is the x axis of child's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.
    joint_axis : Vector
        The axis about which the rotation occurs. Note that the components
        of this axis are the same in the parent_interframe and child_interframe.
    parent_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by parent_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.
    child_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by child_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody or Body
        The joint's parent body.
    child : Particle or RigidBody or Body
        The joint's child body.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates. The default value is
        ``dynamicsymbols(f'q_{joint.name}')``.
    speeds : Matrix
        Matrix of the joint's generalized speeds. The default value is
        ``dynamicsymbols(f'u_{joint.name}')``.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_axis : Vector
        The axis fixed in the parent frame that represents the joint.
    child_axis : Vector
        The axis fixed in the child frame that represents the joint.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    joint_axis : Vector
        The axis about which the rotation occurs. Note that the components of
        this axis are the same in the parent_interframe and child_interframe.
    kdes : Matrix
        Kinematical differential equations of the joint.

    Examples
    =========

    A single pin joint is created from two bodies and has the following basic
    attributes:

    >>> from sympy.physics.mechanics import RigidBody, PinJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = PinJoint('PC', parent, child)
    >>> joint
    PinJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.parent_axis
    P_frame.x
    >>> joint.child_axis
    C_frame.x
    >>> joint.coordinates
    Matrix([[q_PC(t)]])
    >>> joint.speeds
    Matrix([[u_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame)
    u_PC(t)*P_frame.x
    >>> child.frame.dcm(parent.frame)
    Matrix([
    [1,             0,            0],
    [0,  cos(q_PC(t)), sin(q_PC(t))],
    [0, -sin(q_PC(t)), cos(q_PC(t))]])
    >>> joint.child_point.pos_from(joint.parent_point)
    0

    To further demonstrate the use of the pin joint, the kinematics of simple
    double pendulum that rotates about the Z axis of each connected body can be
    created as follows.

    >>> from sympy import symbols, trigsimp
    >>> from sympy.physics.mechanics import RigidBody, PinJoint
    >>> l1, l2 = symbols('l1 l2')

    First create bodies to represent the fixed ceiling and one to represent
    each pendulum bob.

    >>> ceiling = RigidBody('C')
    >>> upper_bob = RigidBody('U')
    >>> lower_bob = RigidBody('L')

    The first joint will connect the upper bob to the ceiling by a distance of
    ``l1`` and the joint axis will be about the Z axis for each body.

    >>> ceiling_joint = PinJoint('P1', ceiling, upper_bob,
    ... child_point=-l1*upper_bob.frame.x,
    ... joint_axis=ceiling.frame.z)

    The second joint will connect the lower bob to the upper bob by a distance
    of ``l2`` and the joint axis will also be about the Z axis for each body.

    >>> pendulum_joint = PinJoint('P2', upper_bob, lower_bob,
    ... child_point=-l2*lower_bob.frame.x,
    ... joint_axis=upper_bob.frame.z)

    Once the joints are established the kinematics of the connected bodies can
    be accessed. First the direction cosine matrices of pendulum link relative
    to the ceiling are found:

    >>> upper_bob.frame.dcm(ceiling.frame)
    Matrix([
    [ cos(q_P1(t)), sin(q_P1(t)), 0],
    [-sin(q_P1(t)), cos(q_P1(t)), 0],
    [            0,            0, 1]])
    >>> trigsimp(lower_bob.frame.dcm(ceiling.frame))
    Matrix([
    [ cos(q_P1(t) + q_P2(t)), sin(q_P1(t) + q_P2(t)), 0],
    [-sin(q_P1(t) + q_P2(t)), cos(q_P1(t) + q_P2(t)), 0],
    [                      0,                      0, 1]])

    The position of the lower bob's masscenter is found with:

    >>> lower_bob.masscenter.pos_from(ceiling.masscenter)
    l1*U_frame.x + l2*L_frame.x

    The angular velocities of the two pendulum links can be computed with
    respect to the ceiling.

    >>> upper_bob.frame.ang_vel_in(ceiling.frame)
    u_P1(t)*C_frame.z
    >>> lower_bob.frame.ang_vel_in(ceiling.frame)
    u_P1(t)*C_frame.z + u_P2(t)*U_frame.z

    And finally, the linear velocities of the two pendulum bobs can be computed
    with respect to the ceiling.

    >>> upper_bob.masscenter.vel(ceiling.frame)
    l1*u_P1(t)*U_frame.y
    >>> lower_bob.masscenter.vel(ceiling.frame)
    l1*u_P1(t)*U_frame.y + l2*(u_P1(t) + u_P2(t))*L_frame.y

    """

    def __init__(self, name, parent, child, coordinates=None, speeds=None,
                 parent_point=None, child_point=None, parent_interframe=None,
                 child_interframe=None, parent_axis=None, child_axis=None,
                 joint_axis=None, parent_joint_pos=None, child_joint_pos=None):

        self._joint_axis = joint_axis
        super().__init__(name, parent, child, coordinates, speeds, parent_point,
                         child_point, parent_interframe, child_interframe,
                         parent_axis, child_axis, parent_joint_pos,
                         child_joint_pos)

    def __str__(self):
        return (f'PinJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    @property
    def joint_axis(self):
        """Axis about which the child rotates with respect to the parent."""
        return self._joint_axis

    def _generate_coordinates(self, coordinate):
        return self._fill_coordinate_list(coordinate, 1, 'q')

    def _generate_speeds(self, speed):
        return self._fill_coordinate_list(speed, 1, 'u')

    def _orient_frames(self):
        self._joint_axis = self._axis(self.joint_axis, self.parent_interframe)
        self.child_interframe.orient_axis(
            self.parent_interframe, self.joint_axis, self.coordinates[0])

    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(self.parent_interframe, self.speeds[
            0] * self.joint_axis.normalize())

    def _set_linear_velocity(self):
        self.child_point.set_pos(self.parent_point, 0)
        self.parent_point.set_vel(self._parent_frame, 0)
        self.child_point.set_vel(self._child_frame, 0)
        self.child.masscenter.v2pt_theory(self.parent_point,
                                          self._parent_frame, self._child_frame)


class PrismaticJoint(Joint):
    """Prismatic (Sliding) Joint.

    .. image:: PrismaticJoint.svg

    Explanation
    ===========

    It is defined such that the child body translates with respect to the parent
    body along the body-fixed joint axis. The location of the joint is defined
    by two points, one in each body, which coincide when the generalized
    coordinate is zero. The direction cosine matrix between the
    parent_interframe and child_interframe is the identity matrix. Therefore,
    the direction cosine matrix between the parent and child frames is fully
    defined by the definition of the intermediate frames. The page on the joints
    framework gives a more detailed explanation of the intermediate frames.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody or Body
        The parent body of joint.
    child : Particle or RigidBody or Body
        The child body of joint.
    coordinates : dynamicsymbol, optional
        Generalized coordinates of the joint. The default value is
        ``dynamicsymbols(f'q_{joint.name}')``.
    speeds : dynamicsymbol, optional
        Generalized speeds of joint. The default value is
        ``dynamicsymbols(f'u_{joint.name}')``.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the parent body which aligns with an axis fixed in the
            child body. The default is the x axis of parent's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    child_axis : Vector, optional
        .. deprecated:: 1.12
            Axis fixed in the child body which aligns with an axis fixed in the
            parent body. The default is the x axis of child's reference frame.
            For more information on this deprecation, see
            :ref:`deprecated-mechanics-joint-axis`.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.
    joint_axis : Vector
        The axis along which the translation occurs. Note that the components
        of this axis are the same in the parent_interframe and child_interframe.
    parent_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by parent_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.
    child_joint_pos : Point or Vector, optional
        .. deprecated:: 1.12
            This argument is replaced by child_point and will be removed in a
            future version.
            See :ref:`deprecated-mechanics-joint-pos` for more information.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody or Body
        The joint's parent body.
    child : Particle or RigidBody or Body
        The joint's child body.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates.
    speeds : Matrix
        Matrix of the joint's generalized speeds.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_axis : Vector
        The axis fixed in the parent frame that represents the joint.
    child_axis : Vector
        The axis fixed in the child frame that represents the joint.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    kdes : Matrix
        Kinematical differential equations of the joint.

    Examples
    =========

    A single prismatic joint is created from two bodies and has the following
    basic attributes:

    >>> from sympy.physics.mechanics import RigidBody, PrismaticJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = PrismaticJoint('PC', parent, child)
    >>> joint
    PrismaticJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.parent_axis
    P_frame.x
    >>> joint.child_axis
    C_frame.x
    >>> joint.coordinates
    Matrix([[q_PC(t)]])
    >>> joint.speeds
    Matrix([[u_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame)
    0
    >>> child.frame.dcm(parent.frame)
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> joint.child_point.pos_from(joint.parent_point)
    q_PC(t)*P_frame.x

    To further demonstrate the use of the prismatic joint, the kinematics of two
    masses sliding, one moving relative to a fixed body and the other relative
    to the moving body. about the X axis of each connected body can be created
    as follows.

    >>> from sympy.physics.mechanics import PrismaticJoint, RigidBody

    First create bodies to represent the fixed ceiling and one to represent
    a particle.

    >>> wall = RigidBody('W')
    >>> Part1 = RigidBody('P1')
    >>> Part2 = RigidBody('P2')

    The first joint will connect the particle to the ceiling and the
    joint axis will be about the X axis for each body.

    >>> J1 = PrismaticJoint('J1', wall, Part1)

    The second joint will connect the second particle to the first particle
    and the joint axis will also be about the X axis for each body.

    >>> J2 = PrismaticJoint('J2', Part1, Part2)

    Once the joint is established the kinematics of the connected bodies can
    be accessed. First the direction cosine matrices of Part relative
    to the ceiling are found:

    >>> Part1.frame.dcm(wall.frame)
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

    >>> Part2.frame.dcm(wall.frame)
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])

    The position of the particles' masscenter is found with:

    >>> Part1.masscenter.pos_from(wall.masscenter)
    q_J1(t)*W_frame.x

    >>> Part2.masscenter.pos_from(wall.masscenter)
    q_J1(t)*W_frame.x + q_J2(t)*P1_frame.x

    The angular velocities of the two particle links can be computed with
    respect to the ceiling.

    >>> Part1.frame.ang_vel_in(wall.frame)
    0

    >>> Part2.frame.ang_vel_in(wall.frame)
    0

    And finally, the linear velocities of the two particles can be computed
    with respect to the ceiling.

    >>> Part1.masscenter.vel(wall.frame)
    u_J1(t)*W_frame.x

    >>> Part2.masscenter.vel(wall.frame)
    u_J1(t)*W_frame.x + Derivative(q_J2(t), t)*P1_frame.x

    """

    def __init__(self, name, parent, child, coordinates=None, speeds=None,
                 parent_point=None, child_point=None, parent_interframe=None,
                 child_interframe=None, parent_axis=None, child_axis=None,
                 joint_axis=None, parent_joint_pos=None, child_joint_pos=None):

        self._joint_axis = joint_axis
        super().__init__(name, parent, child, coordinates, speeds, parent_point,
                         child_point, parent_interframe, child_interframe,
                         parent_axis, child_axis, parent_joint_pos,
                         child_joint_pos)

    def __str__(self):
        return (f'PrismaticJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    @property
    def joint_axis(self):
        """Axis along which the child translates with respect to the parent."""
        return self._joint_axis

    def _generate_coordinates(self, coordinate):
        return self._fill_coordinate_list(coordinate, 1, 'q')

    def _generate_speeds(self, speed):
        return self._fill_coordinate_list(speed, 1, 'u')

    def _orient_frames(self):
        self._joint_axis = self._axis(self.joint_axis, self.parent_interframe)
        self.child_interframe.orient_axis(
            self.parent_interframe, self.joint_axis, 0)

    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(self.parent_interframe, 0)

    def _set_linear_velocity(self):
        axis = self.joint_axis.normalize()
        self.child_point.set_pos(self.parent_point, self.coordinates[0] * axis)
        self.parent_point.set_vel(self._parent_frame, 0)
        self.child_point.set_vel(self._child_frame, 0)
        self.child_point.set_vel(self._parent_frame, self.speeds[0] * axis)
        self.child.masscenter.set_vel(self._parent_frame, self.speeds[0] * axis)


class CylindricalJoint(Joint):
    """Cylindrical Joint.

    .. image:: CylindricalJoint.svg
        :align: center
        :width: 600

    Explanation
    ===========

    A cylindrical joint is defined such that the child body both rotates about
    and translates along the body-fixed joint axis with respect to the parent
    body. The joint axis is both the rotation axis and translation axis. The
    location of the joint is defined by two points, one in each body, which
    coincide when the generalized coordinate corresponding to the translation is
    zero. The direction cosine matrix between the child interframe and parent
    interframe is formed using a simple rotation about the joint axis. The page
    on the joints framework gives a more detailed explanation of the
    intermediate frames.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody or Body
        The parent body of joint.
    child : Particle or RigidBody or Body
        The child body of joint.
    rotation_coordinate : dynamicsymbol, optional
        Generalized coordinate corresponding to the rotation angle. The default
        value is ``dynamicsymbols(f'q0_{joint.name}')``.
    translation_coordinate : dynamicsymbol, optional
        Generalized coordinate corresponding to the translation distance. The
        default value is ``dynamicsymbols(f'q1_{joint.name}')``.
    rotation_speed : dynamicsymbol, optional
        Generalized speed corresponding to the angular velocity. The default
        value is ``dynamicsymbols(f'u0_{joint.name}')``.
    translation_speed : dynamicsymbol, optional
        Generalized speed corresponding to the translation velocity. The default
        value is ``dynamicsymbols(f'u1_{joint.name}')``.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.
    joint_axis : Vector, optional
        The rotation as well as translation axis. Note that the components of
        this axis are the same in the parent_interframe and child_interframe.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody or Body
        The joint's parent body.
    child : Particle or RigidBody or Body
        The joint's child body.
    rotation_coordinate : dynamicsymbol
        Generalized coordinate corresponding to the rotation angle.
    translation_coordinate : dynamicsymbol
        Generalized coordinate corresponding to the translation distance.
    rotation_speed : dynamicsymbol
        Generalized speed corresponding to the angular velocity.
    translation_speed : dynamicsymbol
        Generalized speed corresponding to the translation velocity.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates.
    speeds : Matrix
        Matrix of the joint's generalized speeds.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    kdes : Matrix
        Kinematical differential equations of the joint.
    joint_axis : Vector
        The axis of rotation and translation.

    Examples
    =========

    A single cylindrical joint is created between two bodies and has the
    following basic attributes:

    >>> from sympy.physics.mechanics import RigidBody, CylindricalJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = CylindricalJoint('PC', parent, child)
    >>> joint
    CylindricalJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.parent_axis
    P_frame.x
    >>> joint.child_axis
    C_frame.x
    >>> joint.coordinates
    Matrix([
    [q0_PC(t)],
    [q1_PC(t)]])
    >>> joint.speeds
    Matrix([
    [u0_PC(t)],
    [u1_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame)
    u0_PC(t)*P_frame.x
    >>> child.frame.dcm(parent.frame)
    Matrix([
    [1,              0,             0],
    [0,  cos(q0_PC(t)), sin(q0_PC(t))],
    [0, -sin(q0_PC(t)), cos(q0_PC(t))]])
    >>> joint.child_point.pos_from(joint.parent_point)
    q1_PC(t)*P_frame.x
    >>> child.masscenter.vel(parent.frame)
    u1_PC(t)*P_frame.x

    To further demonstrate the use of the cylindrical joint, the kinematics of
    two cylindrical joints perpendicular to each other can be created as follows.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import RigidBody, CylindricalJoint
    >>> r, l, w = symbols('r l w')

    First create bodies to represent the fixed floor with a fixed pole on it.
    The second body represents a freely moving tube around that pole. The third
    body represents a solid flag freely translating along and rotating around
    the Y axis of the tube.

    >>> floor = RigidBody('floor')
    >>> tube = RigidBody('tube')
    >>> flag = RigidBody('flag')

    The first joint will connect the first tube to the floor with it translating
    along and rotating around the Z axis of both bodies.

    >>> floor_joint = CylindricalJoint('C1', floor, tube, joint_axis=floor.z)

    The second joint will connect the tube perpendicular to the flag along the Y
    axis of both the tube and the flag, with the joint located at a distance
    ``r`` from the tube's center of mass and a combination of the distances
    ``l`` and ``w`` from the flag's center of mass.

    >>> flag_joint = CylindricalJoint('C2', tube, flag,
    ...                               parent_point=r * tube.y,
    ...                               child_point=-w * flag.y + l * flag.z,
    ...                               joint_axis=tube.y)

    Once the joints are established the kinematics of the connected bodies can
    be accessed. First the direction cosine matrices of both the body and the
    flag relative to the floor are found:

    >>> tube.frame.dcm(floor.frame)
    Matrix([
    [ cos(q0_C1(t)), sin(q0_C1(t)), 0],
    [-sin(q0_C1(t)), cos(q0_C1(t)), 0],
    [             0,             0, 1]])
    >>> flag.frame.dcm(floor.frame)
    Matrix([
    [cos(q0_C1(t))*cos(q0_C2(t)), sin(q0_C1(t))*cos(q0_C2(t)), -sin(q0_C2(t))],
    [             -sin(q0_C1(t)),               cos(q0_C1(t)),              0],
    [sin(q0_C2(t))*cos(q0_C1(t)), sin(q0_C1(t))*sin(q0_C2(t)),  cos(q0_C2(t))]])

    The position of the flag's center of mass is found with:

    >>> flag.masscenter.pos_from(floor.masscenter)
    q1_C1(t)*floor_frame.z + (r + q1_C2(t))*tube_frame.y + w*flag_frame.y - l*flag_frame.z

    The angular velocities of the two tubes can be computed with respect to the
    floor.

    >>> tube.frame.ang_vel_in(floor.frame)
    u0_C1(t)*floor_frame.z
    >>> flag.frame.ang_vel_in(floor.frame)
    u0_C1(t)*floor_frame.z + u0_C2(t)*tube_frame.y

    Finally, the linear velocities of the two tube centers of mass can be
    computed with respect to the floor, while expressed in the tube's frame.

    >>> tube.masscenter.vel(floor.frame).to_matrix(tube.frame)
    Matrix([
    [       0],
    [       0],
    [u1_C1(t)]])
    >>> flag.masscenter.vel(floor.frame).to_matrix(tube.frame).simplify()
    Matrix([
    [-l*u0_C2(t)*cos(q0_C2(t)) - r*u0_C1(t) - w*u0_C1(t) - q1_C2(t)*u0_C1(t)],
    [                    -l*u0_C1(t)*sin(q0_C2(t)) + Derivative(q1_C2(t), t)],
    [                                    l*u0_C2(t)*sin(q0_C2(t)) + u1_C1(t)]])

    """

    def __init__(self, name, parent, child, rotation_coordinate=None,
                 translation_coordinate=None, rotation_speed=None,
                 translation_speed=None, parent_point=None, child_point=None,
                 parent_interframe=None, child_interframe=None,
                 joint_axis=None):
        self._joint_axis = joint_axis
        coordinates = (rotation_coordinate, translation_coordinate)
        speeds = (rotation_speed, translation_speed)
        super().__init__(name, parent, child, coordinates, speeds,
                         parent_point, child_point,
                         parent_interframe=parent_interframe,
                         child_interframe=child_interframe)

    def __str__(self):
        return (f'CylindricalJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    @property
    def joint_axis(self):
        """Axis about and along which the rotation and translation occurs."""
        return self._joint_axis

    @property
    def rotation_coordinate(self):
        """Generalized coordinate corresponding to the rotation angle."""
        return self.coordinates[0]

    @property
    def translation_coordinate(self):
        """Generalized coordinate corresponding to the translation distance."""
        return self.coordinates[1]

    @property
    def rotation_speed(self):
        """Generalized speed corresponding to the angular velocity."""
        return self.speeds[0]

    @property
    def translation_speed(self):
        """Generalized speed corresponding to the translation velocity."""
        return self.speeds[1]

    def _generate_coordinates(self, coordinates):
        return self._fill_coordinate_list(coordinates, 2, 'q')

    def _generate_speeds(self, speeds):
        return self._fill_coordinate_list(speeds, 2, 'u')

    def _orient_frames(self):
        self._joint_axis = self._axis(self.joint_axis, self.parent_interframe)
        self.child_interframe.orient_axis(
            self.parent_interframe, self.joint_axis, self.rotation_coordinate)

    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(
            self.parent_interframe,
            self.rotation_speed * self.joint_axis.normalize())

    def _set_linear_velocity(self):
        self.child_point.set_pos(
            self.parent_point,
            self.translation_coordinate * self.joint_axis.normalize())
        self.parent_point.set_vel(self._parent_frame, 0)
        self.child_point.set_vel(self._child_frame, 0)
        self.child_point.set_vel(
            self._parent_frame,
            self.translation_speed * self.joint_axis.normalize())
        self.child.masscenter.v2pt_theory(self.child_point, self._parent_frame,
                                          self.child_interframe)


class PlanarJoint(Joint):
    """Planar Joint.

    .. raw:: html
        :file: ../../../doc/src/modules/physics/mechanics/api/PlanarJoint.svg

    Explanation
    ===========

    A planar joint is defined such that the child body translates over a fixed
    plane of the parent body as well as rotate about the rotation axis, which
    is perpendicular to that plane. The origin of this plane is the
    ``parent_point`` and the plane is spanned by two nonparallel planar vectors.
    The location of the ``child_point`` is based on the planar vectors
    ($\\vec{v}_1$, $\\vec{v}_2$) and generalized coordinates ($q_1$, $q_2$),
    i.e. $\\vec{r} = q_1 \\hat{v}_1 + q_2 \\hat{v}_2$. The direction cosine
    matrix between the ``child_interframe`` and ``parent_interframe`` is formed
    using a simple rotation ($q_0$) about the rotation axis.

    In order to simplify the definition of the ``PlanarJoint``, the
    ``rotation_axis`` and ``planar_vectors`` are set to be the unit vectors of
    the ``parent_interframe`` according to the table below. This ensures that
    you can only define these vectors by creating a separate frame and supplying
    that as the interframe. If you however would only like to supply the normals
    of the plane with respect to the parent and child bodies, then you can also
    supply those to the ``parent_interframe`` and ``child_interframe``
    arguments. An example of both of these cases is in the examples section
    below and the page on the joints framework provides a more detailed
    explanation of the intermediate frames.

    .. list-table::

        * - ``rotation_axis``
          - ``parent_interframe.x``
        * - ``planar_vectors[0]``
          - ``parent_interframe.y``
        * - ``planar_vectors[1]``
          - ``parent_interframe.z``

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody or Body
        The parent body of joint.
    child : Particle or RigidBody or Body
        The child body of joint.
    rotation_coordinate : dynamicsymbol, optional
        Generalized coordinate corresponding to the rotation angle. The default
        value is ``dynamicsymbols(f'q0_{joint.name}')``.
    planar_coordinates : iterable of dynamicsymbols, optional
        Two generalized coordinates used for the planar translation. The default
        value is ``dynamicsymbols(f'q1_{joint.name} q2_{joint.name}')``.
    rotation_speed : dynamicsymbol, optional
        Generalized speed corresponding to the angular velocity. The default
        value is ``dynamicsymbols(f'u0_{joint.name}')``.
    planar_speeds : dynamicsymbols, optional
        Two generalized speeds used for the planar translation velocity. The
        default value is ``dynamicsymbols(f'u1_{joint.name} u2_{joint.name}')``.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody or Body
        The joint's parent body.
    child : Particle or RigidBody or Body
        The joint's child body.
    rotation_coordinate : dynamicsymbol
        Generalized coordinate corresponding to the rotation angle.
    planar_coordinates : Matrix
        Two generalized coordinates used for the planar translation.
    rotation_speed : dynamicsymbol
        Generalized speed corresponding to the angular velocity.
    planar_speeds : Matrix
        Two generalized speeds used for the planar translation velocity.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates.
    speeds : Matrix
        Matrix of the joint's generalized speeds.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    kdes : Matrix
        Kinematical differential equations of the joint.
    rotation_axis : Vector
        The axis about which the rotation occurs.
    planar_vectors : list
        The vectors that describe the planar translation directions.

    Examples
    =========

    A single planar joint is created between two bodies and has the following
    basic attributes:

    >>> from sympy.physics.mechanics import RigidBody, PlanarJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = PlanarJoint('PC', parent, child)
    >>> joint
    PlanarJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.rotation_axis
    P_frame.x
    >>> joint.planar_vectors
    [P_frame.y, P_frame.z]
    >>> joint.rotation_coordinate
    q0_PC(t)
    >>> joint.planar_coordinates
    Matrix([
    [q1_PC(t)],
    [q2_PC(t)]])
    >>> joint.coordinates
    Matrix([
    [q0_PC(t)],
    [q1_PC(t)],
    [q2_PC(t)]])
    >>> joint.rotation_speed
    u0_PC(t)
    >>> joint.planar_speeds
    Matrix([
    [u1_PC(t)],
    [u2_PC(t)]])
    >>> joint.speeds
    Matrix([
    [u0_PC(t)],
    [u1_PC(t)],
    [u2_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame)
    u0_PC(t)*P_frame.x
    >>> child.frame.dcm(parent.frame)
    Matrix([
    [1,              0,             0],
    [0,  cos(q0_PC(t)), sin(q0_PC(t))],
    [0, -sin(q0_PC(t)), cos(q0_PC(t))]])
    >>> joint.child_point.pos_from(joint.parent_point)
    q1_PC(t)*P_frame.y + q2_PC(t)*P_frame.z
    >>> child.masscenter.vel(parent.frame)
    u1_PC(t)*P_frame.y + u2_PC(t)*P_frame.z

    To further demonstrate the use of the planar joint, the kinematics of a
    block sliding on a slope, can be created as follows.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import PlanarJoint, RigidBody, ReferenceFrame
    >>> a, d, h = symbols('a d h')

    First create bodies to represent the slope and the block.

    >>> ground = RigidBody('G')
    >>> block = RigidBody('B')

    To define the slope you can either define the plane by specifying the
    ``planar_vectors`` or/and the ``rotation_axis``. However it is advisable to
    create a rotated intermediate frame, so that the ``parent_vectors`` and
    ``rotation_axis`` will be the unit vectors of this intermediate frame.

    >>> slope = ReferenceFrame('A')
    >>> slope.orient_axis(ground.frame, ground.y, a)

    The planar joint can be created using these bodies and intermediate frame.
    We can specify the origin of the slope to be ``d`` above the slope's center
    of mass and the block's center of mass to be a distance ``h`` above the
    slope's surface. Note that we can specify the normal of the plane using the
    rotation axis argument.

    >>> joint = PlanarJoint('PC', ground, block, parent_point=d * ground.x,
    ...                     child_point=-h * block.x, parent_interframe=slope)

    Once the joint is established the kinematics of the bodies can be accessed.
    First the ``rotation_axis``, which is normal to the plane and the
    ``plane_vectors``, can be found.

    >>> joint.rotation_axis
    A.x
    >>> joint.planar_vectors
    [A.y, A.z]

    The direction cosine matrix of the block with respect to the ground can be
    found with:

    >>> block.frame.dcm(ground.frame)
    Matrix([
    [              cos(a),              0,              -sin(a)],
    [sin(a)*sin(q0_PC(t)),  cos(q0_PC(t)), sin(q0_PC(t))*cos(a)],
    [sin(a)*cos(q0_PC(t)), -sin(q0_PC(t)), cos(a)*cos(q0_PC(t))]])

    The angular velocity of the block can be computed with respect to the
    ground.

    >>> block.frame.ang_vel_in(ground.frame)
    u0_PC(t)*A.x

    The position of the block's center of mass can be found with:

    >>> block.masscenter.pos_from(ground.masscenter)
    d*G_frame.x + h*B_frame.x + q1_PC(t)*A.y + q2_PC(t)*A.z

    Finally, the linear velocity of the block's center of mass can be
    computed with respect to the ground.

    >>> block.masscenter.vel(ground.frame)
    u1_PC(t)*A.y + u2_PC(t)*A.z

    In some cases it could be your preference to only define the normals of the
    plane with respect to both bodies. This can most easily be done by supplying
    vectors to the ``interframe`` arguments. What will happen in this case is
    that an interframe will be created with its ``x`` axis aligned with the
    provided vector. For a further explanation of how this is done see the notes
    of the ``Joint`` class. In the code below, the above example (with the block
    on the slope) is recreated by supplying vectors to the interframe arguments.
    Note that the previously described option is however more computationally
    efficient, because the algorithm now has to compute the rotation angle
    between the provided vector and the 'x' axis.

    >>> from sympy import symbols, cos, sin
    >>> from sympy.physics.mechanics import PlanarJoint, RigidBody
    >>> a, d, h = symbols('a d h')
    >>> ground = RigidBody('G')
    >>> block = RigidBody('B')
    >>> joint = PlanarJoint(
    ...     'PC', ground, block, parent_point=d * ground.x,
    ...     child_point=-h * block.x, child_interframe=block.x,
    ...     parent_interframe=cos(a) * ground.x + sin(a) * ground.z)
    >>> block.frame.dcm(ground.frame).simplify()
    Matrix([
    [               cos(a),              0,               sin(a)],
    [-sin(a)*sin(q0_PC(t)),  cos(q0_PC(t)), sin(q0_PC(t))*cos(a)],
    [-sin(a)*cos(q0_PC(t)), -sin(q0_PC(t)), cos(a)*cos(q0_PC(t))]])

    """

    def __init__(self, name, parent, child, rotation_coordinate=None,
                 planar_coordinates=None, rotation_speed=None,
                 planar_speeds=None, parent_point=None, child_point=None,
                 parent_interframe=None, child_interframe=None):
        # A ready to merge implementation of setting the planar_vectors and
        # rotation_axis was added and removed in PR #24046
        coordinates = (rotation_coordinate, planar_coordinates)
        speeds = (rotation_speed, planar_speeds)
        super().__init__(name, parent, child, coordinates, speeds,
                         parent_point, child_point,
                         parent_interframe=parent_interframe,
                         child_interframe=child_interframe)

    def __str__(self):
        return (f'PlanarJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    @property
    def rotation_coordinate(self):
        """Generalized coordinate corresponding to the rotation angle."""
        return self.coordinates[0]

    @property
    def planar_coordinates(self):
        """Two generalized coordinates used for the planar translation."""
        return self.coordinates[1:, 0]

    @property
    def rotation_speed(self):
        """Generalized speed corresponding to the angular velocity."""
        return self.speeds[0]

    @property
    def planar_speeds(self):
        """Two generalized speeds used for the planar translation velocity."""
        return self.speeds[1:, 0]

    @property
    def rotation_axis(self):
        """The axis about which the rotation occurs."""
        return self.parent_interframe.x

    @property
    def planar_vectors(self):
        """The vectors that describe the planar translation directions."""
        return [self.parent_interframe.y, self.parent_interframe.z]

    def _generate_coordinates(self, coordinates):
        rotation_speed = self._fill_coordinate_list(coordinates[0], 1, 'q',
                                                    number_single=True)
        planar_speeds = self._fill_coordinate_list(coordinates[1], 2, 'q', 1)
        return rotation_speed.col_join(planar_speeds)

    def _generate_speeds(self, speeds):
        rotation_speed = self._fill_coordinate_list(speeds[0], 1, 'u',
                                                    number_single=True)
        planar_speeds = self._fill_coordinate_list(speeds[1], 2, 'u', 1)
        return rotation_speed.col_join(planar_speeds)

    def _orient_frames(self):
        self.child_interframe.orient_axis(
            self.parent_interframe, self.rotation_axis,
            self.rotation_coordinate)

    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(
            self.parent_interframe,
            self.rotation_speed * self.rotation_axis)

    def _set_linear_velocity(self):
        self.child_point.set_pos(
            self.parent_point,
            self.planar_coordinates[0] * self.planar_vectors[0] +
            self.planar_coordinates[1] * self.planar_vectors[1])
        self.parent_point.set_vel(self.parent_interframe, 0)
        self.child_point.set_vel(self.child_interframe, 0)
        self.child_point.set_vel(
            self._parent_frame, self.planar_speeds[0] * self.planar_vectors[0] +
            self.planar_speeds[1] * self.planar_vectors[1])
        self.child.masscenter.v2pt_theory(self.child_point, self._parent_frame,
                                          self._child_frame)


class SphericalJoint(Joint):
    """Spherical (Ball-and-Socket) Joint.

    .. image:: SphericalJoint.svg
        :align: center
        :width: 600

    Explanation
    ===========

    A spherical joint is defined such that the child body is free to rotate in
    any direction, without allowing a translation of the ``child_point``. As can
    also be seen in the image, the ``parent_point`` and ``child_point`` are
    fixed on top of each other, i.e. the ``joint_point``. This rotation is
    defined using the :func:`parent_interframe.orient(child_interframe,
    rot_type, amounts, rot_order)
    <sympy.physics.vector.frame.ReferenceFrame.orient>` method. The default
    rotation consists of three relative rotations, i.e. body-fixed rotations.
    Based on the direction cosine matrix following from these rotations, the
    angular velocity is computed based on the generalized coordinates and
    generalized speeds.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody or Body
        The parent body of joint.
    child : Particle or RigidBody or Body
        The child body of joint.
    coordinates: iterable of dynamicsymbols, optional
        Generalized coordinates of the joint.
    speeds : iterable of dynamicsymbols, optional
        Generalized speeds of joint.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.
    rot_type : str, optional
        The method used to generate the direction cosine matrix. Supported
        methods are:

        - ``'Body'``: three successive rotations about new intermediate axes,
          also called "Euler and Tait-Bryan angles"
        - ``'Space'``: three successive rotations about the parent frames' unit
          vectors

        The default method is ``'Body'``.
    amounts :
        Expressions defining the rotation angles or direction cosine matrix.
        These must match the ``rot_type``. See examples below for details. The
        input types are:

        - ``'Body'``: 3-tuple of expressions, symbols, or functions
        - ``'Space'``: 3-tuple of expressions, symbols, or functions

        The default amounts are the given ``coordinates``.
    rot_order : str or int, optional
        If applicable, the order of the successive of rotations. The string
        ``'123'`` and integer ``123`` are equivalent, for example. Required for
        ``'Body'`` and ``'Space'``. The default value is ``123``.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody or Body
        The joint's parent body.
    child : Particle or RigidBody or Body
        The joint's child body.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates.
    speeds : Matrix
        Matrix of the joint's generalized speeds.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    kdes : Matrix
        Kinematical differential equations of the joint.

    Examples
    =========

    A single spherical joint is created from two bodies and has the following
    basic attributes:

    >>> from sympy.physics.mechanics import RigidBody, SphericalJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = SphericalJoint('PC', parent, child)
    >>> joint
    SphericalJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.parent_interframe
    P_frame
    >>> joint.child_interframe
    C_frame
    >>> joint.coordinates
    Matrix([
    [q0_PC(t)],
    [q1_PC(t)],
    [q2_PC(t)]])
    >>> joint.speeds
    Matrix([
    [u0_PC(t)],
    [u1_PC(t)],
    [u2_PC(t)]])
    >>> child.frame.ang_vel_in(parent.frame).to_matrix(child.frame)
    Matrix([
    [ u0_PC(t)*cos(q1_PC(t))*cos(q2_PC(t)) + u1_PC(t)*sin(q2_PC(t))],
    [-u0_PC(t)*sin(q2_PC(t))*cos(q1_PC(t)) + u1_PC(t)*cos(q2_PC(t))],
    [                             u0_PC(t)*sin(q1_PC(t)) + u2_PC(t)]])
    >>> child.frame.x.to_matrix(parent.frame)
    Matrix([
    [                                            cos(q1_PC(t))*cos(q2_PC(t))],
    [sin(q0_PC(t))*sin(q1_PC(t))*cos(q2_PC(t)) + sin(q2_PC(t))*cos(q0_PC(t))],
    [sin(q0_PC(t))*sin(q2_PC(t)) - sin(q1_PC(t))*cos(q0_PC(t))*cos(q2_PC(t))]])
    >>> joint.child_point.pos_from(joint.parent_point)
    0

    To further demonstrate the use of the spherical joint, the kinematics of a
    spherical joint with a ZXZ rotation can be created as follows.

    >>> from sympy import symbols
    >>> from sympy.physics.mechanics import RigidBody, SphericalJoint
    >>> l1 = symbols('l1')

    First create bodies to represent the fixed floor and a pendulum bob.

    >>> floor = RigidBody('F')
    >>> bob = RigidBody('B')

    The joint will connect the bob to the floor, with the joint located at a
    distance of ``l1`` from the child's center of mass and the rotation set to a
    body-fixed ZXZ rotation.

    >>> joint = SphericalJoint('S', floor, bob, child_point=l1 * bob.y,
    ...                        rot_type='body', rot_order='ZXZ')

    Now that the joint is established, the kinematics of the connected body can
    be accessed.

    The position of the bob's masscenter is found with:

    >>> bob.masscenter.pos_from(floor.masscenter)
    - l1*B_frame.y

    The angular velocities of the pendulum link can be computed with respect to
    the floor.

    >>> bob.frame.ang_vel_in(floor.frame).to_matrix(
    ...     floor.frame).simplify()
    Matrix([
    [u1_S(t)*cos(q0_S(t)) + u2_S(t)*sin(q0_S(t))*sin(q1_S(t))],
    [u1_S(t)*sin(q0_S(t)) - u2_S(t)*sin(q1_S(t))*cos(q0_S(t))],
    [                          u0_S(t) + u2_S(t)*cos(q1_S(t))]])

    Finally, the linear velocity of the bob's center of mass can be computed.

    >>> bob.masscenter.vel(floor.frame).to_matrix(bob.frame)
    Matrix([
    [                           l1*(u0_S(t)*cos(q1_S(t)) + u2_S(t))],
    [                                                             0],
    [-l1*(u0_S(t)*sin(q1_S(t))*sin(q2_S(t)) + u1_S(t)*cos(q2_S(t)))]])

    """
    def __init__(self, name, parent, child, coordinates=None, speeds=None,
                 parent_point=None, child_point=None, parent_interframe=None,
                 child_interframe=None, rot_type='BODY', amounts=None,
                 rot_order=123):
        self._rot_type = rot_type
        self._amounts = amounts
        self._rot_order = rot_order
        super().__init__(name, parent, child, coordinates, speeds,
                         parent_point, child_point,
                         parent_interframe=parent_interframe,
                         child_interframe=child_interframe)

    def __str__(self):
        return (f'SphericalJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    def _generate_coordinates(self, coordinates):
        return self._fill_coordinate_list(coordinates, 3, 'q')

    def _generate_speeds(self, speeds):
        return self._fill_coordinate_list(speeds, len(self.coordinates), 'u')

    def _orient_frames(self):
        supported_rot_types = ('BODY', 'SPACE')
        if self._rot_type.upper() not in supported_rot_types:
            raise NotImplementedError(
                f'Rotation type "{self._rot_type}" is not implemented. '
                f'Implemented rotation types are: {supported_rot_types}')
        amounts = self.coordinates if self._amounts is None else self._amounts
        self.child_interframe.orient(self.parent_interframe, self._rot_type,
                                     amounts, self._rot_order)

    def _set_angular_velocity(self):
        t = dynamicsymbols._t
        vel = self.child_interframe.ang_vel_in(self.parent_interframe).xreplace(
            {q.diff(t): u for q, u in zip(self.coordinates, self.speeds)}
        )
        self.child_interframe.set_ang_vel(self.parent_interframe, vel)

    def _set_linear_velocity(self):
        self.child_point.set_pos(self.parent_point, 0)
        self.parent_point.set_vel(self._parent_frame, 0)
        self.child_point.set_vel(self._child_frame, 0)
        self.child.masscenter.v2pt_theory(self.parent_point, self._parent_frame,
                                          self._child_frame)


class WeldJoint(Joint):
    """Weld Joint.

    .. raw:: html
        :file: ../../../doc/src/modules/physics/mechanics/api/WeldJoint.svg

    Explanation
    ===========

    A weld joint is defined such that there is no relative motion between the
    child and parent bodies. The direction cosine matrix between the attachment
    frame (``parent_interframe`` and ``child_interframe``) is the identity
    matrix and the attachment points (``parent_point`` and ``child_point``) are
    coincident. The page on the joints framework gives a more detailed
    explanation of the intermediate frames.

    Parameters
    ==========

    name : string
        A unique name for the joint.
    parent : Particle or RigidBody or Body
        The parent body of joint.
    child : Particle or RigidBody or Body
        The child body of joint.
    parent_point : Point or Vector, optional
        Attachment point where the joint is fixed to the parent body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the parent's mass
        center.
    child_point : Point or Vector, optional
        Attachment point where the joint is fixed to the child body. If a
        vector is provided, then the attachment point is computed by adding the
        vector to the body's mass center. The default value is the child's mass
        center.
    parent_interframe : ReferenceFrame, optional
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the parent's own frame.
    child_interframe : ReferenceFrame, optional
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated. If a Vector is provided then an interframe
        is created which aligns its X axis with the given vector. The default
        value is the child's own frame.

    Attributes
    ==========

    name : string
        The joint's name.
    parent : Particle or RigidBody or Body
        The joint's parent body.
    child : Particle or RigidBody or Body
        The joint's child body.
    coordinates : Matrix
        Matrix of the joint's generalized coordinates. The default value is
        ``dynamicsymbols(f'q_{joint.name}')``.
    speeds : Matrix
        Matrix of the joint's generalized speeds. The default value is
        ``dynamicsymbols(f'u_{joint.name}')``.
    parent_point : Point
        Attachment point where the joint is fixed to the parent body.
    child_point : Point
        Attachment point where the joint is fixed to the child body.
    parent_interframe : ReferenceFrame
        Intermediate frame of the parent body with respect to which the joint
        transformation is formulated.
    child_interframe : ReferenceFrame
        Intermediate frame of the child body with respect to which the joint
        transformation is formulated.
    kdes : Matrix
        Kinematical differential equations of the joint.

    Examples
    =========

    A single weld joint is created from two bodies and has the following basic
    attributes:

    >>> from sympy.physics.mechanics import RigidBody, WeldJoint
    >>> parent = RigidBody('P')
    >>> parent
    P
    >>> child = RigidBody('C')
    >>> child
    C
    >>> joint = WeldJoint('PC', parent, child)
    >>> joint
    WeldJoint: PC  parent: P  child: C
    >>> joint.name
    'PC'
    >>> joint.parent
    P
    >>> joint.child
    C
    >>> joint.parent_point
    P_masscenter
    >>> joint.child_point
    C_masscenter
    >>> joint.coordinates
    Matrix(0, 0, [])
    >>> joint.speeds
    Matrix(0, 0, [])
    >>> child.frame.ang_vel_in(parent.frame)
    0
    >>> child.frame.dcm(parent.frame)
    Matrix([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]])
    >>> joint.child_point.pos_from(joint.parent_point)
    0

    To further demonstrate the use of the weld joint, two relatively-fixed
    bodies rotated by a quarter turn about the Y axis can be created as follows:

    >>> from sympy import symbols, pi
    >>> from sympy.physics.mechanics import ReferenceFrame, RigidBody, WeldJoint
    >>> l1, l2 = symbols('l1 l2')

    First create the bodies to represent the parent and rotated child body.

    >>> parent = RigidBody('P')
    >>> child = RigidBody('C')

    Next the intermediate frame specifying the fixed rotation with respect to
    the parent can be created.

    >>> rotated_frame = ReferenceFrame('Pr')
    >>> rotated_frame.orient_axis(parent.frame, parent.y, pi / 2)

    The weld between the parent body and child body is located at a distance
    ``l1`` from the parent's center of mass in the X direction and ``l2`` from
    the child's center of mass in the child's negative X direction.

    >>> weld = WeldJoint('weld', parent, child, parent_point=l1 * parent.x,
    ...                  child_point=-l2 * child.x,
    ...                  parent_interframe=rotated_frame)

    Now that the joint has been established, the kinematics of the bodies can be
    accessed. The direction cosine matrix of the child body with respect to the
    parent can be found:

    >>> child.frame.dcm(parent.frame)
    Matrix([
    [0, 0, -1],
    [0, 1,  0],
    [1, 0,  0]])

    As can also been seen from the direction cosine matrix, the parent X axis is
    aligned with the child's Z axis:
    >>> parent.x == child.z
    True

    The position of the child's center of mass with respect to the parent's
    center of mass can be found with:

    >>> child.masscenter.pos_from(parent.masscenter)
    l1*P_frame.x + l2*C_frame.x

    The angular velocity of the child with respect to the parent is 0 as one
    would expect.

    >>> child.frame.ang_vel_in(parent.frame)
    0

    """

    def __init__(self, name, parent, child, parent_point=None, child_point=None,
                 parent_interframe=None, child_interframe=None):
        super().__init__(name, parent, child, [], [], parent_point,
                         child_point, parent_interframe=parent_interframe,
                         child_interframe=child_interframe)
        self._kdes = Matrix(1, 0, []).T  # Removes stackability problems #10770

    def __str__(self):
        return (f'WeldJoint: {self.name}  parent: {self.parent}  '
                f'child: {self.child}')

    def _generate_coordinates(self, coordinate):
        return Matrix()

    def _generate_speeds(self, speed):
        return Matrix()

    def _orient_frames(self):
        self.child_interframe.orient_axis(self.parent_interframe,
                                          self.parent_interframe.x, 0)

    def _set_angular_velocity(self):
        self.child_interframe.set_ang_vel(self.parent_interframe, 0)

    def _set_linear_velocity(self):
        self.child_point.set_pos(self.parent_point, 0)
        self.parent_point.set_vel(self._parent_frame, 0)
        self.child_point.set_vel(self._child_frame, 0)
        self.child.masscenter.set_vel(self._parent_frame, 0)
