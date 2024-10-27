"""Tests for the ``sympy.physics.mechanics.actuator.py`` module."""

import pytest

from sympy import (
    S,
    Matrix,
    Symbol,
    SympifyError,
    sqrt,
    Abs
)
from sympy.physics.mechanics import (
    ActuatorBase,
    Force,
    ForceActuator,
    KanesMethod,
    LinearDamper,
    LinearPathway,
    LinearSpring,
    Particle,
    PinJoint,
    Point,
    ReferenceFrame,
    RigidBody,
    TorqueActuator,
    Vector,
    dynamicsymbols,
    DuffingSpring,
)

from sympy.core.expr import Expr as ExprType

target = RigidBody('target')
reaction = RigidBody('reaction')


class TestForceActuator:

    @pytest.fixture(autouse=True)
    def _linear_pathway_fixture(self):
        self.force = Symbol('F')
        self.pA = Point('pA')
        self.pB = Point('pB')
        self.pathway = LinearPathway(self.pA, self.pB)
        self.q1 = dynamicsymbols('q1')
        self.q2 = dynamicsymbols('q2')
        self.q3 = dynamicsymbols('q3')
        self.q1d = dynamicsymbols('q1', 1)
        self.q2d = dynamicsymbols('q2', 1)
        self.q3d = dynamicsymbols('q3', 1)
        self.N = ReferenceFrame('N')

    def test_is_actuator_base_subclass(self):
        assert issubclass(ForceActuator, ActuatorBase)

    @pytest.mark.parametrize(
        'force, expected_force',
        [
            (1, S.One),
            (S.One, S.One),
            (Symbol('F'), Symbol('F')),
            (dynamicsymbols('F'), dynamicsymbols('F')),
            (Symbol('F')**2 + Symbol('F'), Symbol('F')**2 + Symbol('F')),
        ]
    )
    def test_valid_constructor_force(self, force, expected_force):
        instance = ForceActuator(force, self.pathway)
        assert isinstance(instance, ForceActuator)
        assert hasattr(instance, 'force')
        assert isinstance(instance.force, ExprType)
        assert instance.force == expected_force

    @pytest.mark.parametrize('force', [None, 'F'])
    def test_invalid_constructor_force_not_sympifyable(self, force):
        with pytest.raises(SympifyError):
            _ = ForceActuator(force, self.pathway)

    @pytest.mark.parametrize(
        'pathway',
        [
            LinearPathway(Point('pA'), Point('pB')),
        ]
    )
    def test_valid_constructor_pathway(self, pathway):
        instance = ForceActuator(self.force, pathway)
        assert isinstance(instance, ForceActuator)
        assert hasattr(instance, 'pathway')
        assert isinstance(instance.pathway, LinearPathway)
        assert instance.pathway == pathway

    def test_invalid_constructor_pathway_not_pathway_base(self):
        with pytest.raises(TypeError):
            _ = ForceActuator(self.force, None)

    @pytest.mark.parametrize(
        'property_name, fixture_attr_name',
        [
            ('force', 'force'),
            ('pathway', 'pathway'),
        ]
    )
    def test_properties_are_immutable(self, property_name, fixture_attr_name):
        instance = ForceActuator(self.force, self.pathway)
        value = getattr(self, fixture_attr_name)
        with pytest.raises(AttributeError):
            setattr(instance, property_name, value)

    def test_repr(self):
        actuator = ForceActuator(self.force, self.pathway)
        expected = "ForceActuator(F, LinearPathway(pA, pB))"
        assert repr(actuator) == expected

    def test_to_loads_static_pathway(self):
        self.pB.set_pos(self.pA, 2*self.N.x)
        actuator = ForceActuator(self.force, self.pathway)
        expected = [
            (self.pA, - self.force*self.N.x),
            (self.pB, self.force*self.N.x),
        ]
        assert actuator.to_loads() == expected

    def test_to_loads_2D_pathway(self):
        self.pB.set_pos(self.pA, 2*self.q1*self.N.x)
        actuator = ForceActuator(self.force, self.pathway)
        expected = [
            (self.pA, - self.force*(self.q1/sqrt(self.q1**2))*self.N.x),
            (self.pB, self.force*(self.q1/sqrt(self.q1**2))*self.N.x),
        ]
        assert actuator.to_loads() == expected

    def test_to_loads_3D_pathway(self):
        self.pB.set_pos(
            self.pA,
            self.q1*self.N.x - self.q2*self.N.y + 2*self.q3*self.N.z,
        )
        actuator = ForceActuator(self.force, self.pathway)
        length = sqrt(self.q1**2 + self.q2**2 + 4*self.q3**2)
        pO_force = (
            - self.force*self.q1*self.N.x/length
            + self.force*self.q2*self.N.y/length
            - 2*self.force*self.q3*self.N.z/length
        )
        pI_force = (
            self.force*self.q1*self.N.x/length
            - self.force*self.q2*self.N.y/length
            + 2*self.force*self.q3*self.N.z/length
        )
        expected = [
            (self.pA, pO_force),
            (self.pB, pI_force),
        ]
        assert actuator.to_loads() == expected


class TestLinearSpring:

    @pytest.fixture(autouse=True)
    def _linear_spring_fixture(self):
        self.stiffness = Symbol('k')
        self.l = Symbol('l')
        self.pA = Point('pA')
        self.pB = Point('pB')
        self.pathway = LinearPathway(self.pA, self.pB)
        self.q = dynamicsymbols('q')
        self.N = ReferenceFrame('N')

    def test_is_force_actuator_subclass(self):
        assert issubclass(LinearSpring, ForceActuator)

    def test_is_actuator_base_subclass(self):
        assert issubclass(LinearSpring, ActuatorBase)

    @pytest.mark.parametrize(
        (
            'stiffness, '
            'expected_stiffness, '
            'equilibrium_length, '
            'expected_equilibrium_length, '
            'force'
        ),
        [
            (
                1,
                S.One,
                0,
                S.Zero,
                -sqrt(dynamicsymbols('q')**2),
            ),
            (
                Symbol('k'),
                Symbol('k'),
                0,
                S.Zero,
                -Symbol('k')*sqrt(dynamicsymbols('q')**2),
            ),
            (
                Symbol('k'),
                Symbol('k'),
                S.Zero,
                S.Zero,
                -Symbol('k')*sqrt(dynamicsymbols('q')**2),
            ),
            (
                Symbol('k'),
                Symbol('k'),
                Symbol('l'),
                Symbol('l'),
                -Symbol('k')*(sqrt(dynamicsymbols('q')**2) - Symbol('l')),
            ),
        ]
    )
    def test_valid_constructor(
        self,
        stiffness,
        expected_stiffness,
        equilibrium_length,
        expected_equilibrium_length,
        force,
    ):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        spring = LinearSpring(stiffness, self.pathway, equilibrium_length)

        assert isinstance(spring, LinearSpring)

        assert hasattr(spring, 'stiffness')
        assert isinstance(spring.stiffness, ExprType)
        assert spring.stiffness == expected_stiffness

        assert hasattr(spring, 'pathway')
        assert isinstance(spring.pathway, LinearPathway)
        assert spring.pathway == self.pathway

        assert hasattr(spring, 'equilibrium_length')
        assert isinstance(spring.equilibrium_length, ExprType)
        assert spring.equilibrium_length == expected_equilibrium_length

        assert hasattr(spring, 'force')
        assert isinstance(spring.force, ExprType)
        assert spring.force == force

    @pytest.mark.parametrize('stiffness', [None, 'k'])
    def test_invalid_constructor_stiffness_not_sympifyable(self, stiffness):
        with pytest.raises(SympifyError):
            _ = LinearSpring(stiffness, self.pathway, self.l)

    def test_invalid_constructor_pathway_not_pathway_base(self):
        with pytest.raises(TypeError):
            _ = LinearSpring(self.stiffness, None, self.l)

    @pytest.mark.parametrize('equilibrium_length', [None, 'l'])
    def test_invalid_constructor_equilibrium_length_not_sympifyable(
        self,
        equilibrium_length,
    ):
        with pytest.raises(SympifyError):
            _ = LinearSpring(self.stiffness, self.pathway, equilibrium_length)

    @pytest.mark.parametrize(
        'property_name, fixture_attr_name',
        [
            ('stiffness', 'stiffness'),
            ('pathway', 'pathway'),
            ('equilibrium_length', 'l'),
        ]
    )
    def test_properties_are_immutable(self, property_name, fixture_attr_name):
        spring = LinearSpring(self.stiffness, self.pathway, self.l)
        value = getattr(self, fixture_attr_name)
        with pytest.raises(AttributeError):
            setattr(spring, property_name, value)

    @pytest.mark.parametrize(
        'equilibrium_length, expected',
        [
            (S.Zero, 'LinearSpring(k, LinearPathway(pA, pB))'),
            (
                Symbol('l'),
                'LinearSpring(k, LinearPathway(pA, pB), equilibrium_length=l)',
            ),
        ]
    )
    def test_repr(self, equilibrium_length, expected):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        spring = LinearSpring(self.stiffness, self.pathway, equilibrium_length)
        assert repr(spring) == expected

    def test_to_loads(self):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        spring = LinearSpring(self.stiffness, self.pathway, self.l)
        normal = self.q/sqrt(self.q**2)*self.N.x
        pA_force = self.stiffness*(sqrt(self.q**2) - self.l)*normal
        pB_force = -self.stiffness*(sqrt(self.q**2) - self.l)*normal
        expected = [Force(self.pA, pA_force), Force(self.pB, pB_force)]
        loads = spring.to_loads()

        for load, (point, vector) in zip(loads, expected):
            assert isinstance(load, Force)
            assert load.point == point
            assert (load.vector - vector).simplify() == 0


class TestLinearDamper:

    @pytest.fixture(autouse=True)
    def _linear_damper_fixture(self):
        self.damping = Symbol('c')
        self.l = Symbol('l')
        self.pA = Point('pA')
        self.pB = Point('pB')
        self.pathway = LinearPathway(self.pA, self.pB)
        self.q = dynamicsymbols('q')
        self.dq = dynamicsymbols('q', 1)
        self.u = dynamicsymbols('u')
        self.N = ReferenceFrame('N')

    def test_is_force_actuator_subclass(self):
        assert issubclass(LinearDamper, ForceActuator)

    def test_is_actuator_base_subclass(self):
        assert issubclass(LinearDamper, ActuatorBase)

    def test_valid_constructor(self):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        damper = LinearDamper(self.damping, self.pathway)

        assert isinstance(damper, LinearDamper)

        assert hasattr(damper, 'damping')
        assert isinstance(damper.damping, ExprType)
        assert damper.damping == self.damping

        assert hasattr(damper, 'pathway')
        assert isinstance(damper.pathway, LinearPathway)
        assert damper.pathway == self.pathway

    def test_valid_constructor_force(self):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        damper = LinearDamper(self.damping, self.pathway)

        expected_force = -self.damping*sqrt(self.q**2)*self.dq/self.q
        assert hasattr(damper, 'force')
        assert isinstance(damper.force, ExprType)
        assert damper.force == expected_force

    @pytest.mark.parametrize('damping', [None, 'c'])
    def test_invalid_constructor_damping_not_sympifyable(self, damping):
        with pytest.raises(SympifyError):
            _ = LinearDamper(damping, self.pathway)

    def test_invalid_constructor_pathway_not_pathway_base(self):
        with pytest.raises(TypeError):
            _ = LinearDamper(self.damping, None)

    @pytest.mark.parametrize(
        'property_name, fixture_attr_name',
        [
            ('damping', 'damping'),
            ('pathway', 'pathway'),
        ]
    )
    def test_properties_are_immutable(self, property_name, fixture_attr_name):
        damper = LinearDamper(self.damping, self.pathway)
        value = getattr(self, fixture_attr_name)
        with pytest.raises(AttributeError):
            setattr(damper, property_name, value)

    def test_repr(self):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        damper = LinearDamper(self.damping, self.pathway)
        expected = 'LinearDamper(c, LinearPathway(pA, pB))'
        assert repr(damper) == expected

    def test_to_loads(self):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        damper = LinearDamper(self.damping, self.pathway)
        direction = self.q**2/self.q**2*self.N.x
        pA_force = self.damping*self.dq*direction
        pB_force = -self.damping*self.dq*direction
        expected = [Force(self.pA, pA_force), Force(self.pB, pB_force)]
        assert damper.to_loads() == expected


class TestForcedMassSpringDamperModel():
    r"""A single degree of freedom translational forced mass-spring-damper.

    Notes
    =====

    This system is well known to have the governing equation:

    .. math::
        m \ddot{x} = F - k x - c \dot{x}

    where $F$ is an externally applied force, $m$ is the mass of the particle
    to which the spring and damper are attached, $k$ is the spring's stiffness,
    $c$ is the dampers damping coefficient, and $x$ is the generalized
    coordinate representing the system's single (translational) degree of
    freedom.

    """

    @pytest.fixture(autouse=True)
    def _force_mass_spring_damper_model_fixture(self):
        self.m = Symbol('m')
        self.k = Symbol('k')
        self.c = Symbol('c')
        self.F = Symbol('F')

        self.q = dynamicsymbols('q')
        self.dq = dynamicsymbols('q', 1)
        self.u = dynamicsymbols('u')

        self.frame = ReferenceFrame('N')
        self.origin = Point('pO')
        self.origin.set_vel(self.frame, 0)

        self.attachment = Point('pA')
        self.attachment.set_pos(self.origin, self.q*self.frame.x)

        self.mass = Particle('mass', self.attachment, self.m)
        self.pathway = LinearPathway(self.origin, self.attachment)

        self.kanes_method = KanesMethod(
            self.frame,
            q_ind=[self.q],
            u_ind=[self.u],
            kd_eqs=[self.dq - self.u],
        )
        self.bodies = [self.mass]

        self.mass_matrix = Matrix([[self.m]])
        self.forcing = Matrix([[self.F - self.c*self.u - self.k*self.q]])

    def test_force_acuator(self):
        stiffness = -self.k*self.pathway.length
        spring = ForceActuator(stiffness, self.pathway)
        damping = -self.c*self.pathway.extension_velocity
        damper = ForceActuator(damping, self.pathway)

        loads = [
            (self.attachment, self.F*self.frame.x),
            *spring.to_loads(),
            *damper.to_loads(),
        ]
        self.kanes_method.kanes_equations(self.bodies, loads)

        assert self.kanes_method.mass_matrix == self.mass_matrix
        assert self.kanes_method.forcing == self.forcing

    def test_linear_spring_linear_damper(self):
        spring = LinearSpring(self.k, self.pathway)
        damper = LinearDamper(self.c, self.pathway)

        loads = [
            (self.attachment, self.F*self.frame.x),
            *spring.to_loads(),
            *damper.to_loads(),
        ]
        self.kanes_method.kanes_equations(self.bodies, loads)

        assert self.kanes_method.mass_matrix == self.mass_matrix
        assert self.kanes_method.forcing == self.forcing


class TestTorqueActuator:

    @pytest.fixture(autouse=True)
    def _torque_actuator_fixture(self):
        self.torque = Symbol('T')
        self.N = ReferenceFrame('N')
        self.A = ReferenceFrame('A')
        self.axis = self.N.z
        self.target = RigidBody('target', frame=self.N)
        self.reaction = RigidBody('reaction', frame=self.A)

    def test_is_actuator_base_subclass(self):
        assert issubclass(TorqueActuator, ActuatorBase)

    @pytest.mark.parametrize(
        'torque',
        [
            Symbol('T'),
            dynamicsymbols('T'),
            Symbol('T')**2 + Symbol('T'),
        ]
    )
    @pytest.mark.parametrize(
        'target_frame, reaction_frame',
        [
            (target.frame, reaction.frame),
            (target, reaction.frame),
            (target.frame, reaction),
            (target, reaction),
        ]
    )
    def test_valid_constructor_with_reaction(
        self,
        torque,
        target_frame,
        reaction_frame,
    ):
        instance = TorqueActuator(
            torque,
            self.axis,
            target_frame,
            reaction_frame,
        )
        assert isinstance(instance, TorqueActuator)

        assert hasattr(instance, 'torque')
        assert isinstance(instance.torque, ExprType)
        assert instance.torque == torque

        assert hasattr(instance, 'axis')
        assert isinstance(instance.axis, Vector)
        assert instance.axis == self.axis

        assert hasattr(instance, 'target_frame')
        assert isinstance(instance.target_frame, ReferenceFrame)
        assert instance.target_frame == target.frame

        assert hasattr(instance, 'reaction_frame')
        assert isinstance(instance.reaction_frame, ReferenceFrame)
        assert instance.reaction_frame == reaction.frame

    @pytest.mark.parametrize(
        'torque',
        [
            Symbol('T'),
            dynamicsymbols('T'),
            Symbol('T')**2 + Symbol('T'),
        ]
    )
    @pytest.mark.parametrize('target_frame', [target.frame, target])
    def test_valid_constructor_without_reaction(self, torque, target_frame):
        instance = TorqueActuator(torque, self.axis, target_frame)
        assert isinstance(instance, TorqueActuator)

        assert hasattr(instance, 'torque')
        assert isinstance(instance.torque, ExprType)
        assert instance.torque == torque

        assert hasattr(instance, 'axis')
        assert isinstance(instance.axis, Vector)
        assert instance.axis == self.axis

        assert hasattr(instance, 'target_frame')
        assert isinstance(instance.target_frame, ReferenceFrame)
        assert instance.target_frame == target.frame

        assert hasattr(instance, 'reaction_frame')
        assert instance.reaction_frame is None

    @pytest.mark.parametrize('torque', [None, 'T'])
    def test_invalid_constructor_torque_not_sympifyable(self, torque):
        with pytest.raises(SympifyError):
            _ = TorqueActuator(torque, self.axis, self.target)

    @pytest.mark.parametrize('axis', [Symbol('a'), dynamicsymbols('a')])
    def test_invalid_constructor_axis_not_vector(self, axis):
        with pytest.raises(TypeError):
            _ = TorqueActuator(self.torque, axis, self.target, self.reaction)

    @pytest.mark.parametrize(
        'frames',
        [
            (None, ReferenceFrame('child')),
            (ReferenceFrame('parent'), True),
            (None, RigidBody('child')),
            (RigidBody('parent'), True),
        ]
    )
    def test_invalid_constructor_frames_not_frame(self, frames):
        with pytest.raises(TypeError):
            _ = TorqueActuator(self.torque, self.axis, *frames)

    @pytest.mark.parametrize(
        'property_name, fixture_attr_name',
        [
            ('torque', 'torque'),
            ('axis', 'axis'),
            ('target_frame', 'target'),
            ('reaction_frame', 'reaction'),
        ]
    )
    def test_properties_are_immutable(self, property_name, fixture_attr_name):
        actuator = TorqueActuator(
            self.torque,
            self.axis,
            self.target,
            self.reaction,
        )
        value = getattr(self, fixture_attr_name)
        with pytest.raises(AttributeError):
            setattr(actuator, property_name, value)

    def test_repr_without_reaction(self):
        actuator = TorqueActuator(self.torque, self.axis, self.target)
        expected = 'TorqueActuator(T, axis=N.z, target_frame=N)'
        assert repr(actuator) == expected

    def test_repr_with_reaction(self):
        actuator = TorqueActuator(
            self.torque,
            self.axis,
            self.target,
            self.reaction,
        )
        expected = 'TorqueActuator(T, axis=N.z, target_frame=N, reaction_frame=A)'
        assert repr(actuator) == expected

    def test_at_pin_joint_constructor(self):
        pin_joint = PinJoint(
            'pin',
            self.target,
            self.reaction,
            coordinates=dynamicsymbols('q'),
            speeds=dynamicsymbols('u'),
            parent_interframe=self.N,
            joint_axis=self.axis,
        )
        instance = TorqueActuator.at_pin_joint(self.torque, pin_joint)
        assert isinstance(instance, TorqueActuator)

        assert hasattr(instance, 'torque')
        assert isinstance(instance.torque, ExprType)
        assert instance.torque == self.torque

        assert hasattr(instance, 'axis')
        assert isinstance(instance.axis, Vector)
        assert instance.axis == self.axis

        assert hasattr(instance, 'target_frame')
        assert isinstance(instance.target_frame, ReferenceFrame)
        assert instance.target_frame == self.A

        assert hasattr(instance, 'reaction_frame')
        assert isinstance(instance.reaction_frame, ReferenceFrame)
        assert instance.reaction_frame == self.N

    def test_at_pin_joint_pin_joint_not_pin_joint_invalid(self):
        with pytest.raises(TypeError):
            _ = TorqueActuator.at_pin_joint(self.torque, Symbol('pin'))

    def test_to_loads_without_reaction(self):
        actuator = TorqueActuator(self.torque, self.axis, self.target)
        expected = [
            (self.N, self.torque*self.axis),
        ]
        assert actuator.to_loads() == expected

    def test_to_loads_with_reaction(self):
        actuator = TorqueActuator(
            self.torque,
            self.axis,
            self.target,
            self.reaction,
        )
        expected = [
            (self.N, self.torque*self.axis),
            (self.A, - self.torque*self.axis),
        ]
        assert actuator.to_loads() == expected


class NonSympifyable:
    pass


class TestDuffingSpring:
    @pytest.fixture(autouse=True)
    # Set up common vairables that will be used in multiple tests
    def _duffing_spring_fixture(self):
        self.linear_stiffness = Symbol('beta')
        self.nonlinear_stiffness = Symbol('alpha')
        self.equilibrium_length = Symbol('l')
        self.pA = Point('pA')
        self.pB = Point('pB')
        self.pathway = LinearPathway(self.pA, self.pB)
        self.q = dynamicsymbols('q')
        self.N = ReferenceFrame('N')

    # Simples tests to check that DuffingSpring is a subclass of ForceActuator and ActuatorBase
    def test_is_force_actuator_subclass(self):
        assert issubclass(DuffingSpring, ForceActuator)

    def test_is_actuator_base_subclass(self):
        assert issubclass(DuffingSpring, ActuatorBase)

    @pytest.mark.parametrize(
    # Create parametrized tests that allows running the same test function multiple times with different sets of arguments
    (
        'linear_stiffness,  '
        'expected_linear_stiffness,  '
        'nonlinear_stiffness,   '
        'expected_nonlinear_stiffness,  '
        'equilibrium_length,    '
        'expected_equilibrium_length,   '
        'force'
    ),
    [
            (
                1,
                S.One,
                1,
                S.One,
                0,
                S.Zero,
                -sqrt(dynamicsymbols('q')**2)-(sqrt(dynamicsymbols('q')**2))**3,
            ),
            (
                Symbol('beta'),
                Symbol('beta'),
                Symbol('alpha'),
                Symbol('alpha'),
                0,
                S.Zero,
                -Symbol('beta')*sqrt(dynamicsymbols('q')**2)-Symbol('alpha')*(sqrt(dynamicsymbols('q')**2))**3,
            ),
            (
                Symbol('beta'),
                Symbol('beta'),
                Symbol('alpha'),
                Symbol('alpha'),
                S.Zero,
                S.Zero,
                -Symbol('beta')*sqrt(dynamicsymbols('q')**2)-Symbol('alpha')*(sqrt(dynamicsymbols('q')**2))**3,
            ),
            (
                Symbol('beta'),
                Symbol('beta'),
                Symbol('alpha'),
                Symbol('alpha'),
                Symbol('l'),
                Symbol('l'),
                -Symbol('beta') * (sqrt(dynamicsymbols('q')**2) - Symbol('l')) - Symbol('alpha') * (sqrt(dynamicsymbols('q')**2) - Symbol('l'))**3,
            ),
        ]
    )

    # Check if DuffingSpring correctly inializes its attributes
    # It tests various combinations of linear & nonlinear stiffness, equilibriun length, and the resulting force expression
    def test_valid_constructor(
        self,
        linear_stiffness,
        expected_linear_stiffness,
        nonlinear_stiffness,
        expected_nonlinear_stiffness,
        equilibrium_length,
        expected_equilibrium_length,
        force,
    ):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        spring = DuffingSpring(linear_stiffness, nonlinear_stiffness, self.pathway, equilibrium_length)

        assert isinstance(spring, DuffingSpring)

        assert hasattr(spring, 'linear_stiffness')
        assert isinstance(spring.linear_stiffness, ExprType)
        assert spring.linear_stiffness == expected_linear_stiffness

        assert hasattr(spring, 'nonlinear_stiffness')
        assert isinstance(spring.nonlinear_stiffness, ExprType)
        assert spring.nonlinear_stiffness == expected_nonlinear_stiffness

        assert hasattr(spring, 'pathway')
        assert isinstance(spring.pathway, LinearPathway)
        assert spring.pathway == self.pathway

        assert hasattr(spring, 'equilibrium_length')
        assert isinstance(spring.equilibrium_length, ExprType)
        assert spring.equilibrium_length == expected_equilibrium_length

        assert hasattr(spring, 'force')
        assert isinstance(spring.force, ExprType)
        assert spring.force == force

    @pytest.mark.parametrize('linear_stiffness', [None, NonSympifyable()])
    def test_invalid_constructor_linear_stiffness_not_sympifyable(self, linear_stiffness):
        with pytest.raises(SympifyError):
            _ = DuffingSpring(linear_stiffness, self.nonlinear_stiffness, self.pathway, self.equilibrium_length)

    @pytest.mark.parametrize('nonlinear_stiffness', [None, NonSympifyable()])
    def test_invalid_constructor_nonlinear_stiffness_not_sympifyable(self, nonlinear_stiffness):
        with pytest.raises(SympifyError):
            _ = DuffingSpring(self.linear_stiffness, nonlinear_stiffness, self.pathway, self.equilibrium_length)

    def test_invalid_constructor_pathway_not_pathway_base(self):
        with pytest.raises(TypeError):
            _ = DuffingSpring(self.linear_stiffness, self.nonlinear_stiffness, NonSympifyable(), self.equilibrium_length)

    @pytest.mark.parametrize('equilibrium_length', [None, NonSympifyable()])
    def test_invalid_constructor_equilibrium_length_not_sympifyable(self, equilibrium_length):
        with pytest.raises(SympifyError):
            _ = DuffingSpring(self.linear_stiffness, self.nonlinear_stiffness, self.pathway, equilibrium_length)

    @pytest.mark.parametrize(
        'property_name, fixture_attr_name',
        [
            ('linear_stiffness', 'linear_stiffness'),
            ('nonlinear_stiffness', 'nonlinear_stiffness'),
            ('pathway', 'pathway'),
            ('equilibrium_length', 'equilibrium_length')
        ]
    )
    # Check if certain properties of DuffingSpring object are immutable after initialization
    # Ensure that once DuffingSpring is created, its key properties cannot be changed
    def test_properties_are_immutable(self, property_name, fixture_attr_name):
        spring = DuffingSpring(self.linear_stiffness, self.nonlinear_stiffness, self.pathway, self.equilibrium_length)
        with pytest.raises(AttributeError):
            setattr(spring, property_name, getattr(self, fixture_attr_name))

    @pytest.mark.parametrize(
        'equilibrium_length, expected',
        [
            (0, 'DuffingSpring(beta, alpha, LinearPathway(pA, pB), equilibrium_length=0)'),
            (Symbol('l'), 'DuffingSpring(beta, alpha, LinearPathway(pA, pB), equilibrium_length=l)'),
        ]
    )
    # Check the __repr__ method of DuffingSpring class
    # Check if the actual string representation of DuffingSpring instance matches the expected string for each provided parameter values
    def test_repr(self, equilibrium_length, expected):
        spring = DuffingSpring(self.linear_stiffness, self.nonlinear_stiffness, self.pathway, equilibrium_length)
        assert repr(spring) == expected

    def test_to_loads(self):
        self.pB.set_pos(self.pA, self.q*self.N.x)
        spring = DuffingSpring(self.linear_stiffness, self.nonlinear_stiffness, self.pathway, self.equilibrium_length)

        # Calculate the displacement from the equilibrium length
        displacement = self.q - self.equilibrium_length

        # Make sure this matches the computation in DuffingSpring class
        force = -self.linear_stiffness * displacement - self.nonlinear_stiffness * displacement**3

        # The expected loads on pA and pB due to the spring
        expected_loads = [Force(self.pA, force * self.N.x), Force(self.pB, -force * self.N.x)]

        # Compare expected loads to what is returned from DuffingSpring.to_loads()
        calculated_loads = spring.to_loads()
        for calculated, expected in zip(calculated_loads, expected_loads):
            assert calculated.point == expected.point
            for dim in self.N:  # Assuming self.N is the reference frame
                calculated_component = calculated.vector.dot(dim)
                expected_component = expected.vector.dot(dim)
                # Substitute all symbols with numeric values
                substitutions = {self.q: 1, Symbol('l'): 1, Symbol('alpha'): 1, Symbol('beta'): 1}  # Add other necessary symbols as needed
                diff = (calculated_component - expected_component).subs(substitutions).evalf()
                # Check if the absolute value of the difference is below a threshold
                assert Abs(diff) < 1e-9, f"The forces do not match. Difference: {diff}"
