"""Tests for the ``sympy.physics.biomechanics.musculotendon.py`` module."""

import abc

import pytest

from sympy.core.expr import UnevaluatedExpr
from sympy.core.numbers import Float, Integer, Rational
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import MutableDenseMatrix as Matrix, eye, zeros
from sympy.physics.biomechanics.activation import (
    FirstOrderActivationDeGroote2016
)
from sympy.physics.biomechanics.curve import (
    CharacteristicCurveCollection,
    FiberForceLengthActiveDeGroote2016,
    FiberForceLengthPassiveDeGroote2016,
    FiberForceLengthPassiveInverseDeGroote2016,
    FiberForceVelocityDeGroote2016,
    FiberForceVelocityInverseDeGroote2016,
    TendonForceLengthDeGroote2016,
    TendonForceLengthInverseDeGroote2016,
)
from sympy.physics.biomechanics.musculotendon import (
    MusculotendonBase,
    MusculotendonDeGroote2016,
    MusculotendonFormulation,
)
from sympy.physics.biomechanics._mixin import _NamedMixin
from sympy.physics.mechanics.actuator import ForceActuator
from sympy.physics.mechanics.pathway import LinearPathway
from sympy.physics.vector.frame import ReferenceFrame
from sympy.physics.vector.functions import dynamicsymbols
from sympy.physics.vector.point import Point
from sympy.simplify.simplify import simplify


class TestMusculotendonFormulation:
    @staticmethod
    def test_rigid_tendon_member():
        assert MusculotendonFormulation(0) == 0
        assert MusculotendonFormulation.RIGID_TENDON == 0

    @staticmethod
    def test_fiber_length_explicit_member():
        assert MusculotendonFormulation(1) == 1
        assert MusculotendonFormulation.FIBER_LENGTH_EXPLICIT == 1

    @staticmethod
    def test_tendon_force_explicit_member():
        assert MusculotendonFormulation(2) == 2
        assert MusculotendonFormulation.TENDON_FORCE_EXPLICIT == 2

    @staticmethod
    def test_fiber_length_implicit_member():
        assert MusculotendonFormulation(3) == 3
        assert MusculotendonFormulation.FIBER_LENGTH_IMPLICIT == 3

    @staticmethod
    def test_tendon_force_implicit_member():
        assert MusculotendonFormulation(4) == 4
        assert MusculotendonFormulation.TENDON_FORCE_IMPLICIT == 4


class TestMusculotendonBase:

    @staticmethod
    def test_is_abstract_base_class():
        assert issubclass(MusculotendonBase, abc.ABC)

    @staticmethod
    def test_class():
        assert issubclass(MusculotendonBase, ForceActuator)
        assert issubclass(MusculotendonBase, _NamedMixin)
        assert MusculotendonBase.__name__ == 'MusculotendonBase'

    @staticmethod
    def test_cannot_instantiate_directly():
        with pytest.raises(TypeError):
            _ = MusculotendonBase()


@pytest.mark.parametrize('musculotendon_concrete', [MusculotendonDeGroote2016])
class TestMusculotendonRigidTendon:

    @pytest.fixture(autouse=True)
    def _musculotendon_rigid_tendon_fixture(self, musculotendon_concrete):
        self.name = 'name'
        self.N = ReferenceFrame('N')
        self.q = dynamicsymbols('q')
        self.origin = Point('pO')
        self.insertion = Point('pI')
        self.insertion.set_pos(self.origin, self.q*self.N.x)
        self.pathway = LinearPathway(self.origin, self.insertion)
        self.activation = FirstOrderActivationDeGroote2016(self.name)
        self.e = self.activation.excitation
        self.a = self.activation.activation
        self.tau_a = self.activation.activation_time_constant
        self.tau_d = self.activation.deactivation_time_constant
        self.b = self.activation.smoothing_rate
        self.formulation = MusculotendonFormulation.RIGID_TENDON
        self.l_T_slack = Symbol('l_T_slack')
        self.F_M_max = Symbol('F_M_max')
        self.l_M_opt = Symbol('l_M_opt')
        self.v_M_max = Symbol('v_M_max')
        self.alpha_opt = Symbol('alpha_opt')
        self.beta = Symbol('beta')
        self.instance = musculotendon_concrete(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=self.formulation,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
        )
        self.da_expr = (
            (1/(self.tau_a*(Rational(1, 2) + Rational(3, 2)*self.a)))
            *(Rational(1, 2) + Rational(1, 2)*tanh(self.b*(self.e - self.a)))
            + ((Rational(1, 2) + Rational(3, 2)*self.a)/self.tau_d)
            *(Rational(1, 2) - Rational(1, 2)*tanh(self.b*(self.e - self.a)))
        )*(self.e - self.a)

    def test_state_vars(self):
        assert hasattr(self.instance, 'x')
        assert hasattr(self.instance, 'state_vars')
        assert self.instance.x == self.instance.state_vars
        x_expected = Matrix([self.a])
        assert self.instance.x == x_expected
        assert self.instance.state_vars == x_expected
        assert isinstance(self.instance.x, Matrix)
        assert isinstance(self.instance.state_vars, Matrix)
        assert self.instance.x.shape == (1, 1)
        assert self.instance.state_vars.shape == (1, 1)

    def test_input_vars(self):
        assert hasattr(self.instance, 'r')
        assert hasattr(self.instance, 'input_vars')
        assert self.instance.r == self.instance.input_vars
        r_expected = Matrix([self.e])
        assert self.instance.r == r_expected
        assert self.instance.input_vars == r_expected
        assert isinstance(self.instance.r, Matrix)
        assert isinstance(self.instance.input_vars, Matrix)
        assert self.instance.r.shape == (1, 1)
        assert self.instance.input_vars.shape == (1, 1)

    def test_constants(self):
        assert hasattr(self.instance, 'p')
        assert hasattr(self.instance, 'constants')
        assert self.instance.p == self.instance.constants
        p_expected = Matrix(
            [
                self.l_T_slack,
                self.F_M_max,
                self.l_M_opt,
                self.v_M_max,
                self.alpha_opt,
                self.beta,
                self.tau_a,
                self.tau_d,
                self.b,
                Symbol('c_0_fl_T_name'),
                Symbol('c_1_fl_T_name'),
                Symbol('c_2_fl_T_name'),
                Symbol('c_3_fl_T_name'),
                Symbol('c_0_fl_M_pas_name'),
                Symbol('c_1_fl_M_pas_name'),
                Symbol('c_0_fl_M_act_name'),
                Symbol('c_1_fl_M_act_name'),
                Symbol('c_2_fl_M_act_name'),
                Symbol('c_3_fl_M_act_name'),
                Symbol('c_4_fl_M_act_name'),
                Symbol('c_5_fl_M_act_name'),
                Symbol('c_6_fl_M_act_name'),
                Symbol('c_7_fl_M_act_name'),
                Symbol('c_8_fl_M_act_name'),
                Symbol('c_9_fl_M_act_name'),
                Symbol('c_10_fl_M_act_name'),
                Symbol('c_11_fl_M_act_name'),
                Symbol('c_0_fv_M_name'),
                Symbol('c_1_fv_M_name'),
                Symbol('c_2_fv_M_name'),
                Symbol('c_3_fv_M_name'),
            ]
        )
        assert self.instance.p == p_expected
        assert self.instance.constants == p_expected
        assert isinstance(self.instance.p, Matrix)
        assert isinstance(self.instance.constants, Matrix)
        assert self.instance.p.shape == (31, 1)
        assert self.instance.constants.shape == (31, 1)

    def test_M(self):
        assert hasattr(self.instance, 'M')
        M_expected = Matrix([1])
        assert self.instance.M == M_expected
        assert isinstance(self.instance.M, Matrix)
        assert self.instance.M.shape == (1, 1)

    def test_F(self):
        assert hasattr(self.instance, 'F')
        F_expected = Matrix([self.da_expr])
        assert self.instance.F == F_expected
        assert isinstance(self.instance.F, Matrix)
        assert self.instance.F.shape == (1, 1)

    def test_rhs(self):
        assert hasattr(self.instance, 'rhs')
        rhs_expected = Matrix([self.da_expr])
        rhs = self.instance.rhs()
        assert isinstance(rhs, Matrix)
        assert rhs.shape == (1, 1)
        assert simplify(rhs - rhs_expected) == zeros(1)


@pytest.mark.parametrize(
    'musculotendon_concrete, curve',
    [
        (
            MusculotendonDeGroote2016,
            CharacteristicCurveCollection(
                tendon_force_length=TendonForceLengthDeGroote2016,
                tendon_force_length_inverse=TendonForceLengthInverseDeGroote2016,
                fiber_force_length_passive=FiberForceLengthPassiveDeGroote2016,
                fiber_force_length_passive_inverse=FiberForceLengthPassiveInverseDeGroote2016,
                fiber_force_length_active=FiberForceLengthActiveDeGroote2016,
                fiber_force_velocity=FiberForceVelocityDeGroote2016,
                fiber_force_velocity_inverse=FiberForceVelocityInverseDeGroote2016,
            ),
        )
    ],
)
class TestFiberLengthExplicit:

    @pytest.fixture(autouse=True)
    def _musculotendon_fiber_length_explicit_fixture(
        self,
        musculotendon_concrete,
        curve,
    ):
        self.name = 'name'
        self.N = ReferenceFrame('N')
        self.q = dynamicsymbols('q')
        self.origin = Point('pO')
        self.insertion = Point('pI')
        self.insertion.set_pos(self.origin, self.q*self.N.x)
        self.pathway = LinearPathway(self.origin, self.insertion)
        self.activation = FirstOrderActivationDeGroote2016(self.name)
        self.e = self.activation.excitation
        self.a = self.activation.activation
        self.tau_a = self.activation.activation_time_constant
        self.tau_d = self.activation.deactivation_time_constant
        self.b = self.activation.smoothing_rate
        self.formulation = MusculotendonFormulation.FIBER_LENGTH_EXPLICIT
        self.l_T_slack = Symbol('l_T_slack')
        self.F_M_max = Symbol('F_M_max')
        self.l_M_opt = Symbol('l_M_opt')
        self.v_M_max = Symbol('v_M_max')
        self.alpha_opt = Symbol('alpha_opt')
        self.beta = Symbol('beta')
        self.instance = musculotendon_concrete(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=self.formulation,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
            with_defaults=True,
        )
        self.l_M_tilde = dynamicsymbols('l_M_tilde_name')
        l_MT = self.pathway.length
        l_M = self.l_M_tilde*self.l_M_opt
        l_T = l_MT - sqrt(l_M**2 - (self.l_M_opt*sin(self.alpha_opt))**2)
        fl_T = curve.tendon_force_length.with_defaults(l_T/self.l_T_slack)
        fl_M_pas = curve.fiber_force_length_passive.with_defaults(self.l_M_tilde)
        fl_M_act = curve.fiber_force_length_active.with_defaults(self.l_M_tilde)
        v_M_tilde = curve.fiber_force_velocity_inverse.with_defaults(
            ((((fl_T*self.F_M_max)/((l_MT - l_T)/l_M))/self.F_M_max) - fl_M_pas)
            /(self.a*fl_M_act)
        )
        self.dl_M_tilde_expr = (self.v_M_max/self.l_M_opt)*v_M_tilde
        self.da_expr = (
            (1/(self.tau_a*(Rational(1, 2) + Rational(3, 2)*self.a)))
            *(Rational(1, 2) + Rational(1, 2)*tanh(self.b*(self.e - self.a)))
            + ((Rational(1, 2) + Rational(3, 2)*self.a)/self.tau_d)
            *(Rational(1, 2) - Rational(1, 2)*tanh(self.b*(self.e - self.a)))
        )*(self.e - self.a)

    def test_state_vars(self):
        assert hasattr(self.instance, 'x')
        assert hasattr(self.instance, 'state_vars')
        assert self.instance.x == self.instance.state_vars
        x_expected = Matrix([self.l_M_tilde, self.a])
        assert self.instance.x == x_expected
        assert self.instance.state_vars == x_expected
        assert isinstance(self.instance.x, Matrix)
        assert isinstance(self.instance.state_vars, Matrix)
        assert self.instance.x.shape == (2, 1)
        assert self.instance.state_vars.shape == (2, 1)

    def test_input_vars(self):
        assert hasattr(self.instance, 'r')
        assert hasattr(self.instance, 'input_vars')
        assert self.instance.r == self.instance.input_vars
        r_expected = Matrix([self.e])
        assert self.instance.r == r_expected
        assert self.instance.input_vars == r_expected
        assert isinstance(self.instance.r, Matrix)
        assert isinstance(self.instance.input_vars, Matrix)
        assert self.instance.r.shape == (1, 1)
        assert self.instance.input_vars.shape == (1, 1)

    def test_constants(self):
        assert hasattr(self.instance, 'p')
        assert hasattr(self.instance, 'constants')
        assert self.instance.p == self.instance.constants
        p_expected = Matrix(
            [
                self.l_T_slack,
                self.F_M_max,
                self.l_M_opt,
                self.v_M_max,
                self.alpha_opt,
                self.beta,
                self.tau_a,
                self.tau_d,
                self.b,
            ]
        )
        assert self.instance.p == p_expected
        assert self.instance.constants == p_expected
        assert isinstance(self.instance.p, Matrix)
        assert isinstance(self.instance.constants, Matrix)
        assert self.instance.p.shape == (9, 1)
        assert self.instance.constants.shape == (9, 1)

    def test_M(self):
        assert hasattr(self.instance, 'M')
        M_expected = eye(2)
        assert self.instance.M == M_expected
        assert isinstance(self.instance.M, Matrix)
        assert self.instance.M.shape == (2, 2)

    def test_F(self):
        assert hasattr(self.instance, 'F')
        F_expected = Matrix([self.dl_M_tilde_expr, self.da_expr])
        assert self.instance.F == F_expected
        assert isinstance(self.instance.F, Matrix)
        assert self.instance.F.shape == (2, 1)

    def test_rhs(self):
        assert hasattr(self.instance, 'rhs')
        rhs_expected = Matrix([self.dl_M_tilde_expr, self.da_expr])
        rhs = self.instance.rhs()
        assert isinstance(rhs, Matrix)
        assert rhs.shape == (2, 1)
        assert simplify(rhs - rhs_expected) == zeros(2, 1)


@pytest.mark.parametrize(
    'musculotendon_concrete, curve',
    [
        (
            MusculotendonDeGroote2016,
            CharacteristicCurveCollection(
                tendon_force_length=TendonForceLengthDeGroote2016,
                tendon_force_length_inverse=TendonForceLengthInverseDeGroote2016,
                fiber_force_length_passive=FiberForceLengthPassiveDeGroote2016,
                fiber_force_length_passive_inverse=FiberForceLengthPassiveInverseDeGroote2016,
                fiber_force_length_active=FiberForceLengthActiveDeGroote2016,
                fiber_force_velocity=FiberForceVelocityDeGroote2016,
                fiber_force_velocity_inverse=FiberForceVelocityInverseDeGroote2016,
            ),
        )
    ],
)
class TestTendonForceExplicit:

    @pytest.fixture(autouse=True)
    def _musculotendon_tendon_force_explicit_fixture(
        self,
        musculotendon_concrete,
        curve,
    ):
        self.name = 'name'
        self.N = ReferenceFrame('N')
        self.q = dynamicsymbols('q')
        self.origin = Point('pO')
        self.insertion = Point('pI')
        self.insertion.set_pos(self.origin, self.q*self.N.x)
        self.pathway = LinearPathway(self.origin, self.insertion)
        self.activation = FirstOrderActivationDeGroote2016(self.name)
        self.e = self.activation.excitation
        self.a = self.activation.activation
        self.tau_a = self.activation.activation_time_constant
        self.tau_d = self.activation.deactivation_time_constant
        self.b = self.activation.smoothing_rate
        self.formulation = MusculotendonFormulation.TENDON_FORCE_EXPLICIT
        self.l_T_slack = Symbol('l_T_slack')
        self.F_M_max = Symbol('F_M_max')
        self.l_M_opt = Symbol('l_M_opt')
        self.v_M_max = Symbol('v_M_max')
        self.alpha_opt = Symbol('alpha_opt')
        self.beta = Symbol('beta')
        self.instance = musculotendon_concrete(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=self.formulation,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
            with_defaults=True,
        )
        self.F_T_tilde = dynamicsymbols('F_T_tilde_name')
        l_T_tilde = curve.tendon_force_length_inverse.with_defaults(self.F_T_tilde)
        l_MT = self.pathway.length
        v_MT = self.pathway.extension_velocity
        l_T = l_T_tilde*self.l_T_slack
        l_M = sqrt((l_MT - l_T)**2 + (self.l_M_opt*sin(self.alpha_opt))**2)
        l_M_tilde = l_M/self.l_M_opt
        cos_alpha = (l_MT - l_T)/l_M
        F_T = self.F_T_tilde*self.F_M_max
        F_M = F_T/cos_alpha
        F_M_tilde = F_M/self.F_M_max
        fl_M_pas = curve.fiber_force_length_passive.with_defaults(l_M_tilde)
        fl_M_act = curve.fiber_force_length_active.with_defaults(l_M_tilde)
        fv_M = (F_M_tilde - fl_M_pas)/(self.a*fl_M_act)
        v_M_tilde = curve.fiber_force_velocity_inverse.with_defaults(fv_M)
        v_M = v_M_tilde*self.v_M_max
        v_T = v_MT - v_M/cos_alpha
        v_T_tilde = v_T/self.l_T_slack
        self.dF_T_tilde_expr = (
            Float('0.2')*Float('33.93669377311689')*exp(
                Float('33.93669377311689')*UnevaluatedExpr(l_T_tilde - Float('0.995'))
            )*v_T_tilde
        )
        self.da_expr = (
            (1/(self.tau_a*(Rational(1, 2) + Rational(3, 2)*self.a)))
            *(Rational(1, 2) + Rational(1, 2)*tanh(self.b*(self.e - self.a)))
            + ((Rational(1, 2) + Rational(3, 2)*self.a)/self.tau_d)
            *(Rational(1, 2) - Rational(1, 2)*tanh(self.b*(self.e - self.a)))
        )*(self.e - self.a)

    def test_state_vars(self):
        assert hasattr(self.instance, 'x')
        assert hasattr(self.instance, 'state_vars')
        assert self.instance.x == self.instance.state_vars
        x_expected = Matrix([self.F_T_tilde, self.a])
        assert self.instance.x == x_expected
        assert self.instance.state_vars == x_expected
        assert isinstance(self.instance.x, Matrix)
        assert isinstance(self.instance.state_vars, Matrix)
        assert self.instance.x.shape == (2, 1)
        assert self.instance.state_vars.shape == (2, 1)

    def test_input_vars(self):
        assert hasattr(self.instance, 'r')
        assert hasattr(self.instance, 'input_vars')
        assert self.instance.r == self.instance.input_vars
        r_expected = Matrix([self.e])
        assert self.instance.r == r_expected
        assert self.instance.input_vars == r_expected
        assert isinstance(self.instance.r, Matrix)
        assert isinstance(self.instance.input_vars, Matrix)
        assert self.instance.r.shape == (1, 1)
        assert self.instance.input_vars.shape == (1, 1)

    def test_constants(self):
        assert hasattr(self.instance, 'p')
        assert hasattr(self.instance, 'constants')
        assert self.instance.p == self.instance.constants
        p_expected = Matrix(
            [
                self.l_T_slack,
                self.F_M_max,
                self.l_M_opt,
                self.v_M_max,
                self.alpha_opt,
                self.beta,
                self.tau_a,
                self.tau_d,
                self.b,
            ]
        )
        assert self.instance.p == p_expected
        assert self.instance.constants == p_expected
        assert isinstance(self.instance.p, Matrix)
        assert isinstance(self.instance.constants, Matrix)
        assert self.instance.p.shape == (9, 1)
        assert self.instance.constants.shape == (9, 1)

    def test_M(self):
        assert hasattr(self.instance, 'M')
        M_expected = eye(2)
        assert self.instance.M == M_expected
        assert isinstance(self.instance.M, Matrix)
        assert self.instance.M.shape == (2, 2)

    def test_F(self):
        assert hasattr(self.instance, 'F')
        F_expected = Matrix([self.dF_T_tilde_expr, self.da_expr])
        assert self.instance.F == F_expected
        assert isinstance(self.instance.F, Matrix)
        assert self.instance.F.shape == (2, 1)

    def test_rhs(self):
        assert hasattr(self.instance, 'rhs')
        rhs_expected = Matrix([self.dF_T_tilde_expr, self.da_expr])
        rhs = self.instance.rhs()
        assert isinstance(rhs, Matrix)
        assert rhs.shape == (2, 1)
        assert simplify(rhs - rhs_expected) == zeros(2, 1)


class TestMusculotendonDeGroote2016:

    @staticmethod
    def test_class():
        assert issubclass(MusculotendonDeGroote2016, ForceActuator)
        assert issubclass(MusculotendonDeGroote2016, _NamedMixin)
        assert MusculotendonDeGroote2016.__name__ == 'MusculotendonDeGroote2016'

    @staticmethod
    def test_instance():
        origin = Point('pO')
        insertion = Point('pI')
        insertion.set_pos(origin, dynamicsymbols('q')*ReferenceFrame('N').x)
        pathway = LinearPathway(origin, insertion)
        activation = FirstOrderActivationDeGroote2016('name')
        l_T_slack = Symbol('l_T_slack')
        F_M_max = Symbol('F_M_max')
        l_M_opt = Symbol('l_M_opt')
        v_M_max = Symbol('v_M_max')
        alpha_opt = Symbol('alpha_opt')
        beta = Symbol('beta')
        instance = MusculotendonDeGroote2016(
            'name',
            pathway,
            activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=l_T_slack,
            peak_isometric_force=F_M_max,
            optimal_fiber_length=l_M_opt,
            maximal_fiber_velocity=v_M_max,
            optimal_pennation_angle=alpha_opt,
            fiber_damping_coefficient=beta,
        )
        assert isinstance(instance, MusculotendonDeGroote2016)

    @pytest.fixture(autouse=True)
    def _musculotendon_fixture(self):
        self.name = 'name'
        self.N = ReferenceFrame('N')
        self.q = dynamicsymbols('q')
        self.origin = Point('pO')
        self.insertion = Point('pI')
        self.insertion.set_pos(self.origin, self.q*self.N.x)
        self.pathway = LinearPathway(self.origin, self.insertion)
        self.activation = FirstOrderActivationDeGroote2016(self.name)
        self.l_T_slack = Symbol('l_T_slack')
        self.F_M_max = Symbol('F_M_max')
        self.l_M_opt = Symbol('l_M_opt')
        self.v_M_max = Symbol('v_M_max')
        self.alpha_opt = Symbol('alpha_opt')
        self.beta = Symbol('beta')

    def test_with_defaults(self):
        origin = Point('pO')
        insertion = Point('pI')
        insertion.set_pos(origin, dynamicsymbols('q')*ReferenceFrame('N').x)
        pathway = LinearPathway(origin, insertion)
        activation = FirstOrderActivationDeGroote2016('name')
        l_T_slack = Symbol('l_T_slack')
        F_M_max = Symbol('F_M_max')
        l_M_opt = Symbol('l_M_opt')
        v_M_max = Float('10.0')
        alpha_opt = Float('0.0')
        beta = Float('0.1')
        instance = MusculotendonDeGroote2016.with_defaults(
            'name',
            pathway,
            activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=l_T_slack,
            peak_isometric_force=F_M_max,
            optimal_fiber_length=l_M_opt,
        )
        assert instance.tendon_slack_length == l_T_slack
        assert instance.peak_isometric_force == F_M_max
        assert instance.optimal_fiber_length == l_M_opt
        assert instance.maximal_fiber_velocity == v_M_max
        assert instance.optimal_pennation_angle == alpha_opt
        assert instance.fiber_damping_coefficient == beta

    @pytest.mark.parametrize(
        'l_T_slack, expected',
        [
            (None, Symbol('l_T_slack_name')),
            (Symbol('l_T_slack'), Symbol('l_T_slack')),
            (Rational(1, 2), Rational(1, 2)),
            (Float('0.5'), Float('0.5')),
        ],
    )
    def test_tendon_slack_length(self, l_T_slack, expected):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
        )
        assert instance.l_T_slack == expected
        assert instance.tendon_slack_length == expected

    @pytest.mark.parametrize(
        'F_M_max, expected',
        [
            (None, Symbol('F_M_max_name')),
            (Symbol('F_M_max'), Symbol('F_M_max')),
            (Integer(1000), Integer(1000)),
            (Float('1000.0'), Float('1000.0')),
        ],
    )
    def test_peak_isometric_force(self, F_M_max, expected):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
        )
        assert instance.F_M_max == expected
        assert instance.peak_isometric_force == expected

    @pytest.mark.parametrize(
        'l_M_opt, expected',
        [
            (None, Symbol('l_M_opt_name')),
            (Symbol('l_M_opt'), Symbol('l_M_opt')),
            (Rational(1, 2), Rational(1, 2)),
            (Float('0.5'), Float('0.5')),
        ],
    )
    def test_optimal_fiber_length(self, l_M_opt, expected):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
        )
        assert instance.l_M_opt == expected
        assert instance.optimal_fiber_length == expected

    @pytest.mark.parametrize(
        'v_M_max, expected',
        [
            (None, Symbol('v_M_max_name')),
            (Symbol('v_M_max'), Symbol('v_M_max')),
            (Integer(10), Integer(10)),
            (Float('10.0'), Float('10.0')),
        ],
    )
    def test_maximal_fiber_velocity(self, v_M_max, expected):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
        )
        assert instance.v_M_max == expected
        assert instance.maximal_fiber_velocity == expected

    @pytest.mark.parametrize(
        'alpha_opt, expected',
        [
            (None, Symbol('alpha_opt_name')),
            (Symbol('alpha_opt'), Symbol('alpha_opt')),
            (Integer(0), Integer(0)),
            (Float('0.1'), Float('0.1')),
        ],
    )
    def test_optimal_pennation_angle(self, alpha_opt, expected):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=alpha_opt,
            fiber_damping_coefficient=self.beta,
        )
        assert instance.alpha_opt == expected
        assert instance.optimal_pennation_angle == expected

    @pytest.mark.parametrize(
        'beta, expected',
        [
            (None, Symbol('beta_name')),
            (Symbol('beta'), Symbol('beta')),
            (Integer(0), Integer(0)),
            (Rational(1, 10), Rational(1, 10)),
            (Float('0.1'), Float('0.1')),
        ],
    )
    def test_fiber_damping_coefficient(self, beta, expected):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=beta,
        )
        assert instance.beta == expected
        assert instance.fiber_damping_coefficient == expected

    def test_excitation(self):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
        )
        assert hasattr(instance, 'e')
        assert hasattr(instance, 'excitation')
        e_expected = dynamicsymbols('e_name')
        assert instance.e == e_expected
        assert instance.excitation == e_expected
        assert instance.e is instance.excitation

    def test_excitation_is_immutable(self):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
        )
        with pytest.raises(AttributeError):
            instance.e = None
        with pytest.raises(AttributeError):
            instance.excitation = None

    def test_activation(self):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
        )
        assert hasattr(instance, 'a')
        assert hasattr(instance, 'activation')
        a_expected = dynamicsymbols('a_name')
        assert instance.a == a_expected
        assert instance.activation == a_expected

    def test_activation_is_immutable(self):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
        )
        with pytest.raises(AttributeError):
            instance.a = None
        with pytest.raises(AttributeError):
            instance.activation = None

    def test_repr(self):
        instance = MusculotendonDeGroote2016(
            self.name,
            self.pathway,
            self.activation,
            musculotendon_dynamics=MusculotendonFormulation.RIGID_TENDON,
            tendon_slack_length=self.l_T_slack,
            peak_isometric_force=self.F_M_max,
            optimal_fiber_length=self.l_M_opt,
            maximal_fiber_velocity=self.v_M_max,
            optimal_pennation_angle=self.alpha_opt,
            fiber_damping_coefficient=self.beta,
        )
        expected = (
            'MusculotendonDeGroote2016(\'name\', '
            'pathway=LinearPathway(pO, pI), '
            'activation_dynamics=FirstOrderActivationDeGroote2016(\'name\', '
            'activation_time_constant=tau_a_name, '
            'deactivation_time_constant=tau_d_name, '
            'smoothing_rate=b_name), '
            'musculotendon_dynamics=0, '
            'tendon_slack_length=l_T_slack, '
            'peak_isometric_force=F_M_max, '
            'optimal_fiber_length=l_M_opt, '
            'maximal_fiber_velocity=v_M_max, '
            'optimal_pennation_angle=alpha_opt, '
            'fiber_damping_coefficient=beta)'
        )
        assert repr(instance) == expected
