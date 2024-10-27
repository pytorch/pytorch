"""Tests for the ``sympy.physics.biomechanics.activation.py`` module."""

import pytest

from sympy import Symbol
from sympy.core.numbers import Float, Integer, Rational
from sympy.functions.elementary.hyperbolic import tanh
from sympy.matrices import Matrix
from sympy.matrices.dense import zeros
from sympy.physics.mechanics import dynamicsymbols
from sympy.physics.biomechanics import (
    ActivationBase,
    FirstOrderActivationDeGroote2016,
    ZerothOrderActivation,
)
from sympy.physics.biomechanics._mixin import _NamedMixin
from sympy.simplify.simplify import simplify


class TestZerothOrderActivation:

    @staticmethod
    def test_class():
        assert issubclass(ZerothOrderActivation, ActivationBase)
        assert issubclass(ZerothOrderActivation, _NamedMixin)
        assert ZerothOrderActivation.__name__ == 'ZerothOrderActivation'

    @pytest.fixture(autouse=True)
    def _zeroth_order_activation_fixture(self):
        self.name = 'name'
        self.e = dynamicsymbols('e_name')
        self.instance = ZerothOrderActivation(self.name)

    def test_instance(self):
        instance = ZerothOrderActivation(self.name)
        assert isinstance(instance, ZerothOrderActivation)

    def test_with_defaults(self):
        instance = ZerothOrderActivation.with_defaults(self.name)
        assert isinstance(instance, ZerothOrderActivation)
        assert instance == ZerothOrderActivation(self.name)

    def test_name(self):
        assert hasattr(self.instance, 'name')
        assert self.instance.name == self.name

    def test_order(self):
        assert hasattr(self.instance, 'order')
        assert self.instance.order == 0

    def test_excitation_attribute(self):
        assert hasattr(self.instance, 'e')
        assert hasattr(self.instance, 'excitation')
        e_expected = dynamicsymbols('e_name')
        assert self.instance.e == e_expected
        assert self.instance.excitation == e_expected
        assert self.instance.e is self.instance.excitation

    def test_activation_attribute(self):
        assert hasattr(self.instance, 'a')
        assert hasattr(self.instance, 'activation')
        a_expected = dynamicsymbols('e_name')
        assert self.instance.a == a_expected
        assert self.instance.activation == a_expected
        assert self.instance.a is self.instance.activation is self.instance.e

    def test_state_vars_attribute(self):
        assert hasattr(self.instance, 'x')
        assert hasattr(self.instance, 'state_vars')
        assert self.instance.x == self.instance.state_vars
        x_expected = zeros(0, 1)
        assert self.instance.x == x_expected
        assert self.instance.state_vars == x_expected
        assert isinstance(self.instance.x, Matrix)
        assert isinstance(self.instance.state_vars, Matrix)
        assert self.instance.x.shape == (0, 1)
        assert self.instance.state_vars.shape == (0, 1)

    def test_input_vars_attribute(self):
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

    def test_constants_attribute(self):
        assert hasattr(self.instance, 'p')
        assert hasattr(self.instance, 'constants')
        assert self.instance.p == self.instance.constants
        p_expected = zeros(0, 1)
        assert self.instance.p == p_expected
        assert self.instance.constants == p_expected
        assert isinstance(self.instance.p, Matrix)
        assert isinstance(self.instance.constants, Matrix)
        assert self.instance.p.shape == (0, 1)
        assert self.instance.constants.shape == (0, 1)

    def test_M_attribute(self):
        assert hasattr(self.instance, 'M')
        M_expected = Matrix([])
        assert self.instance.M == M_expected
        assert isinstance(self.instance.M, Matrix)
        assert self.instance.M.shape == (0, 0)

    def test_F(self):
        assert hasattr(self.instance, 'F')
        F_expected = zeros(0, 1)
        assert self.instance.F == F_expected
        assert isinstance(self.instance.F, Matrix)
        assert self.instance.F.shape == (0, 1)

    def test_rhs(self):
        assert hasattr(self.instance, 'rhs')
        rhs_expected = zeros(0, 1)
        rhs = self.instance.rhs()
        assert rhs == rhs_expected
        assert isinstance(rhs, Matrix)
        assert rhs.shape == (0, 1)

    def test_repr(self):
        expected = 'ZerothOrderActivation(\'name\')'
        assert repr(self.instance) == expected


class TestFirstOrderActivationDeGroote2016:

    @staticmethod
    def test_class():
        assert issubclass(FirstOrderActivationDeGroote2016, ActivationBase)
        assert issubclass(FirstOrderActivationDeGroote2016, _NamedMixin)
        assert FirstOrderActivationDeGroote2016.__name__ == 'FirstOrderActivationDeGroote2016'

    @pytest.fixture(autouse=True)
    def _first_order_activation_de_groote_2016_fixture(self):
        self.name = 'name'
        self.e = dynamicsymbols('e_name')
        self.a = dynamicsymbols('a_name')
        self.tau_a = Symbol('tau_a')
        self.tau_d = Symbol('tau_d')
        self.b = Symbol('b')
        self.instance = FirstOrderActivationDeGroote2016(
            self.name,
            self.tau_a,
            self.tau_d,
            self.b,
        )

    def test_instance(self):
        instance = FirstOrderActivationDeGroote2016(self.name)
        assert isinstance(instance, FirstOrderActivationDeGroote2016)

    def test_with_defaults(self):
        instance = FirstOrderActivationDeGroote2016.with_defaults(self.name)
        assert isinstance(instance, FirstOrderActivationDeGroote2016)
        assert instance.tau_a == Float('0.015')
        assert instance.activation_time_constant == Float('0.015')
        assert instance.tau_d == Float('0.060')
        assert instance.deactivation_time_constant == Float('0.060')
        assert instance.b == Float('10.0')
        assert instance.smoothing_rate == Float('10.0')

    def test_name(self):
        assert hasattr(self.instance, 'name')
        assert self.instance.name == self.name

    def test_order(self):
        assert hasattr(self.instance, 'order')
        assert self.instance.order == 1

    def test_excitation(self):
        assert hasattr(self.instance, 'e')
        assert hasattr(self.instance, 'excitation')
        e_expected = dynamicsymbols('e_name')
        assert self.instance.e == e_expected
        assert self.instance.excitation == e_expected
        assert self.instance.e is self.instance.excitation

    def test_excitation_is_immutable(self):
        with pytest.raises(AttributeError):
            self.instance.e = None
        with pytest.raises(AttributeError):
            self.instance.excitation = None

    def test_activation(self):
        assert hasattr(self.instance, 'a')
        assert hasattr(self.instance, 'activation')
        a_expected = dynamicsymbols('a_name')
        assert self.instance.a == a_expected
        assert self.instance.activation == a_expected

    def test_activation_is_immutable(self):
        with pytest.raises(AttributeError):
            self.instance.a = None
        with pytest.raises(AttributeError):
            self.instance.activation = None

    @pytest.mark.parametrize(
        'tau_a, expected',
        [
            (None, Symbol('tau_a_name')),
            (Symbol('tau_a'), Symbol('tau_a')),
            (Float('0.015'), Float('0.015')),
        ]
    )
    def test_activation_time_constant(self, tau_a, expected):
        instance = FirstOrderActivationDeGroote2016(
            'name', activation_time_constant=tau_a,
        )
        assert instance.tau_a == expected
        assert instance.activation_time_constant == expected
        assert instance.tau_a is instance.activation_time_constant

    def test_activation_time_constant_is_immutable(self):
        with pytest.raises(AttributeError):
            self.instance.tau_a = None
        with pytest.raises(AttributeError):
            self.instance.activation_time_constant = None

    @pytest.mark.parametrize(
        'tau_d, expected',
        [
            (None, Symbol('tau_d_name')),
            (Symbol('tau_d'), Symbol('tau_d')),
            (Float('0.060'), Float('0.060')),
        ]
    )
    def test_deactivation_time_constant(self, tau_d, expected):
        instance = FirstOrderActivationDeGroote2016(
            'name', deactivation_time_constant=tau_d,
        )
        assert instance.tau_d == expected
        assert instance.deactivation_time_constant == expected
        assert instance.tau_d is instance.deactivation_time_constant

    def test_deactivation_time_constant_is_immutable(self):
        with pytest.raises(AttributeError):
            self.instance.tau_d = None
        with pytest.raises(AttributeError):
            self.instance.deactivation_time_constant = None

    @pytest.mark.parametrize(
        'b, expected',
        [
            (None, Symbol('b_name')),
            (Symbol('b'), Symbol('b')),
            (Integer('10'), Integer('10')),
        ]
    )
    def test_smoothing_rate(self, b, expected):
        instance = FirstOrderActivationDeGroote2016(
            'name', smoothing_rate=b,
        )
        assert instance.b == expected
        assert instance.smoothing_rate == expected
        assert instance.b is instance.smoothing_rate

    def test_smoothing_rate_is_immutable(self):
        with pytest.raises(AttributeError):
            self.instance.b = None
        with pytest.raises(AttributeError):
            self.instance.smoothing_rate = None

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
        p_expected = Matrix([self.tau_a, self.tau_d, self.b])
        assert self.instance.p == p_expected
        assert self.instance.constants == p_expected
        assert isinstance(self.instance.p, Matrix)
        assert isinstance(self.instance.constants, Matrix)
        assert self.instance.p.shape == (3, 1)
        assert self.instance.constants.shape == (3, 1)

    def test_M(self):
        assert hasattr(self.instance, 'M')
        M_expected = Matrix([1])
        assert self.instance.M == M_expected
        assert isinstance(self.instance.M, Matrix)
        assert self.instance.M.shape == (1, 1)

    def test_F(self):
        assert hasattr(self.instance, 'F')
        da_expr = (
            ((1/(self.tau_a*(Rational(1, 2) + Rational(3, 2)*self.a)))
            *(Rational(1, 2) + Rational(1, 2)*tanh(self.b*(self.e - self.a)))
            + ((Rational(1, 2) + Rational(3, 2)*self.a)/self.tau_d)
            *(Rational(1, 2) - Rational(1, 2)*tanh(self.b*(self.e - self.a))))
            *(self.e - self.a)
        )
        F_expected = Matrix([da_expr])
        assert self.instance.F == F_expected
        assert isinstance(self.instance.F, Matrix)
        assert self.instance.F.shape == (1, 1)

    def test_rhs(self):
        assert hasattr(self.instance, 'rhs')
        da_expr = (
            ((1/(self.tau_a*(Rational(1, 2) + Rational(3, 2)*self.a)))
            *(Rational(1, 2) + Rational(1, 2)*tanh(self.b*(self.e - self.a)))
            + ((Rational(1, 2) + Rational(3, 2)*self.a)/self.tau_d)
            *(Rational(1, 2) - Rational(1, 2)*tanh(self.b*(self.e - self.a))))
            *(self.e - self.a)
        )
        rhs_expected = Matrix([da_expr])
        rhs = self.instance.rhs()
        assert rhs == rhs_expected
        assert isinstance(rhs, Matrix)
        assert rhs.shape == (1, 1)
        assert simplify(self.instance.M.solve(self.instance.F) - rhs) == zeros(1)

    def test_repr(self):
        expected = (
            'FirstOrderActivationDeGroote2016(\'name\', '
            'activation_time_constant=tau_a, '
            'deactivation_time_constant=tau_d, '
            'smoothing_rate=b)'
        )
        assert repr(self.instance) == expected
