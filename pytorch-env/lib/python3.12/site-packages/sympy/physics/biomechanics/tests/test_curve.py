"""Tests for the ``sympy.physics.biomechanics.characteristic.py`` module."""

import pytest

from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import Function
from sympy.core.numbers import Float, Integer
from sympy.core.symbol import Symbol, symbols
from sympy.external.importtools import import_module
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import cosh, sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.biomechanics.curve import (
    CharacteristicCurveCollection,
    CharacteristicCurveFunction,
    FiberForceLengthActiveDeGroote2016,
    FiberForceLengthPassiveDeGroote2016,
    FiberForceLengthPassiveInverseDeGroote2016,
    FiberForceVelocityDeGroote2016,
    FiberForceVelocityInverseDeGroote2016,
    TendonForceLengthDeGroote2016,
    TendonForceLengthInverseDeGroote2016,
)
from sympy.printing.c import C89CodePrinter, C99CodePrinter, C11CodePrinter
from sympy.printing.cxx import (
    CXX98CodePrinter,
    CXX11CodePrinter,
    CXX17CodePrinter,
)
from sympy.printing.fortran import FCodePrinter
from sympy.printing.lambdarepr import LambdaPrinter
from sympy.printing.latex import LatexPrinter
from sympy.printing.octave import OctaveCodePrinter
from sympy.printing.numpy import (
    CuPyPrinter,
    JaxPrinter,
    NumPyPrinter,
    SciPyPrinter,
)
from sympy.printing.pycode import MpmathPrinter, PythonCodePrinter
from sympy.utilities.lambdify import lambdify

jax = import_module('jax')
numpy = import_module('numpy')

if jax:
    jax.config.update('jax_enable_x64', True)


class TestCharacteristicCurveFunction:

    @staticmethod
    @pytest.mark.parametrize(
        'code_printer, expected',
        [
            (C89CodePrinter, '(a + b)*(c + d)*(e + f)'),
            (C99CodePrinter, '(a + b)*(c + d)*(e + f)'),
            (C11CodePrinter, '(a + b)*(c + d)*(e + f)'),
            (CXX98CodePrinter, '(a + b)*(c + d)*(e + f)'),
            (CXX11CodePrinter, '(a + b)*(c + d)*(e + f)'),
            (CXX17CodePrinter, '(a + b)*(c + d)*(e + f)'),
            (FCodePrinter, '      (a + b)*(c + d)*(e + f)'),
            (OctaveCodePrinter, '(a + b).*(c + d).*(e + f)'),
            (PythonCodePrinter, '(a + b)*(c + d)*(e + f)'),
            (NumPyPrinter, '(a + b)*(c + d)*(e + f)'),
            (SciPyPrinter, '(a + b)*(c + d)*(e + f)'),
            (CuPyPrinter, '(a + b)*(c + d)*(e + f)'),
            (JaxPrinter, '(a + b)*(c + d)*(e + f)'),
            (MpmathPrinter, '(a + b)*(c + d)*(e + f)'),
            (LambdaPrinter, '(a + b)*(c + d)*(e + f)'),
        ]
    )
    def test_print_code_parenthesize(code_printer, expected):

        class ExampleFunction(CharacteristicCurveFunction):

            @classmethod
            def eval(cls, a, b):
                pass

            def doit(self, **kwargs):
                a, b = self.args
                return a + b

        a, b, c, d, e, f = symbols('a, b, c, d, e, f')
        f1 = ExampleFunction(a, b)
        f2 = ExampleFunction(c, d)
        f3 = ExampleFunction(e, f)
        assert code_printer().doprint(f1*f2*f3) == expected


class TestTendonForceLengthDeGroote2016:

    @pytest.fixture(autouse=True)
    def _tendon_force_length_arguments_fixture(self):
        self.l_T_tilde = Symbol('l_T_tilde')
        self.c0 = Symbol('c_0')
        self.c1 = Symbol('c_1')
        self.c2 = Symbol('c_2')
        self.c3 = Symbol('c_3')
        self.constants = (self.c0, self.c1, self.c2, self.c3)

    @staticmethod
    def test_class():
        assert issubclass(TendonForceLengthDeGroote2016, Function)
        assert issubclass(TendonForceLengthDeGroote2016, CharacteristicCurveFunction)
        assert TendonForceLengthDeGroote2016.__name__ == 'TendonForceLengthDeGroote2016'

    def test_instance(self):
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants)
        assert isinstance(fl_T, TendonForceLengthDeGroote2016)
        assert str(fl_T) == 'TendonForceLengthDeGroote2016(l_T_tilde, c_0, c_1, c_2, c_3)'

    def test_doit(self):
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants).doit()
        assert fl_T == self.c0*exp(self.c3*(self.l_T_tilde - self.c1)) - self.c2

    def test_doit_evaluate_false(self):
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants).doit(evaluate=False)
        assert fl_T == self.c0*exp(self.c3*UnevaluatedExpr(self.l_T_tilde - self.c1)) - self.c2

    def test_with_defaults(self):
        constants = (
            Float('0.2'),
            Float('0.995'),
            Float('0.25'),
            Float('33.93669377311689'),
        )
        fl_T_manual = TendonForceLengthDeGroote2016(self.l_T_tilde, *constants)
        fl_T_constants = TendonForceLengthDeGroote2016.with_defaults(self.l_T_tilde)
        assert fl_T_manual == fl_T_constants

    def test_differentiate_wrt_l_T_tilde(self):
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants)
        expected = self.c0*self.c3*exp(self.c3*UnevaluatedExpr(-self.c1 + self.l_T_tilde))
        assert fl_T.diff(self.l_T_tilde) == expected

    def test_differentiate_wrt_c0(self):
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants)
        expected = exp(self.c3*UnevaluatedExpr(-self.c1 + self.l_T_tilde))
        assert fl_T.diff(self.c0) == expected

    def test_differentiate_wrt_c1(self):
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants)
        expected = -self.c0*self.c3*exp(self.c3*UnevaluatedExpr(self.l_T_tilde - self.c1))
        assert fl_T.diff(self.c1) == expected

    def test_differentiate_wrt_c2(self):
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants)
        expected = Integer(-1)
        assert fl_T.diff(self.c2) == expected

    def test_differentiate_wrt_c3(self):
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants)
        expected = self.c0*(self.l_T_tilde - self.c1)*exp(self.c3*UnevaluatedExpr(self.l_T_tilde - self.c1))
        assert fl_T.diff(self.c3) == expected

    def test_inverse(self):
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants)
        assert fl_T.inverse() is TendonForceLengthInverseDeGroote2016

    def test_function_print_latex(self):
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants)
        expected = r'\operatorname{fl}^T \left( l_{T tilde} \right)'
        assert LatexPrinter().doprint(fl_T) == expected

    def test_expression_print_latex(self):
        fl_T = TendonForceLengthDeGroote2016(self.l_T_tilde, *self.constants)
        expected = r'c_{0} e^{c_{3} \left(- c_{1} + l_{T tilde}\right)} - c_{2}'
        assert LatexPrinter().doprint(fl_T.doit()) == expected

    @pytest.mark.parametrize(
        'code_printer, expected',
        [
            (
                C89CodePrinter,
                '(-0.25 + 0.20000000000000001*exp(33.93669377311689*(l_T_tilde - 0.995)))',
            ),
            (
                C99CodePrinter,
                '(-0.25 + 0.20000000000000001*exp(33.93669377311689*(l_T_tilde - 0.995)))',
            ),
            (
                C11CodePrinter,
                '(-0.25 + 0.20000000000000001*exp(33.93669377311689*(l_T_tilde - 0.995)))',
            ),
            (
                CXX98CodePrinter,
                '(-0.25 + 0.20000000000000001*exp(33.93669377311689*(l_T_tilde - 0.995)))',
            ),
            (
                CXX11CodePrinter,
                '(-0.25 + 0.20000000000000001*std::exp(33.93669377311689*(l_T_tilde - 0.995)))',
            ),
            (
                CXX17CodePrinter,
                '(-0.25 + 0.20000000000000001*std::exp(33.93669377311689*(l_T_tilde - 0.995)))',
            ),
            (
                FCodePrinter,
                '      (-0.25d0 + 0.2d0*exp(33.93669377311689d0*(l_T_tilde - 0.995d0)))',
            ),
            (
                OctaveCodePrinter,
                '(-0.25 + 0.2*exp(33.93669377311689*(l_T_tilde - 0.995)))',
            ),
            (
                PythonCodePrinter,
                '(-0.25 + 0.2*math.exp(33.93669377311689*(l_T_tilde - 0.995)))',
            ),
            (
                NumPyPrinter,
                '(-0.25 + 0.2*numpy.exp(33.93669377311689*(l_T_tilde - 0.995)))',
            ),
            (
                SciPyPrinter,
                '(-0.25 + 0.2*numpy.exp(33.93669377311689*(l_T_tilde - 0.995)))',
            ),
            (
                CuPyPrinter,
                '(-0.25 + 0.2*cupy.exp(33.93669377311689*(l_T_tilde - 0.995)))',
            ),
            (
                JaxPrinter,
                '(-0.25 + 0.2*jax.numpy.exp(33.93669377311689*(l_T_tilde - 0.995)))',
            ),
            (
                MpmathPrinter,
                '(mpmath.mpf((1, 1, -2, 1)) + mpmath.mpf((0, 3602879701896397, -54, 52))'
                '*mpmath.exp(mpmath.mpf((0, 9552330089424741, -48, 54))*(l_T_tilde + '
                'mpmath.mpf((1, 8962163258467287, -53, 53)))))',
            ),
            (
                LambdaPrinter,
                '(-0.25 + 0.2*math.exp(33.93669377311689*(l_T_tilde - 0.995)))',
            ),
        ]
    )
    def test_print_code(self, code_printer, expected):
        fl_T = TendonForceLengthDeGroote2016.with_defaults(self.l_T_tilde)
        assert code_printer().doprint(fl_T) == expected

    def test_derivative_print_code(self):
        fl_T = TendonForceLengthDeGroote2016.with_defaults(self.l_T_tilde)
        dfl_T_dl_T_tilde = fl_T.diff(self.l_T_tilde)
        expected = '6.787338754623378*math.exp(33.93669377311689*(l_T_tilde - 0.995))'
        assert PythonCodePrinter().doprint(dfl_T_dl_T_tilde) == expected

    def test_lambdify(self):
        fl_T = TendonForceLengthDeGroote2016.with_defaults(self.l_T_tilde)
        fl_T_callable = lambdify(self.l_T_tilde, fl_T)
        assert fl_T_callable(1.0) == pytest.approx(-0.013014055039221595)

    @pytest.mark.skipif(numpy is None, reason='NumPy not installed')
    def test_lambdify_numpy(self):
        fl_T = TendonForceLengthDeGroote2016.with_defaults(self.l_T_tilde)
        fl_T_callable = lambdify(self.l_T_tilde, fl_T, 'numpy')
        l_T_tilde = numpy.array([0.95, 1.0, 1.01, 1.05])
        expected = numpy.array([
            -0.2065693181344816,
            -0.0130140550392216,
            0.0827421191989246,
            1.04314889144172,
        ])
        numpy.testing.assert_allclose(fl_T_callable(l_T_tilde), expected)

    @pytest.mark.skipif(jax is None, reason='JAX not installed')
    def test_lambdify_jax(self):
        fl_T = TendonForceLengthDeGroote2016.with_defaults(self.l_T_tilde)
        fl_T_callable = jax.jit(lambdify(self.l_T_tilde, fl_T, 'jax'))
        l_T_tilde = jax.numpy.array([0.95, 1.0, 1.01, 1.05])
        expected = jax.numpy.array([
            -0.2065693181344816,
            -0.0130140550392216,
            0.0827421191989246,
            1.04314889144172,
        ])
        numpy.testing.assert_allclose(fl_T_callable(l_T_tilde), expected)


class TestTendonForceLengthInverseDeGroote2016:

    @pytest.fixture(autouse=True)
    def _tendon_force_length_inverse_arguments_fixture(self):
        self.fl_T = Symbol('fl_T')
        self.c0 = Symbol('c_0')
        self.c1 = Symbol('c_1')
        self.c2 = Symbol('c_2')
        self.c3 = Symbol('c_3')
        self.constants = (self.c0, self.c1, self.c2, self.c3)

    @staticmethod
    def test_class():
        assert issubclass(TendonForceLengthInverseDeGroote2016, Function)
        assert issubclass(TendonForceLengthInverseDeGroote2016, CharacteristicCurveFunction)
        assert TendonForceLengthInverseDeGroote2016.__name__ == 'TendonForceLengthInverseDeGroote2016'

    def test_instance(self):
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants)
        assert isinstance(fl_T_inv, TendonForceLengthInverseDeGroote2016)
        assert str(fl_T_inv) == 'TendonForceLengthInverseDeGroote2016(fl_T, c_0, c_1, c_2, c_3)'

    def test_doit(self):
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants).doit()
        assert fl_T_inv == log((self.fl_T + self.c2)/self.c0)/self.c3 + self.c1

    def test_doit_evaluate_false(self):
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants).doit(evaluate=False)
        assert fl_T_inv == log(UnevaluatedExpr((self.fl_T + self.c2)/self.c0))/self.c3 + self.c1

    def test_with_defaults(self):
        constants = (
            Float('0.2'),
            Float('0.995'),
            Float('0.25'),
            Float('33.93669377311689'),
        )
        fl_T_inv_manual = TendonForceLengthInverseDeGroote2016(self.fl_T, *constants)
        fl_T_inv_constants = TendonForceLengthInverseDeGroote2016.with_defaults(self.fl_T)
        assert fl_T_inv_manual == fl_T_inv_constants

    def test_differentiate_wrt_fl_T(self):
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants)
        expected = 1/(self.c3*(self.fl_T + self.c2))
        assert fl_T_inv.diff(self.fl_T) == expected

    def test_differentiate_wrt_c0(self):
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants)
        expected = -1/(self.c0*self.c3)
        assert fl_T_inv.diff(self.c0) == expected

    def test_differentiate_wrt_c1(self):
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants)
        expected = Integer(1)
        assert fl_T_inv.diff(self.c1) == expected

    def test_differentiate_wrt_c2(self):
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants)
        expected = 1/(self.c3*(self.fl_T + self.c2))
        assert fl_T_inv.diff(self.c2) == expected

    def test_differentiate_wrt_c3(self):
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants)
        expected = -log(UnevaluatedExpr((self.fl_T + self.c2)/self.c0))/self.c3**2
        assert fl_T_inv.diff(self.c3) == expected

    def test_inverse(self):
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants)
        assert fl_T_inv.inverse() is TendonForceLengthDeGroote2016

    def test_function_print_latex(self):
        fl_T_inv = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants)
        expected = r'\left( \operatorname{fl}^T \right)^{-1} \left( fl_{T} \right)'
        assert LatexPrinter().doprint(fl_T_inv) == expected

    def test_expression_print_latex(self):
        fl_T = TendonForceLengthInverseDeGroote2016(self.fl_T, *self.constants)
        expected = r'c_{1} + \frac{\log{\left(\frac{c_{2} + fl_{T}}{c_{0}} \right)}}{c_{3}}'
        assert LatexPrinter().doprint(fl_T.doit()) == expected

    @pytest.mark.parametrize(
        'code_printer, expected',
        [
            (
                C89CodePrinter,
                '(0.995 + 0.029466630034306838*log(5.0*fl_T + 1.25))',
            ),
            (
                C99CodePrinter,
                '(0.995 + 0.029466630034306838*log(5.0*fl_T + 1.25))',
            ),
            (
                C11CodePrinter,
                '(0.995 + 0.029466630034306838*log(5.0*fl_T + 1.25))',
            ),
            (
                CXX98CodePrinter,
                '(0.995 + 0.029466630034306838*log(5.0*fl_T + 1.25))',
            ),
            (
                CXX11CodePrinter,
                '(0.995 + 0.029466630034306838*std::log(5.0*fl_T + 1.25))',
            ),
            (
                CXX17CodePrinter,
                '(0.995 + 0.029466630034306838*std::log(5.0*fl_T + 1.25))',
            ),
            (
                FCodePrinter,
                '      (0.995d0 + 0.02946663003430684d0*log(5.0d0*fl_T + 1.25d0))',
            ),
            (
                OctaveCodePrinter,
                '(0.995 + 0.02946663003430684*log(5.0*fl_T + 1.25))',
            ),
            (
                PythonCodePrinter,
                '(0.995 + 0.02946663003430684*math.log(5.0*fl_T + 1.25))',
            ),
            (
                NumPyPrinter,
                '(0.995 + 0.02946663003430684*numpy.log(5.0*fl_T + 1.25))',
            ),
            (
                SciPyPrinter,
                '(0.995 + 0.02946663003430684*numpy.log(5.0*fl_T + 1.25))',
            ),
            (
                CuPyPrinter,
                '(0.995 + 0.02946663003430684*cupy.log(5.0*fl_T + 1.25))',
            ),
            (
                JaxPrinter,
                '(0.995 + 0.02946663003430684*jax.numpy.log(5.0*fl_T + 1.25))',
            ),
            (
                MpmathPrinter,
                '(mpmath.mpf((0, 8962163258467287, -53, 53))'
                ' + mpmath.mpf((0, 33972711434846347, -60, 55))'
                '*mpmath.log(mpmath.mpf((0, 5, 0, 3))*fl_T + mpmath.mpf((0, 5, -2, 3))))',
            ),
            (
                LambdaPrinter,
                '(0.995 + 0.02946663003430684*math.log(5.0*fl_T + 1.25))',
            ),
        ]
    )
    def test_print_code(self, code_printer, expected):
        fl_T_inv = TendonForceLengthInverseDeGroote2016.with_defaults(self.fl_T)
        assert code_printer().doprint(fl_T_inv) == expected

    def test_derivative_print_code(self):
        fl_T_inv = TendonForceLengthInverseDeGroote2016.with_defaults(self.fl_T)
        dfl_T_inv_dfl_T = fl_T_inv.diff(self.fl_T)
        expected = '1/(33.93669377311689*fl_T + 8.484173443279222)'
        assert PythonCodePrinter().doprint(dfl_T_inv_dfl_T) == expected

    def test_lambdify(self):
        fl_T_inv = TendonForceLengthInverseDeGroote2016.with_defaults(self.fl_T)
        fl_T_inv_callable = lambdify(self.fl_T, fl_T_inv)
        assert fl_T_inv_callable(0.0) == pytest.approx(1.0015752885)

    @pytest.mark.skipif(numpy is None, reason='NumPy not installed')
    def test_lambdify_numpy(self):
        fl_T_inv = TendonForceLengthInverseDeGroote2016.with_defaults(self.fl_T)
        fl_T_inv_callable = lambdify(self.fl_T, fl_T_inv, 'numpy')
        fl_T = numpy.array([-0.2, -0.01, 0.0, 1.01, 1.02, 1.05])
        expected = numpy.array([
            0.9541505769,
            1.0003724019,
            1.0015752885,
            1.0492347951,
            1.0494677341,
            1.0501557022,
        ])
        numpy.testing.assert_allclose(fl_T_inv_callable(fl_T), expected)

    @pytest.mark.skipif(jax is None, reason='JAX not installed')
    def test_lambdify_jax(self):
        fl_T_inv = TendonForceLengthInverseDeGroote2016.with_defaults(self.fl_T)
        fl_T_inv_callable = jax.jit(lambdify(self.fl_T, fl_T_inv, 'jax'))
        fl_T = jax.numpy.array([-0.2, -0.01, 0.0, 1.01, 1.02, 1.05])
        expected = jax.numpy.array([
            0.9541505769,
            1.0003724019,
            1.0015752885,
            1.0492347951,
            1.0494677341,
            1.0501557022,
        ])
        numpy.testing.assert_allclose(fl_T_inv_callable(fl_T), expected)


class TestFiberForceLengthPassiveDeGroote2016:

    @pytest.fixture(autouse=True)
    def _fiber_force_length_passive_arguments_fixture(self):
        self.l_M_tilde = Symbol('l_M_tilde')
        self.c0 = Symbol('c_0')
        self.c1 = Symbol('c_1')
        self.constants = (self.c0, self.c1)

    @staticmethod
    def test_class():
        assert issubclass(FiberForceLengthPassiveDeGroote2016, Function)
        assert issubclass(FiberForceLengthPassiveDeGroote2016, CharacteristicCurveFunction)
        assert FiberForceLengthPassiveDeGroote2016.__name__ == 'FiberForceLengthPassiveDeGroote2016'

    def test_instance(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *self.constants)
        assert isinstance(fl_M_pas, FiberForceLengthPassiveDeGroote2016)
        assert str(fl_M_pas) == 'FiberForceLengthPassiveDeGroote2016(l_M_tilde, c_0, c_1)'

    def test_doit(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *self.constants).doit()
        assert fl_M_pas == (exp((self.c1*(self.l_M_tilde - 1))/self.c0) - 1)/(exp(self.c1) - 1)

    def test_doit_evaluate_false(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *self.constants).doit(evaluate=False)
        assert fl_M_pas == (exp((self.c1*UnevaluatedExpr(self.l_M_tilde - 1))/self.c0) - 1)/(exp(self.c1) - 1)

    def test_with_defaults(self):
        constants = (
            Float('0.6'),
            Float('4.0'),
        )
        fl_M_pas_manual = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *constants)
        fl_M_pas_constants = FiberForceLengthPassiveDeGroote2016.with_defaults(self.l_M_tilde)
        assert fl_M_pas_manual == fl_M_pas_constants

    def test_differentiate_wrt_l_M_tilde(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = self.c1*exp(self.c1*UnevaluatedExpr(self.l_M_tilde - 1)/self.c0)/(self.c0*(exp(self.c1) - 1))
        assert fl_M_pas.diff(self.l_M_tilde) == expected

    def test_differentiate_wrt_c0(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = (
            -self.c1*exp(self.c1*UnevaluatedExpr(self.l_M_tilde - 1)/self.c0)
            *UnevaluatedExpr(self.l_M_tilde - 1)/(self.c0**2*(exp(self.c1) - 1))
        )
        assert fl_M_pas.diff(self.c0) == expected

    def test_differentiate_wrt_c1(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = (
            -exp(self.c1)*(-1 + exp(self.c1*UnevaluatedExpr(self.l_M_tilde - 1)/self.c0))/(exp(self.c1) - 1)**2
            + exp(self.c1*UnevaluatedExpr(self.l_M_tilde - 1)/self.c0)*(self.l_M_tilde - 1)/(self.c0*(exp(self.c1) - 1))
        )
        assert fl_M_pas.diff(self.c1) == expected

    def test_inverse(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *self.constants)
        assert fl_M_pas.inverse() is FiberForceLengthPassiveInverseDeGroote2016

    def test_function_print_latex(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = r'\operatorname{fl}^M_{pas} \left( l_{M tilde} \right)'
        assert LatexPrinter().doprint(fl_M_pas) == expected

    def test_expression_print_latex(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = r'\frac{e^{\frac{c_{1} \left(l_{M tilde} - 1\right)}{c_{0}}} - 1}{e^{c_{1}} - 1}'
        assert LatexPrinter().doprint(fl_M_pas.doit()) == expected

    @pytest.mark.parametrize(
        'code_printer, expected',
        [
            (
                C89CodePrinter,
                '(0.01865736036377405*(-1 + exp(6.666666666666667*(l_M_tilde - 1))))',
            ),
            (
                C99CodePrinter,
                '(0.01865736036377405*(-1 + exp(6.666666666666667*(l_M_tilde - 1))))',
            ),
            (
                C11CodePrinter,
                '(0.01865736036377405*(-1 + exp(6.666666666666667*(l_M_tilde - 1))))',
            ),
            (
                CXX98CodePrinter,
                '(0.01865736036377405*(-1 + exp(6.666666666666667*(l_M_tilde - 1))))',
            ),
            (
                CXX11CodePrinter,
                '(0.01865736036377405*(-1 + std::exp(6.666666666666667*(l_M_tilde - 1))))',
            ),
            (
                CXX17CodePrinter,
                '(0.01865736036377405*(-1 + std::exp(6.666666666666667*(l_M_tilde - 1))))',
            ),
            (
                FCodePrinter,
                '      (0.0186573603637741d0*(-1 + exp(6.666666666666667d0*(l_M_tilde - 1\n'
                '     @ ))))',
            ),
            (
                OctaveCodePrinter,
                '(0.0186573603637741*(-1 + exp(6.66666666666667*(l_M_tilde - 1))))',
            ),
            (
                PythonCodePrinter,
                '(0.0186573603637741*(-1 + math.exp(6.66666666666667*(l_M_tilde - 1))))',
            ),
            (
                NumPyPrinter,
                '(0.0186573603637741*(-1 + numpy.exp(6.66666666666667*(l_M_tilde - 1))))',
            ),
            (
                SciPyPrinter,
                '(0.0186573603637741*(-1 + numpy.exp(6.66666666666667*(l_M_tilde - 1))))',
            ),
            (
                CuPyPrinter,
                '(0.0186573603637741*(-1 + cupy.exp(6.66666666666667*(l_M_tilde - 1))))',
            ),
            (
                JaxPrinter,
                '(0.0186573603637741*(-1 + jax.numpy.exp(6.66666666666667*(l_M_tilde - 1))))',
            ),
            (
                MpmathPrinter,
                '(mpmath.mpf((0, 672202249456079, -55, 50))*(-1 + mpmath.exp('
                'mpmath.mpf((0, 7505999378950827, -50, 53))*(l_M_tilde - 1))))',
            ),
            (
                LambdaPrinter,
                '(0.0186573603637741*(-1 + math.exp(6.66666666666667*(l_M_tilde - 1))))',
            ),
        ]
    )
    def test_print_code(self, code_printer, expected):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016.with_defaults(self.l_M_tilde)
        assert code_printer().doprint(fl_M_pas) == expected

    def test_derivative_print_code(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016.with_defaults(self.l_M_tilde)
        fl_M_pas_dl_M_tilde = fl_M_pas.diff(self.l_M_tilde)
        expected = '0.12438240242516*math.exp(6.66666666666667*(l_M_tilde - 1))'
        assert PythonCodePrinter().doprint(fl_M_pas_dl_M_tilde) == expected

    def test_lambdify(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016.with_defaults(self.l_M_tilde)
        fl_M_pas_callable = lambdify(self.l_M_tilde, fl_M_pas)
        assert fl_M_pas_callable(1.0) == pytest.approx(0.0)

    @pytest.mark.skipif(numpy is None, reason='NumPy not installed')
    def test_lambdify_numpy(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016.with_defaults(self.l_M_tilde)
        fl_M_pas_callable = lambdify(self.l_M_tilde, fl_M_pas, 'numpy')
        l_M_tilde = numpy.array([0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5])
        expected = numpy.array([
            -0.0179917778,
            -0.0137393336,
            -0.0090783522,
            0.0,
            0.0176822155,
            0.0521224686,
            0.5043387669,
        ])
        numpy.testing.assert_allclose(fl_M_pas_callable(l_M_tilde), expected)

    @pytest.mark.skipif(jax is None, reason='JAX not installed')
    def test_lambdify_jax(self):
        fl_M_pas = FiberForceLengthPassiveDeGroote2016.with_defaults(self.l_M_tilde)
        fl_M_pas_callable = jax.jit(lambdify(self.l_M_tilde, fl_M_pas, 'jax'))
        l_M_tilde = jax.numpy.array([0.5, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5])
        expected = jax.numpy.array([
            -0.0179917778,
            -0.0137393336,
            -0.0090783522,
            0.0,
            0.0176822155,
            0.0521224686,
            0.5043387669,
        ])
        numpy.testing.assert_allclose(fl_M_pas_callable(l_M_tilde), expected)


class TestFiberForceLengthPassiveInverseDeGroote2016:

    @pytest.fixture(autouse=True)
    def _fiber_force_length_passive_arguments_fixture(self):
        self.fl_M_pas = Symbol('fl_M_pas')
        self.c0 = Symbol('c_0')
        self.c1 = Symbol('c_1')
        self.constants = (self.c0, self.c1)

    @staticmethod
    def test_class():
        assert issubclass(FiberForceLengthPassiveInverseDeGroote2016, Function)
        assert issubclass(FiberForceLengthPassiveInverseDeGroote2016, CharacteristicCurveFunction)
        assert FiberForceLengthPassiveInverseDeGroote2016.__name__ == 'FiberForceLengthPassiveInverseDeGroote2016'

    def test_instance(self):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *self.constants)
        assert isinstance(fl_M_pas_inv, FiberForceLengthPassiveInverseDeGroote2016)
        assert str(fl_M_pas_inv) == 'FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, c_0, c_1)'

    def test_doit(self):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *self.constants).doit()
        assert fl_M_pas_inv == self.c0*log(self.fl_M_pas*(exp(self.c1) - 1) + 1)/self.c1 + 1

    def test_doit_evaluate_false(self):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *self.constants).doit(evaluate=False)
        assert fl_M_pas_inv == self.c0*log(UnevaluatedExpr(self.fl_M_pas*(exp(self.c1) - 1)) + 1)/self.c1 + 1

    def test_with_defaults(self):
        constants = (
            Float('0.6'),
            Float('4.0'),
        )
        fl_M_pas_inv_manual = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *constants)
        fl_M_pas_inv_constants = FiberForceLengthPassiveInverseDeGroote2016.with_defaults(self.fl_M_pas)
        assert fl_M_pas_inv_manual == fl_M_pas_inv_constants

    def test_differentiate_wrt_fl_T(self):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *self.constants)
        expected = self.c0*(exp(self.c1) - 1)/(self.c1*(self.fl_M_pas*(exp(self.c1) - 1) + 1))
        assert fl_M_pas_inv.diff(self.fl_M_pas) == expected

    def test_differentiate_wrt_c0(self):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *self.constants)
        expected = log(self.fl_M_pas*(exp(self.c1) - 1) + 1)/self.c1
        assert fl_M_pas_inv.diff(self.c0) == expected

    def test_differentiate_wrt_c1(self):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *self.constants)
        expected = (
            self.c0*self.fl_M_pas*exp(self.c1)/(self.c1*(self.fl_M_pas*(exp(self.c1) - 1) + 1))
            - self.c0*log(self.fl_M_pas*(exp(self.c1) - 1) + 1)/self.c1**2
        )
        assert fl_M_pas_inv.diff(self.c1) == expected

    def test_inverse(self):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *self.constants)
        assert fl_M_pas_inv.inverse() is FiberForceLengthPassiveDeGroote2016

    def test_function_print_latex(self):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *self.constants)
        expected = r'\left( \operatorname{fl}^M_{pas} \right)^{-1} \left( fl_{M pas} \right)'
        assert LatexPrinter().doprint(fl_M_pas_inv) == expected

    def test_expression_print_latex(self):
        fl_T = FiberForceLengthPassiveInverseDeGroote2016(self.fl_M_pas, *self.constants)
        expected = r'\frac{c_{0} \log{\left(fl_{M pas} \left(e^{c_{1}} - 1\right) + 1 \right)}}{c_{1}} + 1'
        assert LatexPrinter().doprint(fl_T.doit()) == expected

    @pytest.mark.parametrize(
        'code_printer, expected',
        [
            (
                C89CodePrinter,
                '(1 + 0.14999999999999999*log(1 + 53.598150033144236*fl_M_pas))',
            ),
            (
                C99CodePrinter,
                '(1 + 0.14999999999999999*log(1 + 53.598150033144236*fl_M_pas))',
            ),
            (
                C11CodePrinter,
                '(1 + 0.14999999999999999*log(1 + 53.598150033144236*fl_M_pas))',
            ),
            (
                CXX98CodePrinter,
                '(1 + 0.14999999999999999*log(1 + 53.598150033144236*fl_M_pas))',
            ),
            (
                CXX11CodePrinter,
                '(1 + 0.14999999999999999*std::log(1 + 53.598150033144236*fl_M_pas))',
            ),
            (
                CXX17CodePrinter,
                '(1 + 0.14999999999999999*std::log(1 + 53.598150033144236*fl_M_pas))',
            ),
            (
                FCodePrinter,
                '      (1 + 0.15d0*log(1.0d0 + 53.5981500331442d0*fl_M_pas))',
            ),
            (
                OctaveCodePrinter,
                '(1 + 0.15*log(1 + 53.5981500331442*fl_M_pas))',
            ),
            (
                PythonCodePrinter,
                '(1 + 0.15*math.log(1 + 53.5981500331442*fl_M_pas))',
            ),
            (
                NumPyPrinter,
                '(1 + 0.15*numpy.log(1 + 53.5981500331442*fl_M_pas))',
            ),
            (
                SciPyPrinter,
                '(1 + 0.15*numpy.log(1 + 53.5981500331442*fl_M_pas))',
            ),
            (
                CuPyPrinter,
                '(1 + 0.15*cupy.log(1 + 53.5981500331442*fl_M_pas))',
            ),
            (
                JaxPrinter,
                '(1 + 0.15*jax.numpy.log(1 + 53.5981500331442*fl_M_pas))',
            ),
            (
                MpmathPrinter,
                '(1 + mpmath.mpf((0, 5404319552844595, -55, 53))*mpmath.log(1 '
                '+ mpmath.mpf((0, 942908627019595, -44, 50))*fl_M_pas))',
            ),
            (
                LambdaPrinter,
                '(1 + 0.15*math.log(1 + 53.5981500331442*fl_M_pas))',
            ),
        ]
    )
    def test_print_code(self, code_printer, expected):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016.with_defaults(self.fl_M_pas)
        assert code_printer().doprint(fl_M_pas_inv) == expected

    def test_derivative_print_code(self):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016.with_defaults(self.fl_M_pas)
        dfl_M_pas_inv_dfl_T = fl_M_pas_inv.diff(self.fl_M_pas)
        expected = '32.1588900198865/(214.392600132577*fl_M_pas + 4.0)'
        assert PythonCodePrinter().doprint(dfl_M_pas_inv_dfl_T) == expected

    def test_lambdify(self):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016.with_defaults(self.fl_M_pas)
        fl_M_pas_inv_callable = lambdify(self.fl_M_pas, fl_M_pas_inv)
        assert fl_M_pas_inv_callable(0.0) == pytest.approx(1.0)

    @pytest.mark.skipif(numpy is None, reason='NumPy not installed')
    def test_lambdify_numpy(self):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016.with_defaults(self.fl_M_pas)
        fl_M_pas_inv_callable = lambdify(self.fl_M_pas, fl_M_pas_inv, 'numpy')
        fl_M_pas = numpy.array([-0.01, 0.0, 0.01, 0.02, 0.05, 0.1])
        expected = numpy.array([
            0.8848253714,
            1.0,
            1.0643754386,
            1.1092744701,
            1.1954331425,
            1.2774998934,
        ])
        numpy.testing.assert_allclose(fl_M_pas_inv_callable(fl_M_pas), expected)

    @pytest.mark.skipif(jax is None, reason='JAX not installed')
    def test_lambdify_jax(self):
        fl_M_pas_inv = FiberForceLengthPassiveInverseDeGroote2016.with_defaults(self.fl_M_pas)
        fl_M_pas_inv_callable = jax.jit(lambdify(self.fl_M_pas, fl_M_pas_inv, 'jax'))
        fl_M_pas = jax.numpy.array([-0.01, 0.0, 0.01, 0.02, 0.05, 0.1])
        expected = jax.numpy.array([
            0.8848253714,
            1.0,
            1.0643754386,
            1.1092744701,
            1.1954331425,
            1.2774998934,
        ])
        numpy.testing.assert_allclose(fl_M_pas_inv_callable(fl_M_pas), expected)


class TestFiberForceLengthActiveDeGroote2016:

    @pytest.fixture(autouse=True)
    def _fiber_force_length_active_arguments_fixture(self):
        self.l_M_tilde = Symbol('l_M_tilde')
        self.c0 = Symbol('c_0')
        self.c1 = Symbol('c_1')
        self.c2 = Symbol('c_2')
        self.c3 = Symbol('c_3')
        self.c4 = Symbol('c_4')
        self.c5 = Symbol('c_5')
        self.c6 = Symbol('c_6')
        self.c7 = Symbol('c_7')
        self.c8 = Symbol('c_8')
        self.c9 = Symbol('c_9')
        self.c10 = Symbol('c_10')
        self.c11 = Symbol('c_11')
        self.constants = (
            self.c0, self.c1, self.c2, self.c3, self.c4, self.c5,
            self.c6, self.c7, self.c8, self.c9, self.c10, self.c11,
        )

    @staticmethod
    def test_class():
        assert issubclass(FiberForceLengthActiveDeGroote2016, Function)
        assert issubclass(FiberForceLengthActiveDeGroote2016, CharacteristicCurveFunction)
        assert FiberForceLengthActiveDeGroote2016.__name__ == 'FiberForceLengthActiveDeGroote2016'

    def test_instance(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        assert isinstance(fl_M_act, FiberForceLengthActiveDeGroote2016)
        assert str(fl_M_act) == (
            'FiberForceLengthActiveDeGroote2016(l_M_tilde, c_0, c_1, c_2, c_3, '
            'c_4, c_5, c_6, c_7, c_8, c_9, c_10, c_11)'
        )

    def test_doit(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants).doit()
        assert fl_M_act == (
            self.c0*exp(-(((self.l_M_tilde - self.c1)/(self.c2 + self.c3*self.l_M_tilde))**2)/2)
            + self.c4*exp(-(((self.l_M_tilde - self.c5)/(self.c6 + self.c7*self.l_M_tilde))**2)/2)
            + self.c8*exp(-(((self.l_M_tilde - self.c9)/(self.c10 + self.c11*self.l_M_tilde))**2)/2)
        )

    def test_doit_evaluate_false(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants).doit(evaluate=False)
        assert fl_M_act == (
            self.c0*exp(-((UnevaluatedExpr(self.l_M_tilde - self.c1)/(self.c2 + self.c3*self.l_M_tilde))**2)/2)
            + self.c4*exp(-((UnevaluatedExpr(self.l_M_tilde - self.c5)/(self.c6 + self.c7*self.l_M_tilde))**2)/2)
            + self.c8*exp(-((UnevaluatedExpr(self.l_M_tilde - self.c9)/(self.c10 + self.c11*self.l_M_tilde))**2)/2)
        )

    def test_with_defaults(self):
        constants = (
            Float('0.814'),
            Float('1.06'),
            Float('0.162'),
            Float('0.0633'),
            Float('0.433'),
            Float('0.717'),
            Float('-0.0299'),
            Float('0.2'),
            Float('0.1'),
            Float('1.0'),
            Float('0.354'),
            Float('0.0'),
        )
        fl_M_act_manual = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *constants)
        fl_M_act_constants = FiberForceLengthActiveDeGroote2016.with_defaults(self.l_M_tilde)
        assert fl_M_act_manual == fl_M_act_constants

    def test_differentiate_wrt_l_M_tilde(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = (
            self.c0*(
                self.c3*(self.l_M_tilde - self.c1)**2/(self.c2 + self.c3*self.l_M_tilde)**3
                + (self.c1 - self.l_M_tilde)/((self.c2 + self.c3*self.l_M_tilde)**2)
            )*exp(-(self.l_M_tilde - self.c1)**2/(2*(self.c2 + self.c3*self.l_M_tilde)**2))
            + self.c4*(
                self.c7*(self.l_M_tilde - self.c5)**2/(self.c6 + self.c7*self.l_M_tilde)**3
                + (self.c5 - self.l_M_tilde)/((self.c6 + self.c7*self.l_M_tilde)**2)
            )*exp(-(self.l_M_tilde - self.c5)**2/(2*(self.c6 + self.c7*self.l_M_tilde)**2))
            + self.c8*(
                self.c11*(self.l_M_tilde - self.c9)**2/(self.c10 + self.c11*self.l_M_tilde)**3
                + (self.c9 - self.l_M_tilde)/((self.c10 + self.c11*self.l_M_tilde)**2)
            )*exp(-(self.l_M_tilde - self.c9)**2/(2*(self.c10 + self.c11*self.l_M_tilde)**2))
        )
        assert fl_M_act.diff(self.l_M_tilde) == expected

    def test_differentiate_wrt_c0(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = exp(-(self.l_M_tilde - self.c1)**2/(2*(self.c2 + self.c3*self.l_M_tilde)**2))
        assert fl_M_act.doit().diff(self.c0) == expected

    def test_differentiate_wrt_c1(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = (
            self.c0*(self.l_M_tilde - self.c1)/(self.c2 + self.c3*self.l_M_tilde)**2
            *exp(-(self.l_M_tilde - self.c1)**2/(2*(self.c2 + self.c3*self.l_M_tilde)**2))
        )
        assert fl_M_act.diff(self.c1) == expected

    def test_differentiate_wrt_c2(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = (
            self.c0*(self.l_M_tilde - self.c1)**2/(self.c2 + self.c3*self.l_M_tilde)**3
            *exp(-(self.l_M_tilde - self.c1)**2/(2*(self.c2 + self.c3*self.l_M_tilde)**2))
        )
        assert fl_M_act.diff(self.c2) == expected

    def test_differentiate_wrt_c3(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = (
            self.c0*self.l_M_tilde*(self.l_M_tilde - self.c1)**2/(self.c2 + self.c3*self.l_M_tilde)**3
            *exp(-(self.l_M_tilde - self.c1)**2/(2*(self.c2 + self.c3*self.l_M_tilde)**2))
        )
        assert fl_M_act.diff(self.c3) == expected

    def test_differentiate_wrt_c4(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = exp(-(self.l_M_tilde - self.c5)**2/(2*(self.c6 + self.c7*self.l_M_tilde)**2))
        assert fl_M_act.diff(self.c4) == expected

    def test_differentiate_wrt_c5(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = (
            self.c4*(self.l_M_tilde - self.c5)/(self.c6 + self.c7*self.l_M_tilde)**2
            *exp(-(self.l_M_tilde - self.c5)**2/(2*(self.c6 + self.c7*self.l_M_tilde)**2))
        )
        assert fl_M_act.diff(self.c5) == expected

    def test_differentiate_wrt_c6(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = (
            self.c4*(self.l_M_tilde - self.c5)**2/(self.c6 + self.c7*self.l_M_tilde)**3
            *exp(-(self.l_M_tilde - self.c5)**2/(2*(self.c6 + self.c7*self.l_M_tilde)**2))
        )
        assert fl_M_act.diff(self.c6) == expected

    def test_differentiate_wrt_c7(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = (
            self.c4*self.l_M_tilde*(self.l_M_tilde - self.c5)**2/(self.c6 + self.c7*self.l_M_tilde)**3
            *exp(-(self.l_M_tilde - self.c5)**2/(2*(self.c6 + self.c7*self.l_M_tilde)**2))
        )
        assert fl_M_act.diff(self.c7) == expected

    def test_differentiate_wrt_c8(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = exp(-(self.l_M_tilde - self.c9)**2/(2*(self.c10 + self.c11*self.l_M_tilde)**2))
        assert fl_M_act.diff(self.c8) == expected

    def test_differentiate_wrt_c9(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = (
            self.c8*(self.l_M_tilde - self.c9)/(self.c10 + self.c11*self.l_M_tilde)**2
            *exp(-(self.l_M_tilde - self.c9)**2/(2*(self.c10 + self.c11*self.l_M_tilde)**2))
        )
        assert fl_M_act.diff(self.c9) == expected

    def test_differentiate_wrt_c10(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = (
            self.c8*(self.l_M_tilde - self.c9)**2/(self.c10 + self.c11*self.l_M_tilde)**3
            *exp(-(self.l_M_tilde - self.c9)**2/(2*(self.c10 + self.c11*self.l_M_tilde)**2))
        )
        assert fl_M_act.diff(self.c10) == expected

    def test_differentiate_wrt_c11(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = (
            self.c8*self.l_M_tilde*(self.l_M_tilde - self.c9)**2/(self.c10 + self.c11*self.l_M_tilde)**3
            *exp(-(self.l_M_tilde - self.c9)**2/(2*(self.c10 + self.c11*self.l_M_tilde)**2))
        )
        assert fl_M_act.diff(self.c11) == expected

    def test_function_print_latex(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = r'\operatorname{fl}^M_{act} \left( l_{M tilde} \right)'
        assert LatexPrinter().doprint(fl_M_act) == expected

    def test_expression_print_latex(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016(self.l_M_tilde, *self.constants)
        expected = (
            r'c_{0} e^{- \frac{\left(- c_{1} + l_{M tilde}\right)^{2}}{2 \left(c_{2} + c_{3} l_{M tilde}\right)^{2}}} '
            r'+ c_{4} e^{- \frac{\left(- c_{5} + l_{M tilde}\right)^{2}}{2 \left(c_{6} + c_{7} l_{M tilde}\right)^{2}}} '
            r'+ c_{8} e^{- \frac{\left(- c_{9} + l_{M tilde}\right)^{2}}{2 \left(c_{10} + c_{11} l_{M tilde}\right)^{2}}}'
        )
        assert LatexPrinter().doprint(fl_M_act.doit()) == expected

    @pytest.mark.parametrize(
        'code_printer, expected',
        [
            (
                C89CodePrinter,
                (
                    '(0.81399999999999995*exp(-19.051973784484073'
                    '*pow(l_M_tilde - 1.0600000000000001, 2)'
                    '/pow(0.39074074074074072*l_M_tilde + 1, 2)) '
                    '+ 0.433*exp(-12.499999999999998'
                    '*pow(l_M_tilde - 0.71699999999999997, 2)'
                    '/pow(l_M_tilde - 0.14949999999999999, 2)) '
                    '+ 0.10000000000000001*exp(-3.9899134986753491'
                    '*pow(l_M_tilde - 1.0, 2)))'
                ),
            ),
            (
                C99CodePrinter,
                (
                    '(0.81399999999999995*exp(-19.051973784484073'
                    '*pow(l_M_tilde - 1.0600000000000001, 2)'
                    '/pow(0.39074074074074072*l_M_tilde + 1, 2)) '
                    '+ 0.433*exp(-12.499999999999998'
                    '*pow(l_M_tilde - 0.71699999999999997, 2)'
                    '/pow(l_M_tilde - 0.14949999999999999, 2)) '
                    '+ 0.10000000000000001*exp(-3.9899134986753491'
                    '*pow(l_M_tilde - 1.0, 2)))'
                ),
            ),
            (
                C11CodePrinter,
                (
                    '(0.81399999999999995*exp(-19.051973784484073'
                    '*pow(l_M_tilde - 1.0600000000000001, 2)'
                    '/pow(0.39074074074074072*l_M_tilde + 1, 2)) '
                    '+ 0.433*exp(-12.499999999999998'
                    '*pow(l_M_tilde - 0.71699999999999997, 2)'
                    '/pow(l_M_tilde - 0.14949999999999999, 2)) '
                    '+ 0.10000000000000001*exp(-3.9899134986753491'
                    '*pow(l_M_tilde - 1.0, 2)))'
                ),
            ),
            (
                CXX98CodePrinter,
                (
                    '(0.81399999999999995*exp(-19.051973784484073'
                    '*std::pow(l_M_tilde - 1.0600000000000001, 2)'
                    '/std::pow(0.39074074074074072*l_M_tilde + 1, 2)) '
                    '+ 0.433*exp(-12.499999999999998'
                    '*std::pow(l_M_tilde - 0.71699999999999997, 2)'
                    '/std::pow(l_M_tilde - 0.14949999999999999, 2)) '
                    '+ 0.10000000000000001*exp(-3.9899134986753491'
                    '*std::pow(l_M_tilde - 1.0, 2)))'
                ),
            ),
            (
                CXX11CodePrinter,
                (
                    '(0.81399999999999995*std::exp(-19.051973784484073'
                    '*std::pow(l_M_tilde - 1.0600000000000001, 2)'
                    '/std::pow(0.39074074074074072*l_M_tilde + 1, 2)) '
                    '+ 0.433*std::exp(-12.499999999999998'
                    '*std::pow(l_M_tilde - 0.71699999999999997, 2)'
                    '/std::pow(l_M_tilde - 0.14949999999999999, 2)) '
                    '+ 0.10000000000000001*std::exp(-3.9899134986753491'
                    '*std::pow(l_M_tilde - 1.0, 2)))'
                ),
            ),
            (
                CXX17CodePrinter,
                (
                    '(0.81399999999999995*std::exp(-19.051973784484073'
                    '*std::pow(l_M_tilde - 1.0600000000000001, 2)'
                    '/std::pow(0.39074074074074072*l_M_tilde + 1, 2)) '
                    '+ 0.433*std::exp(-12.499999999999998'
                    '*std::pow(l_M_tilde - 0.71699999999999997, 2)'
                    '/std::pow(l_M_tilde - 0.14949999999999999, 2)) '
                    '+ 0.10000000000000001*std::exp(-3.9899134986753491'
                    '*std::pow(l_M_tilde - 1.0, 2)))'
                ),
            ),
            (
                FCodePrinter,
                (
                    '      (0.814d0*exp(-19.051973784484073d0*(l_M_tilde - 1.06d0)**2/(\n'
                    '     @ 0.39074074074074072d0*l_M_tilde + 1.0d0)**2) + 0.433d0*exp(\n'
                    '     @ -12.499999999999998d0*(l_M_tilde - 0.717d0)**2/(l_M_tilde -\n'
                    '     @ 0.14949999999999999d0)**2) + 0.1d0*exp(-3.9899134986753491d0*(\n'
                    '     @ l_M_tilde - 1.0d0)**2))'
                ),
            ),
            (
                OctaveCodePrinter,
                (
                    '(0.814*exp(-19.0519737844841*(l_M_tilde - 1.06).^2'
                    './(0.390740740740741*l_M_tilde + 1).^2) '
                    '+ 0.433*exp(-12.5*(l_M_tilde - 0.717).^2'
                    './(l_M_tilde - 0.1495).^2) '
                    '+ 0.1*exp(-3.98991349867535*(l_M_tilde - 1.0).^2))'
                ),
            ),
            (
                PythonCodePrinter,
                (
                    '(0.814*math.exp(-19.0519737844841*(l_M_tilde - 1.06)**2'
                    '/(0.390740740740741*l_M_tilde + 1)**2) '
                    '+ 0.433*math.exp(-12.5*(l_M_tilde - 0.717)**2'
                    '/(l_M_tilde - 0.1495)**2) '
                    '+ 0.1*math.exp(-3.98991349867535*(l_M_tilde - 1.0)**2))'
                ),
            ),
            (
                NumPyPrinter,
                (
                    '(0.814*numpy.exp(-19.0519737844841*(l_M_tilde - 1.06)**2'
                    '/(0.390740740740741*l_M_tilde + 1)**2) '
                    '+ 0.433*numpy.exp(-12.5*(l_M_tilde - 0.717)**2'
                    '/(l_M_tilde - 0.1495)**2) '
                    '+ 0.1*numpy.exp(-3.98991349867535*(l_M_tilde - 1.0)**2))'
                ),
            ),
            (
                SciPyPrinter,
                (
                    '(0.814*numpy.exp(-19.0519737844841*(l_M_tilde - 1.06)**2'
                    '/(0.390740740740741*l_M_tilde + 1)**2) '
                    '+ 0.433*numpy.exp(-12.5*(l_M_tilde - 0.717)**2'
                    '/(l_M_tilde - 0.1495)**2) '
                    '+ 0.1*numpy.exp(-3.98991349867535*(l_M_tilde - 1.0)**2))'
                ),
            ),
            (
                CuPyPrinter,
                (
                    '(0.814*cupy.exp(-19.0519737844841*(l_M_tilde - 1.06)**2'
                    '/(0.390740740740741*l_M_tilde + 1)**2) '
                    '+ 0.433*cupy.exp(-12.5*(l_M_tilde - 0.717)**2'
                    '/(l_M_tilde - 0.1495)**2) '
                    '+ 0.1*cupy.exp(-3.98991349867535*(l_M_tilde - 1.0)**2))'
                ),
            ),
            (
                JaxPrinter,
                (
                    '(0.814*jax.numpy.exp(-19.0519737844841*(l_M_tilde - 1.06)**2'
                    '/(0.390740740740741*l_M_tilde + 1)**2) '
                    '+ 0.433*jax.numpy.exp(-12.5*(l_M_tilde - 0.717)**2'
                    '/(l_M_tilde - 0.1495)**2) '
                    '+ 0.1*jax.numpy.exp(-3.98991349867535*(l_M_tilde - 1.0)**2))'
                ),
            ),
            (
                MpmathPrinter,
                (
                    '(mpmath.mpf((0, 7331860193359167, -53, 53))'
                    '*mpmath.exp(-mpmath.mpf((0, 5362653877279683, -48, 53))'
                    '*(l_M_tilde + mpmath.mpf((1, 2386907802506363, -51, 52)))**2'
                    '/(mpmath.mpf((0, 3519479708796943, -53, 52))*l_M_tilde + 1)**2) '
                    '+ mpmath.mpf((0, 7800234554605699, -54, 53))'
                    '*mpmath.exp(-mpmath.mpf((0, 7036874417766399, -49, 53))'
                    '*(l_M_tilde + mpmath.mpf((1, 6458161865649291, -53, 53)))**2'
                    '/(l_M_tilde + mpmath.mpf((1, 5386305154335113, -55, 53)))**2) '
                    '+ mpmath.mpf((0, 3602879701896397, -55, 52))'
                    '*mpmath.exp(-mpmath.mpf((0, 8984486472937407, -51, 53))'
                    '*(l_M_tilde + mpmath.mpf((1, 1, 0, 1)))**2))'
                ),
            ),
            (
                LambdaPrinter,
                (
                    '(0.814*math.exp(-19.0519737844841*(l_M_tilde - 1.06)**2'
                    '/(0.390740740740741*l_M_tilde + 1)**2) '
                    '+ 0.433*math.exp(-12.5*(l_M_tilde - 0.717)**2'
                    '/(l_M_tilde - 0.1495)**2) '
                    '+ 0.1*math.exp(-3.98991349867535*(l_M_tilde - 1.0)**2))'
                ),
            ),
        ]
    )
    def test_print_code(self, code_printer, expected):
        fl_M_act = FiberForceLengthActiveDeGroote2016.with_defaults(self.l_M_tilde)
        assert code_printer().doprint(fl_M_act) == expected

    def test_derivative_print_code(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016.with_defaults(self.l_M_tilde)
        fl_M_act_dl_M_tilde = fl_M_act.diff(self.l_M_tilde)
        expected = (
            '(0.79798269973507 - 0.79798269973507*l_M_tilde)'
            '*math.exp(-3.98991349867535*(l_M_tilde - 1.0)**2) '
            '+ (10.825*(0.717 - l_M_tilde)/(l_M_tilde - 0.1495)**2 '
            '+ 10.825*(l_M_tilde - 0.717)**2/(l_M_tilde - 0.1495)**3)'
            '*math.exp(-12.5*(l_M_tilde - 0.717)**2/(l_M_tilde - 0.1495)**2) '
            '+ (31.0166133211401*(1.06 - l_M_tilde)/(0.390740740740741*l_M_tilde + 1)**2 '
            '+ 13.6174190361677*(0.943396226415094*l_M_tilde - 1)**2'
            '/(0.390740740740741*l_M_tilde + 1)**3)'
            '*math.exp(-21.4067977442463*(0.943396226415094*l_M_tilde - 1)**2'
            '/(0.390740740740741*l_M_tilde + 1)**2)'
        )
        assert PythonCodePrinter().doprint(fl_M_act_dl_M_tilde) == expected

    def test_lambdify(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016.with_defaults(self.l_M_tilde)
        fl_M_act_callable = lambdify(self.l_M_tilde, fl_M_act)
        assert fl_M_act_callable(1.0) == pytest.approx(0.9941398866)

    @pytest.mark.skipif(numpy is None, reason='NumPy not installed')
    def test_lambdify_numpy(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016.with_defaults(self.l_M_tilde)
        fl_M_act_callable = lambdify(self.l_M_tilde, fl_M_act, 'numpy')
        l_M_tilde = numpy.array([0.0, 0.5, 1.0, 1.5, 2.0])
        expected = numpy.array([
            0.0018501319,
            0.0529122812,
            0.9941398866,
            0.2312431531,
            0.0069595432,
        ])
        numpy.testing.assert_allclose(fl_M_act_callable(l_M_tilde), expected)

    @pytest.mark.skipif(jax is None, reason='JAX not installed')
    def test_lambdify_jax(self):
        fl_M_act = FiberForceLengthActiveDeGroote2016.with_defaults(self.l_M_tilde)
        fl_M_act_callable = jax.jit(lambdify(self.l_M_tilde, fl_M_act, 'jax'))
        l_M_tilde = jax.numpy.array([0.0, 0.5, 1.0, 1.5, 2.0])
        expected = jax.numpy.array([
            0.0018501319,
            0.0529122812,
            0.9941398866,
            0.2312431531,
            0.0069595432,
        ])
        numpy.testing.assert_allclose(fl_M_act_callable(l_M_tilde), expected)


class TestFiberForceVelocityDeGroote2016:

    @pytest.fixture(autouse=True)
    def _muscle_fiber_force_velocity_arguments_fixture(self):
        self.v_M_tilde = Symbol('v_M_tilde')
        self.c0 = Symbol('c_0')
        self.c1 = Symbol('c_1')
        self.c2 = Symbol('c_2')
        self.c3 = Symbol('c_3')
        self.constants = (self.c0, self.c1, self.c2, self.c3)

    @staticmethod
    def test_class():
        assert issubclass(FiberForceVelocityDeGroote2016, Function)
        assert issubclass(FiberForceVelocityDeGroote2016, CharacteristicCurveFunction)
        assert FiberForceVelocityDeGroote2016.__name__ == 'FiberForceVelocityDeGroote2016'

    def test_instance(self):
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants)
        assert isinstance(fv_M, FiberForceVelocityDeGroote2016)
        assert str(fv_M) == 'FiberForceVelocityDeGroote2016(v_M_tilde, c_0, c_1, c_2, c_3)'

    def test_doit(self):
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants).doit()
        expected = (
            self.c0 * log((self.c1 * self.v_M_tilde + self.c2)
            + sqrt((self.c1 * self.v_M_tilde + self.c2)**2 + 1)) + self.c3
        )
        assert fv_M == expected

    def test_doit_evaluate_false(self):
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants).doit(evaluate=False)
        expected = (
            self.c0 * log((self.c1 * self.v_M_tilde + self.c2)
            + sqrt(UnevaluatedExpr(self.c1 * self.v_M_tilde + self.c2)**2 + 1)) + self.c3
        )
        assert fv_M == expected

    def test_with_defaults(self):
        constants = (
            Float('-0.318'),
            Float('-8.149'),
            Float('-0.374'),
            Float('0.886'),
        )
        fv_M_manual = FiberForceVelocityDeGroote2016(self.v_M_tilde, *constants)
        fv_M_constants = FiberForceVelocityDeGroote2016.with_defaults(self.v_M_tilde)
        assert fv_M_manual == fv_M_constants

    def test_differentiate_wrt_v_M_tilde(self):
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants)
        expected = (
            self.c0*self.c1
            /sqrt(UnevaluatedExpr(self.c1*self.v_M_tilde + self.c2)**2 + 1)
        )
        assert fv_M.diff(self.v_M_tilde) == expected

    def test_differentiate_wrt_c0(self):
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants)
        expected = log(
            self.c1*self.v_M_tilde + self.c2
            + sqrt(UnevaluatedExpr(self.c1*self.v_M_tilde + self.c2)**2 + 1)
        )
        assert fv_M.diff(self.c0) == expected

    def test_differentiate_wrt_c1(self):
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants)
        expected = (
            self.c0*self.v_M_tilde
            /sqrt(UnevaluatedExpr(self.c1*self.v_M_tilde + self.c2)**2 + 1)
        )
        assert fv_M.diff(self.c1) == expected

    def test_differentiate_wrt_c2(self):
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants)
        expected = (
            self.c0
            /sqrt(UnevaluatedExpr(self.c1*self.v_M_tilde + self.c2)**2 + 1)
        )
        assert fv_M.diff(self.c2) == expected

    def test_differentiate_wrt_c3(self):
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants)
        expected = Integer(1)
        assert fv_M.diff(self.c3) == expected

    def test_inverse(self):
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants)
        assert fv_M.inverse() is FiberForceVelocityInverseDeGroote2016

    def test_function_print_latex(self):
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants)
        expected = r'\operatorname{fv}^M \left( v_{M tilde} \right)'
        assert LatexPrinter().doprint(fv_M) == expected

    def test_expression_print_latex(self):
        fv_M = FiberForceVelocityDeGroote2016(self.v_M_tilde, *self.constants)
        expected = (
            r'c_{0} \log{\left(c_{1} v_{M tilde} + c_{2} + \sqrt{\left(c_{1} '
            r'v_{M tilde} + c_{2}\right)^{2} + 1} \right)} + c_{3}'
        )
        assert LatexPrinter().doprint(fv_M.doit()) == expected

    @pytest.mark.parametrize(
        'code_printer, expected',
        [
            (
                C89CodePrinter,
                '(0.88600000000000001 - 0.318*log(-8.1489999999999991*v_M_tilde '
                '- 0.374 + sqrt(1 + pow(-8.1489999999999991*v_M_tilde - 0.374, 2))))',
            ),
            (
                C99CodePrinter,
                '(0.88600000000000001 - 0.318*log(-8.1489999999999991*v_M_tilde '
                '- 0.374 + sqrt(1 + pow(-8.1489999999999991*v_M_tilde - 0.374, 2))))',
            ),
            (
                C11CodePrinter,
                '(0.88600000000000001 - 0.318*log(-8.1489999999999991*v_M_tilde '
                '- 0.374 + sqrt(1 + pow(-8.1489999999999991*v_M_tilde - 0.374, 2))))',
            ),
            (
                CXX98CodePrinter,
                '(0.88600000000000001 - 0.318*log(-8.1489999999999991*v_M_tilde '
                '- 0.374 + std::sqrt(1 + std::pow(-8.1489999999999991*v_M_tilde - 0.374, 2))))',
            ),
            (
                CXX11CodePrinter,
                '(0.88600000000000001 - 0.318*std::log(-8.1489999999999991*v_M_tilde '
                '- 0.374 + std::sqrt(1 + std::pow(-8.1489999999999991*v_M_tilde - 0.374, 2))))',
            ),
            (
                CXX17CodePrinter,
                '(0.88600000000000001 - 0.318*std::log(-8.1489999999999991*v_M_tilde '
                '- 0.374 + std::sqrt(1 + std::pow(-8.1489999999999991*v_M_tilde - 0.374, 2))))',
            ),
            (
                FCodePrinter,
                '      (0.886d0 - 0.318d0*log(-8.1489999999999991d0*v_M_tilde - 0.374d0 +\n'
                '     @ sqrt(1.0d0 + (-8.149d0*v_M_tilde - 0.374d0)**2)))',
            ),
            (
                OctaveCodePrinter,
                '(0.886 - 0.318*log(-8.149*v_M_tilde - 0.374 '
                '+ sqrt(1 + (-8.149*v_M_tilde - 0.374).^2)))',
            ),
            (
                PythonCodePrinter,
                '(0.886 - 0.318*math.log(-8.149*v_M_tilde - 0.374 '
                '+ math.sqrt(1 + (-8.149*v_M_tilde - 0.374)**2)))',
            ),
            (
                NumPyPrinter,
                '(0.886 - 0.318*numpy.log(-8.149*v_M_tilde - 0.374 '
                '+ numpy.sqrt(1 + (-8.149*v_M_tilde - 0.374)**2)))',
            ),
            (
                SciPyPrinter,
                '(0.886 - 0.318*numpy.log(-8.149*v_M_tilde - 0.374 '
                '+ numpy.sqrt(1 + (-8.149*v_M_tilde - 0.374)**2)))',
            ),
            (
                CuPyPrinter,
                '(0.886 - 0.318*cupy.log(-8.149*v_M_tilde - 0.374 '
                '+ cupy.sqrt(1 + (-8.149*v_M_tilde - 0.374)**2)))',
            ),
            (
                JaxPrinter,
                '(0.886 - 0.318*jax.numpy.log(-8.149*v_M_tilde - 0.374 '
                '+ jax.numpy.sqrt(1 + (-8.149*v_M_tilde - 0.374)**2)))',
            ),
            (
                MpmathPrinter,
                '(mpmath.mpf((0, 7980378539700519, -53, 53)) '
                '- mpmath.mpf((0, 5728578726015271, -54, 53))'
                '*mpmath.log(-mpmath.mpf((0, 4587479170430271, -49, 53))*v_M_tilde '
                '+ mpmath.mpf((1, 3368692521273131, -53, 52)) '
                '+ mpmath.sqrt(1 + (-mpmath.mpf((0, 4587479170430271, -49, 53))*v_M_tilde '
                '+ mpmath.mpf((1, 3368692521273131, -53, 52)))**2)))',
            ),
            (
                LambdaPrinter,
                '(0.886 - 0.318*math.log(-8.149*v_M_tilde - 0.374 '
                '+ sqrt(1 + (-8.149*v_M_tilde - 0.374)**2)))',
            ),
        ]
    )
    def test_print_code(self, code_printer, expected):
        fv_M = FiberForceVelocityDeGroote2016.with_defaults(self.v_M_tilde)
        assert code_printer().doprint(fv_M) == expected

    def test_derivative_print_code(self):
        fv_M = FiberForceVelocityDeGroote2016.with_defaults(self.v_M_tilde)
        dfv_M_dv_M_tilde = fv_M.diff(self.v_M_tilde)
        expected = '2.591382*(1 + (-8.149*v_M_tilde - 0.374)**2)**(-1/2)'
        assert PythonCodePrinter().doprint(dfv_M_dv_M_tilde) == expected

    def test_lambdify(self):
        fv_M = FiberForceVelocityDeGroote2016.with_defaults(self.v_M_tilde)
        fv_M_callable = lambdify(self.v_M_tilde, fv_M)
        assert fv_M_callable(0.0) == pytest.approx(1.002320622548512)

    @pytest.mark.skipif(numpy is None, reason='NumPy not installed')
    def test_lambdify_numpy(self):
        fv_M = FiberForceVelocityDeGroote2016.with_defaults(self.v_M_tilde)
        fv_M_callable = lambdify(self.v_M_tilde, fv_M, 'numpy')
        v_M_tilde = numpy.array([-1.0, -0.5, 0.0, 0.5])
        expected = numpy.array([
            0.0120816781,
            0.2438336294,
            1.0023206225,
            1.5850003903,
        ])
        numpy.testing.assert_allclose(fv_M_callable(v_M_tilde), expected)

    @pytest.mark.skipif(jax is None, reason='JAX not installed')
    def test_lambdify_jax(self):
        fv_M = FiberForceVelocityDeGroote2016.with_defaults(self.v_M_tilde)
        fv_M_callable = jax.jit(lambdify(self.v_M_tilde, fv_M, 'jax'))
        v_M_tilde = jax.numpy.array([-1.0, -0.5, 0.0, 0.5])
        expected = jax.numpy.array([
            0.0120816781,
            0.2438336294,
            1.0023206225,
            1.5850003903,
        ])
        numpy.testing.assert_allclose(fv_M_callable(v_M_tilde), expected)


class TestFiberForceVelocityInverseDeGroote2016:

    @pytest.fixture(autouse=True)
    def _tendon_force_length_inverse_arguments_fixture(self):
        self.fv_M = Symbol('fv_M')
        self.c0 = Symbol('c_0')
        self.c1 = Symbol('c_1')
        self.c2 = Symbol('c_2')
        self.c3 = Symbol('c_3')
        self.constants = (self.c0, self.c1, self.c2, self.c3)

    @staticmethod
    def test_class():
        assert issubclass(FiberForceVelocityInverseDeGroote2016, Function)
        assert issubclass(FiberForceVelocityInverseDeGroote2016, CharacteristicCurveFunction)
        assert FiberForceVelocityInverseDeGroote2016.__name__ == 'FiberForceVelocityInverseDeGroote2016'

    def test_instance(self):
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants)
        assert isinstance(fv_M_inv, FiberForceVelocityInverseDeGroote2016)
        assert str(fv_M_inv) == 'FiberForceVelocityInverseDeGroote2016(fv_M, c_0, c_1, c_2, c_3)'

    def test_doit(self):
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants).doit()
        assert fv_M_inv == (sinh((self.fv_M - self.c3)/self.c0) - self.c2)/self.c1

    def test_doit_evaluate_false(self):
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants).doit(evaluate=False)
        assert fv_M_inv == (sinh(UnevaluatedExpr(self.fv_M - self.c3)/self.c0) - self.c2)/self.c1

    def test_with_defaults(self):
        constants = (
            Float('-0.318'),
            Float('-8.149'),
            Float('-0.374'),
            Float('0.886'),
        )
        fv_M_inv_manual = FiberForceVelocityInverseDeGroote2016(self.fv_M, *constants)
        fv_M_inv_constants = FiberForceVelocityInverseDeGroote2016.with_defaults(self.fv_M)
        assert fv_M_inv_manual == fv_M_inv_constants

    def test_differentiate_wrt_fv_M(self):
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants)
        expected = cosh((self.fv_M - self.c3)/self.c0)/(self.c0*self.c1)
        assert fv_M_inv.diff(self.fv_M) == expected

    def test_differentiate_wrt_c0(self):
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants)
        expected = (self.c3 - self.fv_M)*cosh((self.fv_M - self.c3)/self.c0)/(self.c0**2*self.c1)
        assert fv_M_inv.diff(self.c0) == expected

    def test_differentiate_wrt_c1(self):
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants)
        expected = (self.c2 - sinh((self.fv_M - self.c3)/self.c0))/self.c1**2
        assert fv_M_inv.diff(self.c1) == expected

    def test_differentiate_wrt_c2(self):
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants)
        expected = -1/self.c1
        assert fv_M_inv.diff(self.c2) == expected

    def test_differentiate_wrt_c3(self):
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants)
        expected = -cosh((self.fv_M - self.c3)/self.c0)/(self.c0*self.c1)
        assert fv_M_inv.diff(self.c3) == expected

    def test_inverse(self):
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants)
        assert fv_M_inv.inverse() is FiberForceVelocityDeGroote2016

    def test_function_print_latex(self):
        fv_M_inv = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants)
        expected = r'\left( \operatorname{fv}^M \right)^{-1} \left( fv_{M} \right)'
        assert LatexPrinter().doprint(fv_M_inv) == expected

    def test_expression_print_latex(self):
        fv_M = FiberForceVelocityInverseDeGroote2016(self.fv_M, *self.constants)
        expected = r'\frac{- c_{2} + \sinh{\left(\frac{- c_{3} + fv_{M}}{c_{0}} \right)}}{c_{1}}'
        assert LatexPrinter().doprint(fv_M.doit()) == expected

    @pytest.mark.parametrize(
        'code_printer, expected',
        [
            (
                C89CodePrinter,
                '(-0.12271444348999878*(0.374 - sinh(3.1446540880503142*(fv_M '
                '- 0.88600000000000001))))',
            ),
            (
                C99CodePrinter,
                '(-0.12271444348999878*(0.374 - sinh(3.1446540880503142*(fv_M '
                '- 0.88600000000000001))))',
            ),
            (
                C11CodePrinter,
                '(-0.12271444348999878*(0.374 - sinh(3.1446540880503142*(fv_M '
                '- 0.88600000000000001))))',
            ),
            (
                CXX98CodePrinter,
                '(-0.12271444348999878*(0.374 - sinh(3.1446540880503142*(fv_M '
                '- 0.88600000000000001))))',
            ),
            (
                CXX11CodePrinter,
                '(-0.12271444348999878*(0.374 - std::sinh(3.1446540880503142'
                '*(fv_M - 0.88600000000000001))))',
            ),
            (
                CXX17CodePrinter,
                '(-0.12271444348999878*(0.374 - std::sinh(3.1446540880503142'
                '*(fv_M - 0.88600000000000001))))',
            ),
            (
                FCodePrinter,
                '      (-0.122714443489999d0*(0.374d0 - sinh(3.1446540880503142d0*(fv_M -\n'
                '     @ 0.886d0))))',
            ),
            (
                OctaveCodePrinter,
                '(-0.122714443489999*(0.374 - sinh(3.14465408805031*(fv_M '
                '- 0.886))))',
            ),
            (
                PythonCodePrinter,
                '(-0.122714443489999*(0.374 - math.sinh(3.14465408805031*(fv_M '
                '- 0.886))))',
            ),
            (
                NumPyPrinter,
                '(-0.122714443489999*(0.374 - numpy.sinh(3.14465408805031'
                '*(fv_M - 0.886))))',
            ),
            (
                SciPyPrinter,
                '(-0.122714443489999*(0.374 - numpy.sinh(3.14465408805031'
                '*(fv_M - 0.886))))',
            ),
            (
                CuPyPrinter,
                '(-0.122714443489999*(0.374 - cupy.sinh(3.14465408805031*(fv_M '
                '- 0.886))))',
            ),
            (
                JaxPrinter,
                '(-0.122714443489999*(0.374 - jax.numpy.sinh(3.14465408805031'
                '*(fv_M - 0.886))))',
            ),
            (
                MpmathPrinter,
                '(-mpmath.mpf((0, 8842507551592581, -56, 53))*(mpmath.mpf((0, '
                '3368692521273131, -53, 52)) - mpmath.sinh(mpmath.mpf((0, '
                '7081131489576251, -51, 53))*(fv_M + mpmath.mpf((1, '
                '7980378539700519, -53, 53))))))',
            ),
            (
                LambdaPrinter,
                '(-0.122714443489999*(0.374 - math.sinh(3.14465408805031*(fv_M '
                '- 0.886))))',
            ),
        ]
    )
    def test_print_code(self, code_printer, expected):
        fv_M_inv = FiberForceVelocityInverseDeGroote2016.with_defaults(self.fv_M)
        assert code_printer().doprint(fv_M_inv) == expected

    def test_derivative_print_code(self):
        fv_M_inv = FiberForceVelocityInverseDeGroote2016.with_defaults(self.fv_M)
        dfv_M_inv_dfv_M = fv_M_inv.diff(self.fv_M)
        expected = (
            '0.385894476383644*math.cosh(3.14465408805031*fv_M '
            '- 2.78616352201258)'
        )
        assert PythonCodePrinter().doprint(dfv_M_inv_dfv_M) == expected

    def test_lambdify(self):
        fv_M_inv = FiberForceVelocityInverseDeGroote2016.with_defaults(self.fv_M)
        fv_M_inv_callable = lambdify(self.fv_M, fv_M_inv)
        assert fv_M_inv_callable(1.0) == pytest.approx(-0.0009548832444487479)

    @pytest.mark.skipif(numpy is None, reason='NumPy not installed')
    def test_lambdify_numpy(self):
        fv_M_inv = FiberForceVelocityInverseDeGroote2016.with_defaults(self.fv_M)
        fv_M_inv_callable = lambdify(self.fv_M, fv_M_inv, 'numpy')
        fv_M = numpy.array([0.8, 0.9, 1.0, 1.1, 1.2])
        expected = numpy.array([
            -0.0794881459,
            -0.0404909338,
            -0.0009548832,
            0.043061991,
            0.0959484397,
        ])
        numpy.testing.assert_allclose(fv_M_inv_callable(fv_M), expected)

    @pytest.mark.skipif(jax is None, reason='JAX not installed')
    def test_lambdify_jax(self):
        fv_M_inv = FiberForceVelocityInverseDeGroote2016.with_defaults(self.fv_M)
        fv_M_inv_callable = jax.jit(lambdify(self.fv_M, fv_M_inv, 'jax'))
        fv_M = jax.numpy.array([0.8, 0.9, 1.0, 1.1, 1.2])
        expected = jax.numpy.array([
            -0.0794881459,
            -0.0404909338,
            -0.0009548832,
            0.043061991,
            0.0959484397,
        ])
        numpy.testing.assert_allclose(fv_M_inv_callable(fv_M), expected)


class TestCharacteristicCurveCollection:

    @staticmethod
    def test_valid_constructor():
        curves = CharacteristicCurveCollection(
            tendon_force_length=TendonForceLengthDeGroote2016,
            tendon_force_length_inverse=TendonForceLengthInverseDeGroote2016,
            fiber_force_length_passive=FiberForceLengthPassiveDeGroote2016,
            fiber_force_length_passive_inverse=FiberForceLengthPassiveInverseDeGroote2016,
            fiber_force_length_active=FiberForceLengthActiveDeGroote2016,
            fiber_force_velocity=FiberForceVelocityDeGroote2016,
            fiber_force_velocity_inverse=FiberForceVelocityInverseDeGroote2016,
        )
        assert curves.tendon_force_length is TendonForceLengthDeGroote2016
        assert curves.tendon_force_length_inverse is TendonForceLengthInverseDeGroote2016
        assert curves.fiber_force_length_passive is FiberForceLengthPassiveDeGroote2016
        assert curves.fiber_force_length_passive_inverse is FiberForceLengthPassiveInverseDeGroote2016
        assert curves.fiber_force_length_active is FiberForceLengthActiveDeGroote2016
        assert curves.fiber_force_velocity is FiberForceVelocityDeGroote2016
        assert curves.fiber_force_velocity_inverse is FiberForceVelocityInverseDeGroote2016

    @staticmethod
    @pytest.mark.skip(reason='kw_only dataclasses only valid in Python >3.10')
    def test_invalid_constructor_keyword_only():
        with pytest.raises(TypeError):
            _ = CharacteristicCurveCollection(
                TendonForceLengthDeGroote2016,
                TendonForceLengthInverseDeGroote2016,
                FiberForceLengthPassiveDeGroote2016,
                FiberForceLengthPassiveInverseDeGroote2016,
                FiberForceLengthActiveDeGroote2016,
                FiberForceVelocityDeGroote2016,
                FiberForceVelocityInverseDeGroote2016,
            )

    @staticmethod
    @pytest.mark.parametrize(
        'kwargs',
        [
            {'tendon_force_length': TendonForceLengthDeGroote2016},
            {
                'tendon_force_length': TendonForceLengthDeGroote2016,
                'tendon_force_length_inverse': TendonForceLengthInverseDeGroote2016,
                'fiber_force_length_passive': FiberForceLengthPassiveDeGroote2016,
                'fiber_force_length_passive_inverse': FiberForceLengthPassiveInverseDeGroote2016,
                'fiber_force_length_active': FiberForceLengthActiveDeGroote2016,
                'fiber_force_velocity': FiberForceVelocityDeGroote2016,
                'fiber_force_velocity_inverse': FiberForceVelocityInverseDeGroote2016,
                'extra_kwarg': None,
            },
        ]
    )
    def test_invalid_constructor_wrong_number_args(kwargs):
        with pytest.raises(TypeError):
            _ = CharacteristicCurveCollection(**kwargs)

    @staticmethod
    def test_instance_is_immutable():
        curves = CharacteristicCurveCollection(
            tendon_force_length=TendonForceLengthDeGroote2016,
            tendon_force_length_inverse=TendonForceLengthInverseDeGroote2016,
            fiber_force_length_passive=FiberForceLengthPassiveDeGroote2016,
            fiber_force_length_passive_inverse=FiberForceLengthPassiveInverseDeGroote2016,
            fiber_force_length_active=FiberForceLengthActiveDeGroote2016,
            fiber_force_velocity=FiberForceVelocityDeGroote2016,
            fiber_force_velocity_inverse=FiberForceVelocityInverseDeGroote2016,
        )
        with pytest.raises(AttributeError):
            curves.tendon_force_length = None
        with pytest.raises(AttributeError):
            curves.tendon_force_length_inverse = None
        with pytest.raises(AttributeError):
            curves.fiber_force_length_passive = None
        with pytest.raises(AttributeError):
            curves.fiber_force_length_passive_inverse = None
        with pytest.raises(AttributeError):
            curves.fiber_force_length_active = None
        with pytest.raises(AttributeError):
            curves.fiber_force_velocity = None
        with pytest.raises(AttributeError):
            curves.fiber_force_velocity_inverse = None
