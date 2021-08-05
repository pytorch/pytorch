import torch
from torch.testing._internal.common_utils import TestCase

class TestAtenPow(TestCase):
    def test_aten_pow_zero_negative_exponent(self):
        '''
        1. Testing a = int, b = int
        '''
        @torch.jit.script
        def fn_int_int(a: int, b: int):
            return a ** b
        # Existing correct behavior of aten::pow
        assert fn_int_int(2, 1) == 2 ** 1
        assert fn_int_int(2, 0) == 2 ** 0
        assert fn_int_int(2, -2) == 2 ** (-2)
        assert fn_int_int(-2, 2) == (-2) ** 2
        assert fn_int_int(-2, 0) == (-2) ** 0
        assert fn_int_int(-2, -2) == (-2) ** (-2)
        assert fn_int_int(-2, -1) == (-2) ** (-1)
        assert fn_int_int(0, 2) == 0 ** 1
        assert fn_int_int(0, 0) == 0 ** 0
        # zero base and negative exponent case that should trigger RunTimeError
        self.assertRaises(RuntimeError, fn_int_int, 0, -2)

        '''
        2. Testing a = int, b = float
        '''
        @torch.jit.script
        def fn_int_float(a: int, b: float):
            return a ** b
        # Existing correct behavior of aten::pow
        assert fn_int_float(2, 2.5) == 2 ** 2.5
        assert fn_int_float(2, -2.5) == 2 ** (-2.5)
        assert fn_int_float(2, -0.0) == 2 ** (-0.0)
        assert fn_int_float(2, 0.0) == 2 ** (0.0)
        assert fn_int_float(-2, 2.0) == (-2) ** 2.0
        assert fn_int_float(-2, -2.0) == (-2) ** (-2.0)
        assert fn_int_float(-2, -3.0) == (-2) ** (-3.0)
        assert fn_int_float(-2, -0.0) == (-2) ** (-0.0)
        assert fn_int_float(-2, 0.0) == (-2) ** (0.0)
        assert fn_int_float(0, 2.0) == 0 ** 2.0
        assert fn_int_float(0, 0.5) == 0 ** 0.5
        assert fn_int_float(0, 0.0) == 0 ** 0.0
        assert fn_int_float(0, -0.0) == 0 ** (-0.0)
        # zero base and negative exponent case that should trigger RunTimeError
        self.assertRaises(RuntimeError, fn_int_float, 0, -2.5)

        '''
        3. Testing a = float, b = int
        '''
        @torch.jit.script
        def fn_float_int(a: float, b: int):
            return a ** b
        assert fn_float_int(2.5, 2) == 2.5 ** 2
        assert fn_float_int(2.5, -2) == 2.5 ** (-2)
        assert fn_float_int(2.5, -0) == 2.5 ** (-0)
        assert fn_float_int(2.5, 0) == 2.5 ** 0
        assert fn_float_int(-2.5, 2) == 2.5 ** 2
        assert fn_float_int(-2.5, -2) == (-2.5) ** (-2)
        assert fn_float_int(-2.5, -3) == (-2.5) ** (-3)
        assert fn_float_int(-2.5, -0) == (-2.5) ** (-0)
        assert fn_float_int(-2.5, 0) == (-2.5) ** 0
        assert fn_float_int(0.0, 2) == 0 ** 2
        assert fn_float_int(0.0, 0) == 0 ** 0
        assert fn_float_int(0.0, -0) == 0 ** (-0)
        # zero base and negative exponent case that should trigger RunTimeError
        self.assertRaises(RuntimeError, fn_float_int, 0.0, -2)

        '''
        4. Testing a = float, b = float
        '''
        @torch.jit.script
        def fn_float_float(a: float, b: float):
            return a ** b
        assert fn_float_float(2.5, 2.0) == 2.5 ** 2.0
        assert fn_float_float(2.5, -2.0) == 2.5 ** (-2.0)
        assert fn_float_float(2.5, -0.0) == 2.5 ** (-0.0)
        assert fn_float_float(2.5, 0.0) == 2.5 ** 0.0
        assert fn_float_float(-2.5, 2.0) == 2.5 ** 2.0
        assert fn_float_float(-2.5, -2.0) == (-2.5) ** (-2.0)
        assert fn_float_float(-2.5, -3.0) == (-2.5) ** (-3.0)
        assert fn_float_float(-2.5, -0.0) == (-2.5) ** (-0.0)
        assert fn_float_float(-2.5, 0.0) == (-2.5) ** 0.0
        assert fn_float_float(0.0, 2.0) == 0.0 ** 2.0
        assert fn_float_float(0.0, 0.0) == 0.0 ** 0.0
        assert fn_float_float(0.0, -0.0) == 0.0 ** (-0.0)
        self.assertRaises(RuntimeError, fn_float_float, 0.0, -2.0)
