import torch
import numpy as np
import unittest

import math
import warnings
from itertools import product, chain
from functools import partial

from torch.testing._internal.common_utils import (TestCase, run_tests,
                                                  torch_to_numpy_dtype_dict, TEST_SCIPY)
from torch.testing._internal.common_device_type import (instantiate_device_type_tests,
                                                        dtypes, dtypesIfCPU, dtypesIfCUDA,
                                                        precisionOverride)

if TEST_SCIPY:
    from scipy import special


# Tests for "universal functions (ufuncs)" which functions that accept
# one or two tensors (unary or binary) and have common properties like
# being methods, supporting the out kwarg, supporting broadcasting and
# participating in type promotion.

# See NumPy's universal function documentation
# (https://numpy.org/doc/1.18/reference/ufuncs.html) for more details
# about the concept of ufuncs.


# Datastructures and helpers for returning a collection of appropriate
# tensors to use during testing.
_bool_finites = [True, False]
_unsigned_int_finites = [0, 1, 55, 127]
_int_finites = [0, -1, 1, -55, 55, -127, 127]
_float_finites = [0., -.001, .001, -.25, .25, -1., 1., -math.pi, math.pi]
_float_nonfinites = [float('inf'), float('-inf'), float('nan')]
_complex_finites = list(complex(x, y) for x, y in product(_float_finites, _float_finites))
_complex_nonfinites = list(complex(x, y) for x, y in (product(_float_nonfinites + [0], _float_nonfinites + [0])))

_nelems_large = 4620  # prime factors: 3, 4, 5, 7, 11
_nelems_medium = 66  # prime factors: 2, 3, 11
_nelems_small = 4  # prime factors: 2^2

_large_factors = (3, 4, 5, 7, 11)
_medium_factors = (2, 3, 11)

# let user supply their own range + special values but that's it
# for binary functions let them supply their own pairs if they want
# only test large if requested
# should create the defaults once? - no because testing inplace in many, anyway
def random_complex(rng, dtype, low, high):
    def generator(length):
        a = np.empty(length, dtype=dtype)
        a.real = rng.uniform(low, high, length)
        a.imag = rng.uniform(low, high, length)
        return a
    return generator

def random_bool(rng):
    def generator(length):
        a = rng.uniform(0, 1, length)
        return a > .5
    return generator

def random_reshape(a, factors, dims):
    reshape_dims = np.random.choice(factors, dims - 1, replace=False)
    reshape_dims = np.append(reshape_dims, -1)
    return np.reshape(a, reshape_dims)

# Makes the NumPy array representable in bfloat16
# Note: especially important when testing the bfloat16 variants of functions
#   like floor
def bfloat16ify(a):
    return torch.from_numpy(a).to(torch.bfloat16).to(torch.float32).numpy()

def pair_arrays_with_tensors(arrays, dtype, device):
    if dtype is torch.bfloat16:
        return ((torch.from_numpy(a).to(dtype=dtype, device=device), bfloat16ify(a)) for a in arrays)
    return ((torch.from_numpy(a).to(dtype=dtype, device=device), a) for a in arrays)

# Returns a list of pairs (torch.Tensor, np.ndarray) to be used as inputs
# to a Torch function and its NumPy or SciPy reference.
# The tensors include scalar tensors (0-dims), "small" 1D tensors, a "medium"
# 1D tensor, and a 3D tensor.
# The arrays are all views of the same underlying NumPy storage, which contains
# all the values specified in the vals kwarg, if given, or a set of default
# "interesting" finite and (if include_nonfinites is True) and nonfinite
# values. The array also contains random values drawn from [low, high).
# If low and high are not specified then dtype-specific defaults are used instead.
# An optional generator kwarg allows the caller to specify a function to
# generate the inputs directly. It should be a function that accepts a single
# argument, the length of a 1D array, and returns a 1D NumPy array of the
# same length.
def make_tensors(device, dtype, *, vals=None, generator=None, low=None, high=None,
                 include_nonfinites=True):
    rng = np.random.default_rng()
    np_dtype = torch_to_numpy_dtype_dict[dtype]

    # Special-cases bool
    if dtype is torch.bool:
        scalar_true = np.array(True)
        scalar_false = np.array(False)
        small = np.array((True, False))
        medium = random_bool(rng)(_nelems_medium)
        threeD = random_reshape(medium, _medium_factors, 3)
        return pair_arrays_with_tensors((scalar_true, scalar_false, small, medium, threeD), dtype, device)

    if vals is None:
        if dtype.is_floating_point:
            vals = _float_finites + _float_nonfinites if include_nonfinites else _float_finites
        elif dtype.is_complex:
            vals = _complex_finites + _complex_nonfinites if include_nonfinites else _complex_finites
        elif dtype is torch.uint8:
            vals = _unsigned_int_finites
        else:  # dtype is a signed integer type
            vals = _int_finites

    if generator is None:
        if dtype.is_floating_point:
            low = low if low is not None else -2
            high = high if high is not None else 2
            generator = partial(rng.uniform, low, high)
        elif dtype.is_complex:
            low = low if low is not None else -2
            high = high if high is not None else 2
            generator = random_complex(rng, np_dtype, low, high)
        elif dtype is torch.uint8:
            low = low if low is not None else 0
            high = high if high is not None else 128
            generator = partial(rng.integers, low, high)
        else:  # dtype is a signed integer type
            low = low if low is not None else -9
            high = high if high is not None else 10
            generator = partial(rng.integers, low, high)

    length = math.ceil((len(vals) + _nelems_medium) / _nelems_medium) * _nelems_medium
    a = generator(length).astype(np_dtype)
    for idx, val in enumerate(vals):
        a[idx] = val
    rng.shuffle(a)

    scalars = (arr.squeeze() for arr in np.split(a, length))

    small_splits = length / 2
    small_arrays = np.split(a, small_splits)

    # selects reshape dims
    threeD = random_reshape(a, _medium_factors, 3)

    rval = pair_arrays_with_tensors(chain(scalars, small_arrays, (a,), (threeD,)), dtype, device)

    return rval

_integral_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
_float_types = (torch.float32, torch.float64)
_float_types_plus_half = (torch.float16, torch.float32, torch.float64)
_float_types_plus_bfloat16 = (torch.bfloat16, torch.float32, torch.float64)
_complex_types = (torch.complex64, torch.complex128)
_float_and_complex_types = (torch.float32, torch.float64, torch.complex64, torch.complex128)
_float_and_complex_types_plus_half = (torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128)
_float_and_complex_types_plus_bfloat16 = (torch.bfloat16, torch.float32, torch.float64, torch.complex64, torch.complex128)
_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
          torch.float32, torch.float64)
_types_plus_half = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                    torch.float16, torch.float32, torch.float64)
_types_plus_bfloat16 = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                        torch.bfloat16, torch.float32, torch.float64)
_types_and_complex = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                      torch.float32, torch.float64, torch.complex64, torch.complex128)
_types_and_complex_plus_half = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                                torch.half, torch.float32, torch.float64,
                                torch.complex64, torch.complex128)
_types_and_complex_plus_bfloat16 = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
                                    torch.bfloat16, torch.float32, torch.float64,
                                    torch.complex64, torch.complex128)


class TestUnaryUfuncs(TestCase):
    exact_dtype = True
    _eps = 1e-5

    def _assertEqualHelper(self, actual, expected, dtype, exact_dtype):
        # Some NumPy functions return scalars, not arrays
        if not isinstance(expected, np.ndarray):
            self.assertEqual(actual.item(), expected)
        else:
            self.assertEqual(actual,
                             torch.from_numpy(expected),
                             exact_device=False,
                             exact_dtype=(exact_dtype if dtype is not torch.bfloat16 else False))

    def _test_unary_ufunc(self, fn, device, dtype, *,
                          ref=None, has_method=None, has_inplace=None,
                          has_out_kwarg=None,
                          low=None, high=None, exact_dtype=True,
                          include_nonfinites=True):
        assert ((low is None) == (high is None))

        if ref is None:
            assert isinstance(fn, str)
            ref = getattr(np, fn)

        if isinstance(fn, str):
            op = getattr(torch, fn)

        # Tests forward
        test_cases = make_tensors(device, dtype, low=low, high=high,
                                  include_nonfinites=include_nonfinites)
        for t, a in test_cases:
            actual = op(t)

            # Suppresses warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                expected = ref(a)

            self._assertEqualHelper(actual, expected, dtype, exact_dtype)

        # Tests method, inplace, and out variants
        # Note: uses the last tensor and array in test_cases
        if isinstance(fn, str):
            if has_method is None or has_method is True:
                op = getattr(torch.Tensor, fn)
                method_actual = op(t)
                self.assertEqual(method_actual, actual)
            if has_out_kwarg is None or has_out_kwarg is True:
                out = torch.empty_like(actual)
                op = partial(getattr(torch, fn), out=out)
                out_actual = op(t)
                self.assertEqual(out_actual, out)
                self.assertEqual(out, actual)
            if has_inplace is None or has_inplace is True:
                op = getattr(torch.Tensor, fn + "_")
                inplace_actual = op(t)
                self.assertEqual(inplace_actual, t)
                self.assertEqual(inplace_actual, actual, exact_dtype=False)

    @precisionOverride({torch.bfloat16: 1e-02})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_cos(self, device, dtype):
        self._test_unary_ufunc('cos', device, dtype)

    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypes(*_float_types)
    def test_cosh(self, device, dtype):
        self._test_unary_ufunc('cosh', device, dtype)

    @precisionOverride({torch.bfloat16: 1e-01})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_acos(self, device, dtype):
        self._test_unary_ufunc('acos', device, dtype,
                               ref=np.arccos, low=-1, high=1)

    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypes(*_float_types)
    def test_acosh(self, device, dtype):
        self._test_unary_ufunc('acosh', device, dtype, ref=np.arccosh)

    @precisionOverride({torch.bfloat16: 1e-02})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_sin(self, device, dtype):
        self._test_unary_ufunc('sin', device, dtype)

    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypes(*_float_types)
    def test_sinh(self, device, dtype):
        self._test_unary_ufunc('sinh', device, dtype)

    @precisionOverride({torch.bfloat16: 1e-01,
                        torch.float16: 1e-04})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_asin(self, device, dtype):
        self._test_unary_ufunc('asin', device, dtype,
                               ref=np.arcsin, low=-1, high=1)

    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypes(*_float_types)
    def test_asinh(self, device, dtype):
        self._test_unary_ufunc('asinh', device, dtype, ref=np.arcsinh)

    # Note: tan(bfloat16) is implemented on the CPU but incredibly innacurate
    #   near the limits of the domain, so it is not tested
    # See https://github.com/pytorch/pytorch/issues/41237
    # Note: CUDAtan(complex) doesn't handle nonfinite values properly
    # See https://github.com/pytorch/pytorch/issues/41244
    @dtypesIfCUDA(*_float_and_complex_types_plus_half)
    @dtypes(*_float_and_complex_types)
    def test_tan(self, device, dtype):
        self._test_unary_ufunc('tan', device, dtype,
                               low=(-math.pi / 2), high=(math.pi / 2),
                               include_nonfinites=(self.device_type == 'cpu'))

    @precisionOverride({torch.bfloat16: 1e-02})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_tanh(self, device, dtype):
        self._test_unary_ufunc('tanh', device, dtype)

    @precisionOverride({torch.bfloat16: 1e-02})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_atan(self, device, dtype):
        self._test_unary_ufunc('atan', device, dtype, ref=np.arctan)

    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypes(*_float_types)
    def test_atanh(self, device, dtype):
        self._test_unary_ufunc('atanh', device, dtype, ref=np.arctanh)

    @precisionOverride({torch.bfloat16: 1e-02})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_sqrt(self, device, dtype):
        self._test_unary_ufunc('sqrt', device, dtype)

    @precisionOverride({torch.float16: 1e-01})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypes(*_float_types)
    def test_rsqrt(self, device, dtype):
        self._test_unary_ufunc('rsqrt', device, dtype,
                               ref=lambda x: np.reciprocal(np.sqrt(x)))

    @precisionOverride({torch.bfloat16: 1e-01})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_log(self, device, dtype):
        self._test_unary_ufunc('log', device, dtype)

    @precisionOverride({torch.bfloat16: 1e-01})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_log2(self, device, dtype):
        self._test_unary_ufunc('log2', device, dtype)

    @precisionOverride({torch.bfloat16: 1e-02})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_log10(self, device, dtype):
        self._test_unary_ufunc('log10', device, dtype)

    @precisionOverride({torch.bfloat16: 1})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_log1p(self, device, dtype):
        self._test_unary_ufunc('log1p', device, dtype, low=-1 + self._eps, high=2)

    # Note: SciPy promotes half->float32 when running erf
    # See https://github.com/pytorch/pytorch/issues/41247
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @precisionOverride({torch.bfloat16: 1e-01,
                        torch.half: 1e-02})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_erf(self, device, dtype):
        self._test_unary_ufunc('erf', device, dtype,
                               ref=special.erf, exact_dtype=(dtype is not torch.float16))

    # Note: SciPy promotes half->float32 when running erfc
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @precisionOverride({torch.bfloat16: 1e-01,
                        torch.half: 1e-02})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_erfc(self, device, dtype):
        self._test_unary_ufunc('erfc', device, dtype,
                               ref=special.erfc, exact_dtype=False)

    @precisionOverride({torch.bfloat16: 1e-01})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_exp(self, device, dtype):
        self._test_unary_ufunc('exp', device, dtype)

    @precisionOverride({torch.bfloat16: 1e-01})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_expm1(self, device, dtype):
        self._test_unary_ufunc('expm1', device, dtype)

    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_floor(self, device, dtype):
        self._test_unary_ufunc('floor', device, dtype)

    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_ceil(self, device, dtype):
        self._test_unary_ufunc('ceil', device, dtype)

    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_trunc(self, device, dtype):
        self._test_unary_ufunc('trunc', device, dtype)

    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_round(self, device, dtype):
        self._test_unary_ufunc('round', device, dtype)

    @precisionOverride({torch.bfloat16: .5,
                        torch.float16: .2})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_rad2deg(self, device, dtype):
        self._test_unary_ufunc('rad2deg', device, dtype)

    @precisionOverride({torch.bfloat16: .5,
                        torch.float16: 1e-01})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_deg2rad(self, device, dtype):
        self._test_unary_ufunc('deg2rad', device, dtype)

    # Note: abs(complex) does not handle nonfinites properly on CPU
    # See https://github.com/pytorch/pytorch/issues/41246
    @dtypesIfCUDA(*_types_and_complex_plus_half)
    @dtypesIfCPU(*_types_and_complex_plus_bfloat16)
    @dtypes(*_float_types)
    def test_abs(self, device, dtype):
        include_nonfinites = (not dtype.is_complex or self.device_type=='cuda')
        self._test_unary_ufunc('abs', device, dtype, include_nonfinites=include_nonfinites)

    @dtypesIfCUDA(*_types_and_complex_plus_half)
    @dtypesIfCPU(*_types_and_complex_plus_bfloat16)
    @dtypes(*_float_types)
    def test_neg(self, device, dtype):
        self._test_unary_ufunc('neg', device, dtype, ref=np.negative)

    # Note: torch.sign(bfloat16) is incorrect for largish tensors on CPU
    # See https://github.com/pytorch/pytorch/issues/41238
    # Note: torch.sign does not handle nonfinite values properly
    # See https://github.com/pytorch/pytorch/issues/41245
    @dtypesIfCUDA(*_types_plus_half)
    @dtypesIfCPU(*_types)
    @dtypes(*_float_types)
    def test_sign(self, device, dtype):
        self._test_unary_ufunc('sign', device, dtype, include_nonfinites=False)

    @dtypesIfCUDA(*_types_and_complex_plus_half)
    @dtypesIfCPU(*_types_and_complex)
    @dtypes(*_float_types)
    def test_isfinite(self, device, dtype):
        self._test_unary_ufunc('isfinite', device, dtype,
                               has_out_kwarg=False, has_inplace=False)

    @dtypesIfCUDA(*_types_and_complex_plus_half)
    @dtypesIfCPU(*_types_and_complex)
    @dtypes(*_float_types)
    def test_isinf(self, device, dtype):
        self._test_unary_ufunc('isinf', device, dtype,
                               has_out_kwarg=False, has_inplace=False)

    @dtypesIfCUDA(*_types_and_complex_plus_half)
    @dtypesIfCPU(*_types_and_complex)
    @dtypes(*_float_types)
    def test_isnan(self, device, dtype):
        self._test_unary_ufunc('isnan', device, dtype,
                               has_out_kwarg=False, has_inplace=False)

    @dtypesIfCUDA(*_types_plus_half)
    @dtypesIfCPU(*_types)
    @dtypes(*_float_types)
    def test_logical_not(self, device, dtype):
        self._test_unary_ufunc('logical_not', device, dtype)

    @dtypes(*_integral_types)
    def test_bitwise_not(self, device, dtype):
        self._test_unary_ufunc('bitwise_not', device, dtype, ref=np.invert)

    @precisionOverride({torch.bfloat16: .5,
                        torch.float16: 1e-01})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypesIfCPU(*_float_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_frac(self, device, dtype):
        self._test_unary_ufunc('frac', device, dtype, ref=lambda x: np.fmod(x, 1))

    @precisionOverride({torch.float16: 1e-02})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypes(*_float_types)
    def test_square(self, device, dtype):
        self._test_unary_ufunc('square', device, dtype, has_out_kwarg=False)

    @precisionOverride({torch.float16: 1e-02})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypes(*_float_types)
    def test_reciprocal(self, device, dtype):
        self._test_unary_ufunc('reciprocal', device, dtype,
                               ref=np.reciprocal)

    @precisionOverride({torch.float16: 1e-02})
    @dtypesIfCUDA(*_float_types_plus_half)
    @dtypes(*_float_types)
    def test_sigmoid(self, device, dtype):
        self._test_unary_ufunc('sigmoid', device, dtype,
                               ref=lambda x: 1 / (1 + np.exp(-x)))

    @unittest.skipIf(True, "See issue https://github.com/pytorch/pytorch/issues/41240")
    @precisionOverride({torch.bfloat16: 1e-02})
    @dtypesIfCUDA(*_float_and_complex_types)
    @dtypesIfCPU(*_float_and_complex_types_plus_bfloat16)
    @dtypes(*_float_types)
    def test_angle(self, device, dtype):
        self._test_unary_ufunc('angle', device, dtype)

    @dtypes(*_complex_types)
    def test_conj(self, device, dtype):
        self._test_unary_ufunc('conj', device, dtype, has_inplace=False)


instantiate_device_type_tests(TestUnaryUfuncs, globals())

if __name__ == '__main__':
    run_tests()
