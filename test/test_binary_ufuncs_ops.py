import torch
import numpy as np

import math
from itertools import product, chain
from numbers import Number

from torch.testing._internal.common_utils import (
    TestCase, run_tests, torch_to_numpy_dtype_dict, numpy_to_torch_dtype_dict,
    make_tensor, )
from torch.testing._internal.common_methods_invocations import (
    binary_ufuncs)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, ops)
from torch.testing import (
    floating_and_complex_types_and)

# Interesting values and extremal values for different dtypes
_unsigned_int_vals = (0, 1, 55, 127)
_int_vals = (0, -1, 1, -55, 55, -127, 127, -128, 128)
_large_int_vals = (-1113, 1113, -10701, 10701)
_float_vals = (0.,
               -.001, .001,
               -.25, .25,
               -1., 1.,
               -math.pi / 2, math.pi / 2,
               -math.pi + .00001, math.pi - .00001,
               -math.pi, math.pi,
               -math.pi - .00001, math.pi + .00001)
_large_float16_vals = (-501, 501,
                       -1001.2, 1001.2,
                       -13437.7, 13437.7)
_large_float_vals = _large_float16_vals + (-4988429.2, 4988429.2, -1e20, 1e20)
_float_extremals = (float('inf'), float('-inf'), float('nan'))
_medium_length = 812
_large_size = (1029, 917)


# Returns generator of tensors of different sizes filled with values in domain
# and with intested region filled with `vals`. This will help test different code
# paths for the given vals
def generate_tensors_from_vals(vals, device, dtype, domain):
    offset = 63

    assert _large_size[1] > (_medium_length + offset)  # large tensor should be large enough
    assert len(vals) < _medium_length  # medium tensor should contain all vals
    assert _medium_length % 4 == 0  # ensure vectorized code coverage

    # Constructs the large tensor containing vals
    large_tensor = make_tensor(_large_size, device=device, dtype=dtype, low=domain[0], high=domain[1])

    # Inserts the vals at an odd place
    large_tensor[57][offset:offset + len(vals)] = torch.tensor(vals, device=device, dtype=dtype)

    # Takes a medium sized copy of the large tensor containing vals
    medium_tensor = large_tensor[57][offset:offset + _medium_length]

    # Constructs scalar tensors
    scalar_tensors = (t.squeeze() for t in torch.split(medium_tensor, 1))

    # Tensors with no elements
    empty_sizes = ((0,), (0, 3, 3), (1, 0, 5), (6, 0, 0, 0), (3, 0, 1, 0))
    empty_tensors = (torch.empty(size, device=device, dtype=dtype) for size in empty_sizes)

    return chain(empty_tensors, scalar_tensors, (medium_tensor,), (large_tensor,))


# [Note generate_numeric_tensors, generate_numeric_tensors_hard,
#  and generate_numeric_tensors_extremal]
#
# Returns an iterable of contiguous tensors with the same storage on the requested
#   device and with the requested dtype.
#
# This function is intended to test the non-vectorized and vectorized code
#   paths of unary functions, as well as their handling of odd tensor
#   sizes (like zero-dim tensors and tensors with zero elements).
#
# The iterable will include an empty tensor, tensors with no elements,
#   zero dim (scalar) tensors, small 1D tensors, a medium 1D tensor, and
#   a large 2D tensor.
#
# These tensors will include interesting values. The generate_numeric_tensors_hard
#   tests larger values (>500) and generate_numeric_tensors_extremal tests extremal
#   values like -inf, inf, and nan.
#
# The randomly generated values can be restricted by the domain
#   argument.
def generate_numeric_tensors(device, dtype, *,
                             domain=(None, None)):
    # Special-cases bool
    if dtype is torch.bool:
        tensors = (torch.empty(0, device=device, dtype=torch.bool),
                   torch.tensor(True, device=device),
                   torch.tensor(False, device=device),
                   torch.tensor((True, False), device=device),
                   make_tensor((_medium_length,), device=device, dtype=dtype, low=None, high=None),
                   make_tensor(_large_size, device=device, dtype=dtype, low=None, high=None))
        return tensors

    # Acquires dtype-specific vals
    if dtype.is_floating_point or dtype.is_complex:
        vals = _float_vals

        # Converts float -> complex vals if dtype is complex
        if dtype.is_complex:
            vals = tuple(complex(x, y) for x, y in product(vals, vals))
    elif dtype is torch.uint8:
        vals = _unsigned_int_vals
    else:  # dtypes is a signed integer type
        assert dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
        vals = _int_vals

    return generate_tensors_from_vals(vals, device, dtype, domain)


def generate_numeric_tensors_hard(device, dtype, *,
                                  domain=(None, None)):
    is_signed_integral = dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
    if not (dtype.is_floating_point or dtype.is_complex or is_signed_integral):
        return ()

    if dtype.is_floating_point:
        if dtype is torch.float16:
            # float16 has smaller range.
            vals = _large_float16_vals
        else:
            vals = _large_float_vals
    elif dtype.is_complex:
        vals = tuple(complex(x, y) for x, y in chain(product(_large_float_vals, _large_float_vals),
                                                     product(_float_vals, _large_float_vals),
                                                     product(_large_float_vals, _float_vals)))
    else:
        vals = _large_int_vals

    return generate_tensors_from_vals(vals, device, dtype, domain)


def generate_numeric_tensors_extremal(device, dtype, *,
                                      domain=(None, None)):
    if not (dtype.is_floating_point or dtype.is_complex):
        return ()

    vals = []
    if dtype.is_floating_point:
        vals = _float_extremals
    elif dtype.is_complex:
        vals = tuple(complex(x, y) for x, y in chain(product(_float_extremals, _float_extremals),
                                                     product(_float_vals, _float_extremals),
                                                     product(_float_extremals, _float_vals)))

    return generate_tensors_from_vals(vals, device, dtype, domain)

0

class TestBinaryUfuncs(TestCase):
    exact_dtype = True

    # Helper for comparing torch tensors and numpy arrays
    # TODO: should this or assertEqual also validate that strides are equal?
    def assertEqualHelper(self, actual, expected, msg, *, dtype, exact_dtype=True, **kwargs):
        assert isinstance(actual, torch.Tensor)

        # Some NumPy functions return scalars, not arrays
        if isinstance(expected, Number):
            self.assertEqual(actual.item(), expected, **kwargs)
        elif isinstance(expected, np.ndarray):
            # Handles exact dtype comparisons between arrays and tensors
            if exact_dtype:
                # Allows array dtype to be float32 when comparing with bfloat16 tensors
                #   since NumPy doesn't support the bfloat16 dtype
                # Also ops like scipy.special.erf, scipy.special.erfc, etc, promote float16
                # to float32
                if expected.dtype == np.float32:
                    assert actual.dtype in (torch.float16, torch.bfloat16, torch.float32)
                else:
                    assert expected.dtype == torch_to_numpy_dtype_dict[actual.dtype]

            self.assertEqual(actual,
                             torch.from_numpy(expected).to(actual.dtype),
                             msg,
                             exact_device=False,
                             **kwargs)
        else:
            self.assertEqual(actual, expected, msg, exact_device=False, **kwargs)

    # Tests that the function and its (array-accepting) reference produce the same
    # values on given tensors
    def _test_reference_numerics(self, dtype, op, tensors, equal_nan=True):
        for t in tensors:
            if dtype is torch.bfloat16:
                a = t.cpu().to(torch.float32).numpy()
            else:
                a = t.cpu().numpy()

            actual = op(t, t)
            expected = op.ref(a, a)

            # Crafts a custom error message for smaller, printable tensors
            if t.numel() < 10:
                msg = ("Failed to produce expected results! Input tensor was"
                       " {0}, torch result is {1}, and reference result is"
                       " {2}.").format(t, actual, expected)
            else:
                msg = None

            exact_dtype = True
            if not torch.can_cast(numpy_to_torch_dtype_dict[expected.dtype.type], dtype):
                exact_dtype = False

                if dtype in [torch.uint8, torch.int8, torch.bool]:
                    # NOTE: For these dtypes, PyTorch computes in the default scalar type (float)
                    # while NumPy computes in float16
                    self.assertEqualHelper(actual, expected, msg, dtype=dtype,
                                           exact_dtype=exact_dtype, rtol=1e-3, atol=1e-2)
                    continue

            self.assertEqualHelper(actual, expected, msg, dtype=dtype, equal_nan=equal_nan, exact_dtype=exact_dtype)

    # Tests that the function and its (array-accepting) reference produce the same
    #   values on a range of tensors, including empty tensors, scalar tensors,
    #   1D tensors and a large 2D tensor with interesting and extremal values
    #   and discontiguities.
    @ops(binary_ufuncs)
    def test_reference_numerics_normal(self, device, dtype, op):
        tensors = generate_numeric_tensors(device, dtype,
                                           domain=op.domain)
        self._test_reference_numerics(dtype, op, tensors)

    @ops(binary_ufuncs, allowed_dtypes=floating_and_complex_types_and(
        torch.bfloat16, torch.half, torch.int8, torch.int16, torch.int32, torch.int64
    ))
    def _test_reference_numerics_hard(self, device, dtype, op):
        if not op.handles_large_floats:
            raise self.skipTest("This op does not handle large values")

        tensors = generate_numeric_tensors_hard(device, dtype,
                                                domain=op.domain)
        self._test_reference_numerics(dtype, op, tensors)

    @ops(binary_ufuncs, allowed_dtypes=floating_and_complex_types_and(torch.bfloat16, torch.half))
    def _test_reference_numerics_extremal(self, device, dtype, op):
        handles_extremals = (op.handles_complex_extremals if
                             dtype in (torch.cfloat, torch.cdouble) else op.handles_extremals)
        if not handles_extremals:
            raise self.skipTest("This op does not handle extremal values")

        tensors = generate_numeric_tensors_extremal(device, dtype,
                                                    domain=op.domain)

        # https://github.com/pytorch/pytorch/issues/50749
        equal_nan = "relaxed" if device.startswith('cuda') else True

        self._test_reference_numerics(dtype, op, tensors, equal_nan)

instantiate_device_type_tests(TestBinaryUfuncs, globals())

if __name__ == '__main__':
    run_tests()
