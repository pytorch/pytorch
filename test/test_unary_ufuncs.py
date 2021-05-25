import torch
import numpy as np

import warnings
import math
from itertools import product, chain
from numbers import Number
import random
import unittest

from torch._six import inf, nan
from torch.testing._internal.common_utils import (
    TestCase, run_tests, torch_to_numpy_dtype_dict, numpy_to_torch_dtype_dict,
    suppress_warnings, make_tensor, TEST_SCIPY, slowTest, skipIfNoSciPy, IS_WINDOWS)
from torch.testing._internal.common_methods_invocations import (
    unary_ufuncs, _NOTHING)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, ops, dtypes, onlyCPU, onlyOnCPUAndCUDA,
    onlyCUDA, dtypesIfCUDA, precisionOverride, skipCUDAIfRocm, dtypesIfCPU,
    OpDTypes)
from torch.testing import (
    floating_types_and, all_types_and_complex_and, floating_and_complex_types_and)

if TEST_SCIPY:
    import scipy

# Refer [scipy reference filter]
# Filter operators for which the reference function
# is available in the current environment (for reference_numerics tests).
reference_filtered_ops = list(filter(lambda op: op.ref is not _NOTHING, unary_ufuncs))

# Tests for unary "universal functions (ufuncs)" that accept a single
# tensor and have common properties like:
#   - they are elementwise functions
#   - the input shape is the output shape
#   - they typically have method and inplace variants
#   - they typically support the out kwarg
#   - they typically have NumPy or SciPy references

# See NumPy's universal function documentation
# (https://numpy.org/doc/1.18/reference/ufuncs.html) for more details
# about the concept of ufuncs.

# Functions tested here:
#

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

    if not dtype.is_complex:
        # Filter values based on Operators domain.
        # Note: Complex numbers don't belong to ordered field,
        #       so we don't filter for them.
        if domain[0] is not None:
            vals = list(filter(lambda x: x >= domain[0], vals))
        if domain[1] is not None:
            vals = list(filter(lambda x: x < domain[1], vals))

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


# TODO: port test_unary_out_op_mem_overlap
# TODO: add out= tests (different devices, dtypes, mismatched sizes,
#                       correct sizes, 0 size, broadcasted out)
# TODO: add test for inplace variants erroring on broadcasted inputs
class TestUnaryUfuncs(TestCase):
    exact_dtype = True

    # Tests bool tensor negation raises the correct error
    def test_neg_error_message(self, device):
        msg = ("Negation, the `\\-` operator, on a bool tensor is not supported."
               " If you are trying to invert a mask, use the `\\~` or"
               " `logical_not\\(\\)` operator instead.")

        t = torch.tensor((False, True), device=device)

        with self.assertRaisesRegex(RuntimeError, msg):
            torch.neg(t)

    @dtypes(*floating_types_and(torch.bfloat16, torch.half))
    @ops((_fn for _fn in unary_ufuncs if _fn.domain != (None, None)))
    def test_float_domains(self, device, dtype, op):
        if not op.supports_dtype(dtype, torch.device(device).type):
            raise unittest.SkipTest('unsupported dtype')

        eps = (1e-5, 1e-3, 1e-1, 1, 2, 10, 20, 50, 100)

        low, high = op.domain
        # NOTE: the following two loops are separated for readability
        if low is not None:
            low_tensor = torch.tensor(low, device=device, dtype=dtype)
            for epsilon in eps:
                lower_tensor = low_tensor - epsilon

                # Skips the test if the difference is not representable,
                #   which can occur if, for example, the difference is small
                #   and the dtype is imprecise (like bfloat16 is)
                if lower_tensor.item() == low_tensor.item():
                    continue

                result = op(lower_tensor)
                self.assertEqual(result.item(), float('nan'),
                                 msg=("input of {0} outside lower domain boundary"
                                      " {1} produced {2}, not nan!").format(lower_tensor.item(),
                                                                            low,
                                                                            result.item()))

        if high is not None:
            high_tensor = torch.tensor(high, device=device, dtype=dtype)
            for epsilon in eps:
                higher_tensor = high_tensor + epsilon

                # See above comment
                if higher_tensor.item() == high_tensor.item():
                    continue

                result = op(higher_tensor)
                self.assertEqual(result.item(), float('nan'),
                                 msg=("input of {0} outside upper domain boundary"
                                      " {1} produced {2}, not nan!").format(higher_tensor.item(),
                                                                            high,
                                                                            result.item()))

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
    #   values on given tensors
    def _test_reference_numerics(self, dtype, op, tensors, equal_nan=True):
        def _helper_reference_numerics(expected, actual, msg, exact_dtype, equal_nan=True):
            if not torch.can_cast(numpy_to_torch_dtype_dict[expected.dtype.type], dtype):
                exact_dtype = False

            if dtype in [torch.uint8, torch.int8, torch.bool]:
                # NOTE: For these dtypes, PyTorch computes in the default scalar type (float)
                # while NumPy computes in float16
                self.assertEqualHelper(actual, expected, msg, dtype=dtype,
                                       exact_dtype=exact_dtype, rtol=1e-3, atol=1e-2)
            elif dtype is torch.bfloat16:
                # Ref: https://github.com/pytorch/pytorch/blob/master/torch/testing/_internal/common_utils.py#L1149
                self.assertEqualHelper(actual, expected, msg, dtype=dtype,
                                       exact_dtype=exact_dtype, rtol=16e-3, atol=1e-5)
            else:
                self.assertEqualHelper(actual, expected, msg, dtype=dtype, equal_nan=equal_nan, exact_dtype=exact_dtype)

        for t in tensors:
            torch_kwargs, numpy_kwargs = op.sample_kwargs(t.device, dtype, t)
            if dtype is torch.bfloat16:
                a = t.cpu().to(torch.float32).numpy()
            else:
                a = t.cpu().numpy()

            actual = op(t, **torch_kwargs)
            expected = op.ref(a, **numpy_kwargs)

            # Crafts a custom error message for smaller, printable tensors
            if t.numel() < 10:
                msg = ("Failed to produce expected results! Input tensor was"
                       " {0}, torch result is {1}, and reference result is"
                       " {2}.").format(t, actual, expected)
            else:
                msg = None

            exact_dtype = True
            if isinstance(actual, torch.Tensor):
                _helper_reference_numerics(expected, actual, msg, exact_dtype, equal_nan)
            else:
                for x, y in zip(expected, actual):
                    # testing multi-outputs results
                    _helper_reference_numerics(x, y, msg, exact_dtype, equal_nan)

    # Tests that the function and its (array-accepting) reference produce the same
    #   values on a range of tensors, including empty tensors, scalar tensors,
    #   1D tensors and a large 2D tensor with interesting and extremal values
    #   and noncontiguities.
    @suppress_warnings
    @ops(reference_filtered_ops)
    def test_reference_numerics_normal(self, device, dtype, op):
        tensors = generate_numeric_tensors(device, dtype,
                                           domain=op.domain)
        self._test_reference_numerics(dtype, op, tensors)

    @suppress_warnings
    @ops(reference_filtered_ops, allowed_dtypes=floating_and_complex_types_and(
        torch.bfloat16, torch.half, torch.int8, torch.int16, torch.int32, torch.int64
    ))
    def test_reference_numerics_hard(self, device, dtype, op):
        if not op.handles_large_floats:
            raise self.skipTest("This op does not handle large values")

        tensors = generate_numeric_tensors_hard(device, dtype,
                                                domain=op.domain)
        self._test_reference_numerics(dtype, op, tensors)

    @suppress_warnings
    @ops(reference_filtered_ops,
         allowed_dtypes=floating_and_complex_types_and(torch.bfloat16, torch.half))
    def test_reference_numerics_extremal(self, device, dtype, op):
        handles_extremals = (op.handles_complex_extremals if
                             dtype in (torch.cfloat, torch.cdouble) else op.handles_extremals)
        if not handles_extremals:
            raise self.skipTest("This op does not handle extremal values")

        tensors = generate_numeric_tensors_extremal(device, dtype,
                                                    domain=op.domain)

        # https://github.com/pytorch/pytorch/issues/50749
        equal_nan = "relaxed" if device.startswith('cuda') else True

        self._test_reference_numerics(dtype, op, tensors, equal_nan)

    # Tests for testing (non)contiguity consistency

    @ops(unary_ufuncs)
    def test_contig_vs_every_other(self, device, dtype, op):
        contig = make_tensor((1026,), device=device, dtype=dtype,
                             low=op.domain[0], high=op.domain[1])
        non_contig = contig[::2]

        self.assertTrue(contig.is_contiguous())
        self.assertFalse(non_contig.is_contiguous())

        torch_kwargs, _ = op.sample_kwargs(device, dtype, non_contig)
        self.assertEqual(op(contig, **torch_kwargs)[::2], op(non_contig, **torch_kwargs))

    @ops(unary_ufuncs)
    def test_contig_vs_transposed(self, device, dtype, op):
        contig = make_tensor((789, 357), device=device, dtype=dtype,
                             low=op.domain[0], high=op.domain[1])
        non_contig = contig.T

        self.assertTrue(contig.is_contiguous())
        self.assertFalse(non_contig.is_contiguous())

        torch_kwargs, _ = op.sample_kwargs(device, dtype, contig)
        self.assertEqual(op(contig, **torch_kwargs).T, op(non_contig, **torch_kwargs))

    @ops(unary_ufuncs)
    def test_non_contig(self, device, dtype, op):
        shapes = [(5, 7), (1024,)]
        for shape in shapes:
            contig = make_tensor(shape, device, dtype,
                                 low=op.domain[0], high=op.domain[1])
            non_contig = torch.empty(shape + (2,), device=device, dtype=dtype)[..., 0]
            non_contig.copy_(contig)

            self.assertTrue(contig.is_contiguous())
            self.assertFalse(non_contig.is_contiguous())

            torch_kwargs, _ = op.sample_kwargs(device, dtype, contig)
            self.assertEqual(op(contig, **torch_kwargs), op(non_contig, **torch_kwargs))

    @ops(unary_ufuncs)
    def test_non_contig_index(self, device, dtype, op):
        contig = make_tensor((2, 2, 1, 2), device, dtype,
                             low=op.domain[0], high=op.domain[1])
        non_contig = contig[:, 1, ...]
        contig = non_contig.contiguous()

        self.assertTrue(contig.is_contiguous())
        self.assertFalse(non_contig.is_contiguous())

        torch_kwargs, _ = op.sample_kwargs(device, dtype, contig)
        self.assertEqual(op(contig, **torch_kwargs), op(non_contig, **torch_kwargs))

    @ops(unary_ufuncs)
    def test_non_contig_expand(self, device, dtype, op):
        shapes = [(1, 3), (1, 7), (5, 7)]
        for shape in shapes:
            contig = make_tensor(shape, device, dtype,
                                 low=op.domain[0], high=op.domain[1])
            non_contig = contig.clone().expand(3, -1, -1)

            self.assertTrue(contig.is_contiguous())
            self.assertFalse(non_contig.is_contiguous())

            torch_kwargs, _ = op.sample_kwargs(device, dtype, contig)
            contig = op(contig, **torch_kwargs)
            non_contig = op(non_contig, **torch_kwargs)
            for i in range(3):
                self.assertEqual(contig, non_contig[i],
                                 msg='non-contiguous expand[' + str(i) + ']')

    @ops(unary_ufuncs)
    def test_contig_size1(self, device, dtype, op):
        contig = make_tensor((5, 100), device, dtype,
                             low=op.domain[0], high=op.domain[1])
        contig = contig[:1, :50]
        contig2 = torch.empty(contig.size(), device=device, dtype=dtype)
        contig2.copy_(contig)

        self.assertTrue(contig.is_contiguous())
        self.assertTrue(contig2.is_contiguous())

        torch_kwargs, _ = op.sample_kwargs(device, dtype, contig)
        self.assertEqual(op(contig, **torch_kwargs), op(contig2, **torch_kwargs))

    @ops(unary_ufuncs)
    def test_contig_size1_large_dim(self, device, dtype, op):
        contig = make_tensor((5, 2, 3, 1, 4, 5, 3, 2, 1, 2, 3, 4), device, dtype,
                             low=op.domain[0], high=op.domain[1])
        contig = contig[:1, :, :, :, :, :, :, :, :, :, :, :]
        contig2 = torch.empty(contig.size(), device=device, dtype=dtype)
        contig2.copy_(contig)

        self.assertTrue(contig.is_contiguous())
        self.assertTrue(contig2.is_contiguous())

        torch_kwargs, _ = op.sample_kwargs(device, dtype, contig)
        self.assertEqual(op(contig, **torch_kwargs), op(contig2, **torch_kwargs))

    # Tests that computation on a multiple batches is the same as
    # per-batch computation.
    @ops(unary_ufuncs)
    def test_batch_vs_slicing(self, device, dtype, op):
        input = make_tensor((1024, 512), dtype=dtype, device=device,
                            low=op.domain[0], high=op.domain[1])

        torch_kwargs, _ = op.sample_kwargs(device, dtype, input)
        actual = op(input, **torch_kwargs)
        expected = torch.stack([op(slice, **torch_kwargs) for slice in input])

        self.assertEqual(actual, expected)

    def _test_out_arg(self, op, input, output, expected, **kwargs):
        if op.safe_casts_outputs:
            expect_fail = not torch.can_cast(expected.dtype, output.dtype)
        else:
            expect_fail = output.dtype != expected.dtype

        if expect_fail:
            with self.assertRaises(RuntimeError):
                op(input, out=output, **kwargs)
        else:
            res = op(input, out=output, **kwargs)
            self.assertTrue(res is output)
            self.assertEqual(output, expected.to(output.dtype))

    @ops(unary_ufuncs, dtypes=OpDTypes.supported)
    def test_out_arg_all_dtypes(self, device, dtype, op):
        if not op.supports_out:
            self.skipTest("Skipped! Op doesn't support out= kwarg.")

        input = make_tensor((64, 64), dtype=dtype, device=device,
                            low=op.domain[0], high=op.domain[1])
        torch_kwargs, _ = op.sample_kwargs(device, dtype, input)
        expected = op(input, **torch_kwargs)

        for out_dtype in all_types_and_complex_and(torch.bool, torch.half):
            out = torch.empty_like(input, dtype=out_dtype)
            self._test_out_arg(op, input, out, expected, **torch_kwargs)

    @dtypes(*(torch.testing.get_all_int_dtypes() + [torch.bool] +
              torch.testing.get_all_fp_dtypes(include_bfloat16=False)))
    def test_nan_to_num(self, device, dtype):
        for contiguous in [False, True]:
            x = make_tensor((64, 64), low=0., high=100., dtype=dtype, device=device)

            if dtype.is_floating_point:
                # Add extremal values.
                extremals = [float('nan'), float('inf'), -float('inf')]
                for idx, extremal in zip(torch.randint(0, 63, (3,)), extremals):
                    x[idx, :] = extremal

            if not contiguous:
                x = x.T

            # With args
            nan = random.random()
            posinf = random.random() * 5
            neginf = random.random() * 10

            self.compare_with_numpy(lambda x: x.nan_to_num(nan=nan, posinf=posinf),
                                    lambda x: np.nan_to_num(x, nan=nan, posinf=posinf),
                                    x)
            self.compare_with_numpy(lambda x: x.nan_to_num(posinf=posinf, neginf=neginf),
                                    lambda x: np.nan_to_num(x, posinf=posinf, neginf=neginf),
                                    x)

            # Out Variant
            out = torch.empty_like(x)
            result = torch.nan_to_num(x)
            torch.nan_to_num(x, out=out)
            self.assertEqual(result, out)

            result = torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
            torch.nan_to_num(x, out=out, nan=nan, posinf=posinf, neginf=neginf)
            self.assertEqual(result, out)

    @dtypes(torch.cfloat, torch.cdouble)
    def test_complex_edge_values(self, device, dtype):
        # sqrt Test Reference: https://github.com/pytorch/pytorch/pull/47424
        x = torch.tensor(0. - 1.0e+20j, dtype=dtype, device=device)
        self.compare_with_numpy(torch.sqrt, np.sqrt, x)
        # acos test reference: https://github.com/pytorch/pytorch/issue/42952
        # Skip on Windows, as CUDA acos  returns conjugate value
        # see https://github.com/pytorch/pytorch/issues/52299
        if not (IS_WINDOWS and dtype == torch.cdouble and "cuda" in device):
            self.compare_with_numpy(torch.acos, np.arccos, x)

        x = torch.tensor((-1.0e+60 if dtype == torch.cdouble else -1.0e+20) - 4988429.2j, dtype=dtype, device=device)
        self.compare_with_numpy(torch.sqrt, np.sqrt, x)

    @unittest.skipIf(not TEST_SCIPY, "Requires SciPy")
    @dtypes(torch.float, torch.double)
    def test_digamma_special(self, device, dtype):
        # Based on SciPy test for the following special values.
        # Reference:
        # https://github.com/scipy/scipy/blob/3a8a3a1d4657254a6611e77e9c28feafa26e6645/scipy/special/tests/test_digamma.py#L22
        euler = 0.57721566490153286
        dataset = [(0., -0.),
                   (1, -euler),
                   (0.5, -2 * math.log(2) - euler),
                   (1 / 3, -math.pi / (2 * math.sqrt(3)) - 3 * math.log(3) / 2 - euler),
                   (1 / 4, -math.pi / 2 - 3 * math.log(2) - euler),
                   (1 / 6, -math.pi * math.sqrt(3) / 2 - 2 * math.log(2) - 3 * math.log(3) / 2 - euler),
                   (1 / 8, -math.pi / 2 - 4 * math.log(2) -
                       (math.pi + math.log(2 + math.sqrt(2)) - math.log(2 - math.sqrt(2))) / math.sqrt(2) - euler)]
        x = torch.tensor(dataset, device=device, dtype=dtype)
        self.compare_with_numpy(torch.digamma, scipy.special.digamma, x)

    @unittest.skipIf(not TEST_SCIPY, "Requires SciPy")
    @dtypes(torch.float, torch.double)
    def test_digamma(self, device, dtype):
        # Tests pole behavior
        tensor = torch.tensor([-0.999999994, -1.999999994, -2.0000000111,
                               -100.99999994, 0.000000111, -1931.99999994,
                               -0.000000111, 0, -0, -1, -2, -931], dtype=dtype, device=device)
        self.compare_with_numpy(torch.digamma, scipy.special.digamma, tensor)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_fp_dtypes(include_half=True, include_bfloat16=False))
    def test_frexp(self, device, dtype):
        input = make_tensor((50, 50), device, dtype)
        mantissa, exponent = torch.frexp(input)
        np_mantissa, np_exponent = np.frexp(input.cpu().numpy())

        self.assertEqual(mantissa, np_mantissa)
        self.assertEqual(exponent, np_exponent)

        # torch.frexp returns exponent in int32 to be compatible with np.frexp
        self.assertTrue(exponent.dtype == torch.int32)
        self.assertTrue(torch_to_numpy_dtype_dict[exponent.dtype] == np_exponent.dtype)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_fp_dtypes(include_half=True, include_bfloat16=False))
    def test_frexp_out(self, device, dtype):
        input = make_tensor((50, 50), device, dtype)
        outputs = (
            (torch.empty_like(input), torch.empty_like(input, dtype=torch.int)),
            (torch.empty_like(input).transpose(0, 1), make_tensor((50, 50), device, torch.int, noncontiguous=True)),
        )
        for mantissa, exponent in outputs:
            torch.frexp(input, out=(mantissa, exponent))
            np_mantissa, np_exponent = np.frexp(input.cpu().numpy())
            self.assertEqual(mantissa, np_mantissa)
            self.assertEqual(exponent, np_exponent)


        # The warning is given when output tensors have wrong shape
        with warnings.catch_warnings(record=True) as w:
            mantissa = torch.empty((2, 2), device=device, dtype=dtype)
            exponent = torch.empty((5, 5), device=device, dtype=torch.int)

            torch.frexp(input, out=(mantissa, exponent))

            self.assertEqual(len(w), 2)
            self.assertTrue("An output with one or more elements was resized" in str(w[0].message))
            self.assertTrue("An output with one or more elements was resized" in str(w[1].message))

    @skipCUDAIfRocm
    def test_frexp_assert_raises(self, device):
        invalid_input_dtypes = torch.testing.get_all_int_dtypes() + \
            torch.testing.get_all_complex_dtypes() + \
            [torch.bool]
        for dtype in invalid_input_dtypes:
            input = make_tensor((50, 50), device, dtype)
            with self.assertRaisesRegex(RuntimeError, r"torch\.frexp\(\) only supports floating-point dtypes"):
                torch.frexp(input)

        for dtype in torch.testing.get_all_fp_dtypes(include_half=True, include_bfloat16=False):
            input = make_tensor((50, 50), device, dtype)

            dtypes = list(torch.testing.all_types_and_complex_and(torch.bool,
                                                                  torch.half,
                                                                  torch.bfloat16))
            dtypes.remove(dtype)
            for mantissa_dtype in dtypes:
                mantissa = torch.empty_like(input, dtype=mantissa_dtype)
                exponent = torch.empty_like(input, dtype=torch.int)
                with self.assertRaisesRegex(RuntimeError,
                                            r"torch\.frexp\(\) expects mantissa to have dtype .+ but got .+"):
                    torch.frexp(input, out=(mantissa, exponent))

            dtypes.append(dtype)
            dtypes.remove(torch.int)
            for exponent_dtype in dtypes:
                mantissa = torch.empty_like(input)
                exponent = torch.empty_like(input, dtype=exponent_dtype)
                with self.assertRaisesRegex(RuntimeError,
                                            r"torch\.frexp\(\) expects exponent to have int dtype but got .+"):
                    torch.frexp(input, out=(mantissa, exponent))

    def test_mvlgamma_argcheck(self, device):
        def run_test(d):
            input = torch.linspace((d - 2) / 2, 10, 10, device=device)
            torch.mvlgamma(input, d)

        with self.assertRaisesRegex(RuntimeError, r"All elements must be greater than \(p-1\)/2"):
            run_test(3)

    def test_polygamma_neg(self, device):
        with self.assertRaisesRegex(RuntimeError, r'polygamma\(n, x\) does not support negative n\.'):
            torch.polygamma(-1, torch.tensor([1.0, 2.0], device=device))

    # TODO resolve with opinfos
    @onlyCPU
    def test_op_invert(self, device):
        res = 0xffff - torch.arange(127, dtype=torch.int8)
        for dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            a = torch.arange(127, dtype=dtype)
            self.assertEqual(res.to(dtype), ~a)

        self.assertEqual(torch.tensor([True, False]), ~torch.tensor([False, True]))

        # test exceptions
        for dtype in (torch.half, torch.float, torch.double):
            a = torch.zeros(10, dtype=dtype)
            with self.assertRaises(TypeError):
                b = ~a

    @dtypes(torch.complex64, torch.complex128)
    def test_abs_angle_complex_to_float(self, device, dtype):
        # Constructs random complex values
        from random import random
        random_vals = []
        for multiplier in (-1, 1, -10, 10, -100, 100):
            for _ in range(10):
                random_vals.append(complex(random() * multiplier, random() * multiplier))

        for vals in (random_vals, []):
            a = np.array(vals, dtype=torch_to_numpy_dtype_dict[dtype])
            t = torch.tensor(vals, device=device, dtype=dtype)

            for fn_name in ('abs', 'angle'):
                torch_fn = getattr(torch, fn_name)
                np_fn = getattr(np, fn_name)

                # Tests function
                np_result = torch.from_numpy(np_fn(a))
                torch_result = torch_fn(t).cpu()
                self.assertEqual(np_result, torch_result, exact_dtype=True)

                # Tests float out
                float_dtype = torch.float32 if dtype is torch.complex64 else torch.float64
                np_float_out = np_fn(a).astype(torch_to_numpy_dtype_dict[float_dtype])
                float_out = torch.empty_like(t).float()
                torch_fn(t, out=float_out)
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(torch.from_numpy(np_float_out), float_out.cpu())

                # Tests float out (resized out)
                float_out = torch.empty(1, device=device, dtype=float_dtype)
                torch_fn(t, out=float_out)
                self.assertEqual(torch.from_numpy(np_float_out), float_out.cpu())

                # Tests complex out
                np_complex_out = np_fn(a)
                complex_out = torch.empty_like(t)
                torch_fn(t, out=complex_out)
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(torch.from_numpy(np_complex_out), complex_out.cpu())

                # Tests complex out (resized out)
                complex_out = torch.empty(0, device=device, dtype=dtype)
                torch_fn(t, out=complex_out)
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(torch.from_numpy(np_complex_out), complex_out.cpu())

                # Tests long out behavior (expected failure)
                long_out = torch.empty(0, device=device, dtype=torch.long)
                with self.assertRaises(RuntimeError):
                    torch_fn(t, out=long_out)

                # Tests inplace
                if fn_name == 'abs':
                    torch_inplace_method = getattr(torch.Tensor, fn_name + "_")
                    np_fn(a, out=a)
                    if dtype.is_complex:
                        with self.assertRaisesRegex(RuntimeError, "In-place abs is not supported for complex tensors."):
                            torch_inplace_method(t)
                        return
                    torch_inplace_method(t)
                    self.assertEqual(torch.from_numpy(a), t.cpu())

                # Note: angle does not have an in-place variant
                if fn_name == 'angle':
                    with self.assertRaises(AttributeError):
                        torch_inplace_method = getattr(torch.Tensor, fn_name + "_")

    def check_internal_mem_overlap(self, inplace_op, num_inputs,
                                   dtype, device,
                                   expected_failure=False):
        if isinstance(inplace_op, str):
            inplace_op = getattr(torch.Tensor, inplace_op)
        input = torch.randn(1, dtype=dtype, device=device).expand(3, 3)
        inputs = [input] + [torch.randn_like(input)
                            for i in range(num_inputs - 1)]
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, 'single memory location'):
                inplace_op(*inputs)
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, 'single memory location'):
                    inplace_op(*inputs)

    def unary_check_input_output_mem_overlap(self, data, sz, op,
                                             expected_failure=False):

        def _test(op, output, input):
            output_exp = torch.empty_like(output)
            op(input, out=output_exp)
            self.assertEqual(op(input, out=output), output_exp, msg=op.__name__)

        # output is identical to input:
        _test(op, output=data[0:sz], input=data[0:sz])
        # output and input are independent:
        _test(op, output=data[0:sz], input=data[sz:2 * sz])
        # output partially overlaps with input:
        if not expected_failure:
            with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                _test(op, data[0:sz], data[1:sz + 1])
        else:
            with self.assertRaises(AssertionError):
                with self.assertRaisesRegex(RuntimeError, 'unsupported operation'):
                    _test(op, data[0:sz], data[1:sz + 1])

    # TODO: run on non-native device types
    @dtypes(torch.double)
    def test_unary_out_op_mem_overlap(self, device, dtype):
        sz = 3
        doubles = torch.randn(2 * sz, dtype=dtype, device=device)
        positives = torch.randint(1, 100, (2 * sz,), device=device).double()
        ints = torch.randint(-100, 100, (2 * sz,), device=device)
        unary_mem_overlap_cases = [
            ("abs", doubles, True, True, 'cpu'),
            ("abs", doubles, True, True, 'cuda'),
            ("acos", doubles, True, True, 'cpu'),
            ("acos", doubles, True, True, 'cuda'),
            ("asin", doubles, True, True, 'cpu'),
            ("asin", doubles, True, True, 'cuda'),
            ("atan", doubles, True, True, 'cpu'),
            ("atan", doubles, True, True, 'cuda'),
            ("acosh", doubles, True, True, 'cpu'),
            ("acosh", doubles, True, True, 'cuda'),
            ("asinh", doubles, True, True, 'cpu'),
            ("asinh", doubles, True, True, 'cuda'),
            ("atanh", doubles, True, True, 'cpu'),
            ("atanh", doubles, True, True, 'cuda'),
            ("bitwise_not", ints, True, True, 'cpu'),
            ("bitwise_not", ints, True, True, 'cuda'),
            ("ceil", doubles, True, True, 'cpu'),
            ("ceil", doubles, True, True, 'cuda'),
            ("cos", doubles, True, True, 'cpu'),
            ("cos", doubles, True, True, 'cuda'),
            ("cosh", doubles, True, True, 'cpu'),
            ("cosh", doubles, True, True, 'cuda'),
            ("digamma", doubles, True, True, 'cpu'),
            ("erf", doubles, True, True, 'cpu'),
            ("erf", doubles, True, True, 'cuda'),
            ("erfc", doubles, True, True, 'cpu'),
            ("erfc", doubles, True, True, 'cuda'),
            ("erfinv", doubles, True, True, 'cpu'),
            ("erfinv", doubles, True, True, 'cuda'),
            ("exp", doubles, True, True, 'cpu'),
            ("exp", doubles, True, True, 'cuda'),
            ("exp2", doubles, True, True, 'cpu'),
            ("exp2", doubles, True, True, 'cuda'),
            ("expm1", doubles, True, True, 'cpu'),
            ("expm1", doubles, True, True, 'cuda'),
            ("floor", doubles, True, True, 'cpu'),
            ("floor", doubles, True, True, 'cuda'),
            ("frac", doubles, True, True, 'cpu'),
            ("frac", doubles, True, True, 'cuda'),
            ("i0", doubles, True, True, 'cpu'),
            ("i0", doubles, True, True, 'cuda'),
            ("log", positives, True, True, 'cpu'),
            ("log", positives, True, True, 'cuda'),
            ("log10", positives, True, True, 'cpu'),
            ("log10", positives, True, True, 'cuda'),
            ("log1p", positives, True, True, 'cpu'),
            ("log1p", positives, True, True, 'cuda'),
            ("log2", positives, True, True, 'cpu'),
            ("log2", positives, True, True, 'cuda'),
            ("neg", doubles, True, True, 'cpu'),
            ("neg", doubles, True, True, 'cuda'),
            ("reciprocal", doubles, True, True, 'cpu'),
            ("reciprocal", doubles, True, True, 'cuda'),
            ("round", doubles, True, True, 'cpu'),
            ("round", doubles, True, True, 'cuda'),
            ("rsqrt", positives, True, True, 'cpu'),
            ("rsqrt", positives, True, True, 'cuda'),
            ("sin", doubles, True, True, 'cpu'),
            ("sin", doubles, True, True, 'cuda'),
            ("sinh", doubles, True, True, 'cpu'),
            ("sinh", doubles, False, True, 'cuda'),
            ("sigmoid", doubles, True, True, 'cpu'),
            ("sigmoid", doubles, True, True, 'cuda'),
            ("logit", doubles, True, True, 'cpu'),
            ("logit", doubles, True, True, 'cuda'),
            ("sqrt", doubles, True, True, 'cpu'),
            ("sqrt", doubles, False, True, 'cuda'),
            ("tan", doubles, True, True, 'cpu'),
            ("tan", doubles, True, True, 'cuda'),
            ("tanh", doubles, True, True, 'cpu'),
            ("tanh", doubles, True, True, 'cuda'),
            ("trunc", doubles, True, True, 'cpu'),
            ("trunc", doubles, True, True, 'cuda')
        ]

        for (fn, inputs, has_input_output_mem_overlap_check,
             has_internal_mem_overlap_check, dev) in unary_mem_overlap_cases:
            if dev != device:
                continue
            out_fn = getattr(torch, fn)
            in_fn = getattr(torch.Tensor, fn + '_')

            self.unary_check_input_output_mem_overlap(inputs, sz, out_fn,
                                                      expected_failure=not has_input_output_mem_overlap_check)

            self.check_internal_mem_overlap(in_fn, 1, dtype, dev,
                                            expected_failure=not has_internal_mem_overlap_check)

    # TODO: opinfo hardshrink
    @onlyCPU
    @dtypes(torch.float, torch.double)
    def test_hardshrink(self, device, dtype):
        data = torch.tensor([1, 0.5, 0.3, 0.6], dtype=dtype, device=device).view(2, 2)
        self.assertEqual(torch.tensor([1, 0.5, 0, 0.6], dtype=dtype, device=device).view(2, 2),
                         data.hardshrink(0.3))
        self.assertEqual(torch.tensor([1, 0, 0, 0.6], dtype=dtype, device=device).view(2, 2),
                         data.hardshrink(0.5))

        # test default lambd=0.5
        self.assertEqual(data.hardshrink(), data.hardshrink(0.5))

        # test non-contiguous case
        self.assertEqual(torch.tensor([1, 0, 0.5, 0.6], dtype=dtype, device=device).view(2, 2),
                         data.t().hardshrink(0.3))

    @onlyCPU
    @dtypes(torch.float, torch.double)
    def test_hardshrink_edge_cases(self, device, dtype) -> None:
        def h(values, l_expected):
            for l, expected in l_expected.items():
                values_tensor = torch.tensor([float(v) for v in values],
                                             dtype=dtype, device=device)
                expected_tensor = torch.tensor([float(v) for v in expected],
                                               dtype=dtype, device=device)
                self.assertEqual(expected_tensor == values_tensor.hardshrink(l),
                                 torch.ones_like(values_tensor, dtype=torch.bool))

        def test_helper(min, max):
            h([0.0, min, -min, 0.1, -0.1, 1.0, -1.0, max, -max, inf, -inf],
              {0.0: [0.0, min, -min, 0.1, -0.1, 1.0, -1.0, max, -max, inf, -inf],
               min: [0.0, 0.0, 0.0, 0.1, -0.1, 1.0, -1.0, max, -max, inf, -inf],
               0.1: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, max, -max, inf, -inf],
               1.0: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, max, -max, inf, -inf],
               max: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, inf, -inf],
               inf: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})

        test_helper(torch.finfo(dtype).tiny, torch.finfo(dtype).max)

    @onlyCPU
    @slowTest
    @dtypes(torch.float)
    def test_exp_slow(self, device, dtype):
        # Test for https://github.com/pytorch/pytorch/issues/17271
        # This is pretty slow on my Macbook but it only takes a few
        # seconds on a beefy Xeon server
        a = torch.exp(torch.ones(2 ** 31, dtype=dtype, device=device))
        b = torch.exp(torch.ones(1, dtype=dtype, device=device))
        self.assertEqual(a, b.expand(2 ** 31))

    @precisionOverride({torch.bfloat16: 1e-2, torch.float: 0.0002, torch.double: 0.0002})
    @dtypesIfCUDA(torch.float, torch.double, torch.bfloat16)
    @dtypes(torch.float, torch.double)
    def test_hardswish(self, device, dtype):
        inputValues = [-1000, -4, -3, -2, 0, 2, 3, 4, 1000]
        expectedOutput = np.multiply(
            inputValues,
            np.minimum(np.maximum((np.add(inputValues, 3)), 0), 6) / 6.0)

        inputTensor = torch.tensor(inputValues, dtype=dtype, device=device)
        expectedOutputTensor = \
            torch.tensor(expectedOutput, dtype=dtype, device=device)

        # normal
        self.assertEqual(torch.nn.functional.hardswish(inputTensor),
                         expectedOutputTensor)

        # inplace
        inputTensorCpy = inputTensor.clone().detach()
        torch.nn.functional.hardswish(inputTensorCpy, inplace=True)
        self.assertEqual(inputTensorCpy, expectedOutputTensor)

    @precisionOverride({torch.bfloat16: 1e-2, torch.float: 0.0002, torch.double: 0.0002})
    @dtypesIfCUDA(torch.float, torch.double, torch.bfloat16)
    @dtypes(torch.float, torch.double)
    def test_hardsigmoid(self, device, dtype):
        inputValues = [-1000, -4, -3, -2, 0, 2, 3, 4, 1000]
        expectedOutput = np.minimum(np.maximum((np.add(inputValues, 3)), 0), 6) / 6.0

        inputTensor = torch.tensor(inputValues, dtype=dtype, device=device)

        # normal
        self.assertEqual(torch.nn.functional.hardsigmoid(inputTensor),
                         torch.tensor(expectedOutput, dtype=dtype, device=device))

        # inplace
        inputTensorCpy = inputTensor.clone().detach()
        self.assertEqual(torch.nn.functional.hardsigmoid(inputTensorCpy, inplace=True),
                         torch.tensor(expectedOutput, dtype=dtype, device=device))

    @precisionOverride({torch.bfloat16: 1e-2, torch.float: 0.0002, torch.double: 0.0002})
    @dtypesIfCUDA(torch.float, torch.double, torch.bfloat16)
    @dtypes(torch.float, torch.double)
    def test_hardsigmoid_backward(self, device, dtype):
        inputValues = [-3.0, 3.0, -2.0, 2.0, -6.0, 6.0]
        expectedValues = [0.0, 0.0, 1.0 / 6.0, 1.0 / 6.0, 0.0, 0.0]
        inputTensor = torch.tensor(inputValues, dtype=dtype, device=device).requires_grad_()
        expetedTensor = torch.tensor(expectedValues, dtype=dtype, device=device)
        out = torch.nn.functional.hardsigmoid(inputTensor)
        out.backward(torch.ones_like(inputTensor))
        self.assertEqual(inputTensor.grad, expetedTensor)

    @skipIfNoSciPy
    @dtypes(torch.float, torch.double)
    def test_silu(self, device, dtype):
        input_np = np.random.randn(5, 8)
        special_input = [[-1000, -1, -0.1, 0, 0.5, 1, 2, 1000]]
        input_np = np.concatenate((input_np, special_input), axis=0).astype(
            torch_to_numpy_dtype_dict[dtype])
        expected_output_np = input_np * scipy.special.expit(input_np)

        expected_output = torch.from_numpy(expected_output_np).to(device)
        expected_output_noncontig = expected_output.transpose(0, 1)

        atol = 1e-6
        rtol = 1e-6

        input = torch.from_numpy(input_np).clone().contiguous().to(device)
        self.assertEqual(torch.nn.functional.silu(input), expected_output,
                         atol=atol, rtol=rtol)
        self.assertEqual(torch.nn.functional.silu(input, inplace=True),
                         expected_output, atol=atol, rtol=rtol)

        input = torch.from_numpy(input_np).clone().to(device)
        input_noncontig = input.transpose(0, 1)
        self.assertEqual(torch.nn.functional.silu(input_noncontig),
                         expected_output_noncontig, atol=atol, rtol=rtol)
        self.assertEqual(torch.nn.functional.silu(
            input_noncontig, inplace=True), expected_output_noncontig,
            atol=atol, rtol=rtol)

    @skipIfNoSciPy
    @dtypes(torch.float, torch.double)
    def test_mish(self, device, dtype):
        input_np = np.random.randn(5, 8)
        special_input = [[-1000, -1, -0.1, 0, 0.5, 1, 2, 1000]]
        input_np = np.concatenate((input_np, special_input), axis=0).astype(
            torch_to_numpy_dtype_dict[dtype])
        expected_output_np = input_np * np.tanh(np.log1p(np.exp(input_np)))

        expected_output = torch.from_numpy(expected_output_np).to(device)
        expected_output_noncontig = expected_output.transpose(0, 1)

        atol = 1e-6
        rtol = 1e-6

        input = torch.from_numpy(input_np).clone().contiguous().to(device)
        self.assertEqual(torch.nn.functional.mish(input), expected_output,
                         atol=atol, rtol=rtol)
        self.assertEqual(torch.nn.functional.mish(input, inplace=True),
                         expected_output, atol=atol, rtol=rtol)

        input = torch.from_numpy(input_np).clone().to(device)
        input_noncontig = input.transpose(0, 1)
        self.assertEqual(torch.nn.functional.mish(input_noncontig),
                         expected_output_noncontig, atol=atol, rtol=rtol)
        self.assertEqual(torch.nn.functional.mish(
            input_noncontig, inplace=True), expected_output_noncontig,
            atol=atol, rtol=rtol)

    # do ops like threshold need a test_unary(_nonufunc) test suite?
    @onlyCPU
    @dtypes(*torch.testing.get_all_math_dtypes('cpu'))
    def test_threshold(self, device, dtype):
        if dtype != torch.uint8 and dtype != torch.float16 and not dtype.is_complex:
            # 100 is wide enough to use AVX2 instructions for all types
            x = torch.randn(100, dtype=torch.float, device=device).sign().to(dtype=dtype)
            y = torch.threshold(x, 0, 0)
            self.assertTrue(y.le(0).any())

    def _helper_test_igamma(self, loglo, loghi, device, dtype,
                            torch_fcn, scipy_fcn):
        exp1 = 2.71828182846
        vec1 = torch.logspace(loglo, loghi, steps=500, base=exp1,
                              dtype=torch.float64, device=device).unsqueeze(-1)
        vec1 = vec1.to(dtype)
        inputs = [
            (vec1, vec1.transpose(0, 1)),
            (vec1, vec1),  # for large number, it should approach 0.5
            (vec1, 0.5 * vec1),  # test for considerable ratio
            (vec1, 2.0 * vec1),
            (vec1[::2, :], vec1[::2, :]),  # contiguous/noncontiguous tests
            (vec1[::2, :], vec1[:vec1.shape[0] // 2, :]),
            (vec1[:vec1.shape[0] // 2, :], vec1[::2, :]),
        ]
        half_prec = dtype in [torch.bfloat16, torch.float16]
        for input0, input1 in inputs:
            actual = torch_fcn(input0, input1)
            if half_prec:
                input0 = input0.to(torch.float)
                input1 = input1.to(torch.float)
            expected = scipy_fcn(input0.cpu().numpy(), input1.cpu().numpy())
            expected = torch.from_numpy(expected).to(dtype)
            self.assertEqual(actual, expected)

    @skipCUDAIfRocm  # see issue https://github.com/pytorch/pytorch/issues/46531
    @dtypesIfCPU(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @dtypes(torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @onlyOnCPUAndCUDA
    def test_igamma_common(self, device, dtype):
        # test igamma for reasonable range of values
        loglo = -4  # approx 0.018
        loghi = 4  # approx 54.6
        self._helper_test_igamma(loglo, loghi, device, dtype,
                                 torch.igamma, scipy.special.gammainc)

    @dtypesIfCPU(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @dtypes(torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    @onlyOnCPUAndCUDA
    def test_igammac_common(self, device, dtype):
        # test igammac for reasonable range of values
        loglo = -4  # approx 0.018
        loghi = 4  # approx 54.6
        self._helper_test_igamma(loglo, loghi, device, dtype,
                                 torch.igammac, scipy.special.gammaincc)

    @dtypesIfCPU(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @dtypes(torch.float32, torch.float64)
    @onlyOnCPUAndCUDA
    def test_igamma_edge_cases(self, device, dtype):
        tkwargs = {"dtype": dtype, "device": device}
        infs = torch.zeros((3,), **tkwargs) + float("inf")
        zeros = torch.zeros((3,), **tkwargs)
        ones = torch.ones((3,), **tkwargs)
        zero_to_large = torch.tensor([0., 1., 1e3], **tkwargs)
        small_to_inf = torch.tensor([1e-3, 1., float("inf")], **tkwargs)
        nans = torch.zeros((3,), **tkwargs) + float("nan")
        inpouts = [
            # (a    ,    x),       out
            ((zeros, small_to_inf), ones),
            ((small_to_inf, zeros), zeros),
            ((infs, zero_to_large), zeros),
            ((zero_to_large, infs), ones),
            ((zeros, zeros), nans),
            ((infs, infs), nans),
            ((-small_to_inf, small_to_inf), nans),
        ]
        for inputs, output in inpouts:
            input0, input1 = inputs
            calc = torch.igamma(input0, input1)
            if torch.all(torch.isnan(output)):
                self.assertTrue(torch.all(torch.isnan(calc)))
            else:
                self.assertEqual(calc, output)

    @dtypesIfCPU(torch.float16, torch.bfloat16, torch.float32, torch.float64)
    @dtypes(torch.float32, torch.float64)
    @onlyOnCPUAndCUDA
    def test_igammac_edge_cases(self, device, dtype):
        tkwargs = {"dtype": dtype, "device": device}
        infs = torch.zeros((3,), **tkwargs) + float("inf")
        zeros = torch.zeros((3,), **tkwargs)
        ones = torch.ones((3,), **tkwargs)
        zero_to_large = torch.tensor([0., 1., 1e3], **tkwargs)
        small_to_inf = torch.tensor([1e-3, 1., float("inf")], **tkwargs)
        nans = torch.zeros((3,), **tkwargs) + float("nan")
        inpouts = [
            # (a    ,    x),       out
            ((zeros, small_to_inf), zeros),
            ((small_to_inf, zeros), ones),
            ((infs, zero_to_large), ones),
            ((zero_to_large, infs), zeros),
            ((zeros, zeros), nans),
            ((infs, infs), nans),
            ((-small_to_inf, small_to_inf), nans),
        ]
        for inputs, output in inpouts:
            input0, input1 = inputs
            calc = torch.igammac(input0, input1)
            if torch.all(torch.isnan(output)):
                self.assertTrue(torch.all(torch.isnan(calc)))
            else:
                self.assertEqual(calc, output)

    def _i0_helper(self, t):
        # Test by comparing to scipy
        dtype = t.dtype
        actual = torch.i0(t)
        if dtype is torch.bfloat16:
            t = t.to(torch.float32)
        expected = scipy.special.i0(t.cpu().numpy())
        # Casting down for dtype float16 is required since scipy upcasts to float32
        if dtype is torch.bfloat16 or dtype is torch.float16:
            expected = torch.from_numpy(expected).to(dtype)
        self.assertEqual(actual, expected)

    def _i0_range_helper(self, range, device, dtype):
        # i0 tests are broken up by the domain for which the function does not overflow for each dtype
        # This is done to ensure that the function performs well across all possible input values, without worrying
        # about inf or nan possibilities
        for r in (range, -range):
            t = torch.rand(1000, device=device).to(dtype) * r
            self._i0_helper(t)

    @dtypesIfCUDA(*torch.testing.get_all_fp_dtypes())
    @dtypes(torch.bfloat16, torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_i0_range1(self, device, dtype):
        # This tests the domain for i0 for which float16 does not overflow
        # The domain is (-13.25, 13.25)
        self._i0_range_helper(13.25, device, dtype)

    @dtypesIfCUDA(*torch.testing.get_all_fp_dtypes())
    @dtypes(torch.bfloat16, torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_i0_range2(self, device, dtype):
        # This tests the domain for i0 for which float32 and bfloat16 does not overflow
        # The domain is (-88.5, 88.5)
        self._i0_range_helper(88.5, device, dtype)

    @dtypes(torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_i0_range3(self, device, dtype):
        # This tests the domain for i0 for which float64 does not overflow
        # The domain is (-709.75, 709.75)
        self._i0_range_helper(709.75, device, dtype)

    @dtypesIfCUDA(*torch.testing.get_all_fp_dtypes())
    @dtypes(torch.bfloat16, torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_i0_special(self, device, dtype):
        t = torch.tensor([], device=device, dtype=dtype)
        self._i0_helper(t)

        t = torch.tensor([inf, -inf, nan], device=device, dtype=dtype)
        self.assertTrue(torch.i0(t).isnan().all())

    @dtypesIfCUDA(*torch.testing.get_all_fp_dtypes())
    @dtypes(torch.bfloat16, torch.float32, torch.float64)
    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_special_i0e_vs_scipy(self, device, dtype):
        def check_equal(t):
            # Test by comparing to scipy
            actual = torch.special.i0e(t)
            if dtype is torch.bfloat16:
                t = t.to(torch.float32)
            expected = scipy.special.i0e(t.cpu().numpy())

            # Casting down for dtype float16 is required since scipy upcasts to float32
            if dtype is torch.bfloat16 or dtype is torch.float16:
                expected = torch.from_numpy(expected).to(dtype)
            self.assertEqual(actual, expected)

        t = torch.tensor([], device=device, dtype=dtype)
        check_equal(t)

        range = (-1e7, 1e7)
        if dtype == torch.half:
            range = (-65000, 65000)

        t = torch.linspace(*range, int(1e4), device=device, dtype=dtype)
        check_equal(t)

        # NaN, inf, -inf are tested in reference_numerics tests.
        info = torch.finfo(dtype)
        min, max, eps, tiny = info.min, info.max, info.eps, info.tiny
        t = torch.tensor([min, max, eps, tiny], dtype=dtype, device=device)
        check_equal(t)

    # TODO: allow large opinfo values to be opted-into via metadata
    @dtypes(torch.long)
    def test_abs_big_number(self, device, dtype):
        bignumber = 2 ** 31 + 1
        res = torch.tensor([bignumber], device=device, dtype=dtype)
        self.assertGreater(res.abs()[0], 0)

    # TODO: add signed zero testing to opinfos
    @dtypes(torch.float, torch.double)
    def test_abs_signed_zero(self, device, dtype):
        # Both abs(0.0) and abs(-0.0) should result in 0.0
        size = 128 + 1  # pick a large enough number with remainder so that
        # both vectorized and nonvectorized op is tested
        inp = torch.zeros(size, device=device, dtype=dtype)
        inp[::2] = -0.0
        inp = inp.abs()
        for v in inp:
            self.assertGreater(math.copysign(1.0, v), 0.0)

    # TODO: update to compare against NumPy by rationalizing with OpInfo
    @onlyCUDA
    @dtypes(torch.float, torch.double)
    def test_abs_zero(self, device, dtype):
        # Both abs(0.0) and abs(-0.0) should result in 0.0
        abs_zeros = torch.tensor([0.0, -0.0], device=device, dtype=dtype).abs().tolist()
        for num in abs_zeros:
            self.assertGreater(math.copysign(1.0, num), 0.0)

    @dtypes(*torch.testing.get_all_fp_dtypes())
    def test_isfinite_isinf_isnan(self, device, dtype):
        vals = (-float('inf'), float('inf'), float('nan'), -1, 0, 1)

        self.compare_with_numpy(torch.isfinite, np.isfinite, vals, device, dtype)
        self.compare_with_numpy(torch.isinf, np.isinf, vals, device, dtype)
        self.compare_with_numpy(torch.isnan, np.isnan, vals, device, dtype)

    @dtypes(torch.int8, torch.int16, torch.int32, torch.int64)
    def test_isfinite_isinf_isnan_int(self, device, dtype):
        vals = (-1, 0, 1)

        self.compare_with_numpy(torch.isfinite, np.isfinite, vals, device, dtype)
        self.compare_with_numpy(torch.isinf, np.isinf, vals, device, dtype)
        self.compare_with_numpy(torch.isnan, np.isnan, vals, device, dtype)

    @dtypes(*(torch.testing.get_all_fp_dtypes()))
    def test_isposinf_isneginf_float(self, device, dtype):
        ops = ((torch.isposinf, np.isposinf), (torch.isneginf, np.isneginf))
        vals = (-float('inf'), float('inf'), float('nan'), -1, 0, 1)

        for torch_op, numpy_op in ops:
            if torch_op == torch.isposinf:
                target_vals = (0, 1, 0, 0, 0, 0)
            else:
                target_vals = (1, 0, 0, 0, 0, 0)

            t = torch.tensor(vals, device=device, dtype=dtype)
            # Manual check here as numpy does not support bfloat16
            if dtype == torch.bfloat16:
                self.assertEqual(torch_op(t),
                                 torch.tensor(target_vals, device=device, dtype=torch.bool))
            else:
                self.compare_with_numpy(torch_op, numpy_op, vals, device, dtype)

            # test the boolean tensor as the `out=` parameter
            out = torch.empty_like(t, dtype=torch.bool)
            t_target = torch.tensor(target_vals, device=device, dtype=torch.bool)
            torch_op(t, out=out)
            self.assertEqual(out, t_target)

    @dtypes(*(torch.testing.get_all_int_dtypes() + [torch.bool]))
    def test_isposinf_isneginf_int_and_bool(self, device, dtype):
        ops = ((torch.isposinf, np.isposinf), (torch.isneginf, np.isneginf))
        vals = (-1, 0, 1)

        for torch_op, numpy_op in ops:
            self.compare_with_numpy(torch_op, numpy_op, vals, device, dtype)

            # test the boolean tensor as the `out=` parameter
            t = torch.tensor(vals, device=device, dtype=dtype)
            out = torch.empty_like(t, dtype=torch.bool)
            t_target = torch.zeros_like(t, dtype=torch.bool)
            torch_op(t, out=out)
            self.assertEqual(out, t_target)

    @dtypes(torch.complex64, torch.complex128)
    def test_isposinf_isneginf_complex(self, device, dtype):
        torch_ops = (torch.isposinf, torch.isneginf)
        vals = (complex(0, float('inf')), complex(1, -float('inf')))
        t = torch.tensor(vals, device=device, dtype=dtype)
        out = torch.empty_like(t)

        for torch_op in torch_ops:
            with self.assertRaisesRegex(RuntimeError, 'does not support complex inputs'):
                torch_op(t)
            with self.assertRaisesRegex(RuntimeError, 'does not support complex inputs'):
                torch_op(t, out=out)

    @dtypes(*(torch.testing.get_all_dtypes(include_bool=False)))
    def test_isposinf_isneginf_non_boolean_output(self, device, dtype):
        # test non-boolean tensors as the `out=` parameters
        # boolean outputs are tested in the above testcases
        vals = (float('inf'), -float('inf'), 1.2)
        t = torch.tensor(vals, device=device)
        for torch_op in (torch.isposinf, torch.isneginf):
            out = torch.empty_like(t, dtype=dtype)
            with self.assertRaisesRegex(RuntimeError, 'does not support non-boolean outputs'):
                torch_op(t, out=out)

    @dtypes(torch.complex64, torch.complex128)
    def test_isfinite_isinf_isnan_complex(self, device, dtype):
        vals = (
            complex(-float('inf'), float('inf')),
            complex(-float('inf'), 0),
            complex(0, float('inf')),
            complex(float('inf'), float('nan')),
            complex(float('nan'), 0),
            complex(-1, 0),
            complex(0, 1)
        )

        self.compare_with_numpy(torch.isfinite, np.isfinite, vals, device, dtype)
        self.compare_with_numpy(torch.isinf, np.isinf, vals, device, dtype)
        self.compare_with_numpy(torch.isnan, np.isnan, vals, device, dtype)

    @dtypes(torch.complex64, torch.complex128)
    def test_isreal_complex(self, device, dtype):
        vals = (1, 1 + 1j, 2 + 0j, 3j, 2 - 1j, 2 - 0j)
        self.compare_with_numpy(torch.isreal, np.isreal, vals, device, dtype)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_isreal_noncomplex(self, device, dtype):
        vals = (1, 2, 3)
        # Manual check here since numpy doesn't support bfloat16
        result = torch.isreal(torch.tensor(vals, dtype=dtype))
        expected = torch.ones(result.size(), dtype=torch.bool, device=device)
        self.assertEqual(result, expected)

    @dtypes(torch.complex64)
    def test_isreal_nan_inf(self, device, dtype):
        vals = (
            complex(-float('inf'), float('inf')),
            complex(-float('inf'), 0),
            complex(0, float('inf')),
            complex(float('inf'), float('nan')),
            complex(float('nan'), 0),
            complex(-1, 0),
            complex(0, 1)
        )
        self.compare_with_numpy(torch.isreal, np.isreal, vals, device, dtype)

    @onlyCPU
    def test_isfinite_type(self, device):
        with self.assertRaises(TypeError):
            torch.isfinite(1)  # Parameter must be a tensor

    @onlyCPU
    def test_isinf_type(self, device):
        with self.assertRaises(TypeError):
            torch.isinf(1)  # Parameter must be a tensor

    def test_nonzero_empty(self, device):
        def assert_tuple_empty(tup, dim):
            self.assertEqual(dim, len(tup))
            for t in tup:
                self.assertEqual(torch.Size([0]), t.shape)

        x = torch.randn(0, 2, 0, 5, 0, device=device)
        y = torch.nonzero(x)
        z = torch.nonzero(x, as_tuple=True)

        self.assertEqual(0, y.numel())
        self.assertEqual(torch.Size([0, 5]), y.shape)
        assert_tuple_empty(z, 5)

        x = torch.tensor(0.5, device=device)
        y = torch.nonzero(x)
        # nonzero with as_tuple returns a
        # tuple of len 1 for a zero-dim tensor.
        # This is done to match Numpy behavior.
        z = torch.nonzero(x, as_tuple=True)
        self.assertEqual(1, len(z))
        self.assertEqual(torch.zeros(1, dtype=torch.long), z[0])

        x = torch.zeros((), device=device)
        y = torch.nonzero(x)
        z = torch.nonzero(x, as_tuple=True)
        self.assertEqual(torch.Size([0, 0]), y.shape)
        self.assertEqual(1, len(z))
        self.assertEqual(torch.empty(0, dtype=torch.long), z[0])

    # TODO: rationalize with exp OpInfo
    @dtypes(*(torch.testing.get_all_fp_dtypes(include_half=False) +
              torch.testing.get_all_complex_dtypes()))
    @dtypesIfCUDA(*(torch.testing.get_all_fp_dtypes(include_half=True) +
                    torch.testing.get_all_complex_dtypes()))
    def test_exp(self, device, dtype):
        for v in (2, -2) + ((1j, 1 + 1j) if dtype.is_complex else ()):
            a = torch.tensor(v, dtype=dtype, device=device) * torch.arange(18, device=device) / 3 * math.pi
            a = a.to(dtype)
            if dtype == torch.bfloat16:
                with self.assertRaises(TypeError):  # compare_with_numpy doesn't support bfloat16
                    self.compare_with_numpy(torch.exp, np.exp, a)
                return
            self.compare_with_numpy(torch.exp, np.exp, a)

            if dtype.is_complex:
                inf_real_zero_imag_in = torch.tensor(complex(float('inf'), 0), device=device, dtype=dtype)
                inf_real_zero_imag_out = torch.exp(inf_real_zero_imag_in).item()
                self.assertTrue(math.isinf(inf_real_zero_imag_out.real))
                if self.device_type == 'cpu':
                    pass
                    # These are commented out because it cannot be consistently reproduced.
                    # This is incorrect. It should be zero. Need fix!
                    # https://github.com/pytorch/pytorch/issues/40590
                    # self.assertNotEqual(inf_real_zero_imag_out.imag, 0)
                    # This is incorrect. They should equal. Need fix!
                    # https://github.com/pytorch/pytorch/issues/40590
                    # with self.assertRaises(AssertionError):
                    #     self.compare_with_numpy(torch.exp, np.exp, inf_real_zero_imag_in)
                else:
                    self.assertEqual(inf_real_zero_imag_out.imag, 0, atol=0, rtol=0)
                    self.compare_with_numpy(torch.exp, np.exp, inf_real_zero_imag_in)

                zero_real_inf_imag_in = torch.tensor(complex(0, float('inf')), device=device, dtype=dtype)
                zero_real_inf_imag_out = torch.exp(zero_real_inf_imag_in).item()
                self.assertTrue(math.isnan(zero_real_inf_imag_out.real))
                self.assertTrue(math.isnan(zero_real_inf_imag_out.imag))
                # Ensure we are notified when NumPy changes its behavior
                self.compare_with_numpy(torch.exp, np.exp, zero_real_inf_imag_in)

                inf_real_imag_in = torch.tensor(complex(float('inf'), float('inf')), device=device, dtype=dtype)
                inf_real_imag_out = torch.exp(inf_real_imag_in).item()
                if self.device_type == 'cpu':
                    pass
                    # This is incorrect. Need fix! https://github.com/pytorch/pytorch/issues/40590
                    # This is commented out because it cannot be consistently reproduced.
                    # with self.assertRaises(AssertionError):
                    #     self.compare_with_numpy(torch.exp, np.exp, inf_real_imag_in)
                else:
                    self.assertTrue(math.isinf(inf_real_imag_out.real))
                    self.assertTrue(math.isnan(inf_real_imag_out.imag))
                    self.compare_with_numpy(torch.exp, np.exp, inf_real_imag_in)

                inf_real_nan_imag_in = torch.tensor(complex(float('inf'), float('nan')), device=device, dtype=dtype)
                inf_real_nan_imag_out = torch.exp(inf_real_nan_imag_in).item()
                if self.device_type == 'cpu':
                    pass
                    # This is incorrect. It should be inf. Need fix! https://github.com/pytorch/pytorch/issues/40590
                    # This is commented out because it cannot be consistently reproduced.
                    # with self.assertRaises(AssertionError):
                    #     self.compare_with_numpy(torch.exp, np.exp, inf_real_nan_imag_in)
                else:
                    self.assertTrue(math.isinf(inf_real_nan_imag_out.real))
                    self.assertTrue(math.isnan(inf_real_nan_imag_out.imag))
                    self.compare_with_numpy(torch.exp, np.exp, inf_real_nan_imag_in)

                nan_real_inf_imag_in = torch.tensor(complex(float('nan'), float('inf')), device=device, dtype=dtype)
                nan_real_inf_imag_out = torch.exp(nan_real_inf_imag_in).item()
                self.assertTrue(math.isnan(nan_real_inf_imag_out.real))
                self.assertTrue(math.isnan(nan_real_inf_imag_out.imag))
                # Ensure we are notified when NumPy changes its behavior
                self.compare_with_numpy(torch.exp, np.exp, nan_real_inf_imag_in)


instantiate_device_type_tests(TestUnaryUfuncs, globals())

if __name__ == '__main__':
    run_tests()
