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
    TestCase, run_tests, torch_to_numpy_dtype_dict, suppress_warnings,
    IS_MACOS, make_tensor, TEST_SCIPY, slowTest, skipIfNoSciPy)
from torch.testing._internal.common_methods_invocations import (
    unary_ufuncs)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, ops, dtypes, onlyCPU, onlyOnCPUAndCUDA,
    onlyCUDA, dtypesIfCUDA, precisionOverride, skipCUDAIfRocm, dtypesIfCPU,
    OpDTypes)
from torch.testing import (
    floating_types_and, integral_types, all_types_and_complex_and, floating_types)

if TEST_SCIPY:
    import scipy

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
_large_float_vals = (-501, 501,
                     -1001.2, 1001.2,
                     -13437.7, 13437.7,
                     -4988429.2, 4988429.2,
                     -1e20, 1e20)
_float_extremals = (float('inf'), float('-inf'), float('nan'))


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
# These tensors will include interesting values. If include_large_values
#   is true they will include larger values (>500), too, and if
#   include_extremal_values is true they will include extremal values
#   like -inf, inf, and nan.
#
# The randomly generated values can be constracted by the domain
#   argument.
def generate_numeric_tensors(device, dtype, *,
                             domain=(None, None),
                             include_large_values=True,
                             include_extremal_values=True):
    medium_length = 812
    large_size = (1029, 917)
    offset = 63

    assert large_size[1] > (medium_length + offset)
    assert medium_length % 4 == 0

    # Special-cases bool
    if dtype is torch.bool:
        tensors = (torch.empty(0, device=device, dtype=torch.bool),
                   torch.tensor(True, device=device),
                   torch.tensor(False, device=device),
                   torch.tensor((True, False), device=device),
                   make_tensor((medium_length,), device=device, dtype=dtype, low=None, high=None),
                   make_tensor(large_size, device=device, dtype=dtype, low=None, high=None))
        return tensors

    # Acquires dtype-specific vals
    if dtype.is_floating_point or dtype.is_complex:
        large_vals = _large_float_vals if include_large_values else tuple()
        extremals = _float_extremals if include_extremal_values else tuple()
        vals = _float_vals + large_vals + extremals

        # Converts float -> complex vals if dtype is complex
        if dtype.is_complex:
            vals = tuple(complex(x, y) for x, y in product(vals, vals))
    elif dtype is torch.uint8:
        vals = _unsigned_int_vals
    else:  # dtypes is a signed integer type
        assert dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
        large_vals = _large_int_vals if include_large_values else tuple()
        vals = _int_vals + large_vals

    assert len(vals) < medium_length

    # Constructs the large tensor containing vals
    large_tensor = make_tensor(large_size, device=device, dtype=dtype, low=domain[0], high=domain[1])

    # Inserts the vals at an odd place
    large_tensor[57][offset:offset + len(vals)] = torch.tensor(vals, device=device, dtype=dtype)

    # Takes a medium sized copy of the large tensor containing vals
    medium_tensor = large_tensor[57][offset:offset + medium_length]

    # Constructs small tensors (4 elements)
    small_tensors = (t for t in torch.split(medium_tensor, 4))

    # Constructs scalar tensors
    scalar_tensors = (t.squeeze() for t in torch.split(medium_tensor, 1))

    # Tensors with no elements
    empty_sizes = ((0,), (0, 3, 3), (1, 0, 5), (6, 0, 0, 0), (3, 0, 1, 0))
    empty_tensors = (torch.empty(size, device=device, dtype=dtype) for size in empty_sizes)

    return chain(empty_tensors, scalar_tensors, small_tensors, (medium_tensor,), (large_tensor,))

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

    # Tests that fn == method == inplace == jit on a simple single tensor input
    # TODO: should this jitting the method and inplace variants, too?
    @ops(unary_ufuncs)
    def test_variant_consistency(self, device, dtype, op):
        def _fn(t):
            return op(t)

        t = make_tensor((5, 5), device, dtype, low=op.domain[0], high=op.domain[1])
        expected = op(t)

        for alt, inplace in ((op.get_method(), False), (op.get_inplace(), True),
                             (torch.jit.script(_fn), False)):
            if alt is None:
                with self.assertRaises(RuntimeError):
                    alt(t.clone())

            if inplace and op.promotes_integers_to_float and dtype in integral_types() + (torch.bool,):
                # Assert that RuntimeError is raised
                # for inplace variant of Operators that
                # promote integer input to floating dtype.
                with self.assertRaises(RuntimeError):
                    alt(t.clone())
                continue

            actual = alt(t.clone())
            self.assertEqual(actual, expected, rtol=0, atol=0)

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
    #   values on a range of tensors, including empty tensors, scalar tensors,
    #   1D tensors and a large 2D tensor with interesting and extremal values
    #   and discontiguities.
    @suppress_warnings
    @ops(unary_ufuncs)
    def test_reference_numerics(self, device, dtype, op):
        include_extremals = (op.handles_complex_extremals if
                             dtype in (torch.cfloat, torch.cdouble) else op.handles_extremals)

        tensors = generate_numeric_tensors(device, dtype,
                                           domain=op.domain,
                                           include_large_values=op.handles_large_floats,
                                           include_extremal_values=include_extremals)
        for t in tensors:
            if dtype is torch.bfloat16:
                a = t.cpu().to(torch.float32).numpy()
            else:
                a = t.cpu().numpy()

            actual = op(t)
            expected = op.ref(a)

            # Crafts a custom error message for smaller, printable tensors
            if t.numel() < 10:
                msg = ("Failed to produce expected results! Input tensor was"
                       " {0}, torch result is {1}, and reference result is"
                       " {2}.").format(t, actual, expected)
            else:
                msg = None

            exact_dtype = True
            if op.promotes_integers_to_float and dtype in integral_types() + (torch.bool,):
                exact_dtype = False

                if dtype in [torch.uint8, torch.int8, torch.bool]:
                    # NOTE: For these dtypes, PyTorch computes in the default scalar type (float)
                    # while NumPy computes in float16
                    self.assertEqualHelper(actual, expected, msg, dtype=dtype,
                                           exact_dtype=exact_dtype, rtol=1e-3, atol=1e-2)
                    continue

            self.assertEqualHelper(actual, expected, msg, dtype=dtype, exact_dtype=exact_dtype)

    # Tests for testing (dis)contiguity consistency

    @ops(unary_ufuncs)
    def test_contig_vs_every_other(self, device, dtype, op):
        contig = make_tensor((1026,), device=device, dtype=dtype,
                             low=op.domain[0], high=op.domain[1])
        non_contig = contig[::2]

        self.assertTrue(contig.is_contiguous())
        self.assertFalse(non_contig.is_contiguous())

        self.assertEqual(op(contig)[::2], op(non_contig))

    @ops(unary_ufuncs)
    def test_contig_vs_transposed(self, device, dtype, op):
        contig = make_tensor((789, 357), device=device, dtype=dtype,
                             low=op.domain[0], high=op.domain[1])
        non_contig = contig.T

        self.assertTrue(contig.is_contiguous())
        self.assertFalse(non_contig.is_contiguous())

        self.assertEqual(op(contig).T, op(non_contig))

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

            self.assertEqual(op(contig), op(non_contig))

    @ops(unary_ufuncs)
    def test_non_contig_index(self, device, dtype, op):
        contig = make_tensor((2, 2, 1, 2), device, dtype,
                             low=op.domain[0], high=op.domain[1])
        non_contig = contig[:, 1, ...]
        contig = non_contig.contiguous()

        self.assertTrue(contig.is_contiguous())
        self.assertFalse(non_contig.is_contiguous())

        self.assertEqual(op(contig), op(non_contig))

    @ops(unary_ufuncs)
    def test_non_contig_expand(self, device, dtype, op):
        shapes = [(1, 3), (1, 7), (5, 7)]
        for shape in shapes:
            contig = make_tensor(shape, device, dtype,
                                 low=op.domain[0], high=op.domain[1])
            non_contig = contig.clone().expand(3, -1, -1)

            self.assertTrue(contig.is_contiguous())
            self.assertFalse(non_contig.is_contiguous())

            contig = op(contig)
            non_contig = op(non_contig)
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

        self.assertEqual(op(contig), op(contig2))

    @ops(unary_ufuncs)
    def test_contig_size1_large_dim(self, device, dtype, op):
        contig = make_tensor((5, 2, 3, 1, 4, 5, 3, 2, 1, 2, 3, 4), device, dtype,
                             low=op.domain[0], high=op.domain[1])
        contig = contig[:1, :, :, :, :, :, :, :, :, :, :, :]
        contig2 = torch.empty(contig.size(), device=device, dtype=dtype)
        contig2.copy_(contig)

        self.assertTrue(contig.is_contiguous())
        self.assertTrue(contig2.is_contiguous())

        self.assertEqual(op(contig), op(contig2))

    # Tests that computation on a multiple batches is the same as
    # per-batch computation.
    @ops(unary_ufuncs)
    def test_batch_vs_slicing(self, device, dtype, op):
        input = make_tensor((1024, 512), dtype=dtype, device=device,
                            low=op.domain[0], high=op.domain[1])

        actual = op(input)
        expected = torch.stack([op(slice) for slice in input])

        self.assertEqual(actual, expected)

    def _test_out_arg(self, op, input, output):
        dtype = input.dtype
        out_dtype = output.dtype
        if dtype is out_dtype:
            expected = op(input)
            op(input, out=output)
            self.assertEqual(output, expected)
        else:
            with self.assertRaises(RuntimeError):
                op(input, out=output)

    def _test_out_promote_int_to_float_op(self, op, input, output):
        def compare_out(op, input, out):
            out_dtype = out.dtype
            expected = op(input)
            op(input, out=out)
            self.assertEqual(out, expected.to(out_dtype))

        dtype = input.dtype
        out_dtype = output.dtype
        if out_dtype.is_floating_point and not dtype.is_complex:
            compare_out(op, input, output)
        elif out_dtype.is_floating_point and dtype.is_complex:
            # Can't cast complex to float
            with self.assertRaises(RuntimeError):
                op(input, out=output)
        elif out_dtype.is_complex:
            compare_out(op, input, output)
        else:
            # Can't cast to Integral types
            with self.assertRaises(RuntimeError):
                op(input, out=output)

    @ops(unary_ufuncs, dtypes=OpDTypes.supported)
    def test_out_arg_all_dtypes(self, device, dtype, op):
        input = make_tensor((64, 64), dtype=dtype, device=device,
                            low=op.domain[0], high=op.domain[1])

        for out_dtype in all_types_and_complex_and(torch.bool, torch.half):
            out = torch.empty_like(input, dtype=out_dtype)
            if op.promotes_integers_to_float:
                self._test_out_promote_int_to_float_op(op, input, out)
            else:
                self._test_out_arg(op, input, out)

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

    @unittest.skipIf(IS_MACOS, "Skip Reference: https://github.com/pytorch/pytorch/issues/47500")
    @dtypes(torch.cfloat, torch.cdouble)
    def test_sqrt_complex_edge_values(self, device, dtype):
        # Test Reference: https://github.com/pytorch/pytorch/pull/47424
        x = torch.tensor(0. - 1.0000e+20j, dtype=dtype, device=device)
        self.compare_with_numpy(torch.sqrt, np.sqrt, x)

        x = torch.tensor(-1.0000e+20 - 4988429.2000j, dtype=dtype, device=device)
        self.compare_with_numpy(torch.sqrt, np.sqrt, x)

    # TODO opinfo mvlgamma
    @unittest.skipIf(not TEST_SCIPY, "Scipy not found")
    def test_mvlgamma(self, device):
        from scipy.special import multigammaln
        for d in range(1, 5):
            input = torch.empty(10, device=device).uniform_(d, 10)
            res_torch = torch.mvlgamma(input, d)
            res_scipy = multigammaln(input.cpu().numpy(), d)
            self.assertEqual(res_torch.cpu().numpy(), res_scipy, atol=1e-5, rtol=0)

    def test_mvlgamma_argcheck(self, device):
        def run_test(d):
            input = torch.linspace((d - 2) / 2, 10, 10, device=device)
            torch.mvlgamma(input, d)

        with self.assertRaisesRegex(RuntimeError, r"All elements must be greater than \(p-1\)/2"):
            run_test(3)

    # TODO opinfo polygamma
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

    # TODO: update sign to use opinfo-based testing
    # XLA tests fail for self.assertRaises for complex dtypes
    @onlyOnCPUAndCUDA
    def test_sign_complex_assert_raises(self, device):
        for dtype in [torch.complex64, torch.complex128]:
            size = [5, 5]
            tensor = torch.rand(size, dtype=dtype, device=device)

            # index_add calls atomicAdd on cuda.
            zeros = torch.zeros(size, dtype=dtype, device=device)

            # index_add is not supported for complex dtypes on cuda yet
            if device.startswith('cuda') and dtype.is_complex:
                self.assertRaises(RuntimeError,
                                  lambda: zeros.index_add(0, torch.arange(0, size[0], dtype=torch.long, device=device), tensor))

            with self.assertRaisesRegex(RuntimeError,
                                        (r'Unlike NumPy, torch.sign is not intended to support complex numbers\. '
                                         r'Please use torch.sgn instead\.')):
                torch.sign(torch.tensor([4j], device=device, dtype=dtype))

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

    # TODO: review with ceil opinfo
    @onlyCUDA
    def test_ceil_out_mismatch(self, device):
        a = torch.randn(1)
        b = torch.randn(1, device=device)
        self.assertRaises(RuntimeError, lambda: torch.ceil(a, out=b))

    # TODO: review with erfinv opinfo
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float, torch.double)
    def test_erfinv(self, device, dtype):
        # general testing. Narrow the range to avoid accuracy issues
        input_values = torch.randn(4, 4, dtype=dtype, device=device).clamp(-0.3, 0.3)
        self.assertEqual(input_values.erf().erfinv(), input_values)
        # test inf
        self.assertTrue(torch.equal(torch.tensor([-1, 1], dtype=dtype, device=device).erfinv(),
                                    torch.tensor([-inf, inf], dtype=dtype, device=device)))
        # test nan
        self.assertEqual(torch.tensor([-2, 2], dtype=dtype, device=device).erfinv(),
                         torch.tensor([nan, nan], dtype=dtype, device=device))

        if dtype == torch.double:
            # double precision
            a = torch.tensor([0.5, 0.8], dtype=torch.double, device=device).erfinv()
            self.assertEqual(a[0].item(), 0.47693627620447, atol=1e-13, rtol=0)
            self.assertEqual(a[1].item(), 0.90619380243682, atol=1e-13, rtol=0)

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

    # Opinfo reciprocal
    @onlyCPU
    @dtypes(torch.float, torch.double)
    def test_reciprocal(self, device, dtype):
        a = torch.randn(100, 89, device=device, dtype=dtype)
        res_div = 1 / a
        res_reciprocal = a.clone()
        res_reciprocal.reciprocal_()
        self.assertEqual(res_reciprocal, res_div)

    @dtypes(torch.float, torch.double, torch.complex64, torch.complex128)
    def test_reciprocal_complex(self, device, dtype):
        t = torch.randn(10, 10, dtype=dtype, device=device)
        expected = torch.from_numpy(np.reciprocal(t.cpu().numpy()))
        actual = torch.reciprocal(t).cpu()
        self.assertEqual(expected, actual)

    @onlyCUDA
    @dtypes(torch.complex64, torch.complex128)
    def test_reciprocal_complex_extremal(self, device, dtype):
        vals = (
            # Inf and Zeros
            complex(float('inf'), float('inf')),
            complex(float('inf'), 0.),
            complex(0., float('inf')),
            complex(0., 0.),

            # Nans and Zeros
            complex(float('nan'), 0.),
            complex(0., float('nan')),
            complex(float('nan'), float('nan')),

            # Inf and Nans
            complex(float('nan'), float('inf')),
            complex(float('inf'), float('nan')),

            # Extremal and Normal Number
            complex(float('nan'), 2.0),
            complex(float('inf'), 2.0),
            complex(2.0, float('nan')),
            complex(2.0, float('inf')),
            complex(2.0, 0.0),
            complex(0.0, 2.0))

        self.compare_with_numpy(torch.reciprocal, np.reciprocal, vals, device, dtype)

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
            (vec1[::2, :], vec1[::2, :]),  # contiguous/discontiguous tests
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

    @skipCUDAIfRocm
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

    # TODO: rationalize with abs testing and verify absolute is tested as an alias
    @dtypes(torch.float)
    def test_absolute(self, device, dtype):
        # absolute is an alias for abs. Just check to see that results
        # are the same.
        t = torch.randn(10, 10, device=device, dtype=dtype)
        r_abs = t.abs()
        r_absolute = t.absolute()
        self.assertEqual(r_abs, r_absolute)

        r_abs = torch.abs(t)
        r_absolute = torch.absolute(t)
        self.assertEqual(r_abs, r_absolute)

        r_abs = torch.empty((10, 10), device=device, dtype=dtype)
        r_absolute = torch.empty((10, 10), device=device, dtype=dtype)
        torch.abs(t, out=r_abs)
        torch.absolute(t, out=r_absolute)
        self.assertEqual(r_abs, r_absolute)

        from copy import deepcopy
        t_copy = deepcopy(t)
        t.absolute_()
        t_copy.abs_()
        self.assertEqual(t, t_copy)

    # Note: ROCm fails when using float tensors
    # TODO: update this test to just compare against NumPy
    @onlyCUDA
    @dtypes(torch.double)
    def test_polygamma(self, device, dtype):
        cpu_tensor = torch.randn(10, 10, 10, dtype=dtype)
        device_tensor = cpu_tensor.to(device)
        zeros = torch.zeros(10, 10, 10, dtype=dtype)
        for n in [0, 1, 2, 3, 4, 5]:
            cpu_out = cpu_tensor.polygamma(n)
            device_out = device_tensor.polygamma(n)
            norm_errors = (device_out - cpu_out.to(device)) / device_out
            self.assertEqual(norm_errors, zeros)

        cpu_tensor.requires_grad = True
        for n in [0, 1, 2, 3, 4, 5]:
            torch.autograd.gradcheck(lambda x: x.polygamma(n),
                                     cpu_tensor)

    # Note: fails when using float tensors
    # TODO: update this test to just compare against NumPy
    @onlyCUDA
    @dtypes(torch.double)
    def test_digamma(self, device, dtype):
        cpu_tensor = torch.randn(10, 10, 10, dtype=dtype)
        device_tensor = cpu_tensor.to(device)
        zeros = torch.zeros(10, 10, 10, dtype=dtype)
        cpu_out = cpu_tensor.digamma()
        device_out = device_tensor.digamma()
        norm_errors = (device_out - cpu_out.to(device)) / device_out
        self.assertEqual(norm_errors, zeros)

        # Tests pole behavior
        cpu_tensor = torch.tensor([-0.999999994, -1.999999994, -2.0000000111,
                                   -100.99999994, -1931.99999994, 0.000000111,
                                   -0.000000111, 0, -1, -2, -931], dtype=dtype)
        expected_errors = torch.tensor([0, 0, 0, 0, 0, 0, 0, nan, nan, nan, nan], dtype=dtype)
        device_tensor = cpu_tensor.to(device)
        cpu_out = cpu_tensor.digamma()
        device_out = device_tensor.digamma()
        norm_errors = (device_out - cpu_out.to(device)) / device_out
        self.assertEqual(norm_errors, expected_errors)

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

    def test_bitwise_not(self, device):
        res = 0xffff - torch.arange(127, dtype=torch.int8, device=device)
        for dtype in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            if dtype == torch.bool:
                a = torch.tensor([True, False], device=device)
                expected_res = torch.tensor([False, True], device=device)
            else:
                a = torch.arange(127, dtype=dtype, device=device)
                expected_res = res.to(dtype)
            # new tensor
            self.assertEqual(expected_res, a.bitwise_not())
            # out
            b = torch.empty(0, dtype=dtype, device=device)
            torch.bitwise_not(a, out=b)
            self.assertEqual(expected_res, b)
            # in-place
            a.bitwise_not_()
            self.assertEqual(expected_res, a)

        # test exceptions
        for dtype in (torch.half, torch.float, torch.double):
            a = torch.zeros(10, dtype=dtype, device=device)
            # new tensor
            with self.assertRaises(RuntimeError):
                a.bitwise_not()
            # out
            b = torch.empty(0, dtype=dtype, device=device)
            with self.assertRaises(RuntimeError):
                torch.bitwise_not(a, out=b)
            # in-place
            with self.assertRaises(RuntimeError):
                a.bitwise_not_()

    @dtypes(*torch.testing.get_all_dtypes())
    def test_logical_not(self, device, dtype):
        data = [10, 1, 0.3, 0, -0.3, -1, -10]
        a = torch.tensor(data, dtype=dtype, device=device)
        if dtype == torch.bfloat16:  # numpy doesn't support these dtypes
            result = [False, False, False, True, False, False, False]
            self.assertEqual(torch.logical_not(a), torch.tensor(result, dtype=torch.bool, device=device))
        else:
            a_np = np.array(data, dtype=torch_to_numpy_dtype_dict[dtype])
            self.assertEqual(np.logical_not(a_np), torch.logical_not(a).to('cpu'))
            self.assertEqual(np.logical_not(a_np, out=a_np), a.logical_not_().to('cpu'))

    @dtypes(*product(torch.testing.get_all_dtypes(),
                     torch.testing.get_all_dtypes()))
    def test_logical_not_out(self, device, dtypes):
        dtype = dtypes[0]
        out_dtype = dtypes[1]
        data = [10, 1, 0.3, 0, -0.3, -1, -10]
        a = torch.tensor(data, dtype=dtype, device=device)
        out = torch.empty_like(a, dtype=out_dtype, device=device)
        if torch.bfloat16 in dtypes:  # numpy doesn't support these dtypes
            result = [not i for i in a]
            self.assertEqual(torch.logical_not(a, out=out), torch.tensor(result, dtype=out_dtype, device=device))
        else:
            out_np = np.empty(a.shape, dtype=torch_to_numpy_dtype_dict[out_dtype])
            self.assertEqual(a, a.cpu().numpy())
            torch.logical_not(a, out=out)
            np.logical_not(a.cpu().numpy(), out=out_np)
            self.assertEqual(out_np, out.to('cpu'))

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

    @dtypes(*torch.testing.get_all_dtypes(include_complex=False))
    def test_sign(self, device, dtype):
        if dtype == torch.bool:
            a_bool = torch.tensor([True, True, False, float('nan')], device=device).bool()
            a_bool_target = torch.tensor([True, True, False, True], device=device).bool()
            self.assertEqual(a_bool.sign(), a_bool_target, msg='sign device={} dtype=bool'.format(device))
            self.assertEqual(torch.sign(a_bool), a_bool_target, msg='sign device={} dtype=bool'.format(device))

            a_out = torch.empty_like(a_bool)
            torch.sign(a_bool, out=a_out)
            self.assertEqual(a_out, a_bool_target, msg='sign_out device={} dtype=bool'.format(device))

            a_bool.sign_()
            self.assertEqual(a_bool, a_bool_target, msg='sign_ device={} dtype=bool'.format(device))
            return

        # Include NaN for floating point numbers
        if dtype.is_floating_point:
            dt_info = torch.finfo(dtype)

            # Create tensor (with NaN checking)
            a = torch.tensor([float('nan'), -12, 0, 71, dt_info.min, dt_info.max], device=device, dtype=dtype)
            a_target = torch.tensor([0, -1, 0, 1, -1, 1], device=device, dtype=dtype)
        else:
            dt_info = torch.iinfo(dtype)

            # If unsigned type, everything should be >= 0
            if dt_info.min == 0:
                a = torch.tensor([12, 0, 71, dt_info.min, dt_info.max], device=device, dtype=dtype)
                a_target = torch.tensor([1, 0, 1, 0, 1], device=device, dtype=dtype)
            else:
                a = torch.tensor([-12, 0, 71, dt_info.min, dt_info.max], device=device, dtype=dtype)
                a_target = torch.tensor([-1, 0, 1, -1, 1], device=device, dtype=dtype)

        self.assertEqual(a.sign(), a_target, msg='sign device={} dtype={}'.format(device, dtype))
        self.assertEqual(torch.sign(a), a_target, msg='sign device={} dtype={}'.format(device, dtype))

        out = torch.empty_like(a)
        torch.sign(a, out=out)
        self.assertEqual(out, a_target, msg='sign_out device={} dtype={}'.format(device, dtype))

        a.sign_()
        self.assertEqual(a, a_target, msg='sign_ device={} dtype={}'.format(device, dtype))

    @dtypes(*(torch.testing.torch.testing.get_all_fp_dtypes()))
    def test_signbit_float(self, device, dtype):
        t = torch.randn(5, 5, device=device)

        if dtype == torch.bfloat16:
            t_bf16 = torch.tensor([1, 0, -1], device=device, dtype=dtype)
            self.assertEqual(torch.signbit(t_bf16), torch.tensor([False, False, True]))
        else:
            self.compare_with_numpy(torch.signbit, np.signbit, t)

        t_target = torch.signbit(t)
        out = torch.empty_like(t, device=device, dtype=torch.bool)
        torch.signbit(t, out=out)
        self.assertEqual(out, t_target)

        t_sp = (0, float('inf'), -float('inf'), float('nan'))
        if dtype == torch.bfloat16:
            t_sp_df16 = torch.tensor(t_sp, device=device, dtype=dtype)
            self.assertEqual(torch.signbit(t_sp_df16), torch.tensor([False, False, True, False]))
        else:
            self.compare_with_numpy(torch.signbit, np.signbit, t_sp, device, dtype)

    @dtypes(*(torch.testing.get_all_int_dtypes() + [torch.bool]))
    def test_signbit_int_and_bool(self, device, dtype):
        t = torch.randint(-5, 5, (5, 5), device=device)
        self.compare_with_numpy(torch.signbit, np.signbit, t)

        t_target = torch.signbit(t)
        out = torch.empty_like(t, device=device, dtype=torch.bool)
        torch.signbit(t, out=out)
        self.assertEqual(out, t_target)

    @dtypes(torch.complex64, torch.complex128)
    def test_signbit_complex(self, device, dtype):
        vals = (complex(0, -1), complex(-1, 2))
        t = torch.tensor(vals, device=device, dtype=dtype)
        out = torch.empty_like(t).real.bool()

        with self.assertRaisesRegex(RuntimeError, 'signbit is not implemented for complex tensors.'):
            torch.signbit(t)
        with self.assertRaisesRegex(RuntimeError, 'signbit is not implemented for complex tensors.'):
            torch.signbit(t, out=out)

    @dtypes(torch.cfloat, torch.cdouble)
    def test_sgn(self, device, dtype):
        x = torch.randn(100, dtype=dtype)
        angle = x.angle()
        out = x.sgn()
        self.assertEqual(out.angle(), angle)
        self.assertEqual(out.abs(), torch.ones_like(x).real)

        x_out = torch.empty_like(x)
        torch.sgn(x, out=x_out)
        self.assertEqual(x_out.angle(), angle)
        self.assertEqual(x_out.abs(), torch.ones_like(x).real)

    @dtypes(*(torch.testing.get_all_dtypes(include_bool=False)))
    def test_signbit_non_boolean_output(self, device, dtype):
        # test non-boolean tensors as the `out=` parameters
        # boolean outputs are tested in the above testcases
        t = torch.randn(5, 5)
        out = torch.empty_like(t, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, 'does not support non-boolean outputs'):
            torch.signbit(t, out=out)

    # This function tests that a nan value is returned for input values not in domain
    @dtypes(torch.float32, torch.float64)
    def test_acosh_domain_float(self, device, dtype):
        # Domain of acosh is [1, inf), for values outside the domain - output is mapped
        # to NaN, except for input value `inf` - output is mapped to `inf`
        sample = torch.tensor([float('-inf'), 1.00, -1.23, -0.06, 0.98, float('inf')],
                              device=device, dtype=dtype)
        nan_mask = torch.tensor([True, False, True, True, True, False], device=device)
        inf_mask = torch.tensor([False, False, False, False, False, True], device=device)
        self.assertEqual(torch.isnan(torch.acosh(sample)), nan_mask)
        self.assertEqual(torch.isnan(sample.acosh()), nan_mask)
        self.assertEqual(torch.isinf(torch.acosh(sample)), inf_mask)
        self.assertEqual(torch.isinf(sample.acosh()), inf_mask)

    # This function tests that a nan value is returned for input values not in domain
    @dtypes(torch.float32, torch.float64)
    def test_atanh_domain_float(self, device, dtype):
        # Domain of atanh is (-1, 1), for edge values (-1 and 1) - output is mapped
        # to inf and for other values outside this range - output is mapped to NaN
        sample = torch.tensor([float('-inf'), -1.00, 1.00, -1.23, 1.06, float('inf')],
                              device=device, dtype=dtype)
        nan_mask = torch.tensor([True, False, False, True, True, True], device=device)
        inf_mask = torch.tensor([False, True, True, False, False, False], device=device)
        # For values not in domain (except -1.0 and 1.0), atanh should return nan
        self.assertEqual(torch.isnan(torch.atanh(sample)), nan_mask)
        self.assertEqual(torch.isnan(sample.atanh()), nan_mask)
        # For values -1.0 and 1.0, atanh should return -inf and inf respectively
        self.assertEqual(torch.isinf(torch.atanh(sample)), inf_mask)
        self.assertEqual(torch.isinf(sample.atanh()), inf_mask)


def _generate_reference_input(dtype, device):
    input = []
    input.append(list(range(-5, 5)))
    input.append([0 for x in range(-5, 5)])
    input.append([x + 1e-6 for x in range(-5, 5)])
    # Some vectorized implementations don't support large values
    input.append([x + 1e10 for x in range(-5, 5)])
    input.append([x - 1e10 for x in range(-5, 5)])
    input.append([*torch.randn(7).tolist(), math.inf, -math.inf, math.nan])
    input.append((torch.randn(10) * 1e6).tolist())
    input.append([math.pi * (x / 2) for x in range(-5, 5)])
    return torch.tensor(input, dtype=dtype, device=device)

def _generate_gamma_input(dtype, device, test_poles=True):
    input = []
    input.append((torch.randn(10).abs() + 1e-4).tolist())
    input.append((torch.randn(10).abs() + 1e6).tolist())
    zeros = torch.linspace(-9.5, -0.5, 10)
    input.append(zeros.tolist())
    input.append((zeros - 0.49).tolist())
    input.append((zeros + 0.49).tolist())
    input.append((zeros + (torch.rand(10) * 0.99) - 0.5).tolist())

    if test_poles:
        input.append([-0.999999994, -1.999999994, -2.0000000111,
                      -100.99999994, -1931.99999994, 0.000000111,
                      -0.000000111, 0, -2, -329])
    return torch.tensor(input, dtype=dtype, device=device)

# this class contains information needed to generate tests for torch math functions
# the generated tests compare torch implementation with the reference numpy/scipy implementation,
# and also check proper behavior for contiguous/discontiguous/inplace outputs.
class _TorchMathTestMeta(object):
    def __init__(self,
                 opstr,
                 args=(),
                 reffn=None,
                 refargs=lambda x: (x.numpy(),),
                 input_fn=_generate_reference_input,
                 inputargs=(),
                 substr='',
                 make_inplace=True,
                 decorators=None,
                 ref_backend='numpy',
                 rtol=None,
                 atol=None,
                 dtypes=floating_types(),
                 replace_inf_with_nan=False):
        self.opstr = opstr
        self.args = args
        self.reffn = reffn  # reffn is either callable or ref_backend attribute, set to opstr if not specified
        self.refargs = refargs
        self.input_fn = input_fn
        self.inputargs = inputargs
        self.substr = substr
        self.make_inplace = make_inplace
        assert ref_backend == 'numpy' or ref_backend == 'scipy'
        self.ref_backend = ref_backend
        if ref_backend == 'scipy':
            self.ref_decorator = [unittest.skipIf(not TEST_SCIPY, "Scipy not found")]
        else:
            self.ref_decorator = []
        self.decorators = decorators
        self.rtol = rtol
        self.atol = atol
        self.dtypes = dtypes
        self.replace_inf_with_nan = replace_inf_with_nan

# TODO: replace with make_tensor
# Converts half/bfloat16 dtype to float when device is cpu
def _convert_t(dtype, device):
    if device == 'cpu' and dtype in {torch.half, torch.bfloat16}:
        return torch.float
    return dtype

# TODO: replace with make_tensor
# Returns a tensor of the requested shape, dtype, and device
# Requesting a half CPU tensor returns a float CPU tensor with
# values representable by a half.
# Initialization uses randint for non-float types and randn for float types.
def _make_tensor(shape, dtype, device, fill_ones=False) -> torch.Tensor:
    # Returns a tensor filled with ones
    if fill_ones:
        return torch.ones(*shape, dtype=_convert_t(dtype, device), device=device)

    # Returns a tensor with random integer values
    if not (dtype.is_floating_point or dtype.is_complex):
        t = torch.randint(0, 10, shape, device=device)
        if dtype != torch.uint8:
            t = t - 5  # generate negative values also
        return t.to(_convert_t(dtype, device))

    # Populates the CPU tensor with floats representable as half/bfloat16
    if dtype == torch.half and device == 'cpu':
        return torch.randn(*shape, dtype=torch.float, device=device).half().float()
    if dtype == torch.bfloat16 and device == 'cpu':
        return torch.randn(*shape, dtype=torch.float, device=device).bfloat16().float()

    # Default: returns a tensor with random float values
    return torch.randn(shape, dtype=dtype, device=device).to(dtype=dtype)

# TODO: replace with make_tensor
def _medium_2d(dtype, device):
    return _make_tensor((50, 50), dtype, device)

# TODO: replace with opinfo
_types_no_half = [
    torch.float, torch.double,
    torch.int8, torch.short, torch.int, torch.long,
    torch.uint8
]

# TODO: all these should be replaced with OpInfos
torch_op_tests = [
    _TorchMathTestMeta('exp'),
    _TorchMathTestMeta('floor'),
    _TorchMathTestMeta('ceil'),
    _TorchMathTestMeta('rad2deg'),
    _TorchMathTestMeta('deg2rad'),
    _TorchMathTestMeta('rsqrt', reffn=lambda x: np.reciprocal(np.sqrt(x))),
    _TorchMathTestMeta('frac', reffn='fmod', refargs=lambda x: (x.numpy(), 1)),
    _TorchMathTestMeta('trunc'),
    _TorchMathTestMeta('round'),
    # FIXME lgamma produces different result compared to scipy at -inf
    _TorchMathTestMeta('lgamma', reffn='gammaln', ref_backend='scipy', replace_inf_with_nan=True),
    _TorchMathTestMeta('polygamma', args=[0], substr='_0', reffn='polygamma',
                       refargs=lambda x: (0, x.numpy()), input_fn=_generate_gamma_input, inputargs=[False],
                       ref_backend='scipy'),
    _TorchMathTestMeta('polygamma', args=[1], substr='_1', reffn='polygamma',
                       refargs=lambda x: (1, x.numpy()), input_fn=_generate_gamma_input, inputargs=[False],
                       ref_backend='scipy', rtol=0.0008, atol=1e-5),
    _TorchMathTestMeta('polygamma', args=[2], substr='_2', reffn='polygamma',
                       refargs=lambda x: (2, x.numpy()), input_fn=_generate_gamma_input, inputargs=[False],
                       ref_backend='scipy', rtol=0.0008, atol=1e-5),
    _TorchMathTestMeta('digamma',
                       input_fn=_generate_gamma_input, inputargs=[True], ref_backend='scipy',
                       replace_inf_with_nan=True),
    _TorchMathTestMeta('abs', input_fn=_medium_2d, dtypes=_types_no_half, rtol=0., atol=0.),
    _TorchMathTestMeta('logit', ref_backend='scipy')]


def generate_torch_test_functions(cls, testmeta, inplace):
    opstr = testmeta.opstr if not inplace else testmeta.opstr + "_"

    def torchfn(x):
        return getattr(x, opstr)(*testmeta.args)

    def fn_check_reference(self, device, dtype):
        def reffn(x):
            backend = np if testmeta.ref_backend == 'numpy' else scipy.special
            opstr = None
            if testmeta.reffn is None:
                opstr = testmeta.opstr
            elif isinstance(testmeta.reffn, str):
                opstr = testmeta.reffn
            if callable(testmeta.reffn):
                fn = testmeta.reffn
            else:
                assert opstr is not None, "invalid reffn"
                fn = getattr(backend, opstr)
            return fn(*testmeta.refargs(x))

        inp = testmeta.input_fn(dtype, device, *testmeta.inputargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            expected = torch.from_numpy(reffn(inp))
        actual = torchfn(inp)
        if testmeta.replace_inf_with_nan:
            actual[(actual == -inf) | (actual == inf)] = nan
            expected[(expected == -inf) | (expected == inf)] = nan

        torch.testing.assert_allclose(actual, expected, rtol=testmeta.rtol, atol=testmeta.atol)

    def fn_non_contig(self, device, dtype) -> None:
        shapes = [(5, 7), (1024,)]
        for shape in shapes:
            contig = _make_tensor(shape, dtype=dtype, device=device)
            non_contig = torch.empty(shape + (2,), dtype=dtype)[..., 0]
            non_contig.copy_(contig)
            self.assertFalse(non_contig.is_contiguous())
            self.assertEqual(torchfn(contig), torchfn(non_contig), msg='non-contiguous')

    def fn_non_contig_index(self, device, dtype):
        contig = _make_tensor((2, 2, 1, 2), dtype=dtype, device=device)
        non_contig = contig[:, 1, ...]
        contig = non_contig.clone()
        self.assertFalse(non_contig.is_contiguous())
        self.assertEqual(torchfn(contig), torchfn(non_contig), msg='non-contiguous index')

    def fn_non_contig_expand(self, device, dtype):
        shapes = [(1, 3), (1, 7), (5, 7)]
        for shape in shapes:
            contig = _make_tensor(shape, dtype=dtype, device=device)
            non_contig = contig.clone().expand(3, -1, -1)
            self.assertFalse(non_contig.is_contiguous())
            contig = torchfn(contig)
            non_contig = torchfn(non_contig)
            for i in range(3):
                self.assertEqual(contig, non_contig[i], msg='non-contiguous expand[' + str(i) + ']')

    def fn_contig_size1(self, device, dtype):
        contig = _make_tensor((5, 100), dtype=dtype, device=device)
        contig = contig[:1, :50]
        contig2 = torch.empty(contig.size(), dtype=dtype)
        contig2.copy_(contig)
        self.assertTrue(contig.is_contiguous())
        self.assertTrue(contig2.is_contiguous())
        self.assertEqual(torchfn(contig), torchfn(contig2), msg='contiguous size1')

    def fn_contig_size1_large_dim(self, device, dtype):
        contig = _make_tensor((5, 2, 3, 1, 4, 5, 3, 2, 1, 2, 3, 4), dtype=dtype, device=device)
        contig = contig[:1, :, :, :, :, :, :, :, :, :, :, :]
        contig2 = torch.empty(contig.size(), dtype=dtype)
        contig2.copy_(contig)
        self.assertTrue(contig.is_contiguous())
        self.assertTrue(contig2.is_contiguous())
        self.assertEqual(torchfn(contig), torchfn(contig2), msg='contiguous size1')

    def fn_large(self, device, dtype):
        input = _make_tensor((1024, 512), dtype=dtype, device=device)
        # clone input to properly test inplace functions
        actual = torchfn(input.clone())
        expected = torch.stack([torchfn(slice) for slice in input])
        self.assertEqual(actual, expected, msg='large')

    test_functions = {"test_reference_": fn_check_reference,
                      "test_non_contig_": fn_non_contig,
                      "test_non_contig_index_": fn_non_contig_index,
                      "test_non_contig_expand_": fn_non_contig_expand,
                      "test_contig_size1_": fn_contig_size1,
                      "test_check_contig_size1_large_dim_": fn_contig_size1_large_dim,
                      "test_large_": fn_large}
    for name in test_functions:
        if inplace and 'expand' in name:
            continue
        test_name = name + testmeta.opstr + testmeta.substr
        if inplace:
            test_name += "_inplace"
        assert not hasattr(cls, test_name), "{0} already in TestUnaryUfuncMathOps".format(test_name)

        decorators = [] if testmeta.decorators is None else testmeta.decorators
        if 'reference' in name:
            decorators = decorators + testmeta.ref_decorator
        decorators = decorators + [dtypes(*testmeta.dtypes)]
        fn_test = test_functions[name]
        for dec in decorators:
            fn_test = dec(fn_test)
        setattr(cls, test_name, fn_test)

class TestUnaryUfuncMathOps(TestCase):
    exact_dtype = True

def generate_torch_op_tests(cls):
    for t in torch_op_tests:
        generate_torch_test_functions(cls, t, False)
        if t.make_inplace:
            generate_torch_test_functions(cls, t, True)


generate_torch_op_tests(TestUnaryUfuncMathOps)
instantiate_device_type_tests(TestUnaryUfuncs, globals())
instantiate_device_type_tests(TestUnaryUfuncMathOps, globals(), only_for='cpu')

if __name__ == '__main__':
    run_tests()
