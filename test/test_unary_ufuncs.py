import math
from itertools import product, chain
from numbers import Number

import unittest

import torch

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, torch_to_numpy_dtype_dict, suppress_warnings,
     TEST_NUMPY, make_tensor)
from torch.testing._internal.common_methods_invocations import \
    (unary_ufuncs)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, dtypes)
from torch.testing import \
    (floating_types_and)

if TEST_NUMPY:
    import numpy as np

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

        for alt in (op.get_method(), op.get_inplace(), torch.jit.script(_fn)):
            if alt is None:
                with self.assertRaises(RuntimeError):
                    alt(t.clone())

            actual = alt(t.clone())
            self.assertEqual(actual, expected, rtol=0, atol=0)

    # Helper for comparing torch tensors and numpy arrays
    # TODO: should this or assertEqual also validate that strides are equal?
    def assertEqualHelper(self, actual, expected, msg, *, dtype, exact_dtype=True, **kwargs):
        assert isinstance(actual, torch.Tensor)

        # Some NumPy functions return scalars, not arrays
        if isinstance(expected, Number):
            self.assertEqual(actual.item(), expected)
        elif isinstance(expected, np.ndarray):
            # Handles exact dtype comparisons between arrays and tensors
            if exact_dtype:
                # Allows array dtype to be float32 when comparing with bfloat16 tensors
                #   since NumPy doesn't support the bfloat16 dtype
                if expected.dtype == np.float32:
                    assert actual.dtype in (torch.bfloat16, torch.float32)
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
    @unittest.skipIf(not TEST_NUMPY, "NumPy not found")
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

            self.assertEqualHelper(actual, expected, msg, dtype=dtype)

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


instantiate_device_type_tests(TestUnaryUfuncs, globals())

if __name__ == '__main__':
    run_tests()
