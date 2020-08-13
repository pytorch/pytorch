import math
from functools import partial
from itertools import product, chain
from numbers import Number
import operator

import unittest

import torch
import numpy as np

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, torch_to_numpy_dtype_dict, suppress_warnings,
     IS_WINDOWS)
from torch.testing._internal.common_methods_invocations import \
    (unary_ufuncs)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, onlyOnCPUAndCUDA, skipCUDAIfRocm,
     dtypes)
from torch.testing import \
    (floating_types_and)

# Tensor generators and helpers

# Interesting values and extremal values for different dtypes
_unsigned_int_vals = (0, 1, 55, 127)
_int_vals = (0, -1, 1, -55, 55, -127, 127)
_small_float_vals = (0.,
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

def _make_tensor(size, device, dtype, *, domain=None):
    if dtype is torch.bool:
        return torch.tensor((True, False), device=device)

    # Windows doesn't support torch.rand(bfloat16) on CUDA
    if IS_WINDOWS and torch.device(device).type == 'cuda' and dtype is torch.bfloat16:
        t = torch.rand(size, device=device, dtype=torch.float32).to(dtype)
    else:
        t = torch.rand(size, device=device, dtype=dtype)

    if domain is None:
        return t

    # creates values [low, min(low + 100, high)) in case domain is unbounded
    #   on the right
    gap = min(domain[1] - domain[0], 100)
    return (t + domain[0]) * gap

# Returns a new array with fp32 values representable in bfloat16
def bfloat16ify(a):
    assert a.dtype == np.float32
    return torch.from_numpy(a).to(torch.bfloat16).to(torch.float32).numpy()

# Given an interable of NumPy arrays, returns a generator of 2-tuples
# (torch.Tensor, numpy.ndarray) containing identical values and of
# the same dtype, except for bfloat16, where the NumPy array will be in
# float32.
# The Torch tensors will be on the specified device and will not share storage,
# but the NumPy arrays may or may not share storage.
def pair_arrays_with_tensors(arrays, device, dtype):
    # Special case for bfloat16 since NumPy doesn't have bfloat16
    if dtype is torch.bfloat16:
        return ((torch.from_numpy(a).to(dtype=dtype, device=device),
                 bfloat16ify(a)) for a in arrays)

    if torch.device(device).type == 'cpu':
        # Copies array on the CPU so the tensors have independent storage
        return ((torch.from_numpy(np.copy(a)).to(dtype=dtype, device=device), a) for a in arrays)
    else:
        return ((torch.from_numpy(a).to(dtype=dtype, device=device), a) for a in arrays)

# Returns a generator (size) -> random complex array
# With the real and imaginary parts in [low, high)
def get_random_complex_generator(rng, dtype, low, high):
    def generator(size):
        a = np.empty(size, dtype=dtype)
        a.real = rng.uniform(low, high, size)
        a.imag = rng.uniform(low, high, size)
        return a
    return generator

# Returns a generator (size) -> random boolean array
def get_random_bool_generator(rng):
    def generator(size):
        a = rng.uniform(0, 1, size)
        return a > .5
    return generator

# Returns a generator (size) -> random array
# The arrays dtype corresponds to the given torch_dtype, and its values
# are in the specified domain or a dtype-specific domain if None is specified.
def get_random_array_generator(torch_dtype, *, domain=None):
    rng = np.random.default_rng()

    # Note: NumPy does not have a bfloat16 type
    if torch_dtype is torch.bfloat16:
        np_dtype = np.float32
    else:
        np_dtype = torch_to_numpy_dtype_dict[torch_dtype]

    low = None if domain is None else domain[0]
    high = None if domain is None else domain[1]

    # Acquires a default domain (if unspecified) and dtype-specific generator
    if torch_dtype is torch.bool:
        return get_random_bool_generator(rng)
    elif torch_dtype.is_floating_point:
        low = low if low is not None else -2
        high = high if high is not None else 2

        def generator(size):
            gen = partial(rng.uniform, low, high)
            generated = gen(size)
            return generated.astype(np_dtype)

        return generator
    elif torch_dtype.is_complex:
        low = low if low is not None else -2
        high = high if high is not None else 2
        return get_random_complex_generator(rng, np_dtype, low, high)
    else:  # dtype is an integer (unsigned or signed)
        assert torch_dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        if torch_dtype is torch.uint8:
            low = low if low is not None else 0
            high = high if high is not None else 128
        else:
            low = low if low is not None else -9
            high = high if high is not None else 10

        def generator(size):
            gen = partial(rng.integers, low, high)
            generated = gen(size)
            return gen.astype(np_dtype)

        return generator

# Returns a list of pairs (torch.Tensor, numpy.ndarray) where the tensor
# and array have identical values and the given dtype (except when the
# dtype is bfloat16, in which case the array will be in float32). The Torch
# tensors will be on the given device and have independent storages.
# The tensors will include scalar tensors (0-dims), "small" 1D tensors,
# a "medium" 1D tensor, and a 1024x512 2D tensor.
# The tensors will "cover" the values in vals, and if vals is None then
# dtype-specific values will be substituted.
# The values, except for those specified in vals, will be in the domain
# [low, high), if specified.
# If include_extremal_values is False, then the default values will not include
# NaN, positive infinity, or negative infinity.
def generate_numeric_tuples(device, dtype, *,
                            vals=None, domain=None,
                            include_large_values=True, include_extremal_values=True):
    medium_length = 812
    large_size = (1029, 917)

    # Special-cases bool
    if dtype is torch.bool:
        generator = get_random_array_generator(dtype)
        arrays = (np.empty(0, dtype=np.bool),
                  np.array(True),
                  np.array(False),
                  np.array((True, False)),
                  generator(medium_size),
                  generator(large_size))
        return pair_arrays_with_tensors(arrays, device, dtype)

    rng = np.random.default_rng()

    # Acquires dtype-specific vals
    if vals is None:
        if dtype.is_floating_point or dtype.is_complex:
            if include_large_values and include_extremal_values:
                vals = _small_float_vals + _large_float_vals + _float_extremals
            elif include_extremal_values:
                vals = _small_float_vals + _float_extremals
            elif include_large_values:
                vals = _small_float_vals + _large_float_vals
            else:
                vals = _small_float_vals

            # Converts float -> complex vals if dtype is complex
            if dtype.is_complex:
                vals = tuple(complex(x, y) for x, y in product(vals, vals))
        elif dtype is torch.uint8:
            vals = _unsigned_int_vals
        else:  # dtypes is a signed integer type
            assert torch_dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
            vals = _int_vals

    assert len(vals) < medium_length

    # Constructs the large array containing vals
    generator = get_random_array_generator(dtype, domain=domain)
    arr = generator((large_size))

    # Inserts the vals at an odd place
    arr[57][63:63 + len(vals)] = vals

    # Takes a medium sized view of the large array containing vals
    medium_array = arr[57][63:63 + medium_length]

    # Constructs small array (4 elements)
    small_arrays = np.split(medium_array, medium_length / 4)

    # Constructs scalar arrays
    scalar_arrays = (arr.squeeze() for arr in np.split(medium_array, medium_length))

    # Empty array
    empty_array = generator((0,))

    arrays = chain(empty_array, scalar_arrays, small_arrays, (medium_array,), (arr,))
    return pair_arrays_with_tensors(arrays, device, dtype)


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

# TODO: port test_unary_out_op_mem_overlap
class TestUnaryUfuncs(TestCase):
    exact_dtype = True

    # Helper for comparing torch tensors and numpy arrays
    # TODO: should this or assertEqual also validate that strides are equal?
    def assertEqualHelper(self, actual, expected, *, dtype, exact_dtype=True, **kwargs):
        assert isinstance(actual, torch.Tensor)

        # Some NumPy functions return scalars, not arrays
        if isinstance(expected, Number):
            self.assertEqual(actual.item(), expected)
        elif isinstance(expected, np.ndarray):
            # Accounts for bfloat16 comparisons
            if exact_dtype:
                if expected.dtype == np.float32:
                    assert actual.dtype in (torch.bfloat16, torch.float32)
                else:
                    assert expected.dtype == torch_to_numpy_dtype_dict[actual.dtype]

            self.assertEqual(actual,
                             torch.from_numpy(expected).to(actual.dtype),
                             exact_device=False,
                             **kwargs)
        else:
            self.assertEqual(actual, expected, exact_device=False, **kwargs)

    # Verifies that the unary ufuncs have their supported dtypes
    #   registered correctly by testing that each unlisted dtype
    #   throws a runtime error
    @skipCUDAIfRocm
    @onlyOnCPUAndCUDA
    @ops(unary_ufuncs, unsupported_dtypes_only=True)
    def test_unsupported_dtypes(self, device, dtype, op):
        t = torch.empty(1, device=device, dtype=dtype)
        with self.assertRaises(RuntimeError):
            op(t)

    @dtypes(*floating_types_and(torch.bfloat16, torch.half))
    @ops((_fn for _fn in unary_ufuncs if _fn.domain is not None))
    def test_float_domains(self, device, dtype, op):
        if not op.supports_dtype(dtype, torch.device(device).type):
            raise unittest.SkipTest('unsupported dtype')

        eps = (1e-1, 1, 2, 10, 20, 50, 100)

        # Adds smaller values except on bfloat16, since these small differences
        #   are often too small to represent in bfloat16
        if dtype is not torch.bfloat16:
            eps = eps + (1e-5, 1e-3)

        for base, _fn in ((op.domain[0], operator.sub), (op.domain[1], operator.add)):
            for epsilon in eps:
                v = _fn(base, epsilon)
                t = torch.tensor(v, device=device, dtype=dtype)
                result = op(t)
                self.assertEqual(result.item(), float('nan'),
                                 msg="input:{0}".format(t.item()))

    # Tests that fn == method == inplace == jit on a simple single tensor input
    @ops(unary_ufuncs)
    def test_variant_consistency(self, device, dtype, op):
        def _fn(t):
            return op(t)

        t = _make_tensor((5, 5), device, dtype, domain=op.domain)
        expected = op(t)

        for alt in (op.get_method(), op.get_inplace(), torch.jit.script(_fn)):
            if alt is None:
                continue

            actual = alt(t.clone())
            self.assertEqual(actual, expected)

    # Tests that the function and its (array-accepting) reference produce the same
    #   values on a range of tensors, including empty tensors, scalar tensors,
    #   1D tensors and a large 2D tensor with interesting and extremal values
    #   and discontiguities.
    @suppress_warnings
    @ops(unary_ufuncs)
    def test_reference_numerics(self, device, dtype, op):
        include_extremals = (op.handles_complex_extremals if
                             dtype in (torch.cfloat, torch.cdouble) else op.handles_extremals)

        pairs = generate_numeric_tuples(device, dtype,
                                        domain=op.domain,
                                        include_large_values=op.handles_large_floats,
                                        include_extremal_values=include_extremals)
        for t, a in pairs:
            actual = op(t)
            expected = op.ref(a)
            self.assertEqualHelper(actual, expected, dtype=dtype)

            if t.ndim > 0:
                actual = op(t[::2])
                expected = op.ref(a[::2])
                self.assertEqualHelper(actual, expected, dtype=dtype)

            if t.ndim > 1:
                actual = op(t.T)
                expected = op.ref(a.T)
                self.assertEqualHelper(actual, expected, dtype=dtype)

    # Tests for testing (dis)contiguity consistency

    @ops(unary_ufuncs)
    def test_non_contig(self, device, dtype, op):
        shapes = [(5, 7), (1024,)]
        for shape in shapes:
            contig = _make_tensor(shape, device, dtype)
            non_contig = torch.empty(shape + (2,), device=device, dtype=dtype)[..., 0]
            non_contig.copy_(contig)

            self.assertTrue(contig.is_contiguous())
            self.assertFalse(non_contig.is_contiguous())

            self.assertEqual(op(contig), op(non_contig))

    @ops(unary_ufuncs)
    def test_non_contig_index(self, device, dtype, op):
        contig = _make_tensor((2, 2, 1, 2), device, dtype)
        non_contig = contig[:, 1, ...]
        contig = non_contig.contiguous()

        self.assertTrue(contig.is_contiguous())
        self.assertFalse(non_contig.is_contiguous())

        self.assertEqual(op(contig), op(non_contig))

    @ops(unary_ufuncs)
    def test_non_contig_expand(self, device, dtype, op):
        shapes = [(1, 3), (1, 7), (5, 7)]
        for shape in shapes:
            contig = _make_tensor(shape, device, dtype)
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
        contig = _make_tensor((5, 100), device, dtype)
        contig = contig[:1, :50]
        contig2 = torch.empty(contig.size(), device=device, dtype=dtype)
        contig2.copy_(contig)

        self.assertTrue(contig.is_contiguous())
        self.assertTrue(contig2.is_contiguous())

        self.assertEqual(op(contig), op(contig2))

    @ops(unary_ufuncs)
    def test_contig_size1_large_dim(self, device, dtype, op):
        contig = _make_tensor((5, 2, 3, 1, 4, 5, 3, 2, 1, 2, 3, 4), device, dtype)
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
        input = _make_tensor((1024, 512), dtype=dtype, device=device)

        actual = op(input)
        expected = torch.stack([op(slice) for slice in input])

        self.assertEqual(actual, expected)


instantiate_device_type_tests(TestUnaryUfuncs, globals())

if __name__ == '__main__':
    run_tests()
