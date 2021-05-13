import torch
import numpy as np
import scipy.special

import unittest
import math
from typing import Dict, List
import random
from functools import partial
from itertools import product, combinations, permutations
import warnings

from torch._six import inf, nan
from torch.testing._internal.common_utils import (
    TestCase, run_tests, TEST_SCIPY, slowTest, torch_to_numpy_dtype_dict,
    IS_WINDOWS, make_tensor)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, onlyCPU, dtypes, dtypesIfCUDA, dtypesIfCPU,
    onlyOnCPUAndCUDA, onlyCUDA, largeTensorTest, precisionOverride)

# TODO: replace with make_tensor
def _generate_input(shape, dtype, device, with_extremal):
    if shape == ():
        x = torch.tensor((), dtype=dtype, device=device)
    else:
        if dtype.is_floating_point or dtype.is_complex:
            # work around torch.randn not being implemented for bfloat16
            if dtype == torch.bfloat16:
                x = torch.randn(*shape, device=device) * random.randint(30, 100)
                x = x.to(torch.bfloat16)
            else:
                x = torch.randn(*shape, dtype=dtype, device=device) * random.randint(30, 100)
            x[torch.randn(*shape) > 0.5] = 0
            if with_extremal and dtype.is_floating_point:
                # Use extremal values
                x[torch.randn(*shape) > 0.5] = float('nan')
                x[torch.randn(*shape) > 0.5] = float('inf')
                x[torch.randn(*shape) > 0.5] = float('-inf')
            elif with_extremal and dtype.is_complex:
                x[torch.randn(*shape) > 0.5] = complex('nan')
                x[torch.randn(*shape) > 0.5] = complex('inf')
                x[torch.randn(*shape) > 0.5] = complex('-inf')
        elif dtype == torch.bool:
            x = torch.zeros(shape, dtype=dtype, device=device)
            x[torch.randn(*shape) > 0.5] = True
        else:
            x = torch.randint(15, 100, shape, dtype=dtype, device=device)

    return x

# TODO: replace with make_tensor
def _rand_shape(dim, min_size, max_size):
    shape = []
    for i in range(dim):
        shape.append(random.randint(min_size, max_size))
    return tuple(shape)

class TestReductions(TestCase):

    def test_var_unbiased(self, device):
        tensor = torch.randn(100, device=device)
        self.assertEqual(tensor.var(0), tensor.var(0, unbiased=True))
        self.assertEqual(tensor.var(), tensor.var(unbiased=True))
        self.assertEqual(tensor.var(unbiased=False), tensor.var(0, unbiased=False))

        tensor = torch.tensor([1.0, 2.0], device=device)
        self.assertEqual(tensor.var(unbiased=True), 0.5)
        self.assertEqual(tensor.var(unbiased=False), 0.25)

        tensor = torch.tensor([1.0, 2.0, 3.0], device=device)
        self.assertEqual(tensor.var(unbiased=True), 1.0)
        self.assertEqual(tensor.var(unbiased=False), 2.0 / 3.0)

        tensor = torch.randn(100, device=device)
        self.assertEqual(tensor.std(0), tensor.std(0, unbiased=True))
        self.assertEqual(tensor.std(), tensor.std(unbiased=True))
        self.assertEqual(tensor.std(unbiased=False), tensor.std(0, unbiased=False))

    def test_var_stability(self, device):
        tensor = torch.tensor([2281.5, 2281.25], device=device)
        self.assertEqual(tensor.var(dim=0), 0.03125)
        self.assertEqual(tensor.var(), 0.03125)

    def test_sum_dim_reduction_uint8_overflow(self, device):
        example = [[-1, 2, 1], [5, 3, 6]]
        x = torch.tensor(example, dtype=torch.uint8, device=device)
        self.assertEqual(x.sum(dtype=torch.uint8).item(), 16)
        self.assertEqual(x.sum(0, dtype=torch.uint8), torch.tensor([4, 5, 7], dtype=torch.uint8, device=device))
        self.assertEqual(x.sum(1, dtype=torch.uint8), torch.tensor([2, 14], dtype=torch.uint8, device=device))
        y = torch.tensor(example, dtype=torch.uint8, device=device)
        torch.sum(x, 0, out=y)
        self.assertEqual(x.sum(0, dtype=torch.uint8), y)

    def test_dim_reduction_less_than_64(self, device):
        sizes = [1] * 65
        x = torch.randn(sizes, device=device)
        ops = [torch.mean, torch.sum, torch.nansum, torch.std, torch.logsumexp, torch.std, torch.var,
               torch.amin, torch.amax, torch.norm]
        for op in ops:
            with self.assertRaisesRegex(RuntimeError, "only tensors with up to 64 dims are supported"):
                op(x, 64)
            with self.assertRaisesRegex(RuntimeError, "only tensors with up to 64 dims are supported"):
                op(x, -1)

    @unittest.skipIf(not TEST_SCIPY, "SciPy not found")
    def test_logsumexp(self, device):
        from scipy.special import logsumexp
        a = torch.randn(5, 4, device=device)
        a[0, 0] = inf
        a[1, :] = -inf
        actual = a.logsumexp(1)
        expected = logsumexp(a.cpu().numpy(), 1)
        self.assertEqual(expected.shape, actual.shape)
        self.assertEqual(expected, actual)
        # check that out is actually inplace
        b = torch.zeros(5, 2, device=device)
        c = b[:, 0]
        torch.logsumexp(a, 1, out=c)
        self.assertEqual(expected, b[:, 0])

    @onlyCPU
    def test_sum_parallel(self, device):
        # To use parallel branches we'll need to compare on tensors
        # that are relatively large. Even if this is run on a single
        # core machine these tests will still give you signal on
        # the correctness

        def _run_test(size):
            for dim in range(len(size) + 1):
                nv = np.round(np.random.rand(*size))  # 0s and 1s
                tv = torch.from_numpy(nv)
                # Parallelisim is only used if numel is
                # larger than grainsize defined in Parallel.h
                self.assertTrue(tv.numel() > 32768)
                if dim == len(size):
                    nvs = nv.sum()
                    tvs = tv.sum()
                else:
                    nvs = nv.sum(dim)
                    tvs = tv.sum(dim)
                diff = np.abs(nvs - tvs.numpy()).sum()
                self.assertEqual(diff, 0)

        _run_test([2, 3, 3, 3, 3, 2, 2, 3, 2, 3, 2, 3, 3])
        _run_test([4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        _run_test([1, 32 * 8 * 32 * 8])
        _run_test([1, 32770])

    # TODO: kill map2_ (and similar) uses and update to compare with NumPy
    # only works on CPU since this uses map2_, which is only supported on CPU
    def _testCSelection(self, torchfn, mathfn):
        # Two tensors
        size = (100, 100)
        a = torch.rand(*size)
        b = torch.rand(*size)
        c = torchfn(a, b)
        expected_c = torch.zeros(*size)
        expected_c.map2_(a, b, lambda _, a, b: mathfn(a, b))
        self.assertEqual(expected_c, c, atol=0, rtol=0)

    @onlyCPU
    def test_max_elementwise(self, device):
        self._testCSelection(torch.max, max)

    @onlyCPU
    def test_min_elementwise(self, device):
        self._testCSelection(torch.min, min)

    def test_all_any(self, device):
        def test(size):
            x = torch.ones(*size, device=device).byte()
            self.assertTrue(x.all())
            self.assertTrue(x.any())

            x[3] = 0
            self.assertFalse(x.all())
            self.assertTrue(x.any())

            x.zero_()
            self.assertFalse(x.all())
            self.assertFalse(x.any())

            x.fill_(2)
            self.assertTrue(x.all())
            self.assertTrue(x.any())

            x = torch.ones(*size, device=device).bool()
            self.assertTrue(x.all())
            self.assertTrue(x.any())

            x[3] = False
            self.assertFalse(x.all())
            self.assertTrue(x.any())

        test((10,))
        test((5, 5))

    def test_all_any_with_dim(self, device):
        def test(x):
            r1 = x.prod(dim=0, keepdim=False).byte()
            r2 = x.all(dim=0, keepdim=False)
            self.assertEqual(r1.shape, r2.shape)
            self.assertTrue((r1 == r2).all())

            r3 = x.sum(dim=1, keepdim=True).clamp(0, 1).byte()
            r4 = x.any(dim=1, keepdim=True)
            self.assertEqual(r3.shape, r4.shape)
            self.assertTrue((r3 == r4).all())

        test(torch.tensor([[0, 0, 0],
                           [0, 0, 1],
                           [0, 1, 1],
                           [1, 1, 1]], device=device, dtype=torch.uint8))

    def test_numpy_named_args(self, device):
        x1 = torch.randn(10, device=device)
        x2 = torch.randn(10, device=device)
        res1 = torch.add(input=x1, other=x2)
        res2 = torch.add(x1=x1, x2=x2)
        self.assertEqual(res1, res2)

        x1 = torch.randn(10, 10, 10, device=device)
        res1 = x1.sum(dim=(0, 2), keepdim=True)
        res2 = x1.sum(axis=(0, 2), keepdims=True)
        self.assertEqual(res1, res2)

    # TODO: kill this ane replace with common creation ops
    def _make_tensors(self, shape, val_range=(-100, 100), use_floating=True, use_integral=True,
                      use_complex=False) -> Dict[str, List[torch.Tensor]]:
        float_types = [torch.double,
                       torch.float]
        int_types = [torch.int64,
                     torch.int32,
                     torch.int16]

        complex_types = [torch.complex64,
                         torch.complex128]

        def make_contiguous(shape, dtype) -> torch.Tensor:
            if dtype in float_types:
                val = torch.randn(shape, dtype=dtype)
                val = val * ((val_range[1] - val_range[0]) / (math.pi * 2.0))
                val = val + ((val_range[1] - val_range[0]) / 2.0)
                val = torch.clamp(val, min=val_range[0], max=val_range[1])
                return val
            result = torch.zeros(shape, dtype=dtype)
            result.apply_(lambda x: random.randint(val_range[0], val_range[1]))
            return result

        def make_non_contiguous(shape, dtype) -> torch.Tensor:
            contig = make_contiguous(shape, dtype)
            non_contig = torch.empty(shape + (2, 2), dtype=dtype)[..., 0]
            non_contig = non_contig.select(-1, -1)
            non_contig.copy_(contig)
            self.assertFalse(non_contig.is_contiguous())
            return non_contig

        def make_contiguous_slice(size, dtype) -> torch.Tensor:
            contig = make_contiguous((1, size), dtype)
            non_contig = contig[:1, 1:size - 1]
            self.assertTrue(non_contig.is_contiguous())
            return contig

        types = []
        if use_floating:
            types += float_types
        if use_integral:
            types += int_types
        if use_complex:
            types += complex_types
        tensors: Dict[str, List[torch.Tensor]] = {"cont": [], "noncont": [], "slice": []}
        for dtype in types:
            tensors["cont"].append(make_contiguous(shape, dtype))
            tensors["noncont"].append(make_non_contiguous(shape, dtype))
            tensors["slice"].append(make_contiguous_slice(sum(list(shape)), dtype))

        return tensors

    # TODO: refactor this to use comparators from common_utils
    def _assert_matches_numpy(self, t, n):
        self.assertEqual(n.shape, t.shape)
        if t.dtype == torch.float:
            self.assertEqual(n, t, rtol=1e-03, atol=1e-05, equal_nan=True)
        else:
            self.assertEqual(n, t, equal_nan=True)

    # TODO: update this and tests that use it to use the device argument properly
    def _test_dim_ops(self, pytorch_op, numpy_op,
                      use_floating=True, use_integral=True, use_complex=False):
        def do_one(tensors_dict, dim):
            for category, tensors in tensors_dict.items():
                if category == "slice":
                    dim = 0
                for tensor in tensors:
                    # we have no control over NumPy warnings...
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        expected = numpy_op(tensor.cpu().numpy(), dim)
                    actual = pytorch_op(tensor, dim)
                    self._assert_matches_numpy(actual, expected)
                    if torch.cuda.is_available():
                        self._assert_matches_numpy(pytorch_op(tensor.cuda(), dim).cpu(), expected)
        do_one(self._make_tensors((5, 400000), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 1)
        do_one(self._make_tensors((3, 5, 7), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 0)
        do_one(self._make_tensors((3, 5, 7), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 1)
        do_one(self._make_tensors((3, 5, 7), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 2)
        do_one(self._make_tensors((100000, ), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), -1)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 0)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 1)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), 2)
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (1, 2))
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (1, -1))
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (0, 2))
        do_one(self._make_tensors((50, 50, 50), use_floating=use_floating,
                                  use_integral=use_integral, use_complex=use_complex), (0, 2, 1))

    @slowTest
    @onlyCPU
    def test_sum_dim(self, device):
        self._test_dim_ops(
            lambda t, d: t.sum(d),
            lambda n, d: n.sum(d),
            use_floating=True, use_integral=True, use_complex=True)

    @onlyCPU
    def test_mean_dim(self, device):
        self._test_dim_ops(
            lambda t, d: t.mean(d),
            lambda n, d: n.mean(d),
            use_integral=False,
            use_complex=True)

    @onlyCPU
    def test_std_dim(self, device):
        for unbiased in [False, True]:
            self._test_dim_ops(
                lambda t, d: t.std(d, unbiased=unbiased),
                lambda n, d: n.std(d, ddof=1 if unbiased else 0),
                use_integral=False)

    @onlyCPU
    def test_var_dim(self, device):
        for unbiased in [False, True]:
            self._test_dim_ops(
                lambda t, d: t.var(d, unbiased=unbiased),
                lambda n, d: n.var(d, ddof=1 if unbiased else 0),
                use_integral=False)

    @onlyCPU
    @unittest.skipIf(not TEST_SCIPY, 'Scipy not found')
    def test_logsumexp_dim(self, device):
        from scipy.special import logsumexp
        self._test_dim_ops(
            lambda t, d: t.logsumexp(d),
            lambda n, d: logsumexp(n, d),
            use_integral=False)

    # TODO: update this and tests that use it to handle device properly
    def _test_reduce_integer_upcast(self, fn, has_out=True, test_complex=True):
        shape = (3, 4, 5)
        reduced_shape = fn(torch.ones(shape)).shape

        def _test_out(dtype, other_dtype):
            out = torch.ones(reduced_shape, dtype=dtype)
            result = fn(x, out=out)
            self.assertIs(out.dtype, result.dtype)
            self.assertEqual(fn(x.to(dtype)), result, exact_dtype=False)
            result = fn(x, out=out, dtype=dtype)
            self.assertIs(out.dtype, result.dtype)
            self.assertEqual(fn(x.to(dtype)), result, exact_dtype=False)
            # 'out' is favored over dtype, check error
            self.assertRaises(RuntimeError, lambda: fn(x, out=out, dtype=other_dtype))

        for dtype in [dtype for dtype in torch.testing.get_all_math_dtypes('cpu') if dtype != torch.float16]:
            x = torch.ones(shape, dtype=dtype)
            expected_dtype = dtype if dtype.is_floating_point or dtype.is_complex else torch.int64
            self.assertIs(expected_dtype, fn(x).dtype)
            self.assertEqual(fn(x.to(expected_dtype)), fn(x))

            if dtype.is_floating_point:
                other_dtype = torch.float32 if dtype == torch.float64 else torch.float64
            elif dtype.is_complex:
                other_dtype = torch.complex64 if dtype == torch.complex128 else torch.complex128
            else:
                other_dtype = torch.int32 if dtype != torch.int32 else torch.int16
            self.assertIs(other_dtype, fn(x, dtype=other_dtype).dtype)
            self.assertEqual(fn(x.to(other_dtype)), fn(x, dtype=other_dtype), exact_dtype=False)

            # test mixed int/float/complex
            if dtype.is_floating_point:
                mixed_dtypes = [torch.int32, torch.complex64]
            elif dtype.is_complex:
                mixed_dtypes = [torch.int32, torch.float32]
            else:
                mixed_dtypes = [torch.float32, torch.complex64]

            for mixed_dtype in mixed_dtypes:
                self.assertIs(mixed_dtype, fn(x, dtype=mixed_dtype).dtype)
                self.assertEqual(fn(x.to(mixed_dtype)), fn(x, dtype=mixed_dtype), exact_dtype=False)

                if has_out:
                    _test_out(dtype, other_dtype)
                    _test_out(dtype, mixed_dtype)

    @onlyCPU
    def test_sum_integer_upcast(self, device):
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.sum(x, **kwargs), False)
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.sum(x, 0, **kwargs))

    @onlyCPU
    def test_prod_integer_upcast(self, device):
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.prod(x, **kwargs), False)
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.prod(x, 0, **kwargs))

    @onlyCPU
    def test_cumsum_integer_upcast(self, device):
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.cumsum(x, 0, **kwargs))

    @onlyCPU
    def test_cumprod_integer_upcast(self, device):
        self._test_reduce_integer_upcast(lambda x, **kwargs: torch.cumprod(x, 0, **kwargs))

    def test_mode(self, device):
        SIZE = 10
        x = torch.arange(1., SIZE * SIZE + 1, device=device).clone().resize_(SIZE, SIZE)
        x[:2] = 1
        x[:, :2] = 1
        x0 = x.clone()

        # Pre-calculated results.
        res1val = torch.ones(SIZE, device=device)
        # The indices are the position of the last appearance of the mode element.
        res1ind = torch.ones(SIZE, device=device, dtype=torch.long)
        res1ind[0] = SIZE - 1
        res1ind[1] = SIZE - 1

        res2val, res2ind = torch.mode(x, keepdim=False)
        self.assertEqual(res1val, res2val, atol=0, rtol=0)
        self.assertEqual(res1ind, res2ind, atol=0, rtol=0)

        # Test use of result tensor
        res2val = torch.tensor((), device=device)
        res2ind = torch.tensor((), device=device, dtype=torch.long)
        torch.mode(x, keepdim=False, out=(res2val, res2ind))
        self.assertEqual(res1val, res2val, atol=0, rtol=0)
        self.assertEqual(res1ind, res2ind, atol=0, rtol=0)

        # Test non-default dim
        res2val, res2ind = torch.mode(x, 0, False)
        self.assertEqual(res1val, res2val, atol=0, rtol=0)
        self.assertEqual(res1ind, res2ind, atol=0, rtol=0)

        # input unchanged
        self.assertEqual(x, x0, atol=0, rtol=0)

    def _test_mode_intervals(self, shape, intervals, device, v=1):
        x = torch.arange(0, shape[0] * shape[1], device=device)
        x[v] = x.numel()
        x = x.resize_(shape)

        # Set the value of each interval to the mode "v"
        for (beg, end) in intervals:
            x[:, beg:end] = v

        values, indices = torch.mode(x, -1, False)

        # Check whether the returned indices correspond to the returned values
        self.assertTrue((x.gather(1, indices.unsqueeze(1)).t() == values).all())
        # Check whether the returned values are the mode
        self.assertTrue((values == v).all().item())

    @onlyCUDA
    def test_mode_large(self, device):
        # i should be less than (d - 2) / 2
        def testset_for_shape(shape, i):
            d = shape[-1]
            # Mode only in the middle.
            self._test_mode_intervals(shape, [(i, d - i)], device)
            # Mode in discontiguous parts of the input.
            self._test_mode_intervals(shape, [(0, i), (i + 1, d - i - 1), (d - i, d)], device)

        # More than one line of (65535) thread blocks
        testset_for_shape((65536, 10), 3)

        # Max slice size (2048)
        testset_for_shape((10, 2048), 10)

        # Naive kernel for big slice sizes (> 2048)
        testset_for_shape((10, 4096), 10)

    @onlyOnCPUAndCUDA
    def test_mode_wrong_dtype(self, device):
        def test_for_dtypes(x_ty, v_ty, i_ty, message):
            x = torch.ones(10, device=device, dtype=x_ty)
            v = torch.ones(10, device=device, dtype=v_ty)
            i = torch.ones(10, device=device, dtype=i_ty)

            with self.assertRaisesRegex(RuntimeError, message):
                torch.mode(x, -1, True, out=(v, i))

        err_msg = "expected scalar type .* but got .* for "
        values_err = err_msg + "values"
        indices_err = err_msg + "indices"

        test_for_dtypes(torch.uint8, torch.int8, torch.long, values_err)
        test_for_dtypes(torch.int8, torch.int16, torch.long, values_err)
        test_for_dtypes(torch.int32, torch.float32, torch.long, values_err)
        test_for_dtypes(torch.float32, torch.float64, torch.long, values_err)

        test_for_dtypes(torch.uint8, torch.uint8, torch.int8, indices_err)
        test_for_dtypes(torch.int8, torch.int8, torch.int16, indices_err)
        test_for_dtypes(torch.int32, torch.int32, torch.float32, indices_err)
        test_for_dtypes(torch.float32, torch.float32, torch.float64, indices_err)

    @onlyCUDA
    def test_mode_wrong_device(self, device):
        # CPU Input Tensor
        x = torch.ones(2)

        with self.assertRaisesRegex(RuntimeError,
                                    "expected device .* but got .* for values"):
            values = torch.tensor([], device=device)
            torch.mode(x, -1, True, out=(values, torch.tensor([], dtype=torch.long)))

        with self.assertRaisesRegex(RuntimeError,
                                    "expected device .* but got .* for indices"):
            indices = torch.tensor([], device=device)
            torch.mode(x, -1, True, out=(torch.tensor([]), indices))

    # TODO: make work on CUDA, too
    @onlyCPU
    def test_accreal_type(self, device) -> None:
        x = torch.ones(2, 3, 4)
        self.assertIsInstance(x.double().sum().item(), float)
        self.assertIsInstance(x.float().sum().item(), float)
        self.assertIsInstance(x.long().sum().item(), int)
        self.assertIsInstance(x.int().sum().item(), int)
        self.assertIsInstance(x.short().sum().item(), int)
        self.assertIsInstance(x.char().sum().item(), int)
        self.assertIsInstance(x.byte().sum().item(), int)

    def test_var_mean_some_dims(self, device):
        sizes = (4, 6, 7, 5, 3)
        dims = len(sizes)

        x = torch.rand(sizes, device=device)
        for num_of_dims in range(2, dims):
            dim_list = list(combinations(list(range(dims)), r=num_of_dims))
            for dim in dim_list:
                for unbiased in [False, True]:
                    for keepdim in [False, True]:
                        var1, mean1 = torch.var_mean(x, dim=dim, unbiased=unbiased, keepdim=keepdim)
                        var2 = x.var(dim=dim, unbiased=unbiased, keepdim=keepdim)
                        mean2 = x.mean(dim=dim, keepdim=keepdim)
                        self.assertEqual(var1, var2)
                        self.assertEqual(mean1, mean2)

    # TODO: this should be a generic opinfo test
    def test_all_any_empty(self, device):
        x = torch.ByteTensor().to(device)
        self.assertTrue(x.all())
        self.assertFalse(x.any())

        x = torch.BoolTensor().to(device)
        self.assertTrue(x.all())
        self.assertFalse(x.any())

    @dtypesIfCUDA(torch.half, torch.bfloat16, torch.float, torch.double)
    @dtypes(torch.half, torch.bfloat16, torch.float, torch.double)
    def test_max_with_inf(self, device, dtype):
        a = torch.tensor([[-inf, -inf, inf, 3], [inf, inf, -inf, -1]], dtype=dtype, device=device)
        self.assertTrue(torch.all(torch.max(a, dim=1).values == inf).item())
        self.assertTrue(torch.all(torch.amax(a, dim=1) == inf).item())
        self.assertTrue(torch.max(a).item() == inf)
        self.assertTrue(torch.amax(a).item() == inf)

    @dtypesIfCUDA(torch.half, torch.bfloat16, torch.float, torch.double)
    @dtypes(torch.half, torch.float, torch.bfloat16, torch.double)
    def test_min_with_inf(self, device, dtype):
        a = torch.tensor([[-inf, -inf, inf, 3], [inf, inf, -inf, -1]], dtype=dtype, device=device)
        self.assertTrue(torch.all(torch.min(a, dim=1).values == (-inf)).item())
        self.assertTrue(torch.all(torch.amin(a, dim=1) == (-inf)).item())
        self.assertTrue(torch.min(a).item() == -inf)
        self.assertTrue(torch.amin(a).item() == -inf)

    def _test_minmax_helper(self, torchfn, reffn, device, dtype, skip_indices=False):
        def create_input(shape, device, dtype):
            if dtype.is_floating_point:
                return torch.randn(*shape, device=device, dtype=dtype)
            else:
                low = 0 if dtype == torch.bool else -1000
                high = 2 if dtype == torch.bool else 1000
                return torch.randint(low, high, shape, device=device, dtype=dtype)
        x = create_input((100, 100), device, dtype)
        self.compare_with_numpy(torchfn, reffn, x)
        # non contiguous
        x = create_input((10, 10, 10), device, dtype)
        x = x[:, 4]
        self.compare_with_numpy(torchfn, reffn, x)

        def get_values(x):
            if isinstance(x, tuple):
                return x[0]
            return x

        # indices
        if not skip_indices:
            size = 5
            x = create_input((size, size), device, dtype)
            inputs = (x, x.t())
            dims = (0, 1)
            for xinp, d in product(inputs, dims):
                self.compare_with_numpy(lambda x: get_values(torchfn(x, d, False)), lambda x: reffn(x, d, keepdims=False), xinp)
                result = torchfn(xinp, d, False)
                if isinstance(result, tuple):
                    v, i = result
                    if d == 1:
                        self.assertEqual(xinp[torch.arange(size), i], v, atol=0, rtol=0)
                    else:
                        self.assertEqual(xinp[i, torch.arange(size)], v, atol=0, rtol=0)
        # nan
        if dtype.is_floating_point:
            for index in (0, 4, 99):
                x = create_input((100,), device, dtype)
                x[index] = nan
                if not skip_indices:
                    result = torchfn(x, 0)
                    v = get_values(result)
                    self.assertEqual(v, nan)
                    if isinstance(result, tuple):
                        i = result[1]
                        self.assertEqual(i, index)
                self.assertEqual(torchfn(x), nan)

    @dtypesIfCPU(torch.float, torch.double, torch.long, torch.bool, torch.half)
    @dtypesIfCUDA(torch.half, torch.float, torch.long, torch.bool)
    @dtypes(torch.half, torch.float, torch.double)
    def test_max(self, device, dtype):
        self._test_minmax_helper(torch.max, np.amax, device, dtype)

    @dtypesIfCPU(torch.float, torch.double, torch.long, torch.bool, torch.half)
    @dtypesIfCUDA(torch.half, torch.float, torch.long, torch.bool)
    @dtypes(torch.half, torch.float, torch.double)
    def test_min(self, device, dtype):
        self._test_minmax_helper(torch.min, np.amin, device, dtype)

    @dtypesIfCPU(torch.half, torch.float, torch.double, torch.int, torch.long, torch.bool)
    @dtypesIfCUDA(torch.half, torch.float, torch.int, torch.long, torch.bool)
    @dtypes(torch.half, torch.float, torch.double)
    def test_amin(self, device, dtype):
        self._test_minmax_helper(torch.amin, np.amin, device, dtype)

    @dtypesIfCPU(torch.half, torch.float, torch.double, torch.int, torch.long, torch.bool)
    @dtypesIfCUDA(torch.half, torch.float, torch.int, torch.long, torch.bool)
    @dtypes(torch.float, torch.double)
    def test_amax(self, device, dtype):
        self._test_minmax_helper(torch.amax, np.amax, device, dtype)

    @onlyOnCPUAndCUDA
    @dtypesIfCPU(torch.float, torch.double)
    @dtypesIfCUDA(torch.half, torch.float)
    def test_aminmax(self, device, dtype):

        def _amin_wrapper(x, dim=None, keepdims=False):
            if dim is None:
                return torch._aminmax(x)[0]
            else:
                return torch._aminmax(x, dim, keepdims)[0]

        def _amax_wrapper(x, dim=None, keepdims=False):
            if dim is None:
                return torch._aminmax(x)[1]
            else:
                return torch._aminmax(x, dim, keepdims)[1]

        self._test_minmax_helper(_amin_wrapper, np.amin, device, dtype)
        self._test_minmax_helper(_amax_wrapper, np.amax, device, dtype)

    # TODO: bincount isn't a classic reduction -- maybe this test suite is
    #   reductions and summary ops?
    def test_bincount(self, device):
        # negative input throws
        with self.assertRaisesRegex(RuntimeError, '1-d non-negative integral'):
            torch.bincount(torch.tensor([1, -1], device=device))
        # n-d input, with n > 1 throws
        with self.assertRaisesRegex(RuntimeError, '1-d non-negative integral'):
            torch.bincount(torch.tensor([[1, 2], [3, 4]], device=device))
        # floating input type throws
        with self.assertRaisesRegex(RuntimeError, 'not implemented'):
            torch.bincount(torch.tensor([1., 0.3], device=device))
        # minlength < 0 throws
        with self.assertRaisesRegex(RuntimeError, 'minlength should be >= 0'):
            torch.bincount(torch.tensor([1, 3], device=device),
                           torch.tensor([.2, .2], device=device),
                           minlength=-1)
        # input and weights dim mismatch
        with self.assertRaisesRegex(RuntimeError, 'same length'):
            torch.bincount(torch.tensor([1, 0], device=device),
                           torch.tensor([1., 0.3, 0.5], device=device))
        # 1-d input with no elements and default minlength
        self.assertEqual(torch.bincount(torch.tensor([], device=device, dtype=torch.long)),
                         torch.zeros(0, dtype=torch.long, device=device))
        # 1-d input with no elements and specified minlength
        self.assertEqual(torch.bincount(torch.tensor([], device=device, dtype=torch.long), minlength=10),
                         torch.zeros(10, dtype=torch.long, device=device))

        # test tensor method without weights
        long_counts = torch.tensor(
            [0, 3, 2, 1, 3], dtype=torch.uint8, device=device).bincount()
        self.assertEqual(
            torch.tensor([1, 1, 1, 2], dtype=torch.int64, device=device),
            long_counts)
        # test minlength functionality
        int_counts = torch.bincount(
            torch.tensor([1, 1, 1, 1], device=device), minlength=5)
        self.assertEqual(
            torch.tensor([0, 4, 0, 0, 0], dtype=torch.int64, device=device),
            int_counts)
        # test weights
        byte_counts = torch.bincount(
            torch.tensor([0, 1, 1, 1, 4], device=device),
            torch.tensor([.1, .2, .3, .4, .5], device=device))
        self.assertEqual(
            torch.tensor([0.1, 0.9, 0, 0, 0.5], device=device), byte_counts)
        byte_counts = torch.bincount(
            torch.tensor([0, 1, 1, 1, 4], device=device),
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.int8, device=device))
        self.assertEqual(
            torch.tensor([1, 9, 0, 0, 5], device=device, dtype=torch.float64), byte_counts)
        # test non-contiguous inputs and weights
        inputs = torch.tensor([[0, 0], [3, 1], [2, 1], [1, 1], [3, 4]], device=device)
        weights = torch.tensor([[.1, 1], [.2, 2], [.3, 3], [.4, 4], [.5, 5]], device=device)
        for i in [0, 1]:
            assert not inputs[:, i].is_contiguous(), "Inputs are supposed to be non-contiguous"
            assert not weights[:, i].is_contiguous(), "Weights are supposed to be non-contiguous"
        # inputs are non-contiguous but weights are contiguous
        self.assertEqual(inputs[:, 0].bincount(), torch.tensor([1, 1, 1, 2]))
        # inputs and weights are non-contiguous
        self.assertEqual(
            inputs[:, 1].bincount(weights[:, 1]),
            torch.tensor([1, 9, 0, 0, 5], dtype=torch.float32))
        # weights are non-contiguous but inputs are contiguous
        self.assertEqual(inputs[:, 1].contiguous().bincount(weights[:, 1]),
                         torch.tensor([1, 9, 0, 0, 5], dtype=torch.float32))

        # test bincount on non-contiguous slices
        all0s = torch.zeros((32, 2), dtype=torch.int64, device=device)
        self.assertEqual(all0s[:, 0].bincount(), torch.tensor([32]))

        all1s = torch.ones((32, 2), dtype=torch.int64, device=device)
        self.assertEqual(all1s[:, 0].bincount(), torch.tensor([0, 32]))

        # test large number of bins - global memory use
        big_exp = torch.zeros(10000000, device=device)
        big_exp[-1] = 50.0
        big_w = torch.tensor([.5] * 100, device=device)
        big_out = torch.tensor([9999999] * 100, device=device).bincount(big_w)
        self.assertEqual(big_exp, big_out)
        # test large input size
        big_exp = torch.zeros(2, device=device, dtype=torch.int64)
        big_exp[1] = 1000000
        big_out = torch.ones(1000000, dtype=torch.int8, device=device).bincount()
        self.assertEqual(big_exp, big_out)

    # TODO: how many var stability tests are there?
    def test_var_stability2(self, device):
        tensor = torch.FloatTensor([2281.5, 2281.25]).to(device)

        # Stability for inner dim
        self.assertEqual(tensor.var(0), 0.03125)

        # General stability
        self.assertEqual(tensor.var(), 0.03125)

        # Stability for outer dimensions
        tensor = tensor.unsqueeze(1)
        self.assertEqual(tensor.var(0), 0.03125)

    @onlyCPU
    @dtypes(torch.bool, torch.double)
    def test_sum_all(self, device, dtype) -> None:
        def check_sum_all(tensor: torch.Tensor) -> None:
            pylist = tensor.reshape(-1).tolist()
            self.assertEqual(tensor.sum(), sum(pylist))

        if dtype != torch.bool:
            check_sum_all(torch.tensor([1, 2, 3, 4, 5], dtype=dtype, device=device))
            check_sum_all(torch.randn(200000, dtype=dtype, device=device))
            check_sum_all(torch.randn(2000, 2, dtype=dtype, device=device)[:, 0])
        else:
            check_sum_all(torch.tensor([True, False, True], dtype=torch.bool, device=device))

    def _test_memory_format_transformations(self, device, input_generator_fn, transformation_fn,
                                            memory_format, compare_data=True, default_is_preserve=False):

        assert(memory_format == torch.channels_last or memory_format == torch.channels_last_3d)

        # xc is a channels last tensor
        xc = input_generator_fn(device)
        # xc is not memory dense, but looks like channels last
        if memory_format == torch.channels_last:
            xc = xc[..., ::2, ::2]
        else:
            xc = xc[..., ::2, ::2, ::2]

        clone = transformation_fn(xc, memory_format=torch.preserve_format)
        self.assertFalse(clone.is_contiguous())
        self.assertTrue(clone.is_contiguous(memory_format=memory_format))
        self.assertFalse(xc.is_contiguous())
        self.assertFalse(xc.is_contiguous(memory_format=memory_format))
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        xc = input_generator_fn(device)
        clone = transformation_fn(xc, memory_format=torch.contiguous_format)
        self.assertTrue(clone.is_contiguous())
        self.assertFalse(clone.is_contiguous(memory_format=memory_format))
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        xc = input_generator_fn(device)
        clone = transformation_fn(xc)

        if default_is_preserve:
            self.assertFalse(clone.is_contiguous())
            self.assertTrue(clone.is_contiguous(memory_format=memory_format))
        else:
            self.assertTrue(clone.is_contiguous())
            self.assertFalse(clone.is_contiguous(memory_format=memory_format))
        if compare_data:
            self.assertEqual(xc, clone.to(xc))

        x = torch.randn((3, 4, 5, 6, 7, 8, 9), device=device)
        for _ in range(10):
            permutation = list(range(len(x.shape)))
            random.shuffle(permutation)
            x = x.permute(permutation)
            self.assertEqual(x.stride(), transformation_fn(x, memory_format=torch.preserve_format).stride())

    @onlyCPU
    @dtypes(torch.double)
    def test_sum_out(self, device, dtype: torch.dtype) -> None:
        x = torch.rand(100, 100, dtype=dtype, device=device)
        res1 = torch.sum(x, 1)
        res2 = torch.tensor((), dtype=dtype, device=device)
        torch.sum(x, 1, out=res2)
        self.assertEqual(res1, res2)
        x = torch.rand(100, 100, 100, dtype=dtype, device=device)
        res1 = x.sum(2).sum(1)
        res2 = torch.tensor((), dtype=dtype, device=device)
        torch.sum(x, (2, 1), out=res2)
        self.assertEqual(res1, res2)

    @onlyCUDA
    @dtypes(torch.float16, torch.float32)
    def test_prod_gpu(self, device, dtype):
        x = torch.tensor([2, 3, 6, 9, 8], dtype=dtype, device=device)

        # Check all combinations: fp16 input - fp16 output, fp16 input - fp32
        # output, fp32 input - fp16 output, fp32 input - fp32 output
        for dtype_output in [torch.float16, torch.float32]:
            result_expected = torch.tensor(2592, dtype=dtype_output, device=device)
            output = torch.prod(x, dtype=dtype_output)
            self.assertEqual(output, result_expected)

            output = x.prod(dtype=dtype_output)
            self.assertEqual(output, result_expected)

    @onlyCPU
    @dtypes(torch.float)
    def test_prod(self, device, dtype):
        x = torch.rand(100, 100, dtype=dtype, device=device)
        res1 = torch.prod(x, 1)
        res2 = torch.tensor((), dtype=dtype, device=device)
        torch.prod(x, 1, out=res2)
        self.assertEqual(res1, res2)

    def test_prod_bool(self, device):
        vals = [[True, True], [True, False], [False, False], []]
        for val in vals:
            result = torch.prod(torch.tensor(val, device=device), dtype=torch.bool).item()
            expect = np.prod(np.array(val), dtype=np.bool)
            self.assertEqual(result, expect)

            result = torch.prod(torch.tensor(val, device=device)).item()
            expect = np.prod(np.array(val))
            self.assertEqual(result, expect)

    @onlyCPU
    def test_max_mixed_devices(self, device):
        a = torch.randn(10, device=device)
        if torch.cuda.is_available():
            values = torch.randn(10).cuda()
            indices = torch.cuda.LongTensor()
            self.assertRaises(RuntimeError,
                              lambda: torch.max(a, 0, out=(values, indices)))
            self.assertRaises(RuntimeError,
                              lambda: torch.amax(a, 0, out=values))

    @onlyCPU
    def test_min_mixed_devices(self, device):
        a = torch.randn(10, device=device)
        if torch.cuda.is_available():
            values = torch.randn(10).cuda()
            indices = torch.cuda.LongTensor()
            self.assertRaises(RuntimeError,
                              lambda: torch.min(a, 0, out=(values, indices)))
            self.assertRaises(RuntimeError,
                              lambda: torch.amin(a, 0, out=values))

    # TODO: consider refactoring with bincount test
    def test_bucketization(self, device):
        values_1d = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], device=device)
        values_3d = torch.tensor([[[1, 3, 5], [2, 4, 6]], [[1, 2, 3], [4, 5, 6]]], device=device)

        # regular case 3d boundary and 3d input value
        boundaries = torch.tensor([[[1, 2, 3, 4], [3, 4, 5, 6]], [[1, 3, 5, 7], [2, 4, 6, 8]]], device=device)
        expected_result = torch.tensor([[[0, 2, 4], [0, 1, 3]], [[0, 1, 1], [1, 2, 2]]], device=device)
        output = torch.empty(2, 2, 3, device=device, dtype=torch.int64)
        self.assertEqual(torch.searchsorted(boundaries, values_3d), expected_result)
        self.assertEqual(torch.searchsorted(boundaries, values_3d, out=output), expected_result)
        expected_result = torch.tensor([[[1, 3, 4], [0, 2, 4]], [[1, 1, 2], [2, 2, 3]]], device=device)
        self.assertEqual(torch.searchsorted(boundaries, values_3d, right=True), expected_result)
        self.assertEqual(torch.searchsorted(boundaries, values_3d, right=True, out=output), expected_result)

        # simple 1d boundary and 3d input value
        boundaries = torch.tensor([1, 2, 3, 4, 5, 6], device=device)
        expected_result = torch.tensor([[[0, 2, 4], [1, 3, 5]], [[0, 1, 2], [3, 4, 5]]], device=device)
        output = torch.empty(2, 2, 3, device=device, dtype=torch.int64)
        self.assertEqual(torch.searchsorted(boundaries, values_3d), expected_result)
        self.assertEqual(torch.bucketize(values_3d, boundaries), expected_result)
        self.assertEqual(torch.bucketize(values_3d, boundaries, out=output), expected_result)
        expected_result = torch.tensor([[[1, 3, 5], [2, 4, 6]], [[1, 2, 3], [4, 5, 6]]], device=device)
        self.assertEqual(torch.searchsorted(boundaries, values_3d, right=True), expected_result)
        self.assertEqual(torch.bucketize(values_3d, boundaries, right=True), expected_result)
        self.assertEqual(torch.bucketize(values_3d, boundaries, out=output, right=True), expected_result)

        # simple float 1d boundary and 1d input with output int32 type
        values_1d_float = values_1d.to(torch.float32)
        boundaries = torch.tensor([0.9, 1, 2, 2, 3, 3, 4, 4.1, 9, 9], device=device, dtype=torch.float32)
        expected_result = torch.tensor([1, 2, 4, 6, 8, 8, 8, 8, 8], device=device, dtype=torch.int32)
        self.assertEqual(torch.searchsorted(boundaries, values_1d_float, out_int32=True), expected_result)
        self.assertEqual(torch.bucketize(values_1d_float, boundaries, out_int32=True), expected_result)

        # multiple dimension input with 0 elements
        boundaries = torch.tensor([1, 2, 3, 4, 5, 6], device=device, dtype=torch.int64)
        values_0_el = torch.tensor([[[]]], device=device, dtype=torch.int64)
        expected_result = values_0_el.to(torch.int64)
        self.assertEqual(torch.searchsorted(boundaries, values_0_el), expected_result)
        self.assertEqual(torch.bucketize(values_0_el, boundaries), expected_result)

        # nan input
        values_nan = torch.tensor([1.0, float('nan'), 2.0, float('nan')], device=device, dtype=torch.float64)
        boundaries = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device, dtype=torch.float64)
        expected_result = torch.tensor([1, 4, 2, 4], device=device)
        self.assertEqual(torch.searchsorted(boundaries, values_nan), expected_result)
        expected_result = torch.tensor([2, 4, 3, 4], device=device)
        self.assertEqual(torch.searchsorted(boundaries, values_nan, right=True), expected_result)

        # type promotion and non contiguous tensors
        values_3d_permute = values_3d.permute(2, 1, 0).to(torch.int32)
        boundaries_permute = values_3d.permute(2, 1, 0).to(torch.float64)
        expected_result = torch.tensor([[[0, 0], [0, 1]], [[2, 0], [0, 1]], [[2, 0], [0, 0]]], device=device)
        if self.device_type != 'xla':
            self.assertWarnsRegex(
                UserWarning, "tensor is non-contiguous",
                lambda: self.assertEqual(torch.searchsorted(boundaries_permute, values_3d_permute), expected_result))
        else:
            # All tensors in XLA is contiguous even doing permute, no warning msg will be generate in XLA
            self.assertEqual(torch.searchsorted(boundaries_permute, values_3d_permute), expected_result)

        # scalar type
        boundaries = torch.tensor([1.5, 2.5, 3.5], device=device)
        expected_result = torch.tensor(1, device=device)
        self.assertEqual(torch.searchsorted(boundaries, 2), expected_result)
        self.assertEqual(torch.bucketize(torch.tensor(2, device=device), boundaries), expected_result)
        expected_result = torch.tensor(3, device=device)
        scalar_tensor_nan = torch.tensor(float('nan'), device=device)
        self.assertEqual(torch.searchsorted(boundaries, scalar_tensor_nan), expected_result)
        self.assertEqual(torch.bucketize(float('nan'), boundaries, right=True), expected_result)

        # invalid input dimensions
        boundaries = torch.tensor([[1, 2, 3], [4, 5, 6]], device=device)
        with self.assertRaisesRegex(
                RuntimeError, "first N-1 dimensions of boundaries tensor and input value tensor must match"):
            torch.searchsorted(boundaries, values_3d)
        with self.assertRaisesRegex(
                RuntimeError, "boundaries tensor must be 1 dimension"):
            torch.bucketize(values_3d, boundaries)
        with self.assertRaisesRegex(
                RuntimeError, "only when boundaries tensor dimension is 1"):
            torch.searchsorted(boundaries, 1)

        # incompatiable output tensor's dtype
        def test_output_dtype(dtype, is_int32):
            output = values_1d.to(dtype)
            with self.assertRaisesRegex(
                    RuntimeError, "output tensor's dtype is wrong"):
                torch.searchsorted(values_1d, values_1d, out=output, out_int32=is_int32)

        test_output_dtype(torch.float32, False)
        test_output_dtype(torch.int32, False)
        test_output_dtype(torch.int64, True)

    @dtypesIfCUDA(torch.half, torch.float, torch.double,
                  torch.int8, torch.short, torch.int, torch.long)
    @dtypes(torch.float, torch.double,
            torch.int8, torch.short, torch.int, torch.long)
    def test_nansum(self, device, dtype):
        x = (torch.randn(3, 3))
        if dtype in [torch.half, torch.float, torch.double]:
            x[x < 0.2] = float('nan')
        # Randomly scale the values
        x = (x * random.randint(10, 100)).tolist()

        self.compare_with_numpy(torch.nansum, np.nansum, x, device, dtype)

    def _test_reduction_function_with_numpy(self, torch_func, np_func, device, dtype,
                                            with_extremal=False, atol=None, rtol=None,
                                            exact_dtype=True, with_keepdim=False):
        # Test 0-d to 3-d tensors.
        for ndims in range(0, 4):
            shape = _rand_shape(ndims, min_size=5, max_size=10)
            for n in range(ndims + 1):
                for c in combinations(list(range(ndims)), n):
                    for count_dim in permutations(c):
                        # Generate Input.
                        x = _generate_input(shape, dtype, device, with_extremal)

                        if count_dim == ():
                            # Default `dims=None` case
                            self.compare_with_numpy(torch_func, np_func, x, device=None, dtype=None,
                                                    atol=atol, rtol=rtol, exact_dtype=exact_dtype)
                        else:
                            # With `dims: tuple of ints` case
                            if with_keepdim:
                                torch_func_partial = partial(torch_func, keepdim=True, dim=count_dim)
                                np_func_partial = partial(np_func, keepdims=True, axis=count_dim)
                            else:
                                torch_func_partial = partial(torch_func, dim=count_dim)
                                np_func_partial = partial(np_func, axis=count_dim)
                            self.compare_with_numpy(torch_func_partial, np_func_partial, x, device=None, dtype=None,
                                                    atol=atol, rtol=rtol, exact_dtype=exact_dtype)

    @dtypes(*(torch.testing.get_all_int_dtypes() + torch.testing.get_all_fp_dtypes(include_bfloat16=False) +
              torch.testing.get_all_complex_dtypes()))
    def test_count_nonzero(self, device, dtype):
        self._test_reduction_function_with_numpy(torch.count_nonzero, np.count_nonzero, device, dtype)
        self._test_reduction_function_with_numpy(torch.count_nonzero, np.count_nonzero, device, dtype, True)

    def _test_sum_reduction_vs_numpy(self, torch_fn, np_fn, device, dtype, with_keepdim=False, with_extremal=False):
        def is_integral(dtype):
            return dtype in torch.testing.get_all_int_dtypes()

        # On Windows CI, the current version of `numpy` promotes all lower integers
        # dtypes to int32 while `torch` promotes them to int64. Hence we skip on checking
        # the exact dtype.
        # Reference : https://dr.pytorch.org/api/view-log-full?build_id=122051580
        # PR : https://github.com/pytorch/pytorch/pull/38628#issuecomment-655905370
        exact_dtype = False if (IS_WINDOWS and is_integral(dtype)) else True

        if dtype == torch.uint8:
            with self.assertRaises(TypeError):
                self._test_reduction_function_with_numpy(torch_fn, np_fn, device, dtype, with_extremal=with_extremal)
        else:
            # TODO: Investigate why the output is not close to numpy.
            if dtype == torch.float16:
                atol = 0.4
                rtol = 1e-2
            elif dtype == torch.float32:
                atol = 7e-05
                rtol = 3e-06
            else:
                # Default values
                atol = None
                rtol = None
            self._test_reduction_function_with_numpy(torch_fn, np_fn, device, dtype,
                                                     atol=atol, rtol=rtol, exact_dtype=exact_dtype,
                                                     with_keepdim=with_keepdim, with_extremal=with_extremal)

    @onlyOnCPUAndCUDA
    @dtypes(*(torch.testing.get_all_int_dtypes() + torch.testing.get_all_fp_dtypes(include_bfloat16=False)))
    def test_sum_vs_numpy(self, device, dtype):
        self._test_sum_reduction_vs_numpy(torch.sum, np.sum, device, dtype)
        self._test_sum_reduction_vs_numpy(torch.sum, np.sum, device, dtype, with_extremal=True)
        self._test_sum_reduction_vs_numpy(torch.sum, np.sum, device, dtype, with_keepdim=True)

    @onlyOnCPUAndCUDA
    @dtypes(*(torch.testing.get_all_int_dtypes() + torch.testing.get_all_fp_dtypes(include_bfloat16=False)))
    def test_nansum_vs_numpy(self, device, dtype):
        self._test_sum_reduction_vs_numpy(torch.nansum, np.nansum, device, dtype)
        self._test_sum_reduction_vs_numpy(torch.nansum, np.nansum, device, dtype, with_extremal=True)
        self._test_sum_reduction_vs_numpy(torch.nansum, np.nansum, device, dtype, with_keepdim=True)

    @dtypes(*(torch.testing.get_all_complex_dtypes()))
    def test_nansum_complex(self, device, dtype):
        x = torch.randn((3, 3, 3), device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, "nansum does not support complex inputs"):
            torch.nansum(x)

    def test_nansum_out_dtype(self, device):
        dtypes = list(torch.testing.get_all_int_dtypes() + torch.testing.get_all_fp_dtypes(include_bfloat16=False))
        for inp_dtype, out_dtype in combinations(dtypes, 2):
            shape = _rand_shape(random.randint(2, 5), min_size=5, max_size=10)
            x = _generate_input(shape, inp_dtype, device, with_extremal=False)
            torch_fn = partial(torch.nansum, dtype=out_dtype)
            np_out_dtype = torch_to_numpy_dtype_dict[out_dtype]
            np_fn = partial(np.nansum, dtype=np_out_dtype)
            self.compare_with_numpy(torch_fn, np_fn, x, device=None, dtype=None)

    @dtypes(*(torch.testing.get_all_int_dtypes() + torch.testing.get_all_fp_dtypes(include_bfloat16=False)))
    def test_argminmax_multiple(self, device, dtype):
        # Case: All Ones
        t = torch.ones(3, 3, device=device, dtype=dtype)
        self.compare_with_numpy(torch.argmax, np.argmax, t)
        self.compare_with_numpy(torch.argmin, np.argmin, t)

        # Case: With single `nan` present.
        if dtype in torch.testing.get_all_fp_dtypes():
            t[2, 2] = float('nan')
            self.compare_with_numpy(torch.argmax, np.argmax, t)
            self.compare_with_numpy(torch.argmin, np.argmin, t)

        # Case: Randomly Generated Tensors
        for ndims in range(1, 5):
            shape = _rand_shape(ndims, min_size=5, max_size=10)
            for with_extremal in [False, True]:
                for contiguous in [False, True]:
                    # Generate Input.
                    x = _generate_input(shape, dtype, device, with_extremal)

                    if dtype == torch.half:
                        max_val = torch.max(x.to(torch.float))
                        min_val = torch.min(x.to(torch.float))
                    else:
                        max_val = torch.max(x)
                        min_val = torch.min(x)

                    mask = torch.randn(x.shape) > 0.5
                    x[mask] = torch.tensor(max_val + 1, dtype=dtype)

                    mask = torch.randn(x.shape) > 0.5
                    x[mask] = torch.tensor(min_val - 1, dtype=dtype)

                    if not contiguous:
                        x = x.T

                    self.compare_with_numpy(torch.argmax, np.argmax, x, device=None, dtype=None)
                    self.compare_with_numpy(torch.argmin, np.argmin, x, device=None, dtype=None)

                    # Verify indices returned by max and min.
                    if dtype != torch.half:
                        rand_dim = random.randint(0, ndims - 1)
                        self.compare_with_numpy(lambda x: torch.max(x, dim=rand_dim)[1],
                                                lambda x: np.argmax(x, axis=rand_dim), x, device=None, dtype=None)
                        self.compare_with_numpy(lambda x: torch.min(x, dim=rand_dim)[1],
                                                lambda x: np.argmin(x, axis=rand_dim), x, device=None, dtype=None)

        def verify_against_numpy(t):
            # Argmax
            torch_fn = partial(torch.argmax, dim=1)
            np_fn = partial(np.argmax, axis=1)
            self.compare_with_numpy(torch_fn, np_fn, t)
            # Non-contiguous input
            self.compare_with_numpy(torch_fn, np_fn, t.T)

            # Verify indices returned by max.
            if dtype != torch.half:
                self.compare_with_numpy(lambda x: torch.max(x, dim=1)[1], np_fn, x, device=None, dtype=None)
                self.compare_with_numpy(lambda x: torch.max(x, dim=1)[1], np_fn, x.T, device=None, dtype=None)

            # Argmin
            torch_fn = partial(torch.argmin, dim=1)
            np_fn = partial(np.argmin, axis=1)
            self.compare_with_numpy(torch_fn, np_fn, t)
            # Non-contiguous input
            self.compare_with_numpy(torch_fn, np_fn, t.T)

            # Verify indices returned by min.
            if dtype != torch.half:
                self.compare_with_numpy(lambda x: torch.min(x, dim=1)[1], np_fn, x, device=None, dtype=None)
                self.compare_with_numpy(lambda x: torch.min(x, dim=1)[1], np_fn, x.T, device=None, dtype=None)

        # Case: Sample from issue: https://github.com/pytorch/pytorch/issues/41998
        t = torch.tensor([[1, 5],
                          [2, 10],
                          [3, 3]], device=device, dtype=dtype)
        verify_against_numpy(t)

        # Case: Sample from issue: https://github.com/pytorch/pytorch/issues/41998
        t = torch.tensor([[1, 5],
                          [2, 10],
                          [0, 0]], device=device, dtype=dtype)
        verify_against_numpy(t)

    @dtypes(*(torch.testing.get_all_dtypes(include_half=True, include_bfloat16=False,
                                           include_bool=True, include_complex=True)))
    def test_all_any_vs_numpy(self, device, dtype):
        # Note [all, any uint8 compatibility]: However for compatibility reason,
        # for `uint8`, they return Tensor of same dtype `uint8`.
        # Reference: https://github.com/pytorch/pytorch/pull/47878#issuecomment-747108561
        exact_dtype = True if dtype != torch.uint8 else False

        def _test_all_any(x):
            self.compare_with_numpy(torch.all, np.all, x)
            self.compare_with_numpy(torch.any, np.any, x)

        def _test_all_any_with_dim(x, dim):
            torch_fn = partial(torch.all, dim=dim)
            np_fn = partial(np.all, axis=dim)
            self.compare_with_numpy(torch_fn, np_fn, x, exact_dtype=exact_dtype)

            torch_fn = partial(torch.any, dim=dim)
            np_fn = partial(np.any, axis=dim)
            self.compare_with_numpy(torch_fn, np_fn, x, exact_dtype=exact_dtype)

        def _test_out_variant(x, dim):
            out = torch.empty_like(x)
            if dtype == torch.bool or dtype == torch.uint8:
                expected = torch.all(x, dim)
                torch.all(x, dim, out=out)
                self.assertEqual(expected, out)

                expected = torch.any(x, dim)
                torch.any(x, dim, out=out)
                self.assertEqual(expected, out)
            else:
                with self.assertRaisesRegex(RuntimeError, "all only supports bool tensor for result, got"):
                    torch.all(x, dim, out=out)

                with self.assertRaisesRegex(RuntimeError, "any only supports bool tensor for result, got"):
                    torch.any(x, dim, out=out)

        def _test_all_any_with_dim_keepdim(x, dim, keepdim):
            torch_fn = partial(torch.all, dim=dim, keepdim=keepdim)
            np_fn = partial(np.all, axis=dim, keepdims=keepdim)
            self.compare_with_numpy(torch_fn, np_fn, x, exact_dtype=exact_dtype)

            torch_fn = partial(torch.any, dim=dim, keepdim=keepdim)
            np_fn = partial(np.any, axis=dim, keepdims=keepdim)
            self.compare_with_numpy(torch_fn, np_fn, x, exact_dtype=exact_dtype)

        def _test_output_dtype(x):
            # This test will fail once the functions return bool output
            # for uint8 input.
            expected_dtype = torch.uint8 if dtype == torch.uint8 else torch.bool
            self.assertEqual(torch.all(x).dtype, expected_dtype)
            self.assertEqual(torch.any(x).dtype, expected_dtype)

            self.assertEqual(torch.all(x, dim=0).dtype, expected_dtype)
            self.assertEqual(torch.any(x, dim=0).dtype, expected_dtype)

        for ndim in range(5):
            shape = _rand_shape(ndim, 1, 5)
            x = _generate_input(shape, dtype, device, with_extremal=False)
            _test_all_any(x)
            _test_all_any(x.T)
            _test_all_any(x[..., ::2])

            x = _generate_input(shape, dtype, device, with_extremal=True)
            _test_all_any(x)
            _test_all_any(x.T)
            _test_all_any(x[..., ::2])

            x = torch.zeros_like(x)
            _test_all_any(x)
            _test_all_any(x.T)
            _test_all_any(x[..., ::2])

            x = torch.ones_like(x)
            _test_all_any(x)
            _test_all_any(x.T)
            _test_all_any(x[..., ::2])
            _test_output_dtype(x)
            for dim in range(ndim):
                x = _generate_input(shape, dtype, device, with_extremal=False)
                _test_all_any_with_dim(x, dim)
                _test_all_any_with_dim(x.T, dim)
                _test_all_any_with_dim(x[..., ::2], dim)
                _test_out_variant(x, dim)
                _test_all_any_with_dim_keepdim(x, dim, keepdim=True)
                _test_all_any_with_dim_keepdim(x, dim, keepdim=False)

                x = _generate_input(shape, dtype, device, with_extremal=True)
                _test_all_any_with_dim(x, dim)
                _test_all_any_with_dim(x.T, dim)
                _test_all_any_with_dim(x[..., ::2], dim)
                _test_out_variant(x, dim)
                _test_all_any_with_dim_keepdim(x, dim, keepdim=True)
                _test_all_any_with_dim_keepdim(x, dim, keepdim=False)

                x = torch.zeros_like(x)
                _test_all_any_with_dim(x, dim)
                _test_all_any_with_dim(x.T, dim)
                _test_all_any_with_dim(x[..., ::2], dim)
                _test_out_variant(x, dim)
                _test_all_any_with_dim_keepdim(x, dim, keepdim=True)
                _test_all_any_with_dim_keepdim(x, dim, keepdim=False)

                x = torch.ones_like(x)
                _test_all_any_with_dim(x, dim)
                _test_all_any_with_dim(x.T, dim)
                _test_all_any_with_dim(x[..., ::2], dim)
                _test_out_variant(x, dim)
                _test_all_any_with_dim_keepdim(x, dim, keepdim=True)
                _test_all_any_with_dim_keepdim(x, dim, keepdim=False)

    # TODO: part of this test covers torch.norm, with should be covered by test_linalg
    @onlyOnCPUAndCUDA
    def test_repeated_dim(self, device):
        ops = [torch.mean, torch.sum, torch.nansum, torch.std, torch.logsumexp, torch.std, torch.var,
               torch.amin, torch.amax, torch.norm]
        x = torch.randn(3, 3, 3, 3, device=device)

        error_msg = r'appears multiple times in the list of dims'
        norm_error_msg = r'Expected dims to be different, got'
        for op in ops:
            for dim in [(0, 0), (0, -4)]:
                e_msg = norm_error_msg if op == torch.norm else error_msg
                with self.assertRaisesRegex(RuntimeError, e_msg):
                    op(x, dim=dim)

    # TODO: update this test to comapre against NumPy
    @onlyCUDA
    def test_var(self, device):
        cpu_tensor = torch.randn(2, 3, 3)
        device_tensor = cpu_tensor.to(device)
        self.assertEqual(device_tensor.var(), cpu_tensor.var())
        self.assertEqual(device_tensor.var(1), cpu_tensor.var(1))
        self.assertEqual(device_tensor.var(2), cpu_tensor.var(2))
        self.assertEqual(device_tensor.std(), cpu_tensor.std())
        self.assertEqual(device_tensor.std(1), cpu_tensor.std(1))
        self.assertEqual(device_tensor.var(2), cpu_tensor.var(2))

        cpu_tensor = torch.randn(100)
        device_tensor = cpu_tensor.to(device)
        self.assertEqual(device_tensor.var(), cpu_tensor.var())

    # TODO: update this test to compare against NumPy
    @onlyCUDA
    def test_var_large_input(self, device):
        # Large, not-nice input
        cpu_tensor = torch.randn(2 * 32 * 1024 + 1, 2, 67)
        device_tensor = cpu_tensor.to(device)

        self.assertEqual(cpu_tensor.var(2), device_tensor.var(2))

    # TODO: update this to compare against NumPy instead of CPU
    @onlyCUDA
    @dtypes(torch.double)
    def test_sum_noncontig(self, device, dtype):
        x = torch.randn(1, 75, 57, 20, dtype=dtype, device=device).permute(0, 3, 1, 2)
        y = x.cpu()
        self.assertEqual(x.sum().cpu(), y.sum())
        self.assertEqual(x.sum(dim=(-1, -2)).cpu(), y.sum(dim=(-1, -2)))
        self.assertEqual(x.sum(dim=(1, 3)).cpu(), y.sum(dim=(1, 3)))

    # TODO: update this to compare against NumPy instead of CPU
    @onlyCUDA
    def test_min_max_nan(self, device):
        tests = [(lambda x: x.min(), 'min'),
                 (lambda x: x.max(), 'max'),
                 (lambda x: x.amin(), 'amin'),
                 (lambda x: x.amax(), 'amax'),
                 (lambda x: x.min(0).values, 'min_dim'),
                 (lambda x: x.max(0).values, 'max_dim'),
                 (lambda x: x.amin(0), 'amin_dim'),
                 (lambda x: x.amax(0), 'amax_dim')]
        for f, name in tests:
            a = torch.arange(25.0).view(5, 5)
            a[2, 2] = nan
            actual = f(a.to(device)).cpu()
            expected = f(a).cpu()
            self.assertEqual(torch.isnan(actual), torch.isnan(expected), msg='nans for {}'.format(name))
            self.assertEqual(actual[~torch.isnan(actual)],
                             expected[~torch.isnan(expected)], msg='nans for {}'.format(name))

    # TODO: make this test generic using OpInfos
    @onlyCUDA
    def test_sum_cpu_device_mismatch(self, device):
        x = torch.randn(20, dtype=torch.float32, device=device)
        y = torch.randn(1, dtype=torch.float32)

        err_string = "Expected all tensors to be on the same device, but found at least two devices, {0}".format(device)

        with self.assertRaisesRegex(RuntimeError, err_string):
            torch.sum(x, dim=[0], dtype=torch.float32, out=y)

        # tests half to float promotion
        if self.device_type == 'cuda':
            x = x.half()
            with self.assertRaisesRegex(RuntimeError, err_string):
                torch.sum(x, dim=[0], dtype=torch.float32, out=y)

    # Assert for illegal dtype would not be raised on XLA
    @onlyOnCPUAndCUDA
    def test_minmax_illegal_dtype(self, device):
        x = torch.randn(5, 5, dtype=torch.float32, device=device)
        valid_values = torch.empty(5, dtype=torch.float32, device=device)
        valid_indices = torch.empty(5, dtype=torch.long, device=device)
        illegal_values = torch.empty(5, dtype=torch.int, device=device)
        illegal_indices = torch.empty(5, dtype=torch.double, device=device)
        torch.max(x, dim=0, out=(valid_values, valid_indices))
        torch.min(x, dim=0, out=(valid_values, valid_indices))
        torch.amax(x, dim=0, out=valid_values)
        torch.amin(x, dim=0, out=valid_values)
        rmsg = r'scalar type|dtype'
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.max(x, dim=0, out=(illegal_values, valid_indices))
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.min(x, dim=0, out=(illegal_values, valid_indices))
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.amax(x, dim=0, out=illegal_values)
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.amin(x, dim=0, out=illegal_values)
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.max(x, dim=0, out=(valid_values, illegal_indices))
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.min(x, dim=0, out=(valid_values, illegal_indices))
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.max(x, dim=0, out=(illegal_values, illegal_indices))
        with self.assertRaisesRegex(RuntimeError, rmsg):
            torch.min(x, dim=0, out=(illegal_values, illegal_indices))

    @dtypes(*torch.testing.get_all_dtypes(include_bool=False, include_complex=False))
    def test_dim_arg_reduction_scalar(self, device, dtype):
        example = 4.0

        x = torch.tensor(example, device=device, dtype=dtype)
        self.assertEqual(x.argmax().item(), 0)
        self.assertEqual(x.argmax(dim=None).item(), 0)
        self.assertEqual(x.argmax(dim=0).item(), 0)
        self.assertEqual(x.argmax(dim=0, keepdim=True), torch.tensor(0, dtype=torch.int64))

        x = torch.tensor(example, device=device, dtype=dtype)
        self.assertEqual(x.argmin().item(), 0)
        self.assertEqual(x.argmin(dim=None).item(), 0)
        self.assertEqual(x.argmin(dim=0).item(), 0)
        self.assertEqual(x.argmin(dim=0, keepdim=True), torch.tensor(0, dtype=torch.int64))


    @precisionOverride({torch.float16: 1e-2, torch.bfloat16: 1e-2})
    @dtypes(*(set(torch.testing.get_all_dtypes(include_bool=False, include_complex=False)) - {torch.uint8}))
    def test_dim_reduction(self, device, dtype):
        example = [[-1, 2, 1], [5, 3, 6]]

        sum_dtype = {
            torch.bfloat16: torch.bfloat16,
            torch.double: torch.double,
            torch.float: torch.float,
            torch.half: torch.half,
            torch.int64: torch.int64,
            torch.int32: torch.int64,
            torch.int16: torch.int64,
            torch.int8: torch.int64
        }

        # This won't test for 256bit instructions, since we usually
        # only work on 1 cacheline (512bit) at a time and these
        # examples aren't big enough to trigger that.
        x = torch.tensor(example, device=device, dtype=dtype)
        self.assertEqual(x.sum().item(), 16)
        self.assertEqual(x.sum(0), torch.tensor([4, 5, 7], dtype=sum_dtype[dtype]))
        self.assertEqual(x.sum(1), torch.tensor([2, 14], dtype=sum_dtype[dtype]))
        y = torch.tensor(example, device=device, dtype=sum_dtype[dtype])
        torch.sum(x, 0, out=y)
        self.assertEqual(x.sum(0), y)

        # Mean not supported for Int types
        if dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            x = torch.tensor(example, device=device, dtype=dtype)
            self.assertEqual(x.mean().item(), 16.0 / 6)
            self.assertEqual(x.mean(0), torch.tensor([2.0, 2.5, 7.0 / 2], dtype=dtype))
            self.assertEqual(x.mean(1), torch.tensor([2.0 / 3, 14.0 / 3], dtype=dtype))
            self.assertEqual(x.mean(), x.mean((0, 1)))

        prod_dtype = {
            torch.bfloat16: torch.bfloat16,
            torch.double: torch.double,
            torch.float: torch.float,
            torch.float16: torch.float16,
            torch.int64: torch.int64,
            torch.int32: torch.int64,
            torch.int16: torch.int64,
            torch.int8: torch.int64,
        }

        # prod is not supported for float16 & bfloat16 on CPU
        if not (self.device_type == 'cpu' and dtype in [torch.float16, torch.bfloat16]):
            x = torch.tensor(example, device=device, dtype=dtype)
            self.assertEqual(x.prod().item(), -180)
            self.assertEqual(x.prod(0), torch.tensor([-5, 6, 6], dtype=prod_dtype[dtype]))
            self.assertEqual(x.prod(1), torch.tensor([-2, 90], dtype=prod_dtype[dtype]))

        x = torch.tensor(example, device=device, dtype=dtype)

        self.assertEqual(x.min().item(), -1)
        self.assertEqual(x.argmin().item(), 0)

        # TODO: torch.min does not support the same operation as argmin
        # for the same case, should we enable it?
        self.assertEqual(x.argmin(dim=None).item(), 0)

        self.assertEqual(x.min(0), (torch.tensor([-1, 2, 1], dtype=dtype),
                                    torch.tensor([0, 0, 0], dtype=torch.int64)))
        self.assertEqual(x.amin(0), torch.tensor([-1, 2, 1], dtype=dtype))
        self.assertEqual(x.argmin(0), torch.tensor([0, 0, 0], dtype=torch.int64))

        self.assertEqual(x.min(dim=0, keepdim=True), (torch.tensor([[-1, 2, 1]], dtype=dtype),
                         torch.tensor([[0, 0, 0]], dtype=torch.int64)))
        self.assertEqual(x.amin(dim=0, keepdim=True), torch.tensor([[-1, 2, 1]], dtype=dtype))
        self.assertEqual(x.argmin(dim=0, keepdim=True), torch.tensor([[0, 0, 0]], dtype=torch.int64))

        self.assertEqual(x.min(1), (torch.tensor([-1, 3], dtype=dtype),
                         torch.tensor([0, 1], dtype=torch.int64)))
        self.assertEqual(x.amin(1), torch.tensor([-1, 3], dtype=dtype))
        self.assertEqual(x.argmin(1), torch.tensor([0, 1], dtype=torch.int64))

        self.assertEqual(x.min(dim=1, keepdim=True), (torch.tensor([[-1], [3]], dtype=dtype),
                         torch.tensor([[0], [1]], dtype=torch.int64)))
        self.assertEqual(x.amin(dim=1, keepdim=True), torch.tensor([[-1], [3]], dtype=dtype))
        self.assertEqual(x.argmin(dim=1, keepdim=True), torch.tensor([[0], [1]], dtype=torch.int64))

        # test that non-contiguous tensors work
        self.assertEqual(x[:, :2].min().item(), -1)
        self.assertEqual(x[:, :2].amin().item(), -1)
        self.assertEqual(x[:, :2].argmin().item(), 0)

        x = torch.tensor(example, device=device, dtype=dtype)

        self.assertEqual(x.max().item(), 6)
        self.assertEqual(x.amax().item(), 6)
        self.assertEqual(x.argmax().item(), 5)

        self.assertEqual(x.max(0), (torch.tensor([5, 3, 6], dtype=dtype),
                                    torch.tensor([1, 1, 1], dtype=torch.int64)))
        self.assertEqual(x.amax(0), torch.tensor([5, 3, 6], dtype=dtype))
        self.assertEqual(x.argmax(dim=0), torch.tensor([1, 1, 1], dtype=torch.int64))

        self.assertEqual(x.max(dim=0, keepdim=True), (torch.tensor([[5, 3, 6]], dtype=dtype),
                                                      torch.tensor([[1, 1, 1]], dtype=torch.int64)))
        self.assertEqual(x.amax(dim=0, keepdim=True), torch.tensor([[5, 3, 6]], dtype=dtype))
        self.assertEqual(x.argmax(dim=0, keepdim=True), torch.tensor([[1, 1, 1]], dtype=torch.int64))

        self.assertEqual(x.max(1), (torch.tensor([2, 6], dtype=dtype),
                                    torch.tensor([1, 2], dtype=torch.int64)))
        self.assertEqual(x.amax(1), torch.tensor([2, 6], dtype=dtype))
        self.assertEqual(x.argmax(dim=1), torch.tensor([1, 2], dtype=torch.int64))

        self.assertEqual(x.max(1, keepdim=True), (torch.tensor([[2], [6]], dtype=dtype),
                                                  torch.tensor([[1], [2]], dtype=torch.int64)))
        self.assertEqual(x.amax(1, keepdim=True), torch.tensor([[2], [6]], dtype=dtype))
        self.assertEqual(x.argmax(dim=1, keepdim=True), torch.tensor([[1], [2]], dtype=torch.int64))

        # test that non-contiguous tensors work
        self.assertEqual(x[:, :2].max().item(), 5)
        self.assertEqual(x[:, :2].amax().item(), 5)
        self.assertEqual(x[:, :2].argmax().item(), 2)

        dim_red_fns = [
            "mean", "median", "nanmedian", "mode", "norm", "prod",
            "std", "sum", "var", "max", "min", "amax", "amin"]

        def normfn_attr(t, dim, keepdim=False, out=None):
            attr = torch.norm
            return attr(t, 2, dim, keepdim, out=out)

        for fn_name in dim_red_fns:
            fn_attr = getattr(torch, fn_name) if fn_name != "norm" else normfn_attr

            def fn(x, dim, keepdim=False, out=None):
                ans = fn_attr(x, dim, keepdim=keepdim, out=out)
                return ans if not isinstance(ans, tuple) else ans[0]

            def fn_tuple(x, dim, keepdim=False, out=None):
                return fn_attr(x, dim, keepdim=keepdim, out=out)

            def test_multidim(x, dim):
                self.assertEqual(fn(x, dim).unsqueeze(dim), fn(x, dim, keepdim=True))
                self.assertEqual(x.ndimension() - 1, fn(x, dim).ndimension())
                self.assertEqual(x.ndimension(), fn(x, dim, keepdim=True).ndimension())

            # general case
            x = torch.randn(3, 4, 5, device=device)
            dim = random.randint(0, 2)
            test_multidim(x, dim)

            # check 1-d behavior
            x = torch.randn(1, device=device)
            dim = 0
            self.assertEqual(fn(x, dim).shape, ())
            self.assertEqual(fn(x, dim, keepdim=True).shape, (1,))

            # check reducing of a singleton dimension
            dims = [3, 4, 5]
            singleton_dim = random.randint(0, 2)
            dims[singleton_dim] = 1
            x = torch.randn(dims, device=device)
            test_multidim(x, singleton_dim)

            # check reducing with output kwargs
            if fn_name in ['median', 'nanmedian', 'mode', 'max', 'min']:
                y = torch.randn(5, 3, device=device)
                values = torch.randn(5, 3, device=device)
                indices = torch.zeros(5, 3, device=device).long() - 1
                fn_tuple(y, 1, keepdim=False, out=(values[:, 1], indices[:, 1]))
                values_expected, indices_expected = fn_tuple(y, 1, keepdim=False)
                self.assertEqual(values[:, 1], values_expected,
                                 msg='{} values with out= kwarg'.format(fn_name))
                self.assertEqual(indices[:, 1], indices_expected,
                                 msg='{} indices with out= kwarg'.format(fn_name))
                continue

            x = torch.randn(5, 3, device=device)
            y = torch.randn(5, 3, device=device)
            fn(y, 1, keepdim=False, out=x[:, 1])
            expected = fn(y, 1, keepdim=False)
            self.assertEqual(x[:, 1], expected, msg='{} with out= kwarg'.format(fn_name))

    @onlyCUDA
    @largeTensorTest('10GB')
    def test_reduction_split(self, device):
        # Test reduction when there is a 32bit-indexing split
        # https://github.com/pytorch/pytorch/issues/37583
        input_ = torch.randn(5, 14400, 14400, device=device)
        result = input_.sum(dim=0)
        expect = input_[0] + input_[1] + input_[2] + input_[3] + input_[4]
        self.assertEqual(result, expect)

    @onlyCUDA
    @dtypes(torch.half, torch.float, torch.double, torch.bfloat16)
    def test_reduction_vectorize_along_input_corner(self, device, dtype):
        # 1D case: sum
        size = 1024 * 1024 * 64 + 3
        shift = 1
        x = torch.zeros(size, dtype=dtype, device=device)
        y = x[shift:]
        for i in range(100):
            x.zero_()
            x[i] = 1
            self.assertEqual(x.sum(), 1.0)
            if i < shift:
                self.assertEqual(y.sum(), 0.0)
            else:
                self.assertEqual(y.sum(), 1.0)
        for i in range(1, 100):
            x.zero_()
            x[-i] = 1
            self.assertEqual(x.sum(), 1.0)
            self.assertEqual(y.sum(), 1.0)
        # 1D case: argmax
        size = 1024 * 1024 * 64 + 3
        shift = 1
        ysize = size - shift
        x = torch.zeros(size, dtype=dtype, device=device)
        y = x[shift:]
        for i in range(100):
            x.zero_()
            x[i] = 1
            self.assertEqual(x.argmax().item(), i)
            if i >= shift:
                self.assertEqual(y.argmax().item(), i - shift)
        for i in range(1, 100):
            x.zero_()
            x[-i] = 1
            self.assertEqual(x.argmax().item(), size - i)
            self.assertEqual(y.argmax().item(), ysize - i)
        # 2D case: sum
        size = (7, 1024 * 1024 + 3)
        x = torch.zeros(size, dtype=dtype, device=device)
        for i in range(100):
            x.zero_()
            for j in range(7):
                x[j][i] = j
            xs = x.sum(dim=-1)
            for j in range(7):
                self.assertEqual(xs[j].item(), float(j))
        for i in range(100):
            x.zero_()
            for j in range(7):
                x[j][-i] = j
            xs = x.sum(dim=-1)
            for j in range(7):
                self.assertEqual(xs[j].item(), float(j))
        # 2D case: max/argmax
        size = (7, 1024 * 1024 + 3)
        x = torch.zeros(size, dtype=dtype, device=device)
        for i in range(100):
            x.zero_()
            for j in range(7):
                x[j][i] = j + 1
            xs1 = x.argmax(dim=-1)
            xs2 = x.max(dim=-1).indices
            for j in range(7):
                self.assertEqual(xs1[j].item(), i)
                self.assertEqual(xs2[j].item(), i)
        for i in range(1, 100):
            x.zero_()
            for j in range(7):
                x[j][-i] = j + 1
            xs1 = x.argmax(dim=-1)
            xs2 = x.max(dim=-1).indices
            for j in range(7):
                self.assertEqual(xs1[j].item(), size[1] - i)
                self.assertEqual(xs2[j].item(), size[1] - i)
        # 2D case: min/argmin
        size = (7, 1024 * 1024 + 3)
        x = torch.zeros(size, dtype=dtype, device=device)
        for i in range(100):
            x.zero_()
            for j in range(7):
                x[j][i] = -(j + 1)
            xs1 = x.argmin(dim=-1)
            xs2 = x.min(dim=-1).indices
            for j in range(7):
                self.assertEqual(xs1[j].item(), i)
                self.assertEqual(xs2[j].item(), i)
        for i in range(1, 100):
            x.zero_()
            for j in range(7):
                x[j][-i] = -(j + 1)
            xs1 = x.argmin(dim=-1)
            xs2 = x.min(dim=-1).indices
            for j in range(7):
                self.assertEqual(xs1[j].item(), size[1] - i)
                self.assertEqual(xs2[j].item(), size[1] - i)

    @onlyCUDA
    @dtypes(torch.half, torch.float, torch.double, torch.bfloat16)
    def test_reduction_vectorize_along_output(self, device, dtype):
        def run_test(input_):
            M, N = input_.shape
            input_.zero_()
            for i in range(min(M, N)):
                input_[i][i] = 1
            output1 = input_.argmax(dim=0)
            output2 = input_.sum(dim=0)
            for i in range(min(M, N)):
                self.assertEqual(output1[i], i)
                self.assertEqual(output2[i], 1)
        # vec 4
        run_test(torch.zeros(64, 64, dtype=dtype, device=device))
        # vec 2
        run_test(torch.zeros(64 * 64 + 2, dtype=dtype, device=device)[2:].view(64, 64))
        run_test(torch.zeros(64, 62, dtype=dtype, device=device))
        run_test(torch.zeros(64, 2, dtype=dtype, device=device))
        # vec 1
        run_test(torch.zeros(64 * 64 + 1, dtype=dtype, device=device)[1:].view(64, 64))
        run_test(torch.zeros(64, 61, dtype=dtype, device=device))
        run_test(torch.zeros(64, 1, dtype=dtype, device=device))

    @slowTest
    def test_argminmax_large_axis(self, device):
        # Regression test for gh-32863
        x = torch.zeros(2**31, device=device, dtype=torch.int8)
        x[-1] = 1
        self.assertEqual(x.argmax(0), x.shape[0] - 1)
        self.assertEqual(x.max(0).indices, x.shape[0] - 1)
        x[-1] = -1
        self.assertEqual(x.argmin(0), x.shape[0] - 1)
        self.assertEqual(x.min(0).indices, x.shape[0] - 1)

    def test_argminmax_axis_with_dim_one(self, device):
        # See: https://github.com/pytorch/pytorch/issues/38922
        n = 32768
        x = torch.zeros(1, n)
        self.assertEqual(x.argmax(dim=0), torch.zeros(n, dtype=torch.int64))
        self.assertEqual(x.argmin(dim=0), torch.zeros(n, dtype=torch.int64))

        self.assertEqual(x.argmax(dim=-2), torch.zeros(n, dtype=torch.int64))
        self.assertEqual(x.argmin(dim=-2), torch.zeros(n, dtype=torch.int64))

        self.assertEqual(x.argmax(dim=0, keepdim=True), torch.zeros(1, n, dtype=torch.int64))
        self.assertEqual(x.argmin(dim=0, keepdim=True), torch.zeros(1, n, dtype=torch.int64))

        self.assertEqual(x.argmax(dim=-2, keepdim=True), torch.zeros(1, n, dtype=torch.int64))
        self.assertEqual(x.argmin(dim=-2, keepdim=True), torch.zeros(1, n, dtype=torch.int64))

    @dtypes(torch.int, torch.long, torch.float, torch.double)
    @dtypesIfCUDA(torch.int, torch.long, torch.half, torch.float, torch.double)
    def test_median_real_values(self, device, dtype):
        # Generate random 0-3D sizes
        sizes = [random.sample(range(1, 32), i) for i in range(4) for _ in range(2)]
        for size in sizes:
            # Create random input tensor
            t = torch.randn(size, device=device).type(dtype)
            t_numpy = t.cpu().numpy()
            res = t.median()
            self.assertEqual(res, t.nanmedian())
            k = int((t.numel() - 1) / 2)
            self.assertEqual(res, t.view(-1).sort()[0][k])
            if t.numel() % 2 == 1:
                # We can only test agains numpy for odd reductions because numpy
                # returns the mean of the two medians and torch returns the lower
                self.assertEqual(res.cpu().numpy(), np.median(t_numpy))
            for dim in range(t.ndim):
                res = t.median(dim, True)
                self.assertEqual(res, t.nanmedian(dim, True))
                size = t.size(dim) if t.ndim > 0 else 1
                k = int((size - 1) / 2)
                self.assertEqual(res[0], (t.sort(dim)[0]).select(dim, k).unsqueeze_(dim))
                self.assertEqual(res[0], t.gather(dim, res[1]))
                if size % 2 == 1:
                    # We can only test agains numpy for odd reductions because numpy
                    # returns the mean of the two medians and torch returns the lower
                    self.assertEqual(res[0].cpu().numpy(), np.median(t_numpy, dim, keepdims=True))

    @dtypes(torch.float, torch.double)
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    def test_median_nan_values(self, device, dtype):
        # Generate random 0-3D sizes
        sizes = [random.sample(range(1, 32), i) for i in range(4) for _ in range(2)]
        for size in sizes:
            # Create random input tensor with nan values
            t = torch.rand(size, device=device, dtype=dtype)
            t.masked_fill_(t < 0.1, float('nan'))
            t_numpy = t.cpu().numpy()
            for op in [torch.median, torch.nanmedian]:
                numpy_op = np.median if op == torch.median else np.nanmedian
                res = op(t)
                num_nan = t.isnan().sum()
                if op == torch.median and num_nan > 0:
                    k = t.numel() - 1
                else:
                    k = int((t.numel() - num_nan - 1) / 2)
                self.assertEqual(res, t.view(-1).sort()[0][k])
                if (t.numel() - num_nan) % 2 == 1:
                    # We can only test agains numpy for odd reductions because numpy
                    # returns the mean of the two medians and torch returns the lower
                    self.assertEqual(res.item(), numpy_op(t.cpu().numpy()))
                for dim in range(t.ndim):
                    res = op(t, dim, True)
                    size = t.size(dim) if t.ndim > 0 else 1
                    num_nan = t.isnan().sum(dim, True)
                    if op == torch.median:
                        k = torch.where(num_nan > 0, size - 1, int((size - 1) / 2))
                    else:
                        k = ((size - num_nan - 1) / 2).type(torch.long)
                    self.assertEqual(res[0], (t.sort(dim)[0]).gather(dim, k))
                    self.assertEqual(res[0], t.gather(dim, res[1]))
                    # We can only test agains numpy for odd reductions because numpy
                    # returns the mean of the two medians and torch returns the lower
                    mask = (size - num_nan) % 2 == 1
                    res = res[0].masked_select(mask).cpu()
                    ref = numpy_op(t_numpy, dim, keepdims=True)[mask.cpu().numpy()]
                    self.assertEqual(res, torch.from_numpy(ref))

    def test_median_corner_cases(self, device):
        def check(op, a, args, key):
            t = torch.tensor(a, device=device)
            res = op(t, *args)
            if not args:
                key = torch.tensor(key, device=device)
            else:
                if len(key) == 1:
                    key = torch.tensor(key[0], device=device)
                    res = res[0]
                else:
                    key = (torch.tensor(key[0], device=device), torch.tensor(key[1], device=device))
            self.assertEqual(res, key)

        nan = float('nan')
        check(torch.median, nan, [], nan)
        check(torch.nanmedian, nan, [], nan)
        check(torch.median, nan, [0], [nan, 0])
        check(torch.nanmedian, nan, [0], [nan, 0])
        check(torch.median, [nan], [0, True], [[nan], [0]])
        check(torch.nanmedian, [nan], [0, True], [[nan], [0]])
        check(torch.median, [nan], [0, True], [[nan], [0]])
        check(torch.nanmedian, [nan], [0, True], [[nan], [0]])

        # Indices are not deterministic here so can only check values
        check(torch.median, [[nan, nan], [1, 2]], [0], [[nan, nan]])
        check(torch.nanmedian, [[nan, nan], [1, 2]], [0], [[1, 2.]])
        check(torch.median, [[nan, nan], [1, 2]], [1], [[nan, 1]])
        check(torch.nanmedian, [[nan, nan], [1, 2]], [1], [[nan, 1.]])

        # Discontiguous and strided tensors
        a = torch.arange(12, device=device)
        self.assertEqual(a[::2].median(), torch.tensor(4, device=device))
        self.assertEqual(a[::2].nanmedian(), torch.tensor(4, device=device))

        a.resize_(3, 4)
        self.assertEqual(a.T.median(), torch.tensor(5, device=device))
        self.assertEqual(a.T.nanmedian(), torch.tensor(5, device=device))
        self.assertEqual(a[::2, ::2].median(-1)[0], torch.tensor([0, 8], device=device))
        self.assertEqual(a[::2, ::2].nanmedian(-1)[0], torch.tensor([0, 8], device=device))

        a.resize_(2, 3, 2)
        self.assertEqual(a.T.median(), torch.tensor(5, device=device))
        self.assertEqual(a.T.nanmedian(), torch.tensor(5, device=device))
        self.assertEqual(a[:, ::2, :].median(-1)[0], torch.tensor([[0, 4], [6, 10]], device=device))
        self.assertEqual(a[:, ::2, :].nanmedian(-1)[0], torch.tensor([[0, 4], [6, 10]], device=device))


    @onlyOnCPUAndCUDA
    @dtypes(torch.float, torch.double)
    def test_quantile(self, device, dtype):
        # Generate some random test cases
        ops = ['quantile', 'nanquantile']
        inputs = [tuple(np.random.randint(2, 10, size=i)) for i in range(1, 4)]
        quantiles = [tuple(np.random.rand(i)) for i in range(0, 5)]
        keepdims = [True, False]

        # Add corner cases
        inputs.extend([0.75, (1,), (1, 1), (1, 2, 1)])
        inputs.extend([[float('nan')], [[float('nan'), float('nan')], [1, 2]]])
        inputs.extend([[[float('nan'), float('nan')], [float('nan'), 2]]])
        quantiles.extend([0.5, [0., 1.], np.random.rand(10)])

        # Enumerate all input combinations
        for op, x, q, keepdim in product(ops, inputs, quantiles, keepdims):
            if type(x) is tuple:
                a = torch.randn(x, dtype=dtype, device=device)
                # Make some random elements NaN
                a.masked_fill_(torch.randint_like(a, 20) == 0, float('nan'))
            else:
                a = torch.tensor(x, dtype=dtype, device=device)

            q = torch.tensor(q, dtype=dtype, device=device)

            torch_op = getattr(torch, op)
            numpy_op = getattr(np, op)

            # Compute quantile along every dimension and flattened tensor
            interpolations = ('linear', 'lower', 'higher', 'midpoint', 'nearest')
            for interpolation, dim in product(interpolations,
                                              [None] + list(range(a.ndim))):
                result = torch_op(a, q, dim=dim, keepdim=keepdim, interpolation=interpolation)
                expected = numpy_op(a.cpu().numpy(), q.cpu().numpy(), dim,
                                    interpolation=interpolation, keepdims=keepdim)
                self.assertEqual(result.cpu(), torch.from_numpy(np.array(expected)).type(result.type()))

                # Test out variation
                out = torch.empty_like(result)
                torch_op(a, q, dim=dim, keepdim=keepdim, interpolation=interpolation, out=out)
                self.assertEqual(out.cpu(), result.cpu())

    def test_quantile_backward(self, device):
        def check(a, q, dim, expected_grad, ops=(torch.quantile, torch.nanquantile)):
            for op in ops:
                t = torch.tensor(a, device=device, requires_grad=True)
                op(t, torch.tensor(q, device=device), dim).sum().backward()
                self.assertEqual(t.grad, expected_grad)

        check([1., 2, 3], 0.5, 0, [0, 1, 0])
        check([1., 2, 3, 4], 0.5, 0, [0, 0.5, 0.5, 0])
        check([3., 1, 4, 2], 0.5, 0, [0.5, 0, 0, 0.5])
        check([1., 2, 3, 4], [0.25, 0.5, 0.75], 0, [0.25, 1.25, 1.25, 0.25])
        check([[1., 2], [2, 1]], 0., 0, [[1, 0], [0, 1]])
        check([[1., 2], [4, 3]], 1., 1, [[0, 1], [1, 0]])
        check([1, float('nan'), 2], 0.5, 0, [0, 1, 0], [torch.quantile])
        check([1, float('nan'), 2], 0.5, 0, [0.5, 0, 0.5], [torch.nanquantile])

    def test_quantile_error(self, device):
        def check(a, q, args, kwargs, message):
            with self.assertRaisesRegex(RuntimeError, r'quantile\(\) ' + message):
                at = torch.tensor(a, device=device)
                qt = torch.tensor(q, device=device) if isinstance(q, list) else q
                torch.quantile(at, qt, *args, **kwargs)

        check([], 0.5, [], {}, r'input tensor must be non-empty')
        check([1.], [[1.]], [], {}, r'q must be a scalar or 1D tensor')
        check([1], 0.5, [], {}, r'input tensor must be either float or double dtype')
        check([1.], [1], [], {}, r'q tensor must be same dtype as the input tensor')
        check([1.], -1., [], {}, r'q must be in the range \[0, 1\] but got -1')
        check([1.], 1.1, [], {}, r'q must be in the range \[0, 1\] but got 1.1')
        check([1.], 0.5, [], {'out': torch.empty([], dtype=torch.int32, device=device)},
              r'out tensor must be same dtype as the input tensor')
        check([1.], [1.], [None, False], {'interpolation': 'random_mode'},
              r"interpolation must be one of linear, lower, higher, midpoint or nearest, but got random_mode")

        if self.device_type == "cpu":
            check([1.], [0.5, 1.1, -1], [], {}, r'q values must be in the range \[0, 1\]')

        if self.device_type == "cuda":
            with self.assertRaisesRegex(
                    RuntimeError, r'quantile\(\) q tensor must be on the same device as the input tensor'):
                torch.randn(1, device=device).quantile(torch.tensor(0.5))
            with self.assertRaisesRegex(
                    RuntimeError, r'quantile\(\) out tensor must be on the same device as the input tensor'):
                torch.quantile(torch.randn(1, device=device), 0.5, out=torch.scalar_tensor(1))

    def test_std_mean(self, device):
        x = torch.rand(100, 50, 20, device=device)
        for dim in range(x.dim()):
            for unbiased in [False, True]:
                for keepdim in [False, True]:
                    std1, mean1 = torch.std_mean(x, dim=dim, unbiased=unbiased, keepdim=keepdim)
                    std2 = x.std(dim=dim, unbiased=unbiased, keepdim=keepdim)
                    mean2 = x.mean(dim=dim, keepdim=keepdim)
                    self.assertEqual(std1, std2)
                    self.assertEqual(mean1, mean2)

    def test_std_mean_all_dims(self, device):
        x = torch.rand(100, 50, 20, device=device)
        for unbiased in [False, True]:
            std1, mean1 = torch.std_mean(x, unbiased=unbiased)
            std2 = x.std(unbiased=unbiased)
            mean2 = x.mean()
            self.assertEqual(std1, std2)
            self.assertEqual(mean1, mean2)

    def test_var_mean(self, device):
        x = torch.rand(100, 300, 50, device=device)
        for dim in range(x.dim()):
            for unbiased in [False, True]:
                for keepdim in [False, True]:
                    var1, mean1 = torch.var_mean(x, dim=dim, unbiased=unbiased, keepdim=keepdim)
                    var2 = x.var(dim=dim, unbiased=unbiased, keepdim=keepdim)
                    mean2 = x.mean(dim=dim, keepdim=keepdim)
                    self.assertEqual(var1, var2)
                    self.assertEqual(mean1, mean2)

    def test_var_mean_all_dims(self, device):
        x = torch.rand(100, 50, 20, device=device)
        for unbiased in [False, True]:
            var1, mean1 = torch.var_mean(x, unbiased=unbiased)
            var2 = x.var(unbiased=unbiased)
            mean2 = x.mean()
            self.assertEqual(var1, var2)
            self.assertEqual(mean1, mean2)

    def test_std_mean_some_dims(self, device):
        sizes = (4, 6, 7, 5, 3)
        dims = len(sizes)
        x = torch.rand(sizes, device=device)
        for num_of_dims in range(2, dims):
            dim_list = list(combinations(list(range(dims)), r=num_of_dims))
            for dim in dim_list:
                for unbiased in [False, True]:
                    for keepdim in [False, True]:
                        std1, mean1 = torch.std_mean(x, dim=dim, unbiased=unbiased, keepdim=keepdim)
                        std2 = x.std(dim=dim, unbiased=unbiased, keepdim=keepdim)
                        mean2 = x.mean(dim=dim, keepdim=keepdim)
                        self.assertEqual(std1, std2)
                        self.assertEqual(mean1, mean2)

    def _compare_std_var_with_numpy(self, op, device, dtype, input, dim,
                                    keepdim, unbiased, use_out):
        a = input.cpu().numpy() if input.dtype is not torch.bfloat16 else input.float().cpu().numpy()
        numpy_kwargs = {
            'axis' : dim,
            'keepdims' : keepdim,
            'ddof' : 1 if unbiased else 0,
        }

        if dim is None:
            del numpy_kwargs['axis']
            del numpy_kwargs['keepdims']

        if op == 'var':
            torch_op = torch.var
            numpy_op = np.var
        elif op == 'std':
            torch_op = torch.std
            numpy_op = np.std
        else:
            self.fail("Unknown op!")

        numpy_result = numpy_op(a, **numpy_kwargs)

        if dim is None and use_out is False:
            torch_result = torch_op(input, unbiased)
        elif dim is not None and use_out is False:
            torch_result = torch_op(input, dim, unbiased, keepdim)
        elif dim is not None and use_out is True:
            out = torch.empty(0, device=device, dtype=dtype)
            torch_result = torch_op(input, dim, unbiased, keepdim, out=out)
        else:
            out = torch.empty(0, device=device, dtype=dtype)
            try:
                torch_result = torch_op(input, dim, unbiased, keepdim, out=out)
            except RuntimeError:
                return
            self.fail("Failed to hit RuntimeError!")

        exact_dtype = input.dtype != torch.bfloat16
        self.assertEqual(torch_result, numpy_result, exact_dtype=exact_dtype)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_var_vs_numpy(self, device, dtype):
        _size = (20, 20)

        for test_case in product((torch.randn(_size, device=device, dtype=dtype),),
                                 (None, 0, 1),
                                 (False, True),
                                 (False, True),
                                 (False, True),):
            self._compare_std_var_with_numpy('var', device, dtype, *test_case)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_std_vs_numpy(self, device, dtype):
        _size = (20, 20)

        for test_case in product((torch.randn(_size, device=device, dtype=dtype),),
                                 (None, 0, 1),
                                 (False, True),
                                 (False, True),
                                 (False, True),):
            self._compare_std_var_with_numpy('std', device, dtype, *test_case)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_var_correction_vs_numpy(self, device, dtype):
        _size = (20, 20)
        test_args = [
            *product(
                # dim
                (None, 0, 1),
                # correction
                (None, 0, 10, 30),
                # keepdim
                (False, True),
            ),
            [None, -100, True],  # Negative correction
        ]

        tensor = make_tensor(_size, device=device, dtype=dtype)
        array = tensor.cpu().numpy()

        for dim, correction, keepdim in test_args:
            numpy_kwargs = dict(axis=dim, ddof=correction, keepdims=keepdim)
            if correction is None:
                # NumPy default is not compatible with torch.std (gh-50010)
                numpy_kwargs['ddof'] = 1

            numpy_res = np.asarray(np.var(array, **numpy_kwargs))
            torch_res = torch.var(tensor, dim=dim, correction=correction, keepdim=keepdim)

            # inf vs. nan results are sensitive to machine precision,
            # just treat them as equivalent
            numpy_res[np.isinf(numpy_res)] = np.nan
            torch_res[torch_res.isinf()] = np.nan

            self.assertEqual(torch_res, numpy_res)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_std_correction_vs_numpy(self, device, dtype):
        _size = (20, 20)
        test_args = [
            *product(
                # dim
                (None, 0, 1),
                # correction
                (None, 0, 10, 30),
                # keepdim
                (False, True),
            ),
            [None, -100, True],  # Negative correction
        ]

        tensor = make_tensor(_size, device=device, dtype=dtype)
        array = tensor.cpu().numpy()

        for dim, correction, keepdim in test_args:
            numpy_kwargs = dict(axis=dim, ddof=correction, keepdims=keepdim)
            if correction is None:
                # NumPy default is incompatible with torch.std (gh-50010)
                numpy_kwargs['ddof'] = 1

            numpy_res = np.asarray(np.std(array, **numpy_kwargs))
            torch_res = torch.std(tensor, dim=dim, correction=correction, keepdim=keepdim)

            # inf vs. nan results are sensitive to machine precision,
            # just treat them as equivalent
            numpy_res[np.isinf(numpy_res)] = np.nan
            torch_res[torch_res.isinf()] = np.nan

            self.assertEqual(torch_res, numpy_res)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_std_mean_correction(self, device, dtype):
        _size = (20, 20)
        test_args = [
            *product(
                # dim
                (None, 0, 1),
                # correction
                (None, 0, 10, 30),
                # keepdim
                (False, True),
            ),
            [None, -100, True],  # Negative correction
        ]

        tensor = make_tensor(_size, device=device, dtype=dtype)

        for dim, correction, keepdim in test_args:
            kwargs = dict(dim=dim, correction=correction, keepdim=keepdim)
            std1 = torch.std(tensor, **kwargs)
            if dim is not None:
                mean1 = torch.mean(tensor, dim=dim, keepdim=keepdim)
            else:
                mean1 = torch.mean(tensor)
                if keepdim:
                    mean1 = mean1.reshape((1,) * tensor.ndim)
            std2, mean2 = torch.std_mean(tensor, **kwargs)

            self.assertEqual(std1, std2)
            self.assertEqual(mean1, mean2)

    @dtypes(torch.float, torch.double, torch.cfloat, torch.cdouble)
    def test_var_mean_correction(self, device, dtype):
        _size = (20, 20)
        test_args = [
            *product(
                # dim
                (None, 0, 1),
                # correction
                (None, 0, 10, 30),
                # keepdim
                (False, True),
            ),
            [None, -100, True],  # Negative correction
        ]

        tensor = make_tensor(_size, device=device, dtype=dtype)

        for dim, correction, keepdim in test_args:
            kwargs = dict(dim=dim, correction=correction, keepdim=keepdim)
            var1 = torch.var(tensor, **kwargs)
            if dim is not None:
                mean1 = torch.mean(tensor, dim=dim, keepdim=keepdim)
            else:
                mean1 = torch.mean(tensor)
                if keepdim:
                    mean1 = mean1.reshape((1,) * tensor.ndim)
            var2, mean2 = torch.var_mean(tensor, **kwargs)

            self.assertEqual(var1, var2)
            self.assertEqual(mean1, mean2)

    def test_amin_amax_some_dims(self, device):
        sizes = (4, 6, 7, 5, 3)
        dims = len(sizes)
        x = torch.rand(sizes, device=device)
        for num_of_dims in range(2, dims):
            dim_list = list(combinations(list(range(dims)), r=num_of_dims))
            for dim in dim_list:
                for keepdim in [False, True]:
                    amin1 = torch.amin(x, dim=dim, keepdim=keepdim)
                    amax1 = torch.amax(x, dim=dim, keepdim=keepdim)
                    amin2 = x
                    amax2 = x
                    for i, d in enumerate(dim):
                        if not keepdim:
                            d -= i
                        amin2 = torch.amin(amin2, dim=d, keepdim=keepdim)
                        amax2 = torch.amax(amax2, dim=d, keepdim=keepdim)
                    self.assertEqual(amin1, amin2)
                    self.assertEqual(amax1, amax2)

    def test_histc(self, device):
        # negative nbins throws
        with self.assertRaisesRegex(RuntimeError, 'bins must be > 0'):
            torch.histc(torch.tensor([1], dtype=torch.float, device=device), bins=-1)
        # empty tensor
        actual = torch.histc(torch.tensor([], device=device), min=0, max=3)
        expected = torch.zeros(100, dtype=torch.float, device=device)
        self.assertEqual(expected, actual)

        # without nbins
        actual = torch.histc(
            torch.tensor([2, 5], dtype=torch.float, device=device))
        expected = torch.zeros(100, dtype=torch.float, device=device)
        expected[0] = 1
        expected[99] = 1
        self.assertEqual(expected, actual)
        # tensor with the same element
        actual = torch.histc(torch.ones(5, dtype=torch.float, device=device), bins=5)
        self.assertEqual(
            torch.tensor([0, 0, 5, 0, 0], dtype=torch.float, device=device),
            actual)
        # no element falls between [min, max]
        actual = torch.histc(
            torch.ones(5, dtype=torch.float, device=device), bins=5, min=2, max=3)
        self.assertEqual(
            torch.tensor([0, 0, 0, 0, 0], dtype=torch.float, device=device),
            actual)
        # element falls below min + integral bin size and
        actual = torch.histc(
            torch.tensor([2, 4, 2, 2, 5, 4], dtype=torch.float, device=device),
            bins=5, min=1, max=5)
        self.assertEqual(
            torch.tensor([0, 3, 0, 2, 1], dtype=torch.float, device=device),
            actual)
        # non-integral bin size
        actual = torch.histc(
            torch.tensor([1, 2, 1], dtype=torch.float, device=device),
            bins=4, min=0, max=3)
        self.assertEqual(
            torch.tensor([0, 2, 1, 0], dtype=torch.float, device=device),
            actual)
        # double input
        actual = torch.histc(
            torch.tensor([1, 2, 1], dtype=torch.double, device=device), bins=4, min=0, max=3)
        self.assertEqual(
            torch.tensor([0, 2, 1, 0], dtype=torch.double, device=device),
            actual)
        self.assertEqual(actual.dtype, torch.double)
        # mixed input
        actual = torch.histc(
            torch.tensor([1., 2, 1], dtype=torch.float, device=device),
            bins=4, min=0, max=3)
        self.assertEqual(
            torch.tensor([0, 2, 1, 0], dtype=torch.float, device=device),
            actual)
        self.assertEqual(actual.dtype, torch.float)
        # scalar input and 1 bin -- should return a 1-dimensional tensor, not a scalar.
        actual = torch.histc(
            torch.tensor(0, dtype=torch.float, device=device),
            bins=1, min=0, max=3)
        self.assertEqual(
            torch.tensor([1], dtype=torch.float, device=device),
            actual)
        # tensors with inf; min, max not provided -- should throw a RuntimeError
        with self.assertRaisesRegex(RuntimeError, r'range of \[inf, inf\] is not finite'):
            torch.histc(torch.tensor([float("inf")], dtype=torch.float, device=device))
        with self.assertRaisesRegex(RuntimeError, r'range of \[1, inf\] is not finite'):
            torch.histc(torch.tensor([1., 2., float("inf")], dtype=torch.float, device=device))
        # tensors with inf; min, max provided
        self.assertEqual(
            torch.histc(torch.tensor([float("inf")], dtype=torch.float, device=device),
                        bins=1, min=0, max=3),
            torch.tensor([0], dtype=torch.float, device=device))
        self.assertEqual(
            torch.histc(torch.tensor([1., 2., float("inf")], dtype=torch.float, device=device),
                        bins=4, max=3),
            torch.tensor([0, 1, 1, 0], dtype=torch.float, device=device))
        # tensor with nan -- should throw a RuntimeError
        with self.assertRaisesRegex(RuntimeError, r'range of \[nan, nan\] is not finite'):
            torch.histc(torch.tensor([float("nan")], dtype=torch.float, device=device))
        # tensors with min > max -- should throw a RuntimeError
        with self.assertRaisesRegex(RuntimeError, "max must be larger than min"):
            torch.histc(torch.tensor([1., 2., 3.], dtype=torch.float, device=device),
                        bins=4, min=5, max=1)

        # test against numpy.histogram()
        def test_against_np(tensor, bins=100, min=0, max=0):
            if min == 0 and max == 0:
                min = tensor.min().item()
                max = tensor.max().item()
            nparr = tensor.cpu().numpy()
            actual = torch.histc(tensor, bins=bins, min=min, max=max)
            expected = torch.from_numpy(np.histogram(nparr, bins=bins, range=(min, max))[0])
            actual_cpu = actual.cpu()
            # NB: Numpy returns a int64 tensor, like normal people...
            self.assertEqual(actual, expected.to(actual_cpu))

        test_against_np(torch.tensor([1., 2, 1], device=device))
        test_against_np(torch.randn(5000, device=device))

        # Test bins arg
        test_against_np(torch.randn(301, device=device), bins=10)

        # Test truncated range
        test_against_np(torch.randn(201, device=device), min=0.1, max=1)

        noncontig = torch.randn(100, 3, device=device)[:, 2]
        test_against_np(noncontig)

        multidim = torch.randn(3, 5, 7, 2, device=device)
        test_against_np(multidim)

        expanded = torch.randn(1, 5, 1, 2, device=device).expand(3, 5, 7, 2)
        test_against_np(expanded)

    # Tests to ensure that reduction functions employing comparison operators are usable when there
    # exists a zero dimension (i.e. when the the tensors are empty) in the tensor. These tests specifically
    # cater to functions where specifying the `dim` parameter is necessary.
    def test_tensor_compare_ops_empty(self, device):
        shape = (2, 0, 4)
        master_input = torch.randn(shape, device=device)
        np_input = np.empty(shape)
        test_functions = [
            ('amax', torch.amax, np.amax),
            ('amin', torch.amin, np.amin),
            ('max', lambda *args, **kwargs: torch.max(*args, **kwargs).values, np.max),
            ('min', lambda *args, **kwargs: torch.min(*args, **kwargs).values, np.min),
            ('median', lambda *args, **kwargs: torch.median(*args, **kwargs).values, np.median),
        ]

        for name, fn, np_function in test_functions:
            # Check if reduction happens along the specified dim with and without keepdim. Check with
            # numpy to maintain compatibility with numpy functions.
            error_msg = f"test function: {name}"
            self.assertEqual(torch.empty((2, 0), device=device), fn(master_input, dim=2), msg=error_msg)
            self.assertEqual(np_function(np_input, axis=2),
                             fn(master_input, dim=2).cpu().numpy(), msg=error_msg)

            self.assertEqual(torch.empty((2, 0), device=device), fn(master_input, dim=-1), msg=error_msg)
            self.assertEqual(np_function(np_input, axis=-1),
                             fn(master_input, dim=-1).cpu().numpy(), msg=error_msg)

            self.assertEqual(torch.empty((2, 0, 1), device=device), fn(master_input, dim=2, keepdim=True),
                             msg=error_msg)
            self.assertEqual(np_function(np_input, axis=2, keepdims=True),
                             fn(master_input, dim=2, keepdim=True).cpu().numpy(), msg=error_msg)

            self.assertEqual(torch.empty((2, 0, 1), device=device), fn(master_input, dim=-1, keepdim=True),
                             msg=error_msg)
            self.assertEqual(np_function(np_input, axis=-1, keepdims=True),
                             fn(master_input, dim=-1, keepdim=True).cpu().numpy(), msg=error_msg)

            # Check if function raises error on specified zero'd dimension as reduction dim.
            self.assertRaisesRegex(IndexError, "Expected reduction dim", lambda: fn(master_input, dim=1))

    # Tests to ensure that reduction of zero-dim tensors (i.e. empty tensors) using comparison operators
    # raises an error if no `dim` parameter is specified. This exists separately from tests in
    # test_tensot_compare_ops_empty because not specifying a `dim` parameter in the former tests does
    # not throw errors. Also, checking the return type of argmax requires supplying a different dtype
    # argument than that for the input tensor. There is also variantion in numpy testing.
    def test_tensor_compare_ops_argmax_argmix_kthvalue_dim_empty(self, device):
        shape = (2, 0, 4)
        master_input = torch.randn(shape, device=device)
        np_input = np.empty(shape)
        test_functions = [
            ('argmax', torch.argmax, {'dtype': torch.int64}, np.argmax),
            ('argmin', torch.argmin, {'dtype': torch.int64}, np.argmin),
            ('kthvalue', lambda *args, **kwargs: torch.kthvalue(*args, k=1, **kwargs).values,
             {}, lambda *args, **kwargs: np.partition(*args, 1, **kwargs))
        ]

        for name, fn, dtype, np_function in test_functions:
            error_msg = f"test function: {name}"
            self.assertEqual(torch.empty((2, 0), device=device, **dtype), fn(master_input, dim=2), msg=error_msg)
            self.assertEqual(np_function(np_input, axis=2),
                             fn(master_input, dim=2).cpu().numpy(), msg=error_msg)

            self.assertEqual(torch.empty((2, 0), device=device, **dtype), fn(master_input, dim=-1), msg=error_msg)
            self.assertEqual(np_function(np_input, axis=-1),
                             fn(master_input, dim=-1).cpu().numpy(), msg=error_msg)

            # keepdim variant does not exist for numpy
            self.assertEqual(torch.empty((2, 0, 1), device=device, **dtype), fn(master_input, dim=2, keepdim=True),
                             msg=error_msg)
            self.assertEqual(torch.empty((2, 0, 1), device=device, **dtype), fn(master_input, dim=-1, keepdim=True),
                             msg=error_msg)

            # Check if function raises error on specified zero'd dimension as reduction dim.
            self.assertRaisesRegex(IndexError, "Expected reduction dim", lambda: fn(master_input, dim=1))
            if name != 'kthvalue':
                self.assertRaisesRegex(IndexError, "Expected reduction dim", lambda: fn(master_input))

    # Tests to ensure that reduction of zero-dim tensors (i.e. empty tensors) using math operators works when a
    # non-zero dim is specified for the reduction and throws an error when the dim specified is 0. Although
    # there is some repetition with test_tensor_compare_ops_optional_dim_empty and test_tensor_compare_ops_empty,
    # these tests are kept separate since tests for math operators also require checking for correctness of the
    # returned data using allclose() or isinf() which does not exists in the former tests.
    def test_tensor_reduce_ops_empty(self, device):
        shape = (2, 0, 4)
        master_input = torch.randn(shape, device=device)
        np_input = np.empty(shape)
        test_functions = [
            ('prod', torch.prod, 1., np.prod),
            ('sum', torch.sum, 0., np.sum),
            ('norm', torch.norm, 0., np.linalg.norm),
            ('mean', torch.mean, nan, np.mean),
            ('var', torch.var, nan, np.var),
            ('std', torch.std, nan, np.std),
            ('logsumexp', torch.logsumexp, -inf, scipy.special.logsumexp),
        ]

        for name, fn, return_value, np_function in test_functions:
            # Check if reduction happens along the specified dimension.
            error_msg = f"test function: {name}"
            self.assertEqual(torch.empty((2, 0), device=device), fn(master_input, dim=2), msg=error_msg)
            self.assertEqual(np_function(np_input, axis=2), fn(master_input, dim=2).cpu().numpy(), msg=error_msg)

            self.assertEqual(torch.empty((2, 0), device=device), fn(master_input, dim=-1), msg=error_msg)
            self.assertEqual(np_function(np_input, axis=-1), fn(master_input, dim=-1).cpu().numpy(), msg=error_msg)

            self.assertEqual(torch.empty((2, 0, 1), device=device), fn(master_input, dim=2, keepdim=True), msg=error_msg)
            self.assertEqual(np_function(np_input, axis=2, keepdims=True), fn(master_input, dim=2, keepdim=True),
                             msg=error_msg)

            self.assertEqual(torch.empty((2, 0, 1), device=device), fn(master_input, dim=-1, keepdim=True), msg=error_msg)
            self.assertEqual(np_function(np_input, axis=-1, keepdims=True), fn(master_input, dim=-1, keepdim=True),
                             msg=error_msg)

            # Check if returned data is correct.
            check_func = (torch.testing.assert_allclose if math.isnan(return_value) or math.isinf(return_value) else
                          self.assertEqual)

            check_func(torch.full((2, 4), return_value, device=device), fn(master_input, dim=1), msg=error_msg)
            check_func(torch.full((2, 4), return_value, device=device), fn(master_input, dim=-2), msg=error_msg)
            check_func(torch.full((2, 1, 4), return_value, device=device), fn(master_input, dim=1, keepdim=True), msg=error_msg)
            check_func(torch.full((2, 1, 4), return_value, device=device), fn(master_input, dim=-2, keepdim=True), msg=error_msg)

            if name != 'logsumexp':
                # The scipy function does not work for reduction the zero dimension
                check_func(np.float32(np_function(np_input, axis=1)), fn(master_input, dim=1).cpu().numpy(), msg=error_msg)
                check_func(np.float32(np_function(np_input, axis=-2)), fn(master_input, dim=-2).cpu().numpy(), msg=error_msg)
                check_func(np.float32(np_function(np_input, axis=1, keepdims=True)),
                           fn(master_input, dim=1, keepdim=True).cpu().numpy(),
                           msg=error_msg)
                check_func(np.float32(np_function(np_input, axis=-2, keepdims=True)),
                           fn(master_input, dim=-2, keepdim=True).cpu().numpy(),
                           msg=error_msg)

                # logsumexp throws a type error when not specifying dim so test separately.
                check_func(torch.full((), return_value, device=device), fn(master_input), msg=error_msg)
            else:
                self.assertRaises(TypeError, lambda: fn(master_input))

    # Tests to ensure that any() and all() functions work with zero-dim tensors. Kept separate from
    # other tests for checking reduction with zero-dim tensors because these tests have significantly
    # different testing behaviour than that used for the former tests.
    def test_reduction_empty_any_all(self, device):
        shape = (2, 0, 4)
        x = torch.randn(shape, device=device)

        for dtype in torch.testing.get_all_dtypes(include_half=True, include_bfloat16=False,
                                                  include_bool=True, include_complex=True):
            # Refer: [all, any uint8 compatibility]
            if dtype == torch.uint8:
                out_dtype = torch.uint8
            else:
                out_dtype = torch.bool  # output of all/any is bool irrespective of input dtype

            xb = x.to(dtype)
            yb = x.to(dtype)
            # any
            self.assertEqual((2, 0), xb.any(2).shape)
            self.assertEqual((2, 0, 1), xb.any(2, keepdim=True).shape)
            self.assertEqual(torch.zeros((2, 4), device=device, dtype=out_dtype), xb.any(1))
            self.assertEqual(torch.zeros((2, 1, 4), device=device, dtype=out_dtype), xb.any(1, keepdim=True))
            self.assertEqual(torch.zeros((), device=device, dtype=out_dtype), xb.any())

            # all
            self.assertEqual((2, 0), xb.all(2).shape)
            self.assertEqual((2, 0, 1), xb.all(2, keepdim=True).shape)
            self.assertEqual(torch.ones((2, 4), device=device, dtype=out_dtype), xb.all(1))
            self.assertEqual(torch.ones((2, 1, 4), device=device, dtype=out_dtype), xb.all(1, keepdim=True))
            self.assertEqual(torch.ones((), device=device, dtype=out_dtype), xb.all())

instantiate_device_type_tests(TestReductions, globals())

if __name__ == '__main__':
    run_tests()
