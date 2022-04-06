# -*- coding: utf-8 -*-
# Owner(s): ["module: scatter & gather ops"]

import random

import torch

from torch.testing import make_tensor
from torch.testing._internal.common_utils import \
    (run_tests, TestCase,)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, dtypesIfCUDA,
     toleranceOverride, tol,)
from torch.testing._internal.common_dtype import \
    (get_all_dtypes, get_all_fp_dtypes,)

# Protects against includes accidentally setting the default dtype
assert torch.get_default_dtype() is torch.float32


# Note: test_scatter_gather_ops.py
# This test file tests scatter and gather operations,
#   like torch.scatter and torch.gather.

class TestScatterGather(TestCase):
    # Fills an index tensor with valid indices
    def _fill_indices(self, idx, dim, dim_size, elems_per_row, m, n, o, unique_indices=True):
        for i in range(1 if dim == 0 else m):
            for j in range(1 if dim == 1 else n):
                for k in range(1 if dim == 2 else o):
                    ii = [i, j, k]
                    ii[dim] = slice(0, idx.size(dim) + 1)
                    if unique_indices:
                        idx[tuple(ii)] = torch.randperm(dim_size)[0:elems_per_row]
                    else:
                        idx[tuple(ii)] = torch.randint(dim_size, (elems_per_row,))

    @dtypes(torch.float32, torch.complex64)
    def test_gather(self, device, dtype):
        m, n, o = random.randint(10, 20), random.randint(10, 20), random.randint(10, 20)
        elems_per_row = random.randint(1, 10)
        dim = random.randrange(3)

        src = make_tensor((m, n, o), device=device, dtype=dtype)
        idx_size = [m, n, o]
        idx_size[dim] = elems_per_row
        idx = make_tensor(idx_size, device=device, dtype=torch.long)
        self._fill_indices(idx, dim, src.size(dim), elems_per_row, m, n, o)

        actual = torch.gather(src, dim, idx)
        expected = torch.zeros(idx_size, device=device, dtype=dtype)
        for i in range(idx_size[0]):
            for j in range(idx_size[1]):
                for k in range(idx_size[2]):
                    ii = [i, j, k]
                    ii[dim] = idx[i, j, k]
                    expected[i, j, k] = src[tuple(ii)]
        self.assertEqual(actual, expected, atol=0, rtol=0)

        # Guarded because torch.max isn't defined for complex types
        if not dtype.is_complex:
            src = make_tensor((3, 4, 5), device=device, dtype=dtype)
            expected, idx = src.max(2, True)
            actual = torch.gather(src, 2, idx)
            self.assertEqual(actual, expected, atol=0, rtol=0)

    @dtypes(torch.bool)
    def test_gather_bool(self, device, dtype):
        src = torch.tensor(((False, True), (True, True)), device=device, dtype=dtype)
        idx = torch.tensor(((0, 0), (1, 0)), device=device, dtype=torch.long)
        actual = torch.gather(src, 1, idx)
        expected = torch.tensor(((False, False), (True, True)), device=device, dtype=dtype)
        self.assertEqual(actual, expected, atol=0, rtol=0)

    def _test_scatter_base(self, fn, *, device, dtype, is_scalar, reduction, unique_indices=True):
        m, n, o = random.randint(10, 20), random.randint(10, 20), random.randint(10, 20)
        elems_per_row = random.randint(1, 10)
        dim = random.randrange(3)

        idx_size = [m, n, o]
        idx_size[dim] = elems_per_row
        idx = torch.empty(tuple(idx_size), device=device, dtype=torch.long)
        self._fill_indices(idx, dim, ([m, n, o])[dim], elems_per_row, m, n, o, unique_indices)

        if is_scalar:
            src = random.random()
        else:
            src_size = [random.randint(1, 5) + s for s in idx_size]
            src = make_tensor(tuple(src_size), device=device, dtype=dtype)

        base = make_tensor((m, n, o), device=device, dtype=dtype)
        if reduction is not None:
            actual = fn(base.clone(), dim, idx, src, reduce=reduction)
        else:
            actual = fn(base.clone(), dim, idx, src)

        expected = base.clone()
        counts = torch.ones(base.shape, dtype=torch.long, device=device)
        for i in range(idx_size[0]):
            for j in range(idx_size[1]):
                for k in range(idx_size[2]):
                    ii = [i, j, k]
                    ii[dim] = idx[i, j, k]
                    if fn is torch.Tensor.scatter_add_:
                        expected[tuple(ii)] += src[i, j, k]
                    else:
                        # method may be 'scatter_', 'scatter', 'scatter_reduce'
                        # or 'scatter_reduce_', the former two might have a reduction argument
                        # while the latter two always do
                        value = src if is_scalar else src[i, j, k]

                        if reduction == "add" or reduction == "sum":
                            expected[tuple(ii)] += value
                        elif reduction == "multiply" or reduction == "prod":
                            expected[tuple(ii)] *= value
                        elif reduction == "amax":
                            expected[tuple(ii)] = max(expected[tuple(ii)], value)
                        elif reduction == "amin":
                            expected[tuple(ii)] = min(expected[tuple(ii)], value)
                        elif reduction == "mean":
                            expected[tuple(ii)] += value
                            counts[tuple(ii)] += 1
                        else:
                            expected[tuple(ii)] = value

        if (reduction == "mean"):
            if (dtype.is_floating_point or dtype.is_complex):
                expected /= counts
            else:
                expected.div_(counts, rounding_mode="floor")

        self.assertEqual(actual, expected, atol=0, rtol=0)

        # Tests empty index
        dst = make_tensor((2, 2), device=device, dtype=dtype)
        idx = torch.tensor((), device=device, dtype=torch.long)
        src = make_tensor((2, 2), device=device, dtype=dtype)
        if reduction is not None:
            actual = fn(dst, 0, idx, src, reduce=reduction)
        else:
            actual = fn(dst, 0, idx, src)
        self.assertEqual(actual, dst, atol=0, rtol=0)

    @dtypes(torch.float16, torch.float32, torch.complex64)
    def test_scatter_(self, device, dtype):
        self._test_scatter_base(torch.Tensor.scatter_, device=device, dtype=dtype,
                                is_scalar=False, reduction=None)

    @dtypes(torch.float16, torch.float32, torch.complex64)
    def test_scatter__scalar(self, device, dtype):
        self._test_scatter_base(torch.Tensor.scatter_, device=device, dtype=dtype,
                                is_scalar=True, reduction=None)

    # FIXME: RuntimeError: "cuda_scatter_gather_base_kernel_reduce_multiply" not implemented for 'ComplexFloat'
    @toleranceOverride({torch.float16: tol(atol=1e-2, rtol=0)})
    @dtypesIfCUDA(torch.float16, torch.float32)
    @dtypes(torch.float16, torch.float32, torch.complex64)
    def test_scatter__reductions(self, device, dtype):
        for reduction in ("add", "multiply"):
            self._test_scatter_base(torch.Tensor.scatter_, device=device, dtype=dtype,
                                    is_scalar=False, reduction=reduction)
            self._test_scatter_base(torch.Tensor.scatter_, device=device, dtype=dtype,
                                    is_scalar=True, reduction=reduction)

    @dtypes(torch.float16, torch.float32, torch.complex64)
    def test_scatter_add_(self, device, dtype):
        self._test_scatter_base(torch.Tensor.scatter_add_, device=device, dtype=dtype,
                                is_scalar=False, reduction=None)

    @dtypes(torch.float32)
    def test_scatter_add_mult_index_base(self, device, dtype):
        m, n = 30, 40
        idx = torch.zeros(m, n, device=device, dtype=torch.long)
        src = torch.ones(m, n, device=device, dtype=dtype)
        res0 = torch.zeros(m, n, device=device, dtype=dtype).scatter_add_(0, idx, src)
        res1 = torch.zeros(m, n, device=device, dtype=dtype).scatter_add_(1, idx, src)

        self.assertEqual(res0[0, :], m * torch.ones(n, device=device, dtype=dtype), atol=0, rtol=0)
        self.assertEqual(res1[:, 0], n * torch.ones(m, device=device, dtype=dtype), atol=0, rtol=0)

    # FIXME: discrepancy between bool ReduceAdd on CUDA and CPU (a + b on CPU and buggy a && b on CUDA)
    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_bool=False))
    def test_scatter_reduce_sum(self, device, dtype):
        self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                is_scalar=False, reduction='sum', unique_indices=False)

    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True))
    @dtypesIfCUDA(*get_all_fp_dtypes(include_half=True, include_bfloat16=True))
    def test_scatter_reduce_prod(self, device, dtype):
        self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                is_scalar=False, reduction='prod', unique_indices=False)

    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_bool=False))
    @dtypesIfCUDA(*get_all_fp_dtypes(include_half=True, include_bfloat16=True))
    def test_scatter_reduce_mean(self, device, dtype):
        self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                is_scalar=False, reduction='mean', unique_indices=False)

    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False))
    @dtypesIfCUDA(*get_all_fp_dtypes(include_half=True, include_bfloat16=True))
    def test_scatter_reduce_amax(self, device, dtype):
        self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                is_scalar=False, reduction='amax', unique_indices=False)

    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False))
    @dtypesIfCUDA(*get_all_fp_dtypes(include_half=True, include_bfloat16=True))
    def test_scatter_reduce_amin(self, device, dtype):
        self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                is_scalar=False, reduction='amin', unique_indices=False)


# Generic Device Test Framework instantation, see
#   https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests
#   for details.
instantiate_device_type_tests(TestScatterGather, globals())

if __name__ == '__main__':
    run_tests()
