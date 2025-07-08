# Owner(s): ["module: scatter & gather ops"]

import random

import torch

from torch.testing import make_tensor
from torch.testing._internal.common_utils import \
    (parametrize, run_tests, TestCase, DeterministicGuard, TEST_WITH_ROCM)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, onlyCPU, dtypes, dtypesIfCUDA,
     toleranceOverride, tol,)
from torch.testing._internal.common_dtype import \
    (get_all_dtypes,)

from torch.testing._internal.common_cuda import CDNA3OrLater

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

    @dtypes(torch.int8, torch.bfloat16)
    def test_gather_large(self, device, dtype):
        # test larger shapes to check vectorized implementation
        for (m, n, k) in ((4096, 3072, 4096), (4096, 3072, 4100)):
            src = make_tensor((m, k), device=device, dtype=dtype)
            alloc0 = torch.empty(src.nelement() * 2, device=device, dtype=dtype)
            discontig = alloc0.view(m, 2 * k)[:, ::2].copy_(src)
            alloc1 = torch.empty(src.nelement() + 1, device=device, dtype=dtype)
            misaligned = alloc1[1:].view(m, k).copy_(src)
            alloc2 = torch.empty(m, k + 4, device=device, dtype=dtype)
            misaligned1 = alloc2[:, :-4].copy_(src)
            num_ind = n
            for dim in (0, 1):
                max_ind = src.shape[dim]
                ind0 = torch.randint(max_ind, (num_ind,), device=device)
                ind_discontig0 = torch.empty(num_ind * 2, device=device, dtype=torch.int64)[::2].copy_(ind0)
                shape_ind = [1] * src.ndim
                shape_ind[dim] = ind0.shape[0]
                shape_out = list(src.shape)
                shape_out[dim] = ind0.shape[0]
                ind = ind0.view(shape_ind).expand(shape_out)
                ind_discontig = ind_discontig0.view(shape_ind).expand(shape_out)
                res = torch.gather(src, dim=dim, index=ind)
                ref = src[ind0] if dim == 0 else src[:, ind0]
                self.assertEqual(res, ref, atol=0, rtol=0)
                if res.device.type == "cuda":
                    ref_cpu = src.cpu()[ind0.cpu()] if dim == 0 else src.cpu()[:, ind0.cpu()]
                    self.assertEqual(res.cpu(), ref_cpu, atol=0, rtol=0)
                res = torch.gather(src, dim=dim, index=ind_discontig)
                self.assertEqual(res, ref, atol=0, rtol=0)
                res_ind = src[ind_discontig0] if dim == 0 else src[:, ind_discontig0]
                self.assertEqual(res_ind, ref, atol=0, rtol=0)
                res_ind_neg = src[ind0 - src.shape[dim]] if dim == 0 else src[:, ind0 - src.shape[1]]
                self.assertEqual(res_ind_neg, ref, atol=0, rtol=0)
                res = torch.gather(discontig, dim=dim, index=ind)
                self.assertEqual(res, ref, atol=0, rtol=0)
                res_ind = discontig[ind0] if dim == 0 else discontig[:, ind0]
                self.assertEqual(res_ind, ref, atol=0, rtol=0)
                res = torch.gather(misaligned, dim=dim, index=ind)
                self.assertEqual(res, ref, atol=0, rtol=0)
                res_ind = misaligned[ind0] if dim == 0 else misaligned[:, ind0]
                self.assertEqual(res_ind, ref, atol=0, rtol=0)
                res_ind = misaligned1[ind0] if dim == 0 else misaligned[:, ind0]
                self.assertEqual(res_ind, ref, atol=0, rtol=0)
                res_gather = torch.gather(misaligned1, dim=dim, index=ind)
                self.assertEqual(res_gather, ref, atol=0, rtol=0)
        # test gather along 1st dim that can accidentally trigger fast path
        # because due to index dimension in the gather dim being 1
        # an unexpected squashing in tensorIterator happens
        src = make_tensor((16, 2, 16), device=device, dtype=dtype)
        ind = torch.randint(2, (16, 1), device=device).view(16, 1, 1).expand(16, 1, 16)
        res = torch.gather(src, dim=1, index=ind)
        if res.device.type == "cuda":
            ref_cpu = torch.gather(src.cpu(), dim=1, index=ind.cpu())
            self.assertEqual(res.cpu(), ref_cpu, atol=0, rtol=0)


    @dtypes(torch.bool)
    def test_gather_bool(self, device, dtype):
        src = torch.tensor(((False, True), (True, True)), device=device, dtype=dtype)
        idx = torch.tensor(((0, 0), (1, 0)), device=device, dtype=torch.long)
        actual = torch.gather(src, 1, idx)
        expected = torch.tensor(((False, False), (True, True)), device=device, dtype=dtype)
        self.assertEqual(actual, expected, atol=0, rtol=0)

    @parametrize("sparse_grad", [False, True])
    @dtypes(torch.float32, torch.float64)
    def test_gather_backward_with_empty_index_tensor(self, device, dtype, sparse_grad):
        dim = -1
        input = torch.rand([10, 5], dtype=dtype, device=device, requires_grad=True)
        index = torch.randint(0, 2, [3, 0], dtype=torch.int64, device=device)
        res = torch.gather(input, dim, index, sparse_grad=sparse_grad)
        res.sum().backward()
        grad = input.grad.to_dense() if sparse_grad else input.grad
        expected_grad = torch.zeros_like(input, requires_grad=False)
        self.assertEqual(grad, expected_grad, atol=0, rtol=0)

    def _test_scatter_base(self, fn, *, device, dtype, is_scalar, reduction,
                           unique_indices=True, include_self=True):
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
            if fn is torch.Tensor.scatter_reduce_:
                actual = fn(base.clone(), dim, idx, src, reduce=reduction, include_self=include_self)
            else:
                actual = fn(base.clone(), dim, idx, src, reduce=reduction)
        else:
            actual = fn(base.clone(), dim, idx, src)

        expected = base.clone()
        counts = torch.zeros(base.shape, dtype=torch.long, device=device) + include_self
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

                        if ((not include_self) and counts[tuple(ii)] == 0):
                            expected[tuple(ii)] = value
                        else:
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
                            else:
                                expected[tuple(ii)] = value

                        counts[tuple(ii)] += 1

        if (reduction == "mean"):
            counts.masked_fill_(counts == 0, 1)
            if (dtype.is_floating_point or dtype.is_complex):
                expected /= counts
            else:
                expected.div_(counts, rounding_mode="floor")

        if dtype == torch.float16 or dtype == torch.bfloat16:
            # Some CUDA kernels (e.g. indexing_backward_kernel_stride_1) that are called during
            # the test use fp32 for internal accumulation for improved accuracy. When using 16 bit
            # precision types can be small differences
            self.assertEqual(actual, expected, atol=0.04, rtol=0.05)
        else:
            # When we are running opportunistic_fastatomics, we will expect some floating point rounding
            # errors as the order of operation is not guaranteed.
            if TEST_WITH_ROCM and CDNA3OrLater() \
                    and not torch.are_deterministic_algorithms_enabled():
                self.assertEqual(actual, expected, atol=1e-9, rtol=1e-6)
            else:
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
        for deterministic in [False, True]:
            with DeterministicGuard(deterministic):
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
        for deterministic in [False, True]:
            with DeterministicGuard(deterministic):
                self._test_scatter_base(torch.Tensor.scatter_add_, device=device, dtype=dtype,
                                        is_scalar=False, reduction=None)

    @dtypes(torch.float32)
    def test_scatter_add_mult_index_base(self, device, dtype):
        for deterministic in [False, True]:
            with DeterministicGuard(deterministic):
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
        for include_self in (True, False):
            for deterministic in [False, True]:
                with DeterministicGuard(deterministic):
                    self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                            is_scalar=False, reduction='sum', unique_indices=False,
                                            include_self=include_self)

    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True))
    @dtypesIfCUDA(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False, include_bool=False))
    def test_scatter_reduce_prod(self, device, dtype):
        for include_self in (True, False):
            self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                    is_scalar=False, reduction='prod', unique_indices=False,
                                    include_self=include_self)

    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_bool=False))
    @dtypesIfCUDA(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False, include_bool=False))
    def test_scatter_reduce_mean(self, device, dtype):
        for include_self in (True, False):
            for deterministic in [False, True]:
                with DeterministicGuard(deterministic):
                    self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                            is_scalar=False, reduction='mean', unique_indices=False,
                                            include_self=include_self)

    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False))
    @dtypesIfCUDA(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False, include_bool=False))
    def test_scatter_reduce_amax(self, device, dtype):
        for include_self in (True, False):
            self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                    is_scalar=False, reduction='amax', unique_indices=False,
                                    include_self=include_self)
            # simple test for nan/inf propagation
            if (dtype.is_floating_point):
                input = torch.zeros(3, device=device, dtype=dtype)
                src = torch.tensor([1, float('nan'), -float('inf'), -float('inf'), 2, float('inf')], device=device, dtype=dtype)
                idx = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
                input.scatter_reduce_(0, idx, src, 'amax', include_self=include_self)
                expected_result = torch.tensor([float('nan'), -float('inf'), float('inf')], device=device, dtype=dtype)
                if (include_self):
                    expected_result[1] = 0
                self.assertEqual(input, expected_result)


    @dtypes(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False))
    @dtypesIfCUDA(*get_all_dtypes(include_half=True, include_bfloat16=True, include_complex=False, include_bool=False))
    def test_scatter_reduce_amin(self, device, dtype):
        for include_self in (True, False):
            self._test_scatter_base(torch.Tensor.scatter_reduce_, device=device, dtype=dtype,
                                    is_scalar=False, reduction='amin', unique_indices=False,
                                    include_self=include_self)
            # simple test for nan/inf propagation
            if (dtype.is_floating_point):
                input = torch.zeros(3, device=device, dtype=dtype)
                src = torch.tensor([1, float('nan'), -2, -float('inf'), float('inf'), float('inf')], device=device, dtype=dtype)
                idx = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
                input.scatter_reduce_(0, idx, src, 'amin', include_self=include_self)
                expected_result = torch.tensor([float('nan'), -float('inf'), float('inf')], device=device, dtype=dtype)
                if (include_self):
                    expected_result[2] = 0
                self.assertEqual(input, expected_result)

    @onlyCPU
    @dtypes(torch.float32, torch.float64, torch.bfloat16, torch.float16)
    def test_scatter_expanded_index(self, device, dtype):
        def helper(input_size, idx_size):
            input = torch.randn(input_size, device=device).to(dtype=dtype)
            input2 = input.clone()

            shape = [1] * len(input_size)
            shape[0] = idx_size
            dim_size = input_size[0]
            idx = torch.randint(0, dim_size, shape)

            # The fast path on scatter when index is expanded
            # will depend on sorted index where the collected src indice
            # for each row in input will be mapped to rowptrs in a CSR format.
            # Create some empty rows by masking:
            mask = (idx > 1) * (idx < 4)
            idx[mask] = 0

            expanded_shape = input_size
            expanded_shape[0] = idx_size
            idx = idx.expand(expanded_shape)
            idx2 = idx.contiguous()
            src = torch.randn(expanded_shape, device=device).to(dtype=dtype)

            out = input.scatter_add(0, idx, src)
            out2 = input2.scatter_add(0, idx2, src)
            self.assertEqual(out, out2)

            for reduce in ["sum", "prod", "mean", "amax", "amin"]:
                for include_self in [True, False]:
                    out = input.scatter_reduce(0, idx, src, reduce=reduce, include_self=include_self)
                    out2 = input2.scatter_reduce(0, idx2, src, reduce=reduce, include_self=include_self)
                    self.assertEqual(out, out2)

        helper([50, 17], 100)
        helper([50, 1], 100)
        helper([50, 8, 7], 100)
        helper([50, 3, 4, 5], 100)

    @dtypes(torch.float32)
    def test_scatter_add_broadcasted_index_deterministic(self, device, dtype):
        for d in (0, 1):
            inp = torch.randn(3, 4, device=device, dtype=dtype)
            idx_1d = torch.randint(3, (10,), device=device)
            src_shape = list(inp.shape)
            src_shape[d] = 10
            src = torch.randn(src_shape, device=device, dtype=dtype)
            idx = idx_1d.unsqueeze(1 - d).expand(src_shape)
            print(idx.stride())
            ref = inp.clone().scatter_add_(d, idx, src)
            with DeterministicGuard(True):
                res = inp.clone().scatter_add_(d, idx, src)
            self.assertEqual(res, ref)


    @onlyCPU
    @dtypes(torch.float32, torch.float64, torch.bfloat16)
    def test_gather_expanded_index(self, device, dtype):
        # Test when index is [N, 1], which would have stride [1, 0]
        # should be excluded from the fast path when index ix expanded
        input = torch.arange(25).view(5, 5)
        input2 = input.to(dtype=dtype)

        idx = torch.arange(5).view(5, 1)
        out = torch.gather(input, 0, idx)
        out2 = torch.gather(input2, 0, idx)

        self.assertEqual(out.to(dtype=dtype), out2)

        def helper(input_size, idx_size):
            input = torch.randn(input_size, device=device).to(dtype=dtype)
            input2 = input.clone()

            shape = [1] * len(input_size)
            shape[0] = idx_size
            dim_size = input_size[0]
            idx = torch.randint(0, dim_size, shape)

            # Test the fast path on gather when index is expanded
            expanded_shape = input_size
            expanded_shape[0] = idx_size
            idx = idx.expand(expanded_shape)
            idx2 = idx.contiguous()

            out = torch.gather(input, 0, idx)
            out2 = torch.gather(input2, 0, idx2)

            self.assertEqual(out, out2)

            # test unsqueezed index
            # expanded_index kernel can not handle the case:
            # the size > 1 and stride == 1 at a dimension.
            # for example: the index with size of [1, 8, 7],  stride of [1, 1, 0].
            # see https://github.com/pytorch/pytorch/issues/129093
            def unsqueeze_helper(idx, dim):
                if dim == 2:
                    return idx.unsqueeze(1).t()
                else:
                    return unsqueeze_helper(idx, dim - 1).unsqueeze(dim - 1)

            idx = torch.randint(0, dim_size, (input.shape[1],))
            idx = unsqueeze_helper(idx, len(input_size))
            expanded_shape[0] = 1
            idx = idx.expand(expanded_shape)
            idx2 = idx.contiguous()
            out = torch.gather(input, 0, idx)
            out2 = torch.gather(input2, 0, idx2)
            self.assertEqual(out, out2)

        helper([50, 17], 100)
        helper([50, 1], 100)
        helper([50, 8, 7], 100)
        helper([50, 3, 4, 5], 100)

# Generic Device Test Framework instantation, see
#   https://github.com/pytorch/pytorch/wiki/Running-and-writing-tests
#   for details.
instantiate_device_type_tests(TestScatterGather, globals())

if __name__ == '__main__':
    run_tests()
