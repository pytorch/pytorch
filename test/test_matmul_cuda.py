# Owner(s): ["module: linear algebra"]

import contextlib
import json
import math
import re
import tempfile
import unittest
from itertools import product
from functools import partial
from typing import Optional

import torch

from torch.quantization._quantized_conversions import (
    pack_int4_to_int8,
    quantized_weight_reorder_for_mixed_dtypes_linear_cutlass,
)

from torch.testing import make_tensor
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_BF16,
    SM53OrLater,
    SM89OrLater,
    SM90OrLater,
    xfailIfSM100OrLater,
    xfailIfSM120OrLater,
    _get_torch_cuda_version,
    PLATFORM_SUPPORTS_FP8,
    PLATFORM_SUPPORTS_MX_GEMM,
)
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
    tol as xtol,
    toleranceOverride,
    e4m3_type,
    e5m2_type,
    E4M3_MAX_POS,
    E5M2_MAX_POS,
)

from torch.testing._internal.common_utils import (
    IS_JETSON,
    IS_WINDOWS,
    parametrize,
    run_tests,
    skipIfRocm,
    skipIfRocmVersionLessThan,
    TEST_CUDA,
    TEST_WITH_ROCM,
    TestCase,
)
from torch.testing._internal.common_quantized import _f32_to_floatx_unpacked, _floatx_unpacked_to_f32, ceil_div, to_blocked

_IS_SM8X = False
if TEST_CUDA:
    _IS_SM8X = torch.cuda.get_device_capability(0)[0] == 8

# Protects against includes accidentally setting the default dtype
assert torch.get_default_dtype() is torch.float32


@contextlib.contextmanager
def blas_library_context(backend):
    prev_backend = torch.backends.cuda.preferred_blas_library()
    torch.backends.cuda.preferred_blas_library(backend)
    try:
        yield
    finally:
        torch.backends.cuda.preferred_blas_library(prev_backend)

class TestMatmulCuda(TestCase):
    def setUp(self):
        super().setUp()
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        super().tearDown()

    def cublas_addmm(self, size: int, dtype: torch.dtype, reduced_precision: bool = False, fp16_accumulate: bool = False):
        #
        # Check for catastrophic cuBLAS inaccuracy by measuring the deviation between
        # results from the CUDA invocation of torch.addmm and the CPU invocation
        # (which does not use CUDA backend).
        #
        # Get dims
        n, m, p = (size + 1, size, size + 2)
        # Disable reduced precision reductions in BFloat16 to bypass some kernels
        # which fail the threshold check
        orig_bf16 = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        orig_fp16 = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        orig_fp16_accumulate = torch.backends.cuda.matmul.allow_fp16_accumulation
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced_precision
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced_precision
        torch.backends.cuda.matmul.allow_fp16_accumulation = fp16_accumulate
        # Make random tensors on CPU (seed set on common_utils.py import)
        # (Not using numpy because it does not support bfloat16)
        make_arg = partial(make_tensor, dtype=dtype, device="cpu")
        m_beta = make_arg(1)
        m_input = make_arg((n, p))
        m_1 = make_arg((n, m))
        m_2 = make_arg((m, p))
        # scale to abate overflows in fp16 accum
        if fp16_accumulate:
            m_1 = m_1 / 100
            m_2 = m_2 / 100
        # *(B)FLOAT16 Special Handling*
        # Backend does not tensorize float16 on CPU,
        # and bloat16 may present accuracy issues,
        # so convert to float32 for these cases
        # (but keep same for other types, e.g. float32 and int*)
        if dtype == torch.float16 or dtype == torch.bfloat16:
            m_beta = m_beta.to(dtype=torch.float32)
            m_input = m_input.to(dtype=torch.float32)
            m_1 = m_1.to(dtype=torch.float32)
            m_2 = m_2.to(dtype=torch.float32)
        # Get CPU result
        res_cpu = torch.addmm(m_input, m_1, m_2, beta=m_beta.item())
        # *(B)FLOAT16 Special Handling*``
        # Convert back to (b)float16
        if dtype == torch.float16 or dtype == torch.bfloat16:
            m_beta = m_beta.to(dtype=dtype)
            m_input = m_input.to(dtype=dtype)
            m_1 = m_1.to(dtype=dtype)
            m_2 = m_2.to(dtype=dtype)
            res_cpu = res_cpu.to(dtype=dtype)
        # Move arg tensors to CUDA
        m_beta = m_beta.to("cuda")
        m_input = m_input.to("cuda")
        m_1 = m_1.to("cuda")
        m_2 = m_2.to("cuda")
        # Get CUDA result
        res_cuda = torch.addmm(m_input, m_1, m_2, beta=m_beta.item())
        # Move to CPU for comparison
        res_cuda = res_cuda.to("cpu")
        # Compare
        self.assertEqual(res_cpu, res_cuda)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = orig_bf16
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = orig_fp16
        torch.backends.cuda.matmul.allow_fp16_accumulation = orig_fp16_accumulate

    @onlyCUDA
    @skipIfRocmVersionLessThan((5, 2))
    # imported 'tol' as 'xtol' to avoid aliasing in code above
    @toleranceOverride({torch.float16: xtol(atol=1e-1, rtol=1e-1),
                        torch.bfloat16: xtol(atol=1e-1, rtol=1e-1),
                        torch.float32: xtol(atol=1e-1, rtol=1e-1)})
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("size", [100, 1000, 10000])
    @parametrize("backend", ["cublas", "cublaslt"])
    def test_cublas_addmm(self, size: int, dtype: torch.dtype, backend):
        with blas_library_context(backend):
            self.cublas_addmm(size, dtype, False)

    @onlyCUDA
    @skipIfRocmVersionLessThan((5, 2))
    # imported 'tol' as 'xtol' to avoid aliasing in code above
    @toleranceOverride({torch.float16: xtol(atol=7e-1, rtol=2e-1),
                        torch.bfloat16: xtol(atol=1e1, rtol=2e-1)})
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("size", [100, 1000, 10000])
    @parametrize("backend", ["cublas", "cublaslt"])
    def test_cublas_addmm_reduced_precision(self, size: int, dtype: torch.dtype, backend):
        with blas_library_context(backend):
            self.cublas_addmm(size, dtype, True)

    @onlyCUDA
    @skipIfRocmVersionLessThan((5, 2))
    @dtypes(torch.float16)
    # m == 4 chooses OUTPUT_TYPE reduction on H200
    # m == 8 chooses OUTPUT_TYPE reduction on A100
    @parametrize("small_size", [4, 8])
    @parametrize("size", [32768])
    @parametrize("backend", ["cublaslt", "cublas"])
    def test_cublas_addmm_no_reduced_precision(self, small_size: int, size: int, dtype: torch.dtype, backend):
        with blas_library_context(backend):
            torch.backends.cuda.preferred_blas_library(backend)
            orig_precision = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            m1 = torch.full((small_size, size), 65504.0, dtype=dtype, device='cuda')
            m2 = torch.ones((size, small_size), dtype=dtype, device='cuda')
            m2[size // 2:, :] = -1.0
            b = torch.zeros((small_size,), dtype=dtype, device='cuda')
            out = torch.addmm(b, m1, m2, beta=1.0)
            self.assertEqual(out.sum().item(), 0.0)
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = orig_precision

    @onlyCUDA
    @skipIfRocmVersionLessThan((5, 2))
    # imported 'tol' as 'xtol' to avoid aliasing in code above
    @toleranceOverride({torch.float16: xtol(atol=7e-1, rtol=2e-1),
                        torch.bfloat16: xtol(atol=1e1, rtol=2e-1)})
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("size", [100, 1000, 10000])
    @parametrize("backend", ["cublas", "cublaslt"])
    def test_cublas_addmm_reduced_precision_fp16_accumulate(self, size: int, dtype: torch.dtype, backend):
        with blas_library_context(backend):
            self.cublas_addmm(size, dtype, False, True)

    @onlyCUDA
    @skipIfRocm
    def test_cublas_and_lt_reduced_precision_fp16_accumulate(self):
        orig_fp16_accumulate = torch.backends.cuda.matmul.allow_fp16_accumulation
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        x = torch.rand(32, 512, 512, device='cuda', dtype=torch.half)
        w = torch.rand(512, 512, device='cuda', dtype=torch.half)
        b = torch.rand(512, device='cuda', dtype=torch.half)
        out = torch.nn.functional.linear(x, w, b)
        out_cpu = torch.nn.functional.linear(x.cpu(), w.cpu(), b.cpu())
        self.assertEqual(out, out_cpu, atol=5e-3, rtol=8e-3)

        a = torch.rand(16, 128, 128, device='cuda', dtype=torch.half)
        b = torch.rand(16, 128, 128, device='cuda', dtype=torch.half)
        c = torch.rand(16, 128, 128, device='cuda', dtype=torch.half)
        out = torch.baddbmm(a, b, c)
        out_cpu = torch.baddbmm(a.cpu(), b.cpu(), c.cpu())
        self.assertEqual(out, out_cpu, atol=1e-3, rtol=5e-3)
        torch.backends.cuda.matmul.allow_fp16_accumulation = orig_fp16_accumulate

    @onlyCUDA
    @toleranceOverride({torch.float16: xtol(atol=1e-3, rtol=2e-3)})
    @dtypes(torch.float16)
    def test_cublas_addmm_alignment(self, dtype):
        device = 'cuda'
        # perturb X, A, or B alignment
        for idx in range(0, 3):
            for offset in range(1, 3):
                offsets = [0, 0, 0]
                offsets[idx] = offset
                x_offset, a_offset, b_offset = offsets
                A = torch.rand((5120 * 2560 + a_offset), requires_grad=True, dtype=dtype, device=device)
                A = A[a_offset:].reshape(5120, 2560)
                X = torch.rand((26 * 2560 + x_offset), requires_grad=True, dtype=dtype, device=device)
                X = X[x_offset:].reshape(26, 1, 2560)
                B = torch.rand((5120 + b_offset), requires_grad=True, dtype=dtype, device=device)
                B = B[b_offset:].reshape(5120)
                out = torch.nn.functional.linear(X, A, B)
                self.assertEqual(out, torch.matmul(X, A.transpose(1, 0)) + B)

    @onlyCUDA
    @unittest.skipIf(IS_JETSON, "Too large for Jetson")
    @toleranceOverride({torch.float32: xtol(atol=1e-5, rtol=1.1e-5)})
    @dtypes(*([torch.float32, torch.float16] +
              [torch.bfloat16] if TEST_WITH_ROCM or SM53OrLater else []))
    @parametrize(
        "batch_size, N, M, P",
        [(2, 100, 100, 100),
         (2, 1000, 1000, 1000),
         (1, 10000, 1000, 10000),
         (1, 10000, 10000, 10000)],
        name_fn=lambda batch_size, N, M, P: f"{batch_size}_{N}_{M}_{P}",
    )
    @skipIfRocm
    def test_cublas_baddbmm_large_input(self, device, batch_size, N, M, P, dtype):
        cpu_dtype = dtype
        if dtype == torch.float16 or dtype == torch.bfloat16:
            cpu_dtype = torch.float32

        M1 = torch.rand((N, M), device=device, dtype=dtype)
        M2 = torch.rand((M, P), device=device, dtype=dtype)
        A = torch.rand((N, P), device=device, dtype=dtype)

        def _convert_to_cpu(t):
            return t.to(device='cpu', dtype=cpu_dtype)
        M1_cpu, M2_cpu, A_cpu = map(_convert_to_cpu, [M1, M2, A])

        # linear
        out1_cpu = torch.nn.functional.linear(M1_cpu, M2_cpu.t(), A_cpu).to(dtype=dtype)
        out1_gpu = torch.nn.functional.linear(M1, M2.t(), A).cpu()
        self.assertEqual(out1_cpu, out1_gpu)
        # test multiply the identity matrix
        if N == M and M == P:
            M2_eye = torch.eye(N, device=device, dtype=dtype)
            out1_eye_gpu = torch.nn.functional.linear(M1, M2_eye.t(), torch.zeros_like(A))
            self.assertEqual(M1_cpu.to(dtype=dtype), out1_eye_gpu.cpu())

        # baddbmm
        def _expand_to_batch(t: torch.Tensor):
            return t.expand((batch_size, ) + t.size())
        alpha, beta = 1.0, 1.0
        M1, M2, A, M1_cpu, M2_cpu, A_cpu = map(_expand_to_batch, [M1, M2, A, M1_cpu, M2_cpu, A_cpu])

        out2_cpu = torch.baddbmm(A_cpu, M1_cpu, M2_cpu, beta=beta, alpha=alpha).to(dtype=dtype)
        out2_gpu = torch.baddbmm(A, M1, M2, beta=beta, alpha=alpha).cpu()
        self.assertEqual(out2_cpu, out2_gpu)
        # test multiply the identity matrix
        if N == M and M == P:
            M2_eye = torch.eye(N, device=device, dtype=dtype).expand(batch_size, N, N)
            out2_eye_gpu = torch.baddbmm(torch.zeros_like(A), M1, M2_eye, beta=beta, alpha=alpha)
            self.assertEqual(M1_cpu.to(dtype=dtype), out2_eye_gpu.cpu())

        # cross comparison
        self.assertEqual(out1_gpu, out2_gpu[0])

    def grouped_mm_helper(self, alist, blist, gOlist, agradlist, bgradlist, outlist):
        for a, b, gO, agrad, bgrad, out in zip(alist, blist, gOlist, agradlist, bgradlist, outlist):
            a = a.clone().detach().requires_grad_()
            b = b.clone().detach().requires_grad_()
            out_ref = torch.mm(a, b.t())
            out_ref.backward(gO)
            self.assertEqual(out, out_ref)
            if agrad is not None:
                self.assertEqual(agrad, a.grad)
                self.assertEqual(bgrad, b.grad)

    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support CUTLASS")
    @xfailIfSM120OrLater
    @unittest.skipIf(not SM90OrLater, "Grouped gemm supported only on SM90 and SM100")
    @parametrize("strided", [False, True])
    @parametrize("a_row_major", [False, True])
    @parametrize("b_row_major", [False, True])
    def test_grouped_gemm_2d_2d(self, strided, a_row_major, b_row_major):
        device = "cuda"
        dtype = torch.bfloat16
        m, n, k, n_groups = 16, 32, 64, 4
        if a_row_major:
            a = torch.randn(m, k * n_groups + k * int(strided), device=device, dtype=dtype)[:, :k * n_groups]
        else:
            a = torch.randn(k * n_groups + k * int(strided), m, device=device, dtype=dtype).t()[:, :k * n_groups]

        if b_row_major:
            b = torch.randn(n, k * n_groups + k * int(strided), device=device, dtype=dtype)[:, :k * n_groups]
        else:
            b = torch.randn(k * n_groups + k * int(strided), n, device=device, dtype=dtype).t()[:, :k * n_groups]

        a.requires_grad_(True)
        b.requires_grad_(True)
        offs = torch.arange(k, n_groups * k + 1, k, device=device, dtype=torch.int32)

        f = torch._grouped_mm
        out = f(a, b.t(), offs=offs, out_dtype=torch.bfloat16)
        gO = torch.rand_like(out)
        out.backward(gO)
        offs_cpu = offs.cpu()
        alist, blist, agradlist, bgradlist = [], [], [], []
        start = 0
        for i in range(n_groups):
            alist.append(a[:, start:offs_cpu[i]])
            blist.append(b[:, start:offs_cpu[i]])
            agradlist.append(a.grad[:, start:offs_cpu[i]])
            bgradlist.append(b.grad[:, start:offs_cpu[i]])
            start = offs_cpu[i]
        self.grouped_mm_helper(alist, blist, gO, agradlist, bgradlist, out)

    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support CUTLASS")
    @xfailIfSM120OrLater
    @unittest.skipIf(not SM90OrLater, "Grouped gemm supported only on SM90 and SM100")
    @parametrize("strided", [False, True])
    @parametrize("a_row_major", [False, True])
    @parametrize("b_row_major", [False, True])
    def test_grouped_gemm_2d_3d(self, strided, a_row_major, b_row_major):
        device = "cuda"
        dtype = torch.bfloat16
        s_int = int(strided)
        m, n, k, n_groups = 16, 32, 64, 4
        if a_row_major:
            a = torch.randn(m * n_groups, k * (1 + s_int), device=device, dtype=dtype)[:, :k]
        else:
            a = torch.randn(k, (m + 2 * s_int) * n_groups, device=device, dtype=dtype).t()[:m * n_groups, :]

        if b_row_major:
            b = torch.randn(n_groups * (1 + s_int), n, k * (1 + s_int), device=device, dtype=dtype)[::(1 + s_int), :, :k]
        else:
            b = torch.randn(n_groups * (1 + s_int), k * (1 + s_int), n, device=device,
                            dtype=dtype).transpose(-2, -1)[::(1 + s_int), :, :k]

        a.requires_grad_(True)
        b.requires_grad_(True)

        a_contig = a if a_row_major else a.t()
        self.assertTrue(a_contig.is_contiguous() is not strided)
        b_contig = b if b_row_major else b.transpose(-2, -1)
        self.assertTrue(b_contig.is_contiguous() is not strided)
        for check_zero_size in (False, True):
            if check_zero_size and n_groups <= 1:
                continue

            a.grad = None
            b.grad = None
            offs = torch.arange(m, n_groups * m + 1, m, device="cuda", dtype=torch.int32)
            if check_zero_size:
                offs[0] = offs[1]

            f = torch._grouped_mm
            out = f(a, b.transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16)
            gO = torch.rand_like(out)
            if not check_zero_size:
                out.backward(gO)
            offs_cpu = offs.cpu()
            alist, agradlist, gOlist, outlist = [], [], [], []
            bgradlist = [None] * n_groups if check_zero_size else b.grad
            start = 0
            for i in range(n_groups):
                alist.append(a[start:offs_cpu[i]])
                agradlist.append(None if check_zero_size else a.grad[start:offs_cpu[i]])
                outlist.append(out[start:offs_cpu[i]])
                gOlist.append(gO[start:offs_cpu[i]])
                start = offs_cpu[i]
            self.grouped_mm_helper(alist, b, gOlist, agradlist, bgradlist, outlist)


    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support CUTLASS")
    @xfailIfSM120OrLater
    @unittest.skipIf(not SM90OrLater, "Grouped gemm supported only on SM90 and SM100")
    @parametrize("strided", [False, True])
    @parametrize("a_row_major", [False, True])
    @parametrize("b_row_major", [False, True])
    def test_grouped_gemm_3d_3d(self, strided, a_row_major, b_row_major):
        device = "cuda"
        dtype = torch.bfloat16
        s_int = int(strided)
        m, n, k, n_groups = 16, 32, 64, 4
        if a_row_major:
            a = torch.randn(n_groups * (1 + s_int), m, k * (1 + s_int), device=device, dtype=dtype)[::(1 + s_int), :, :k]
        else:
            a = torch.randn(n_groups * (1 + s_int), k * (1 + s_int), m, device=device,
                            dtype=dtype).transpose(-2, -1)[::(1 + s_int), :, :k]
        if b_row_major:
            b = torch.randn(n_groups * (1 + s_int), n, k * (1 + s_int), device=device, dtype=dtype)[::(1 + s_int), :, :k]
        else:
            b = torch.randn(n_groups * (1 + s_int), k * (1 + s_int), n, device=device,
                            dtype=dtype).transpose(-2, -1)[::(1 + s_int), :, :k]
        a.requires_grad_(True)
        b.requires_grad_(True)

        a_contig = a if a_row_major else a.transpose(-2, -1)
        self.assertTrue(a_contig.is_contiguous() is not strided)
        b_contig = b if b_row_major else b.transpose(-2, -1)
        self.assertTrue(b_contig.is_contiguous() is not strided)

        f = torch._grouped_mm
        out = f(a, b.transpose(-2, -1), out_dtype=torch.bfloat16)
        gO = torch.rand_like(out)
        out.backward(gO)
        self.grouped_mm_helper(a, b, gO, a.grad, b.grad, out)

    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support CUTLASS")
    @xfailIfSM120OrLater
    @unittest.skipIf(not SM90OrLater, "Grouped gemm supported only on SM90 and SM100")
    @parametrize("strided", [False, True])
    @parametrize("a_row_major", [False, True])
    @parametrize("b_row_major", [False, True])
    def test_grouped_gemm_3d_2d(self, strided, a_row_major, b_row_major):
        device = "cuda"
        dtype = torch.bfloat16
        s_int = int(strided)
        m, n, k, n_groups = 16, 32, 64, 4
        if a_row_major:
            a = torch.randn(n_groups * (1 + s_int), m, k * (1 + s_int), device=device, dtype=dtype)[::(1 + s_int), :, :k]
        else:
            a = torch.randn(n_groups * (1 + s_int), k * (1 + s_int), m, device=device,
                            dtype=dtype).transpose(-2, -1)[::(1 + s_int), :, :k]
        if b_row_major:
            b = torch.randn(n * n_groups, k * (1 + s_int), device=device, dtype=dtype)[:, :k]
        else:
            b = torch.randn(k, n * (n_groups + s_int), device=device, dtype=dtype).transpose(-2, -1)[:n * n_groups, :]

        a.requires_grad_(True)
        b.requires_grad_(True)

        a_contig = a if a_row_major else a.transpose(-2, -1)
        self.assertTrue(a_contig.is_contiguous() is not strided)
        b_contig = b if b_row_major else b.transpose(-2, -1)
        self.assertTrue(b_contig.is_contiguous() is not strided)
        for check_zero_size in (False, True):
            if check_zero_size and n_groups <= 1:
                continue

            offs = torch.arange(n, n_groups * n + 1, n, device="cuda", dtype=torch.int32)
            if check_zero_size:
                offs[0] = offs[1]

            f = torch._grouped_mm
            out = f(a, b.transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16)
            gO = torch.rand_like(out)
            if not check_zero_size:
                out.backward(gO)
            offs_cpu = offs.cpu()
            blist, outlist, bgradlist, gOlist = [], [], [], []
            agradlist = [None] * n_groups if check_zero_size else a.grad
            start = 0
            for i in range(n_groups):
                blist.append(b[start:offs_cpu[i]])
                bgradlist.append(b.grad[start:offs_cpu[i]])
                outlist.append(out[:, start:offs_cpu[i]])
                gOlist.append(gO[:, start:offs_cpu[i]])
                start = offs_cpu[i]
            self.grouped_mm_helper(a, blist, gOlist, agradlist, bgradlist, outlist)

    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support CUTLASS")
    @xfailIfSM100OrLater
    @unittest.skipIf(not SM90OrLater, "Grouped gemm supported on SM90")
    @parametrize("op", ["2d/2d", "2d/3d", "3d/2d", "3d/3d"])
    @parametrize("a_row_major", [False, True])
    @parametrize("b_row_major", [False, True])
    @parametrize("max_autotune", [False, True])
    def test_grouped_gemm_compiled(self, op, a_row_major, b_row_major, max_autotune):
        torch._dynamo.reset()

        device = "cuda"
        dtype_AB = torch.bfloat16
        dtype_offset = torch.int32

        align = 16 // dtype_AB.itemsize

        f_ref = torch._grouped_mm

        options = {}
        if max_autotune:
            options.update(
                {
                    "max_autotune": True,
                    "max_autotune_gemm_backends": "TRITON",
                }
            )
        f = torch.compile(
            f_ref,
            options=options,
        )

        if op == "2d/2d":
            m, n = 3, 7
            m_align = (m + align - 1) // align * align
            n_align = (n + align - 1) // align * align
            if not a_row_major and not b_row_major:
                offs = torch.tensor([0, 1, 6, 6, 7], device=device, dtype=dtype_offset)
            else:
                offs = torch.tensor([0, 8, 16, 16, 27], device=device, dtype=dtype_offset)
            ngroups = offs.shape[0]
            k = offs[-1]
            k_align = (k + align - 1) // align * align

            if a_row_major:
                A = torch.randn(m, k_align, device=device, dtype=dtype_AB)[:, :k]
            else:
                A = torch.randn(k, m_align, device=device, dtype=dtype_AB).t()[:m, :]
            if b_row_major:
                B = torch.randn(n, k_align, device=device, dtype=dtype_AB)[:, :k]
            else:
                B = torch.randn(k, n_align, device=device, dtype=dtype_AB).t()[:n, :]
        elif op == "2d/3d":
            n, k = 7, 13
            n_align = (n + align - 1) // align * align
            k_align = (k + align - 1) // align * align
            if a_row_major:
                offs = torch.tensor([0, 1, 3, 3, 5], device=device, dtype=dtype_offset)
            else:
                offs = torch.tensor([0, 8, 16, 16, 19], device=device, dtype=dtype_offset)
            ngroups = offs.shape[0]
            m = offs[-1]
            m_align = (m + align - 1) // align * align

            if a_row_major:
                A = torch.randn(m, k_align, device=device, dtype=dtype_AB)[:, :k]
            else:
                A = torch.randn(k, m_align, device=device, dtype=dtype_AB).t()[:m, :]
            if b_row_major:
                B = torch.randn(ngroups, n, k_align, device=device, dtype=dtype_AB)[:, :, :k]
            else:
                B = torch.randn(ngroups, k, n_align, device=device, dtype=dtype_AB).transpose(
                    -2, -1
                )[:, :n, :]
        elif op == "3d/2d":
            m, k = 3, 13
            m_align = (m + align - 1) // align * align
            k_align = (k + align - 1) // align * align
            offs = torch.tensor([0, 8, 16, 16, 19], device=device, dtype=dtype_offset)
            ngroups = offs.shape[0]
            n = offs[-1]
            n_align = (n + align - 1) // align * align

            if a_row_major:
                A = torch.randn(ngroups, m, k_align, device=device, dtype=dtype_AB)[:, :, :k]
            else:
                A = torch.randn(ngroups, k, m_align, device=device, dtype=dtype_AB).transpose(
                    -2, -1
                )[:, :m, :]
            if b_row_major:
                B = torch.randn(n, k_align, device=device, dtype=dtype_AB)[:, :k]
            else:
                B = torch.randn(k, n_align, device=device, dtype=dtype_AB).t()[:n, :]
        elif op == "3d/3d":
            offs = None
            ngroups = 5
            m, n, k = 3, 7, 13
            m_align = (m + align - 1) // align * align
            n_align = (n + align - 1) // align * align
            k_align = (k + align - 1) // align * align
            if a_row_major:
                A = torch.randn(ngroups, m, k_align, device=device, dtype=dtype_AB)[:, :, :k]
            else:
                A = torch.randn(ngroups, k, m_align, device=device, dtype=dtype_AB).transpose(
                    -2, -1
                )[:, :m, :]
            if b_row_major:
                B = torch.randn(ngroups, n, k_align, device=device, dtype=dtype_AB)[:, :, :k]
            else:
                B = torch.randn(ngroups, k, n_align, device=device, dtype=dtype_AB).transpose(
                    -2, -1
                )[:, :n, :]
        else:
            raise AssertionError(f"Invalid op: {op}")

        C_ref = f_ref(A, B.transpose(-2, -1), offs=offs)
        C = f(A, B.transpose(-2, -1), offs=offs)
        torch.testing.assert_close(C, C_ref)


    @onlyCUDA
    @skipIfRocm
    @parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
    @parametrize("M", [1, 32, 64])
    @parametrize("N", [1, 32, 64])
    @parametrize("K", [1, 32, 64])
    @parametrize("batch_size", [None, 1, 16])
    @parametrize("backend", ["cublas", "cublaslt"])
    def test_mm_bmm_dtype_overload(self, input_dtype, M, N, K, batch_size, backend):
        device = "cuda"
        dtype = input_dtype
        with blas_library_context(backend):
            def create_inputs(B=None):
                if B is None:
                    a = torch.randn(M, K, device=device, dtype=dtype)
                    b = torch.randn(K, N, device=device, dtype=dtype)
                else:
                    a = torch.randn(B, M, K, device=device, dtype=dtype)
                    b = torch.randn(B, K, N, device=device, dtype=dtype)
                return a, b

            a, b = create_inputs(batch_size)

            a_fp32, b_fp32 = a.to(torch.float32), b.to(torch.float32)

            output_dtypes = [torch.float32]

            if input_dtype != torch.float32:
                output_dtypes.append(input_dtype)

            for output_dtype in output_dtypes:
                # Catch edge case of incompat with bfloat16 and major version < 8
                if input_dtype == torch.bfloat16 and not PLATFORM_SUPPORTS_BF16:
                    if output_dtype == torch.bfloat16:
                        continue

                    if batch_size:
                        with self.assertRaises(RuntimeError):
                            torch.bmm(a, b, out_dtype=output_dtype)
                    else:
                        with self.assertRaises(RuntimeError):
                            torch.mm(a, b, out_dtype=output_dtype)
                else:
                    if batch_size:
                        out = torch.bmm(a, b, out_dtype=output_dtype)
                        baseline = torch.bmm(a_fp32, b_fp32) if output_dtype == torch.float32 else torch.bmm(a, b)
                    else:
                        out = torch.mm(a, b, out_dtype=output_dtype)
                        baseline = torch.mm(a_fp32, b_fp32) if output_dtype == torch.float32 else torch.mm(a, b)

                    self.assertEqual(out.dtype, output_dtype)

                    torch.testing.assert_close(out, baseline, atol=1e-3, rtol=1e-3)


    @onlyCUDA
    @skipIfRocm
    @parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
    @parametrize("M", [1, 32, 64])
    @parametrize("N", [1, 32, 64])
    @parametrize("K", [1, 32, 64])
    @parametrize("batch_size", [None, 1, 32])
    @parametrize("backend", ["cublas", "cublaslt"])
    def test_addmm_baddmm_dtype_overload(self, input_dtype, M, N, K, batch_size, backend):
        device = "cuda"
        dtype = input_dtype
        with blas_library_context(backend):
            def create_inputs(B=None):
                if B is None:
                    a = torch.randn(M, K, device=device, dtype=dtype)
                    b = torch.randn(K, N, device=device, dtype=dtype)
                    c = torch.randn(M, N, device=device, dtype=dtype)
                else:
                    a = torch.randn(B, M, K, device=device, dtype=dtype)
                    b = torch.randn(B, K, N, device=device, dtype=dtype)
                    c = torch.randn(B, M, N, device=device, dtype=dtype)

                return a, b, c

            a, b, c = create_inputs(batch_size)

            a_fp32, b_fp32, c_fp32 = a.to(torch.float32), b.to(torch.float32), c.to(torch.float32)

            output_dtypes = [torch.float32]

            if input_dtype != torch.float32:
                output_dtypes.append(input_dtype)

            for output_dtype in output_dtypes:
                # Catch edge case of incompat with bfloat16 and major version < 8
                if input_dtype == torch.bfloat16 and not PLATFORM_SUPPORTS_BF16:
                    if output_dtype == torch.bfloat16:
                        continue

                    if batch_size:
                        with self.assertRaises(RuntimeError):
                            torch.baddbmm(c, a, b, out_dtype=output_dtype)
                    else:
                        with self.assertRaises(RuntimeError):
                            torch.addmm(c, a, b, out_dtype=output_dtype)
                else:
                    if batch_size:
                        out = torch.baddbmm(c, a, b, out_dtype=output_dtype)
                        if output_dtype == torch.float32:
                            baseline = torch.baddbmm(c_fp32, a_fp32, b_fp32)
                        else:
                            baseline = torch.baddbmm(c, a, b)
                    else:
                        out = torch.addmm(c, a, b, out_dtype=output_dtype)
                        if output_dtype == torch.float32:
                            baseline = torch.addmm(c_fp32, a_fp32, b_fp32)
                        else:
                            baseline = torch.addmm(c, a, b)

                    self.assertEqual(out.dtype, output_dtype)
                    torch.testing.assert_close(out, baseline, atol=1e-3, rtol=1e-3)


    @onlyCUDA
    @skipIfRocm
    @parametrize("batch_size", [1, 32])
    @parametrize("backend", ["cublas", "cublaslt"])
    def test_fp16_accum_and_fp32_out_failure(self, batch_size, backend):
        M, N, K = 32, 32, 32
        device = "cuda"
        dtype = torch.float16
        with blas_library_context(backend):
            torch.backends.cuda.preferred_blas_library(backend)

            orig_fp16_accum = torch.backends.cuda.matmul.allow_fp16_accumulation
            torch.backends.cuda.matmul.allow_fp16_accumulation = True

            def create_inputs():
                a = torch.randn(M, K, device=device, dtype=dtype)
                b = torch.randn(K, N, device=device, dtype=dtype)
                c = torch.randn(M, N, device=device, dtype=dtype)
                return a, b, c

            def expand(tensor):
                return tensor.unsqueeze(0).expand(batch_size, *tensor.shape)

            a, b, c = create_inputs()

            with self.assertRaises(Exception):
                torch.baddbmm(expand(c), expand(a), expand(b), out_dtype=torch.float32)

            with self.assertRaises(Exception):
                torch.addmm(c, a, b, out_dtype=torch.float32)

            with self.assertRaises(Exception):
                torch.bmm(expand(a,), expand(b), out_dtype=torch.float32)

            with self.assertRaises(Exception):
                torch.mm(a, b, out_dtype=torch.float32)

            torch.backends.cuda.matmul.allow_fp16_accumulation = orig_fp16_accum

f8_msg = "FP8 is only supported on H100+, SM 8.9 and MI300+ devices"
mx_skip_msg = "MX gemm is only supported on CUDA capability 10.0+"

# avoid division by zero when calculating scale
EPS = 1e-12

def amax_to_scale(
    amax: torch.Tensor, float8_dtype: torch.dtype, orig_dtype: torch.dtype
):
    """ Converts the amax value of a tensor to the fp8 scale.
    Args:
        amax: The amax value of the tensor.
        float8_dtype: the float8 dtype.
        orig_dtype: The original dtype of the tensor.
    """
    scale = torch.empty_like(amax, dtype=torch.float32)
    if float8_dtype == e4m3_type:
        res = E4M3_MAX_POS / torch.clamp(amax, min=EPS)
    elif float8_dtype == e5m2_type:
        res = E4M3_MAX_POS / torch.clamp(amax, min=EPS)
    else:
        raise ValueError(f"Unsupported float8_dtype: {float8_dtype}")

    # Ensure the scale is representable in float16,
    # this helps when amax is small. We are assuming that we don't need
    # to care about this for float32/bfloat16
    if orig_dtype is torch.float16:
        res = torch.clamp(res, max=torch.finfo(torch.float16).max)

    scale.copy_(res)
    return scale

def tensor_to_scale(x: torch.Tensor, float8_dtype: torch.dtype, dim=None):
    if dim is None:
        amax = torch.max(torch.abs(x))
    else:
        amax = torch.max(torch.abs(x), dim=dim, keepdim=True).values

    return amax_to_scale(amax, float8_dtype, x.dtype)

def mm_float8_emulated(x, x_scale, y, y_scale, out_dtype) -> torch.Tensor:
    # naive implementation: dq -> op -> q
    x_fp32 = x.to(torch.float) / x_scale
    y_fp32 = y.to(torch.float) / y_scale
    out_fp32 = torch.mm(x_fp32, y_fp32)

    return out_fp32.to(out_dtype)

def addmm_float8_unwrapped(
    a_data: torch.Tensor,
    a_scale: torch.Tensor,
    b_data: torch.Tensor,
    b_scale: torch.tensor,
    output_dtype: torch.dtype,
    output_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    a_inverse_scale = a_scale.reciprocal()
    b_inverse_scale = b_scale.reciprocal()
    if output_dtype == torch.float32 and bias is not None:
        # Bias is not supported by _scaled_mm when output is fp32
        output = torch._scaled_mm(
            a_data,
            b_data,
            scale_a=a_inverse_scale,
            scale_b=b_inverse_scale,
            scale_result=output_scale,
            out_dtype=output_dtype,
        )
        output += bias
        return output
    output = torch._scaled_mm(
        a_data,
        b_data,
        bias=bias,
        scale_a=a_inverse_scale,
        scale_b=b_inverse_scale,
        scale_result=output_scale,
        out_dtype=output_dtype,
    )
    return output

def mm_float8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    output_dtype: torch.dtype,  # output dtype
    output_scale: Optional[torch.Tensor] = None,  # output scale, precomputed
) -> torch.Tensor:
    return addmm_float8_unwrapped(
        a, a_scale, b, b_scale, output_dtype, output_scale
    )

def to_fp8_saturated(
    x: torch.Tensor,
    fp8_dtype: torch.dtype
):
    if fp8_dtype == e4m3_type:
        x = x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
    elif fp8_dtype == e5m2_type:
        x = x.clamp(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
    else:
        raise ValueError(f"to_fp8_saturated(): Unsupported fp8_dtype: {fp8_dtype}")

    return x.to(fp8_dtype)

def compute_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the error between two tensors in dB.

    For more details see:
        https://en.wikipedia.org/wiki/Signal-to-noise_ratio

    Args:
        x: The original tensor.
        y: The tensor to compare to the original tensor.
    """
    Ps = torch.norm(x)
    Pn = torch.norm(x - y)
    return 20 * torch.log10(Ps / Pn)


# largest power of 2 representable in `torch.float8_e4m3fn`
F8E4M3_LARGEST_POW2 = 8
# max value of `torch.float8_e4m3fn` (448)
F8E4M3_MAX_VAL = torch.finfo(torch.float8_e4m3fn).max
# exponent bias of `torch.float8_e8m0fnu`
F8E8M0_EXP_BIAS = 127
# exponent and mantissa bits of `torch.float4_e2m1fn_x2`
FP4_EBITS, FP4_MBITS = 2, 1
FP4_MAX_VAL = 6.0

def data_to_mx_scale(x, block_size):
    # simple implementation of https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
    # section 6.3, not all edge cases (such as NaN) are handled/tested
    orig_shape = x.shape
    x = x.reshape(-1, block_size)
    max_abs = torch.amax(torch.abs(x), 1)
    largest_p2_lt_max_abs = torch.floor(torch.log2(max_abs))
    scale_e8m0_unbiased = largest_p2_lt_max_abs - F8E4M3_LARGEST_POW2
    scale_e8m0_unbiased = torch.clamp(scale_e8m0_unbiased, -1 * F8E8M0_EXP_BIAS, F8E8M0_EXP_BIAS)
    scale_e8m0_biased = scale_e8m0_unbiased + F8E8M0_EXP_BIAS
    scale_e8m0_biased = scale_e8m0_biased.to(torch.uint8)
    scale_e8m0_biased = scale_e8m0_biased.view(torch.float8_e8m0fnu)
    return scale_e8m0_biased.reshape(orig_shape[0], -1)


def data_to_nvfp4_scale(x, block_size):
    orig_shape = x.shape
    x = x.reshape(-1, block_size)
    max_abs = torch.amax(torch.abs(x), 1) + 1e-12

    # x_orig_max / scale = x_in_fp4_domain_max
    # x_orig_max / x_in_fp4_domain_max = scale
    scale = max_abs / FP4_MAX_VAL

    # for the purposes of this function, just clamp to representable range of
    # `torch.float8_e4m3fn`. In real code, we would expect the modeling code to
    # handle this before the input data hits this function.
    scale = scale.clamp(max=F8E4M3_MAX_VAL)

    # cast to target dtype
    scale = scale.to(torch.float8_e4m3fn)
    scale = scale.reshape(orig_shape[0], -1)
    return scale


def down_size(size):
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def pack_uint4(uint8_data) -> torch.Tensor:
    # converting to uint8 for operations
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[1::2] << 4 | uint8_data[::2]).view(down_size(shape))


def _bfloat16_to_float4_e2m1fn_x2(x):
    assert x.dtype == torch.bfloat16
    x = _f32_to_floatx_unpacked(x.float(), FP4_EBITS, FP4_MBITS)
    x = pack_uint4(x)
    x = x.view(torch.float4_e2m1fn_x2)
    return x


class TestFP8Matmul(TestCase):

    def _test_tautological_mm(self, device: str = "cuda",
                              x_dtype: torch.dtype = e4m3_type,
                              y_dtype: torch.dtype = e4m3_type,
                              out_dtype: Optional[torch.dtype] = None,
                              size: int = 16) -> None:
        if device != "cpu" and torch.cuda.is_available() and not PLATFORM_SUPPORTS_FP8:
            raise unittest.SkipTest(f8_msg)
        x_fp8 = torch.rand(size, size, device=device).to(x_dtype)
        y_fp8 = torch.eye(size, device=device, dtype=y_dtype).t()
        out_fp32 = torch.mm(x_fp8.to(torch.float), y_fp8.to(torch.float))
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        out_fp8 = torch._scaled_mm(x_fp8, y_fp8, scale_a, scale_b, out_dtype=out_dtype)
        if out_dtype is not None:
            self.assertEqual(out_dtype, out_fp8.dtype)
        self.assertEqual(out_fp32, out_fp8.to(torch.float))

    def test_float8_basics(self, device) -> None:
        if device != "cpu" and torch.cuda.is_available() and not PLATFORM_SUPPORTS_FP8:
            raise unittest.SkipTest(f8_msg)
        self._test_tautological_mm(device, e4m3_type, e4m3_type, size=16)
        # According to https://docs.nvidia.com/cuda/cublas/#id99 8F_E5M2 MM is unsupported
        # supported on ROCm but fails on CUDA
        ctx = self.assertRaises(RuntimeError) if torch.version.hip is None and device != "cpu" else contextlib.nullcontext()
        with ctx:
            self._test_tautological_mm(device, e5m2_type, e5m2_type)

        self._test_tautological_mm(device, e4m3_type, e5m2_type, size=32)
        self._test_tautological_mm(device, e5m2_type, e4m3_type, size=48)

        self._test_tautological_mm(device, size=64, out_dtype=torch.float16)
        self._test_tautological_mm(device, size=96, out_dtype=torch.float32)
        self._test_tautological_mm(device, size=80, out_dtype=torch.bfloat16)

        with self.assertRaises(AssertionError if torch.version.hip or device == "cpu" else RuntimeError):
            self._test_tautological_mm(device, out_dtype=e5m2_type)

    def test_float8_scale(self, device) -> None:
        if device != "cpu" and torch.cuda.is_available() and not PLATFORM_SUPPORTS_FP8:
            raise unittest.SkipTest(f8_msg)
        size = (16, 16)
        x = torch.full(size, .5, device=device, dtype=e4m3_type)
        # hipblaslt does not yet support mixed e4m3_type input
        y_type = e4m3_type if torch.version.hip else e5m2_type
        y = torch.full(size, .5, device=device, dtype=y_type).t()
        scale_one = torch.tensor(1.0, device=device)
        scale_a = torch.tensor(1.5, device=device)
        scale_b = torch.tensor(0.66, device=device)
        out_fp8 = torch._scaled_mm(x, y, scale_a=scale_one, scale_b=scale_one)
        self.assertEqual(out_fp8.to(torch.float), torch.full(size, 4., device=device))
        out_fp8_s = torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b)
        self.assertEqual(out_fp8, out_fp8_s)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("base_dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_scaled_mm_vs_emulated(self, base_dtype):
        torch.manual_seed(42)
        input_dtype = e4m3_type
        output_dtype = base_dtype
        compare_type = torch.float32

        x = torch.randn(16, 16, device="cuda", dtype=base_dtype)
        y = torch.randn(32, 16, device="cuda", dtype=base_dtype).t()

        x_scale = tensor_to_scale(x, input_dtype).float()
        y_scale = tensor_to_scale(y, input_dtype).float()

        x_fp8 = to_fp8_saturated(x * x_scale, input_dtype)
        y_fp8 = to_fp8_saturated(y * y_scale, input_dtype)

        # Calculate actual F8 mm
        out_scaled_mm = mm_float8(
            x_fp8,
            y_fp8,
            a_scale=x_scale,
            b_scale=y_scale,
            output_dtype=output_dtype
        )

        # Calculate emulated F8 mm
        out_emulated = mm_float8_emulated(
            x_fp8,
            x_scale,
            y_fp8,
            y_scale,
            output_dtype
        )

        if output_dtype != base_dtype:
            out_scaled_mm = out_scaled_mm.to(compare_type)
            out_scaled_mm = out_scaled_mm / tensor_to_scale(out_scaled_mm, input_dtype)

            out_emulated = out_emulated.to(compare_type)
            out_emulated = out_emulated / tensor_to_scale(out_emulated, input_dtype)

        if base_dtype in {torch.bfloat16, torch.float16}:
            atol, rtol = 7e-2, 7e-2
        else:
            atol, rtol = 3e-3, 3e-3

        torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("base_dtype", [torch.float16, torch.bfloat16, torch.float32])
    def test_scaled_mm_change_stride(self, base_dtype):
        torch.manual_seed(42)
        input_dtype = e4m3_type
        output_dtype = base_dtype
        compare_type = torch.float32

        x = torch.empty_strided((16, 16), (16, 1), device="cuda", dtype=base_dtype)
        y = torch.empty_strided((16, 32), (1, 64), device="cuda", dtype=base_dtype)

        x.normal_()
        y.normal_()

        x_scale = tensor_to_scale(x, input_dtype).float()
        y_scale = tensor_to_scale(y, input_dtype).float()

        x_fp8 = to_fp8_saturated(x * x_scale, input_dtype)
        y_fp8 = to_fp8_saturated(y * y_scale, input_dtype)

        # Calculate actual F8 mm
        out_scaled_mm = mm_float8(
            x_fp8,
            y_fp8,
            a_scale=x_scale,
            b_scale=y_scale,
            output_dtype=output_dtype
        )

        # Calculate emulated F8 mm
        out_emulated = mm_float8_emulated(
            x_fp8,
            x_scale,
            y_fp8,
            y_scale,
            output_dtype
        )

        if output_dtype != base_dtype:
            out_scaled_mm = out_scaled_mm.to(compare_type)
            out_scaled_mm = out_scaled_mm / tensor_to_scale(out_scaled_mm, input_dtype)

            out_emulated = out_emulated.to(compare_type)
            out_emulated = out_emulated / tensor_to_scale(out_emulated, input_dtype)

        if base_dtype in {torch.bfloat16, torch.float16}:
            atol, rtol = 7e-2, 7e-2
        else:
            atol, rtol = 3e-3, 3e-3

        torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)

    @onlyCUDA
    def test_float8_bias(self, device) -> None:
        if device != "cpu" and torch.cuda.is_available() and not PLATFORM_SUPPORTS_FP8:
            raise unittest.SkipTest(f8_msg)
        (k, l, m) = (16, 48, 32)
        x = torch.ones((k, l), device=device).to(e4m3_type)
        y = torch.full((m, l), .25, device=device, dtype=e4m3_type).t()
        bias = torch.full((m,), 4.0, device=device, dtype=torch.half)
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        out_fp8 = torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b)
        outb_fp8 = torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b, bias=bias)
        # this fails on ROCm currently because hipblaslt doesn't have amax op
        out_fp32 = out_fp8.to(torch.float32)
        outb_fp32 = outb_fp8.to(torch.float32)
        difference = torch.abs(out_fp32 - outb_fp32)
        self.assertEqual(difference, torch.tensor(4.0, device=device).expand_as(out_fp32))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("bias", [True, False])
    def test_non_divisible_leading_dim(self, device, bias: bool) -> None:
        x = torch.rand((17, 16), device=device).to(e4m3_type)
        y = torch.rand((16, 16), device=device).to(e4m3_type).t()
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        input_bias = None
        if bias:
            input_bias = torch.rand((16,), device=device).to(torch.half)
        _ = torch._scaled_mm(x, y, scale_a, scale_b, bias=input_bias)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    def test_float8_bias_relu_edgecase(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.full((k, l), 0.0, device=device).to(e4m3_type)
        y = torch.full((m, l), 1.0, device=device, dtype=e4m3_type).t()
        bias = torch.full((m,), -3.0, device=device, dtype=torch.half)
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        outb_fp8 = torch._scaled_mm(x, y, scale_a, scale_b, bias=bias)
        outb_fp32 = outb_fp8.to(torch.float32)
        self.assertEqual(outb_fp32, torch.tensor(-3.0, device=device).expand_as(outb_fp32))

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    def test_float32_output_errors_with_bias(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.rand((k, l), device=device).to(e4m3_type)
        y = torch.full((m, l), .25, device=device, dtype=e4m3_type).t()
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        bias = torch.full((m,), 4.0, device=device, dtype=torch.bfloat16)
        self.assertRaisesRegex(
            RuntimeError,
            "Bias is not supported when out_dtype is set to Float32",
            lambda: torch._scaled_mm(x, y, scale_a, scale_b, bias=bias, out_dtype=torch.float32),
        )

    @onlyCUDA
    @unittest.skipIf(PLATFORM_SUPPORTS_FP8 or not torch.cuda.is_available(), f8_msg)
    def test_error_message_fp8_pre_sm89(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.rand((k, l), device=device).to(e4m3_type)
        y = torch.rand((m, l), device=device).to(e4m3_type).t()
        scale_a = torch.tensor(1.0, device=device)
        scale_b = torch.tensor(1.0, device=device)
        self.assertRaisesRegex(
            RuntimeError,
            r"torch\.\_scaled\_mm is only supported on CUDA devices with compute capability \>\= 9\.0 or 8\.9, or ROCm MI300\+",
            lambda: torch._scaled_mm(x, y, scale_a, scale_b, out_dtype=torch.float32),
        )

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    def test_float8_scale_fast_accum(self, device) -> None:
        size = (16, 16)
        x = torch.full(size, .5, device=device, dtype=e4m3_type)
        # hipblaslt does not yet support mixed e4m3_type input
        y_type = e4m3_type if torch.version.hip else e5m2_type
        y = torch.full(size, .5, device=device, dtype=y_type).t()
        scale_a = torch.tensor(1.5, device=device)
        scale_b = torch.tensor(0.66, device=device)
        out_fp8 = torch._scaled_mm(x, y, scale_a, scale_b, use_fast_accum=True)
        self.assertEqual(out_fp8.to(torch.float), torch.full(size, 4., device=device))
        out_fp8_s = torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b, use_fast_accum=True)
        self.assertEqual(out_fp8, out_fp8_s)

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8 or IS_WINDOWS, f8_msg)
    @unittest.skipIf(not SM89OrLater, "rowwise implementation is currently sm89-sm100 specific")
    @parametrize("use_fast_accum", [True, False])
    def test_float8_rowwise_scaling_sanity(self, device, use_fast_accum: bool) -> None:
        M, K, N = (1024, 512, 2048)
        fill_value = 0.5
        x = torch.full((M, K), fill_value, device=device)
        y = torch.full((N, K), fill_value, device=device)

        x_scales = torch.ones((x.shape[0], 1), device=device, dtype=torch.float32)
        y_scales = torch.ones((1, y.shape[0]), device=device, dtype=torch.float32)

        x_fp8 = x.to(e4m3_type)
        y_fp8 = y.to(e4m3_type).t()

        out_fp8 = torch._scaled_mm(
            x_fp8,
            y_fp8,
            scale_a=x_scales,
            scale_b=y_scales,
            out_dtype=torch.bfloat16,
            use_fast_accum=use_fast_accum,
        )
        self.assertEqual(
            out_fp8.to(torch.float32), torch.full((M, N), K * (fill_value**2), device=device)
        )

    @onlyCUDA
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8 or IS_WINDOWS, f8_msg)
    def test_float8_error_messages(self, device) -> None:
        M, K, N = (1024, 512, 2048)
        fill_value = 0.5
        x = torch.full((M, K), fill_value, device=device)
        y = torch.full((N, K), fill_value, device=device)

        x_fp8 = x.to(e4m3_type)
        y_fp8 = y.to(e4m3_type).t()

        with self.assertRaisesRegex(
            RuntimeError,
            re.escape(
                "For RowWise scaling, scale_a should be (1024, 1) and scale_b "
                "should be (1, 2048). Got scale_a.size()=(1, 1) and scale_b.size()=(1, 2)"
            ),
        ):
            torch._scaled_mm(
                x_fp8,
                y_fp8,
                scale_a=torch.ones((1, 1), device="cuda"),
                scale_b=torch.ones((1, 2), device="cuda"),
                out_dtype=torch.bfloat16,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            re.escape(
                " For RowWise scaling, scale_a should be (1024, 1) and scale_b "
                "should be (1, 2048). Got scale_a.size()=(1024, 1) and scale_b.size()=(1, 2049)"
            ),
        ):
            torch._scaled_mm(
                x_fp8,
                y_fp8,
                scale_a=torch.ones((M, 1), device="cuda"),
                scale_b=torch.ones((1, N + 1), device="cuda"),
                out_dtype=torch.bfloat16,
            )
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape("For non-TensorWise scaling, scale tensors must be 2-dimensional"),
        ):
            torch._scaled_mm(
                x_fp8,
                y_fp8,
                scale_a=torch.ones((M), device="cuda"),
                scale_b=torch.ones((N, N), device="cuda"),
                out_dtype=torch.bfloat16,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            re.escape(
                "Both scale_a and scale_b must be contiguous for RowWise scaling."
            ),
        ):
            torch._scaled_mm(
                x_fp8,
                y_fp8,
                scale_a=torch.ones((M, 1), device="cuda"),
                scale_b=torch.ones((1, N * 2), device="cuda")[:, ::2],
                out_dtype=torch.bfloat16,
            )

        # Note re.compile is used, not re.escape. This is to accommodate fn vs fnuz type message.
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected b\.dtype\(\) == at::kFloat8_e4m3fnu?z? to be true, but got false\.",
        ):
            torch._scaled_mm(
                x_fp8,
                y_fp8.to(e5m2_type),
                scale_a=torch.ones((M, 1), device="cuda"),
                scale_b=torch.ones((1, N), device="cuda"),
                out_dtype=torch.bfloat16,
            )

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8 or IS_WINDOWS, f8_msg)
    @unittest.skipIf(not SM89OrLater, "rowwise implementation is currently sm89-sm100 specific")
    @parametrize("base_dtype", [torch.bfloat16, torch.float32])
    def test_scaled_mm_vs_emulated_row_wise(self, base_dtype):
        # Fp32 out_dtype is only supported by cuBLAS, which however only started
        # shipping row-wise kernels in CUDA 12.9, and only for sm90+.
        if base_dtype is torch.float32:
            if _get_torch_cuda_version() < (12, 9):
                raise unittest.SkipTest("Need CUDA 12.9+ for row-wise fp8 w/ cuBLAS")
            if torch.cuda.get_device_capability() < (9, 0):
                raise unittest.SkipTest("Need sm90+ for row-wise fp8 w/ cuBLAS")

        torch.manual_seed(42)
        input_dtype = e4m3_type
        output_dtype = base_dtype

        x = torch.randn(16, 16, device="cuda", dtype=base_dtype)
        y = torch.randn(32, 16, device="cuda", dtype=base_dtype).t()

        x_scales = tensor_to_scale(x, input_dtype, dim=1).float()
        y_scales = tensor_to_scale(y, input_dtype, dim=0).float()

        x_fp8 = to_fp8_saturated(x * x_scales, e4m3_type)
        y_fp8 = to_fp8_saturated(y * y_scales, e4m3_type)

        # Calculate actual F8 mm
        out_scaled_mm = mm_float8(
            x_fp8, y_fp8, a_scale=x_scales, b_scale=y_scales, output_dtype=output_dtype
        )

        # Calculate emulated F8 mm
        out_emulated = mm_float8_emulated(
            x_fp8, x_scales, y_fp8, y_scales, output_dtype
        )

        if base_dtype in {torch.bfloat16, torch.float16}:
            atol, rtol = 7e-2, 7e-2
        else:
            atol, rtol = 2e-3, 2e-3

        torch.testing.assert_close(out_scaled_mm, out_emulated, atol=atol, rtol=rtol)

    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @parametrize("which_dim_zero", [0, 1, 2])
    @parametrize("use_torch_compile", [False, True])
    def test_zero_dim_tensorwise(self, which_dim_zero, use_torch_compile) -> None:
        device = "cuda"
        x_dtype, y_dtype = torch.float8_e4m3fn, torch.float8_e4m3fn
        out_dtype = torch.bfloat16
        M, K, N = 32, 32, 32
        if which_dim_zero == 0:
            M = 0
        elif which_dim_zero == 1:
            K = 0
        elif which_dim_zero == 2:
            N = 0

        x_fp8 = torch.zeros(M, K, device=device).to(x_dtype)
        y_fp8 = torch.zeros(N, K, device=device, dtype=y_dtype).t()
        out_fp32 = torch.mm(x_fp8.to(torch.float), y_fp8.to(torch.float))
        scale_a = torch.tensor(float('-inf'), device=device)
        scale_b = torch.tensor(float('-inf'), device=device)
        f = torch._scaled_mm
        if use_torch_compile:
            f = torch.compile(torch._scaled_mm)
        out_fp8 = f(x_fp8, y_fp8, scale_a, scale_b, out_dtype=out_dtype)
        self.assertEqual(out_dtype, out_fp8.dtype)
        self.assertEqual(out_fp32, out_fp8.to(torch.float))

    @unittest.skipIf(IS_WINDOWS, "Windows doesn't support row-wise scaling")
    @unittest.skipIf(not PLATFORM_SUPPORTS_FP8, f8_msg)
    @unittest.skipIf(not SM90OrLater, "sm89 kernel isn't opted into carveout yet")
    def test_honor_sm_carveout(self) -> None:
        torch.manual_seed(42)

        x = torch.randn(8192, 2048, device="cuda", dtype=torch.float32)
        y = torch.randn(8192, 2048, device="cuda", dtype=torch.float32).t()
        x_scales = tensor_to_scale(x, e4m3_type, dim=1).reciprocal()
        y_scales = tensor_to_scale(y, e4m3_type, dim=0).reciprocal()
        x_fp8 = to_fp8_saturated(x / x_scales, e4m3_type)
        y_fp8 = to_fp8_saturated(y / y_scales, e4m3_type)

        with tempfile.NamedTemporaryFile() as f:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                self.assertIsNone(torch._C._get_sm_carveout_experimental())
                torch._scaled_mm(x_fp8, y_fp8, scale_a=x_scales, scale_b=y_scales, out_dtype=torch.bfloat16)
                torch._C._set_sm_carveout_experimental(0)
                self.assertEqual(torch._C._get_sm_carveout_experimental(), 0)
                torch._scaled_mm(x_fp8, y_fp8, scale_a=x_scales, scale_b=y_scales, out_dtype=torch.bfloat16)
                torch._C._set_sm_carveout_experimental(66)
                self.assertEqual(torch._C._get_sm_carveout_experimental(), 66)
                torch._scaled_mm(x_fp8, y_fp8, scale_a=x_scales, scale_b=y_scales, out_dtype=torch.bfloat16)
                torch._C._set_sm_carveout_experimental(None)
                self.assertIsNone(torch._C._get_sm_carveout_experimental())
                torch._scaled_mm(x_fp8, y_fp8, scale_a=x_scales, scale_b=y_scales, out_dtype=torch.bfloat16)

            prof.export_chrome_trace(f.name)
            if torch.version.hip:
                events = [evt for evt in json.load(open(f.name))["traceEvents"] if evt.get("cat", "") == "kernel"]
                # events were returned out of order; need to be sorted on "ts" timestamp
                events = sorted(events, key=lambda x: x['ts'])
                # ROCm carveout is invisible except for kernels running slower on fewer CUs
                no_carveout, carveout_0, carveout_66, no_carveout_again = [float(evt.get("dur", "0.0")) for evt in events]
                self.assertTrue(no_carveout < carveout_66)
                self.assertTrue(carveout_0 < carveout_66)
                self.assertTrue(no_carveout_again < carveout_66)
                # ROCm carveout will create new streams when enabled, and go back to the original stream when disabled
                no_carveout, carveout_0, carveout_66, no_carveout_again = [int(evt.get("tid", "0")) for evt in events]
                self.assertTrue(no_carveout == no_carveout_again)
                self.assertTrue(no_carveout != carveout_0)
                self.assertTrue(no_carveout != carveout_66)
                self.assertTrue(carveout_0 != carveout_66)
            else:
                no_carveout, carveout_0, carveout_66, no_carveout_again = [
                    math.prod(evt.get("args", {}).get("grid", []))
                    for evt in json.load(open(f.name))["traceEvents"]
                    if evt.get("cat", "") == "kernel"
                ]

                self.assertEqual(no_carveout, no_carveout_again)
                capability = torch.cuda.get_device_capability()
                if capability == (10, 0):
                    # expected failure
                    # CUTLASS only supports SM carveout via green contexts on SM100
                    self.assertEqual(no_carveout, carveout_66)
                    self.assertEqual(carveout_66, carveout_0)
                else:
                    # correct behavior
                    self.assertNotEqual(no_carveout, carveout_66)
                    self.assertNotEqual(carveout_66, carveout_0)

    def test_pack_uint4(self):
        """
        Verify that given a tensor with high precision values [val0, val1],
        the x2 packed representation is val1:val0 (from MSB to LSB), and
        not val0:val1.

        Note that the packing function is private to this file, but it's still
        good to test that we are packing in the expected way.
        """
        hp_data = torch.tensor([0b00000010, 0b00001011], dtype=torch.uint8)
        lp_data_actual = pack_uint4(hp_data)
        lp_data_expected = torch.tensor([0b10110010], dtype=torch.uint8)
        torch.testing.assert_close(lp_data_actual, lp_data_expected, atol=0, rtol=0)

    @unittest.skipIf(not PLATFORM_SUPPORTS_MX_GEMM, mx_skip_msg)
    @parametrize("test_case_name", [
        "a_eye_b_eye",
        "a_ones_b_ones",
        "a_ones_modified_b_ones",
        "a_ones_b_ones_modified",
        "a_scale_modified_b_ones",
        "a_ones_b_scale_modified",
        "data_random_scales_one",
        "data_random_scales_from_data",
    ])
    @parametrize("fast_accum", [False, True])
    @parametrize("mkn", [
        # Nice shapes
        (128, 128, 128),
        (256, 256, 256),
        (128, 256, 512),
        (256, 512, 128),
        (512, 128, 256),

        # Non block multiples
        (65, 96, 112),
        (197, 224, 272),
        # K not multiple of 32 (skipped for fp4)
        (197, 240, 272),

        # Very unbalanced
        (1023, 64, 48),
        (31, 1024, 64),
        (45, 96, 1024),

        # Mixed large and small
        (2, 1024, 128),
        (127, 96, 1024),
        (1025, 128, 96)
    ], name_fn=lambda mkn: f"{mkn[0]}_{mkn[1]}_{mkn[2]}")
    @parametrize("recipe", ["mxfp8", "nvfp4"])
    def test_blockwise_mxfp8_nvfp4_numerics(self, test_case_name, fast_accum, mkn, recipe) -> None:
        if recipe == "nvfp4" and fast_accum:
            return unittest.skip("fast_accum not supported in nvfp4 cublas gemm, skipping")

        device = "cuda"
        M, K, N = mkn
        if recipe == "nvfp4" and K % 32 != 0:
            return unittest.skip("K must be divisible by 32 for nvfp4 cublas gemm, skipping")

        BLOCK_SIZE = 16 if recipe == "nvfp4" else 32
        require_exact_match = True
        approx_match_sqnr_target = 22.0

        if test_case_name == "a_eye_b_eye":
            if not ((M == K) and (M == N)):
                return unittest.skip("this test is only defined for M == K == N, skipping")
            A_ref = torch.eye(M, device=device, dtype=torch.bfloat16)
            B_ref = torch.eye(M, device=device, dtype=torch.bfloat16)

            if recipe == "mxfp8":
                A = A_ref.to(torch.float8_e4m3fn)
                B = B_ref.to(torch.float8_e4m3fn)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
                B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
            else:  # nvfp4
                A = _bfloat16_to_float4_e2m1fn_x2(A_ref)
                B = _bfloat16_to_float4_e2m1fn_x2(B_ref)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e4m3fn)
                B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e4m3fn)

        elif test_case_name == "a_ones_b_ones":
            A_ref = torch.ones(M, K, device=device, dtype=torch.bfloat16)
            B_ref = torch.ones(N, K, device=device, dtype=torch.bfloat16)

            if recipe == "mxfp8":
                A = A_ref.to(torch.float8_e4m3fn)
                B = B_ref.to(torch.float8_e4m3fn)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
                B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
            else:  # nvfp4
                A = _bfloat16_to_float4_e2m1fn_x2(A_ref)
                B = _bfloat16_to_float4_e2m1fn_x2(B_ref)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e4m3fn)
                B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e4m3fn)

        elif test_case_name == "a_ones_modified_b_ones":
            A_ref = torch.ones(M, K, device=device, dtype=torch.bfloat16)
            B_ref = torch.ones(N, K, device=device, dtype=torch.bfloat16)
            A_ref[1][0:BLOCK_SIZE] = 2

            if recipe == "mxfp8":
                A = A_ref.to(torch.float8_e4m3fn)
                B = B_ref.to(torch.float8_e4m3fn)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
                B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
            else:  # nvfp4
                A = _bfloat16_to_float4_e2m1fn_x2(A_ref)
                B = _bfloat16_to_float4_e2m1fn_x2(B_ref)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e4m3fn)
                B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e4m3fn)

        elif test_case_name == "a_ones_b_ones_modified":
            A_ref = torch.ones(M, K, device=device, dtype=torch.bfloat16)
            B_ref = torch.ones(N, K, device=device, dtype=torch.bfloat16)
            B_ref[1][0:BLOCK_SIZE] = 2

            if recipe == "mxfp8":
                A = A_ref.to(torch.float8_e4m3fn)
                B = B_ref.to(torch.float8_e4m3fn)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
                B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
            else:  # nvfp4
                A = _bfloat16_to_float4_e2m1fn_x2(A_ref)
                B = _bfloat16_to_float4_e2m1fn_x2(B_ref)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e4m3fn)
                B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e4m3fn)

        elif test_case_name == "a_scale_modified_b_ones":
            A_ref = torch.ones(M, K, device=device, dtype=torch.bfloat16)
            B_ref = torch.ones(N, K, device=device, dtype=torch.bfloat16)

            if recipe == "mxfp8":
                A = A_ref.to(torch.float8_e4m3fn)
                B = B_ref.to(torch.float8_e4m3fn)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
                B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
                A_ref[1][0:BLOCK_SIZE] = 4
                A[1][0:BLOCK_SIZE] = 2
                A_scale[1][0] = 2
            else:  # nvfp4
                A = _bfloat16_to_float4_e2m1fn_x2(A_ref)
                B = _bfloat16_to_float4_e2m1fn_x2(B_ref)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e4m3fn)
                B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e4m3fn)
                A_ref[1][0:BLOCK_SIZE] = 4
                A.view(torch.uint8)[1][0:(BLOCK_SIZE // 2)] = 0b01000100
                A_scale[1][0] = 2

        elif test_case_name == "a_ones_b_scale_modified":
            A_ref = torch.ones(M, K, device=device, dtype=torch.bfloat16)
            B_ref = torch.ones(N, K, device=device, dtype=torch.bfloat16)

            if recipe == "mxfp8":
                A = A_ref.to(torch.float8_e4m3fn)
                B = B_ref.to(torch.float8_e4m3fn)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
                B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
                B_ref[1][0:BLOCK_SIZE] = 4
                B[1][0:BLOCK_SIZE] = 2
                B_scale[1][0] = 2
            else:  # nvfp4
                A = _bfloat16_to_float4_e2m1fn_x2(A_ref)
                B = _bfloat16_to_float4_e2m1fn_x2(B_ref)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e4m3fn)
                B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e4m3fn)
                B_ref[1][0:BLOCK_SIZE] = 4
                B.view(torch.uint8)[1][0:(BLOCK_SIZE // 2)] = 0b01000100
                B_scale[1][0] = 2

        elif test_case_name == "data_random_scales_one":
            require_exact_match = False

            if recipe == "mxfp8":
                # scales all-ones, element data random while being exactly representable in float8_e4m3fn
                # generate integers in [0, 255] and interpret as float8_e4m3fn
                A_ref = torch.randint(0, 255, (M, K), device=device, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.bfloat16)
                B_ref = torch.randint(0, 255, (N, K), device=device, dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.bfloat16)
                # modification: don't allow NaN values
                A_ref[torch.isnan(A_ref)] = 0
                B_ref[torch.isnan(B_ref)] = 0
                A = A_ref.to(torch.float8_e4m3fn)
                B = B_ref.to(torch.float8_e4m3fn)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
                B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
            else:  # nvfp4
                # scales all-ones, element data random while being exactly representable in float4_e2m1fn_x2
                # generate integers in [0, 16] and cast to bfloat16
                A_ref = _floatx_unpacked_to_f32(
                    torch.randint(0, 16, (M, K), device=device, dtype=torch.uint8),
                    FP4_EBITS,
                    FP4_MBITS
                ).bfloat16()
                B_ref = _floatx_unpacked_to_f32(
                    torch.randint(0, 16, (N, K), device=device, dtype=torch.uint8),
                    FP4_EBITS,
                    FP4_MBITS
                ).bfloat16()
                A = _bfloat16_to_float4_e2m1fn_x2(A_ref)
                B = _bfloat16_to_float4_e2m1fn_x2(B_ref)
                A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e4m3fn)
                B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e4m3fn)

        elif test_case_name == "data_random_scales_from_data":
            if not K % BLOCK_SIZE == 0:
                return unittest.skip(f"this test is only defined for K a multiple of {BLOCK_SIZE}, skipping")
            require_exact_match = False
            # random data, scales from data
            A_ref = torch.randn((M, K), device=device, dtype=torch.bfloat16) * 1000
            B_ref = torch.randn((N, K), device=device, dtype=torch.bfloat16) * 1000

            if recipe == "mxfp8":
                # Calculate scales based on the inputs
                A_scale = data_to_mx_scale(A_ref, BLOCK_SIZE)
                B_scale = data_to_mx_scale(B_ref, BLOCK_SIZE)
                max_val = F8E4M3_MAX_VAL
                min_val = -1 * max_val
                A = (A_ref.reshape(-1, BLOCK_SIZE) / A_scale.reshape(M * ceil_div(K, BLOCK_SIZE), 1).float()).reshape(M, K)
                A = A.clamp(min=min_val, max=max_val).to(torch.float8_e4m3fn)
                B = (B_ref.reshape(-1, BLOCK_SIZE) / B_scale.reshape(N * ceil_div(K, BLOCK_SIZE), 1).float()).reshape(N, K)
                B = B.clamp(min=min_val, max=max_val).to(torch.float8_e4m3fn)
            else:  # nvfp4
                A_scale = data_to_nvfp4_scale(A_ref, BLOCK_SIZE)
                B_scale = data_to_nvfp4_scale(B_ref, BLOCK_SIZE)
                max_val = FP4_MAX_VAL
                min_val = -1 * max_val

                A = (A_ref.reshape(-1, BLOCK_SIZE) / A_scale.reshape(M * ceil_div(K, BLOCK_SIZE), 1).bfloat16()).reshape(M, K)
                A = A.clamp(min=min_val, max=max_val)
                A = _bfloat16_to_float4_e2m1fn_x2(A)
                B = (B_ref.reshape(-1, BLOCK_SIZE) / B_scale.reshape(N * ceil_div(K, BLOCK_SIZE), 1).bfloat16()).reshape(N, K)
                B = B.clamp(min=min_val, max=max_val)
                B = _bfloat16_to_float4_e2m1fn_x2(B)

                approx_match_sqnr_target = 15.8

        C_ref = A_ref @ B_ref.t()

        # convert to swizzled format
        A_scale = to_blocked(A_scale)
        B_scale = to_blocked(B_scale)

        C = torch._scaled_mm(
            A,
            B.t(),
            A_scale,
            B_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=fast_accum,
        )

        if require_exact_match:
            torch.testing.assert_close(C, C_ref, atol=0, rtol=0)
        else:
            sqnr = compute_error(C_ref, C)
            assert sqnr.item() > approx_match_sqnr_target

    @unittest.skipIf(not PLATFORM_SUPPORTS_MX_GEMM or IS_WINDOWS, mx_skip_msg)
    @parametrize("recipe", ["mxfp8", "nvfp4"])
    def test_blockwise_mxfp8_nvfp4_error_messages(self, device, recipe) -> None:
        M, K, N = (1024, 512, 2048)
        BLOCK_SIZE_K = 16 if recipe == "nvfp4" else 32
        BLOCK_SIZE_MN = 128
        fill_value = 0.5
        scale_dtype = torch.float8_e4m3fn if recipe == "nvfp4" else torch.float8_e8m0fnu

        x = torch.full((M, K), fill_value, device=device)
        y = torch.full((N, K), fill_value, device=device)

        if recipe == "mxfp8":
            x_lowp = x.to(e4m3_type)
            y_lowp = y.to(e4m3_type).t()
        else:  # nvfp4
            x_lowp = _bfloat16_to_float4_e2m1fn_x2(x.bfloat16())
            y_lowp = _bfloat16_to_float4_e2m1fn_x2(y.bfloat16()).t()

        num_k_blocks = ceil_div(K, BLOCK_SIZE_K)
        padded_num_k_blocks = ceil_div(num_k_blocks, 4) * 4
        expected_a_size = BLOCK_SIZE_MN * ceil_div(M, BLOCK_SIZE_MN) * padded_num_k_blocks
        expected_b_size = BLOCK_SIZE_MN * ceil_div(N, BLOCK_SIZE_MN) * padded_num_k_blocks

        # Test wrong scale tensor size for scale_a with correct dtype
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape(
                f"For BlockWise scaling: Expected scale_a size to be {expected_a_size} "
                f"but got {expected_a_size - 1}"
            ),
        ):
            incorrect_size_a = torch.ones(expected_a_size - 1, device=device, dtype=scale_dtype)
            correct_size_b = torch.ones(expected_b_size, device=device, dtype=scale_dtype)
            torch._scaled_mm(
                x_lowp,
                y_lowp,
                scale_a=incorrect_size_a,
                scale_b=correct_size_b,
                out_dtype=torch.bfloat16,
            )

        # Test wrong scale tensor size for scale_b with correct dtype
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape(
                f"For BlockWise scaling: Expected scale_b size to be {expected_b_size} "
                f"but got {expected_b_size + 1}"
            ),
        ):
            correct_size_a = torch.ones(expected_a_size, device=device, dtype=scale_dtype)
            incorrect_size_b = torch.ones(expected_b_size + 1, device=device, dtype=scale_dtype)
            torch._scaled_mm(
                x_lowp,
                y_lowp,
                scale_a=correct_size_a,
                scale_b=incorrect_size_b,
                out_dtype=torch.bfloat16,
            )

        # Test non-contiguous scale tensors with correct dtype
        with self.assertRaisesRegex(
            RuntimeError,
            re.escape(
                "For BlockWise scaling: Both scale_a and scale_b must be contiguous"
            ),
        ):
            non_contiguous_a = torch.ones(expected_a_size * 2, device=device, dtype=scale_dtype)[::2]
            contiguous_b = torch.ones(expected_b_size, device=device, dtype=scale_dtype)
            torch._scaled_mm(
                x_lowp,
                y_lowp,
                scale_a=non_contiguous_a,
                scale_b=contiguous_b,
                out_dtype=torch.bfloat16,
            )

    def scaled_grouped_mm_helper(self, alist, blist, ascalelist, bscalelist, outlist, use_fast_accum):
        for a, b, ascale, bscale, out in zip(alist, blist, ascalelist, bscalelist, outlist):
            out_ref = torch._scaled_mm(a, b.t(), ascale.view(-1, 1), bscale.view(1, -1),
                                       out_dtype=torch.bfloat16, use_fast_accum=use_fast_accum)
            self.assertEqual(out, out_ref, atol=5e-2, rtol=5e-4)

    # Testing only _scaled_grouped_mm() with multiple shapes, as
    # _scaled_mm() already has more combinations of parameters than
    # _scaled_grouped_mm(), for supporting more than one inputs layout
    # combinations.

    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support CUTLASS")
    @xfailIfSM100OrLater
    @unittest.skipIf(not SM90OrLater, "Grouped gemm supported on SM90")
    @parametrize("fast_accum", [False, True])
    @parametrize("strided", [False, True])
    def test_scaled_grouped_gemm_2d_2d(self, fast_accum, strided):
        device = "cuda"
        m, n, k, n_groups = 16, 32, 64, 4
        a = torch.randn(m, k * n_groups + k * int(strided), device=device).to(torch.float8_e4m3fn)[:, :k * n_groups]
        b = torch.randn(n, k * n_groups + k * int(strided), device=device).to(torch.float8_e4m3fn)[:, :k * n_groups]
        scale_a = torch.rand(m * n_groups, device=device, dtype=torch.float32)
        scale_b = torch.rand(n * n_groups, device=device, dtype=torch.float32)
        offs = torch.arange(k, n_groups * k + 1, k, device=device, dtype=torch.int32)
        f = torch._scaled_grouped_mm
        out = f(a, b.t(), scale_a, scale_b, offs=offs,
                out_dtype=torch.bfloat16, use_fast_accum=fast_accum)
        offs_cpu = offs.cpu()
        alist, blist, ascalelist, bscalelist = [], [], [], []
        start = 0
        for i in range(n_groups):
            alist.append(a[:, start:offs_cpu[i]])
            blist.append(b[:, start:offs_cpu[i]])
            ascalelist.append(scale_a[i * m : (i + 1) * m])
            bscalelist.append(scale_b[i * n : (i + 1) * n])
            start = offs_cpu[i]
        self.scaled_grouped_mm_helper(alist, blist, ascalelist, bscalelist, out, fast_accum)


    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support CUTLASS")
    @xfailIfSM100OrLater
    @unittest.skipIf(not SM90OrLater, "Grouped gemm supported on SM90")
    @parametrize("fast_accum", [False, True])
    @parametrize("strided", [False, True])
    def test_scaled_grouped_gemm_2d_3d(self, fast_accum, strided):
        device = "cuda"
        m, n, k, n_groups = 16, 32, 64, 4
        s_int = int(strided)
        a = torch.randn(m * n_groups, k * (1 + s_int), device=device).to(torch.float8_e4m3fn)[:, :k]
        b = torch.randn(n_groups * (1 + s_int), n, k * (1 + s_int), device=device).to(torch.float8_e4m3fn)[::(1 + s_int), :, :k]
        self.assertTrue(a.is_contiguous() is not strided)
        self.assertTrue(b.is_contiguous() is not strided)
        for check_zero_size in (True, False):
            if check_zero_size and n_groups <= 1:
                continue

            offs = torch.arange(m, n_groups * m + 1, m, device="cuda", dtype=torch.int32)
            if check_zero_size:
                offs[0] = offs[1]
            scale_a = torch.rand(n_groups * m, device="cuda", dtype=torch.float32)
            scale_b = torch.rand(n_groups * n, device="cuda", dtype=torch.float32).view(n_groups, n)

            f = torch._scaled_grouped_mm
            out = f(a, b.transpose(-2, -1), scale_a, scale_b, offs=offs,
                    out_dtype=torch.bfloat16, use_fast_accum=fast_accum)

            offs_cpu = offs.cpu()
            alist, ascalelist, outlist = [], [], []
            start = 0
            for i in range(n_groups):
                alist.append(a[start:offs_cpu[i]])
                ascalelist.append(scale_a[start:offs_cpu[i]])
                outlist.append(out[start:offs_cpu[i]])
                start = offs_cpu[i]
                self.scaled_grouped_mm_helper(alist, b, ascalelist, scale_b, outlist, fast_accum)


    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support CUTLASS")
    @xfailIfSM100OrLater
    @unittest.skipIf(not SM90OrLater, "Grouped gemm supported on SM90")
    @parametrize("fast_accum", [False, True])
    @parametrize("strided", [False, True])
    def test_scaled_grouped_gemm_3d_3d(self, fast_accum, strided):
        device = "cuda"
        m, n, k, n_groups = 16, 32, 64, 4
        s_int = int(strided)
        a = torch.randn(n_groups * (1 + s_int), m, k * (1 + s_int), device=device).to(torch.float8_e4m3fn)[::(1 + s_int), :, :k]
        b = torch.randn(n_groups * (1 + s_int), n, k * (1 + s_int), device=device).to(torch.float8_e4m3fn)[::(1 + s_int), :, :k]
        self.assertTrue(a.is_contiguous() is not strided)
        self.assertTrue(b.is_contiguous() is not strided)
        scale_a = torch.rand(n_groups * m, device="cuda", dtype=torch.float32).view(n_groups, m)
        scale_b = torch.rand(n_groups * n, device="cuda", dtype=torch.float32).view(n_groups, n)

        f = torch._scaled_grouped_mm
        out = f(a, b.transpose(-2, -1), scale_a, scale_b,
                out_dtype=torch.bfloat16, use_fast_accum=fast_accum)

        self.scaled_grouped_mm_helper(a, b, scale_a, scale_b, out, fast_accum)


    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support CUTLASS")
    @xfailIfSM100OrLater
    @unittest.skipIf(not SM90OrLater, "Grouped gemm supported on SM90")
    @parametrize("fast_accum", [False, True])
    @parametrize("strided", [False, True])
    def test_scaled_grouped_gemm_3d_2d(self, fast_accum, strided):
        device = "cuda"
        m, n, k, n_groups = 16, 32, 64, 4
        s_int = int(strided)
        a = torch.randn(n_groups * (1 + s_int), m, k * (1 + s_int), device=device).to(torch.float8_e4m3fn)[::(1 + s_int), :, :k]
        b = torch.randn(n * n_groups, k * (1 + s_int), device=device).to(torch.float8_e4m3fn)[:, :k]
        self.assertTrue(a.is_contiguous() is not strided)
        self.assertTrue(b.is_contiguous() is not strided)
        scale_a = torch.rand(n_groups * m, device="cuda", dtype=torch.float32).view(n_groups, m)
        scale_b = torch.rand(n_groups * n, device="cuda", dtype=torch.float32)
        for check_zero_size in (True, False):
            if check_zero_size and n_groups <= 1:
                continue

            offs = torch.arange(n, n_groups * n + 1, n, device="cuda", dtype=torch.int32)
            if check_zero_size:
                offs[0] = offs[1]

            f = torch._scaled_grouped_mm
            out = f(a, b.transpose(-2, -1), scale_a, scale_b, offs=offs,
                    out_dtype=torch.bfloat16, use_fast_accum=fast_accum)
            offs_cpu = offs.cpu()
            blist, bscalelist, outlist = [], [], []
            start = 0
            for i in range(n_groups):
                blist.append(b[start:offs_cpu[i]])
                bscalelist.append(scale_b[start:offs_cpu[i]])
                outlist.append(out[:, start:offs_cpu[i]])
                start = offs_cpu[i]
                self.scaled_grouped_mm_helper(a, blist, scale_a, bscalelist, outlist, fast_accum)


    @unittest.skipIf(not PLATFORM_SUPPORTS_MX_GEMM, mx_skip_msg)
    def test_blockwise_mxfp8_compile(self) -> None:

        device = "cuda"
        M, K, N = 128, 128, 128
        BLOCK_SIZE = 32

        A_ref = torch.eye(M, device=device, dtype=torch.bfloat16)
        B_ref = torch.eye(M, device=device, dtype=torch.bfloat16)

        A = A_ref.to(torch.float8_e4m3fn)
        B = B_ref.to(torch.float8_e4m3fn)

        A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
        B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e8m0fnu)
        C_ref = A_ref @ B_ref.t()

        compiled_scaled_mm = torch.compile(torch._scaled_mm, backend="inductor")
        C = compiled_scaled_mm(
            A,
            B.t(),
            A_scale,
            B_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=False,
        )
        torch.testing.assert_close(C, C_ref, atol=0, rtol=0)

    @unittest.skipIf(not PLATFORM_SUPPORTS_MX_GEMM, mx_skip_msg)
    def test_blockwise_nvfp4_compile(self) -> None:

        device = "cuda"
        M, K, N = 128, 128, 128
        BLOCK_SIZE = 16

        A_ref = torch.eye(M, device=device, dtype=torch.bfloat16)
        B_ref = torch.eye(M, device=device, dtype=torch.bfloat16)

        A = _bfloat16_to_float4_e2m1fn_x2(A_ref)
        B = _bfloat16_to_float4_e2m1fn_x2(B_ref)

        A_scale = torch.full((M, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e4m3fn)
        B_scale = torch.full((N, ceil_div(K, BLOCK_SIZE)), 1.0, device=device, dtype=torch.float8_e4m3fn)
        C_ref = A_ref @ B_ref.t()

        compiled_scaled_mm = torch.compile(torch._scaled_mm, backend="inductor")
        # C = torch._scaled_mm(
        C = compiled_scaled_mm(
            A,
            B.t(),
            A_scale,
            B_scale,
            out_dtype=torch.bfloat16,
            use_fast_accum=False,
        )
        torch.testing.assert_close(C, C_ref, atol=0, rtol=0)


@unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support CUTLASS")
@unittest.skipIf(IS_WINDOWS, "Windows doesn't support CUTLASS extensions")
@unittest.skipIf(not _IS_SM8X, "mixed dtypes linear only supported on SM 8.x")
class TestMixedDtypesLinearCuda(TestCase):
    @dtypes(torch.float16, torch.bfloat16)
    def test_mixed_dtypes_linear(self, dtype: torch.dtype, device: str = "cuda"):
        version = _get_torch_cuda_version()
        if version < (11, 8):
            self.skipTest("_mixed_dtypes_linear only compiled for CUDA 11.8+")

        def run_test(
            batch_shape,
            m,
            n,
            k,
            add_bias,
            activation,
            dtype,
            dtypeq,
            device,
            rtol,
            atol,
        ):
            if not add_bias and activation != "none":
                return

            val_lo, val_hi = -1, 1
            valq_lo, valq_hi = -2, 2
            input = make_tensor(
                *batch_shape, m, k, low=val_lo, high=val_hi, dtype=dtype, device=device
            )
            weight = make_tensor(
                n, k, low=valq_lo, high=valq_hi, dtype=torch.int8, device=device
            )
            scale = make_tensor(
                (n,), low=val_lo, high=val_hi, dtype=input.dtype, device=device
            )
            bias = (
                make_tensor(
                    (n,), low=val_lo, high=val_hi, dtype=input.dtype, device=device
                )
                if add_bias
                else None
            )

            input_ref = input.reshape(-1, input.shape[-1])

            # First, test plain multiplication.
            weight_ref = weight.T.to(input.dtype) * scale.view(1, n)
            weightq = (
                pack_int4_to_int8(weight.T) if dtypeq == torch.quint4x2 else weight.T
            )
            output_ref = torch.mm(input_ref, weight_ref).reshape(*input.shape[:-1], n)
            output = torch.ops.aten._mixed_dtypes_linear(
                input,
                quantized_weight_reorder_for_mixed_dtypes_linear_cutlass(
                    weightq, dtypeq, transpose=False
                ),
                scale,
            )
            torch.testing.assert_close(output, output_ref, rtol=rtol, atol=atol)

            # Second, test the linear operator itself.
            weight_ref = weight.to(input.dtype) * scale.view(n, 1)
            weightq = pack_int4_to_int8(weight) if dtypeq == torch.quint4x2 else weight
            bias_ref = bias.view(1, n) if add_bias else None
            output_ref = torch.nn.functional.linear(
                input_ref, weight_ref, bias=bias_ref
            ).reshape(*input.shape[:-1], n)
            if activation == "relu":
                relu = torch.nn.ReLU()
                output_ref = relu(output_ref)
            elif activation == "silu":
                silu = torch.nn.SiLU()
                output_ref = silu(output_ref)
            output = torch.ops.aten._mixed_dtypes_linear(
                input,
                quantized_weight_reorder_for_mixed_dtypes_linear_cutlass(
                    weightq, dtypeq, transpose=True
                ),
                scale,
                bias=bias,
                activation=activation,
            )
            torch.testing.assert_close(output, output_ref, rtol=rtol, atol=atol)

        dtypeqs = [torch.int8, torch.quint4x2]
        batch_shapes = [[], [2], [2, 1]]
        shapes = [
            [8, 64, 64],
            [8, 64, 128],
            [8, 128, 64],
            [8, 128, 128],
            [8, 128, 192],
            [8, 128, 256],
            [8, 256, 128],
            [8, 256, 384],
            [8, 384, 256],
        ]
        activations = [None, "relu", "silu"]
        rtol, atol = 1e-3, 1e-3
        if dtype == torch.bfloat16:
            rtol, atol = 1e-2, 1e-3
        for dtypeq, batch_shape, (m, n, k), add_bias, activation in product(
            dtypeqs, batch_shapes, shapes, (False, True), activations
        ):
            run_test(
                batch_shape,
                m,
                n,
                k,
                add_bias,
                activation,
                dtype,
                dtypeq,
                device,
                rtol,
                atol,
            )

instantiate_device_type_tests(TestMatmulCuda, globals(), except_for="cpu")
instantiate_device_type_tests(TestMixedDtypesLinearCuda, globals(), except_for="cpu")
instantiate_device_type_tests(TestFP8Matmul, globals(), except_for="cpu")

if __name__ == '__main__':
    TestCase._default_dtype_check_enabled = True
    run_tests()
