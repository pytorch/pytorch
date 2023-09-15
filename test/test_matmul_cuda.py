# Owner(s): ["module: linear algebra"]

import unittest
from functools import partial
from typing import Optional

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import SM53OrLater
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
    onlyCUDA,
    tol as xtol,
    toleranceOverride,
)

from torch.testing._internal.common_utils import (
    IS_ARM64,
    IS_JETSON,
    parametrize,
    run_tests,
    skipIfRocmVersionLessThan,
    TEST_WITH_ROCM,
    TestCase,
)

# Protects against includes accidentally setting the default dtype
assert torch.get_default_dtype() is torch.float32


@unittest.skipIf(IS_ARM64, "Issue with numpy version on arm")
class TestMatmulCuda(TestCase):
    def setUp(self):
        super(self.__class__, self).setUp()
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = True
        super(self.__class__, self).tearDown()

    @onlyCUDA
    @skipIfRocmVersionLessThan((5, 2))
    # imported 'tol' as 'xtol' to avoid aliasing in code above
    @toleranceOverride({torch.float16: xtol(atol=1e-1, rtol=1e-1),
                        torch.bfloat16: xtol(atol=1e-1, rtol=1e-1),
                        torch.float32: xtol(atol=1e-1, rtol=1e-1)})
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("size", [100, 1000, 10000])
    def test_cublas_addmm(self, size: int, dtype: torch.dtype):
        #
        # Check for catastrophic cuBLAS inaccuracy by measuring the deviation between
        # results from the CUDA invocation of torch.addmm and the CPU invocation
        # (which does not use CUDA backend).
        #
        # Get dims
        n, m, p = (size + 1, size, size + 2)
        # Disable reduced precision reductions in BFloat16 to bypass some kernels
        # which fail the threshold check
        orig = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        if dtype == torch.bfloat16 and torch.cuda.get_device_capability() >= (8, 6):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        # Make random tensors on CPU (seed set on common_utils.py import)
        # (Not using numpy because it does not support bfloat16)
        make_arg = partial(make_tensor, dtype=dtype, device="cpu")
        m_beta = make_arg(1)
        m_input = make_arg((n, p))
        m_1 = make_arg((n, m))
        m_2 = make_arg((m, p))
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
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = orig

    @onlyCUDA
    def test_cublas_addmm_alignment(self):
        dtype = torch.half
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
    @toleranceOverride({torch.float32: xtol(atol=1e-5, rtol=1e-5)})
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


@unittest.skipIf(TEST_WITH_ROCM, "FP8 is not supported on ROCM")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA not found")
class TestFP8MatmulCuda(TestCase):
    @unittest.skipIf(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0), "FP8 is only supported on H100+")
    def _test_tautological_mm(self, device: str = "cuda",
                              x_dtype: torch.dtype = torch.float8_e4m3fn,
                              y_dtype: torch.dtype = torch.float8_e4m3fn,
                              out_dtype: Optional[torch.dtype] = None,
                              size: int = 16) -> None:
        x_fp8 = torch.rand(size, size, device=device).to(x_dtype)
        y_fp8 = torch.eye(size, device=device, dtype=y_dtype).t()
        out_fp32 = torch.mm(x_fp8.to(torch.float), y_fp8.to(torch.float))
        (out_fp8, amax_fp8) = torch._scaled_mm(x_fp8, y_fp8, out_dtype=out_dtype)
        if out_dtype is not None:
            self.assertEqual(out_dtype, out_fp8.dtype)
        if out_dtype not in [torch.float16, torch.bfloat16, torch.float]:
            self.assertEqual(out_fp32.amax(), amax_fp8)
        self.assertEqual(out_fp32, out_fp8.to(torch.float))

    @unittest.skipIf(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0), "FP8 is only supported on H100+")
    def test_float8_basics(self, device) -> None:
        self._test_tautological_mm(device, torch.float8_e4m3fn, torch.float8_e4m3fn, size=16)
        self._test_tautological_mm(device, torch.float8_e4m3fn, torch.float8_e5m2, size=32)
        self._test_tautological_mm(device, torch.float8_e5m2, torch.float8_e4m3fn, size=48)
        # According to https://docs.nvidia.com/cuda/cublas/#id99 8F_E5M2 MM is unsupported
        with self.assertRaises(RuntimeError):
            self._test_tautological_mm(device, torch.float8_e5m2, torch.float8_e5m2)

    @unittest.skipIf(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0), "FP8 is only supported on H100+")
    def test_float8_out_dtype(self, device) -> None:
        self._test_tautological_mm(device, size=64, out_dtype=torch.float16)
        self._test_tautological_mm(device, size=96, out_dtype=torch.float32)
        self._test_tautological_mm(device, size=80, out_dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            self._test_tautological_mm(device, out_dtype=torch.float8_e5m2)

    @unittest.skipIf(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0), "FP8 is only supported on H100+")
    def test_float8_scale(self, device) -> None:
        size = (16, 16)
        x = torch.full(size, .5, device=device, dtype=torch.float8_e4m3fn)
        y = torch.full(size, .5, device=device, dtype=torch.float8_e5m2).t()
        scale_a = torch.tensor(1.5, device=device)
        scale_b = torch.tensor(0.66, device=device)
        out_fp8, amax_fp8 = torch._scaled_mm(x, y)
        self.assertEqual(out_fp8.to(torch.float), torch.full(size, 4., device=device))
        out_fp8_s, amax_fp8_s = torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b)
        self.assertEqual(out_fp8, out_fp8_s)

    @unittest.skipIf(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0), "FP8 is only supported on H100+")
    def test_float8_bias(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.rand((k, l), device=device).to(torch.float8_e4m3fn)
        y = torch.full((m, l), .25, device=device, dtype=torch.float8_e4m3fn).t()
        bias = torch.full((m,), 4.0, device=device, dtype=torch.half)
        out_fp8, amax_fp8 = torch._scaled_mm(x, y)
        outb_fp8, amaxb_fp8 = torch._scaled_mm(x, y, bias=bias)
        self.assertEqual((amaxb_fp8 - amax_fp8).item(), 4.0)

    @unittest.skipIf(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0), "FP8 is only supported on H100+")
    @parametrize("bias", [True, False])
    def test_non_divisible_leading_dim(self, device, bias: torch.bool) -> None:
        x = torch.rand((17, 16), device=device).to(torch.float8_e4m3fn)
        y = torch.rand((16, 16), device=device).to(torch.float8_e4m3fn).t()
        input_bias = None
        if bias:
            input_bias = torch.rand((16,), device=device).to(torch.half)
        out_fp8, amax_fp8 = torch._scaled_mm(x, y, bias=input_bias)

    @unittest.skipIf(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0), "FP8 is only supported on H100+")
    def test_float8_bias_relu_edgecase(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.full((k, l), 0.0, device=device).to(torch.float8_e4m3fn)
        y = torch.full((m, l), 1.0, device=device, dtype=torch.float8_e4m3fn).t()
        bias = torch.full((m,), -3.0, device=device, dtype=torch.half)
        outb_fp8, amaxb_fp8 = torch._scaled_mm(x, y, bias=bias)
        self.assertEqual(amaxb_fp8.item(), 3.0)

    @unittest.skipIf(not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0), "FP8 is only supported on H100+")
    def test_float32_output_errors_with_bias(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.rand((k, l), device=device).to(torch.float8_e4m3fn)
        y = torch.full((m, l), .25, device=device, dtype=torch.float8_e4m3fn).t()
        bias = torch.full((m,), 4.0, device=device, dtype=torch.bfloat16)
        self.assertRaisesRegex(
            RuntimeError,
            "Bias is not supported when out_dtype is set to Float32",
            lambda: torch._scaled_mm(x, y, bias=bias, out_dtype=torch.float32),
        )

    @unittest.skipIf(not torch.cuda.is_available() or torch.cuda.get_device_capability() >= (9, 0),
                     "This test is only for devices with compute capability < 9.0")
    def test_error_message_fp8_non_h100(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.rand((k, l), device=device).to(torch.float8_e4m3fn)
        y = torch.rand((m, l), device=device).to(torch.float8_e4m3fn).t()
        self.assertRaisesRegex(
            RuntimeError,
            r"torch\.\_scaled\_mm is only supported on devices with compute capability \>\= 9\.0",
            lambda: torch._scaled_mm(x, y, out_dtype=torch.float32),
        )



instantiate_device_type_tests(TestMatmulCuda, globals(), except_for="cpu")
instantiate_device_type_tests(TestFP8MatmulCuda, globals(), except_for="cpu")

if __name__ == '__main__':
    TestCase._default_dtype_check_enabled = True
    run_tests()
