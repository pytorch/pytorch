# Owner(s): ["module: linear algebra"]

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
from torch.testing._internal.common_cuda import SM53OrLater, _get_torch_cuda_version
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
    IS_WINDOWS,
    parametrize,
    run_tests,
    skipIfRocmVersionLessThan,
    TEST_WITH_ROCM,
    skipIfRocm,
    TestCase,
)

_IS_SM8X = False
if torch.cuda.is_available():
    _IS_SM8X = torch.cuda.get_device_capability(0)[0] == 8

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

    def cublas_addmm(self, size: int, dtype: torch.dtype, reduced_precision: bool = False):
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
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced_precision
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = reduced_precision
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
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = orig_bf16
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = orig_fp16

    @onlyCUDA
    @skipIfRocmVersionLessThan((5, 2))
    # imported 'tol' as 'xtol' to avoid aliasing in code above
    @toleranceOverride({torch.float16: xtol(atol=1e-1, rtol=1e-1),
                        torch.bfloat16: xtol(atol=1e-1, rtol=1e-1),
                        torch.float32: xtol(atol=1e-1, rtol=1e-1)})
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    @parametrize("size", [100, 1000, 10000])
    def test_cublas_addmm(self, size: int, dtype: torch.dtype):
        self.cublas_addmm(size, dtype, False)

    @onlyCUDA
    @skipIfRocmVersionLessThan((5, 2))
    # imported 'tol' as 'xtol' to avoid aliasing in code above
    @toleranceOverride({torch.float16: xtol(atol=7e-1, rtol=2e-1),
                        torch.bfloat16: xtol(atol=1e1, rtol=2e-1)})
    @dtypes(torch.float16, torch.bfloat16)
    @parametrize("size", [100, 1000, 10000])
    def test_cublas_addmm_reduced_precision(self, size: int, dtype: torch.dtype):
        self.cublas_addmm(size, dtype, True)

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



f8_msg = "FP8 is only supported on H100+ and sm_89 and MI300+ devices"

if torch.version.hip:
    e4m3_type = torch.float8_e4m3fnuz
    e5m2_type = torch.float8_e5m2fnuz
else:
    e4m3_type = torch.float8_e4m3fn
    e5m2_type = torch.float8_e5m2

def scaled_mm_supported_device():
    if torch.cuda.is_available():
        if torch.version.hip:
            return 'gfx94' in torch.cuda.get_device_properties(0).gcnArchName
        else:
            return torch.cuda.get_device_capability() >= (9, 0) or torch.cuda.get_device_capability() == (8, 9)
    return False


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not found")
class TestFP8MatmulCuda(TestCase):



    @unittest.skipIf(not scaled_mm_supported_device(), f8_msg)
    def _test_tautological_mm(self, device: str = "cuda",
                              x_dtype: torch.dtype = e4m3_type,
                              y_dtype: torch.dtype = e4m3_type,
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

    @unittest.skipIf(not scaled_mm_supported_device(), f8_msg)
    def test_float8_basics(self, device) -> None:
        self._test_tautological_mm(device, e4m3_type, e4m3_type, size=16)
        # hipblaslt does not yet support mixed e4m3_type input
        if torch.version.hip is None:
            self._test_tautological_mm(device, e4m3_type, e5m2_type, size=32)
            self._test_tautological_mm(device, e5m2_type, e4m3_type, size=48)
        # According to https://docs.nvidia.com/cuda/cublas/#id99 8F_E5M2 MM is unsupported
        with self.assertRaises(RuntimeError):
            self._test_tautological_mm(device, e5m2_type, e5m2_type)

        self._test_tautological_mm(device, size=64, out_dtype=torch.float16)
        self._test_tautological_mm(device, size=96, out_dtype=torch.float32)
        # hipblaslt does not yet support bfloat16 output
        if torch.version.hip is None:
            self._test_tautological_mm(device, size=80, out_dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            self._test_tautological_mm(device, out_dtype=e5m2_type)

    @unittest.skipIf(not scaled_mm_supported_device(), f8_msg)
    def test_float8_scale(self, device) -> None:
        size = (16, 16)
        x = torch.full(size, .5, device=device, dtype=e4m3_type)
        # hipblaslt does not yet support mixed e4m3_type input
        y_type = e4m3_type if torch.version.hip else e5m2_type
        y = torch.full(size, .5, device=device, dtype=y_type).t()
        scale_a = torch.tensor(1.5, device=device)
        scale_b = torch.tensor(0.66, device=device)
        out_fp8, amax_fp8 = torch._scaled_mm(x, y)
        self.assertEqual(out_fp8.to(torch.float), torch.full(size, 4., device=device))
        out_fp8_s, amax_fp8_s = torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b)
        self.assertEqual(out_fp8, out_fp8_s)

    @unittest.skipIf(not scaled_mm_supported_device(), f8_msg)
    def test_float8_bias(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.rand((k, l), device=device).to(e4m3_type)
        y = torch.full((m, l), .25, device=device, dtype=e4m3_type).t()
        bias = torch.full((m,), 4.0, device=device, dtype=torch.half)
        out_fp8, amax_fp8 = torch._scaled_mm(x, y)
        outb_fp8, amaxb_fp8 = torch._scaled_mm(x, y, bias=bias)
        # this fails on ROCm currently because hipblaslt doesn't have amax op
        if torch.version.hip is None:
            self.assertEqual((amaxb_fp8 - amax_fp8).item(), 4.0)

    @unittest.skipIf(not scaled_mm_supported_device(), f8_msg)
    @parametrize("bias", [True, False])
    def test_non_divisible_leading_dim(self, device, bias: torch.bool) -> None:
        x = torch.rand((17, 16), device=device).to(e4m3_type)
        y = torch.rand((16, 16), device=device).to(e4m3_type).t()
        input_bias = None
        if bias:
            input_bias = torch.rand((16,), device=device).to(torch.half)
        out_fp8, amax_fp8 = torch._scaled_mm(x, y, bias=input_bias)

    @unittest.skipIf(not scaled_mm_supported_device(), f8_msg)
    def test_float8_bias_relu_edgecase(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.full((k, l), 0.0, device=device).to(e4m3_type)
        y = torch.full((m, l), 1.0, device=device, dtype=e4m3_type).t()
        bias = torch.full((m,), -3.0, device=device, dtype=torch.half)
        outb_fp8, amaxb_fp8 = torch._scaled_mm(x, y, bias=bias)
        self.assertEqual(amaxb_fp8.item(), 3.0)

    @unittest.skipIf(not scaled_mm_supported_device(), f8_msg)
    def test_float32_output_errors_with_bias(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.rand((k, l), device=device).to(e4m3_type)
        y = torch.full((m, l), .25, device=device, dtype=e4m3_type).t()
        bias = torch.full((m,), 4.0, device=device, dtype=torch.bfloat16)
        self.assertRaisesRegex(
            RuntimeError,
            "Bias is not supported when out_dtype is set to Float32",
            lambda: torch._scaled_mm(x, y, bias=bias, out_dtype=torch.float32),
        )

    @unittest.skipIf(scaled_mm_supported_device(),
                     "This test is only for devices with compute capability < 8.9")
    def test_error_message_fp8_pre_sm89(self, device) -> None:
        (k, l, m) = (16, 48, 32)
        x = torch.rand((k, l), device=device).to(e4m3_type)
        y = torch.rand((m, l), device=device).to(e4m3_type).t()
        self.assertRaisesRegex(
            RuntimeError,
            r"torch\.\_scaled\_mm is only supported on CUDA devices with compute capability \>\= 9\.0 or 8\.9, or ROCm MI300\+",
            lambda: torch._scaled_mm(x, y, out_dtype=torch.float32),
        )

    @unittest.skipIf(not scaled_mm_supported_device(), f8_msg)
    def test_float8_scale_fast_accum(self, device) -> None:
        size = (16, 16)
        x = torch.full(size, .5, device=device, dtype=e4m3_type)
        # hipblaslt does not yet support mixed e4m3_type input
        y_type = e4m3_type if torch.version.hip else e5m2_type
        y = torch.full(size, .5, device=device, dtype=y_type).t()
        scale_a = torch.tensor(1.5, device=device)
        scale_b = torch.tensor(0.66, device=device)
        out_fp8, amax_fp8 = torch._scaled_mm(x, y, use_fast_accum=True)
        self.assertEqual(out_fp8.to(torch.float), torch.full(size, 4., device=device))
        out_fp8_s, amax_fp8_s = torch._scaled_mm(x, y, scale_a=scale_a, scale_b=scale_b, use_fast_accum=True)
        self.assertEqual(out_fp8, out_fp8_s)


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
instantiate_device_type_tests(TestFP8MatmulCuda, globals(), except_for="cpu")
instantiate_device_type_tests(TestMixedDtypesLinearCuda, globals(), except_for="cpu")

if __name__ == '__main__':
    TestCase._default_dtype_check_enabled = True
    run_tests()
