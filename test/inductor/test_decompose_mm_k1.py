# Owner(s): ["module: inductor"]

import torch
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.testing._internal.triton_utils import requires_gpu


@requires_gpu
@instantiate_parametrized_tests
class TestDecomposeMmK1(TestCase):
    """Tests for the K==1 pointwise decomposition in tuned_mm and tuned_addmm."""

    @parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
    @config.patch(
        {
            "max_autotune": True,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    def test_mm_k_1(self, dtype):
        def mm(x, y):
            return x @ y

        a = torch.randn((64, 1), device=GPU_TYPE, dtype=dtype)
        b = torch.randn((1, 64), device=GPU_TYPE, dtype=dtype)
        compiled_f = torch.compile(mm)

        out, code = run_and_get_code(compiled_f, a, b)
        torch.testing.assert_close(out, mm(a, b), atol=1e-2, rtol=1e-2)
        # K==1 should produce a pointwise kernel, not a GEMM
        FileCheck().check("triton_poi").run(code[0])

    @config.patch(
        {
            "max_autotune": True,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    def test_mm_k_not_1_no_decompose(self):
        """K > 1 should not trigger the pointwise decomposition."""

        def mm(x, y):
            return x @ y

        a = torch.randn((64, 16), device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn((16, 64), device=GPU_TYPE, dtype=torch.float32)
        compiled_f = torch.compile(mm)

        out, code = run_and_get_code(compiled_f, a, b)
        torch.testing.assert_close(out, mm(a, b), atol=1e-2, rtol=1e-2)

    @parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
    @config.patch(
        {
            "max_autotune": True,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    def test_addmm_k_1(self, dtype):
        """addmm with K==1 and default alpha=1, beta=1."""

        def addmm(bias, x, y):
            return torch.addmm(bias, x, y)

        a = torch.randn((64, 1), device=GPU_TYPE, dtype=dtype)
        b = torch.randn((1, 64), device=GPU_TYPE, dtype=dtype)
        bias = torch.randn((64,), device=GPU_TYPE, dtype=dtype)
        compiled_f = torch.compile(addmm)

        out, code = run_and_get_code(compiled_f, bias, a, b)
        torch.testing.assert_close(out, addmm(bias, a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("triton_poi").run(code[0])

    @config.patch(
        {
            "max_autotune": True,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    def test_addmm_k_1_alpha(self):
        """addmm with K==1 and alpha != 1."""

        def addmm(bias, x, y):
            return torch.addmm(bias, x, y, alpha=2.0)

        a = torch.randn((64, 1), device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn((1, 64), device=GPU_TYPE, dtype=torch.float32)
        bias = torch.randn((64,), device=GPU_TYPE, dtype=torch.float32)
        compiled_f = torch.compile(addmm)

        out, code = run_and_get_code(compiled_f, bias, a, b)
        torch.testing.assert_close(out, addmm(bias, a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("triton_poi").run(code[0])

    @config.patch(
        {
            "max_autotune": True,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    def test_addmm_k_1_beta_zero(self):
        """addmm with K==1 and beta=0 (bias should be ignored)."""

        def addmm(bias, x, y):
            return torch.addmm(bias, x, y, beta=0.0)

        a = torch.randn((64, 1), device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn((1, 64), device=GPU_TYPE, dtype=torch.float32)
        bias = torch.randn((64,), device=GPU_TYPE, dtype=torch.float32)
        compiled_f = torch.compile(addmm)

        out, code = run_and_get_code(compiled_f, bias, a, b)
        torch.testing.assert_close(out, addmm(bias, a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("triton_poi").run(code[0])

    @config.patch(
        {
            "max_autotune": True,
            "max_autotune_gemm_backends": "TRITON",
        }
    )
    def test_addmm_k_1_beta_non_unit(self):
        """addmm with K==1 and beta != 0 and beta != 1."""

        def addmm(bias, x, y):
            return torch.addmm(bias, x, y, alpha=2.0, beta=0.5)

        a = torch.randn((64, 1), device=GPU_TYPE, dtype=torch.float32)
        b = torch.randn((1, 64), device=GPU_TYPE, dtype=torch.float32)
        bias = torch.randn((64,), device=GPU_TYPE, dtype=torch.float32)
        compiled_f = torch.compile(addmm)

        out, code = run_and_get_code(compiled_f, bias, a, b)
        torch.testing.assert_close(out, addmm(bias, a, b), atol=1e-2, rtol=1e-2)
        FileCheck().check("triton_poi").run(code[0])


if __name__ == "__main__":
    run_tests()
