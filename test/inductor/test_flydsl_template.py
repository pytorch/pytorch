# Owner(s): ["module: inductor"]
import unittest
from dataclasses import asdict
from unittest import mock

import torch

from torch._inductor.test_case import TestCase


try:
    import flydsl  # noqa: F401

    HAS_FLYDSL = True
except ImportError:
    HAS_FLYDSL = False

if HAS_FLYDSL:
    from torch._inductor.codegen.flydsl import flydsl_utils
    from torch._inductor.codegen.flydsl.flydsl_kernel import FlyDSLTemplateKernel


class TestFlyDSLTemplate(TestCase):
    def test_gen_imports(self):
        if not HAS_FLYDSL:
            self.skipTest("requires flydsl")

        kernel = FlyDSLTemplateKernel(
            kernel_name="test_kernel",
            input_nodes=[],
            output_node=None,
        )

        imports = kernel.gen_imports()

        self.assertIn("import torch", imports)
        self.assertIn("import flydsl.compiler as flyc", imports)
        self.assertIn("import flydsl.expr as fx", imports)

    @unittest.skipUnless(HAS_FLYDSL, "requires flydsl")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
    )
    def test_flydsl_gemm_transposed_rhs_e2e(self):
        from torch._inductor.utils import run_and_get_code

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        def fn(a, b):
            return torch.mm(a, b.t())

        cases = [
            (32, 128, 128),
            (32, 256, 128),
            (48, 96, 96),
        ]
        for dtype in (torch.bfloat16,):
            for m, n, k in cases:
                with self.subTest(dtype=dtype, m=m, n=n, k=k):
                    a = torch.randn(m, k, device="cuda", dtype=dtype)
                    b = torch.randn(n, k, device="cuda", dtype=dtype)

                    compiled_fn = torch.compile(fn, backend="inductor")
                    result, (code,) = run_and_get_code(compiled_fn, a, b)

                    self.assertIn("async_compile.flydsl", code)
                    self.assertIn("_flydsl_mm", code)
                    self.assertIn("TILE_M: fx.Constexpr", code)
                    self.assertIn("STAGES: fx.Constexpr", code)
                    self.assertIn("BLOCK_N_WARPS: fx.Constexpr", code)
                    self.assertIn("BLOCK_K_WARPS: fx.Constexpr", code)
                    self.assertTrue(
                        torch.allclose(result, fn(a, b), atol=3e-2, rtol=3e-2)
                    )

    @unittest.skipUnless(HAS_FLYDSL, "requires flydsl")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA/ROCm not available")
    @unittest.skipIf(torch.version.hip is None, "requires ROCm")
    @torch._inductor.config.patch(
        max_autotune_gemm=True,
        max_autotune_gemm_backends="FLYDSL",
        max_autotune_gemm_search_space="EXHAUSTIVE",
        flydsl_enable_autotuning=True,
    )
    def test_flydsl_autotune_transposed_rhs_uses_view_tensor(self):
        from torch._inductor.template_heuristics import flydsl as flydsl_heuristics
        from torch._inductor.utils import run_and_get_code

        if not flydsl_utils.runtime_available():
            self.skipTest("FlyDSL runtime unavailable")

        def fn(a, b):
            return torch.mm(a, b.t())

        configs = [
            asdict(config)
            for config in flydsl_heuristics.get_exhaustive_gemm_configs()[:2]
        ]
        a = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)

        with mock.patch.object(
            flydsl_heuristics, "get_gemm_configs", return_value=configs
        ):
            compiled_fn = torch.compile(fn, backend="inductor")
            result, (code,) = run_and_get_code(compiled_fn, a, b)

        self.assertIn("_flydsl_mm", code)
        self.assertTrue(torch.allclose(result, fn(a, b), atol=3e-2, rtol=3e-2))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
