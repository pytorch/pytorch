# Owner(s): ["module: inductor"]
"""Tests for cvt_e8m0_rceil inductor prim with PTX lowering on SM100+ (Blackwell)."""

import unittest

import torch
from torch._inductor import inductor_prims
from torch._inductor.fx_passes.misc_patterns import _misc_patterns_init
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_cuda import SM100OrLater
from torch.testing._internal.inductor_utils import HAS_GPU


@unittest.skipIf(not SM100OrLater, "Requires SM100+ (Blackwell) for PTX instruction")
class TestCvtE8M0Rceil(TestCase):
    """Tests for cvt_e8m0_rceil prim with PTX lowering on Blackwell."""

    def test_correctness(self):
        """Test correctness for various dtypes."""

        def fn(inp):
            return inductor_prims.cvt_e8m0_rceil(inp)

        for dtype in [torch.float32, torch.bfloat16, torch.float16]:
            inp = torch.cat(
                [
                    torch.arange(-1024, 0, device="cuda", dtype=dtype),
                    torch.arange(1, 1025, device="cuda", dtype=dtype),
                ]
            )
            eager_result = fn(inp)
            compiled_result = torch.compile(fn)(inp)
            self.assertEqual(compiled_result, eager_result)

    def test_pattern_match(self):
        """Test that the log2+ceil pattern gets matched and replaced."""
        _misc_patterns_init()

        E8M0_BIAS = 127

        def fn_with_log2_pattern(inp):
            log2_val = torch.log2(inp)
            ceil_val = torch.ceil(log2_val)
            clamped = torch.clamp(ceil_val, min=-E8M0_BIAS, max=E8M0_BIAS)
            biased = clamped + E8M0_BIAS
            return biased.to(torch.uint8)

        inp = torch.tensor(
            [1.0, 2.0, 4.0, 3.0, 1.5], device="cuda", dtype=torch.float32
        )

        eager_result = fn_with_log2_pattern(inp)
        compiled_result = torch.compile(fn_with_log2_pattern)(inp)
        self.assertEqual(compiled_result, eager_result)

    def test_ptx_code_generation(self):
        """Test that PTX instruction appears in generated code."""

        def fn(inp):
            return inductor_prims.cvt_e8m0_rceil(inp)

        inp = torch.rand(32, device="cuda", dtype=torch.float32)
        compiled_fn = torch.compile(fn)
        _, code = run_and_get_code(compiled_fn, inp)

        code_str = "\n".join(code)
        self.assertIn("cvt.rp.satfinite.ue8m0x2.f32", code_str)


if __name__ == "__main__":
    if HAS_GPU and SM100OrLater:
        run_tests()
