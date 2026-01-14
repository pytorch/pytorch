# Owner(s): ["module: inductor"]
"""Tests for cvt_e8m0_rceil inductor prim with PTX lowering."""

from unittest import skipIf

import torch
from torch._inductor import inductor_prims
from torch._inductor.fx_passes.misc_patterns import _misc_patterns_init
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_cuda import SM100OrLater
from torch.testing._internal.common_utils import run_tests, skipIfRocm, skipIfXpu
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_gpu


class TestCvtE8M0Rceil(InductorTestCase):
    """Tests for cvt_e8m0_rceil prim and pattern matching."""

    @requires_gpu()
    @skipIfRocm
    @skipIfXpu(msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs")
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    def test_correctness(self):
        """Test correctness for powers of 2 and ceiling rounding cases."""

        def fn(inp):
            return inductor_prims.cvt_e8m0_rceil(inp)

        # Powers of 2: biased exponent = log2(value) + 127
        # Non-powers of 2: ceiling rounding (e.g., 1.5 -> exp 1 -> biased 128)
        inp = torch.tensor(
            [1.0, 2.0, 4.0, 0.5, 0.25, 1.5, 3.0, 5.0],
            device=GPU_TYPE,
            dtype=torch.float32,
        )

        eager_result = fn(inp)
        compiled_result = torch.compile(fn)(inp)
        self.assertEqual(compiled_result, eager_result)

        expected = torch.tensor(
            [127, 128, 129, 126, 125, 128, 129, 130],
            device=GPU_TYPE,
            dtype=torch.uint8,
        )
        self.assertEqual(eager_result, expected)

    @requires_gpu()
    @skipIfRocm
    @skipIfXpu(msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs")
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    def test_random_values(self):
        """Test with random values and various dtypes."""

        def fn(inp):
            return inductor_prims.cvt_e8m0_rceil(inp)

        for dtype in [torch.float32, torch.bfloat16, torch.float16]:
            inp = torch.rand(1024, device=GPU_TYPE, dtype=dtype) * 100 + 0.01
            eager_result = fn(inp)
            compiled_result = torch.compile(fn)(inp)
            self.assertEqual(compiled_result, eager_result)

    @requires_gpu()
    @skipIfRocm
    @skipIfXpu(msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs")
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    @skipIf(not SM100OrLater, "Pattern matching only enabled on SM100+")
    def test_pattern_match(self):
        """Test that patterns get matched and replaced."""
        _misc_patterns_init()

        # Test log2+ceil pattern (used by torchao)
        E8M0_BIAS = 127

        def fn_with_log2_pattern(inp):
            log2_val = torch.log2(inp)
            ceil_val = torch.ceil(log2_val)
            clamped = torch.clamp(ceil_val, min=-E8M0_BIAS, max=E8M0_BIAS)
            biased = clamped + E8M0_BIAS
            return biased.to(torch.uint8)

        inp = torch.tensor(
            [1.0, 2.0, 4.0, 3.0, 1.5], device=GPU_TYPE, dtype=torch.float32
        )

        eager_result = fn_with_log2_pattern(inp)
        compiled_result = torch.compile(fn_with_log2_pattern)(inp)
        self.assertEqual(compiled_result, eager_result)

    @requires_gpu()
    @skipIfRocm
    @skipIfXpu(msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs")
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    @skipIf(not SM100OrLater, "PTX instruction requires SM100+")
    def test_ptx_code_generation(self):
        """Test that PTX instruction appears in generated code on SM100+."""

        def fn(inp):
            return inductor_prims.cvt_e8m0_rceil(inp)

        inp = torch.rand(32, device=GPU_TYPE, dtype=torch.float32)
        compiled_fn = torch.compile(fn)
        _, code = run_and_get_code(compiled_fn, inp)

        code_str = "\n".join(code)
        self.assertIn("cvt.rp.satfinite.ue8m0x2.f32", code_str)


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
