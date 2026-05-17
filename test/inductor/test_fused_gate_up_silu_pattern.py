# Owner(s): ["module: inductor"]

import unittest

import torch
import torch.nn.functional as F
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code


class LlamaMLP(torch.nn.Module):
    """Minimal Llama MLP for pattern matching test."""

    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


@unittest.skipIf(not torch.xpu.is_available(), "XPU not available")
class TestPatternMatch(TestCase):
    def _test_pattern_fires(self, dtype):
        """Verify the pattern matcher replaces silu(mm)*mm with fused op."""
        model = LlamaMLP(512, 1384).to(dtype).to("xpu")
        x = torch.randn(32, 512, device="xpu", dtype=dtype)

        out, codes = run_and_get_code(torch.compile(model, backend="inductor"), x)

        # Verify the fused kernel appears in the generated code
        code = codes[0] if len(codes) == 1 else "\n".join(codes)
        self.assertIn(
            "fused_gate_up_silu",
            code,
            f"fused_gate_up_silu not found in generated inductor code for {dtype}",
        )

        # Verify correctness
        ref = model(x)
        rtol, atol = (2e-3, 0.5) if dtype == torch.float16 else (1.6e-2, 0.5)
        self.assertTrue(torch.allclose(ref, out, rtol=rtol, atol=atol))

    def test_pattern_fires_fp16(self):
        self._test_pattern_fires(torch.float16)

    def test_pattern_fires_bf16(self):
        self._test_pattern_fires(torch.bfloat16)

    def _test_fallback_on_cpu(self, dtype):
        """Pattern should NOT fire on CPU — output must still be correct."""
        model = LlamaMLP(512, 1384).to(dtype)
        x = torch.randn(32, 512, dtype=dtype)

        out, codes = run_and_get_code(torch.compile(model, backend="inductor"), x)

        code = codes[0] if len(codes) == 1 else "\n".join(codes)
        self.assertNotIn("fused_gate_up_silu", code)

        ref = model(x)
        rtol, atol = (2e-3, 0.5) if dtype == torch.float16 else (1.6e-2, 0.5)
        self.assertTrue(torch.allclose(ref, out, rtol=rtol, atol=atol))

    def test_fallback_on_cpu_fp16(self):
        self._test_fallback_on_cpu(torch.float16)

    def test_fallback_on_cpu_bf16(self):
        self._test_fallback_on_cpu(torch.bfloat16)

    def test_fallback_on_fp32(self):
        """Pattern should NOT fire on fp32 — output must still be correct."""
        model = LlamaMLP(512, 1384).to("xpu")
        x = torch.randn(32, 512, device="xpu")

        out, codes = run_and_get_code(torch.compile(model, backend="inductor"), x)

        code = codes[0] if len(codes) == 1 else "\n".join(codes)
        self.assertNotIn("fused_gate_up_silu", code)

        ref = model(x)
        self.assertTrue(torch.allclose(ref, out, rtol=1e-5, atol=1e-5))


if __name__ == "__main__":
    run_tests()
