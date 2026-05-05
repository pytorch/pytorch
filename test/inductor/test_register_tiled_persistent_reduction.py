# Owner(s): ["module: inductor"]

"""Tests for register-tiled persistent reductions.

When a fused reduction+epilogue kernel shares source reads between the
reduction and epilogue, the reduction dimension can be split into tiles
so shared inputs stay in registers across both phases.  These tests
verify correctness, generated code shape, and fallback behaviour.
"""

import functools
import unittest

import torch
import torch._inductor.config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON


register_tiled_config = {
    "triton.register_tiled_persistent_reductions": True,
}


def expects_register_tiled(expected: bool):
    """Decorator marking whether a test expects register-tiled codegen."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapper._expects_register_tiled = expected
        return wrapper

    return decorator


@unittest.skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
@instantiate_parametrized_tests
class TestRegisterTiledPersistentReduction(TestCase):
    def _run_and_check(self, fn, args, *, atol=None, rtol=None):
        """Compile fn with register-tiled config, check correctness against eager.

        Also asserts register-tiled codegen based on the @expects_register_tiled decorator.
        """
        expected = fn(*args)
        compiled = torch.compile(fn)
        with inductor_config.patch(register_tiled_config):
            actual, code = run_and_get_code(compiled, *args)
        self.assertEqual(actual, expected, atol=atol, rtol=rtol)
        # Check register-tiled expectation from decorator
        test_method = getattr(self, self._testMethodName)
        expect = getattr(test_method, "_expects_register_tiled", None)
        self.assertIsNotNone(expect, "test must use @expects_register_tiled decorator")
        if expect:
            self.assertIn("tl.static_range(NUM_TILES)", code[0])
        else:
            self.assertNotIn("tl.static_range(NUM_TILES)", code[0])
        return code

    # ---- register-tiled tests (bf16/fp16 only) ----

    @expects_register_tiled(True)
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_sum_reduction(self, dtype):
        """Simple sum reduction with epilogue — should tile correctly."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        x = torch.randn(16, 8192, device=GPU_TYPE, dtype=dtype).abs()

        self._run_and_check(fn, (x,))

    @expects_register_tiled(True)
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_max_reduction(self, dtype):
        """max reduction with epilogue — should tile correctly."""

        def fn(x):
            return x - x.max(dim=-1, keepdim=True).values

        x = torch.randn(16, 8192, device=GPU_TYPE, dtype=dtype)

        self._run_and_check(fn, (x,))

    @expects_register_tiled(True)
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_rmsnorm(self, dtype):
        """RMSNorm-like pattern."""

        def fn(x, weight):
            variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + 1e-6)
            return x_normed * weight

        x = torch.randn(32, 8192, device=GPU_TYPE, dtype=dtype)
        weight = torch.randn(8192, device=GPU_TYPE, dtype=dtype)

        self._run_and_check(fn, (x, weight))

    @expects_register_tiled(True)
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_residual_rmsnorm(self, dtype):
        """Residual + RMSNorm fused pattern (common in transformers)."""

        def fn(x, residual, weight):
            h = x.to(torch.float32) + residual.to(torch.float32)
            variance = h.pow(2).mean(-1, keepdim=True)
            h_normed = h * torch.rsqrt(variance + 1e-6)
            return h_normed * weight.to(torch.float32)

        x = torch.randn(16, 8192, device=GPU_TYPE, dtype=dtype)
        residual = torch.randn(16, 8192, device=GPU_TYPE, dtype=dtype)
        weight = torch.randn(8192, device=GPU_TYPE, dtype=dtype)

        self._run_and_check(fn, (x, residual, weight))

    # ---- fallback tests ----

    def test_small_rnumel_stays_single_tile(self):
        """rnumel=1024 is below min_numel — should use standard persistent, no tiling."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True)

        x = torch.randn(16, 1024, device=GPU_TYPE)
        compiled = torch.compile(fn)
        with inductor_config.patch(register_tiled_config):
            _, code = run_and_get_code(compiled, x)

        self.assertIn("persistent_reduction", code[0])
        self.assertNotIn("tl.static_range", code[0])
        self.assertNotIn("persistent_reduction_num_tiles", code[0])

    def test_indivisible_rnumel_falls_back(self):
        """rnumel=5003 (prime) — not evenly divisible by any tile count, should fall back."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True)

        x = torch.randn(16, 5003, device=GPU_TYPE)
        compiled = torch.compile(fn)
        with inductor_config.patch(register_tiled_config):
            _, code = run_and_get_code(compiled, x)

        self.assertNotIn("tl.static_range", code[0])

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_many_shared_inputs_falls_back(self, dtype):
        """More than 2 shared inputs — falls back to avoid register pressure."""

        def fn(a, b, c, d):
            h = a.float() + b.float() + c.float() + d.float()
            s = h.sum(dim=-1, keepdim=True)
            return (h - s).to(dtype)

        a = torch.randn(16, 8192, device=GPU_TYPE, dtype=dtype)
        b = torch.randn(16, 8192, device=GPU_TYPE, dtype=dtype)
        c = torch.randn(16, 8192, device=GPU_TYPE, dtype=dtype)
        d = torch.randn(16, 8192, device=GPU_TYPE, dtype=dtype)

        compiled = torch.compile(fn)
        with inductor_config.patch(register_tiled_config):
            _, code = run_and_get_code(compiled, a, b, c, d)

        self.assertNotIn("tl.static_range(NUM_TILES)", code[0])

    @expects_register_tiled(False)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_var_mean_falls_back(self, dtype):
        """torch.var_mean uses welford reduction — unsupported, should fall back."""

        def fn(x):
            var, mean = torch.var_mean(x, dim=-1, keepdim=True)
            return x * mean + var

        x = torch.randn(16, 8192, device=GPU_TYPE, dtype=dtype)

        self._run_and_check(fn, (x,))

    @expects_register_tiled(False)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_layer_norm_falls_back(self, dtype):
        """nn.functional.layer_norm uses welford — should fall back correctly."""

        def fn(x, weight, bias):
            return torch.nn.functional.layer_norm(x, [8192], weight, bias)

        x = torch.randn(16, 8192, device=GPU_TYPE, dtype=dtype)
        weight = torch.randn(8192, device=GPU_TYPE, dtype=dtype)
        bias = torch.randn(8192, device=GPU_TYPE, dtype=dtype)

        self._run_and_check(fn, (x, weight, bias))

    @expects_register_tiled(False)
    @parametrize("dtype", [torch.float32, torch.bfloat16])
    def test_softmax(self, dtype):
        """softmax has two reductions (max + sum) — falls back."""

        def fn(x):
            return torch.nn.functional.softmax(x, dim=-1)

        x = torch.randn(16, 8192, device=GPU_TYPE, dtype=dtype)

        self._run_and_check(fn, (x,))

    # ---- metadata / gating tests ----

    def test_metadata_emitted(self):
        """Check that inductor_meta contains tile metadata."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True)

        x = torch.randn(16, 8192, device=GPU_TYPE, dtype=torch.bfloat16)
        compiled = torch.compile(fn)
        with inductor_config.patch(register_tiled_config):
            _, code = run_and_get_code(compiled, x)

        self.assertIn("persistent_reduction_num_tiles", code[0])
        self.assertIn("persistent_reduction_rnumel", code[0])

    def test_feature_disabled_by_default(self):
        """Without the config flag, register-tiled should not activate."""

        def fn(x):
            return x / x.sum(dim=-1, keepdim=True)

        x = torch.randn(16, 8192, device=GPU_TYPE)
        compiled = torch.compile(fn)
        _, code = run_and_get_code(compiled, x)
        self.assertNotIn("tl.static_range", code[0])
        self.assertNotIn("persistent_reduction_num_tiles", code[0])


if __name__ == "__main__":
    run_tests()
