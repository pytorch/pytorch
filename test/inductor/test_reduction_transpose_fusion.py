# Owner(s): ["module: inductor"]
"""
Adversarial tests for reduction + transpose fusion via loop reordering.

Tests that torch.compile fuses x.permute(1,0).contiguous() and x.sum(dim=0)
into a single kernel (or fewer kernels) and produces correct results.
"""

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
import torch._inductor.metrics as metrics
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU

import unittest


@unittest.skipUnless(HAS_GPU, "requires GPU")
class ReductionTransposeFusionTest(TestCase):
    def setUp(self):
        super().setUp()
        metrics.reset()
        torch._dynamo.reset()

    def _check_pattern(self, x, max_kernels=1, rtol=1e-3, atol=1e-3):
        """
        Compile and run x.permute(1,0).contiguous(), x.sum(dim=0).
        Asserts kernel count and correctness.
        """

        @torch.compile
        def pattern(x):
            return x.permute(1, 0).contiguous(), x.sum(dim=0)

        metrics.reset()
        torch._dynamo.reset()
        out_t, out_s = pattern(x)
        kernel_count = metrics.generated_kernel_count

        # Correctness
        ref_t = x.permute(1, 0).contiguous()
        ref_s = x.sum(dim=0)
        torch.testing.assert_close(out_t, ref_t, rtol=0, atol=0)
        torch.testing.assert_close(out_s, ref_s, rtol=rtol, atol=atol)

        self.assertLessEqual(
            kernel_count,
            max_kernels,
            f"Expected at most {max_kernels} kernel(s), got {kernel_count}",
        )
        return kernel_count

    def test_large_shape_fusion(self):
        """Large [25216, 3072] should fuse into 1 kernel."""
        x = torch.randn(25216, 3072, device=GPU_TYPE)
        self._check_pattern(x, max_kernels=1)

    def test_medium_shape_fusion(self):
        """Medium [4096, 1024] should also fuse."""
        x = torch.randn(4096, 1024, device=GPU_TYPE)
        self._check_pattern(x, max_kernels=1)

    def test_small_tensor_no_fusion_overhead(self):
        """
        Small tensors [32, 32] -- fusion is still allowed (1 kernel) but
        we verify correctness is maintained.
        """
        x = torch.randn(32, 32, device=GPU_TYPE)
        # Even small tensors fuse -- we just check correctness
        self._check_pattern(x, max_kernels=2)  # allow up to 2

    def test_very_small_tensor(self):
        """[4, 4] -- trivial case, check correctness."""
        x = torch.randn(4, 4, device=GPU_TYPE)
        self._check_pattern(x, max_kernels=2)

    def test_non_square(self):
        """Non-square: tall and wide shapes.

        For tall tensors (M >> N), sum(dim=0) reduces over many elements with
        few outputs. The split suppression only fires when numel >= 2*num_sm,
        so tall tensors with small N may still split.

        For wide tensors (N >> M), there are many outputs so fusion is easy.
        """
        # Wide: many outputs (N=8192), small reduction (M=64) -- fuses easily
        x_wide = torch.randn(64, 8192, device=GPU_TYPE)
        self._check_pattern(x_wide, max_kernels=1)

        torch._dynamo.reset()
        metrics.reset()

        # Tall: few outputs (N=64), large reduction (M=8192) -- may split
        # Key: correctness still holds
        x_tall = torch.randn(8192, 64, device=GPU_TYPE)
        self._check_pattern(x_tall, max_kernels=4)

    def test_non_contiguous_input(self):
        """Non-contiguous input (sliced) should still produce correct results."""
        x_base = torch.randn(1000, 2048, device=GPU_TYPE)
        x = x_base[::2, :]  # stride(0) != shape[1]
        self._check_pattern(x, max_kernels=2)  # may not fuse, but must be correct

    def test_float16(self):
        """fp16 dtype correctness."""
        x = torch.randn(4096, 1024, device=GPU_TYPE, dtype=torch.float16)
        self._check_pattern(x, max_kernels=1, rtol=1e-3, atol=1e-2)

    def test_bfloat16(self):
        """bf16 dtype correctness."""
        x = torch.randn(4096, 1024, device=GPU_TYPE, dtype=torch.bfloat16)
        self._check_pattern(x, max_kernels=1, rtol=1e-2, atol=1e-1)

    def test_float64(self):
        """float64 correctness with tighter tolerance."""
        x = torch.randn(2048, 512, device=GPU_TYPE, dtype=torch.float64)
        self._check_pattern(x, max_kernels=2, rtol=1e-12, atol=1e-12)

    def test_correctness_various_shapes(self):
        """Sweep shapes to catch edge cases -- correctness is key."""
        shapes = [
            (128, 128),
            (256, 512),
            (512, 256),
            (1024, 3072),
            (3072, 1024),
            (7, 1024),   # prime number of rows
            (1024, 7),   # prime number of cols
            (1, 1024),   # single row
            (1024, 1),   # single col
        ]
        for M, N in shapes:
            with self.subTest(shape=(M, N)):
                torch._dynamo.reset()
                metrics.reset()
                x = torch.randn(M, N, device=GPU_TYPE)
                # Allow up to 4 kernels for edge cases (small dims may split)
                # The key assertion is correctness, not kernel count
                self._check_pattern(x, max_kernels=4)

    @inductor_config.patch("triton.cooperative_reductions", True)
    def test_cooperative_reductions_enabled(self):
        """With cooperative reductions enabled, should still fuse and be correct."""
        x = torch.randn(25216, 3072, device=GPU_TYPE)
        # Cooperative reduction may reorder accumulation, so allow some tolerance
        self._check_pattern(x, max_kernels=1, rtol=1e-3, atol=1e-3)

    def test_performance_not_regressed(self):
        """
        The fused kernel should not be slower than running separate kernels.
        This is a sanity check -- we allow 20% overhead tolerance for noise.
        """
        from triton.testing import do_bench

        x = torch.randn(8192, 2048, device=GPU_TYPE)

        @torch.compile
        def fused(x):
            return x.permute(1, 0).contiguous(), x.sum(dim=0)

        # Warmup
        fused(x)
        fused(x)
        fused_ms = do_bench(lambda: fused(x))

        # Baseline: separate ops
        def separate(x):
            t = x.permute(1, 0).contiguous()
            s = x.sum(dim=0)
            return t, s

        separate(x)
        separate_ms = do_bench(lambda: separate(x))

        # Fused should not be more than 1.2x slower than separate
        self.assertLess(
            fused_ms,
            separate_ms * 1.2,
            f"Fused ({fused_ms:.3f}ms) is more than 20% slower than "
            f"separate ({separate_ms:.3f}ms)",
        )


if __name__ == "__main__":
    run_tests()
