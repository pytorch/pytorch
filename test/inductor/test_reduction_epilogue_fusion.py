# Owner(s): ["module: inductor"]
"""
Adversarial tests for the reduction epilogue fusion patch
(small-reduction-epilogue-fusion branch).

The patch suppresses split reductions when numel_hint >= 2 * num_sm,
enabling downstream broadcast pointwise ops (e.g. batch norm epilogue)
to fuse into the reduction kernel.

Tests verify:
1. Kernel count reduction for eligible shapes
2. Numerical correctness vs eager
3. No regression for cases that SHOULD still split (small numel < 2*num_sm)
"""

import torch
import torch._inductor.config as inductor_config
import torch._inductor.metrics as metrics
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import DeviceProperties
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


class TestReductionEpilogueFusion(TestCase):
    """Test that reduction epilogue fusion works correctly."""

    def setUp(self):
        super().setUp()
        metrics.reset()
        torch._dynamo.reset()

    def _get_num_sm(self):
        props = DeviceProperties.create(torch.device(GPU_TYPE))
        return props.multi_processor_count

    def _check_correctness(self, fn, args, tol=1e-3):
        """Check that compiled output matches eager within tolerance."""
        ref = fn(*args)
        torch._dynamo.reset()
        metrics.reset()
        act = torch.compile(fn)(*args)

        if isinstance(ref, (tuple, list)):
            for r, a in zip(ref, act):
                torch.testing.assert_close(r, a, rtol=tol, atol=tol)
        else:
            torch.testing.assert_close(ref, act, rtol=tol, atol=tol)
        return metrics.generated_kernel_count

    # ---------------------------------------------------------------
    # Tests for shapes where fusion SHOULD occur (C >= 2*num_sm)
    # ---------------------------------------------------------------

    def test_batch_norm_eval_fused(self):
        """BatchNorm eval: should always produce 1 kernel (no Welford needed)."""
        num_sm = self._get_num_sm()
        C = 2 * num_sm + 10  # Above threshold
        bn = torch.nn.BatchNorm2d(C).to(GPU_TYPE).eval()
        x = torch.randn(32, C, 16, 16, device=GPU_TYPE)
        kernels = self._check_correctness(bn, (x,))
        self.assertEqual(kernels, 1, f"Expected 1 kernel for eval BN with C={C}")

    def test_var_mean_epilogue_large_C(self):
        """var+mean followed by normalization with C >= 2*num_sm should fuse to 1 kernel."""
        num_sm = self._get_num_sm()
        C = 2 * num_sm + 10
        spatial = 1024

        def fn(x):
            mean = x.mean(dim=1)
            var = x.var(dim=1)
            return (x - mean.unsqueeze(1)) / (var.unsqueeze(1) + 1e-5).sqrt()

        x = torch.randn(C, spatial, device=GPU_TYPE)
        kernels = self._check_correctness(fn, (x,))
        self.assertEqual(kernels, 1, f"Expected 1 kernel for var_mean epilogue with C={C}")

    def test_var_mean_epilogue_at_threshold(self):
        """Test at exactly the threshold boundary (C = 2*num_sm)."""
        num_sm = self._get_num_sm()
        C = 2 * num_sm  # Exactly at threshold

        def fn(x):
            mean = x.mean(dim=1)
            var = x.var(dim=1)
            return (x - mean.unsqueeze(1)) / (var.unsqueeze(1) + 1e-5).sqrt()

        x = torch.randn(C, 8192, device=GPU_TYPE)
        kernels = self._check_correctness(fn, (x,))
        self.assertEqual(kernels, 1, f"Expected 1 kernel at threshold C={C}")

    def test_layer_norm_large_batch(self):
        """LayerNorm with large batch should fuse."""
        num_sm = self._get_num_sm()
        batch = 2 * num_sm + 50
        hidden = 768

        ln = torch.nn.LayerNorm(hidden).to(GPU_TYPE).eval()
        x = torch.randn(batch, hidden, device=GPU_TYPE)
        kernels = self._check_correctness(ln, (x,))
        self.assertEqual(kernels, 1, f"Expected 1 kernel for LN with batch={batch}")

    def test_various_spatial_dims(self):
        """Test batch norm pattern across various spatial dimensions."""
        num_sm = self._get_num_sm()
        C = 2 * num_sm + 10

        def fn(x):
            mean = x.mean(dim=1)
            var = x.var(dim=1)
            return (x - mean.unsqueeze(1)) / (var.unsqueeze(1) + 1e-5).sqrt()

        for spatial in [256, 1024, 4096, 16384, 65536]:
            x = torch.randn(C, spatial, device=GPU_TYPE)
            kernels = self._check_correctness(fn, (x,))
            self.assertEqual(
                kernels, 1,
                f"Expected 1 kernel for C={C}, spatial={spatial}"
            )

    # ---------------------------------------------------------------
    # Tests for shapes that SHOULD still split (C < 2*num_sm)
    # ---------------------------------------------------------------

    def test_small_C_still_splits(self):
        """With small C, split should still happen (more kernels expected)."""
        num_sm = self._get_num_sm()
        C = num_sm // 2  # Well below threshold

        def fn(x):
            mean = x.mean(dim=1)
            var = x.var(dim=1)
            return (x - mean.unsqueeze(1)) / (var.unsqueeze(1) + 1e-5).sqrt()

        x = torch.randn(C, 100000, device=GPU_TYPE)
        kernels = self._check_correctness(fn, (x,))
        # Should NOT be 1 kernel - split should still happen
        self.assertGreater(
            kernels, 1,
            f"Expected split (>1 kernels) for small C={C} < 2*num_sm={2*num_sm}"
        )

    def test_below_threshold_correctness(self):
        """Verify correctness for shapes just below the threshold."""
        num_sm = self._get_num_sm()
        C = 2 * num_sm - 1  # Just below threshold

        def fn(x):
            mean = x.mean(dim=1)
            var = x.var(dim=1)
            return (x - mean.unsqueeze(1)) / (var.unsqueeze(1) + 1e-5).sqrt()

        x = torch.randn(C, 50000, device=GPU_TYPE)
        kernels = self._check_correctness(fn, (x,))
        # Should still split (>1 kernels) since we're below threshold
        self.assertGreater(
            kernels, 1,
            f"Expected split for C={C} just below threshold 2*num_sm={2*num_sm}"
        )

    def test_tiny_numel_still_splits(self):
        """Very small output count should definitely still split."""
        def fn(x):
            mean = x.mean(dim=1)
            var = x.var(dim=1)
            return (x - mean.unsqueeze(1)) / (var.unsqueeze(1) + 1e-5).sqrt()

        # C=4, spatial=1M: definitely should split
        x = torch.randn(4, 1000000, device=GPU_TYPE)
        kernels = self._check_correctness(fn, (x,))
        self.assertGreater(
            kernels, 1,
            "Expected split for C=4 (very small numel)"
        )

    # ---------------------------------------------------------------
    # Correctness stress tests across diverse shapes
    # ---------------------------------------------------------------

    def test_correctness_sweep(self):
        """Sweep across many shapes to verify correctness."""
        num_sm = self._get_num_sm()

        def fn(x):
            mean = x.mean(dim=1)
            var = x.var(dim=1)
            return (x - mean.unsqueeze(1)) / (var.unsqueeze(1) + 1e-5).sqrt()

        shapes = [
            (num_sm * 4, 512),       # Above threshold, small spatial
            (num_sm * 4, 8192),      # Above threshold, medium spatial
            (num_sm * 4, 65536),     # Above threshold, large spatial
            (num_sm // 2, 8192),     # Below threshold
            (num_sm * 2, 1024),      # At threshold
            (1, 1000000),            # Single output, large reduction
            (num_sm * 8, 256),       # Way above threshold, small spatial
        ]

        for C, spatial in shapes:
            x = torch.randn(C, spatial, device=GPU_TYPE)
            self._check_correctness(fn, (x,), tol=1e-2)

    def test_batch_norm_training_correctness(self):
        """BatchNorm training correctness across shapes."""
        num_sm = self._get_num_sm()

        for C in [num_sm // 2, num_sm * 2, num_sm * 4]:
            bn = torch.nn.BatchNorm2d(C).to(GPU_TYPE).train()
            x = torch.randn(8, C, 16, 16, device=GPU_TYPE)
            self._check_correctness(bn, (x,), tol=1e-2)

    def test_welford_reduction_correctness(self):
        """Direct Welford (var_mean) test across shapes."""
        num_sm = self._get_num_sm()

        for C in [num_sm * 3, num_sm * 2, num_sm * 6]:
            x = torch.randn(C, 10000, device=GPU_TYPE)

            def fn(x):
                return torch.var_mean(x, dim=1)

            self._check_correctness(fn, (x,), tol=1e-2)


    # ---------------------------------------------------------------
    # Test cooperative reduction extension (choices.py change)
    # ---------------------------------------------------------------

    @inductor_config.patch(
        {
            "triton.cooperative_reductions": True,
            "triton.force_cooperative_reductions": False,
        }
    )
    def test_cooperative_reduction_no_launch_error(self):
        """
        The choices.py change extends cooperative reduction to larger xhint.
        Verify it does not produce 'too many blocks in cooperative launch'.
        """
        num_sm = self._get_num_sm()

        def fn(x):
            mean = x.mean(dim=1)
            var = x.var(dim=1)
            return (x - mean.unsqueeze(1)) / (var.unsqueeze(1) + 1e-5).sqrt()

        # Shapes that the patch now makes eligible for cooperative reduction
        # These should NOT exceed cooperative launch grid limits
        for C in [num_sm * 4, num_sm * 6, num_sm * 8]:
            for spatial in [8192, 16384]:
                x = torch.randn(C, spatial, device=GPU_TYPE)
                try:
                    kernels = self._check_correctness(fn, (x,), tol=1e-2)
                except RuntimeError as e:
                    if "too many blocks" in str(e):
                        self.fail(
                            f"Cooperative reduction failed with too many blocks "
                            f"for C={C}, spatial={spatial}: {e}"
                        )
                    raise


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
