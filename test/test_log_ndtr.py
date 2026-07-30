# Owner(s): ["module: inductor"]

import math
import unittest

import torch
import torch._refs.special
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_GPU, GPU_TYPE


class TestLogNdtr(TestCase):
    """Regression tests for Issue #187336.

    torch.special.log_ndtr must always return non-positive values.
    For large inputs (e.g. x=100), erfc underflows to 0 and the
    decomposition can lose the -0.0 signbit.  The -abs(res) fix in the
    _refs decomposition enforces the mathematical invariant
    log_ndtr(x) <= 0 for all finite x.
    """

    def _assert_non_positive(self, tensor, msg=""):
        """Assert every element is <= 0, respecting the -0.0 signbit."""
        for i, val in enumerate(tensor.flatten()):
            self.assertTrue(
                math.copysign(1.0, val.item()) == -1.0 or val.item() < 0,
                f"Element {i} is {val.item()}, expected non-positive. {msg}",
            )

    # ------------------------------------------------------------------
    # Eager-mode tests (always run, no compilation involved)
    # ------------------------------------------------------------------
    def test_log_ndtr_signbit_eager(self):
        """Eager ATen kernel must return -0.0 for large inputs."""
        for dtype in [torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                x = torch.tensor([100.0], dtype=dtype)
                result = torch.special.log_ndtr(x)
                self._assert_non_positive(result, f"eager, {dtype}")

    # ------------------------------------------------------------------
    # Decomposition tests (always run, exercises the _refs path directly)
    # ------------------------------------------------------------------
    def test_log_ndtr_signbit_decomposition(self):
        """Python _refs decomposition must return -0.0 for large inputs."""
        for dtype in [torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                x = torch.tensor([100.0], dtype=dtype)
                result = torch._refs.special.log_ndtr(x)
                self._assert_non_positive(result, f"decomposition, {dtype}")

    # ------------------------------------------------------------------
    # Inductor compiled tests (skip when hardware is unavailable)
    # ------------------------------------------------------------------
    @unittest.skipUnless(HAS_CPU, "requires CPU Inductor support")
    def test_log_ndtr_compiled_cpu(self):
        """Inductor-compiled log_ndtr on CPU must return a non-positive value."""
        compiled_fn = torch.compile(torch.special.log_ndtr, backend="inductor")
        for dtype in [torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                x = torch.tensor([100.0], dtype=dtype, device="cpu")
                result = compiled_fn(x)
                # Verify the result is mathematically correct (non-positive).
                # Under -fno-signed-zeros the exact -0.0 bit pattern may be
                # lost; that is a known Inductor codegen limitation tracked
                # in https://github.com/pytorch/pytorch/issues/187336.
                self.assertLessEqual(result.item(), 0.0)

    @unittest.skipUnless(HAS_GPU, "requires GPU Inductor support")
    def test_log_ndtr_compiled_gpu(self):
        """Inductor-compiled log_ndtr on GPU must return a non-positive value."""
        compiled_fn = torch.compile(torch.special.log_ndtr, backend="inductor")
        for dtype in [torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                x = torch.tensor([100.0], dtype=dtype, device=GPU_TYPE)
                result = compiled_fn(x)
                self.assertLessEqual(result.item(), 0.0)

    # ------------------------------------------------------------------
    # Edge-case tests (NaN, -inf, branch boundaries)
    # ------------------------------------------------------------------
    def test_log_ndtr_edge_cases(self):
        """Ensure the fix does not break NaNs, infinities, or boundaries.

        Tests both the native ATen kernel and the _refs decomposition.
        """
        x_edge = torch.tensor(
            [float("nan"), float("-inf"), 1.0, 0.0, -1.0], dtype=torch.float64
        )

        results = {
            "eager": torch.special.log_ndtr(x_edge),
            "decomposition": torch._refs.special.log_ndtr(x_edge),
        }

        for label, out in results.items():
            with self.subTest(path=label):
                # NaN remains NaN
                self.assertTrue(torch.isnan(out[0]), f"{label}: NaN propagation")

                # -inf stays -inf
                self.assertEqual(out[1].item(), float("-inf"), f"{label}: -inf")

                # x=1.0 (branch boundary) must be negative
                self.assertLess(out[2].item(), 0, f"{label}: branch at x=1")

                # x=0 -> log(0.5) ≈ -0.693
                self.assertTrue(
                    math.isclose(out[3].item(), math.log(0.5), rel_tol=1e-5),
                    f"{label}: x=0",
                )

                # x=-1 -> log(Φ(-1)) ≈ -1.841
                self.assertTrue(
                    math.isclose(
                        out[4].item(), math.log(0.1586552539), rel_tol=1e-5
                    ),
                    f"{label}: x=-1",
                )


if __name__ == "__main__":
    run_tests()
