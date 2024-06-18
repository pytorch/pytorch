# Owner(s): ["module: inductor"]

import sys
import unittest
from unittest import mock

import torch
from torch._inductor.runtime.hints import TRITON_MAX_BLOCK

from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU

try:
    import triton
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904

from torch._inductor import config
from torch._inductor.runtime.coordinate_descent_tuner import CoordescTuner

config.benchmark_kernel = True
config.coordinate_descent_tuning = True

orig_compare_config = CoordescTuner.compare_config


def mock_compare_config_prefer_larger_XBLOCK(
    self, func, candidate_config, best_config, best_timing
):
    """
    self is the CoordescTuner object
    """
    if "XBLOCK" in candidate_config.kwargs:
        assert "XBLOCK" in best_config.kwargs
        if candidate_config.kwargs["XBLOCK"] < best_config.kwargs["XBLOCK"]:
            func(candidate_config)  # run func so the launcher will be created
            return False, best_timing * 1.1
        elif candidate_config.kwargs["XBLOCK"] > best_config.kwargs["XBLOCK"]:
            func(candidate_config)
            return True, best_timing * 0.9

    return orig_compare_config(self, func, candidate_config, best_config, best_timing)


class TestCoordinateDescentTuner(TestCase):
    def test_abs_function(self):
        """
        The benchmark result is simply abs(XBLOCK - 15)
        """
        tuner = CoordescTuner()
        baseline_config = triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=1)

        def func(config):
            return abs(config.kwargs["XBLOCK"] - 15)

        best_config = tuner.autotune(func, baseline_config)
        self.assertTrue(best_config.kwargs.get("XBLOCK") == 16, str(best_config))

    def test_no_neighbors(self):
        """
        Test the case that there is no available neighbor values for a field.
        """

        # size hint for x being 1 limits the max XBLOCK we try to be 1
        tuner = CoordescTuner(size_hints=[1])
        baseline_config = triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=1)

        def func(config):
            return abs(config.kwargs["XBLOCK"] - 15)

        best_config = tuner.autotune(func, baseline_config)
        self.assertTrue(best_config.kwargs.get("XBLOCK") == 1, str(best_config))

    def test_get_neighbour_values(self):
        tuner = CoordescTuner()

        neighbours = tuner.get_neighbour_values("num_stages", 2, radius=2)
        self.assertEqual(set(neighbours), {1, 3, 4})
        neighbours = tuner.get_neighbour_values("num_warps", 2, radius=2)
        self.assertEqual(set(neighbours), {1, 4, 8})

    def test_persistent_reduction(self):
        def f(x):
            return x / x.sum(dim=-1, keepdim=True)

        with mock.patch.object(
            CoordescTuner, "compare_config", mock_compare_config_prefer_larger_XBLOCK
        ):
            x = torch.ones(2, 256).to(GPU_TYPE)
            expected = f(x)
            # the first call get correct result when cache miss. Don't know why yet
            _ = torch.compile(f)(x)
            actual = torch.compile(f)(x)
            self.assertTrue(
                torch.allclose(expected, actual, atol=1e-4, rtol=1e-4),
                f"Expected:\n{expected}\nActual:\n{actual}",
            )

    def test_value_too_large(self):
        # Simulate a reduction
        size_hints = [2**20, 2**20]

        tuner = CoordescTuner(size_hints=size_hints)

        max_block = TRITON_MAX_BLOCK
        self.assertFalse(tuner.value_too_large("XBLOCK", max_block["X"]))
        self.assertTrue(tuner.value_too_large("XBLOCK", max_block["X"] * 2))
        self.assertFalse(tuner.value_too_large("RBLOCK", max_block["R"]))
        self.assertTrue(tuner.value_too_large("RBLOCK", max_block["R"] * 2))


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU:
        run_tests()
