# Owner(s): ["module: inductor"]

import sys
import unittest

from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA

try:
    import triton
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")

from torch._inductor.coordinate_descent_tuner import CoordescTuner


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


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA:
        run_tests()
