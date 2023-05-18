# Owner(s): ["module: inductor"]

import sys
import unittest

from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_LINUX, TEST_WITH_ROCM
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
        self.assertTrue(best_config.kwargs.get("XBLOCK") == 16)


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA and not TEST_WITH_ROCM:
        run_tests()
