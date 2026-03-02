# Owner(s): ["module: inductor"]

import unittest
import torch
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_cuda import skipCUDAIfRocm
from torch.testing._internal.inductor_utils import HAS_CUDA


@skipCUDAIfRocm
@unittest.skipIf(not HAS_CUDA, "CUDA required for this test")
class TestPowMixedDtypeInductor(unittest.TestCase):
    def test_pow_mixed_float_dtype_no_crash(self):
        """
        This test ensures that torch.compile does not generate
        mixed-dtype Triton pow (e.g. float64, float32), which
        is unsupported by libdevice.pow.
        """

        @torch.compile
        def f(x):
            # 1e-56 forces float64 promotion in eager
            return torch.pow(x + 1e-56, -0.5)

        x = torch.randn(4, 4, device="cuda", dtype=torch.float32)

        # The test passes if this does NOT raise InductorError
        y = f(x)

        # Sanity checks
        self.assertTrue(torch.isfinite(y).all())
        self.assertEqual(y.device.type, "cuda")


if __name__ == "__main__":
    run_tests()
