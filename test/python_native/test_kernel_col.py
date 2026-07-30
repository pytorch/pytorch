# Owner(s): ["module: dsl-native-ops"]
#
# Minimal smoke test for the K2 vectorized column kernel (reduce dim 0). Proves it
# compiles and runs on a tiny input; real numeric coverage comes from the reduction
# overrides' OpInfo suites in a later commit.

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestKernelCol(TestCase):
    def test_reduce_col(self):
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_col

        x = torch.randn(256, 1024, device="cuda")
        out = kernel_col.reduce_col(
            T.SumOps(acc=cutlass.Float32), "smoke", x, torch.float32
        )
        self.assertEqual(out, x.sum(dim=0), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
