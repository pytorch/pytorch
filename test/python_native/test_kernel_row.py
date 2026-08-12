# Owner(s): ["module: dsl-native-ops"]
#
# Minimal smoke test for the K1 vectorized row kernel. Proves it compiles and reduces
# a contiguous last dim on a tiny input; real numeric coverage comes from the
# reduction overrides' OpInfo suites in a later commit.

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestKernelRow(TestCase):
    def test_reduce_row(self):
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_row

        x = torch.randn(128, 512, device="cuda")
        out = kernel_row.reduce_row(
            T.SumOps(acc=cutlass.Float32), "smoke", x, torch.float32
        )
        self.assertEqual(out, x.sum(dim=1), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
