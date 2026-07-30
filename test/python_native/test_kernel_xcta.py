# Owner(s): ["module: dsl-native-ops"]
#
# Minimal smoke test for the fused two-stage cross-CTA kernel (few-row / huge-N).
# Proves it compiles and runs on a tiny single huge row; real numeric coverage comes
# from the reduction overrides' OpInfo suites in a later commit.

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestKernelXcta(TestCase):
    def test_reduce_row_xcta(self):
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_xcta

        x = torch.randn(1, 1 << 20, device="cuda")
        out = kernel_xcta.reduce_row_xcta(
            T.SumOps(acc=cutlass.Float32), "smoke", x, torch.float32
        )
        self.assertEqual(out, x.sum(dim=1), atol=1e-1, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
