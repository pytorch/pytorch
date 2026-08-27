# Owner(s): ["module: dsl-native-ops"]
#
# Minimal smoke test for the general (K0) reduction kernel + dispatcher. This only
# proves the kernel COMPILES and runs the general ReduceBlock path end to end on a
# tiny input; real numeric coverage arrives with the reduction OVERRIDES (via the
# numpy-referenced OpInfo suites) in a later commit. Reduces a MIDDLE dim so the
# dispatcher stays in the general path and does not pull in the row/col/xcta fast
# kernels (added in later commits).

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestKernelGeneral(TestCase):
    def test_reduce_dim_general_path(self):
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_general as kg

        x = torch.randn(8, 16, 32, device="cuda")
        out = kg.reduce_dim(
            T.SumOps(acc=cutlass.Float32), "smoke", x, [1], torch.float32
        )
        self.assertEqual(out, x.sum(dim=1), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
