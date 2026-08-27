# Owner(s): ["module: dsl-native-ops"]
#
# Minimal smoke test for the column reduction on the shared tile datapath. Covers the two
# shapes that exercise different machinery: one that fits a single stage (no reduced-axis
# split) and a tall one that must split the reduced axis and combine partials. Real numeric
# coverage comes from the reduction overrides' OpInfo suites in a later commit.

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestKernelColTile(TestCase):
    def test_reduce_col_tile_single_stage(self):
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_coltile as ct

        x = torch.randn(32, 512, device="cuda")
        out = ct.reduce_col_tile(
            T.SumOps(acc=cutlass.Float32), "smoke", x, torch.float32, npar=1
        )
        self.assertEqual(out, x.sum(dim=0), atol=1e-2, rtol=1e-2)

    def test_reduce_col_tile_split_reduced_axis(self):
        # Tall input: the driver splits the reduced axis, so stage 2 has to combine the
        # per-chunk partials. A ragged split (rows not a multiple of the chunk) too.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_coltile as ct

        x = torch.randn(4097, 256, device="cuda")
        out = ct.reduce_col_tile(
            T.MeanOps(acc=cutlass.Float32), "smoke_split", x, torch.float32
        )
        self.assertEqual(out, x.mean(dim=0), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    run_tests()
