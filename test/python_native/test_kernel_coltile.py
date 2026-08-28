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

    def test_stage2_combine_body(self):
        # Stage 2 has TWO shapes: a ReduceBlock fold, and the shared body in `combine` mode, which
        # is selected only when C >= _C_THREAD_STAGE2 AND the reduced axis is split. The existing
        # tests use C=512/256, so both took the ReduceBlock branch and the combine body -- the
        # only construction of TileReduce(combine=True) in the tree -- ran nowhere.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_coltile as ct

        c = ct._C_THREAD_STAGE2 + 256  # over the thread-per-column crossover
        x = torch.randn(4096, c, device="cuda")  # tall enough that _split_p splits it
        self.assertGreater(
            ct._split_p(4096), 1, "the shape no longer splits -- test is stale"
        )
        out = ct.reduce_col_tile(
            T.SumOps(acc=cutlass.Float32), "combine2", x, torch.float32
        )
        self.assertEqual(out, x.double().sum(dim=0).float(), atol=2e-3, rtol=1e-4)

    def test_dim0_argmax_is_exact_including_ties(self):
        # The commit's headline claim is that this path carries the ABSOLUTE reduced index, so
        # argmax/argmin over dim 0 are exact with ATen's first-wins tie-break. randn has no ties,
        # so build them: two rows hold the same maximum and the LOWER row must win.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_coltile as ct

        x = torch.randn(2048, 256, device="cuda")
        x[100, :] = 5.0  # the winner
        x[900, :] = 5.0  # an exact tie, further down: must NOT win
        for trait, ref in (
            (T.ArgMaxOps, torch.full((256,), 100, device="cuda", dtype=torch.int32)),
            (T.ArgMinOps, x.argmin(dim=0).to(torch.int32)),
        ):
            with self.subTest(trait=trait.__name__):
                got = ct.reduce_col_tile(
                    trait(acc=cutlass.Float32),
                    f"argdim0_{trait.__name__}",
                    x,
                    torch.int32,
                )
                self.assertEqual(got, ref)

    def test_three_field_welford_launch(self):
        # A 3-field accumulator is the only thing that selects the _NT_WIDE_ACC branch of the
        # column launch (register pressure per thread scales with nfields), and nothing crossed it.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_coltile as ct

        x = torch.randn(4096, 256, device="cuda")
        out = ct.reduce_col_tile(
            T.WelfordOps(correction=1, acc=cutlass.Float32),
            "welford_col",
            x,
            torch.float32,
        )
        self.assertEqual(out, x.var(dim=0), atol=1e-4, rtol=1e-4)

    def test_dispatcher_routes_a_column_reduction(self):
        # The other tests call reduce_col_tile directly, so the line this commit actually changes
        # for users -- fast_kind's col arm plus the reshape/_as_shape round-trip in _reduce -- was
        # untested. Drive it through the dispatcher, and through an n-D input that has to coalesce
        # to a single reduced + single kept run before the col arm can take it.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import kernel_general as kg

        trait = T.SumOps(acc=cutlass.Float32)
        x = torch.randn(512, 256, device="cuda")
        got = kg.reduce_dim(trait, "disp_col", x, 0, torch.float32)
        self.assertEqual(got, x.double().sum(dim=0).float(), atol=1e-3, rtol=1e-4)

        nd = torch.randn(8, 64, 128, device="cuda")
        got_nd = kg.reduce_dim(trait, "disp_col_nd", nd, (0, 1), torch.float32)
        self.assertEqual(
            got_nd, nd.double().sum(dim=(0, 1)).float(), atol=1e-3, rtol=1e-4
        )


@skipIfNoCuteDSL
class TestColTileHost(TestCase):
    def test_col_axis_carries_no_tile(self):
        # Host-only, so it runs wherever the DSL imports. The col axis takes `vec` from the
        # driver (accumulators per thread) rather than from a load width, so it has no tile --
        # and asking for one must fail loudly, not dereference None inside a kernel build.
        import cutlass

        from torch._native.ops._cutedsl import traits as T
        from torch._native.ops.reductions import tile

        trait = T.SumOps(acc=cutlass.Float32)
        row = tile.TileReduce(trait, cutlass.Float32, "row", 1024, tpr=32)
        self.assertIs(row.tilemap, row.tm)

        col = tile.TileReduce(trait, cutlass.Float32, "col", 1024, vec=4)
        self.assertIsNone(col.tm)
        with self.assertRaisesRegex(AssertionError, "no tile"):
            col.tilemap


if __name__ == "__main__":
    run_tests()
