# Owner(s): ["module: dsl-native-ops"]
# Smoke tests for single-stage and split column reduction; OpInfo covers numerics.

import unittest

import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfNoCuteDSL, TestCase


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@skipIfNoCuteDSL
class TestKernelColTile(TestCase):
    def test_reduce_col_tile_single_stage(self):
        import cutlass

        from torch._native.ops.reductions import kernel_coltile as ct, traits as T

        x = torch.randn(32, 512, device="cuda")
        out = ct.reduce_col_tile(
            T.SumOps(acc=cutlass.Float32), "smoke", x, torch.float32, npar=1
        )
        self.assertEqual(out, x.sum(dim=0), atol=1e-2, rtol=1e-2)

    def test_reduce_col_tile_split_reduced_axis(self):
        # Exercise a ragged reduced-axis split and stage-2 partial combination.
        import cutlass

        from torch._native.ops.reductions import kernel_coltile as ct, traits as T

        x = torch.randn(4097, 256, device="cuda")
        out = ct.reduce_col_tile(
            T.MeanOps(acc=cutlass.Float32), "smoke_split", x, torch.float32
        )
        self.assertEqual(out, x.mean(dim=0), atol=1e-3, rtol=1e-3)

    def test_stage2_combine_body(self):
        # Only wide, split inputs select the otherwise untested shared combine body.
        from unittest import mock

        import cutlass

        from torch._native.ops.reductions import (
            kernel_coltile as ct,
            tile as tl,
            traits as T,
        )

        c = ct._C_THREAD_STAGE2 + 256  # over the thread-per-column crossover
        x = torch.randn(4096, c, device="cuda")  # tall enough that _split_p splits it
        self.assertGreater(
            ct._split_p(4096), 1, "the shape no longer splits -- test is stale"
        )
        with mock.patch.object(tl, "TileReduce", wraps=tl.TileReduce) as built:
            out = ct.reduce_col_tile(
                T.SumOps(acc=cutlass.Float32), "combine2", x, torch.float32
            )
        # ReduceBlock is numerically identical, so assert which branch ran.
        self.assertTrue(
            any(call.kwargs.get("combine") for call in built.call_args_list),
            "stage 2 took the ReduceBlock branch, so the combine body ran nowhere",
        )
        self.assertEqual(out, x.double().sum(dim=0).float(), atol=2e-3, rtol=1e-4)

    def test_dim0_argmax_is_exact_including_ties(self):
        # Verify absolute reduced indices and ATen's first-wins tie-break.
        import cutlass

        from torch._native.ops.reductions import kernel_coltile as ct, traits as T

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
        # Three fields select the otherwise untested high-register-pressure launch.
        import cutlass

        from torch._native.ops.reductions import kernel_coltile as ct, traits as T

        x = torch.randn(4096, 256, device="cuda")
        out = ct.reduce_col_tile(
            T.WelfordOps(correction=1, acc=cutlass.Float32),
            "welford_col",
            x,
            torch.float32,
        )
        self.assertEqual(out, x.var(dim=0), atol=1e-4, rtol=1e-4)

    def test_dispatcher_routes_a_column_reduction(self):
        # Numerics cannot distinguish the column arm from K0; assert routing and reshape.
        from unittest import mock

        import cutlass

        from torch._native.ops.reductions import (
            kernel_coltile as ct,
            kernel_general as kg,
            traits as T,
        )

        trait = T.SumOps(acc=cutlass.Float32)
        x = torch.randn(512, 256, device="cuda")
        nd = torch.randn(8, 64, 128, device="cuda")
        real = ct.reduce_col_tile
        with mock.patch.object(ct, "reduce_col_tile", wraps=real) as served:
            got = kg.reduce_dim(trait, "disp_col", x, 0, torch.float32)
            got_nd = kg.reduce_dim(trait, "disp_col_nd", nd, (0, 1), torch.float32)
        self.assertEqual(served.call_count, 2, "K0 served these, not the col arm")
        self.assertEqual(got, x.double().sum(dim=0).float(), atol=1e-3, rtol=1e-4)
        self.assertEqual(
            got_nd, nd.double().sum(dim=(0, 1)).float(), atol=1e-3, rtol=1e-4
        )


@skipIfNoCuteDSL
class TestColTileHost(TestCase):
    def test_col_axis_carries_no_tile(self):
        # Driver-supplied column vec needs no tile; requesting one must fail clearly.
        import cutlass

        from torch._native.ops.reductions import tile, traits as T

        trait = T.SumOps(acc=cutlass.Float32)
        row = tile.TileReduce(trait, cutlass.Float32, "row", 1024, tpr=32)
        self.assertIs(row.tilemap, row.tm)

        col = tile.TileReduce(trait, cutlass.Float32, "col", 1024, vec=4)
        self.assertIsNone(col.tm)
        with self.assertRaisesRegex(AssertionError, "no tile"):
            col.tilemap

    def test_over_reported_split_stays_correct(self):
        # A capped split can leave blocks empty; they must load nothing and combine identities.
        import cutlass

        from torch._native.ops.reductions import kernel_coltile as ct, traits as T

        r = ct._P_MAX * ct._Q_TARGET + 1  # the smallest R whose split over-reports
        self.assertEqual(ct._split_p(r), ct._P_MAX, "shape no longer over-reports")
        x = torch.randn(r, 8, device="cuda")
        out = ct.reduce_col_tile(
            T.SumOps(acc=cutlass.Float32), "overrep", x, torch.float32
        )
        self.assertEqual(out, x.double().sum(dim=0).float(), atol=2e-3, rtol=1e-4)
        # Index traits expose stray partials as wrong indices.
        idx = ct.reduce_col_tile(
            T.ArgMaxOps(acc=cutlass.Float32), "overrep_idx", x, torch.int32
        )
        self.assertEqual(idx, x.argmax(dim=0).to(torch.int32))

    def test_explicit_vec_must_divide_the_column_count(self):
        # Explicit vec must divide C or trailing outputs remain uninitialized.
        import cutlass

        from torch._native.ops.reductions import kernel_coltile as ct, traits as T

        x = torch.randn(64, 30, device="cuda")
        with self.assertRaisesRegex(AssertionError, "vec must divide"):
            ct.reduce_col_tile(
                T.SumOps(acc=cutlass.Float32), "badvec", x, torch.float32, vec=4
            )


if __name__ == "__main__":
    run_tests()
