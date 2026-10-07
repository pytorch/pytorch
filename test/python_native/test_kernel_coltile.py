# Owner(s): ["module: dsl-native-ops"]
# Smoke tests for single-stage and split column reduction; OpInfo covers numerics.

import sys
import unittest
from unittest import mock

import torch
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUTEDSL,
    TestCase,
)


if not TEST_CUTEDSL:
    sys.stderr.write("CuTeDSL not available\n")
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("CuTeDSL not available")

import cutlass

from torch._native.ops.reductions import (
    kernel_coltile as ct,
    kernel_general as kg,
    tile,
    traits as T,
)


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@unittest.skipUnless(SM90OrLater, "Hopper+ required")
class TestKernelColTile(TestCase):
    def test_reduce_col_tile_single_stage(self):
        x = torch.randn(32, 512, device="cuda")
        out = ct.reduce_col_tile(
            T.SumOps(acc=cutlass.Float32), "smoke", x, torch.float32, npar=1
        )
        torch.testing.assert_close(out, x.sum(dim=0), atol=1e-2, rtol=1e-2)

    def test_reduce_col_tile_split_reduced_axis(self):
        # Exercise a ragged reduced-axis split and stage-2 partial combination.
        x = torch.randn(4097, 256, device="cuda")
        out = ct.reduce_col_tile(
            T.MeanOps(acc=cutlass.Float32), "smoke_split", x, torch.float32
        )
        torch.testing.assert_close(out, x.mean(dim=0), atol=1e-3, rtol=1e-3)

    def test_stage2_combine_body(self):
        # Only wide, split inputs select the otherwise untested shared combine body.
        c = ct._C_THREAD_STAGE2 + 256  # over the thread-per-column crossover
        x = torch.randn(4096, c, device="cuda")  # tall enough that _split_p splits it
        self.assertGreater(
            ct._split_p(4096), 1, "the shape no longer splits -- test is stale"
        )
        with mock.patch.object(tile, "TileReduce", wraps=tile.TileReduce) as built:
            out = ct.reduce_col_tile(
                T.SumOps(acc=cutlass.Float32), "combine2", x, torch.float32
            )
        # ReduceBlock is numerically identical, so assert which branch ran.
        self.assertTrue(
            any(call.kwargs.get("combine") for call in built.call_args_list),
            "stage 2 took the ReduceBlock branch, so the combine body ran nowhere",
        )
        torch.testing.assert_close(
            out, x.double().sum(dim=0).float(), atol=2e-3, rtol=1e-4
        )

    def test_dim0_argmax_is_exact_including_ties(self):
        # Verify absolute reduced indices and ATen's first-wins tie-break.
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
        trait = T.SumOps(acc=cutlass.Float32)
        x = torch.randn(512, 256, device="cuda")
        nd = torch.randn(8, 64, 128, device="cuda")
        real = ct.reduce_col_tile
        with mock.patch.object(ct, "reduce_col_tile", wraps=real) as served:
            got = kg.reduce_dim(trait, "disp_col", x, 0, torch.float32)
            got_nd = kg.reduce_dim(trait, "disp_col_nd", nd, (0, 1), torch.float32)
        self.assertEqual(served.call_count, 2, "K0 served these, not the col arm")
        torch.testing.assert_close(
            got, x.double().sum(dim=0).float(), atol=1e-3, rtol=1e-4
        )
        torch.testing.assert_close(
            got_nd, nd.double().sum(dim=(0, 1)).float(), atol=1e-3, rtol=1e-4
        )


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@unittest.skipUnless(SM90OrLater, "Hopper+ required")
class TestColTileHost(TestCase):
    def test_col_axis_carries_no_tile(self):
        # Driver-supplied column vec needs no tile; requesting one must fail clearly.
        trait = T.SumOps(acc=cutlass.Float32)
        row = tile.TileReduce(trait, cutlass.Float32, "row", 1024, threads_per_row=32)
        self.assertIs(row.tilemap, row.tm)

        col = tile.TileReduce(trait, cutlass.Float32, "col", 1024, vec=4)
        self.assertIsNone(col.tm)
        with self.assertRaisesRegex(AssertionError, "no tile"):
            col.tilemap

    def test_over_reported_split_stays_correct(self):
        # A capped split can leave blocks empty; they must load nothing and combine identities.
        r = ct._P_MAX * ct._Q_TARGET + 1  # the smallest R whose split over-reports
        self.assertEqual(ct._split_p(r), ct._P_MAX, "shape no longer over-reports")
        x = torch.randn(r, 8, device="cuda")
        out = ct.reduce_col_tile(
            T.SumOps(acc=cutlass.Float32), "overrep", x, torch.float32
        )
        torch.testing.assert_close(
            out, x.double().sum(dim=0).float(), atol=2e-3, rtol=1e-4
        )
        # Index traits expose stray partials as wrong indices.
        idx = ct.reduce_col_tile(
            T.ArgMaxOps(acc=cutlass.Float32), "overrep_idx", x, torch.int32
        )
        self.assertEqual(idx, x.argmax(dim=0).to(torch.int32))

    def test_explicit_vec_must_divide_the_column_count(self):
        # Explicit vec must divide C or trailing outputs remain uninitialized.
        x = torch.randn(64, 30, device="cuda")
        with self.assertRaisesRegex(AssertionError, "vec must divide"):
            ct.reduce_col_tile(
                T.SumOps(acc=cutlass.Float32), "badvec", x, torch.float32, vec=4
            )

    def test_explicit_sizes_must_be_positive(self):
        x = torch.randn(64, 32, device="cuda")
        trait = T.SumOps(acc=cutlass.Float32)
        with self.assertRaisesRegex(ValueError, "vec must be positive"):
            ct.reduce_col_tile(trait, "badvec0", x, torch.float32, vec=0)
        with self.assertRaisesRegex(ValueError, "npar must be positive"):
            ct.reduce_col_tile(trait, "badnpar0", x, torch.float32, npar=0)


class TestOrderedColHost(TestCase):
    @parametrize(
        "updates",
        [
            {"worker_warps": 3},
            {"tile_columns": 7},
            {"column_pack": 3},
            {"column_pack": 4, "C": 131},
            {"column_pack": 4, "nouts": 2},
        ],
    )
    def test_mapping_guards(self, updates):
        """Reject column mappings that violate load, layout, or ordered-plan constraints."""
        from torch._native.ops.reductions import kernel_rowtile as rt

        trait = T.SumOps(acc=cutlass.Float32)
        plan = rt.trait_itree_plan(
            trait, 32768, 132, 4, stage=False, arch=rt._ITREE_ARCH["default"]
        )
        kwargs = dict(R=32768, C=132, B=1, nouts=1)
        kwargs.update(updates)
        with self.assertRaises(ValueError):
            ct._OrderedColReduce(trait, plan, **kwargs)


@unittest.skipUnless(TEST_CUDA and SM90OrLater, "requires Hopper or later")
class TestOrderedColDevice(TestCase):
    def test_direct_plan_matches_the_indexed_tree(self, device):
        """The unsplit baseline must preserve the reference DAG bit for bit."""
        x = torch.randn(2, 128, 33, device=device)
        trait = T.SumOps(acc=cutlass.Float32)
        got = ct.reduce_ordered_col(trait, "ordered_direct", x, [torch.float32], 1)
        reference = kg._try_indexed_itree(
            trait,
            "ordered_direct_reference",
            x,
            [(128, 33)],
            [(33, 1), (2, 128 * 33)],
            2 * 33,
            128,
            [torch.float32],
            1,
        )
        self.assertIsNotNone(got)
        self.assertIsNotNone(reference)
        self.assertEqual(
            got[0].flatten().view(torch.uint8), reference[0].view(torch.uint8)
        )

    def test_split_plan_matches_the_indexed_tree(self, device):
        """Preserve each row's inner-tree DAG across ordered column partials."""
        rows, columns = 100003, 32
        x = torch.randn(rows, columns, device=device)
        trait = T.SumOps(acc=cutlass.Float32)
        got = ct.reduce_ordered_col(
            trait,
            "ordered_split",
            x,
            [torch.float32],
            1,
            allow_split=True,
        )
        reference = kg._try_indexed_itree(
            trait,
            "ordered_split_reference",
            x,
            [(rows, columns)],
            [(columns, 1)],
            columns,
            rows,
            [torch.float32],
            1,
        )
        self.assertIsNotNone(got)
        self.assertIsNotNone(reference)
        self.assertEqual(got[0].view(torch.uint8), reference[0].view(torch.uint8))

    @parametrize(
        "op,dtype,rows,columns,pack,workers,width,offset",
        [
            ("sum", torch.float32, 128, 132, 2, 8, 16, 4),
            ("argmax", torch.float32, 32769, 33, 1, 8, 8, 4),
            ("var_mean", torch.bfloat16, 32768, 33, 1, 16, 32, 4),
            ("mean", torch.bfloat16, 32769, 36, 4, 16, 8, 1),
        ],
    )
    def test_ordered_mapping_bits(
        self, device, op, dtype, rows, columns, pack, workers, width, offset
    ):
        """Keep packed, indexed, and multi-output column mappings bitwise exact."""
        from torch import _native as native

        batches = 2
        storage = torch.empty(
            batches * rows * columns + offset, dtype=dtype, device=device
        )
        x = storage[offset:].view(batches, rows, columns)
        x.uniform_(-0.5, 0.5)
        trait = {
            "sum": T.SumOps(),
            "mean": T.MeanOps(),
            "argmax": T.ArgMaxOps(),
            "var_mean": T.VarMeanOps(correction=0),
        }[op]
        output = torch.int64 if op == "argmax" else torch.float32
        outputs = [output] * (2 if op == "var_mean" else 1)
        parts = {}
        original = kg._launch

        def record(block, key, ins, outs):
            if key[0] in ("genitree2", "ordered_col2"):
                parts[key[0]] = ins
            return original(block, key, ins, outs)

        cfg = ct.OrderedColConfig(
            0,
            worker_warps=workers,
            tile_columns=width,
            full_tiles=True,
            column_pack=pack,
        )

        def run(**kwargs):
            return ct.reduce_ordered_col(
                trait,
                "mapping_" + op,
                x,
                outputs,
                len(outputs),
                **kwargs,
            )

        with (
            native._unconditional_masked(),
            torch.backends.python_native.cutedsl.disabled(),
            mock.patch.object(ct, "_launch", record),
            mock.patch.object(kg, "_launch", record),
            mock.patch.object(ct, "select_ordered_col_config", return_value=cfg),
        ):
            if offset % pack:
                with self.assertRaisesRegex(ValueError, "aligned contiguous"):
                    run(column_pack=pack)
            with mock.patch.object(
                ct, "_OrderedColReduce", wraps=ct._OrderedColReduce
            ) as kernel:
                actual = run()
            self.assertEqual(
                kernel.call_args.kwargs["column_pack"], 1 if offset % pack else pack
            )
            self.assertEqual(
                kernel.call_args.kwargs["tile_columns"], 32 if offset % pack else width
            )
            self.assertIsNotNone(actual)
            reference = kg._try_indexed_itree(
                trait,
                "mapping_" + op,
                x,
                [(rows, columns)],
                [(columns, 1), (batches, rows * columns)],
                batches * columns,
                rows,
                outputs,
                len(outputs),
            )
            self.assertIsNotNone(reference)
            self.assertEqual(len(actual), len(reference))
            for got, expected in zip(actual, reference):
                self.assertEqual(
                    got.flatten().view(torch.uint8), expected.view(torch.uint8)
                )
            self.assertEqual("ordered_col2" in parts, "genitree2" in parts)
            if "genitree2" in parts:
                self.assertEqual(len(parts["ordered_col2"]), trait.nfields)
                for got, expected in zip(parts["ordered_col2"], parts["genitree2"]):
                    self.assertEqual(got.view(torch.uint8), expected.view(torch.uint8))


instantiate_parametrized_tests(TestOrderedColHost)
instantiate_device_type_tests(TestOrderedColDevice, globals(), only_for="cuda")
instantiate_parametrized_tests(TestKernelColTile)


if __name__ == "__main__":
    run_tests()
