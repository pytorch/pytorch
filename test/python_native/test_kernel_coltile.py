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

from torch._native.cutedsl import launch as _L
from torch._native.ops.reductions import (
    kernel_coltile as ct,
    kernel_general as kg,
    tile,
    traits as T,
)


# A column count of 4 mod 8 declares 16-byte fp32 rows but 8-byte 16-bit rows.
_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
_TOL = {
    torch.float32: {"atol": 1e-3, "rtol": 1e-4},
    torch.bfloat16: {"atol": 1e-1, "rtol": 1.6e-2},
    torch.float16: {"atol": 1e-2, "rtol": 1e-3},
}


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@unittest.skipUnless(SM90OrLater, "Hopper+ required")
class TestKernelColTile(TestCase):
    def setUp(self):
        super().setUp()
        # The batched and two-output column routes are measured, and enabled, on B200 only.
        current = (torch.cuda.get_device_capability(),)
        patcher = mock.patch.object(kg, "_BATCHED_COL_CCS", current)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_reduce_col_tile_single_stage(self):
        # A misaligned input needs an element-aligned descriptor.
        raw = torch.randn(32 * 512 + 1, device="cuda")
        x = raw[1:].view(32, 512)
        self.assertEqual(
            _L.supported_alignment(x, tile.align_bytes(512, x.element_size())),
            x.element_size(),
        )
        out = ct.reduce_col_tile(
            T.SumOps(acc=cutlass.Float32),
            "smoke",
            x,
            torch.float32,
            npar=1,
        )
        self.assertEqual(out, x.sum(dim=0), atol=1e-2, rtol=1e-2)

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

    @parametrize("dtype", _DTYPES)
    def test_batched_columns_split_and_combine(self, dtype):
        # (B, R, C) stage one offsets each batch's rows; cover both splits and combines.
        trait = T.SumOps(acc=cutlass.Float32)
        for npar in (1, 4):
            for columns in (260, ct._C_THREAD_STAGE2 + 4):
                x = torch.randn(3, 515, columns, device="cuda").to(dtype)
                with self.subTest(npar=npar, pc=3 * columns >= ct._C_THREAD_STAGE2):
                    (got,) = ct.reduce_batched_col_tile(
                        trait, "batched_col", x, [dtype], 1, npar=npar
                    )
                    want = x.double().sum(dim=1).to(dtype)
                    torch.testing.assert_close(got, want, **_TOL[dtype])

    def test_batched_argmax_reports_the_reduced_index(self):
        # Batched views must report positions along the reduced axis, first tie winning.
        x = torch.randn(2, 2048, 256, device="cuda")
        x[:, 100, :] = 5.0
        x[:, 900, :] = 5.0
        real = ct.reduce_batched_col_tile
        with mock.patch.object(ct, "reduce_batched_col_tile", wraps=real) as served:
            got = kg.reduce_dim(
                T.ArgMaxOps(acc=cutlass.Float32), "batched_argmax", x, 1, torch.int32
            )
        self.assertEqual(served.call_count, 1)
        self.assertEqual(
            got, torch.full((2, 256), 100, device="cuda", dtype=torch.int32)
        )

    @parametrize("dtype", _DTYPES)
    def test_two_output_columns(self, dtype):
        x = torch.randn(4097, 260, device="cuda").to(dtype)
        with torch.backends.python_native.cutedsl.disabled():
            var_mean, aminmax = torch.var_mean(x, dim=0), x.aminmax(dim=0)
        expected = {"var_mean": var_mean, "aminmax": aminmax}
        traits = {
            "var_mean": T.VarMeanOps(correction=1, acc=cutlass.Float32),
            "aminmax": T.AMinMaxOps(acc=cutlass.Float32),
        }
        for name, trait in traits.items():
            with self.subTest(trait=name):
                got = ct.reduce_col_tile_2out(trait, f"{name}_col", x, [dtype, dtype])
                for g, e in zip(got, expected[name]):
                    torch.testing.assert_close(g, e, **_TOL[dtype])

    @parametrize("dtype", _DTYPES)
    def test_batched_two_output_columns(self, dtype):
        # Two-output batched views fold both outputs in stage two; cover both combines.
        trait = T.VarMeanOps(correction=1, acc=cutlass.Float32)
        for columns in (260, ct._C_THREAD_STAGE2 + 4):
            x = torch.randn(3, 515, columns, device="cuda").to(dtype)
            with torch.backends.python_native.cutedsl.disabled():
                expected = torch.var_mean(x, dim=1)
            with self.subTest(pc=3 * columns >= ct._C_THREAD_STAGE2):
                got = ct.reduce_batched_col_tile(
                    trait, "batched_var_mean", x, [dtype, dtype], 2, npar=4
                )
                for g, e in zip(got, expected):
                    torch.testing.assert_close(g, e, **_TOL[dtype])

    def test_batched_column_route_gate(self):
        # Middle-axis reductions use the batched kernel only where it measured faster.
        f32 = torch.float32
        one = T.SumOps(acc=cutlass.Float32)
        two = T.VarMeanOps(correction=1, acc=cutlass.Float32)
        cases = (
            ((4, 64, 2), 1, True),
            ((4, 1024, 2), 1, False),
            ((4, 1024, 16), 1, True),
            ((4, 64, 16), 2, True),
            ((4, 1024, 16), 2, False),
            ((4, 1024, 2048), 2, True),
        )
        for shape, nouts, routed in cases:
            x = torch.randn(shape, device="cuda")
            with torch.backends.python_native.cutedsl.disabled():
                expected = (x.sum(dim=1),) if nouts == 1 else torch.var_mean(x, dim=1)
            real = ct.reduce_batched_col_tile
            with (
                self.subTest(shape=shape, nouts=nouts),
                mock.patch.object(ct, "reduce_batched_col_tile", wraps=real) as served,
            ):
                if nouts == 1:
                    got = (kg.reduce_dim(one, "gate", x, 1, f32),)
                else:
                    got = kg.reduce_dim2(two, "gate_2out", x, 1, [f32, f32])
                self.assertEqual(served.call_count, int(routed))
                for g, e in zip(got, expected):
                    torch.testing.assert_close(g, e, atol=1e-3, rtol=1e-4)

    def test_batched_view_collapses_adjacent_axes(self):
        # Leading, reduced and trailing runs each fold into one (B, R, C) axis and back.
        x = torch.randn(2, 3, 32, 5, 16, device="cuda")
        real = ct.reduce_batched_col_tile
        with mock.patch.object(ct, "reduce_batched_col_tile", wraps=real) as served:
            got = kg.reduce_dim(
                T.SumOps(acc=cutlass.Float32), "collapse", x, [2, 3], torch.float32
            )
        served.assert_called_once()
        self.assertEqual(served.call_args.args[2].shape, (6, 160, 16))
        torch.testing.assert_close(
            got, x.double().sum(dim=(2, 3)).float(), atol=1e-4, rtol=1e-4
        )

    def test_two_output_columns_route_to_coltile(self):
        # Dense dim-0 and transposed two-output reductions reach the column kernel when
        # they pass the batched gate as B == 1; narrow columns stay on the general kernel.
        trait = T.VarMeanOps(correction=1, acc=cutlass.Float32)
        f32 = [torch.float32, torch.float32]
        wide = torch.randn(1100, 8192, device="cuda")
        narrow = torch.randn(4096, 256, device="cuda")
        cases = (
            ("dim0", wide, 0, True),
            ("transposed", wide.t(), 1, True),
            ("narrow", narrow, 0, False),
        )
        for name, x, dim, routed in cases:
            with torch.backends.python_native.cutedsl.disabled():
                expected = torch.var_mean(x, dim=dim)
            real = ct.reduce_col_tile_2out
            with (
                self.subTest(layout=name),
                mock.patch.object(ct, "reduce_col_tile_2out", wraps=real) as served,
            ):
                got = kg.reduce_dim2(trait, f"var_mean_{name}", x, dim, f32)
                self.assertEqual(served.call_count, int(routed))
                for g, e in zip(got, expected):
                    torch.testing.assert_close(g, e, atol=1e-4, rtol=1e-4)

    @parametrize("dtype", _DTYPES)
    def test_value_index_columns_keep_the_first_tie(self, dtype):
        # max.dim/min.dim project a value and the first tied index through stage two.
        i32 = torch.int32
        cases = (
            ("max", T.MaxDimOps(acc=cutlass.Float32), 10.0),
            ("min", T.MinDimOps(acc=cutlass.Float32), -10.0),
        )
        for name, trait, tie in cases:
            for batches, columns in ((3, 260), (1, ct._C_THREAD_STAGE2 + 4)):
                x = torch.randn(batches, 1100, columns, device="cuda").to(dtype)
                x[:, 100] = tie
                x[:, 900] = tie
                with self.subTest(op=name, pc=batches * columns >= ct._C_THREAD_STAGE2):
                    vals, idx = ct.reduce_batched_col_tile(
                        trait, f"{name}dim_batched", x, [dtype, i32], 2, npar=4
                    )
                    self.assertEqual(vals, torch.full_like(vals, tie))
                    self.assertEqual(idx, torch.full_like(idx, 100))
            two_out, batched = ct.reduce_col_tile_2out, ct.reduce_batched_col_tile
            for shape, dim in (((1100, 8196), 0), ((2, 1100, 4100), 1)):
                x = torch.randn(shape, device="cuda").to(dtype)
                x.select(dim, 100).fill_(tie)
                x.select(dim, 900).fill_(tie)
                with (
                    self.subTest(op=name, routed=shape),
                    mock.patch.object(ct, "reduce_col_tile_2out", wraps=two_out) as two,
                    mock.patch.object(
                        ct, "reduce_batched_col_tile", wraps=batched
                    ) as bat,
                ):
                    key = f"{name}dim_route"
                    vals, idx = kg.reduce_dim2(trait, key, x, dim, [dtype, i32])
                    self.assertEqual(two.call_count, int(x.dim() == 2))
                    self.assertEqual(bat.call_count, int(x.dim() == 3))
                    self.assertEqual(vals, torch.full_like(vals, tie))
                    self.assertEqual(idx, torch.full_like(idx, 100))

    def test_transposed_dense_columns_route_to_coltile(self):
        # A transposed view whose kept axis is dense is served by the column kernel.
        x = torch.randn(256, 4096, device="cuda").t()
        real = ct.reduce_col_tile
        with mock.patch.object(ct, "reduce_col_tile", wraps=real) as served:
            got = kg.reduce_dim(
                T.SumOps(acc=cutlass.Float32), "transposed_col", x, 1, torch.float32
            )
        self.assertEqual(served.call_count, 1)
        torch.testing.assert_close(
            got, x.double().sum(dim=1).float(), atol=1e-3, rtol=1e-4
        )


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@unittest.skipUnless(SM90OrLater, "Hopper+ required")
class TestColTileHost(TestCase):
    def test_route_gate_keeps_other_architectures_on_general(self):
        # Off B200, only 2D one-output column views leave the general kernel.
        wide = torch.empty(1100, 8192, device="cuda")
        batched = torch.empty(4, 64, 16, device="cuda")
        for view, nouts in ((wide, 1), (wide, 2), (batched, 1), (batched, 2)):
            with self.subTest(shape=tuple(view.shape), nouts=nouts):
                self.assertTrue(kg._col_route_ok(view, nouts, (10, 0)))
                self.assertEqual(
                    kg._col_route_ok(view, nouts, (9, 0)),
                    view.dim() == 2 and nouts == 1,
                )
        x = torch.randn(4, 64, 16, device="cuda")
        real = ct.reduce_batched_col_tile
        with (
            mock.patch.object(kg, "_BATCHED_COL_CCS", ()),
            mock.patch.object(ct, "reduce_batched_col_tile", wraps=real) as served,
        ):
            got = kg.reduce_dim(
                T.SumOps(acc=cutlass.Float32), "gate_off", x, 1, torch.float32
            )
        served.assert_not_called()
        torch.testing.assert_close(
            got, x.double().sum(dim=1).float(), atol=1e-4, rtol=1e-4
        )

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

    def test_explicit_npar_must_fit_int32_partials(self):
        x = torch.randn(1, 1 << 20, device="cuda")
        with self.assertRaisesRegex(ValueError, "overflows Int32"):
            ct.reduce_col_tile(
                T.SumOps(acc=cutlass.Float32), "bignpar", x, torch.float32, npar=4096
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
        default = rt._ITREE_ARCH["default"]
        with mock.patch.object(rt, "_itree_arch", return_value=default):
            plan = rt.trait_itree_plan(trait, 32768, 132, 4, stage=False)
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
