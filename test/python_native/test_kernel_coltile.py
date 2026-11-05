# Owner(s): ["module: dsl-native-ops"]
# Smoke tests for single-stage and split column reduction; OpInfo covers numerics.

import sys
import unittest
from unittest import mock

import torch
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
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
_VALUE_INDEX = {
    "max": (T.MaxDimOps(acc=cutlass.Float32), 10.0),
    "min": (T.MinDimOps(acc=cutlass.Float32), -10.0),
}


def _tags(trait_key):
    """Tags of the column plans compiled for trait_key."""
    return {k[0] for k in ct._CACHE if k[1] == trait_key}


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@unittest.skipUnless(SM90OrLater, "Hopper+ required")
class TestKernelColTile(TestCase):
    def setUp(self):
        super().setUp()
        # Enable the batched and two-output column routes on the test GPU.
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

    def test_plans_key_and_require_the_current_device(self):
        trait = T.SumOps(acc=cutlass.Float32)
        # A split over a thread-per-column width adds a "coltile2" stage-two plan.
        x = torch.randn(4096, ct._C_THREAD_STAGE2 + 256, device="cuda")
        before = set(ct._CACHE)
        # One GPU has no other device, so pretend another one is current.
        other = x.get_device() + 1
        with mock.patch.object(torch.cuda, "current_device", return_value=other):
            with self.assertRaisesRegex(AssertionError, "is current"):
                ct.reduce_col_tile(trait, "bydevice", x, torch.float32)
        self.assertEqual(set(ct._CACHE), before)
        ct.reduce_col_tile(trait, "bydevice", x, torch.float32)
        # Both keys put the index sixth; without it, the slot holds the "col" axis.
        keys = {k[0]: k[5] for k in ct._CACHE if k[1] == "bydevice"}
        index = x.get_device()
        self.assertEqual(keys, {"coltile": index, "coltile2": index})

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
        x = torch.randn(4096, c, device="cuda")  # tall enough that stage one splits it
        npar, _ = ct._col_geometry(4096, c, 1, x.element_size(), 2**31 // c)
        self.assertGreater(npar, 1, "the shape no longer splits -- test is stale")
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
        torch.testing.assert_close(out, x.var(dim=0), atol=1e-4, rtol=1e-4)

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

    @parametrize("columns", [260, ct._C_THREAD_STAGE2 + 4], name_fn=lambda c: f"c{c}")
    @parametrize("npar", [1, 4])
    @parametrize("dtype", _DTYPES)
    def test_batched_columns_split_and_combine(self, dtype, npar, columns):
        # (B, R, C) stage one offsets each batch's rows; cover both splits and combines.
        x = torch.randn(3, 515, columns, device="cuda").to(dtype)
        (got,) = ct.reduce_batched_col_tile(
            T.SumOps(acc=cutlass.Float32), "batched_col", x, [dtype], 1, npar=npar
        )
        want = x.double().sum(dim=1).to(dtype)
        torch.testing.assert_close(got, want, **_TOL[dtype])

    def test_batched_argmax_reports_the_reduced_index(self):
        # Batched views must report positions along the reduced axis, first tie winning.
        x = torch.randn(2, 2048, 256, device="cuda")
        x[:, 100, :] = 5.0
        x[:, 900, :] = 5.0
        got = kg.reduce_dim(
            T.ArgMaxOps(acc=cutlass.Float32), "batched_argmax", x, 1, torch.int32
        )
        self.assertIn("coltile", _tags("batched_argmax"))
        self.assertEqual(
            got, torch.full((2, 256), 100, device="cuda", dtype=torch.int32)
        )

    @parametrize("name", ["var_mean", "aminmax"])
    @parametrize("dtype", _DTYPES)
    def test_two_output_columns(self, dtype, name):
        x = torch.randn(4097, 260, device="cuda").to(dtype)
        with torch.backends.python_native.cutedsl.disabled():
            expected = (
                torch.var_mean(x, dim=0) if name == "var_mean" else x.aminmax(dim=0)
            )
        trait = (
            T.VarMeanOps(correction=1, acc=cutlass.Float32)
            if name == "var_mean"
            else T.AMinMaxOps(acc=cutlass.Float32)
        )
        got = ct.reduce_col_tile_2out(trait, f"{name}_col", x, [dtype, dtype])
        for g, e in zip(got, expected):
            torch.testing.assert_close(g, e, **_TOL[dtype])

    @parametrize("columns", [260, ct._C_THREAD_STAGE2 + 4], name_fn=lambda c: f"c{c}")
    @parametrize("dtype", _DTYPES)
    def test_batched_two_output_columns(self, dtype, columns):
        # Two-output batched views fold both outputs in stage two; cover both combines.
        trait = T.VarMeanOps(correction=1, acc=cutlass.Float32)
        x = torch.randn(3, 515, columns, device="cuda").to(dtype)
        with torch.backends.python_native.cutedsl.disabled():
            expected = torch.var_mean(x, dim=1)
        got = ct.reduce_batched_col_tile(
            trait, "batched_var_mean", x, [dtype, dtype], 2, npar=4
        )
        for g, e in zip(got, expected):
            torch.testing.assert_close(g, e, **_TOL[dtype])

    @parametrize(
        "shape,nouts,routed",
        [
            ((4, 64, 2), 1, True),
            ((4, 1024, 2), 1, False),
            ((4, 1024, 16), 1, True),
            ((4, 64, 16), 2, True),
            ((4, 1024, 16), 2, False),
            ((4, 1024, 2048), 2, True),
        ],
        name_fn=lambda shape,
        nouts,
        routed: f"{'x'.join(map(str, shape))}_nouts{nouts}",
    )
    def test_batched_column_route_gate(self, shape, nouts, routed):
        # Middle-axis reductions use the batched kernel only where it measured faster.
        f32 = torch.float32
        x = torch.randn(shape, device="cuda")
        with torch.backends.python_native.cutedsl.disabled():
            expected = (x.sum(dim=1),) if nouts == 1 else torch.var_mean(x, dim=1)
        key = f"gate_{'x'.join(map(str, shape))}_{nouts}"
        if nouts == 1:
            got = (kg.reduce_dim(T.SumOps(acc=cutlass.Float32), key, x, 1, f32),)
        else:
            two = T.VarMeanOps(correction=1, acc=cutlass.Float32)
            got = kg.reduce_dim2(two, key, x, 1, [f32, f32])
        self.assertEqual("coltile" in _tags(key), routed)
        for g, e in zip(got, expected):
            torch.testing.assert_close(g, e, atol=1e-3, rtol=1e-4)

    def test_batched_view_collapses_adjacent_axes(self):
        # Leading, reduced and trailing runs each fold into one (B, R, C) axis and back.
        x = torch.randn(2, 3, 32, 5, 16, device="cuda")
        got = kg.reduce_dim(
            T.SumOps(acc=cutlass.Float32), "collapse", x, [2, 3], torch.float32
        )
        self.assertIn("coltile", _tags("collapse"))
        torch.testing.assert_close(
            got, x.double().sum(dim=(2, 3)).float(), atol=1e-4, rtol=1e-4
        )

    @parametrize("layout", ["dim0", "transposed", "narrow"])
    def test_two_output_columns_route_to_coltile(self, layout):
        # Dense dim-0 and transposed two-output reductions reach the column kernel when
        # they pass the batched gate as B == 1; narrow columns stay on the general kernel.
        trait = T.VarMeanOps(correction=1, acc=cutlass.Float32)
        x, dim, routed = {
            "dim0": (torch.randn(1100, 8192, device="cuda"), 0, True),
            "transposed": (torch.randn(1100, 8192, device="cuda").t(), 1, True),
            "narrow": (torch.randn(4096, 256, device="cuda"), 0, False),
        }[layout]
        with torch.backends.python_native.cutedsl.disabled():
            expected = torch.var_mean(x, dim=dim)
        f32 = [torch.float32, torch.float32]
        got = kg.reduce_dim2(trait, f"var_mean_{layout}", x, dim, f32)
        self.assertEqual("coltile" in _tags(f"var_mean_{layout}"), routed)
        for g, e in zip(got, expected):
            torch.testing.assert_close(g, e, atol=1e-4, rtol=1e-4)

    @parametrize(
        "batches,columns",
        [(3, 260), (1, ct._C_THREAD_STAGE2 + 4)],
        name_fn=lambda b, c: f"{b}x{c}",
    )
    @parametrize("op", ["max", "min"])
    @parametrize("dtype", _DTYPES)
    def test_value_index_columns_keep_the_first_tie(self, dtype, op, batches, columns):
        # max.dim/min.dim project a value and the first tied index through stage two.
        trait, tie = _VALUE_INDEX[op]
        x = torch.randn(batches, 1100, columns, device="cuda").to(dtype)
        x[:, 100] = tie
        x[:, 900] = tie
        vals, idx = ct.reduce_batched_col_tile(
            trait, f"{op}dim_batched", x, [dtype, torch.int32], 2, npar=4
        )
        self.assertEqual(vals, torch.full_like(vals, tie))
        self.assertEqual(idx, torch.full_like(idx, 100))

    @parametrize(
        "shape,dim",
        [((1100, 8196), 0), ((2, 1100, 4100), 1)],
        name_fn=lambda s, d: f"{len(s)}d",
    )
    @parametrize("op", ["max", "min"])
    @parametrize("dtype", _DTYPES)
    def test_value_index_columns_route_to_coltile(self, dtype, op, shape, dim):
        # Dispatch reaches the 2D two-output or batched kernel and keeps the first tie.
        trait, tie = _VALUE_INDEX[op]
        x = torch.randn(shape, device="cuda").to(dtype)
        x.select(dim, 100).fill_(tie)
        x.select(dim, 900).fill_(tie)
        key = f"{op}dim_route_{x.dim()}d"
        vals, idx = kg.reduce_dim2(trait, key, x, dim, [dtype, torch.int32])
        served = {(k[0], k[2]) for k in ct._CACHE if k[1] == key}
        self.assertIn(("coltile", dtype), served)
        self.assertEqual(vals, torch.full_like(vals, tie))
        self.assertEqual(idx, torch.full_like(idx, 100))

    def test_transposed_dense_columns_route_to_coltile(self):
        # A transposed view whose kept axis is dense is served by the column kernel.
        x = torch.randn(256, 4096, device="cuda").t()
        got = kg.reduce_dim(
            T.SumOps(acc=cutlass.Float32), "transposed_col", x, 1, torch.float32
        )
        self.assertIn("coltile", _tags("transposed_col"))
        torch.testing.assert_close(
            got, x.double().sum(dim=1).float(), atol=1e-3, rtol=1e-4
        )


@unittest.skipUnless(TEST_CUDA, "CUDA required")
@unittest.skipUnless(SM90OrLater, "Hopper+ required")
class TestColTileHost(TestCase):
    @parametrize("nouts", [1, 2])
    @parametrize("shape", [(1100, 8192), (4, 64, 16)], name_fn=lambda s: f"{len(s)}d")
    def test_route_gate_by_architecture(self, shape, nouts):
        # Off H100 and B200, only 2D one-output column views leave the general kernel.
        view = torch.empty(shape, device="cuda")
        self.assertTrue(kg._col_route_ok(view, nouts, (9, 0)))
        self.assertTrue(kg._col_route_ok(view, nouts, (10, 0)))
        self.assertEqual(
            kg._col_route_ok(view, nouts, (12, 0)), view.dim() == 2 and nouts == 1
        )

    @parametrize(
        "shape,nouts,routed",
        [
            ((4, 64, 2), 1, True),
            ((4, 65, 2), 1, False),
            ((4, 65, 15), 1, False),
            ((4, 65, 16), 1, True),
            ((1, 65, 8191), 2, False),
            ((2, 65, 4096), 2, True),
        ],
        name_fn=lambda shape, nouts, routed: "x".join(map(str, shape)) + f"_{nouts}",
    )
    def test_route_gate_boundaries(self, shape, nouts, routed):
        view = torch.empty(shape, device="meta")
        self.assertEqual(kg._col_route_ok(view, nouts, (10, 0)), routed)

    def test_route_gate_keeps_other_architectures_on_general(self):
        x = torch.randn(4, 64, 16, device="cuda")
        with mock.patch.object(kg, "_BATCHED_COL_CCS", ()):
            got = kg.reduce_dim(
                T.SumOps(acc=cutlass.Float32), "gate_off", x, 1, torch.float32
            )
        self.assertNotIn("coltile", _tags("gate_off"))
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
        # 64 chunks of ceil(65 / 64) = 2 rows leave the last 31 empty; they must load
        # nothing and combine identities.
        x = torch.randn(65, 8, device="cuda")
        out = ct.reduce_col_tile(
            T.SumOps(acc=cutlass.Float32), "overrep", x, torch.float32, npar=64
        )
        torch.testing.assert_close(
            out, x.double().sum(dim=0).float(), atol=2e-3, rtol=1e-4
        )
        # Index traits expose stray partials as wrong indices.
        idx = ct.reduce_col_tile(
            T.ArgMaxOps(acc=cutlass.Float32), "overrep_idx", x, torch.int32, npar=64
        )
        self.assertEqual(idx, x.argmax(dim=0).to(torch.int32))

    def test_default_split_fits_grid_y(self):
        # Two lanes would want 65536 chunks; the default caps npar at grid y's 65535.
        x = torch.randn(1 << 20, 2, device="cuda")
        out = ct.reduce_col_tile(
            T.SumOps(acc=cutlass.Float32), "grid_y", x, torch.float32
        )
        torch.testing.assert_close(
            out, x.double().sum(dim=0).float(), atol=2e-3, rtol=1e-4
        )

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

    def test_batched_rows_must_fit_int32(self):
        # Stage one indexes batched rows as batch * R + row in Int32.
        x = torch.randn(4, 8, 32, device="cuda")
        args = (x, {1}, [], [], 8, 128)
        self.assertEqual(kg._physical_col_view(*args).shape, (4, 8, 32))
        with mock.patch.object(kg, "_INT32_LIMIT", 32):
            self.assertIsNone(kg._physical_col_view(*args))
        trait = T.SumOps(acc=cutlass.Float32)
        with mock.patch.object(ct, "_INT32_LIMIT", 32):
            with self.assertRaisesRegex(AssertionError, r"want B \* R < 2\^31"):
                ct.reduce_batched_col_tile(trait, "rows", x, [torch.float32], 1)

    @parametrize(
        "launcher, rank, args",
        [
            ("reduce_col_tile", 3, (torch.float32,)),
            ("reduce_col_tile_2out", 3, ([torch.float32] * 2,)),
            ("reduce_batched_col_tile", 2, ([torch.float32], 1)),
        ],
        name_fn=lambda launcher, rank, args: launcher,
    )
    def test_launchers_check_rank(self, launcher, rank, args):
        x = torch.randn((4,) * rank, device="cuda")
        fn, trait = getattr(ct, launcher), T.SumOps(acc=cutlass.Float32)
        with self.assertRaisesRegex(AssertionError, "want [23]D x"):
            fn(trait, "badrank", x, *args)


instantiate_parametrized_tests(TestKernelColTile)
instantiate_parametrized_tests(TestColTileHost)

if __name__ == "__main__":
    run_tests()
