# Owner(s): ["module: dsl-native-ops"]
# Smoke tests for single-stage and split column reduction; OpInfo covers numerics.

import sys
import unittest
from unittest import mock

import torch
from torch.testing._internal.common_cuda import SM90OrLater, TEST_CUDA
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    largeTensorTest,
)
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
        real = ct.reduce_batched_col_tile
        with mock.patch.object(ct, "reduce_batched_col_tile", wraps=real) as served:
            got = kg.reduce_dim(
                T.ArgMaxOps(acc=cutlass.Float32), "batched_argmax", x, 1, torch.int32
            )
        self.assertEqual(served.call_count, 1)
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
        real = ct.reduce_batched_col_tile
        with mock.patch.object(ct, "reduce_batched_col_tile", wraps=real) as served:
            if nouts == 1:
                got = (kg.reduce_dim(T.SumOps(acc=cutlass.Float32), "gate", x, 1, f32),)
            else:
                two = T.VarMeanOps(correction=1, acc=cutlass.Float32)
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
        real = ct.reduce_col_tile_2out
        with mock.patch.object(ct, "reduce_col_tile_2out", wraps=real) as served:
            f32 = [torch.float32, torch.float32]
            got = kg.reduce_dim2(trait, f"var_mean_{layout}", x, dim, f32)
        self.assertEqual(served.call_count, int(routed))
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
        two_out, batched = ct.reduce_col_tile_2out, ct.reduce_batched_col_tile
        with (
            mock.patch.object(ct, "reduce_col_tile_2out", wraps=two_out) as two,
            mock.patch.object(ct, "reduce_batched_col_tile", wraps=batched) as bat,
        ):
            key = f"{op}dim_route"
            vals, idx = kg.reduce_dim2(trait, key, x, dim, [dtype, torch.int32])
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


def _sparse(x):
    return torch.where(x.abs() > 2, x, torch.zeros_like(x))


def _with_nans(x):
    # NaNs scattered along the reduced axis leave each column mixed.
    return torch.where(torch.rand_like(x) < 0.1, float("nan"), x)


_F32, _I32, _I64 = torch.float32, torch.int32, torch.int64


# Traits kernel_general routes to ordered columns: factory, outputs (None = input dtype),
# and an input transform that exercises the trait. The kernels store no bool, so all and
# any write int32 flags.
_ROUTED_TRAITS = {
    "sum": (lambda: T.SumOps(), [_F32], lambda x: x),
    "mean": (lambda: T.MeanOps(), [None], lambda x: x),
    "prod": (lambda: T.ProdOps(), [_F32], lambda x: 1 + 1e-3 * x),
    "nansum": (lambda: T.NanSumOps(), [_F32], _with_nans),
    "norm1": (lambda: T.NormOps(1), [_F32], lambda x: x),
    "norm2": (lambda: T.NormOps(2), [_F32], lambda x: x),
    "norm3": (lambda: T.NormOps(3), [_F32], lambda x: x),
    "amax": (lambda: T.AMaxOps(), [None], lambda x: x),
    "amin": (lambda: T.AMinOps(), [None], lambda x: x),
    "absmax": (lambda: T.AbsMaxOps(), [None], lambda x: x),
    "absmin": (lambda: T.AbsMinOps(), [None], lambda x: x),
    # Rare hits (|x| > 3.5 or |x| >= 3.9) leave columns mixed at R = 256 and R = 3000.
    "all": (lambda: T.AllOps(), [_I32], lambda x: (x.abs() < 3.9).float()),
    "any": (lambda: T.AnyOps(), [_I32], lambda x: (x.abs() > 3.5).float()),
    "count_nonzero": (lambda: T.CountNonzeroOps(), [_I64], _sparse),
    "argmax": (lambda: T.ArgMaxOps(), [_I64], lambda x: x),
    "argmin": (lambda: T.ArgMinOps(), [_I64], lambda x: x),
    "max_dim": (lambda: T.MaxDimOps(), [None, _I64], lambda x: x),
    "min_dim": (lambda: T.MinDimOps(), [None, _I64], lambda x: x),
    "aminmax": (lambda: T.AMinMaxOps(), [None, None], lambda x: x),
    "var": (lambda: T.WelfordOps(correction=1), [_F32], lambda x: x),
    "std": (lambda: T.WelfordOps(correction=1, take_sqrt=True), [_F32], lambda x: x),
    "var_mean": (lambda: T.VarMeanOps(correction=1), [_F32, _F32], lambda x: x),
}


class TestOrderedColRoute(TestCase):
    @parametrize(
        "nfields,shape,cc,routed",
        [
            (1, (16, 128, 128), (10, 0), True),
            (1, (16, 128, 127), (10, 0), False),
            (1, (16, 127, 128), (10, 0), False),
            (2, (128, 128, 128), (10, 0), True),
            (2, (127, 128, 128), (10, 0), False),
            (3, (256, 128, 128), (10, 0), True),
            (3, (255, 128, 128), (10, 0), False),
            (4, (256, 128, 128), (10, 0), True),
            (1, (16, 128, 128), (9, 0), False),
            (1, (16, 8192, 128), (10, 0), True),
            (1, (16, 8193, 128), (10, 0), False),
            (1, (128, 2048), (10, 0), True),
            (1, (127, 2048), (10, 0), False),
            (1, (128, 2047), (10, 0), False),
        ],
        name_fn=lambda nfields, shape, cc, routed: "_".join(
            [f"{nfields}f", "x".join(map(str, shape)), f"sm{cc[0]}{cc[1]}"]
        ),
    )
    def test_gate_boundaries(self, nfields, shape, cc, routed):
        view = torch.empty(shape, device="meta")
        self.assertEqual(kg._ordered_col_route_ok(view, nfields, cc), routed)

    @parametrize("dtype", [*_DTYPES, torch.float64, torch.int8])
    def test_gate_admits_measured_dtypes(self, dtype):
        view = torch.empty((16, 128, 128), device="meta", dtype=dtype)
        routed = kg._ordered_col_route_ok(view, 1, (10, 0))
        self.assertEqual(routed, dtype in _DTYPES)

    @parametrize(
        "field,error",
        [
            ("shape", "do not support"),
            ("vec_linear", "tree vector folds"),
            ("batches", "single planner batch"),
        ],
    )
    def test_constructor_invariants_raise(self, field, error):
        # reduce_ordered_col declines these plans first, so only direct callers reach them.
        from torch._native.ops.reductions import kernel_rowtile as rt

        trait = T.SumOps(acc=cutlass.Float32)
        default = rt._ITREE_ARCH["default"]
        with mock.patch.object(rt, "_itree_arch", return_value=default):
            plan = rt.trait_itree_plan(trait, 1024, 32, 4, stage=False)
        state = (plan.shape, plan.vec_linear, len(plan.batches))
        self.assertEqual(state, ("looped", False, 1))
        invalid = {"shape": "multirow", "vec_linear": True, "batches": plan.batches * 2}
        bad = plan._replace(**{field: invalid[field]})
        with self.assertRaisesRegex(ValueError, error):
            ct._OrderedColReduce(trait, bad, 1024, 32, 1)

    def test_row_offsets_widen_past_int32(self):
        from torch._native.ops.reductions import kernel_rowtile as rt

        trait = T.SumOps(acc=cutlass.Float32)
        default = rt._ITREE_ARCH["default"]
        with mock.patch.object(rt, "_itree_arch", return_value=default):
            plan = rt.trait_itree_plan(trait, 1024, 32, 4, stage=False)
        narrow = ct._OrderedColReduce(trait, plan, 1024, (1 << 21) - 1, 1)
        wide = ct._OrderedColReduce(trait, plan, 1024, 1 << 21, 1)
        self.assertEqual((narrow.wide, wide.wide), (False, True))


@unittest.skipUnless(TEST_CUDA and SM90OrLater, "requires Hopper or later")
class TestOrderedColDevice(TestCase):
    @parametrize(
        "op,dtype,shape",
        [
            ("sum", torch.float32, (2, 128, 33)),
            ("sum", torch.float32, (8000, 32)),
            ("sum", torch.float16, (2, 3000, 32)),
            ("sum", torch.bfloat16, (2, 3000, 32)),
            ("mean", torch.float32, (2, 3000, 32)),
            ("argmax", torch.float32, (4, 2048, 64)),
            ("var_mean", torch.bfloat16, (2, 2048, 40)),
            ("argmax", torch.float32, (8000, 32)),
            ("var_mean", torch.float32, (2, 8000, 32)),
        ],
        name_fn=lambda op, dtype, shape: (
            f"{op}_{str(dtype)[6:]}_{len(shape)}d_{shape[-2]}"
        ),
    )
    def test_direct_plan_matches_the_indexed_tree(self, device, op, dtype, shape):
        """Preserve each row's inner-tree DAG across plan shapes, fields, and ranks."""
        x = torch.randn(shape, device=device).to(dtype)
        rows, columns = x.shape[-2:]
        batches = x.numel() // (rows * columns)
        kept = [(columns, 1)] + ([(batches, rows * columns)] if x.dim() == 3 else [])
        trait, outputs = {
            "sum": (T.SumOps(acc=cutlass.Float32), [torch.float32]),
            "mean": (T.MeanOps(acc=cutlass.Float32), [torch.float32]),
            "argmax": (T.ArgMaxOps(acc=cutlass.Float32), [torch.int64]),
            "var_mean": (T.VarMeanOps(correction=0), [torch.float32] * 2),
        }[op]
        nouts = len(outputs)
        got = ct.reduce_ordered_col(trait, f"ordered_{op}", x, outputs, nouts)
        reference = kg._try_indexed_itree(
            trait,
            f"ordered_{op}_reference",
            x,
            [(rows, columns)],
            kept,
            batches * columns,
            rows,
            outputs,
            nouts,
        )
        self.assertIsNotNone(got)
        self.assertIsNotNone(reference)
        for a, b in zip(got, reference, strict=True):
            self.assertEqual(a.flatten().view(torch.uint8), b.view(torch.uint8))

    @parametrize("pattern", ["negative_zero", "ties"])
    def test_special_values_match_the_indexed_tree(self, device, pattern):
        """Keep signed-zero sums and argmax ties bitwise equal to the indexed tree."""
        rows, columns = 2048, 32
        if pattern == "negative_zero":
            x = torch.full((rows, columns), -0.0, device=device)
            trait, outputs = T.SumOps(), [torch.float32]
        else:
            x = torch.ones(rows, columns, device=device)
            trait, outputs = T.ArgMaxOps(), [torch.int64]
        key = f"special_{pattern}"
        got = ct.reduce_ordered_col(trait, key, x, outputs, 1)
        reference = kg._try_indexed_itree(
            trait,
            key + "_reference",
            x,
            [(rows, columns)],
            [(columns, 1)],
            columns,
            rows,
            outputs,
            1,
        )
        self.assertEqual(got[0].view(torch.uint8), reference[0].view(torch.uint8))

    @parametrize("op", ["sum", "argmax"])
    @parametrize("rows,served", [(8192, True), (8193, False), (16384, False)])
    def test_replay_is_bounded(self, device, op, rows, served):
        """Decline looped plans whose static replay passes MAX_UNROLL."""
        x = torch.randn(rows, 32, device=device)
        trait = T.SumOps() if op == "sum" else T.ArgMaxOps()
        output = torch.float32 if op == "sum" else torch.int64
        got = ct.reduce_ordered_col(trait, "bounded_" + op, x, [output], 1)
        self.assertEqual(got is not None, served)

    def test_batch_and_column_counts_share_one_kernel(self, device):
        trait = T.SumOps(acc=cutlass.Float32)
        for batches, columns in ((2, 33), (3, 33), (2, 64), (1, 1000)):
            x = torch.randn(batches, 128, columns, device=device)
            got = ct.reduce_ordered_col(trait, "ordered_reuse", x, [torch.float32], 1)
            self.assertEqual(got[0], x.sum(dim=1), atol=1e-4, rtol=1e-4)
        keys = [k for k in ct._CACHE if k[:2] == ("ordered_col", "ordered_reuse")]
        self.assertEqual(len(keys), 1)

    @largeTensorTest("10GB", device="cuda")
    def test_wide_offsets(self, device):
        # The last row starts (R - 1) * C = 2**31 elements in, past Int32 offsets.
        x = torch.randn(129, 1 << 24, device=device, dtype=torch.bfloat16)
        trait = T.SumOps(acc=cutlass.Float32)
        (got,) = ct.reduce_ordered_col(trait, "ordered_wide", x, [torch.float32], 1)
        want = x.sum(dim=0, dtype=torch.float32)
        self.assertEqual(got, want, atol=1e-2, rtol=1e-3)

    def _reduce(self, device, trait, key, x, dim, outputs, *, route):
        """Inner-tree _reduce with the column route on (this device) or off; also return
        what each reduce_ordered_col call returned."""
        real, served = ct.reduce_ordered_col, []

        def record(*args, **kwargs):
            served.append(real(*args, **kwargs))
            return served[-1]

        ccs = (torch.cuda.get_device_capability(device),) if route else ()
        with (
            mock.patch.object(kg, "_ORDERED_COL_CCS", ccs),
            mock.patch.object(ct, "reduce_ordered_col", record),
        ):
            got = kg._reduce(
                trait, key, x, dim, outputs, len(outputs), order="inner_tree"
            )
        return got, served

    def _assert_bitwise(self, got, want):
        for a, b in zip(got, want, strict=True):
            a, b = a.contiguous().flatten(), b.contiguous().flatten()
            self.assertEqual(a.view(torch.uint8), b.view(torch.uint8))

    @parametrize("layout", ["3d", "offset", "2d", "transposed"])
    def test_inner_tree_columns_route_to_ordered_columns(self, device, layout):
        """Serve dense column views through the kernel, bitwise equal to the indexed tree."""
        make, dim = {
            "3d": (lambda: torch.randn(16, 128, 128, device=device), 1),
            "offset": (lambda: torch.randn(17, 128, 128, device=device)[1:], 1),
            "2d": (lambda: torch.randn(128, 4096, device=device), 0),
            "transposed": (lambda: torch.randn(4096, 2048, device=device).t(), 1),
        }[layout]
        x = make()
        trait, outputs = T.SumOps(acc=cutlass.Float32), [torch.float32]
        key = f"route_{layout}"
        got, served = self._reduce(device, trait, key, x, dim, outputs, route=True)
        self.assertEqual(len(served), 1)
        self.assertIsNotNone(served[0])
        want, _ = self._reduce(device, trait, key, x, dim, outputs, route=False)
        self._assert_bitwise(got, want)

    @parametrize("dtype", _DTYPES)
    @parametrize("name", list(_ROUTED_TRAITS))
    def test_routed_traits_match_the_indexed_tree(self, device, name, dtype):
        """Every trait the gate admits stays bitwise equal to the indexed tree."""
        make, outputs, prepare = _ROUTED_TRAITS[name]
        # 32768 outputs clear the widest (three-field) gate threshold.
        x = prepare(torch.randn(32, 256, 1024, device=device)).to(dtype)
        outputs = [dtype if out is None else out for out in outputs]
        key = f"parity_{name}"
        got, served = self._reduce(device, make(), key, x, 1, outputs, route=True)
        self.assertEqual(len(served), 1)
        self.assertIsNotNone(served[0])
        want, _ = self._reduce(device, make(), key, x, 1, outputs, route=False)
        if name in ("all", "any"):
            self.assertEqual(want[0].unique().numel(), 2, "uniform flags test nothing")
        self._assert_bitwise(got, want)

    @parametrize("dtype", _DTYPES)
    @parametrize("name", list(_ROUTED_TRAITS))
    def test_routed_traits_with_padded_warps(self, device, name, dtype):
        """Padded warps and masked leaves keep every trait bitwise equal."""
        make, outputs, prepare = _ROUTED_TRAITS[name]
        rows, columns = 3000, 64
        x = prepare(torch.randn(2, rows, columns, device=device)).to(dtype)
        outputs = [dtype if out is None else out for out in outputs]
        nouts, key = len(outputs), f"padded_{name}"
        got = ct.reduce_ordered_col(make(), key, x, outputs, nouts)
        kept = [(columns, 1), (2, rows * columns)]
        want = kg._try_indexed_itree(
            make(),
            key + "_ref",
            x,
            [(rows, columns)],
            kept,
            2 * columns,
            rows,
            outputs,
            nouts,
        )
        self.assertIsNotNone(got)
        if name in ("all", "any"):
            self.assertEqual(want[0].unique().numel(), 2, "uniform flags test nothing")
        self._assert_bitwise(got, want)

    def test_declined_ordered_columns_fall_back_to_the_indexed_tree(self, device):
        # The gate admits 4096 outputs, but the kernel declines fewer than 32 columns.
        x = torch.randn(256, 128, 16, device=device)
        trait = T.SumOps(acc=cutlass.Float32)
        with mock.patch.object(
            kg, "_try_indexed_itree", wraps=kg._try_indexed_itree
        ) as fallback:
            (got,), served = self._reduce(
                device, trait, "fallback", x, 1, [torch.float32], route=True
            )
        self.assertEqual(served, [None])
        fallback.assert_called_once()
        self.assertEqual(got, x.sum(dim=1), atol=1e-4, rtol=1e-4)


instantiate_device_type_tests(TestOrderedColDevice, globals(), only_for="cuda")
instantiate_parametrized_tests(TestOrderedColRoute)

instantiate_parametrized_tests(TestKernelColTile)
instantiate_parametrized_tests(TestColTileHost)

if __name__ == "__main__":
    run_tests()
