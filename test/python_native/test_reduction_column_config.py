# Owner(s): ["module: dsl-native-ops"]

import sys
import unittest
from unittest.mock import patch

import torch
from torch import _native as native
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUTEDSL,
    TestCase,
)


if not TEST_CUTEDSL:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("CuTeDSL not available")

import cutlass

from torch._native.ops.reductions import (
    kernel_coltile as ct,
    kernel_general as kg,
    traits as T,
)


# Views select() builds, each with the rule it takes without explicit geometry.
_VIEWS = {
    "default": ({}, "default"),
    "b200": ({"cc": (10, 0), "columns": 4095}, "b200_ragged_c4095"),
    "sm107": ({"cc": (10, 7)}, "rubin_sum_c4096_tiled"),
}


class TestColumnConfig(TestCase):
    hw_classification = HardwareClassification.CUDA

    def select(self, dtype=torch.float32, size=256, columns=4096, **kwargs):
        itemsize = dtype.itemsize
        args = dict(
            cc=(10, 3),
            dtype=dtype,
            rows=(size << 20) // (columns * itemsize),
            columns=columns,
            batches=1,
            nfields=1,
            trait_type=T.SumOps,
            itemsize=itemsize,
            acc_bits=32,
            nouts=1,
            welford_outs=0,
            alignment=16,
            out_dtypes=(torch.float32,),
        )
        args.update(kwargs)
        return ct.select_col_config(**args)

    def test_default_partitions_fit_grid_y(self):
        """Cap default partitions at grid y's 65535 blocks."""
        # Two lanes would want 65536 chunks.
        cfg = self.select(rows=1 << 20, columns=2)
        self.assertLessEqual(cfg.npar, 65535)

    def test_explicit_geometry_is_validated(self):
        """Reject explicit partitions past Int32 partial indices or grid y, nonpositive
        sizes, and vec that does not divide the columns; past 2^31 outputs even one
        partition overflows."""
        with self.assertRaisesRegex(ValueError, "overflows Int32"):
            self.select(rows=1 << 20, columns=1 << 20, npar=2048)
        with self.assertRaisesRegex(ValueError, "grid y"):
            self.select(rows=1 << 20, columns=2, npar=65536)
        with self.assertRaisesRegex(ValueError, "npar must be positive"):
            self.select(npar=0)
        with self.assertRaisesRegex(ValueError, "vec must be positive"):
            self.select(vec=0)
        with self.assertRaisesRegex(ValueError, "vec must divide the column count"):
            self.select(columns=30, vec=4)
        for npar in (None, 1):
            with self.assertRaisesRegex(ValueError, "overflow Int32"):
                self.select(rows=4, columns=1 << 31, npar=npar)

    @parametrize(
        "view,kwargs,geometry",
        [
            ("default", {"threads_per_block": 128}, (128, 4)),
            ("default", {"npar": 7}, (7, 4)),
            # Two-wide vectors make 2048 lanes, which want 64 partitions, not 128.
            ("default", {"vec": 2}, (64, 2)),
            # B200's anchors are odd widths, so they take no wider vectors.
            ("b200", {"threads_per_block": 128}, (33, 1)),
            ("b200", {"npar": 7}, (7, 1)),
            ("b200", {"vec": 1}, (33, 1)),
            ("sm107", {"threads_per_block": 128}, (128, 4)),
            ("sm107", {"npar": 7}, (7, 4)),
            ("sm107", {"vec": 2}, (64, 2)),
        ],
        name_fn=lambda view, kw, g: view + "".join(f"_{k}_{v}" for k, v in kw.items()),
    )
    def test_explicit_geometry_disables_policy(self, view, kwargs, geometry):
        """Honor explicit stage-one geometry without rewriting it through the rule the
        view takes otherwise, and derive the free field from the explicit one."""
        anchor, rule = _VIEWS[view]
        self.assertEqual(self.select(**anchor).rule, rule)
        cfg = self.select(**anchor, **kwargs)
        self.assertEqual(cfg.rule, "explicit")
        self.assertEqual((cfg.npar, cfg.vec), geometry)
        for key, value in kwargs.items():
            self.assertEqual(getattr(cfg, key), value)
        self.assertEqual(cfg.partial_layout, "column")
        self.assertEqual(cfg.combine_columns, 0)

    def test_physical_col_view_declines(self):
        """Decline unaligned pitches, unpacked batches, reduced axes out of logical
        order and batched rows past Int32."""

        def view(x, dims, count, num_o):
            pairs = kg._ti_pairs(x, kg._probe(x, dims))
            return kg._physical_col_view(x, dims, *pairs, count, num_o)

        base = torch.empty(64, 68, device="meta")
        self.assertEqual(view(base[:, :64], {0}, 64, 64).stride(), (68, 1))
        unaligned = torch.empty(64, 66, device="meta")[:, :64]
        self.assertIsNone(view(unaligned, {0}, 64, 64))
        unpacked = torch.empty(4, 10, 32, device="meta")[:, :8]
        self.assertIsNone(view(unpacked, {1}, 8, 128))
        # Reduced (6, 5) axes at strides (32, 192) merge in storage order.
        permuted = torch.empty(5, 6, 32, device="meta").permute(1, 0, 2)
        self.assertIsNone(view(permuted, {0, 1}, 30, 32))
        # Stage one indexes batched rows as batch * R + row in Int32.
        packed = torch.empty(2, 2**30 - 1, 2, device="meta")
        self.assertEqual(view(packed, {1}, 2**30 - 1, 4).dim(), 3)
        wide = torch.empty(2, 2**30, 2, device="meta")
        self.assertIsNone(view(wide, {1}, 2**30, 4))


class TestColumnCombine(TestCase):
    hw_classification = HardwareClassification.CUDA

    @parametrize(
        "dtype,op,pattern,width,shape",
        [
            (torch.float32, "sum", "signed", 8, (257, 260)),
            (torch.bfloat16, "argmax", "nonfinite_ties", 32, (257, 260)),
            (torch.float32, "var_mean", "offset", 32, (257, 260)),
            (torch.float32, "var_mean", "offset", 16, (3, 257, 36)),
            (torch.float32, "mean", "signed", 16, (3, 257, 36)),
            # Chunks of two rows leave the last two of seven partitions empty.
            (torch.float32, "var_mean", "offset", 8, (9, 260)),
        ],
    )
    def test_tiled_combine_numerics(self, device, dtype, op, pattern, width, shape):
        """Validate one-field, indexed, multi-output, and batched tiled combines."""
        x = torch.randn(shape, device=device, dtype=dtype)
        if pattern == "offset":
            x.mul_(8).add_(1024)
        elif pattern == "nonfinite_ties":
            x.fill_(0)
            x[0, :] = 1
            x[-1, :] = 1
            x[0, 0] = float("nan")
            x[1, 1] = float("inf")
            x[2, 2] = -float("inf")
        traits = {
            "sum": T.SumOps,
            "argmax": T.ArgMaxOps,
            "var_mean": T.VarMeanOps,
            "mean": T.MeanOps,
        }
        kw = {"correction": 0} if op == "var_mean" else {}
        trait = traits[op](acc=cutlass.Float32, **kw)
        nouts = 2 if op == "var_mean" else 1
        odt = torch.int64 if op == "argmax" else torch.float32
        cfg = ct.ColConfig("test_tiled", 64, 7, 4, "partition", width)
        key = f"test_tiled_{op}_{width}_{len(shape)}d"

        def reference():
            with (
                native._unconditional_masked(),
                torch.backends.python_native.cutedsl.disabled(),
            ):
                ref_kw = dict(kw, dtype=torch.float32) if op == "sum" else kw
                result = getattr(torch, op)(x, dim=x.dim() - 2, **ref_kw)
            return tuple(result) if nouts == 2 else (result,)

        def run():
            return ct._reduce_col_tile(
                trait,
                key,
                x,
                [odt] * nouts,
                nouts,
                None,
                None,
                None,
            )

        tol = 0 if op == "argmax" else 0.016 if dtype == torch.bfloat16 else 1e-4
        with patch.object(ct, "select_col_config", return_value=cfg):
            got = run()
        # Stage-two keys end with PartialColumns.cache_sig, (tile_columns, nouts).
        tiled = {k[-2] for k in ct._CACHE if k[:2] == ("coltile2t", key)}
        self.assertEqual(tiled, {width})
        self.assertEqual(got, reference(), rtol=tol, atol=tol, equal_nan=True)

    @parametrize(
        "layout,width,error",
        [
            ("column", 8, "partition-major"),
            ("rows", 0, "unknown partial layout"),
            ("partition", 64, "combine_columns must be"),
        ],
        name_fn=lambda layout, width, error: f"{layout}_{width}",
    )
    def test_invalid_partial_config(self, device, layout, width, error):
        """Reject partial configs before stage one launches."""
        x = torch.randn((257, 260), device=device)
        cfg = ct.ColConfig("test_tiled", 64, 7, 4, layout, width)
        trait = T.SumOps(acc=cutlass.Float32)
        args = (trait, "tiled_layout", x, [torch.float32], 1, None, None, None)
        with patch.object(ct, "select_col_config", return_value=cfg):
            with self.assertRaisesRegex(ValueError, error):
                ct._reduce_col_tile(*args)
        self.assertFalse(any(k[:2] == ("coltile", "tiled_layout") for k in ct._CACHE))

    def test_column_unroll_reaches_stage_one(self, device):
        """Pass the config's unroll through to the stage-one tile."""
        x = torch.randn((257, 260), device=device)
        cfg = ct.ColConfig("test_unroll", 64, 7, 4, "column", 0, 2)
        trait = T.SumOps(acc=cutlass.Float32)
        args = (trait, "unroll", x, [torch.float32], 1, None, None, None)
        with patch.object(ct, "select_col_config", return_value=cfg):
            got = ct._reduce_col_tile(*args)[0]
        # Six key fields precede TileReduce.cache_sig, whose unroll is its seventh field.
        unrolls = {k[12] for k in ct._CACHE if k[:2] == ("coltile", "unroll")}
        self.assertEqual(unrolls, {2})
        with (
            native._unconditional_masked(),
            torch.backends.python_native.cutedsl.disabled(),
        ):
            expected = x.sum(0)
        self.assertEqual(got, expected, atol=1e-4, rtol=1e-4)

    @parametrize("columns", [16, 64])
    def test_inner_tree_order_hands_off(self, device, columns):
        """inner_tree configs serve views the ordered kernel accepts, which takes the
        default looped mapping at 64 columns, and fall back to coltile when it declines
        views narrower than a warp."""
        x = torch.randn((257, columns), device=device)
        cfg = ct.ColConfig("test_order", 64, 7, 4, "column", kernel_order="inner_tree")
        trait = T.SumOps(acc=cutlass.Float32)
        key = f"order_{columns}"
        args = (trait, key, x, [torch.float32], 1, None, None, None)
        # No policy picks inner_tree at these shapes, so patch one in.
        with patch.object(ct, "select_col_config", return_value=cfg):
            got = ct._reduce_col_tile(*args)[0]
        tags = {k[0] for k in ct._CACHE if k[1] == key} & {"ordered_col", "coltile"}
        self.assertEqual(tags, {"ordered_col" if columns >= 32 else "coltile"})
        if columns >= 32:
            # The ordered kernel replays the indexed tree bit for bit.
            red, kept = [(257, columns)], [(columns, 1)]
            ref = kg._try_indexed_itree(
                trait, key + "_ref", x, red, kept, columns, 257, [torch.float32], 1
            )
            self.assertEqual(got.view(torch.uint8), ref[0].view(torch.uint8))
        else:
            with (
                native._unconditional_masked(),
                torch.backends.python_native.cutedsl.disabled(),
            ):
                expected = x.sum(0)
            self.assertEqual(got, expected, atol=1e-4, rtol=1e-4)


instantiate_parametrized_tests(TestColumnConfig)
instantiate_device_type_tests(TestColumnCombine, globals(), only_for=("cuda",))


if __name__ == "__main__":
    run_tests()
