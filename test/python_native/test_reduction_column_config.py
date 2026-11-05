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

from torch._native.ops.reductions import kernel_coltile as ct, traits as T


# Welford outputs of the traits the test keys name.
_WELFORD_OUTS = {"var0": 1, "var1": 1, "varmean0": 2}


class TestColumnConfig(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def select(self, dtype=torch.float32, size=256, columns=4096, **kwargs):
        itemsize = dtype.itemsize
        args = dict(
            cc=(10, 3),
            dtype=dtype,
            rows=(size << 20) // (columns * itemsize),
            columns=columns,
            batches=1,
            nfields=1,
            trait_key="sum",
            itemsize=itemsize,
            acc_bits=32,
            nouts=1,
            alignment=16,
        )
        args.update(kwargs)
        args.setdefault("welford_outs", _WELFORD_OUTS.get(args["trait_key"], 0))
        return ct.select_col_config(**args)

    def test_default_geometry(self):
        """Fill about _TARGET_LANES threads, narrowing vec before chunks get short."""
        cfg = self.select(size=16, columns=256)
        self.assertEqual(cfg.rule, "default")
        self.assertEqual(cfg.threads_per_block, 32)
        # Wider vectors would leave 16-row chunks; 256 lanes take 512 chunks of 32 rows.
        self.assertEqual((cfg.npar, cfg.vec), (512, 1))
        self.assertEqual(cfg.partial_layout, "column")
        # Tall views keep full-width vectors: 1024 lanes split 128 ways.
        cfg = self.select(size=256, columns=4096)
        self.assertEqual((cfg.npar, cfg.vec), (128, 4))
        # 16 columns keep vec 1 for lanes.
        cfg = self.select(size=16, columns=16)
        self.assertEqual((cfg.npar, cfg.vec), (8192, 1))
        # Short views reduce in one stage.
        cfg = self.select(rows=64, columns=4096)
        self.assertEqual((cfg.npar, cfg.vec), (1, 2))
        # Two lanes would want 65536 chunks; grid y holds 65535.
        cfg = self.select(rows=1 << 20, columns=2)
        self.assertEqual((cfg.npar, cfg.vec), (65535, 1))
        # Partition-major partials start at B * C >= 8192 kept outputs.
        for kw in ({"columns": 8192}, {"columns": 1024, "batches": 8}):
            self.assertEqual(self.select(**kw).partial_layout, "partition")

    def test_explicit_partitions_are_bounded(self):
        """Reject explicit partitions past Int32 partial indices or grid y."""
        with self.assertRaisesRegex(ValueError, "overflows Int32"):
            self.select(rows=1 << 20, columns=1 << 20, npar=2048)
        with self.assertRaisesRegex(ValueError, "grid y"):
            self.select(rows=1 << 20, columns=2, npar=65536)

    @parametrize(
        "dtype,size,columns,key,fields,rule,expected",
        [
            (
                torch.float32,
                256,
                256,
                "sum",
                1,
                "rubin_c256",
                (4096, 4, "column", 0, "linear"),
            ),
            (
                torch.bfloat16,
                2048,
                1024,
                "argmaxi32",
                2,
                "rubin_c1024",
                (4096, 8, "column", 0, "inner_tree"),
            ),
            (
                torch.float32,
                256,
                1024,
                "sum",
                1,
                "rubin_c1024",
                (1024, 8, "column", 0, "linear"),
            ),
            (
                torch.float32,
                256,
                4096,
                "mean",
                1,
                "rubin_mean_c4096_tiled",
                (256, 8, "partition", 16, "linear"),
            ),
            (
                torch.bfloat16,
                256,
                4096,
                "sum",
                1,
                "rubin_sum_c4096_tiled",
                (256, 8, "partition", 16, "linear"),
            ),
            (
                torch.float32,
                2048,
                4096,
                "sum",
                1,
                "rubin_sum_c4096_tiled",
                (1024, 4, "partition", 8, "linear"),
            ),
            (
                torch.bfloat16,
                2048,
                4096,
                "mean",
                1,
                "rubin_mean_c4096_tiled",
                (2048, 8, "partition", 8, "linear"),
            ),
            (
                torch.bfloat16,
                2048,
                4096,
                "var0",
                3,
                "rubin_var_c4096",
                (2048, 8, "column", 0, "inner_tree"),
            ),
            (
                torch.float32,
                256,
                4096,
                "var0",
                3,
                "rubin_var_c4096",
                (256, 8, "column", 0, "linear"),
            ),
            # rubin_var_c4096 runs the ordered kernel for bf16 or 2 GiB views.
            (
                torch.bfloat16,
                256,
                4096,
                "var0",
                3,
                "rubin_var_c4096",
                (256, 8, "column", 0, "inner_tree"),
            ),
            (
                torch.float32,
                2048,
                4096,
                "var0",
                3,
                "rubin_var_c4096",
                (1024, 8, "column", 0, "inner_tree"),
            ),
            (
                torch.bfloat16,
                2048,
                4096,
                "argmaxi32",
                2,
                "rubin_argmax_c4096_tree",
                (2048, 4, "partition", 8, "inner_tree"),
            ),
            (
                torch.float32,
                2048,
                4096,
                "argmaxi32",
                2,
                "rubin_argmax_c4096_tiled",
                (1024, 4, "partition", 8, "linear"),
            ),
            (
                torch.bfloat16,
                256,
                4096,
                "argmaxi32",
                2,
                "rubin_argmax_c4096_tiled",
                (512, 4, "partition", 8, "linear"),
            ),
        ],
        name_fn=lambda dtype, size, columns, key, fields, rule, expected: (
            f"{rule}_{str(dtype)[6:]}_{size}mib"
        ),
    )
    def test_sm107_unordered_profiles(
        self, dtype, size, columns, key, fields, rule, expected
    ):
        """Pin each SM107 unordered-column launch family at its measured size anchors."""
        cfg = self.select(
            dtype,
            size,
            columns,
            cc=(10, 7),
            trait_key=key,
            nfields=fields,
        )
        got = (cfg.npar, cfg.vec, cfg.partial_layout, cfg.combine_columns)
        self.assertEqual(got + (cfg.kernel_order,), expected)
        self.assertEqual(cfg.rule, rule)
        self.assertEqual(cfg.threads_per_block, 64)

    @parametrize(
        "dtype,key,rows,columns,partials,fields,outputs,expected",
        [
            (
                torch.float32,
                "sum",
                65536,
                256,
                8,
                1,
                (torch.float32,),
                (16, 16),
            ),
            (
                torch.bfloat16,
                "argmaxi32",
                262144,
                4096,
                32,
                2,
                (torch.int64,),
                (64, 32),
            ),
            # The 256 MiB mean_anchor arm, the 65536 x 1024 rule, the c256 Welford and
            # bf16 minimums, and a column_anchor seed for a simple trait.
            (torch.float32, "mean", 262144, 256, 32, 1, (torch.float32,), (32, 16)),
            (torch.float32, "var0", 65536, 1024, 8, 3, (torch.float32,), (64, 16)),
            # One-output Welford traits share the var0 entry.
            (torch.float32, "var1", 65536, 1024, 8, 3, (torch.float32,), (64, 16)),
            (torch.float32, "var0", 65536, 256, 8, 3, (torch.float32,), (0, 16)),
            (torch.bfloat16, "sum", 131072, 256, 16, 1, (torch.float32,), (8, 16)),
            (torch.float32, "mean", 524288, 1024, 64, 1, (torch.float32,), (32, 32)),
            # The 65536 x 1024 rule skips argmax; Welford anchors; mean_anchor's 2 GiB and
            # bf16 64 MiB arms.
            (torch.float32, "argmaxi32", 65536, 1024, 8, 2, (torch.int64,), (64, 32)),
            (torch.bfloat16, "var0", 8192, 4096, 1, 3, (torch.bfloat16,), (0, 32)),
            (torch.bfloat16, "var0", 8192, 1024, 1, 3, (torch.bfloat16,), (0, 16)),
            (torch.float32, "mean", 2097152, 256, 256, 1, (torch.float32,), (32, 32)),
            (torch.bfloat16, "amax", 131072, 256, 16, 1, (torch.bfloat16,), (32, 32)),
            (torch.bfloat16, "mean", 32768, 1024, 4, 1, (torch.float32,), (32, 32)),
        ],
    )
    def test_sm107_ordered_profiles(
        self, dtype, key, rows, columns, partials, fields, outputs, expected
    ):
        """Pin SM107 ordered mappings."""
        cfg = self._sm107_ordered(dtype, key, rows, columns, partials, fields, outputs)
        self.assertEqual((cfg.min_blocks, cfg.tile_columns), expected)
        self.assertEqual((cfg.worker_warps, cfg.full_tiles), (16, True))

    @parametrize(
        "dtype,key,rows,columns,partials,fields,outputs",
        [
            (torch.float32, "sum", 65536, 256, 8, 1, (torch.bfloat16,)),
            (torch.bfloat16, "argmaxi32", 262144, 4096, 32, 2, (torch.int32,)),
            (torch.bfloat16, "amax", 131072, 256, 16, 1, (torch.float32,)),
            (torch.bfloat16, "var0", 8192, 4096, 1, 3, (torch.float32,)),
        ],
    )
    def test_sm107_ordered_needs_measured_outputs(
        self, dtype, key, rows, columns, partials, fields, outputs
    ):
        """Map no SM107 ordered view whose outputs differ from _sm107_outputs."""
        cfg = self._sm107_ordered(dtype, key, rows, columns, partials, fields, outputs)
        self.assertIsNone(cfg)

    def _sm107_ordered(self, dtype, key, rows, columns, partials, fields, outputs):
        return ct.select_ordered_col_config(
            (10, 7),
            dtype,
            key,
            columns,
            1,
            partials,
            nfields=fields,
            nouts=len(outputs),
            welford_outs=_WELFORD_OUTS.get(key, 0),
            acc_bits=32,
            out_dtypes=outputs,
            rows=rows,
            full_tiles=True,
        )

    @parametrize(
        "dtype,size,columns,key,fields,order",
        [
            (torch.float32, 64, 256, "sum", 1, "inner_tree"),
            (torch.bfloat16, 16, 256, "var0", 3, "inner_tree"),
            (torch.float32, 16, 4096, "sum", 1, "inner_tree"),
            (torch.bfloat16, 64, 4096, "mean", 1, "inner_tree"),
            (torch.bfloat16, 64, 4096, "amax", 1, "linear"),
            (torch.float32, 2048, 1024, "sum", 1, "inner_tree"),
            # 8192-row bf16 Welford views.
            (torch.bfloat16, 16, 1024, "var0", 3, "inner_tree"),
            (torch.bfloat16, 64, 4096, "var0", 3, "inner_tree"),
            # 16384-row fp32 plans exceed reduce_ordered_col's unroll cap.
            (torch.float32, 16, 256, "var0", 3, "linear"),
            (torch.float32, 64, 1024, "var0", 3, "linear"),
            (torch.float32, 256, 4096, "var0", 3, "linear"),
        ],
    )
    def test_sm107_hand_offs(self, dtype, size, columns, key, fields, order):
        """Pin which SM107 columns hand off to the ordered kernel."""
        cfg = self.select(
            dtype, size, columns, cc=(10, 7), trait_key=key, nfields=fields
        )
        self.assertEqual(cfg.kernel_order, order)

    @parametrize(
        "dtype,size,columns,key,fields,out",
        [
            (torch.bfloat16, 64, 256, "sum", 1, torch.bfloat16),
            (torch.bfloat16, 64, 1024, "mean", 1, torch.bfloat16),
            (torch.bfloat16, 2048, 4096, "var0", 3, torch.float32),
        ],
    )
    def test_sm107_hand_offs_need_ordered_outputs(
        self, dtype, size, columns, key, fields, out
    ):
        """Outputs the ordered kernel cannot store keep SM107 columns on the linear kernel."""
        args = dict(cc=(10, 7), trait_key=key, nfields=fields)
        natural = torch.float32 if fields == 1 else dtype
        cfg = self.select(dtype, size, columns, out_dtypes=(natural,), **args)
        self.assertEqual(cfg.kernel_order, "inner_tree")
        cfg = self.select(dtype, size, columns, out_dtypes=(out,), **args)
        self.assertEqual(cfg.kernel_order, "linear")

    @parametrize(
        "updates",
        [
            {"size": 255},
            {"dtype": torch.float16},
            {"acc_bits": 64},
            {"nouts": 2},
            {"nfields": 2},
            {"batches": 2},
            {"alignment": 8},
        ],
    )
    def test_sm107_column_guards(self, updates):
        """Keep a 256 MiB SM107 anchor that breaks one guard on generic policies."""
        self.assertEqual(self.select(cc=(10, 7)).rule, "rubin_sum_c4096_tiled")
        self.assertEqual(self.select(cc=(10, 7), **updates), self.select(**updates))

    def test_sm107_ordered_acc_bits_guard(self):
        """Decline SM107 ordered mappings for 64-bit accumulators."""
        cfg = ct.select_ordered_col_config(
            (10, 7),
            torch.float32,
            "mean",
            1024,
            1,
            64,
            nfields=1,
            nouts=1,
            welford_outs=0,
            acc_bits=64,
            out_dtypes=(torch.float32,),
            rows=524288,
            full_tiles=True,
        )
        self.assertIsNone(cfg)

    @parametrize("key,nouts", [("var0", 1), ("varmean0", 2)])
    def test_sm107_ordered_welford_outputs(self, key, nouts):
        """Map only one-output Welford columns at the SM107 Welford anchors."""
        cfg = ct.select_ordered_col_config(
            (10, 7),
            torch.bfloat16,
            key,
            4096,
            1,
            1,
            nfields=3,
            nouts=nouts,
            welford_outs=nouts,
            acc_bits=32,
            out_dtypes=(torch.bfloat16,) * nouts,
            rows=8192,
            full_tiles=True,
        )
        selected = ct.OrderedColConfig(0, full_tiles=True, out_dtypes=(torch.bfloat16,))
        self.assertEqual(cfg, selected if nouts == 1 else None)

    @parametrize("kwargs", [{"threads_per_block": 128}, {"npar": 7}, {"vec": 2}])
    @parametrize("cc", [(10, 3), (10, 7)])
    def test_explicit_geometry_disables_policy(self, cc, kwargs):
        """Honor explicit column geometry without rewriting it through policy, even at
        SM107's rubin_sum_c4096_tiled anchor."""
        cfg = self.select(cc=cc, **kwargs)
        self.assertEqual(cfg.rule, "explicit")
        for key, value in kwargs.items():
            self.assertEqual(getattr(cfg, key), value)
        self.assertEqual(cfg.partial_layout, "column")
        self.assertEqual(cfg.combine_columns, 0)

    @parametrize(
        "dtype,key,rows,columns,partials,full_tiles,expected",
        [
            # Looped views keep the default mapping, with or without full tiles.
            (torch.float32, "sum", 8192, 4096, 1, False, None),
            (torch.float16, "sum", 8192, 4096, 1, True, None),
            (torch.float32, "sum", 65536, 1024, 8, True, (0, 16)),
            (torch.float32, "sum", 32768, 257, 4, True, (0, 16)),
            (torch.float32, "sum", 131072, 257, 16, True, (0, 32)),
            (torch.bfloat16, "amax", 32768, 4096, 4, True, (0, 16)),
            (torch.bfloat16, "amax", 262144, 4096, 32, True, (0, 32)),
            (torch.bfloat16, "amin", 32768, 4096, 4, True, (0, 32)),
        ],
    )
    def test_b200_ordered_column_mappings(
        self, dtype, key, rows, columns, partials, full_tiles, expected
    ):
        """Pin B200 ordered column widths, which apply to any outputs, independent of
        SM107 guards."""
        for output in {dtype, torch.float32}:
            with self.subTest(output=output):
                cfg = ct.select_ordered_col_config(
                    (10, 0),
                    dtype,
                    key,
                    columns,
                    1,
                    partials,
                    nfields=1,
                    nouts=1,
                    welford_outs=0,
                    acc_bits=32,
                    out_dtypes=(output,),
                    rows=rows,
                    full_tiles=full_tiles,
                )
                if expected is None:
                    self.assertIsNone(cfg)
                    continue
                min_blocks, tile_columns = expected
                want = ct.OrderedColConfig(min_blocks, tile_columns=tile_columns)
                self.assertEqual(cfg, want._replace(full_tiles=full_tiles))

    def test_b200_ordered_column_mapping_guards(self):
        """Split views the fallback does not serve get None; multi-field split views keep
        full-warp tiles."""
        args = dict(
            cc=(10, 0),
            dtype=torch.float32,
            trait_key="sum",
            columns=1024,
            batches=1,
            partials=8,
            nfields=1,
            nouts=1,
            welford_outs=0,
            acc_bits=32,
            out_dtypes=(torch.float32,),
            rows=65536,
            full_tiles=True,
        )
        for updates in ({"batches": 2}, {"acc_bits": 64}, {"cc": (10, 3)}):
            with self.subTest(**updates):
                self.assertIsNone(ct.select_ordered_col_config(**(args | updates)))
        index = {"trait_key": "argmaxi32", "nfields": 2, "out_dtypes": (torch.int32,)}
        cfg = ct.select_ordered_col_config(**(args | index))
        want = ct.OrderedColConfig(0, tile_columns=ct.WARP, full_tiles=True)
        self.assertEqual(cfg, want)


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

    @parametrize("columns,served", [(256, True), (4095, True), (16, False)])
    def test_inner_tree_order_hands_off(self, device, columns, served):
        """inner_tree configs use the ordered kernel and fall back when it declines."""
        x = torch.randn((257, columns), device=device)
        cfg = ct.ColConfig("test_order", 64, 7, 4, "column", kernel_order="inner_tree")
        trait = T.SumOps(acc=cutlass.Float32)
        key = f"order_{columns}"
        args = (trait, key, x, [torch.float32], 1, None, None, None)
        with patch.object(ct, "select_col_config", return_value=cfg):
            got = ct._reduce_col_tile(*args)[0]
        tags = {k[0] for k in ct._CACHE if k[1] == key} & {"ordered_col", "coltile"}
        self.assertEqual(tags, {"ordered_col" if served else "coltile"})
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
