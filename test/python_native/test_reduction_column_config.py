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
            contiguous=True,
        )
        args.update(kwargs)
        return ct.select_col_config(**args)

    def test_default_geometry(self):
        """Size column geometry from the reduced and column extents by default."""
        cfg = self.select(size=16, columns=256)
        self.assertEqual(cfg.rule, "default")
        self.assertEqual(cfg.threads_per_block, 32)
        self.assertEqual(cfg.npar, 256)
        self.assertEqual(cfg.vec, 4)
        self.assertEqual(cfg.partial_layout, "column")
        # Half-width elements double the reduced rows in the same bytes.
        cfg = self.select(dtype=torch.float16, size=16, columns=256)
        self.assertEqual(cfg.npar, 512)
        self.assertEqual(cfg.vec, 4)
        # Partition-major partials start at B * C >= 8192 kept outputs.
        for kw in ({"columns": 8192}, {"columns": 1024, "batches": 8}):
            self.assertEqual(self.select(**kw).partial_layout, "partition")

    def test_partitions_keep_int32_partial_indices(self):
        """Cap default partitions at Int32 partial indices; reject explicit overflow."""
        cfg = self.select(rows=1 << 20, columns=1 << 20)
        self.assertEqual(cfg.npar, 2047)
        with self.assertRaisesRegex(ValueError, "overflows Int32"):
            self.select(rows=1 << 20, columns=1 << 20, npar=2048)

    @parametrize(
        "dtype,size,columns,key,fields,rule",
        [
            (torch.float32, 256, 256, "sum", 1, "rubin_c256"),
            (torch.bfloat16, 2048, 1024, "argmaxi32", 2, "rubin_c1024"),
            (torch.float32, 256, 4096, "mean", 1, "rubin_mean_c4096_tiled"),
            (torch.bfloat16, 2048, 4096, "var0", 3, "rubin_var_c4096"),
            (
                torch.bfloat16,
                2048,
                4096,
                "argmaxi32",
                2,
                "rubin_argmax_c4096_tree",
            ),
        ],
    )
    def test_sm107_unordered_profiles(self, dtype, size, columns, key, fields, rule):
        """Pin each distinct SM107 unordered-column launch family."""
        expected = {
            "rubin_c256": (4096, 4, "column", 0, "linear"),
            "rubin_c1024": (4096, 8, "column", 0, "inner_tree"),
            "rubin_mean_c4096_tiled": (256, 8, "partition", 16, "linear"),
            "rubin_var_c4096": (2048, 8, "column", 0, "inner_tree"),
            "rubin_argmax_c4096_tree": (2048, 4, "partition", 8, "inner_tree"),
        }
        cfg = self.select(
            dtype,
            size,
            columns,
            cc=(10, 7),
            trait_key=key,
            nfields=fields,
        )
        self.assertEqual(
            (
                cfg.npar,
                cfg.vec,
                cfg.partial_layout,
                cfg.combine_columns,
                cfg.kernel_order,
            ),
            expected[rule],
        )
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
                torch.float32,
                "var0",
                16384,
                1024,
                2,
                3,
                (torch.float32,),
                (0, 16),
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
            # full_column_anchor, the 65536 x 1024 rule, the c256 Welford and bf16
            # minimums, and a column_anchor seed for a simple trait.
            (torch.float32, "mean", 262144, 256, 32, 1, (torch.float32,), (32, 16)),
            (torch.float32, "var0", 65536, 1024, 8, 3, (torch.float32,), (64, 16)),
            (torch.float32, "var0", 65536, 256, 8, 3, (torch.float32,), (0, 16)),
            (torch.bfloat16, "sum", 131072, 256, 16, 1, (torch.float32,), (8, 16)),
            (torch.float32, "mean", 524288, 1024, 64, 1, (torch.float32,), (32, 32)),
        ],
    )
    def test_sm107_ordered_profiles(
        self, dtype, key, rows, columns, partials, fields, outputs, expected
    ):
        """Pin SM107 ordered mappings."""
        cfg = ct.select_ordered_col_config(
            (10, 7),
            dtype,
            key,
            columns,
            1,
            partials,
            field_bits=(32,) * fields,
            out_dtypes=outputs,
            rows=rows,
            full_tiles=True,
        )
        self.assertEqual((cfg.min_blocks, cfg.tile_columns), expected)
        self.assertEqual((cfg.worker_warps, cfg.full_tiles), (16, True))

    @parametrize(
        "dtype,size,columns,key,fields,order",
        [
            (torch.float32, 64, 256, "sum", 1, "inner_tree"),
            (torch.bfloat16, 16, 256, "var0", 3, "inner_tree"),
            (torch.float32, 16, 4096, "sum", 1, "inner_tree"),
            (torch.bfloat16, 64, 4096, "mean", 1, "inner_tree"),
            (torch.bfloat16, 64, 4096, "amax", 1, "linear"),
            (torch.float32, 64, 1024, "var0", 3, "inner_tree"),
        ],
    )
    def test_sm107_small_hand_offs(self, dtype, size, columns, key, fields, order):
        """Pin which 16 and 64 MiB SM107 columns hand off to the ordered kernel."""
        cfg = self.select(
            dtype, size, columns, cc=(10, 7), trait_key=key, nfields=fields
        )
        self.assertEqual(cfg.kernel_order, order)

    def test_sm107_column_guards(self):
        """Keep incompatible SM107 calls on generic policies."""
        self.assertEqual(self.select(cc=(10, 7), size=255).rule, "default")
        self.assertIsNone(
            ct.select_ordered_col_config(
                (10, 7),
                torch.bfloat16,
                "mean",
                4096,
                1,
                32,
                field_bits=(64,),
                out_dtypes=(torch.float32,),
                rows=262144,
                full_tiles=True,
            )
        )

    @parametrize("kwargs", [{"threads_per_block": 128}, {"npar": 7}, {"vec": 2}])
    def test_explicit_geometry_disables_policy(self, kwargs):
        """Honor explicit column geometry without rewriting it through policy."""
        cfg = self.select(**kwargs)
        self.assertEqual(cfg.rule, "explicit")
        for key, value in kwargs.items():
            self.assertEqual(getattr(cfg, key), value)
        self.assertEqual(cfg.partial_layout, "column")
        self.assertEqual(cfg.combine_columns, 0)

    @parametrize(
        "dtype,key,rows,columns,partials,full_tiles,expected",
        [
            (torch.float32, "sum", 8192, 4096, 1, False, None),
            (torch.float16, "sum", 8192, 4096, 1, True, (0, 32)),
            (torch.float32, "sum", 65536, 1024, 8, True, (0, 16)),
            (torch.float32, "sum", 32768, 257, 4, True, (0, 16)),
            (torch.float32, "sum", 131072, 257, 16, True, (0, 32)),
            (torch.bfloat16, "amax", 32768, 4096, 4, True, (0, 16)),
            (torch.bfloat16, "amax", 262144, 4096, 32, True, (0, 32)),
        ],
    )
    def test_b200_ordered_column_mappings(
        self, dtype, key, rows, columns, partials, full_tiles, expected
    ):
        """Pin B200 ordered column widths, independent of SM107 guards."""
        # Single-field float outputs may be the input dtype or fp32.
        for output in {dtype, dtype if key == "amax" else torch.float32}:
            with self.subTest(output=output):
                cfg = ct.select_ordered_col_config(
                    (10, 0),
                    dtype,
                    key,
                    columns,
                    1,
                    partials,
                    field_bits=(32,),
                    out_dtypes=(output,),
                    rows=rows,
                    full_tiles=full_tiles,
                )
                if expected is None:
                    self.assertIsNone(cfg)
                    continue
                min_blocks, tile_columns = expected
                want = ct.OrderedColConfig(min_blocks, tile_columns=tile_columns)
                self.assertEqual(cfg, want._replace(full_tiles=True))


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
        stage = patch.object(ct.tile, "TileReduce", wraps=ct.tile.TileReduce)
        with patch.object(ct, "select_col_config", return_value=cfg), stage as built:
            got = ct._reduce_col_tile(*args)[0]
        self.assertEqual(built.call_args_list[0].kwargs["unroll"], 2)
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
        args = (trait, "order", x, [torch.float32], 1, None, None, None)
        results = []
        original = ct.reduce_ordered_col

        def record(*a, **kw):
            results.append(original(*a, **kw))
            return results[-1]

        ordered = patch.object(ct, "reduce_ordered_col", record)
        with patch.object(ct, "select_col_config", return_value=cfg), ordered:
            got = ct._reduce_col_tile(*args)[0]
        self.assertEqual([r is not None for r in results], [served])
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
