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

    @parametrize("kwargs", [{"threads_per_block": 128}, {"npar": 7}, {"vec": 2}])
    def test_explicit_geometry_disables_policy(self, kwargs):
        """Honor explicit column geometry without rewriting it through policy."""
        cfg = self.select(**kwargs)
        self.assertEqual(cfg.rule, "explicit")
        for key, value in kwargs.items():
            self.assertEqual(getattr(cfg, key), value)
        self.assertEqual(cfg.partial_layout, "column")
        self.assertEqual(cfg.combine_columns, 0)

    def test_ordered_columns_require_inner_tree(self):
        """Reject an unordered request on the ordered column kernel."""
        with self.assertRaisesRegex(ValueError, "require inner_tree"):
            ct.reduce_ordered_col(
                None, "sum", None, [torch.float32], 1, order="unordered"
            )

    def test_b200_ordered_column_packing(self):
        """Pin B200 packing separately so Rubin guards cannot reject its valid rule."""
        cfg = ct.select_ordered_col_config(
            (10, 0),
            torch.bfloat16,
            "mean",
            4096,
            1,
            16,
            field_bits=(32,),
            out_dtypes=(torch.float32,),
            rows=262144,
            full_tiles=True,
        )
        self.assertEqual(
            cfg,
            ct.OrderedColConfig(0, tile_columns=8, full_tiles=True, column_pack=4),
        )


class TestColumnCombine(TestCase):
    hw_classification = HardwareClassification.CUDA

    @parametrize(
        "dtype,op,pattern,width,shape",
        [
            (torch.float32, "sum", "signed", 8, (257, 260)),
            (torch.bfloat16, "argmax", "nonfinite_ties", 8, (257, 260)),
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
                f"test_tiled_{op}",
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
        self.assertIn(("coltile2t", f"test_tiled_{op}"), {k[:2] for k in ct._CACHE})
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


instantiate_parametrized_tests(TestColumnConfig)
instantiate_device_type_tests(TestColumnCombine, globals(), only_for=("cuda",))


if __name__ == "__main__":
    run_tests()
