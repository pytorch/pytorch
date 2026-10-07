# Owner(s): ["module: dsl-native-ops"]

import sys
import unittest
from unittest.mock import patch

import torch
from torch import _native as native
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
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
    def select(self, dtype=torch.float32, size=256, columns=4096, **kwargs):
        itemsize = 2 if dtype == torch.bfloat16 else 4
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
        """Keep unmeasured architectures on conservative column geometry."""
        cfg = self.select(size=16, columns=256)
        self.assertEqual(cfg.rule, "default")
        self.assertEqual(cfg.threads_per_block, 32)
        self.assertEqual(cfg.npar, 256)
        self.assertEqual(cfg.vec, 4)
        self.assertEqual(cfg.partial_layout, "column")

    @parametrize("kwargs", [{"threads_per_block": 128}, {"npar": 7}, {"vec": 2}])
    def test_explicit_geometry_disables_policy(self, kwargs):
        """Honor explicit column geometry without rewriting it through policy."""
        cfg = self.select(**kwargs)
        self.assertEqual(cfg.rule, "explicit")
        for key, value in kwargs.items():
            self.assertEqual(getattr(cfg, key), value)
        self.assertEqual(cfg.partial_layout, "column")
        self.assertEqual(cfg.combine_columns, 0)

    def test_inner_tree_never_selects_unordered_plan(self):
        """Never satisfy an ordered request with an unordered column plan."""
        self.assertIsNone(self.select(order="inner_tree"))
        with self.assertRaisesRegex(ValueError, "unknown reduction order"):
            self.select(order="linear")
        with self.assertRaisesRegex(ValueError, "requires unordered"):
            ct.reduce_col_tile(None, "sum", None, torch.float32, order="inner_tree")


class TestColumnCombine(TestCase):
    @onlyCUDA
    @parametrize(
        "dtype,op,pattern,tile_columns",
        [
            (torch.float32, "sum", "signed", 8),
            (torch.bfloat16, "argmax", "nonfinite_ties", 8),
            (torch.float32, "var_mean", "offset", 32),
        ],
    )
    def test_tiled_combine_numerics(self, device, dtype, op, pattern, tile_columns):
        """Validate one-field, indexed, and multi-output tiled combines."""
        x = torch.randn((257, 260), device=device, dtype=dtype)
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
        }
        kw = {"correction": 0} if op == "var_mean" else {}
        trait = traits[op](acc=cutlass.Float32, **kw)
        nouts = 2 if op == "var_mean" else 1
        odt = torch.int64 if op == "argmax" else torch.float32
        cfg = ct.ColConfig(
            "unordered", "test_tiled", 64, 7, 4, "partition", tile_columns
        )

        def reference():
            with (
                native._unconditional_masked(),
                torch.backends.python_native.cutedsl.disabled(),
            ):
                kw = {"correction": 0} if op == "var_mean" else {}
                if op == "sum":
                    kw["dtype"] = torch.float32
                result = getattr(torch, op)(x, dim=0, **kw)
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
        self.assertEqual(got, reference(), rtol=tol, atol=tol, equal_nan=True)


instantiate_parametrized_tests(TestColumnConfig)
instantiate_device_type_tests(TestColumnCombine, globals(), only_for=("cuda",))


if __name__ == "__main__":
    run_tests()
