# Owner(s): ["module: dsl-native-ops"]

import sys
import unittest

import torch
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

from torch._native.ops.reductions import kernel_coltile as ct, kernel_general as kg


class TestColumnConfig(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def select(self, dtype=torch.float32, size=256, columns=4096, **kwargs):
        itemsize = dtype.itemsize
        args = dict(
            rows=(size << 20) // (columns * itemsize),
            columns=columns,
            batches=1,
            nfields=1,
            itemsize=itemsize,
        )
        args.update(kwargs)
        return ct.select_col_config(**args)

    def test_default_geometry(self):
        """Fill about _TARGET_LANES threads, narrowing vec before chunks get short."""
        cfg = self.select(size=16, columns=256)
        self.assertEqual(cfg.rule, "default")
        self.assertEqual(cfg.threads_per_block, 32)
        # Wider vectors would leave 16-row chunks; 256 lanes take 512 chunks of 32 rows.
        self.assertEqual((cfg.npar, cfg.vec), (512, 1))
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
        "kwargs,rule,geometry",
        [
            ({"threads_per_block": 128}, "default", (128, 4)),
            ({"npar": 7}, "explicit", (7, 4)),
            # Two-wide vectors make 2048 lanes, which want 64 partitions, not 128.
            ({"vec": 2}, "explicit", (64, 2)),
        ],
    )
    def test_explicit_geometry_disables_policy(self, kwargs, rule, geometry):
        """Honor explicit stage-one geometry without rewriting it through policy, and
        derive the free field from the explicit one."""
        cfg = self.select(**kwargs)
        self.assertEqual(cfg.rule, rule)
        self.assertEqual((cfg.npar, cfg.vec), geometry)
        for key, value in kwargs.items():
            self.assertEqual(getattr(cfg, key), value)

    @parametrize("nouts", [1, 2])
    @parametrize("shape", [(1100, 8192), (4, 64, 16)], name_fn=lambda s: f"{len(s)}d")
    def test_route_gate_by_architecture(self, shape, nouts):
        # Off H100 and B200, only 2D one-output column views leave the general kernel.
        view = torch.empty(shape, device="meta")
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


instantiate_parametrized_tests(TestColumnConfig)


if __name__ == "__main__":
    run_tests()
