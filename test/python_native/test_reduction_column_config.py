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
    hw_classification = HardwareClassification.CUDA

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
