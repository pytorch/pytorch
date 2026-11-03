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

from torch._native.ops.reductions import kernel_coltile as ct


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

    def test_explicit_partitions_are_bounded(self):
        """Reject explicit partitions past Int32 partial indices or grid y."""
        with self.assertRaisesRegex(ValueError, "overflows Int32"):
            self.select(rows=1 << 20, columns=1 << 20, npar=2048)
        with self.assertRaisesRegex(ValueError, "grid y"):
            self.select(rows=1 << 20, columns=2, npar=65536)

    @parametrize("kwargs", [{"threads_per_block": 128}, {"npar": 7}, {"vec": 2}])
    def test_explicit_geometry_disables_policy(self, kwargs):
        """Honor explicit stage-one geometry without rewriting it through policy."""
        cfg = self.select(**kwargs)
        self.assertEqual(cfg.rule, "explicit")
        for key, value in kwargs.items():
            self.assertEqual(getattr(cfg, key), value)


instantiate_parametrized_tests(TestColumnConfig)


if __name__ == "__main__":
    run_tests()
