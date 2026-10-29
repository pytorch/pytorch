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
        itemsize = 2 if dtype == torch.bfloat16 else 4
        args = dict(
            rows=(size << 20) // (columns * itemsize),
            columns=columns,
            nfields=1,
            itemsize=itemsize,
        )
        args.update(kwargs)
        return ct.select_col_config(**args)

    def test_default_geometry(self):
        """Size stage one from the reduced and column extents by default."""
        cfg = self.select(size=16, columns=256)
        self.assertEqual(cfg.rule, "default")
        self.assertEqual(cfg.threads_per_block, 32)
        self.assertEqual(cfg.npar, 256)
        self.assertEqual(cfg.vec, 4)

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
