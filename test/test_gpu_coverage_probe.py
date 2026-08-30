# Owner(s): ["module: ci"]

"""Scratch file for exercising the gpu-coverage check. Not for landing."""

import unittest

from torch.testing._internal.common_cuda import SM100OrLater
from torch.testing._internal.common_utils import run_tests, TestCase


class TestGpuCoverageProbe(TestCase):
    @unittest.skipIf(not SM100OrLater, "needs a B200")
    def test_gpu_coverage_probe(self):
        self.assertTrue(True)


if __name__ == "__main__":
    run_tests()
