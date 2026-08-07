# Owner(s): ["module: ci"]
"""Probe for the GPU_ARCH_COVERAGE linter. DO NOT LAND."""

import unittest

import torch
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_utils import run_tests, TestCase


class GacProbeTest(TestCase):
    @unittest.skipUnless(SM90OrLater, "probe requires sm90+")
    def test_gac_probe_sm90(self):
        major, minor = torch.cuda.get_device_capability()
        print(f"GAC probe executed on sm_{major}{minor}")
        self.assertGreaterEqual((major, minor), (9, 0))


if __name__ == "__main__":
    run_tests()
