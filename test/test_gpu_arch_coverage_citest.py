# Owner(s): ["module: ci"]
"""Throwaway probe for the GPU_ARCH_COVERAGE linter. DO NOT LAND.

Exists to prove the chain end to end: an sm90-gated test added to a PR gets
wired into .github/labeler.yml and .ci/pytorch/test.sh by `lintrunner -a`, and
the ciflow/h100 job then actually executes it.

The assertions are written so that running on the wrong hardware fails rather
than silently passing -- a skipped test is also a green test, so the point is to
be able to tell the two apart in the uploaded report.
"""

import unittest

import torch
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_utils import run_tests, TestCase


class GpuArchCoverageCiTest(TestCase):
    @unittest.skipUnless(SM90OrLater, "probe requires sm90+")
    def test_probe_runs_on_sm90_or_later(self):
        self.assertTrue(torch.cuda.is_available())
        major, minor = torch.cuda.get_device_capability()
        print(f"GPU_ARCH_COVERAGE probe executed on sm_{major}{minor}")
        self.assertGreaterEqual((major, minor), (9, 0))


if __name__ == "__main__":
    run_tests()
