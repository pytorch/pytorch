# Owner(s): ["oncall: package/deploy"]

import sys
import unittest

import torch
from torch.package import analyze
from torch.testing._internal.common_utils import IS_LINUX, run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestAnalyze(PackageTestCase):
    """Dependency analysis API tests."""

    def test_trace_dependencies_restores_profile(self):
        def profile(frame, event, arg):
            pass

        previous_profile = sys.getprofile()
        self.addCleanup(sys.setprofile, previous_profile)
        sys.setprofile(profile)

        analyze.trace_dependencies(lambda: None, [()])

        self.assertIs(sys.getprofile(), profile)

    def test_trace_dependencies_restores_profile_when_callable_raises(self):
        def profile(frame, event, arg):
            pass

        def fail():
            raise RuntimeError("boom")

        previous_profile = sys.getprofile()
        self.addCleanup(sys.setprofile, previous_profile)
        sys.setprofile(profile)

        with self.assertRaisesRegex(RuntimeError, "boom"):
            analyze.trace_dependencies(fail, [()])

        self.assertIs(sys.getprofile(), profile)

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/81213")
    def test_trace_dependencies(self):
        import test_trace_dep

        obj = test_trace_dep.SumMod()

        used_modules = analyze.trace_dependencies(obj, [(torch.randn(4),)])

        self.assertNotIn("yaml", used_modules)
        self.assertIn("test_trace_dep", used_modules)


if __name__ == "__main__":
    run_tests()
