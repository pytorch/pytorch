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

    @unittest.skipIf(IS_LINUX, "https://github.com/pytorch/pytorch/issues/81213")
    def test_trace_dependencies(self):
        import test_trace_dep

        obj = test_trace_dep.SumMod()

        used_modules = analyze.trace_dependencies(obj, [(torch.randn(4),)])

        self.assertNotIn("yaml", used_modules)
        self.assertIn("test_trace_dep", used_modules)

    def test_trace_dependencies_restores_prior_profiler(self):
        def prior_profile(frame, event, arg):
            pass

        sys.setprofile(prior_profile)
        try:
            analyze.trace_dependencies(lambda x: x, [(1,)])
            self.assertIs(sys.getprofile(), prior_profile)
        finally:
            sys.setprofile(None)

    def test_trace_dependencies_restores_prior_profiler_on_exception(self):
        def prior_profile(frame, event, arg):
            pass

        def raises(x):
            raise ValueError("boom")

        sys.setprofile(prior_profile)
        try:
            with self.assertRaises(ValueError):
                analyze.trace_dependencies(raises, [(1,)])
            self.assertIs(sys.getprofile(), prior_profile)
        finally:
            sys.setprofile(None)

    def test_trace_dependencies_clears_profiler_when_none_installed(self):
        self.assertIsNone(sys.getprofile())
        analyze.trace_dependencies(lambda x: x, [(1,)])
        self.assertIsNone(sys.getprofile())


if __name__ == "__main__":
    run_tests()
