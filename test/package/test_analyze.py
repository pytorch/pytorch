from unittest import skipIf

import torch
from torch.package import analyze
from torch.testing._internal.common_utils import IS_FBCODE, IS_SANDCASTLE, run_tests

try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase  # type: ignore


class TestAnalyze(PackageTestCase):
    """Dependency analysis API tests."""

    @skipIf(IS_FBCODE or IS_SANDCASTLE, "yaml not available")
    def test_trace_dependencies(self):
        import test_trace_dep

        obj = test_trace_dep.SumMod()

        used_modules = analyze.trace_dependencies(obj, [(torch.randn(4),)])

        self.assertNotIn("yaml", used_modules)
        self.assertIn("test_trace_dep", used_modules)


if __name__ == "__main__":
    run_tests()
