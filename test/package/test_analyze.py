import torch
from torch.package import analyze

from torch.testing._internal.common_utils import run_tests

try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase


class TestAnalyze(PackageTestCase):
    """Dependency analysis API tests."""

    def test_trace_dependencies(self):
        import test_trace_dep

        obj = test_trace_dep.SumMod()

        used_modules = analyze.trace_dependencies(obj, [(torch.randn(4),)])

        self.assertNotIn("yaml", used_modules)
        self.assertIn("test_trace_dep", used_modules)

if __name__ == "__main__":
    run_tests()
