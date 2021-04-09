import torch
from torch.package import analyze

from torch.testing._internal.common_utils import run_tests

try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase  # type: ignore


class TestAnalyze(PackageTestCase):
    """Dependency analysis API tests."""

    def test_trace_dependencies(self):
        import automock

        obj = automock.SumMod()

        used_modules = analyze.trace_dependencies(obj, [(torch.randn(4),)])

        self.assertNotIn("yaml", used_modules)
        self.assertIn("automock", used_modules)

if __name__ == "__main__":
    run_tests()
