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
        import test_trace_dep

        obj = test_trace_dep.SumMod()

        used_modules = analyze.trace_dependencies(obj, [(torch.randn(4),)])

        self.assertNotIn("yaml", used_modules)
        self.assertIn("test_trace_dep", used_modules)

    def test_dependency_explorer(self):
        de = analyze.DependencyExplorer("test_dependency_explorer")

        self.assertFalse(de.can_package())
        de.extern(["numpy.**.**", "scipy.**.**", "_sentencepiece", "mkl_random.mklrand"])
        self.assertFalse(de.can_package())
        unresolved_dependencies = [mod.name for mod in de.get_unresolved_dependencies()]
        self.assertIn("_io", unresolved_dependencies)
        self.assertIn("sys", unresolved_dependencies)
        de.extern(["_io", "sys"])
        self.assertTrue(de.can_package())

if __name__ == "__main__":
    run_tests()
