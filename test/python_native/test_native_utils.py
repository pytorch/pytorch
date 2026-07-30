# Owner(s): ["module: dsl-native-ops"]
#
# Minimal tests for the DSL-agnostic native-op utils. These are torch-only helpers
# (no cutlass), so they are testable with no GPU/DSL. Broader behavior is exercised
# by every override family's cond in later commits.

from torch.testing._internal.common_utils import run_tests, TestCase


class TestNativeUtils(TestCase):
    def test_lazy_module_defers_import(self):
        # LazyModule must NOT import its target until first attribute access -- this is
        # what keeps `import torch` free of DSL runtimes (the lazy-DSL-import contract).
        import sys

        from torch._native.utils.lazy import LazyModule

        # A stdlib module not yet imported: pick one unlikely to be loaded.
        name = "wave"
        sys.modules.pop(name, None)
        mod = LazyModule(name)
        self.assertNotIn(name, sys.modules)  # constructing the proxy imports nothing
        self.assertTrue(callable(mod.open))  # first attr access triggers the import
        self.assertIn(name, sys.modules)

    def test_capability_is_traced_on_meta(self):
        # is_traced declines meta tensors (no real storage to launch on). Pure torch,
        # no CUDA needed.
        import torch
        from torch._native.utils import capability as cap

        self.assertTrue(cap.is_traced(torch.empty(2, device="meta")))
        self.assertFalse(cap.is_traced(torch.empty(2)))


if __name__ == "__main__":
    run_tests()
