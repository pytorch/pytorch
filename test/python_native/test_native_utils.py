# Owner(s): ["module: dsl-native-ops"]
#
# Tests for the DSL-agnostic native-op utils. torch-only helpers (no cutlass), so no GPU or DSL
# is needed and both branches of every cond primitive are reachable here.

import sys

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import run_tests, TestCase


class TestNativeUtils(TestCase):
    def test_lazy_module_defers_import(self):
        # LazyModule must NOT import its target until first attribute access -- this is what
        # keeps `import torch` free of DSL runtimes (the lazy-DSL-import contract).
        from torch._native.utils.lazy import LazyModule

        name = "wave"  # a stdlib module unlikely to be loaded already
        prior = sys.modules.pop(name, None)
        # Restore whatever was there: the assertions below need the module ABSENT, and leaving
        # the interpreter mutated would break any later test that imports it.
        self.addCleanup(
            lambda: sys.modules.__setitem__(name, prior)
            if prior is not None
            else sys.modules.pop(name, None)
        )
        mod = LazyModule(name)
        self.assertNotIn(name, sys.modules)  # constructing the proxy imports nothing
        self.assertTrue(callable(mod.open))  # first attr access triggers the import
        self.assertIn(name, sys.modules)

    def test_is_traced_exact_tensor_branch(self):
        # The fast branch: an exact torch.Tensor can only be "traced" by being on meta, so it
        # never pays for is_fake().
        from torch._native.utils import capability as cap

        self.assertTrue(cap.is_traced(torch.empty(2, device="meta")))
        self.assertFalse(cap.is_traced(torch.empty(2)))

    def test_is_traced_fake_tensor_branch(self):
        # The slow branch, and the reason the helper is more than a device read: a FakeTensor is
        # never EXACTLY torch.Tensor, so it falls through to is_fake(). Declining these is what
        # keeps a compile/export trace on aten's reference instead of our kernel.
        from torch._native.utils import capability as cap

        with FakeTensorMode() as fake:
            ft = fake.from_tensor(torch.empty(2))
            self.assertTrue(cap.is_traced(ft))

    def test_device_ok_short_circuits_off_cuda(self):
        # Must answer False for a CPU tensor WITHOUT touching torch.cuda: these conds run on
        # every eager dispatch, including on builds with no CUDA at all.
        from torch._native.utils import capability as cap

        self.assertFalse(cap.device_ok(torch.empty(2)))
        self.assertFalse(cap.device_ok(torch.empty(2, device="meta")))

    def test_device_ok_memoizes_per_device(self):
        # The arch answer is immutable per device and the query is ~1.4us, so it is cached by
        # device index. Assert the cache is populated once and reused (a miss on every call
        # would put a device-property query on the hot path).
        from torch._native.utils import capability as cap

        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device to populate the arch cache")
        idx = torch.cuda.current_device()
        prior = cap._ARCH_OK.copy()
        self.addCleanup(lambda: (cap._ARCH_OK.clear(), cap._ARCH_OK.update(prior)))
        cap._ARCH_OK.clear()
        x = torch.empty(2, device="cuda")
        first = cap.device_ok(x)
        self.assertIn(idx, cap._ARCH_OK)
        # Poison the cache: a second call that re-queried the device would disagree.
        cap._ARCH_OK[idx] = not first
        self.assertEqual(cap.device_ok(x), not first)

    def test_on_current_device_never_raises(self):
        # capability.py's contract is that a cond NEVER raises -- a throwing cond takes down the
        # dispatcher instead of falling back to aten. A CPU tensor (or a build without CUDA) has
        # to answer False rather than raise from torch.cuda.current_device().
        from torch._native.utils import capability as cap

        self.assertFalse(cap.on_current_device(torch.empty(2)))
        self.assertFalse(cap.on_current_device(torch.empty(2, device="meta")))
        if torch.cuda.is_available():
            self.assertTrue(cap.on_current_device(torch.empty(2, device="cuda")))


if __name__ == "__main__":
    run_tests()
