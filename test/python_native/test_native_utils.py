# Owner(s): ["module: dsl-native-ops"]
#
# Tests for the DSL-agnostic native-op utils. torch-only helpers (no cutlass), so no DSL is
# needed; the is_traced and device_ok branches below run without a GPU, and the arch-cache and
# current-device cases skip when there is no CUDA.

import sys

import torch
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


# These are host-side predicates over tensor IDENTITY, and dynamo rewrites exactly that:
# it graph-breaks at a FakeTensorMode entry and the resumed frame holds a tensor that is
# no longer fake, so `is_traced` answers False about it -- correctly, but about the wrong
# object. Compiling them adds no coverage either way.
@skipIfTorchDynamo("host-side capability predicates need no dynamo compilation")
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

    def test_is_traced_exact_type_cpp_wrappers(self):
        # The fast path's premise is that an exact-type tensor cannot be traced except on meta.
        # That is false for the two C++-level wrappers, which are dispatch-key wrappers rather
        # than Python subclasses: functionalization over a fake tensor is EXACTLY torch.Tensor,
        # is_fake() says True, and answering False here would launch a kernel mid-trace.
        from torch._native.utils import capability as cap

        with FakeTensorMode() as mode:
            wrapped = torch._to_functional_tensor(mode.from_tensor(torch.empty(2)))
            self.assertIs(
                type(wrapped), torch.Tensor
            )  # the premise the fast path relied on
            self.assertTrue(is_fake(wrapped))
            self.assertTrue(cap.is_traced(wrapped))

    def test_is_traced_fake_tensor_branch(self):
        # The slow branch, and the reason the helper is more than a device read: a FakeTensor is
        # never EXACTLY torch.Tensor, so it falls through to is_fake(). Declining these is what
        # keeps a compile/export trace on aten's reference instead of our kernel.
        from torch._native.utils import capability as cap

        with FakeTensorMode() as fake:
            ft = fake.from_tensor(torch.empty(2))
            self.assertTrue(cap.is_traced(ft))

    def test_device_ok_short_circuits_off_cuda(self):
        # False for a non-CUDA tensor, and reached WITHOUT querying the device -- these conds run
        # on every eager dispatch, including on builds with no CUDA at all. Patching the query
        # asserts the short-circuit; a return value alone cannot tell the two apart.
        from unittest.mock import patch

        from torch._native.utils import capability as cap

        with patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=AssertionError("queried the device"),
        ):
            self.assertFalse(cap.device_ok(torch.empty(2), 9))
            self.assertFalse(cap.device_ok(torch.empty(2, device="meta"), 9))

    def test_device_ok_memoizes_per_device(self):
        # The arch answer is immutable per device, so it is asked once. Assert that as BEHAVIOUR
        # -- the second call must not query the device -- rather than by inspecting the memo,
        # which would weld the test to the cache's representation.
        from unittest.mock import patch

        from torch._native.utils import capability as cap

        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device to populate the arch cache")
        self.addCleanup(cap._arch_ok.cache_clear)
        cap._arch_ok.cache_clear()
        x = torch.empty(2, device="cuda")
        with patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=torch.cuda.get_device_capability,
        ) as query:
            first = cap.device_ok(x, 9)
            self.assertEqual(query.call_count, 1)
            self.assertEqual(cap.device_ok(x, 9), first)
            self.assertEqual(query.call_count, 1, "the arch query was repeated")

    def test_device_ok_honours_the_callers_bound(self):
        # The bound is the caller's, so it has to be part of the memo KEY: families disagree
        # about the minimum (topk sm100+, reductions sm90+) and one must not be served the
        # other's answer. Asserting both directions on one device is what catches that.
        from unittest.mock import patch

        from torch._native.utils import capability as cap

        if not torch.cuda.is_available():
            self.skipTest("needs a CUDA device to reach the arch query")
        self.addCleanup(cap._arch_ok.cache_clear)
        cap._arch_ok.cache_clear()
        x = torch.empty(2, device="cuda")
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            self.assertTrue(cap.device_ok(x, 9))
            self.assertFalse(
                cap.device_ok(x, 10), "the sm90 answer was reused for sm100+"
            )

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
