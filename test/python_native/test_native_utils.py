# Owner(s): ["module: dsl-native-ops"]
#
# Tests for the DSL-agnostic native-op utils: torch-only, so no DSL is needed. The cases
# needing a real arch query are gated on CUDA AND NOT ROCm -- is_available() is True on a
# ROCm build while device_ok declines HIP by design, so a CUDA-only guard FAILS there.

import sys
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfRocm,
    skipIfTorchDynamo,
    TestCase,
)


# Host-side predicates over tensor IDENTITY, which is exactly what dynamo rewrites: it
# graph-breaks at a FakeTensorMode entry and the resumed frame holds a tensor that is no
# longer fake, so is_traced answers correctly about the wrong object.
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
        # The fast path's premise -- an exact-type tensor cannot be traced except on meta -- is false
        # for the two C++-level wrappers, which are dispatch-key wrappers rather than Python
        # subclasses: functionalization over a fake tensor is EXACTLY torch.Tensor, and answering
        # False would launch a kernel mid-trace.
        from torch._native.utils import capability as cap

        with FakeTensorMode() as mode:
            wrapped = torch._to_functional_tensor(mode.from_tensor(torch.empty(2)))
            self.assertIs(
                type(wrapped), torch.Tensor
            )  # the premise the fast path relied on
            self.assertTrue(is_fake(wrapped))
            self.assertTrue(cap.is_traced(wrapped))

    def test_is_traced_fake_tensor_branch(self):
        # The slow branch: a FakeTensor is never EXACTLY torch.Tensor, so it falls through to
        # is_fake(). Declining these keeps a trace on aten's reference instead of our kernel.
        from torch._native.utils import capability as cap

        with FakeTensorMode() as fake:
            ft = fake.from_tensor(torch.empty(2))
            self.assertTrue(cap.is_traced(ft))

    def test_device_ok_short_circuits_off_cuda(self):
        # False for a non-CUDA tensor, and reached WITHOUT querying the device -- these run on every
        # eager dispatch, including on builds with no CUDA. Patching the query asserts the
        # short-circuit; a return value alone cannot tell the two apart.
        from unittest.mock import patch

        from torch._native.utils import capability as cap

        with patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=AssertionError("queried the device"),
        ):
            self.assertFalse(cap.device_ok(torch.empty(2), (9, 10)))
            self.assertFalse(cap.device_ok(torch.empty(2, device="meta"), (9, 10)))

    def test_device_ok_declines_hip(self):
        # A ROCm build reports device.type == "cuda", so the HIP arm is what keeps these kernels off
        # it. Patch torch.version.hip rather than requiring a ROCm machine, and make the arch query
        # raise so this also proves HIP is refused BEFORE the device is touched.
        from unittest.mock import patch

        from torch._native.utils import capability as cap

        with FakeTensorMode():
            x = torch.empty(2, device="cuda")
        with (
            patch.object(torch.version, "hip", "6.0"),
            patch.object(
                torch.cuda,
                "get_device_capability",
                side_effect=AssertionError("queried the device on a HIP build"),
            ),
        ):
            self.assertFalse(cap.device_ok(x, (9, 10)))

    @unittest.skipUnless(TEST_CUDA, "needs a CUDA device to populate the arch cache")
    @skipIfRocm
    def test_device_ok_memoizes_per_device(self):
        # The arch answer is immutable per device, so assert the memo as BEHAVIOUR -- the second call
        # must not query -- rather than by inspecting it, which would weld the test to its shape.
        from unittest.mock import patch

        from torch._native.utils import capability as cap

        self.addCleanup(cap._arch_ok.cache_clear)
        cap._arch_ok.cache_clear()
        x = torch.empty(2, device="cuda")
        with patch.object(
            torch.cuda,
            "get_device_capability",
            side_effect=torch.cuda.get_device_capability,
        ) as query:
            first = cap.device_ok(x, (9, 10, 12))
            self.assertEqual(query.call_count, 1)
            self.assertEqual(cap.device_ok(x, (9, 10, 12)), first)
            self.assertEqual(query.call_count, 1, "the arch query was repeated")

    @skipIfRocm
    def test_device_ok_honours_the_callers_set(self):
        # The accepted set is the caller's, so it must be part of the memo KEY: families disagree and
        # one must not be served the other's answer. Thor (SM 11.0) is the case that makes it
        # load-bearing -- a family enumerating (9, 10, 12) must refuse it.
        from unittest.mock import patch

        from torch._native.utils import capability as cap

        self.addCleanup(cap._arch_ok.cache_clear)
        cap._arch_ok.cache_clear()
        with FakeTensorMode():
            x = torch.empty(2, device="cuda")
        with patch.object(torch.cuda, "get_device_capability", return_value=(11, 0)):
            # Thor (SM 11.0): a family enumerating (9, 10, 12) must refuse it, while one accepting 11 must
            # not be handed the first answer out of the memo.
            self.assertFalse(cap.device_ok(x, (9, 10, 12)))
            self.assertTrue(cap.device_ok(x, (11,)), "the (9,10,12) answer was reused")

    def test_on_current_device_never_raises(self):
        # capability.py's contract is that a cond NEVER raises -- a throwing cond takes down the
        # dispatcher instead of falling back. A CPU tensor must answer False, not raise.
        from torch._native.utils import capability as cap

        self.assertFalse(cap.on_current_device(torch.empty(2)))
        self.assertFalse(cap.on_current_device(torch.empty(2, device="meta")))
        if torch.cuda.is_available():
            self.assertTrue(cap.on_current_device(torch.empty(2, device="cuda")))

    @unittest.skipUnless(TEST_CUDA, "needs a CUDA tensor to compare device indices")
    @skipIfRocm
    def test_on_current_device_declines_another_device(self):
        # A tensor on a device that is NOT the current one must be declined, since the kernel and
        # stream caches are bound to the current device. Without this, `return True` passes the rest
        # of the suite -- the CPU and meta cases short-circuit before reaching it.
        from unittest.mock import patch

        from torch._native.utils import capability as cap

        x = torch.empty(2, device="cuda")
        self.assertTrue(cap.on_current_device(x))
        with patch.object(
            torch.cuda, "current_device", return_value=x.device.index + 1
        ):
            self.assertFalse(cap.on_current_device(x))

    def test_conds_never_raise_without_a_usable_device(self):
        # Tracing a CUDA model on a CPU-only box: a fake cuda tensor normalizes to cuda:0 without
        # initializing CUDA, so both queries raise. Both conds must answer False instead -- the
        # never-raises contract is the only thing between that and a dead dispatcher.
        from unittest.mock import patch

        from torch._native.utils import capability as cap

        with FakeTensorMode():
            x = torch.empty(2, device="cuda")
        self.addCleanup(cap._arch_ok.cache_clear)
        cap._arch_ok.cache_clear()
        boom = RuntimeError("No CUDA GPUs are available")
        with (
            patch.object(torch.cuda, "get_device_capability", side_effect=boom),
            patch.object(torch.cuda, "current_device", side_effect=boom),
        ):
            self.assertFalse(cap.device_ok(x, (9, 10, 12)))
            self.assertFalse(cap.on_current_device(x))


if __name__ == "__main__":
    run_tests()
