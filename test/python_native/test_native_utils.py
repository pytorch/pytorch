# Owner(s): ["module: dsl-native-ops"]
# Torch-only tests for DSL-agnostic native-op utilities. Real architecture queries require
# CUDA, not ROCm: is_available() is true on ROCm while device_ok intentionally declines HIP.

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


# These predicates inspect tensor identity, which Dynamo rewrites across a FakeTensorMode
# graph break; the resumed frame can therefore hold a non-fake tensor.
@skipIfTorchDynamo("host-side capability predicates need no dynamo compilation")
class TestNativeUtils(TestCase):
    def test_lazy_module_defers_import(self):
        # Deferring the target import until attribute access keeps `import torch` DSL-free.
        from torch._native.utils.lazy import LazyModule

        name = "wave"  # a stdlib module unlikely to be loaded already
        prior = sys.modules.pop(name, None)
        # Restore the module afterward; this test needs it absent without affecting later tests.
        self.addCleanup(
            lambda: sys.modules.__setitem__(name, prior)
            if prior is not None
            else sys.modules.pop(name, None)
        )
        mod = LazyModule(name)
        self.assertNotIn(name, sys.modules)
        self.assertTrue(callable(mod.open))
        self.assertIn(name, sys.modules)

    def test_is_traced_exact_tensor_branch(self):
        # Exact tensors avoid is_fake()'s subclass walk via cheap meta and C++ wrapper checks.
        from torch._native.utils import capability as cap

        self.assertTrue(cap.is_traced(torch.empty(2, device="meta")))
        self.assertFalse(cap.is_traced(torch.empty(2)))

    def test_is_traced_exact_type_cpp_wrappers(self):
        # C++ dispatch-key wrappers are exact torch.Tensor instances, not Python subclasses.
        # Functionalization over a fake tensor must still be detected to avoid a mid-trace launch.
        from torch._native.utils import capability as cap

        with FakeTensorMode() as mode:
            wrapped = torch._to_functional_tensor(mode.from_tensor(torch.empty(2)))
            self.assertIs(type(wrapped), torch.Tensor)
            self.assertTrue(is_fake(wrapped))
            self.assertTrue(cap.is_traced(wrapped))

    def test_is_traced_fake_tensor_branch(self):
        # FakeTensor subclasses reach is_fake() and stay on ATen's reference while tracing.
        from torch._native.utils import capability as cap

        with FakeTensorMode() as fake:
            ft = fake.from_tensor(torch.empty(2))
            self.assertTrue(cap.is_traced(ft))

    def test_device_ok_short_circuits_off_cuda(self):
        # Non-CUDA tensors must short-circuit before device queries, including on CPU-only builds.
        # Patching the query distinguishes that behavior from merely returning False.
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
        # ROCm reports device.type == "cuda"; patch torch.version.hip and make the architecture
        # query raise to prove HIP is rejected before querying the device.
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
        # The caller's accepted set belongs in the cache key because families differ. In
        # particular, (9, 10, 12) must reject Thor (SM 11.0).
        from unittest.mock import patch

        from torch._native.utils import capability as cap

        self.addCleanup(cap._arch_ok.cache_clear)
        cap._arch_ok.cache_clear()
        with FakeTensorMode():
            x = torch.empty(2, device="cuda")
        with patch.object(torch.cuda, "get_device_capability", return_value=(11, 0)):
            self.assertFalse(cap.device_ok(x, (9, 10, 12)))
            self.assertTrue(cap.device_ok(x, (11,)), "the (9,10,12) answer was reused")

    def test_on_current_device_never_raises(self):
        # Conditions must return False rather than raise, or the dispatcher cannot fall back.
        from torch._native.utils import capability as cap

        self.assertFalse(cap.on_current_device(torch.empty(2)))
        self.assertFalse(cap.on_current_device(torch.empty(2, device="meta")))
        if torch.cuda.is_available():
            self.assertTrue(cap.on_current_device(torch.empty(2, device="cuda")))

    @unittest.skipUnless(TEST_CUDA, "needs a CUDA tensor to compare device indices")
    @skipIfRocm
    def test_on_current_device_declines_another_device(self):
        # Reject non-current devices because kernel and stream caches bind to the current device.
        # This case is needed because CPU and meta inputs short-circuit before that check.
        from unittest.mock import patch

        from torch._native.utils import capability as cap

        x = torch.empty(2, device="cuda")
        self.assertTrue(cap.on_current_device(x))
        with patch.object(
            torch.cuda, "current_device", return_value=x.device.index + 1
        ):
            self.assertFalse(cap.on_current_device(x))

    def test_conds_never_raise_without_a_usable_device(self):
        # On CPU-only systems, fake CUDA tensors normalize to cuda:0 but device queries raise.
        # Both conditions must return False to preserve the dispatcher's fallback.
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
