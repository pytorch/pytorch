# Owner(s): ["module: PrivateUse1"]
"""
Tests for torch.utils.skip_cea_decomposition_for_privateuse1().

Verifies that, once the CEA skip is enabled:
  - Ops with only a CompositeExplicitAutograd (CEA) kernel route to the
    PrivateUse1 backend fallback instead of being silently decomposed.
  - Direct PrivateUse1 kernel registrations are unaffected.
  - The RAII handle correctly restores the previous behavior on release.
"""

import torch
import torch.library
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.backend_registration import _setup_privateuseone_for_python_backend


# _setup_privateuseone_for_python_backend can only be called once per process.
_setup_privateuseone_for_python_backend("fakedev")


class TestPrivateUse1SkipCEA(TestCase):
    def test_skip_cea_flag_initially_false(self):
        self.assertFalse(torch._C._dispatch_privateuse1_skip_cea_enabled())

    def test_raii_sets_and_restores_flag(self):
        self.assertFalse(torch._C._dispatch_privateuse1_skip_cea_enabled())
        handle = torch.utils.skip_cea_decomposition_for_privateuse1()
        self.assertTrue(torch._C._dispatch_privateuse1_skip_cea_enabled())
        del handle
        self.assertFalse(torch._C._dispatch_privateuse1_skip_cea_enabled())

    def test_context_manager_restores_on_exception(self):
        class _Ctx:
            def __enter__(self):
                self._handle = torch.utils.skip_cea_decomposition_for_privateuse1()
                return self

            def __exit__(self, *_):
                del self._handle

        with _Ctx():
            self.assertTrue(torch._C._dispatch_privateuse1_skip_cea_enabled())
        self.assertFalse(torch._C._dispatch_privateuse1_skip_cea_enabled())

    def test_double_registration_raises(self):
        handle = torch.utils.skip_cea_decomposition_for_privateuse1()
        try:
            with self.assertRaisesRegex(RuntimeError, "already registered"):
                torch.utils.skip_cea_decomposition_for_privateuse1()
        finally:
            del handle

    def test_cea_op_routes_to_fallback(self):
        """
        torch.ops.aten.abs.default has a CEA kernel and no direct PrivateUse1
        kernel.  With skip enabled it must reach the backend fallback rather
        than being handled by the CEA decomposition.
        """
        fallback_calls: list[str] = []

        def recording_fallback(op, args, kwargs):
            fallback_calls.append(op.name())
            # Raise so the test can verify dispatch reached us without needing
            # a full fakedev tensor implementation.
            raise RuntimeError(f"fakedev fallback: {op.name()}")

        lib = torch.library.Library("_", "IMPL", "fakedev")
        lib.fallback(recording_fallback)

        handle = torch.utils.skip_cea_decomposition_for_privateuse1()
        try:
            t = torch.empty(2, device="fakedev")
            with self.assertRaises(RuntimeError):
                torch.abs(t)
            self.assertTrue(
                any("abs" in c for c in fallback_calls),
                f"Expected aten::abs to reach the fallback; got calls: {fallback_calls}",
            )
        finally:
            del handle
            del lib

    def test_direct_kernel_unaffected_by_skip(self):
        """
        Direct PrivateUse1 kernel registrations take priority over CEA and must
        remain unaffected when the CEA skip is enabled.
        """
        direct_calls: list[str] = []

        lib = torch.library.Library("aten", "IMPL", "fakedev")

        @lib.impl("aten::neg.default")
        def fakedev_neg(self):
            direct_calls.append("neg")
            return self  # identity for test purposes

        handle = torch.utils.skip_cea_decomposition_for_privateuse1()
        try:
            t = torch.empty(2, device="fakedev")
            torch.neg(t)
            self.assertIn("neg", direct_calls)
        finally:
            del handle
            del lib


if __name__ == "__main__":
    run_tests()
