# Owner(s): ["module: prim"]
#
# Verifies that graph-safe RNG higher-order operators (run_and_save_rng_state
# and run_with_rng_state) correctly dispatch to dynamically registered
# PrivateUse1 backends (e.g., torch_npu) instead of hitting the hardcoded
# device whitelist AssertionError.

import types
import unittest
from unittest.mock import patch, MagicMock

import torch
from torch._C import DispatchKey
from torch._prims import rng_prims


# ---------------------------------------------------------------------------
# Fake Torch Module Setup
# ---------------------------------------------------------------------------
# PyTorch's C++ built-in functions (like torch._C._get_privateuse1_backend_name)
# cannot be directly mocked via unittest.mock. To bypass this, we inject a fake
# torch module into the rng_prims namespace. This intercepts all internal calls
# to torch._C and getattr(torch, ...) within rng_prims.py.
# ---------------------------------------------------------------------------
_fake_torch = types.ModuleType("fake_torch_for_rng_prims")
_fake_torch.__dict__.update(torch.__dict__)


class _FakeC:
    @staticmethod
    def _get_privateuse1_backend_name():
        # Simulate a third-party backend registered via
        # torch.utils.rename_privateuse1_backend("fake_device")
        return "fake_device"


_fake_torch._C = _FakeC
_fake_torch.fake_device = MagicMock()


class TestRngPrimsPrivateUse1(unittest.TestCase):
    """
    Test suite for PrivateUse1 backend support in torch._prims.rng_prims.

    Validates that the dynamic dispatch logic correctly routes graph-safe RNG
    operations to out-of-tree backends without triggering hardcoded whitelist
    assertion errors.
    """

    @classmethod
    def setUpClass(cls):
        # Inject the fake torch module into rng_prims to override C++ bindings
        cls._original_torch = rng_prims.torch
        rng_prims.torch = _fake_torch

        # Extract the BackendSelect Python implementations for the HOPs.
        # Wrapped in staticmethod to prevent Python's descriptor protocol from
        # implicitly passing `self` (the TestCase instance) as the first argument.
        cls.run_and_save_impl = staticmethod(
            rng_prims.run_and_save_rng_state.py_kernels[DispatchKey.BackendSelect]
        )
        cls.run_with_impl = staticmethod(
            rng_prims.run_with_rng_state.py_kernels[DispatchKey.BackendSelect]
        )

    @classmethod
    def tearDownClass(cls):
        # Restore the original torch reference to prevent test pollution
        rng_prims.torch = cls._original_torch
        _fake_torch.fake_device.reset_mock()

    def setUp(self):
        # Reset mock states before each test to ensure isolation
        _fake_torch.fake_device.reset_mock()
        _fake_torch.fake_device.get_rng_state.return_value = b"fake_state_123"

    def test_registration_and_dispatch(self):
        """
        End-to-End Validation: Verifies that the simulated PrivateUse1 backend
        registration is effective and that run_and_save_rng_state correctly
        routes to the backend's get_rng_state.

        This combines registration verification and dispatch routing into a
        single behavioral test, avoiding redundant implementation-detail checks.
        """
        # Verify the simulated registration is active
        self.assertEqual(
            _fake_torch._C._get_privateuse1_backend_name(), "fake_device"
        )

        def dummy_op(x, *args, **kwargs):
            return x * 2

        with patch(
            "torch._prims.rng_prims.get_device", return_value="fake_device"
        ):
            state, res = self.run_and_save_impl(
                dummy_op, torch.tensor(3.0), device="fake_device"
            )

        self.assertEqual(state, b"fake_state_123")
        self.assertEqual(res.item(), 6.0)
        _fake_torch.fake_device.get_rng_state.assert_called_once()

    @patch("torch._prims.rng_prims.get_device", return_value="fake_device")
    def test_run_with_dispatch_and_restore(self, mock_get_device):
        """
        Validates that run_with_rng_state correctly sets the new RNG state
        before executing the operation, and restores the original state
        immediately after the operation completes.
        """
        _fake_torch.fake_device.get_rng_state.return_value = b"original_state"

        def dummy_op(*args, **kwargs):
            return "op_result"

        res = self.run_with_impl(
            b"new_state", dummy_op, "dummy_arg", device="fake_device"
        )

        self.assertEqual(res, "op_result")
        # Verify state was set to the new state, then restored to the original
        self.assertEqual(_fake_torch.fake_device.set_rng_state.call_count, 2)
        _fake_torch.fake_device.set_rng_state.assert_any_call(b"new_state")
        _fake_torch.fake_device.set_rng_state.assert_any_call(b"original_state")

    @patch("torch._prims.rng_prims.get_device", return_value="fake_device")
    def test_run_with_exception_safety(self, mock_get_device):
        """
        Validates the exception safety of the try...finally block in
        _privateuse1_run_with_rng_state.

        Ensures that even if the inner operation raises an exception, the
        original RNG state is guaranteed to be restored, preventing state
        corruption in compiled graphs.
        """
        _fake_torch.fake_device.get_rng_state.return_value = b"original_state"

        def failing_op(*args, **kwargs):
            raise RuntimeError("Simulated op failure")

        with self.assertRaises(RuntimeError):
            self.run_with_impl(
                b"new_state", failing_op, "dummy_arg", device="fake_device"
            )

        # Crucial check: finally block must execute despite the RuntimeError
        self.assertEqual(_fake_torch.fake_device.set_rng_state.call_count, 2)
        _fake_torch.fake_device.set_rng_state.assert_any_call(b"new_state")
        _fake_torch.fake_device.set_rng_state.assert_any_call(b"original_state")

    def test_privateuse1_helper_functions(self):
        """
        Directly tests the newly added helper functions
        _privateuse1_save_rng_state and _privateuse1_run_with_rng_state
        to ensure they correctly interact with the dynamically resolved
        device module via getattr(torch, backend_name).
        """

        def dummy_op(x):
            return x + 1

        # Test save semantics
        state, res = rng_prims._privateuse1_save_rng_state(dummy_op, 1)
        self.assertEqual(state, b"fake_state_123")
        self.assertEqual(res, 2)

        # Test run/restore semantics
        _fake_torch.fake_device.get_rng_state.return_value = b"orig"
        res = rng_prims._privateuse1_run_with_rng_state(b"new", dummy_op, 2)
        self.assertEqual(res, 3)
        _fake_torch.fake_device.set_rng_state.assert_any_call(b"new")
        _fake_torch.fake_device.set_rng_state.assert_any_call(b"orig")


if __name__ == "__main__":
    unittest.main()
