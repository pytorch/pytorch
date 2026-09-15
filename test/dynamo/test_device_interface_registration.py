# Owner(s): ["module: dynamo", "module: PrivateUse1"]
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import torch
import torch.utils.backend_registration
from torch._dynamo.device_interface import DeviceInterface, get_interface_for_device
from torch._dynamo.test_case import TestCase
from torch.testing._internal.common_utils import HardwareClassification


class TestPrivateuse1DeviceInterface(TestCase):
    """
    Integration tests for privateuse1 device interface registration.

    These tests verify that when a privateuse1 backend is configured,
    get_interface_for_device() returns the correct interface, and various
    failure modes in the backend module do not crash the system.
    """

    hw_classification = HardwareClassification.GENERIC

    def _make_dummy_interface(self):
        """Create a dummy DeviceInterface subclass for testing."""

        class DummyInterface(DeviceInterface):
            pass

        return DummyInterface

    def _save_device_reg(self):
        """Snapshot the current device registration state."""
        import torch._dynamo.device_interface as di

        self._saved_initialized = di._device_initialized
        self._saved_initialization_in_progress = di._device_initialization_in_progress
        self._saved_interfaces = dict(di.device_interfaces)

    def _restore_device_reg(self):
        """Restore the saved device registration state."""
        import torch._dynamo.device_interface as di

        di.device_interfaces.clear()
        di.device_interfaces.update(self._saved_interfaces)
        di._device_initialized = self._saved_initialized
        di._device_initialization_in_progress = self._saved_initialization_in_progress
        # Clear the memoization cache so it picks up the restored registry
        from torch._dynamo.variables.user_defined import UserDefinedClassVariable

        UserDefinedClassVariable._in_graph_classes.cache_clear()

    def _reset_device_reg(self):
        """Reset device registration state so init_device_reg() re-runs."""
        import torch._dynamo.device_interface as di

        di._device_initialized = False
        di._device_initialization_in_progress = False
        di.device_interfaces.clear()
        from torch._dynamo.variables.user_defined import UserDefinedClassVariable

        UserDefinedClassVariable._in_graph_classes.cache_clear()

    def setUp(self):
        super().setUp()
        self._save_device_reg()

    def tearDown(self):
        self._restore_device_reg()
        super().tearDown()

    def _setup_fakebackend(self, get_device_interface_fn, device_count_fn=lambda: 0):
        """Set up a fake backend module on torch with the given get_device_interface
        and device_count, and patch _get_privateuse1_backend_name to return its name."""
        mod = MagicMock()
        mod.get_device_interface = get_device_interface_fn
        mod.device_count = device_count_fn
        _pu1_patch = patch.object(
            torch._C,
            "_get_privateuse1_backend_name",
            return_value="fakebackend",
        )
        # backend_registration imports this C binding by name, so patch both
        # references to keep the fake backend consistent across the two modules.
        _pu1_br_patch = patch.object(
            torch.utils.backend_registration,
            "_get_privateuse1_backend_name",
            return_value="fakebackend",
        )
        return (
            _pu1_patch,
            _pu1_br_patch,
            patch.object(torch, "fakebackend", mod, create=True),
        )

    def _patch_no_backend(self, backend_name="privateuseone"):
        """Patch both _get_privateuse1_backend_name references."""
        return (
            patch.object(
                torch._C,
                "_get_privateuse1_backend_name",
                return_value=backend_name,
            ),
            patch.object(
                torch.utils.backend_registration,
                "_get_privateuse1_backend_name",
                return_value=backend_name,
            ),
        )

    def test_no_backend_registered(self):
        """When no privateuse1 backend is set, get_interface_for_device should
        raise NotImplementedError for the privateuse1 device."""
        self._reset_device_reg()
        p1, p2 = self._patch_no_backend()
        with p1, p2:
            with self.assertRaises(NotImplementedError):
                get_interface_for_device("privateuseone")

    def test_backend_registers_interface(self):
        """When a privateuse1 backend is properly configured,
        get_interface_for_device should return the correct interface for the
        device name."""
        DummyInterface = self._make_dummy_interface()

        self._reset_device_reg()
        p1, p2, p3 = self._setup_fakebackend(
            get_device_interface_fn=lambda: DummyInterface,
            device_count_fn=lambda: 1,
        )
        with p1, p2, p3:
            self.assertIs(get_interface_for_device("fakebackend"), DummyInterface)

    def test_backend_missing_module(self):
        """When the backend name is set but no module is registered on torch,
        get_interface_for_device should raise NotImplementedError."""
        self._reset_device_reg()
        p1, p2 = self._patch_no_backend("fakebackend")
        with p1, p2:
            with patch.object(torch, "fakebackend", None, create=True):
                with self.assertRaises(NotImplementedError):
                    get_interface_for_device("fakebackend")

    def test_backend_missing_get_device_interface(self):
        """When the backend module exists but lacks get_device_interface,
        get_interface_for_device should raise NotImplementedError."""
        mod = MagicMock(spec=[])

        self._reset_device_reg()
        p1, p2 = self._patch_no_backend("fakebackend")
        with p1, p2:
            with patch.object(torch, "fakebackend", mod, create=True):
                with self.assertRaises(NotImplementedError):
                    get_interface_for_device("fakebackend")

    def test_backend_get_device_interface_returns_none(self):
        """When get_device_interface returns None,
        get_interface_for_device should raise NotImplementedError."""
        self._reset_device_reg()
        p1, p2, p3 = self._setup_fakebackend(
            get_device_interface_fn=lambda: None,
        )
        with p1, p2, p3:
            with self.assertRaises(NotImplementedError):
                get_interface_for_device("fakebackend")

    def test_backend_get_device_interface_raises(self):
        """Hook failures warn but do not escape from registry initialization."""
        for error in (
            RuntimeError("driver missing"),
            ValueError("invalid interface"),
        ):
            with self.subTest(error=error):
                self._reset_device_reg()
                p1, p2, p3 = self._setup_fakebackend(
                    get_device_interface_fn=MagicMock(side_effect=error),
                )
                with p1, p2, p3:
                    with self.assertWarnsRegex(
                        UserWarning, r"get_device_interface.*raised"
                    ):
                        with self.assertRaises(NotImplementedError):
                            get_interface_for_device("fakebackend")

    def test_backend_device_count_is_not_queried(self):
        """DeviceInterface registration only needs the bare device type."""
        DummyInterface = self._make_dummy_interface()
        device_count = MagicMock(side_effect=RuntimeError("driver error"))

        self._reset_device_reg()
        p1, p2, p3 = self._setup_fakebackend(
            get_device_interface_fn=lambda: DummyInterface,
            device_count_fn=device_count,
        )
        with p1, p2, p3:
            self.assertIs(get_interface_for_device("fakebackend"), DummyInterface)
        device_count.assert_not_called()

    def test_backend_returns_non_device_interface(self):
        """When get_device_interface returns a non-DeviceInterface subclass,
        get_interface_for_device should raise NotImplementedError."""
        self._reset_device_reg()
        p1, p2, p3 = self._setup_fakebackend(
            get_device_interface_fn=lambda: str,  # not a DeviceInterface subclass
        )
        with p1, p2, p3:
            with self.assertWarnsRegex(
                UserWarning, r"is not a DeviceInterface subclass"
            ):
                with self.assertRaises(NotImplementedError):
                    get_interface_for_device("fakebackend")

    def test_hook_can_reenter_device_registration(self):
        """A hook may import code that reads the registry on the same thread."""
        from torch._dynamo.device_interface import get_registered_device_interfaces

        DummyInterface = self._make_dummy_interface()
        hook_calls = MagicMock()

        def get_device_interface():
            hook_calls()
            tuple(get_registered_device_interfaces())
            return DummyInterface

        self._reset_device_reg()
        p1, p2, p3 = self._setup_fakebackend(get_device_interface)
        with p1, p2, p3:
            self.assertIs(get_interface_for_device("fakebackend"), DummyInterface)
        hook_calls.assert_called_once_with()

    def test_hook_is_serialized_across_threads(self):
        """Concurrent first lookups wait for one hook invocation to finish."""
        DummyInterface = self._make_dummy_interface()
        hook_started = threading.Event()
        allow_hook_to_finish = threading.Event()
        second_started = threading.Event()
        second_finished = threading.Event()
        hook_calls = MagicMock()

        def get_device_interface():
            hook_calls()
            hook_started.set()
            if not allow_hook_to_finish.wait(timeout=5):
                raise RuntimeError("timed out waiting to finish hook")
            return DummyInterface

        def second_lookup():
            second_started.set()
            try:
                return get_interface_for_device("fakebackend")
            finally:
                second_finished.set()

        self._reset_device_reg()
        p1, p2, p3 = self._setup_fakebackend(get_device_interface)
        with p1, p2, p3, ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(get_interface_for_device, "fakebackend")
            self.assertTrue(hook_started.wait(timeout=5))
            second = pool.submit(second_lookup)
            self.assertTrue(second_started.wait(timeout=5))
            try:
                self.assertFalse(second_finished.wait(timeout=0.1))
                self.assertEqual(hook_calls.call_count, 1)
            finally:
                allow_hook_to_finish.set()
            self.assertIs(first.result(timeout=5), DummyInterface)
            self.assertIs(second.result(timeout=5), DummyInterface)

        hook_calls.assert_called_once_with()

    def test_import_torch_does_not_import_dynamo(self):
        subprocess.check_call(
            [
                sys.executable,
                "-c",
                "import sys; import torch; assert 'torch._dynamo' not in sys.modules",
            ]
        )

    def test_direct_registration_not_clobbered_by_hook(self):
        """When a backend has already registered its interface directly via
        register_interface_for_device, the lazy hook should NOT overwrite it
        during init_device_reg()."""
        DirectInterface = self._make_dummy_interface()
        HookInterface = self._make_dummy_interface()

        self._reset_device_reg()
        # Simulate direct registration (as openreg does at import time)
        from torch._dynamo.device_interface import register_interface_for_device

        register_interface_for_device("fakebackend", DirectInterface)
        p1, p2, p3 = self._setup_fakebackend(
            get_device_interface_fn=lambda: HookInterface,
            device_count_fn=lambda: 1,
        )
        with p1, p2, p3:
            # init_device_reg() should skip the hook because the backend
            # is already registered
            result = get_interface_for_device("fakebackend")
            self.assertIs(result, DirectInterface)
            self.assertIsNot(result, HookInterface)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
