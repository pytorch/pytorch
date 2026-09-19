# Owner(s): ["module: dynamo"]
from unittest.mock import MagicMock, patch

import torch
import torch.utils.backend_registration
from torch._dynamo.device_interface import (
    DeviceInterface,
    get_interface_for_device,
)
from torch.testing._internal.common_utils import HardwareClassification, TestCase


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

    def _reset_device_reg(self):
        """Reset device registration state so init_device_reg() re-runs."""
        import torch._dynamo.device_interface as di

        di._device_initialized = False
        di.device_interfaces.clear()

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
        get_interface_for_device should return the correct interface for both
        the device name and device:index."""
        DummyInterface = self._make_dummy_interface()

        self._reset_device_reg()
        try:
            p1, p2, p3 = self._setup_fakebackend(
                get_device_interface_fn=lambda: DummyInterface,
                device_count_fn=lambda: 1,
            )
            with p1, p2, p3:
                self.assertIs(get_interface_for_device("fakebackend"), DummyInterface)
                self.assertIs(
                    get_interface_for_device("fakebackend:0"), DummyInterface
                )
        finally:
            self._reset_device_reg()

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
        del mod.get_device_interface

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
        """When get_device_interface raises, the exception should not propagate
        to the caller of get_interface_for_device; it should raise
        NotImplementedError instead."""
        self._reset_device_reg()
        p1, p2, p3 = self._setup_fakebackend(
            get_device_interface_fn=MagicMock(
                side_effect=RuntimeError("driver missing")
            ),
        )
        with p1, p2, p3:
            with self.assertRaises(NotImplementedError):
                get_interface_for_device("fakebackend")

    def test_backend_device_count_raises(self):
        """When device_count raises, the main device is already registered
        but indexed devices are not; the exception should not propagate."""
        DummyInterface = self._make_dummy_interface()

        self._reset_device_reg()
        p1, p2, p3 = self._setup_fakebackend(
            get_device_interface_fn=lambda: DummyInterface,
            device_count_fn=MagicMock(side_effect=RuntimeError("driver error")),
        )
        with p1, p2, p3:
            self.assertIs(get_interface_for_device("fakebackend"), DummyInterface)
            with self.assertRaises(NotImplementedError):
                get_interface_for_device("fakebackend:0")

    def test_backend_returns_non_device_interface(self):
        """When get_device_interface returns a non-DeviceInterface subclass,
        get_interface_for_device should raise NotImplementedError."""
        self._reset_device_reg()
        p1, p2, p3 = self._setup_fakebackend(
            get_device_interface_fn=lambda: str,  # not a DeviceInterface subclass
        )
        with p1, p2, p3:
            with self.assertRaises(NotImplementedError):
                get_interface_for_device("fakebackend")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
