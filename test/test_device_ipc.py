# Owner(s): ["module: tests"]

"""Unit tests for RFC #192099 device IPC additions.

Covers:
  - torch._C._acc._is_privateuse1_ipc_supported()
  - New torch.UntypedStorage methods: _share_device_, _new_shared_device,
    _release_ipc_counter_device
  - Error paths that fire before any device-specific code
"""

import unittest

import torch
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


TEST_PRIVATEUSE1_IPC = hasattr(torch.UntypedStorage, "_share_device_") and hasattr(
    torch._C._acc, "_is_privateuse1_ipc_supported"
)


@unittest.skipIf(not TEST_PRIVATEUSE1_IPC, "PrivateUse1 IPC build not available")
class TestIsPrivateUse1IpcSupported(TestCase):
    """Tests for torch._C._acc._is_privateuse1_ipc_supported()."""

    hw_classification = HardwareClassification.GENERIC

    def test_returns_false_no_hooks(self):
        # assertIs checks both value (False) and type (bool), not just truthiness.
        # 0 would pass assertFalse but fail here.
        self.assertIs(torch._C._acc._is_privateuse1_ipc_supported(), False)


@unittest.skipIf(not TEST_PRIVATEUSE1_IPC, "PrivateUse1 IPC build not available")
class TestStorageSharingMethods(TestCase):
    """Tests that the new sharing methods are registered on torch.UntypedStorage."""

    hw_classification = HardwareClassification.GENERIC

    def test_share_device_method_exists(self):
        self.assertTrue(hasattr(torch.UntypedStorage, "_share_device_"))

    def test_new_shared_device_method_exists(self):
        self.assertTrue(hasattr(torch.UntypedStorage, "_new_shared_device"))

    def test_release_ipc_counter_device_method_exists(self):
        self.assertTrue(hasattr(torch.UntypedStorage, "_release_ipc_counter_device"))


@unittest.skipIf(not TEST_PRIVATEUSE1_IPC, "PrivateUse1 IPC build not available")
class TestDeviceIPCErrors(TestCase):
    """Tests error and safety behavior of the three new IPC methods."""

    hw_classification = HardwareClassification.GENERIC

    def test_share_device_no_hooks_raises(self):
        storage = torch.UntypedStorage(16)
        with self.assertRaisesRegex(
            RuntimeError, "PrivateUse1 hooks are not registered"
        ):
            storage._share_device_()

    def test_new_shared_device_no_hooks_raises(self):
        with self.assertRaisesRegex(
            RuntimeError, "PrivateUse1 hooks are not registered"
        ):
            torch.UntypedStorage._new_shared_device(
                0, b"handle", 16, 0, b"ref", 0, b"", False
            )

    def test_release_ipc_counter_device_nonexistent_handle(self):
        # Best-effort: missing shm file is silently ignored so B's process
        # continues even if A exited before B called this.
        torch.UntypedStorage._release_ipc_counter_device(
            b"nonexistent_shm_handle_xyz", 0
        )


if __name__ == "__main__":
    run_tests()
