# Owner(s): ["module: dynamo"]
from unittest.mock import Mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.device_interface import DeviceGuard, get_interface_for_device
from torch.testing._internal.common_device_type import instantiate_device_type_tests


class TestDeviceGuard(torch._dynamo.test_case.TestCase):
    """
    Unit tests for the DeviceGuard class using a mock DeviceInterface.
    """

    def setUp(self):
        super().setUp()
        self.device_interface = Mock()

        self.device_interface.exchange_device = Mock(return_value=0)
        self.device_interface.maybe_exchange_device = Mock(return_value=1)

    def test_device_guard(self):
        device_guard = DeviceGuard(self.device_interface, 1)

        with device_guard as _:
            self.device_interface.exchange_device.assert_called_once_with(1)
            self.assertEqual(device_guard.prev_idx, 0)
            self.assertEqual(device_guard.idx, 1)

        self.device_interface.maybe_exchange_device.assert_called_once_with(0)
        self.assertEqual(device_guard.prev_idx, 0)
        self.assertEqual(device_guard.idx, 1)

    def test_device_guard_no_index(self):
        device_guard = DeviceGuard(self.device_interface, None)

        with device_guard as _:
            self.device_interface.exchange_device.assert_not_called()
            self.assertEqual(device_guard.prev_idx, -1)
            self.assertEqual(device_guard.idx, None)

        self.device_interface.maybe_exchange_device.assert_not_called()
        self.assertEqual(device_guard.prev_idx, -1)
        self.assertEqual(device_guard.idx, None)


class TestDeviceGuardWithInterface(torch._dynamo.test_case.TestCase):
    """
    Unit tests for the DeviceGuard class using a real DeviceInterface.
    """

    def test_device_guard_no_index(self, device):
        device_interface = get_interface_for_device(torch.device(device).type)
        current_device = device_interface.current_device()

        device_guard = DeviceGuard(device_interface, None)

        with device_guard as _:
            self.assertEqual(device_interface.current_device(), current_device)
            self.assertEqual(device_guard.prev_idx, -1)
            self.assertEqual(device_guard.idx, None)

        self.assertEqual(device_guard.prev_idx, -1)
        self.assertEqual(device_guard.idx, None)


instantiate_device_type_tests(
    TestDeviceGuardWithInterface, globals(), allow_mps=True, allow_xpu=True
)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
