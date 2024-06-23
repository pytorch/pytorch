# Owner(s): ["module: dynamo"]
import unittest
from unittest.mock import Mock

import torch

import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.device_interface import CudaInterface, DeviceGuard
from torch.testing._internal.common_cuda import TEST_CUDA, TEST_MULTIGPU


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


@unittest.skipIf(not TEST_CUDA, "No CUDA available.")
class TestCUDADeviceGuard(torch._dynamo.test_case.TestCase):
    """
    Unit tests for the DeviceGuard class using a CudaInterface.
    """

    def setUp(self):
        super().setUp()
        self.device_interface = CudaInterface

    @unittest.skipIf(not TEST_MULTIGPU, "need multiple GPU")
    def test_device_guard(self):
        current_device = torch.cuda.current_device()

        device_guard = DeviceGuard(self.device_interface, 1)

        with device_guard as _:
            self.assertEqual(torch.cuda.current_device(), 1)
            self.assertEqual(device_guard.prev_idx, 0)
            self.assertEqual(device_guard.idx, 1)

        self.assertEqual(torch.cuda.current_device(), current_device)
        self.assertEqual(device_guard.prev_idx, 0)
        self.assertEqual(device_guard.idx, 1)

    def test_device_guard_no_index(self):
        current_device = torch.cuda.current_device()

        device_guard = DeviceGuard(self.device_interface, None)

        with device_guard as _:
            self.assertEqual(torch.cuda.current_device(), current_device)
            self.assertEqual(device_guard.prev_idx, -1)
            self.assertEqual(device_guard.idx, None)

        self.assertEqual(device_guard.prev_idx, -1)
        self.assertEqual(device_guard.idx, None)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
