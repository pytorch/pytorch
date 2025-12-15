# Owner(s): ["module: dynamo"]
import unittest
from unittest.mock import Mock

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.device_interface import CudaInterface, DeviceGuard, XpuInterface
from torch.testing._internal.common_utils import TEST_CUDA, TEST_XPU


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
TEST_GPU = TEST_CUDA or TEST_XPU
TEST_MULTIGPU = torch.cuda.device_count() >= 2 or torch.xpu.device_count() >= 2


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


@unittest.skipIf(not TEST_GPU, "No GPU available.")
class TestGPUDeviceGuard(torch._dynamo.test_case.TestCase):
    """
    Unit tests for the DeviceGuard class using a GPU=Interface.
    """

    def setUp(self):
        super().setUp()
        if device_type == "cuda":
            self.device_interface = CudaInterface
        elif device_type == "xpu":
            self.device_interface = XpuInterface
        else:
            raise ValueError("Not supported GPU type")

    @unittest.skipIf(not TEST_MULTIGPU, "need multiple GPU")
    def test_device_guard(self):
        current_device = torch.accelerator.current_device_index()

        device_guard = DeviceGuard(self.device_interface, 1)

        with device_guard as _:
            self.assertEqual(torch.accelerator.current_device_index(), 1)
            self.assertEqual(device_guard.prev_idx, 0)
            self.assertEqual(device_guard.idx, 1)

        self.assertEqual(torch.accelerator.current_device_index(), current_device)
        self.assertEqual(device_guard.prev_idx, 0)
        self.assertEqual(device_guard.idx, 1)

    def test_device_guard_no_index(self):
        current_device = torch.accelerator.current_device_index()

        device_guard = DeviceGuard(self.device_interface, None)

        with device_guard as _:
            self.assertEqual(torch.accelerator.current_device_index(), current_device)
            self.assertEqual(device_guard.prev_idx, -1)
            self.assertEqual(device_guard.idx, None)

        self.assertEqual(device_guard.prev_idx, -1)
        self.assertEqual(device_guard.idx, None)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
