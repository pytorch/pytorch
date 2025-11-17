# Owner(s): ["module: PrivateUse1"]

import torch
import torch_openreg  # noqa: F401
from torch.testing._internal.common_utils import run_tests, TestCase


class TestDevice(TestCase):
    def test_device_count(self):
        """Test device count query"""
        count = torch.accelerator.device_count()
        self.assertEqual(count, 2)

    def test_device_switch(self):
        """Test switching between devices"""
        torch.accelerator.set_device_index(1)
        self.assertEqual(torch.accelerator.current_device_index(), 1)

        torch.accelerator.set_device_index(0)
        self.assertEqual(torch.accelerator.current_device_index(), 0)

    def test_device_context(self):
        """Test device context manager"""
        device = torch.accelerator.current_device_index()
        with torch.accelerator.device_index(None):
            self.assertEqual(torch.accelerator.current_device_index(), device)
        self.assertEqual(torch.accelerator.current_device_index(), device)

        with torch.accelerator.device_index(1):
            self.assertEqual(torch.accelerator.current_device_index(), 1)
        self.assertEqual(torch.accelerator.current_device_index(), device)

    def test_invalid_device_index(self):
        """Test error handling for invalid device index"""
        with self.assertRaisesRegex(RuntimeError, "The device index is out of range"):
            torch.accelerator.set_device_index(2)

    def test_device_properties(self):
        """Test device properties"""
        device = torch.device("openreg:0")
        self.assertEqual(device.type, "openreg")
        self.assertEqual(device.index, 0)
        
        device = torch.device("openreg")
        self.assertEqual(device.type, "openreg")
        self.assertIsNone(device.index)

    def test_tensor_device(self):
        """Test tensor device assignment"""
        x = torch.randn(2, 3, device="openreg")
        self.assertEqual(x.device.type, "openreg")
        
        x = torch.randn(2, 3, device="openreg:1")
        self.assertEqual(x.device.type, "openreg")
        self.assertEqual(x.device.index, 1)

    def test_device_guard(self):
        """Test device guard context manager"""
        original_device = torch.accelerator.current_device_index()
        
        with torch.accelerator.device_index(1):
            self.assertEqual(torch.accelerator.current_device_index(), 1)
        
        self.assertEqual(torch.accelerator.current_device_index(), original_device)

    def test_device_switch_persistence(self):
        """Test that device switch persists across operations"""
        torch.accelerator.set_device_index(1)
        x = torch.randn(2, 3, device="openreg")
        self.assertEqual(x.device.index, 1)
        
        y = torch.randn(3, 3, device="openreg")
        self.assertEqual(y.device.index, 1)

    def test_device_count_consistency(self):
        """Test device count consistency"""
        count = torch.accelerator.device_count()
        self.assertGreater(count, 0)
        self.assertLess(count, 10)  # Reasonable upper bound
        
        # Test that we can access all devices
        for i in range(count):
            torch.accelerator.set_device_index(i)
            self.assertEqual(torch.accelerator.current_device_index(), i)


if __name__ == "__main__":
    run_tests()
