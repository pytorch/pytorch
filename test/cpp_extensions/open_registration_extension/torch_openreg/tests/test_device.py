# Owner(s): ["module: PrivateUse1"]

import torch
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.testing._internal.common_utils import run_tests, TestCase


class TestDevice(TestCase):
    def test_device_count(self):
        count = torch.accelerator.device_count()
        self.assertEqual(count, 2)

    def test_device_switch(self):
        torch.accelerator.set_device_index(1)
        self.assertEqual(torch.accelerator.current_device_index(), 1)

        torch.accelerator.set_device_index(0)
        self.assertEqual(torch.accelerator.current_device_index(), 0)

    def test_device_context(self):
        device = torch.accelerator.current_device_index()
        with torch.accelerator.device_index(None):
            self.assertEqual(torch.accelerator.current_device_index(), device)
        self.assertEqual(torch.accelerator.current_device_index(), device)

        with torch.accelerator.device_index(1):
            self.assertEqual(torch.accelerator.current_device_index(), 1)
        self.assertEqual(torch.accelerator.current_device_index(), device)

    def test_invalid_device_index(self):
        with self.assertRaisesRegex(RuntimeError, "The device index is out of range"):
            torch.accelerator.set_device_index(2)

    def test_device_capability(self):
        capability = torch.accelerator.get_device_capability("openreg:0")
        supported_dtypes = capability["supported_dtypes"]
        expected_dtypes = get_all_dtypes(include_complex32=True, include_qint=True)

        self.assertTrue(all(dtype in supported_dtypes for dtype in expected_dtypes))


if __name__ == "__main__":
    run_tests()
