# Owner(s): ["module: PrivateUse1"]

import torch
import torch_openreg  # noqa: F401
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

    def test_device_capability(self):
        capability = torch.accelerator.get_device_capability("openreg:0")
        supported_dtypes = capability["supported_dtypes"]
        expected_dtypes = [torch.float32, torch.float64, torch.int32, torch.int64]

        self.assertTrue(all(dtype in supported_dtypes for dtype in expected_dtypes))


if __name__ == "__main__":
    run_tests()
