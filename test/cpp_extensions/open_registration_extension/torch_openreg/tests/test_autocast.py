# Owner(s): ["module: PrivateUse1"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestAutocast(TestCase):
    def test_autocast_with_unsupported_type(self):
        with self.assertWarnsRegex(
            UserWarning,
            "In openreg autocast, but the target dtype is not supported. Disabling autocast.\n"
            "openreg Autocast only supports dtypes of torch.float16, torch.bfloat16 currently.",
        ):
            with torch.autocast(device_type="openreg", dtype=torch.float32):
                _ = torch.ones(10)

    def test_autocast_operator_not_supported(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.",
        ):
            x = torch.randn(2, 3, device="openreg")
            y = torch.randn(2, 3, device="openreg")
            with torch.autocast(device_type="openreg", dtype=torch.float16):
                _ = torch.nn.functional.binary_cross_entropy(x, y)

    def test_autocast_low_precision(self):
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, 3, device="openreg")
            y = torch.randn(3, 3, device="openreg")
            result = torch.mm(x, y)
            self.assertEqual(result.dtype, torch.float16)

    def test_autocast_fp32(self):
        with torch.amp.autocast(device_type="openreg"):
            x = torch.randn(2, device="openreg", dtype=torch.float16)
            result = torch.asin(x)
            self.assertEqual(result.dtype, torch.float32)

    def test_autocast_default_dtype(self):
        openreg_fast_dtype = torch.get_autocast_dtype(device_type="openreg")
        self.assertEqual(openreg_fast_dtype, torch.half)

    def test_autocast_set_dtype(self):
        for dtype in [torch.float16, torch.bfloat16]:
            torch.set_autocast_dtype("openreg", dtype)
            self.assertEqual(torch.get_autocast_dtype("openreg"), dtype)


if __name__ == "__main__":
    run_tests()
