# Owner(s): ["module: PrivateUse1"]

import torch
import torch_openreg  # noqa: F401
from torch.testing._internal.common_utils import run_tests, TestCase


class TestAutocastoOpenReg(TestCase):
    def test_openreg_autocast_error_message(self):
        with self.assertWarnsRegex(
            UserWarning,
            "In openreg autocast, but the target dtype torch.float32 is not supported.",
        ):
            with torch.autocast(device_type="openreg", dtype=torch.float32):
                _ = torch.ones(10)

    def test_openreg_autocast_float16_supported(self):
        with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
            x = torch.randn(2, 3, device="openreg")
            y = torch.randn(3, 3, device="openreg")
            result = torch.mm(x, y)
            self.assertEqual(result.dtype, torch.float16)

    def test_openreg_autocast_float32_supported(self):
        with torch.amp.autocast(device_type="openreg"):
            x = torch.randn(2, device="openreg", dtype=torch.float16)
            y = torch.randn(2, device="openreg", dtype=torch.float16)
            result = torch.dot(x, y)
            self.assertEqual(result.dtype, torch.float32)

    def test_autocast_fast_dtype(self):
        openreg_fast_dtype = torch.get_autocast_dtype(device_type="openreg")
        self.assertEqual(openreg_fast_dtype, torch.half)


if __name__ == "__main__":
    run_tests()
