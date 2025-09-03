import torch
import torch_openreg

from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase

class TestTorchAutocast(TestCase):
    def test_autocast_fast_dtype(self):
        openreg_fast_dtype = torch.get_autocast_dtype(device_type="openreg")
        self.assertEqual(openreg_fast_dtype, torch.half)

if __name__ == "__main__":
    run_tests()