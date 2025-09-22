# Owner(s): ["module: PrivateUse1"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestDLPack(TestCase):
    def test_open_device_dlpack(self):
        x_in = torch.randn(2, 3).to("openreg")
        capsule = torch.utils.dlpack.to_dlpack(x_in)
        x_out = torch.from_dlpack(capsule)
        self.assertTrue(x_out.device == x_in.device)

        x_in = x_in.to("cpu")
        x_out = x_out.to("cpu")
        self.assertEqual(x_in, x_out)


if __name__ == "__main__":
    run_tests()
