# Owner(s): ["module: PrivateUse1"]

import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestDLPack(TestCase):
    def test_open_device_dlpack(self):
        """Test DLPack conversion for openreg device"""
        x_in = torch.randn(2, 3).to("openreg")
        capsule = torch.utils.dlpack.to_dlpack(x_in)
        x_out = torch.from_dlpack(capsule)
        self.assertTrue(x_out.device == x_in.device)

        x_in = x_in.to("cpu")
        x_out = x_out.to("cpu")
        self.assertEqual(x_in, x_out)

    def test_dlpack_roundtrip(self):
        """Test DLPack roundtrip conversion"""
        x = torch.randn(2, 3, device="openreg")
        capsule = torch.utils.dlpack.to_dlpack(x)
        y = torch.from_dlpack(capsule)

        self.assertEqual(x.device, y.device)
        self.assertEqual(x, y)

    def test_dlpack_different_shapes(self):
        """Test DLPack with different tensor shapes"""
        shapes = [(1,), (2, 3), (4, 5, 6), (1, 2, 3, 4)]

        for shape in shapes:
            x = torch.randn(*shape, device="openreg")
            capsule = torch.utils.dlpack.to_dlpack(x)
            y = torch.from_dlpack(capsule)

            self.assertEqual(x.shape, y.shape)
            self.assertEqual(x, y)

    @unittest.skip("Abs kernel only supports float type when assertEuqal")
    def test_dlpack_different_dtypes(self):
        """Test DLPack with different dtypes"""
        dtypes = [torch.float32, torch.float16, torch.int32, torch.int64]

        for dtype in dtypes:
            x = torch.randn(2, 3, device="openreg", dtype=dtype)
            capsule = torch.utils.dlpack.to_dlpack(x)
            y = torch.from_dlpack(capsule)

            self.assertEqual(x.dtype, y.dtype)
            self.assertEqual(x, y)

    def test_dlpack_cross_device(self):
        """Test DLPack conversion across devices"""
        x_cpu = torch.randn(2, 3)
        x_openreg = x_cpu.to("openreg")

        capsule = torch.utils.dlpack.to_dlpack(x_openreg)
        y = torch.from_dlpack(capsule)

        self.assertEqual(y.device.type, "openreg")
        self.assertEqual(x_cpu, y.cpu())

    def test_dlpack_non_contiguous(self):
        """Test DLPack with non-contiguous tensors"""
        x = torch.randn(3, 4, device="openreg")
        x_t = x.t()  # Transpose creates non-contiguous tensor

        capsule = torch.utils.dlpack.to_dlpack(x_t)
        y = torch.from_dlpack(capsule)

        self.assertEqual(x_t.shape, y.shape)
        self.assertEqual(x_t, y)


if __name__ == "__main__":
    run_tests()
