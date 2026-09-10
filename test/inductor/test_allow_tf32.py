# Owner(s): ["module: inductor"]

import torch
from torch._dynamo.device_interface import (
    CudaInterface,
    DeviceInterface,
    XpuInterface,
)
from torch._inductor.test_case import run_tests, TestCase


class TestAllowTf32(TestCase):
    def test_device_interface_default_false(self):
        self.assertFalse(DeviceInterface.allow_tf32())
        self.assertFalse(DeviceInterface.allow_tf32(size_threshold=True))

    def test_cuda_allow_tf32_respects_threshold(self):
        orig = torch.backends.cuda.matmul.fp32_precision
        try:
            torch.backends.cuda.matmul.fp32_precision = "tf32"
            self.assertTrue(CudaInterface.allow_tf32(size_threshold=True))
            self.assertFalse(CudaInterface.allow_tf32(size_threshold=False))
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            self.assertFalse(CudaInterface.allow_tf32(size_threshold=True))
        finally:
            torch.backends.cuda.matmul.fp32_precision = orig

    def test_xpu_allow_tf32_matches_mkldnn(self):
        expected = torch.backends.mkldnn.allow_tf32
        self.assertEqual(XpuInterface.allow_tf32(size_threshold=False), expected)
        self.assertEqual(XpuInterface.allow_tf32(size_threshold=True), expected)


if __name__ == "__main__":
    run_tests()
