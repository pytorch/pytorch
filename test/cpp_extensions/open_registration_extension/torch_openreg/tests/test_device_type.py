# Owner(s): ["module: PrivateUse1"]
"""
Device-Agnostic Test Patterns for PrivateUse1 Backends.

This file serves as a **reference implementation** demonstrating how to write
tests for PrivateUse1 backends using PyTorch's device type testing framework.

These tests automatically run on PrivateUse1 backends when registered.
This file contains only:
1. Device transfer tests specific to validating H2D/D2H data integrity
2. Reference examples for PrivateUse1-specific test decorators and skip patterns

For more details on the testing framework, see:
torch/testing/_internal/common_device_type.py
"""

import torch
import torch_openreg  # noqa: F401  # Ensures OpenReg backend is registered
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    onlyNativeDeviceTypes,
    onlyPRIVATEUSE1,
    skipCPUIf,
    skipCUDAIf,
    skipPRIVATEUSE1If,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestDeviceTransfer(TestCase):
    """
    These tests validate that data is correctly transferred between CPU and
    the PrivateUse1 device.
    """

    def test_to_device(self, device):
        x_cpu = torch.randn(3, 4)
        x = x_cpu.to(device=device)
        self.assertEqual(x.device.type, torch.device(device).type)
        self.assertEqual(x.cpu(), x_cpu)

    def test_to_device_with_copy(self, device):
        x = torch.randn(3, 4, device=device)
        y = x.to(device=device)
        self.assertEqual(x.data_ptr(), y.data_ptr())

        # With copy=True, should be different tensor
        z = x.to(device=device, copy=True)
        self.assertNotEqual(x.data_ptr(), z.data_ptr())
        self.assertEqual(x, z)

    def test_clone(self, device):
        x = torch.randn(3, 4, device=device)
        y = x.clone()
        self.assertEqual(y.device.type, torch.device(device).type)
        self.assertEqual(y, x)
        self.assertNotEqual(y.data_ptr(), x.data_ptr())

    def test_copy_(self, device):
        src = torch.randn(3, 4, device=device)
        dst = torch.empty(3, 4, device=device)
        dst.copy_(src)
        self.assertEqual(dst, src)
        self.assertNotEqual(dst.data_ptr(), src.data_ptr())

    def test_copy_from_cpu(self, device):
        x_cpu = torch.randn(3, 4)
        x_device = torch.empty(3, 4, device=device)
        x_device.copy_(x_cpu)
        self.assertEqual(x_device.cpu(), x_cpu)

    def test_copy_to_cpu(self, device):
        x_device = torch.randn(3, 4, device=device)
        x_cpu = torch.empty(3, 4)
        x_cpu.copy_(x_device)
        self.assertEqual(x_cpu, x_device.cpu())

    def test_round_trip(self, device):
        """Test data survives CPU -> device -> CPU round trip."""
        x_original = torch.randn(3, 4)
        x_device = x_original.to(device=device)
        x_back = x_device.to(device="cpu")
        self.assertEqual(x_back, x_original)


class TestDeviceSpecificSkips(TestCase):
    """Reference examples for device-specific test decorators.

    Available decorators for PrivateUse1:
    - @onlyPRIVATEUSE1: Only run on PrivateUse1 backends
    - @skipPRIVATEUSE1If(condition, reason): Skip on PrivateUse1 if condition is True
    - @onlyNativeDeviceTypes: Run on CPU, CUDA, and registered PrivateUse1

    For a complete list, see torch/testing/_internal/common_device_type.py
    """

    @onlyCPU
    def test_cpu_only_feature(self, device):
        self.assertEqual(device, "cpu")

    @onlyCUDA
    def test_cuda_only_feature(self, device):
        self.assertTrue("cuda" in device)

    @skipCPUIf(True, "Example of skipping on CPU")
    def test_skip_on_cpu(self, device):
        self.assertNotEqual(device, "cpu")

    @skipCUDAIf(True, "Example of skipping on CUDA")
    def test_skip_on_cuda(self, device):
        self.assertFalse("cuda" in device)

    @onlyNativeDeviceTypes
    def test_native_device_only(self, device):
        """Example: Test that runs on native device types.

        @onlyNativeDeviceTypes includes:
        - CPU, CUDA, XPU, Meta, MPS, MTIA
        - Registered PrivateUse1 backend (e.g., OpenReg)

        """
        x = make_tensor((2, 2), device=device, dtype=torch.float32)
        self.assertIsNotNone(x)

    @onlyPRIVATEUSE1
    def test_privateuse1_only(self, device):
        x = make_tensor((2, 2), device=device, dtype=torch.float32)
        self.assertIsNotNone(x)
        # Verify we're on the PrivateUse1 backend
        privateuse1_name = torch._C._get_privateuse1_backend_name()
        self.assertEqual(torch.device(device).type, privateuse1_name)

    @skipPRIVATEUSE1If(True, "Example: Skip on PrivateUse1 if condition is met")
    def test_conditional_skip_privateuse1(self, device):
        x = make_tensor((2, 2), device=device, dtype=torch.float32)
        self.assertIsNotNone(x)

    def test_runtime_skip_pattern(self, device):
        device_type = torch.device(device).type
        privateuse1_backend_name = torch._C._get_privateuse1_backend_name()

        if device_type == privateuse1_backend_name:
            # Check capability at runtime
            # if not hasattr(torch_openreg, 'some_advanced_feature'):
            #     self.skipTest("Feature not available on this backend")
            pass

        x = make_tensor((2, 2), device=device, dtype=torch.float32)
        self.assertIsNotNone(x)


class TestExceptForPattern(TestCase):
    """ 
    Runs on all devices EXCEPT those listed.
    """

    def test_tensor_operations(self, device):
        # Verify we're not on CPU or CUDA (demonstrating except_for)
        device_type = torch.device(device).type
        self.assertNotIn(device_type, ["cpu", "cuda"])

        a_cpu = torch.randn(3, 4)
        b_cpu = torch.randn(3, 4)
        a = a_cpu.to(device=device)
        b = b_cpu.to(device=device)

        self.assertEqual((a + b).cpu(), a_cpu + b_cpu)
        self.assertEqual((a * b).cpu(), a_cpu * b_cpu)
        self.assertEqual((a - b).cpu(), a_cpu - b_cpu)
        self.assertTrue(torch.allclose((a / b).cpu(), a_cpu / b_cpu))


class TestGlobalsPattern(TestCase):
    """
    Demonstrates globals() pattern + Device placement and metadata.
    """

    def test_basic_tensor_creation(self, device):
        shape = (3, 4)
        dtype = torch.float32

        x = torch.empty(shape, device=device, dtype=dtype)
        self.assertEqual(x.device.type, torch.device(device).type)
        self.assertEqual(x.shape, torch.Size(shape))
        self.assertEqual(x.dtype, dtype)

        x = torch.zeros(shape, device=device, dtype=dtype)
        self.assertTrue(torch.all(x == 0))

        x = torch.ones(shape, device=device, dtype=dtype)
        self.assertTrue(torch.all(x == 1))


class TestDtypeSupport(TestCase):
    """ 
    Instantiated with only_for=("openreg",) to focus on PrivateUse1 validation.
    """

    def test_dtype_propagation(self, device):
        x = torch.randn(3, 4, device=device, dtype=torch.float32)
        self.assertEqual(x.dtype, torch.float32)
        self.assertEqual((x + x).dtype, torch.float32)

        y = torch.randn(3, 4, device=device, dtype=torch.float64)
        self.assertEqual(y.dtype, torch.float64)
        self.assertEqual((y * 2).dtype, torch.float64)

        z = x.to(dtype=torch.float64)
        self.assertEqual(z.dtype, torch.float64)
        self.assertEqual(z.device.type, torch.device(device).type)

        i = torch.randint(0, 100, (3, 4), device=device, dtype=torch.int64)
        self.assertEqual(i.dtype, torch.int64)

        b = torch.zeros(3, 4, device=device, dtype=torch.bool)
        self.assertEqual(b.dtype, torch.bool)
        self.assertTrue(torch.all(~b))  # All False


# Device transfer tests specific to PrivateUse1 backend
instantiate_device_type_tests(TestDeviceTransfer, globals(), only_for=("openreg",))

# Validate dtype handling on the backend
instantiate_device_type_tests(TestDtypeSupport, globals(), only_for=("openreg",))

# Demonstrate skip decorators across all devices
instantiate_device_type_tests(TestDeviceSpecificSkips, globals())

# Tests should only run on non-native devices
instantiate_device_type_tests(TestExceptForPattern, globals(), except_for=("cpu", "cuda"))

# globals() enables test discovery
instantiate_device_type_tests(TestGlobalsPattern, globals())


if __name__ == "__main__":
    run_tests()
