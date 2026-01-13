# Example reproducer and minimal test to exercise OpenReg test instantiation patterns

import torch
from torch.testing._internal.common_utils import TestCase, run_tests, dtypes
from torch.testing._internal.common_device_type import instantiate_device_type_tests


class ExampleTestInstantiation(TestCase):
    """
    This is a minimal example test demonstrating OpenReg test patterns.
    
    When run, this test will be automatically expanded to:
    - test_device_aware_cpu_float32
    - test_device_aware_cpu_float64
    - test_device_aware_cuda_float32
    - test_device_aware_cuda_float64
    - test_device_aware_privateuse1_float32 (if OpenReg is available)
    - test_device_aware_privateuse1_float64 (if OpenReg is available)
    """
    
    @dtypes(torch.float32, torch.float64)
    def test_device_aware(self, device, dtype):
        """
        Demonstrates device and dtype parametrization.
        
        This test method accepts TWO extra parameters:
        - device: string like "cpu", "cuda:0", or "openreg:0"
        - dtype: torch dtype like torch.float32
        
        The instantiation framework creates one test per (device, dtype) pair.
        """
        # Create test data using the device and dtype parameters
        x = torch.randn(3, 3, dtype=dtype, device=device)
        y = torch.randn(3, 3, dtype=dtype, device=device)
        
        # Verify the device is correct
        device_type = device.split(":")[0]  # "cpu", "cuda", or "openreg"
        self.assertEqual(x.device.type, device_type)
        self.assertEqual(y.device.type, device_type)
        
        # Verify the dtype is correct
        self.assertEqual(x.dtype, dtype)
        self.assertEqual(y.dtype, dtype)
        
        # Perform a simple operation
        result = x + y
        
        # Verify result is on the same device and has the same dtype
        self.assertEqual(result.device.type, device_type)
        self.assertEqual(result.dtype, dtype)
        
        # Compare against CPU reference (move to CPU for comparison)
        expected = x.cpu().float() + y.cpu().float()
        # Use rtol/atol for floating-point comparison
        self.assertTrue(torch.allclose(result.cpu().float(), expected, rtol=1e-5, atol=1e-6))
    
    def test_simple_no_parametrization(self, device):
        """
        Test without dtype parametrization (device only).
        
        This will run once per device (CPU, CUDA, OpenReg, etc.)
        """
        x = torch.randn(2, 2, device=device)
        self.assertEqual(x.shape, torch.Size([2, 2]))
    
    def test_device_context(self, device):
        """Test device context managers."""
        initial_device = torch.device(device)
        
        # Create a tensor on the device
        x = torch.randn(2, 2, device=initial_device)
        self.assertEqual(x.device, initial_device)


# This is the critical call: instantiate device-specific test classes
# It replaces ExampleTestInstantiation with ExampleTestInstantiationCPU,
# ExampleTestInstantiationCUDA, ExampleTestInstantiationPrivateUse1, etc.
instantiate_device_type_tests(ExampleTestInstantiation, globals())


# For direct running:
if __name__ == "__main__":
    run_tests()
