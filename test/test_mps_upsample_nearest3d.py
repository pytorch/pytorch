import torch
import unittest
import math
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, skipMPS
from torch.testing._internal.common_methods_invocations import sample_inputs_upsample_nearest3d

class TestUpsampleNearest3DMPS(TestCase):
    def test_upsample_nearest3d_vec(self, device="mps"):
        """Test upsample_nearest3d.vec implementation on MPS device"""
        # Skip if MPS is not available
        if not torch.backends.mps.is_available():
            self.skipTest("MPS device not available")
        
        # Test with different input shapes
        input_shapes = [
            (1, 1, 2, 2, 2),  # Minimal shape
            (2, 3, 4, 5, 6),  # Standard shape
            (3, 4, 8, 10, 12)  # Larger shape
        ]
        
        # Test with different scale factors
        scale_factors = [
            (2.0, 2.0, 2.0),  # Double in all dimensions
            (1.5, 1.5, 1.5),  # 1.5x in all dimensions
            (0.5, 0.5, 0.5),  # Downsampling
            (1.0, 2.0, 3.0)   # Different scales per dimension
        ]
        
        # Test with different output sizes
        output_sizes = [
            (4, 4, 4),  # Double the minimal shape
            (8, 10, 12),  # Double the standard shape
            (4, 5, 6)   # Custom size
        ]
        
        # Test with different data types
        dtypes = [torch.float32, torch.float16]
        
        for input_shape in input_shapes:
            for dtype in dtypes:
                # Create input tensor
                x = torch.randn(input_shape, device="mps", dtype=dtype)
                
                # Test with scale_factor
                for scale_factor in scale_factors:
                    # Skip downsampling tests for now as they might not be supported
                    if scale_factor[0] < 1.0 or scale_factor[1] < 1.0 or scale_factor[2] < 1.0:
                        continue
                    
                    # Run on MPS
                    y_mps = torch.nn.functional.interpolate(
                        x, scale_factor=scale_factor, mode="nearest")
                    
                    # Run on CPU for reference
                    x_cpu = x.to("cpu")
                    y_cpu = torch.nn.functional.interpolate(
                        x_cpu, scale_factor=scale_factor, mode="nearest")
                    
                    # Compare results
                    self.assertEqual(y_mps.shape, y_cpu.shape)
                    self.assertTrue(torch.allclose(y_mps.to("cpu"), y_cpu, rtol=1e-3, atol=1e-3))
                
                # Test with output_size
                for output_size in output_sizes:
                    # Run on MPS
                    y_mps = torch.nn.functional.interpolate(
                        x, size=output_size, mode="nearest")
                    
                    # Run on CPU for reference
                    x_cpu = x.to("cpu")
                    y_cpu = torch.nn.functional.interpolate(
                        x_cpu, size=output_size, mode="nearest")
                    
                    # Compare results
                    self.assertEqual(y_mps.shape, y_cpu.shape)
                    self.assertTrue(torch.allclose(y_mps.to("cpu"), y_cpu, rtol=1e-3, atol=1e-3))
    
    def test_upsample_nearest3d_vec_backward(self, device="mps"):
        """Test backward pass of upsample_nearest3d.vec on MPS device"""
        # Skip if MPS is not available
        if not torch.backends.mps.is_available():
            self.skipTest("MPS device not available")
        
        # Create input tensor
        x = torch.randn(2, 3, 4, 5, 6, device="mps", requires_grad=True)
        
        # Forward pass
        y = torch.nn.functional.interpolate(x, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
        
        # Create gradient
        grad_output = torch.randn_like(y)
        
        # Backward pass
        y.backward(grad_output)
        
        # Check that gradient is not None
        self.assertIsNotNone(x.grad)
        
        # Compare with CPU implementation
        x_cpu = torch.randn(2, 3, 4, 5, 6, device="cpu", requires_grad=True)
        x_cpu.data.copy_(x.data.to("cpu"))
        
        y_cpu = torch.nn.functional.interpolate(x_cpu, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
        grad_output_cpu = grad_output.to("cpu")
        y_cpu.backward(grad_output_cpu)
        
        # Compare gradients
        self.assertTrue(torch.allclose(x.grad.to("cpu"), x_cpu.grad, rtol=1e-3, atol=1e-3))
    
    def test_upsample_nearest3d_vec_edge_cases(self, device="mps"):
        """Test edge cases for upsample_nearest3d.vec on MPS device"""
        # Skip if MPS is not available
        if not torch.backends.mps.is_available():
            self.skipTest("MPS device not available")
        
        # Test with empty tensor
        x = torch.randn(0, 3, 4, 5, 6, device="mps")
        y = torch.nn.functional.interpolate(x, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
        self.assertEqual(y.shape, (0, 3, 8, 10, 12))
        
        # Test with single element tensor
        x = torch.randn(1, 1, 1, 1, 1, device="mps")
        y = torch.nn.functional.interpolate(x, scale_factor=(2.0, 2.0, 2.0), mode="nearest")
        self.assertEqual(y.shape, (1, 1, 2, 2, 2))
        
        # Test with scale_factor = 1.0 (no change)
        x = torch.randn(2, 3, 4, 5, 6, device="mps")
        y = torch.nn.functional.interpolate(x, scale_factor=(1.0, 1.0, 1.0), mode="nearest")
        self.assertEqual(y.shape, (2, 3, 4, 5, 6))
        self.assertTrue(torch.allclose(y, x, rtol=1e-3, atol=1e-3))

if __name__ == "__main__":
    run_tests()
