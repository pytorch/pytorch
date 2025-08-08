"""
Unit tests for Intel XPU backend in PyTorch Inductor.

This module contains tests to verify the correctness and performance
of the Intel XPU backend optimizations for PyTorch Inductor.
"""

import os
import unittest
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from torch.inductor.xpu_backends import matmul, kernels, integration, utils
    from torch.inductor.xpu_backends.config import config as xpu_config
    HAS_XPU_BACKEND = True
except ImportError:
    HAS_XPU_BACKEND = False


@unittest.skipIf(not HAS_XPU_BACKEND, "XPU backend not available")
class TestXPUBackend(unittest.TestCase):
    """Test cases for Intel XPU backend implementation."""
    
    def setUp(self):
        """Set up test environment."""
        # Set a consistent random seed for reproducibility
        torch.manual_seed(12345)
        
        # Initialize XPU integration
        if HAS_XPU_BACKEND:
            self.integration = integration.XPUInductorIntegration()
            
    def test_xpu_available(self):
        """Test XPU availability check."""
        is_available = matmul.is_available()
        
        # This might be False on systems without Intel GPU
        # Just verify that the function runs without error
        self.assertIsInstance(is_available, bool)
        
    def test_matmul_optimal_tile_size(self):
        """Test optimal tile size calculation for matrix multiplication."""
        # Test different matrix sizes
        test_cases = [
            ((128, 128, 128), (16, 16, 16)),
            ((1024, 1024, 1024), (32, 32, 16)),
            ((4096, 4096, 4096), (128, 128, 32)),
        ]
        
        for (m, n, k), expected in test_cases:
            tile_size = matmul.XPUMatmulKernel.get_optimal_tile_size(m, n, k)
            self.assertEqual(tile_size, expected)
            
    @unittest.skipIf(not hasattr(torch, 'xpu') or not torch.xpu.is_available(), 
                    "XPU device not available")
    def test_matmul_kernel(self):
        """Test XPU optimized matrix multiplication kernel."""
        # Create test tensors
        a = torch.randn(128, 64, device="xpu")
        b = torch.randn(64, 32, device="xpu")
        
        # Reference result using standard PyTorch
        expected = torch.matmul(a, b)
        
        # Result using our XPU kernel
        result = matmul.XPUMatmulKernel.matmul(a, b)
        
        # Verify results match (within tolerance)
        self.assertTrue(torch.allclose(result, expected, rtol=1e-4, atol=1e-4))
        
    def test_xpu_config(self):
        """Test XPU configuration from environment variables."""
        # Temporarily set environment variables
        os.environ["PYTORCH_XPU_TILE_SIZE_M"] = "32"
        os.environ["PYTORCH_XPU_FAST_MATH"] = "true"
        
        # Create new config from environment
        config = xpu_config.__class__.from_env()
        
        # Verify config values were set from environment
        self.assertEqual(config.tile_size_m, 32)
        self.assertTrue(config.fast_math)
        
        # Clean up
        os.environ.pop("PYTORCH_XPU_TILE_SIZE_M", None)
        os.environ.pop("PYTORCH_XPU_FAST_MATH", None)
        
    def test_kernel_performance_estimation(self):
        """Test kernel performance estimation utilities."""
        # Test matrix multiplication performance estimation
        matmul_shapes = [[1024, 1024], [1024, 1024]]
        output_shape = [1024, 1024]
        
        estimates = utils.estimate_kernel_performance(
            "matmul", 
            matmul_shapes, 
            output_shape, 
            dtype=torch.float32
        )
        
        # Verify we get reasonable estimates (not checking exact values)
        self.assertIn("estimated_flops", estimates)
        self.assertGreater(estimates["estimated_flops"], 0)
        self.assertIn("estimated_memory_bytes", estimates)
        self.assertGreater(estimates["estimated_memory_bytes"], 0)
        
    def test_integration_singleton(self):
        """Test that integration class is a singleton."""
        integration1 = integration.XPUInductorIntegration()
        integration2 = integration.XPUInductorIntegration()
        
        # Verify both variables refer to the same instance
        self.assertIs(integration1, integration2)
        
    def test_launch_config(self):
        """Test kernel launch configuration generation."""
        # Test for matrix multiplication
        matmul_config = utils.get_optimal_launch_config(
            "matmul", 
            [[1024, 1024], [1024, 1024]]
        )
        
        # Verify we get a valid configuration
        self.assertIn("block_size_x", matmul_config)
        self.assertIn("grid_size_x", matmul_config)
        self.assertGreater(matmul_config["shared_memory_bytes"], 0)


@unittest.skipIf(not HAS_XPU_BACKEND, "XPU backend not available")
class TestXPUKernels(unittest.TestCase):
    """Test cases for Intel XPU kernel implementations."""
    
    def setUp(self):
        """Set up test environment."""
        torch.manual_seed(12345)
        
    @unittest.skipIf(not hasattr(torch, 'xpu') or not torch.xpu.is_available(), 
                    "XPU device not available")
    def test_conv2d_kernel(self):
        """Test XPU optimized conv2d kernel."""
        # Create test tensors
        batch_size = 8
        in_channels = 16
        out_channels = 32
        input_size = 24
        kernel_size = 3
        
        input = torch.randn(batch_size, in_channels, input_size, input_size, device="xpu")
        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device="xpu")
        bias = torch.randn(out_channels, device="xpu")
        
        # Reference result using standard PyTorch
        expected = torch.nn.functional.conv2d(
            input, weight, bias, stride=1, padding=1
        )
        
        # Result using our XPU kernel
        result = kernels.XPUConvolutionKernel.conv2d(
            input, weight, bias, stride=1, padding=1
        )
        
        # Verify results match (within tolerance)
        self.assertTrue(torch.allclose(result, expected, rtol=1e-4, atol=1e-4))
        
    @unittest.skipIf(not hasattr(torch, 'xpu') or not torch.xpu.is_available(), 
                    "XPU device not available")
    def test_gelu_activation(self):
        """Test XPU optimized GELU activation."""
        # Create test tensor
        input = torch.randn(1024, 1024, device="xpu")
        
        # Reference result using standard PyTorch
        expected = torch.nn.functional.gelu(input)
        
        # Result using our XPU kernel
        result = kernels.XPUActivationKernel.gelu(input)
        
        # Verify results match (within tolerance)
        self.assertTrue(torch.allclose(result, expected, rtol=1e-4, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
