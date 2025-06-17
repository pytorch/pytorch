"""
Tests for the kernel filters functionality.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import torch
from torch._inductor import config
from torch._inductor.kernel.kernel_filters import (
    gemm_config_registry,
    kernel_lut_filter,
    model_predicted_top_k,
    get_filter_context,
    set_scale_params,
)
from torch._inductor.template_heuristics import GemmConfig


class TestKernelFilters(unittest.TestCase):
    """Tests for the kernel filters functionality."""

    def setUp(self):
        """Set up test environment."""
        # Create sample configs for testing
        self.configs = [
            GemmConfig(block_m=128, block_n=128, block_k=32, num_stages=3, num_warps=4),
            GemmConfig(block_m=64, block_n=64, block_k=32, num_stages=2, num_warps=4),
            GemmConfig(block_m=32, block_n=32, block_k=16, num_stages=1, num_warps=2),
        ]
        
        # Set up filter context
        set_scale_params(m=1024, n=1024, k=1024)
    
    def test_kernel_lut_filter_no_lut(self):
        """Test kernel_lut_filter with no lookup table."""
        with patch('torch._inductor.kernel.kernel_filters.get_lookup_table', return_value=None):
            filtered_configs = kernel_lut_filter(self.configs)
            # Should return all configs when no lookup table is found
            self.assertEqual(len(filtered_configs), 3)
    
    def test_kernel_lut_filter_with_lut(self):
        """Test kernel_lut_filter with a lookup table."""
        # Create a mock lookup table
        mock_lut = MagicMock()
        mock_lut.filter_configs.return_value = [self.configs[0]]
        
        with patch('torch._inductor.kernel.kernel_filters.get_lookup_table', return_value=mock_lut):
            filtered_configs = kernel_lut_filter(self.configs)
            # Should return filtered configs from the lookup table
            self.assertEqual(len(filtered_configs), 1)
            self.assertEqual(filtered_configs[0].block_m, 128)
            self.assertEqual(filtered_configs[0].block_n, 128)
            self.assertEqual(filtered_configs[0].block_k, 32)
            
            # Verify the lookup table was called with the correct parameters
            mock_lut.filter_configs.assert_called_once_with(self.configs, 1024, 1024, 1024)
    
    def test_kernel_lut_filter_exception(self):
        """Test kernel_lut_filter with an exception."""
        # Create a mock lookup table that raises an exception
        mock_lut = MagicMock()
        mock_lut.filter_configs.side_effect = Exception("Test exception")
        
        with patch('torch._inductor.kernel.kernel_filters.get_lookup_table', return_value=mock_lut):
            filtered_configs = kernel_lut_filter(self.configs)
            # Should return all configs when an exception occurs
            self.assertEqual(len(filtered_configs), 3)
    
    def test_model_predicted_top_k_default(self):
        """Test model_predicted_top_k with default parameters."""
        # Mock the necessary components
        mock_wrappedmodel = MagicMock()
        mock_wrappedmodel.encode.return_value = torch.tensor([1.0, 2.0, 3.0])
        mock_wrappedmodel.inference.return_value = torch.tensor([0.1, 0.3, 0.2])
        
        with patch('torch._inductor.kernel.kernel_filters.wrappedmodel', mock_wrappedmodel), \
             patch('torch._inductor.config.matmul_gemm_autotune_benchmarking_space', 2):
            filtered_configs = model_predicted_top_k(self.configs)
            # Should return top 2 configs based on predictions
            self.assertEqual(len(filtered_configs), 2)
    
    def test_model_predicted_top_k_explicit(self):
        """Test model_predicted_top_k with explicit top_k."""
        # Mock the necessary components
        mock_wrappedmodel = MagicMock()
        mock_wrappedmodel.encode.return_value = torch.tensor([1.0, 2.0, 3.0])
        mock_wrappedmodel.inference.return_value = torch.tensor([0.1, 0.3, 0.2])
        
        with patch('torch._inductor.kernel.kernel_filters.wrappedmodel', mock_wrappedmodel):
            filtered_configs = model_predicted_top_k(self.configs, top_k=1)
            # Should return top 1 config based on predictions
            self.assertEqual(len(filtered_configs), 1)
    
    def test_model_predicted_top_k_exception(self):
        """Test model_predicted_top_k with an exception."""
        # Mock the necessary components to raise an exception
        mock_wrappedmodel = MagicMock()
        mock_wrappedmodel.encode.side_effect = Exception("Test exception")
        
        with patch('torch._inductor.kernel.kernel_filters.wrappedmodel', mock_wrappedmodel):
            filtered_configs = model_predicted_top_k(self.configs)
            # Should return all configs when an exception occurs
            self.assertEqual(len(filtered_configs), 3)
    
    def test_filter_registry(self):
        """Test the filter registry functionality."""
        # Clear the registry
        original_filters = gemm_config_registry.filters.copy()
        original_filter_names = gemm_config_registry.filter_names.copy()
        gemm_config_registry.clear()
        
        try:
            # Register a test filter
            @gemm_config_registry.register(name="test_filter")
            def test_filter(configs):
                return [configs[0]]
            
            # Check if the filter was registered
            self.assertEqual(len(gemm_config_registry.filters), 1)
            self.assertEqual(gemm_config_registry.filter_names, ["test_filter"])
            
            # Apply the filter
            filtered_configs = gemm_config_registry.apply_filters(self.configs)
            self.assertEqual(len(filtered_configs), 1)
            self.assertEqual(filtered_configs[0].block_m, 128)
        finally:
            # Restore the original filters
            gemm_config_registry.filters = original_filters
            gemm_config_registry.filter_names = original_filter_names


class TestKernelLUTFilterIntegration(unittest.TestCase):
    """Integration tests for the kernel LUT filter."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create sample configs for testing
        self.configs = [
            GemmConfig(block_m=128, block_n=128, block_k=32, num_stages=3, num_warps=4),
            GemmConfig(block_m=64, block_n=64, block_k=32, num_stages=2, num_warps=4),
            GemmConfig(block_m=32, block_n=32, block_k=16, num_stages=1, num_warps=2),
        ]
        
        # Set up filter context
        set_scale_params(m=1024, n=1024, k=1024)
        
        # Create a sample JSON lookup table file
        self.json_path = os.path.join(self.temp_dir.name, "test_lut.json")
        import json
        data = [
            {
                "problem_size": {"m": 1024, "n": 1024, "k": 1024},
                "config": {
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "BLOCK_K": 32,
                    "num_stages": 3,
                    "num_warps": 4,
                    "GROUP_M": 8
                }
            }
        ]
        with open(self.json_path, "w") as f:
            json.dump(data, f)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_kernel_lut_filter_integration(self):
        """Test kernel_lut_filter with a real lookup table file."""
        # Set the lookup table path in the config
        with patch.object(config, "kernel_lut_path", self.json_path):
            # Reset the global lookup table instance
            from torch._inductor.kernel.kernel_lut import _global_lut
            _global_lut = None
            
            # Apply the filter
            filtered_configs = kernel_lut_filter(self.configs)
            
            # Should return filtered configs from the lookup table
            self.assertEqual(len(filtered_configs), 1)
            self.assertEqual(filtered_configs[0].block_m, 128)
            self.assertEqual(filtered_configs[0].block_n, 128)
            self.assertEqual(filtered_configs[0].block_k, 32)


if __name__ == "__main__":
    unittest.main()
