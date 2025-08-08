"""
Mock-based test for XPU backend modules.
This script tests the modules with mocked PyTorch dependencies.
"""

import os
import sys
import unittest
from unittest import mock

# Create mock PyTorch modules
mock_torch = mock.MagicMock()
mock_torch._inductor = mock.MagicMock()
mock_torch._inductor.config = mock.MagicMock()

# Apply mocks to sys.modules
sys.modules['torch'] = mock_torch
sys.modules['torch._inductor'] = mock_torch._inductor
sys.modules['torch._inductor.config'] = mock_torch._inductor.config

# Now import our modules (after mocks are in place)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(current_dir)))  # Add parent directory to path

class TestXPUBackend(unittest.TestCase):
    """Test cases for XPU backend modules"""
    
    def setUp(self):
        """Set up test environment"""
        # Import modules now that mocks are in place
        from torch.inductor.xpu_backends import matmul, utils, integration, config
        self.matmul = matmul
        self.utils = utils
        self.integration = integration
        self.config = config
    
    def test_matmul_tile_size(self):
        """Test optimal tile size calculation"""
        # Test small matrices
        small_tile = self.matmul.XPUMatmulKernel.get_optimal_tile_size(128, 128, 128)
        self.assertEqual(small_tile, (16, 16, 16))
        
        # Test medium matrices
        medium_tile = self.matmul.XPUMatmulKernel.get_optimal_tile_size(1024, 1024, 1024)
        self.assertEqual(medium_tile, (32, 32, 16))
        
        # Test large matrices
        large_tile = self.matmul.XPUMatmulKernel.get_optimal_tile_size(4096, 4096, 4096)
        self.assertEqual(large_tile, (128, 128, 32))
    
    def test_integration_singleton(self):
        """Test that XPUInductorIntegration is a singleton"""
        instance1 = self.integration.XPUInductorIntegration()
        instance2 = self.integration.XPUInductorIntegration()
        self.assertIs(instance1, instance2)
    
    def test_launch_config(self):
        """Test kernel launch configuration generation"""
        config = self.utils.get_optimal_launch_config("matmul", [[1024, 1024], [1024, 1024]])
        self.assertIn("block_size_x", config)
        self.assertIn("grid_size_x", config)
        self.assertGreater(config["shared_memory_bytes"], 0)

if __name__ == '__main__':
    unittest.main()
