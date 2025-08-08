"""
Simple direct test for XPU backend modules.
This script tests the modules by importing them directly.
"""

import os
import sys
import importlib.util
from unittest import mock

def load_module(name, path):
    """Load a module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Mock torch and its submodules
torch_mock = mock.MagicMock()
torch_mock._inductor = mock.MagicMock()
torch_mock._inductor.config = mock.MagicMock()
torch_mock.is_tensor = lambda x: isinstance(x, mock.MagicMock)
torch_mock.Tensor = mock.MagicMock

# Apply mocks
sys.modules['torch'] = torch_mock
sys.modules['torch._inductor'] = torch_mock._inductor
sys.modules['torch._inductor.config'] = torch_mock._inductor.config

try:
    # Load modules directly
    matmul = load_module('matmul', os.path.join(current_dir, 'matmul.py'))
    utils = load_module('utils', os.path.join(current_dir, 'utils.py'))
    
    # Run basic tests
    print("Testing matmul.XPUMatmulKernel.get_optimal_tile_size...")
    tile_size = matmul.XPUMatmulKernel.get_optimal_tile_size(1024, 1024, 1024)
    assert tile_size == (32, 32, 16), f"Expected (32, 32, 16) but got {tile_size}"
    print(f"✓ get_optimal_tile_size returned {tile_size} as expected")
    
    print("Testing utils.get_optimal_launch_config...")
    launch_config = utils.get_optimal_launch_config("matmul", [[1024, 1024], [1024, 1024]])
    assert "block_size_x" in launch_config, "Missing block_size_x in launch config"
    assert "grid_size_x" in launch_config, "Missing grid_size_x in launch config"
    assert launch_config["shared_memory_bytes"] > 0, "Expected positive shared_memory_bytes"
    print(f"✓ get_optimal_launch_config returned valid configuration")
    
    print("\nAll basic tests passed!")
    sys.exit(0)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
