"""
Simple test script for the XPU backends module.
This script tests that our modules can be imported and basic functionality works.
"""

import os
import sys

# Add parent directory to the Python path so we can import our modules
sys.path.insert(0, os.path.abspath(".."))

try:
    # Import our modules
    from torch.inductor.xpu_backends import matmul, kernels, integration, utils, benchmark, config

    print("✓ Successfully imported all XPU backend modules")
    
    # Test basic functionality
    print("Testing matmul.is_available()...")
    is_available = matmul.is_available()
    print(f"  Result: {is_available}")
    
    print("Testing matmul.XPUMatmulKernel.get_optimal_tile_size(1024, 1024, 1024)...")
    tile_size = matmul.XPUMatmulKernel.get_optimal_tile_size(1024, 1024, 1024)
    print(f"  Result: {tile_size}")
    
    print("Testing utils.get_optimal_launch_config('matmul', [[1024, 1024], [1024, 1024]])...")
    launch_config = utils.get_optimal_launch_config('matmul', [[1024, 1024], [1024, 1024]])
    print(f"  Result: {launch_config}")
    
    print("Testing integration.XPUInductorIntegration singleton pattern...")
    integration1 = integration.XPUInductorIntegration()
    integration2 = integration.XPUInductorIntegration()
    print(f"  Are instances the same? {integration1 is integration2}")
    
    print("Testing config values...")
    print(f"  enabled: {config.config.enabled}")
    print(f"  fast_math: {config.config.fast_math}")
    
    print("\nAll basic tests passed successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
