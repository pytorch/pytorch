"""
Standalone test for XPU backend modules.
This script tests the modules directly without relying on the PyTorch import structure.
"""

import os
import sys
import importlib.util

def import_module_from_path(module_name, file_path):
    """Import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        print(f"Could not find module {module_name} at {file_path}")
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import modules directly
try:
    print("Importing modules directly from file paths...")
    
    # Patch torch module if needed
    if 'torch' not in sys.modules:
        import types
        torch = types.ModuleType('torch')
        sys.modules['torch'] = torch
        torch.is_tensor = lambda x: False  # Minimal torch functionality
    
    # Import our modules
    matmul = import_module_from_path('matmul', os.path.join(current_dir, 'matmul.py'))
    utils = import_module_from_path('utils', os.path.join(current_dir, 'utils.py'))
    integration = import_module_from_path('integration', os.path.join(current_dir, 'integration.py'))
    kernels = import_module_from_path('kernels', os.path.join(current_dir, 'kernels.py'))
    config_module = import_module_from_path('config', os.path.join(current_dir, 'config.py'))
    
    print("✓ Successfully imported all modules directly")
    
    # Test basic functions that don't require PyTorch
    print("\nTesting module structure and basic functionality:")
    
    # Check if matmul module has expected functions
    print("- Checking matmul module structure...")
    assert hasattr(matmul, 'XPUMatmulKernel'), "matmul module missing XPUMatmulKernel class"
    assert hasattr(matmul.XPUMatmulKernel, 'get_optimal_tile_size'), "XPUMatmulKernel missing get_optimal_tile_size method"
    print("  ✓ matmul module structure OK")
    
    # Test get_optimal_tile_size function
    print("- Testing XPUMatmulKernel.get_optimal_tile_size...")
    tile_size = matmul.XPUMatmulKernel.get_optimal_tile_size(1024, 1024, 1024)
    assert tile_size == (32, 32, 16), f"Expected (32, 32, 16) but got {tile_size}"
    print(f"  ✓ get_optimal_tile_size returned {tile_size} as expected")
    
    # Check if utils module has expected functions
    print("- Checking utils module structure...")
    assert hasattr(utils, 'get_optimal_launch_config'), "utils module missing get_optimal_launch_config function"
    print("  ✓ utils module structure OK")
    
    # Check if integration module has expected classes
    print("- Checking integration module structure...")
    assert hasattr(integration, 'XPUInductorIntegration'), "integration module missing XPUInductorIntegration class"
    print("  ✓ integration module structure OK")
    
    # Check if config module has expected classes
    print("- Checking config module structure...")
    assert hasattr(config_module, 'XPUInductorConfig'), "config module missing XPUInductorConfig class"
    print("  ✓ config module structure OK")
    
    # Test singleton pattern in XPUInductorIntegration
    print("- Testing XPUInductorIntegration singleton pattern...")
    integration_instance1 = integration.XPUInductorIntegration()
    integration_instance2 = integration.XPUInductorIntegration()
    assert integration_instance1 is integration_instance2, "XPUInductorIntegration is not a singleton"
    print("  ✓ XPUInductorIntegration singleton pattern works correctly")
    
    print("\nAll tests passed successfully! The module structure looks good.")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
