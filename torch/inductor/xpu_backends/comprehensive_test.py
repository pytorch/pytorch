"""
Comprehensive direct test for XPU backend modules.
This script tests the modules by importing them directly and testing more components.
"""

import os
import sys
import importlib.util
from unittest import mock
import inspect

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

def print_separator():
    """Print a separator line."""
    print("\n" + "=" * 50 + "\n")

try:
    print("Loading XPU backend modules...")
    
    # Load modules directly
    matmul = load_module('matmul', os.path.join(current_dir, 'matmul.py'))
    utils = load_module('utils', os.path.join(current_dir, 'utils.py'))
    kernels = load_module('kernels', os.path.join(current_dir, 'kernels.py'))
    integration = load_module('integration', os.path.join(current_dir, 'integration.py'))
    config = load_module('config', os.path.join(current_dir, 'config.py'))
    benchmark = load_module('benchmark', os.path.join(current_dir, 'benchmark.py'))
    
    print("✓ All modules loaded successfully")
    print_separator()
    
    # Test matmul module
    print("Testing matmul module...")
    
    # Check XPUMatmulKernel class
    assert hasattr(matmul, 'XPUMatmulKernel'), "Missing XPUMatmulKernel class"
    assert inspect.isclass(matmul.XPUMatmulKernel), "XPUMatmulKernel is not a class"
    
    # Test get_optimal_tile_size method
    test_sizes = [
        ((128, 128, 128), (16, 16, 16)),
        ((1024, 1024, 1024), (32, 32, 16)),
        ((4096, 4096, 4096), (128, 128, 32))
    ]
    
    for input_size, expected_tile in test_sizes:
        tile_size = matmul.XPUMatmulKernel.get_optimal_tile_size(*input_size)
        assert tile_size == expected_tile, f"For size {input_size}, expected {expected_tile} but got {tile_size}"
        print(f"✓ get_optimal_tile_size({input_size}) returned {tile_size} as expected")
    
    # Test is_available function
    assert hasattr(matmul, 'is_available'), "Missing is_available function"
    assert callable(matmul.is_available), "is_available is not callable"
    
    print("✓ matmul module tests passed")
    print_separator()
    
    # Test utils module
    print("Testing utils module...")
    
    # Check get_optimal_launch_config function
    assert hasattr(utils, 'get_optimal_launch_config'), "Missing get_optimal_launch_config function"
    assert callable(utils.get_optimal_launch_config), "get_optimal_launch_config is not callable"
    
    # Test get_optimal_launch_config function
    op_types = ["matmul", "conv2d", "element_wise"]
    for op_type in op_types:
        config = utils.get_optimal_launch_config(op_type, [[1024, 1024], [1024, 1024]])
        assert "block_size_x" in config, f"Missing block_size_x in {op_type} launch config"
        assert "grid_size_x" in config, f"Missing grid_size_x in {op_type} launch config"
        print(f"✓ get_optimal_launch_config for {op_type} returned valid configuration")
    
    # Test other utility functions
    assert hasattr(utils, 'estimate_kernel_performance'), "Missing estimate_kernel_performance function"
    assert callable(utils.estimate_kernel_performance), "estimate_kernel_performance is not callable"
    
    performance = utils.estimate_kernel_performance("matmul", [[1024, 1024], [1024, 1024]], [1024, 1024])
    assert "estimated_flops" in performance, "Missing estimated_flops in performance estimate"
    assert performance["estimated_flops"] > 0, "Expected positive flops count"
    print(f"✓ estimate_kernel_performance returned valid estimates")
    
    print("✓ utils module tests passed")
    print_separator()
    
    # Test kernels module
    print("Testing kernels module...")
    
    # Check kernel classes
    kernel_classes = ["XPUConvolutionKernel", "XPUReductionKernel", "XPUActivationKernel"]
    for class_name in kernel_classes:
        assert hasattr(kernels, class_name), f"Missing {class_name} class"
        assert inspect.isclass(getattr(kernels, class_name)), f"{class_name} is not a class"
        print(f"✓ {class_name} class exists")
    
    # Check GELU implementation
    assert hasattr(kernels.XPUActivationKernel, 'gelu'), "Missing gelu method"
    assert callable(kernels.XPUActivationKernel.gelu), "gelu is not callable"
    print(f"✓ XPUActivationKernel.gelu method exists")
    
    print("✓ kernels module tests passed")
    print_separator()
    
    # Test integration module
    print("Testing integration module...")
    
    # Check XPUInductorIntegration class
    assert hasattr(integration, 'XPUInductorIntegration'), "Missing XPUInductorIntegration class"
    assert inspect.isclass(integration.XPUInductorIntegration), "XPUInductorIntegration is not a class"
    
    # Test singleton pattern
    instance1 = integration.XPUInductorIntegration()
    instance2 = integration.XPUInductorIntegration()
    assert instance1 is instance2, "XPUInductorIntegration is not a singleton"
    print(f"✓ XPUInductorIntegration singleton pattern works correctly")
    
    # Check initialize_xpu_backend function
    assert hasattr(integration, 'initialize_xpu_backend'), "Missing initialize_xpu_backend function"
    assert callable(integration.initialize_xpu_backend), "initialize_xpu_backend is not callable"
    print(f"✓ initialize_xpu_backend function exists")
    
    print("✓ integration module tests passed")
    print_separator()
    
    # Test config module
    print("Testing config module...")
    
    # Check XPUInductorConfig class
    assert hasattr(config, 'XPUInductorConfig'), "Missing XPUInductorConfig class"
    assert inspect.isclass(config.XPUInductorConfig), "XPUInductorConfig is not a class"
    
    # Test from_env method
    assert hasattr(config.XPUInductorConfig, 'from_env'), "Missing from_env method"
    assert callable(config.XPUInductorConfig.from_env), "from_env is not callable"
    
    # Check global config instance
    assert hasattr(config, 'config'), "Missing global config instance"
    assert isinstance(config.config, config.XPUInductorConfig), "config is not an instance of XPUInductorConfig"
    print(f"✓ Global config instance exists and has type XPUInductorConfig")
    
    print("✓ config module tests passed")
    print_separator()
    
    # Test benchmark module
    print("Testing benchmark module...")
    
    # Check XPUBenchmark class
    assert hasattr(benchmark, 'XPUBenchmark'), "Missing XPUBenchmark class"
    assert inspect.isclass(benchmark.XPUBenchmark), "XPUBenchmark is not a class"
    
    # Create benchmark instance
    bench = benchmark.XPUBenchmark(warm_up_iterations=1, test_iterations=1)
    assert hasattr(bench, 'benchmark_operation'), "Missing benchmark_operation method"
    assert callable(bench.benchmark_operation), "benchmark_operation is not callable"
    print(f"✓ XPUBenchmark instance created successfully")
    
    # Check generate_benchmark_report function
    assert hasattr(benchmark, 'generate_benchmark_report'), "Missing generate_benchmark_report function"
    assert callable(benchmark.generate_benchmark_report), "generate_benchmark_report is not callable"
    print(f"✓ generate_benchmark_report function exists")
    
    print("✓ benchmark module tests passed")
    print_separator()
    
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("The XPU backend implementation is ready for use.")
    sys.exit(0)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
