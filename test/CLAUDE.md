# Test/ Directory - PyTorch Testing Framework

Comprehensive test suite for PyTorch covering all components from Python API to C++ kernels.

## 🏗️ Directory Organization

### Core Test Framework
- **`run_test.py`** - Main test runner script with sharding, filtering, and CI integration
- **`conftest.py`** - Pytest configuration and shared fixtures
- **`HowToWriteTestsUsingFileCheck.md`** - Guide for testing with FileCheck (primarily for Inductor tests)

### Test Categories

#### Core Functionality Tests
- **`test_torch.py`** - Core tensor operations and functions
- **`test_autograd.py`** - Automatic differentiation system
- **`test_nn.py`** - Neural network modules and functions
- **`test_ops.py`** - Operator correctness and edge cases

#### Device-Specific Tests
- **`test_cuda.py`** - CUDA functionality and GPU operations
- **`test_mps.py`** - Apple Metal Performance Shaders backend
- **`test_xpu.py`** - Intel XPU backend testing
- **`test_vulkan.py`** - Vulkan compute backend

#### Advanced Features
- **`test_fx.py`** - Graph capture and transformation
- **`test_export.py`** - Model export functionality
- **`test_quantization.py`** - Quantization algorithms
- **`test_distributed.py`** - Multi-process training
- **`test_profiler.py`** - Performance profiling tools
- **`test_jit.py`** - TorchScript compilation and execution (note: TorchScript is deprecated)

#### Backend-Specific Directories
- **`autograd/`** - Autograd-specific test cases
- **`backends/`** - Backend implementations testing
- **`ao/`** - AO (Accelerated Operations) quantization tests

### Supporting Infrastructure
- **`benchmark_utils/`** - Performance benchmarking utilities
- **`bottleneck_test/`** - Performance bottleneck identification
- **`compiled_autograd_skips/`** - Known compilation issues tracking

## 🧪 Running Tests

### Basic Test Execution
```bash
# Run all tests (rarely used - very slow)
python test/run_test.py

# Run specific test file
python test/run_test.py -i test_torch
python test/run_test.py -i test_nn
python test/run_test.py -i test_autograd

# Run tests with specific patterns
python test/run_test.py -k "test_add"
python test/run_test.py test_torch.py::TestTorch::test_add_cuda
python test/run_test.py -k "TestNN"
```

### Device-Specific Testing
```bash
# CUDA tests (requires CUDA)
python test/run_test.py -i test_cuda

# CPU-only tests
python test/run_test.py -i test_torch --cpu-only

# MPS tests (requires macOS with Metal)
python test/run_test.py -i test_mps
```

### Performance and Debugging
```bash
# Run with profiling
python test/run_test.py -i test_torch --profile

# Verbose output
python test/run_test.py -i test_torch -v

# Run single test method
python -m pytest test/test_torch.py::TestTorch::test_add_cpu -v
```

## 🔧 Test Development Patterns

### Identifying Relevant Tests
Part of development work is figuring out which tests are relevant for your changes:

- **Operator changes**: Look at `test_ops.py`, `test_torch.py`
- **Autograd changes**: Check `test_autograd.py`, `autograd/` directory
- **NN module changes**: Run `test_nn.py`, specific module tests
- **Backend changes**: Test device-specific files (`test_cuda.py`, etc.)
- **Build system changes**: Test core functionality and CI integration

### Writing New Tests

Test organization principles:
- Add new test class only for major new features
- Add new test methods for unit tests within existing classes
- For new operators, add OpInfo entries rather than individual test methods
```python
# Standard test pattern
class TestMyFeature(TestCase):
    def test_basic_functionality(self):
        # Test basic functionality
        pass
    
    @skipIfNoLapack
    @deviceCountAtLeast(2)
    def test_advanced_case(self):
        # Test with decorators for requirements
        pass
```

### Common Test Utilities
- **`torch.testing.assert_close()`** - Tensor comparison with tolerances
- **`@parametrize`** - Test multiple configurations
- **`@skipIf`** - Conditional test skipping
- **`gradcheck`** - Automatic gradient verification

### Test Instantiation Patterns
- `instantiate_device_type_tests()` - Generates device-specific test variants (CPU, CUDA, etc.)
- `instantiate_parametrized_tests()` - Creates parameterized test combinations
- Test methods with extra args (device, dtype) are automatically called from these instantiated new tests

## 🐛 Common Testing Issues

### Test Failures
- **Numerical precision**: Changing tolerance should be last resort. Refactor the test to be numerically stable.
- **Device availability**: Check CUDA/MPS availability before device-specific tests
- **Random seed**: Set seeds for reproducible tests
- **Resource cleanup**: Ensure proper cleanup of GPU memory, files, etc.

### Test Environment
- **Dependencies**: Some tests require optional dependencies (CUDA, MKL, etc.)
- **Hardware**: Device-specific tests need appropriate hardware

### CI Integration
- **Sharding**: Tests are automatically sharded in CI
- **Skip files**: Some tests may be temporarily skipped in CI
- **Timeouts**: Long-running tests may hit CI timeouts

## 📝 Notes for Claude

- Test discovery is based on filename patterns (`test_*.py`)
- Tests are designed to be run independently and in parallel
- Device-specific tests automatically skip if hardware unavailable
- Heavy use of parameterization for testing multiple configurations
- CI runs tests in sharded mode across multiple workers
- Test results are often XML-formatted for CI integration
- Memory and performance tests require special attention to cleanup

### Development Testing Strategy
- Run specific tests locally (single file maximum)
- Use GitHub CI for broad coverage across OS/hardware/platforms
- Iterate locally based on CI failures