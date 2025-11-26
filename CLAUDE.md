# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Building PyTorch

### Quick Development Setup
```bash
# Setup development environment with virtual environment
make setup-env              # CPU-only development
make setup-env-cuda         # With pre-built CUDA binaries
make setup-env-rocm         # With pre-built ROCm binaries

# Build PyTorch from source (after setup-env)
python setup.py develop     # Editable install for development
```

### Build Customization
Key environment variables for controlling the build:
- `DEBUG=1` - Debug build
- `REL_WITH_DEB_INFO=1` - Release with debug info
- `MAX_JOBS=4` - Limit parallel compilation jobs
- `USE_CUDA=0` - Disable CUDA support
- `BUILD_TEST=0` - Skip building C++ tests
- `CMAKE_FRESH=1` - Clean CMake cache before building
- `CMAKE_ONLY=1` - Only run CMake, don't build

### Clean Build
```bash
make clean                  # Remove all build folders
python setup.py clean       # Clean Python build artifacts
```

## Testing

### Test Framework
Use PyTorch's test utilities for consistency:

```python
from torch.testing._internal.common_utils import run_tests, TestCase

class TestFeature(TestCase):
    def test_something(self):
        # Use assertEqual for tensor comparisons
        self.assertEqual(tensor1, tensor2)

        # Other useful assertions
        self.assertRaisesRegex(RuntimeError, "expected error", fn, args)
        self.assertLeaksNoCudaTensors()  # Check for CUDA memory leaks

if __name__ == "__main__":
    run_tests()
```

### Running Tests

#### Run Single Test Method
```bash
# Direct execution (fastest for debugging)
python test/test_torch.py TestTorch.test_add

# With pytest (more features)
pytest test/test_torch.py::TestTorch::test_add -v
```

#### Run Test File or Pattern
```bash
# Run all tests in a file
python test/test_nn.py

# Run tests matching pattern with pytest
pytest test/test_nn.py -k "Loss" -v

# Run in parallel (4 workers)
RUN_PARALLEL=4 python test/test_torch.py
```

#### Test Discovery
```bash
# List all tests without running
pytest test/ --collect-only

# Run full test suite
python test/run_test.py
```

### Test Environment Variables
- `TEST_IN_SUBPROCESS=1` - Run each test in separate process
- `PYTORCH_TEST_WITH_ASAN=1` - Enable address sanitizer
- `PYTORCH_TEST_WITH_ROCM=1` - Enable ROCm testing

## Code Architecture

### Core Components

**c10/** - Core library (minimal dependencies, works on mobile)
- Device abstractions (CUDA, CPU, MPS, XPU)
- Memory management and reference counting
- Core utilities used everywhere

**aten/** - C++ Tensor Library (no autograd)
- `src/ATen/native/` - Operator implementations
  - `cpu/` - CPU implementations with vectorization
  - `cuda/` - CUDA kernels
  - `mps/` - Apple Metal GPU ops
  - `sparse/` - Sparse tensor operations
  - `quantized/` - Quantization operators

**torch/** - Python API and high-level functionality
- `csrc/` - Python bindings and C++ API
  - `autograd/` - Automatic differentiation engine
  - `jit/` - TorchScript compiler
  - `distributed/` - Distributed training
- `nn/` - Neural network modules
- `_inductor/` - TorchInductor compiler
- `_dynamo/` - TorchDynamo graph capture

### Key Files for Operators
- `aten/src/ATen/native/native_functions.yaml` - Operator definitions
- `aten/src/ATen/native/*.cpp` - CPU implementations
- `aten/src/ATen/native/cuda/*.cu` - CUDA implementations
- Generated code appears in `build/aten/src/ATen/`

### Adding New Operators
1. Define in `native_functions.yaml`
2. Implement in appropriate `native/` subdirectory
3. Run torchgen to generate bindings
4. Write tests in `test/test_torch.py` or appropriate test file

## Linting and Code Quality

### Setup and Run Linters
```bash
# Initial setup
make setup-lint

# Run all linters
make lint

# Run on changed files only (fast)
make quicklint

# Auto-fix issues where possible
make quickfix
```

### Code Style
- Python: 120 char line limit (flake8), black-compatible formatting
- C++: Clang-format configured, 80 char soft limit
- Use type hints where possible (checked by mypy)
- Follow existing patterns in nearby code

### Linter-Specific Commands
```bash
# Run specific linter
lintrunner flake8 --all-files
lintrunner clang-format --all-files

# Type checking
python -m mypy torch/nn/
```

## Development Tips

### Iterating on C++ Code
After initial build, for C++ changes only:
```bash
# Rebuild C++ extensions without full setup.py
python setup.py build_ext --inplace

# Or use CMAKE_ONLY for faster iteration
CMAKE_ONLY=1 python setup.py develop
cmake --build build
```

### CUDA Development
```bash
# Set specific CUDA architectures to speed up build
export TORCH_CUDA_ARCH_LIST="8.0;8.6"  # Ampere GPUs

# Debug CUDA kernels
export CUDA_LAUNCH_BLOCKING=1
```

### Useful Make Targets
```bash
make setup-env       # Setup development environment
make clean          # Clean all build artifacts
make setup-lint     # Setup linting tools
make lint           # Run all linters
make quicklint      # Lint changed files only
```

### Common Development Patterns

#### Working with Tensors
- Always use `torch.testing.assert_close()` for floating point comparisons
- Be aware of device placement (CPU vs CUDA)
- Check for memory leaks with `self.assertLeaksNoCudaTensors()`

#### Debugging Test Failures
```bash
# Run single test with verbose output
python test/test_torch.py TestTorch.test_add -v

# Use pdb for debugging
python -m pdb test/test_torch.py TestTorch.test_add

# Check for flaky tests
PYTORCH_TEST_RERUN_DISABLED_TESTS=1 python test/test_torch.py
```

#### Performance Testing
- Use `torch.utils.benchmark` for microbenchmarks
- Profile with `torch.profiler` for performance analysis
- Check for CUDA synchronization issues with `CUDA_LAUNCH_BLOCKING=1`