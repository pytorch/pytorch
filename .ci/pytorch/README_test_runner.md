# PyTorch CI Python Test Runner

This is a Python-based replacement for the `test.sh` shell script, designed to provide better maintainability and extensibility for PyTorch CI testing.

## Overview

The original `test.sh` script (~1,784 lines) has grown complex with numerous if/elif branches and environment variable dependencies. This Python implementation provides:

- **Modular Structure**: Clear separation of concerns with dedicated modules
- **Type Safety**: Python type hints for better reliability
- **Extensibility**: Easy to add new test configurations
- **Maintainability**: Configuration-driven approach instead of nested conditionals
- **Testability**: Individual components can be unit tested

## Architecture

```
.ci/pytorch/
├── test_runner.py          # Main entry point
├── test_config/
│   ├── __init__.py
│   ├── base.py            # Base test configuration classes
│   ├── environment.py     # Environment detection and setup
│   └── test_registry.py   # Test function registry
├── test_suites/
│   ├── __init__.py
│   ├── python_tests.py    # Python test implementations
│   ├── inductor_tests.py  # Inductor-specific tests
│   └── ...                # Other specialized test suites
└── utils/
    ├── __init__.py
    ├── shell_utils.py     # Shell command execution helpers
    └── install_utils.py   # Package installation utilities
```

## Usage

### Basic Usage

```bash
# Run tests for current environment
python .ci/pytorch/test_runner.py

# Dry run to see what would be executed
python .ci/pytorch/test_runner.py --dry-run

# Verbose output
python .ci/pytorch/test_runner.py --verbose
```

### Environment Variables

The test runner uses the same environment variables as the original `test.sh`:

- `BUILD_ENVIRONMENT`: Specifies the build configuration (e.g., `pytorch-linux-xenial-py3.8-gcc7-cuda11`)
- `TEST_CONFIG`: Specifies the test configuration (e.g., `inductor`, `distributed`, `smoke`)
- `SHARD_NUMBER`: Current shard number for parallel testing
- `NUM_TEST_SHARDS`: Total number of test shards

## Test Suite Selection

The test runner automatically selects the appropriate test suite based on environment configuration:

### Supported Test Configurations

- **numpy_2**: NumPy 2.0 compatibility tests
- **backward**: Forward/backward compatibility tests
- **xla**: XLA integration tests
- **executorch**: ExecutorTorch tests
- **jit_legacy**: Legacy JIT tests
- **distributed**: Distributed training tests
- **inductor**: Inductor compiler tests
- **inductor_distributed**: Inductor distributed tests
- **inductor_cpp_wrapper**: Inductor C++ wrapper tests
- **torchbench**: TorchBench performance tests
- **smoke**: Smoke tests
- **docs_test**: Documentation tests
- And many more...

### Build Environment Support

- **CUDA builds**: `*cuda*`
- **ROCm builds**: `*rocm*`
- **ASAN builds**: `*asan*`
- **AArch64 builds**: `*aarch64*`
- **XPU builds**: `*xpu*`
- **Vulkan builds**: `*vulkan*`
- **Bazel builds**: `*-bazel-*`
- **LibTorch builds**: `*libtorch*`
- **Mobile builds**: `*-mobile-lightweight-dispatch*`

## Migration Strategy

This implementation uses a **gradual migration approach**:

1. **Phase 1**: Python wrapper calls existing shell functions
2. **Phase 2**: Incrementally convert shell functions to native Python
3. **Phase 3**: Remove shell dependencies and retire `test.sh`

Currently in Phase 1, the Python test runner provides the structure and logic while delegating actual test execution to existing shell functions. This ensures compatibility while enabling incremental migration.

## Adding New Test Suites

To add a new test suite:

1. Create a new class inheriting from `TestSuite` or `ConditionalTestSuite`
2. Implement the required methods (`matches`, `get_test_names`, `run`)
3. Register the suite in `test_registry.py`

Example:

```python
class MyNewTestSuite(ConditionalTestSuite):
    def __init__(self):
        super().__init__(
            name="my_new_test",
            description="My new test suite",
            test_config_patterns=["my_pattern"]
        )
    
    def run(self, env_config: EnvironmentConfig) -> bool:
        # Implement test logic
        return True
```

## Benefits Over Shell Script

1. **Reduced Complexity**: Eliminates deeply nested if/elif chains
2. **Better Error Handling**: Structured exception handling and logging
3. **Type Safety**: Python type hints catch errors at development time
4. **Modularity**: Easy to test and maintain individual components
5. **Extensibility**: Simple to add new test configurations
6. **IDE Support**: Better development experience with autocomplete and refactoring
7. **Documentation**: Self-documenting code with docstrings and type hints

## Compatibility

The Python test runner is designed to be a drop-in replacement for `test.sh`. It:

- Uses the same environment variables
- Produces the same test outputs
- Maintains the same exit codes
- Supports all existing test configurations

## Testing the Migration

To test the new Python test runner:

```bash
# Set up test environment
export BUILD_ENVIRONMENT="pytorch-linux-xenial-py3.8-gcc7"
export TEST_CONFIG="smoke"
export SHARD_NUMBER="1"
export NUM_TEST_SHARDS="1"

# Run with dry-run first
python .ci/pytorch/test_runner.py --dry-run --verbose

# Run actual tests
python .ci/pytorch/test_runner.py --verbose
```

## Future Enhancements

- **Native Python Test Implementations**: Convert shell functions to Python
- **Parallel Test Execution**: Run independent tests in parallel
- **Better Reporting**: Structured test result reporting
- **Configuration Validation**: Validate environment configurations
- **Test Discovery**: Automatic discovery of test functions
- **Caching**: Cache test results and dependencies
