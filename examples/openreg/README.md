# OpenReg Examples and Reproducers

This directory contains working examples and helper scripts to understand and test OpenReg testing patterns.

## Contents

### Example Tests

- **`example_test_instantiation.py`** — Minimal working example demonstrating device and dtype parametrization. Run this first to understand how tests expand.

### Templates

- **`opinfo_template.py`** — Template for defining new operator metadata (`OpInfo`). Use when adding tests for new operations.

### Build and Test Scripts

- **`build_and_test.ps1`** (Windows) — PowerShell helper to build OpenReg extension and run tests
- **`build_and_test.sh`** (Linux/macOS) — Bash helper for Unix-like systems

## Quick Start

### Windows

```powershell
# Run the example test
.\build_and_test.ps1 example_test_instantiation.py

# Run a specific test file
.\build_and_test.ps1 test_device.py

# Run a single test with verbose output
.\build_and_test.ps1 test_device.py::TestDevice::test_device_count -Verbose

# Force rebuild of the extension
.\build_and_test.ps1 -Rebuild
```

### Linux / macOS

```bash
# Run the example test
./build_and_test.sh example_test_instantiation.py

# Run a specific test file
./build_and_test.sh test_device.py

# Run a single test with verbose output
VERBOSE=true ./build_and_test.sh test_device.py::TestDevice::test_device_count

# Force rebuild of the extension
REBUILD=true ./build_and_test.sh
```

## Understanding the Output

When you run a test, you'll see something like:

```
test_device_aware_cpu_float32 PASSED
test_device_aware_cpu_float64 PASSED
test_device_aware_cuda_float32 PASSED
test_device_aware_cuda_float64 PASSED
test_device_aware_privateuse1_float32 PASSED
test_device_aware_privateuse1_float64 PASSED
```

Each line is a **separate instantiated test**. The test framework automatically generated 6 test cases from 1 template, covering the cross-product of:
- **Devices:** CPU, CUDA, PrivateUse1 (OpenReg)
- **Dtypes:** float32, float64

See [../docs/openreg/test_instantiation.md](../docs/openreg/test_instantiation.md) for details.

## Debugging Failed Tests

If a test fails, the output will show which configuration failed:

```
FAILED test_device_aware_privateuse1_float32 - AssertionError: device type mismatch
```

**Next steps:**

1. Run with more verbose output: `build_and_test.ps1 example_test_instantiation.py -Verbose`
2. Check [../docs/openreg/failure_interpretation.md](../docs/openreg/failure_interpretation.md) for troubleshooting
3. Look for the failure category (import error, assertion, dtype mismatch, etc.)
4. Apply the triage steps for that category

## Adding Your Own Tests

1. Copy `example_test_instantiation.py` to a new file: `my_test.py`
2. Modify the test class and methods
3. Run with: `build_and_test.ps1 my_test.py`

See [../docs/openreg/adding_tests.md](../docs/openreg/adding_tests.md) for step-by-step guidance.

## Common Commands

```powershell
# (Windows examples; adapt for Linux/macOS)

# Run a subset of tests matching a pattern
.\build_and_test.ps1 "test_device.py -k device_count"

# Show test collection without running
python -m pytest test/cpp_extensions/open_registration_extension/torch_openreg/tests/ --collect-only

# Run with custom pytest options
python -m pytest test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_device.py -v --tb=long

# Enable C++ stack traces for debugging
$env:TORCH_SHOW_CPP_STACKTRACES = "1"
.\build_and_test.ps1 test_device.py
```

## Troubleshooting

**Problem:** "torch_openreg module not found"

**Solution:** Run `build_and_test.ps1` without arguments first (it will build the extension).

**Problem:** "cmake not found"

**Solution:** Install CMake from https://cmake.org or `pip install cmake`

**Problem:** "ninja not found"

**Solution:** Install Ninja: `pip install ninja`

**Problem:** Build fails with "compiler not found" (Windows)

**Solution:** Install Visual Studio Build Tools from https://visualstudio.microsoft.com/downloads/

See [../docs/openreg/failure_interpretation.md](../docs/openreg/failure_interpretation.md) for more debugging tips.
