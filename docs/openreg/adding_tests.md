# Adding New Tests Guide

## Overview

This guide walks you through the process of adding a new test to OpenReg, from choosing the test base to running and validating your changes.

---

## Step-by-Step Workflow

### Step 1: Decide on the Test Type

**What are you testing?**

| Test Type | Use When | Location | Example |
|-----------|----------|----------|---------|
| **Device test** | Testing device operations (allocation, queries, context) | `test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_device.py` | `test_device_count()` |
| **Operator test** | Testing a specific operation (add, mul, matmul, etc.) | `test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_ops.py` | `test_add()`, `test_matmul()` |
| **Autograd test** | Testing gradient computation | `test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_autograd.py` | `test_backward()` |
| **Utility test** | Testing helper functions or APIs | `test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_utils.py` | `test_helper_function()` |
| **Feature test** | Testing a specific feature (e.g., streams, memory) | `test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_streams.py` | `test_stream_synchronize()` |

**Decision:** Pick the location that best fits your test.

---

### Step 2: Choose Your Test Base

**Option A: Basic TestCase (No Device Parametrization)**

Use `TestCase` for single-device tests (usually CPU or a specific fixed device):

```python
from torch.testing._internal.common_utils import TestCase

class TestMyFeature(TestCase):
    def test_something(self):
        # No device or dtype parameters
        x = torch.randn(2, 2)
        y = x + 1
        self.assertEqual(y.shape, torch.Size([2, 2]))
```

**Option B: DeviceTypeTestBase (Device Parametrization)**

Use `instantiate_device_type_tests()` for tests that should run on all available devices:

```python
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_device_type import instantiate_device_type_tests

class TestMyOp(TestCase):
    def test_add(self, device):
        # This test will run on CPU, CUDA, OpenReg, etc.
        x = torch.randn(2, 2, device=device)
        y = torch.randn(2, 2, device=device)
        result = x + y
        self.assertEqual(result.device.type, device.split(":")[0])

# At module scope:
instantiate_device_type_tests(TestMyOp, globals())
```

**Option C: DeviceTypeTestBase + Dtype Parametrization**

Use `@dtypes` for tests that vary over both devices and data types:

```python
from torch.testing._internal.common_utils import TestCase, dtypes
from torch.testing._internal.common_device_type import instantiate_device_type_tests

class TestNumericOps(TestCase):
    @dtypes(torch.float32, torch.float64)
    def test_reduction(self, device, dtype):
        # This test runs on all device/dtype combinations
        # Example: CPU/float32, CPU/float64, CUDA/float32, CUDA/float64, ...
        x = torch.randn(10, dtype=dtype, device=device)
        result = x.sum()
        expected = x.cpu().float().sum()
        self.assertTrue(torch.allclose(result, expected, rtol=1e-4))

instantiate_device_type_tests(TestNumericOps, globals())
```

**Decision:** Choose **Option B** for most new tests. It ensures your test runs on all available backends.

---

### Step 3: Decide on Parametrization

**No parametrization:**
```python
def test_simple(self):
    # Runs once
    pass
```

**Device parametrization only:**
```python
def test_device_specific(self, device):
    # Runs once per device: CPU, CUDA, OpenReg, ...
    pass
```

**Device + dtype parametrization:**
```python
@dtypes(torch.float32, torch.float64, torch.int32)
def test_multi_dtype(self, device, dtype):
    # Runs once per (device, dtype) pair
    # Example: 3 devices × 3 dtypes = 9 test instances
    pass
```

**Decision:** Use device parametrization for most tests. Add dtype parametrization if your test needs to validate behavior across multiple dtypes.

---

### Step 4: Write the Test

Start with a simple template:

```python
class TestNewFeature(TestCase):
    def test_basic_operation(self, device):
        """Test basic operation on the specified device."""
        # Step 1: Create test data
        x = torch.randn(3, 3, device=device)
        y = torch.randn(3, 3, device=device)
        
        # Step 2: Perform the operation
        result = x + y
        
        # Step 3: Compute expected result (usually on CPU)
        expected = x.cpu() + y.cpu()
        
        # Step 4: Assert correctness
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-6))

instantiate_device_type_tests(TestNewFeature, globals())
```

**Common Assertion Methods:**

| Method | Use Case | Example |
|--------|----------|---------|
| `self.assertEqual(a, b)` | Exact equality (integers, booleans) | `self.assertEqual(x.shape, torch.Size([2, 3]))` |
| `self.assertAlmostEqual(a, b, places=7)` | Floating-point with tolerance | `self.assertAlmostEqual(result, 1.5, places=5)` |
| `self.assertTrue(torch.allclose(a, b, rtol=..., atol=...))` | Tensor comparison with relative/absolute tolerance | `self.assertTrue(torch.allclose(result, expected, rtol=1e-4))` |
| `self.assertRaisesRegex(Exception, pattern)` | Error handling | `self.assertRaisesRegex(RuntimeError, "dimension out of range")` |

---

### Step 5: Add Skips (if needed)

If your test does not apply to certain devices or dtypes, add a skip:

```python
from torch.testing._internal.common_utils import skipIfRunningOn, skipIf

class TestConditionalFeature(TestCase):
    @skipIfRunningOn("cpu")
    def test_gpu_only(self, device):
        # This test will not run on CPU
        pass
    
    @skipIfRunningOn("openreg")
    def test_skip_openreg(self, device):
        # This test will not run on OpenReg
        pass
    
    @skipIf(not torch.cuda.is_available(), "CUDA required")
    @dtypes(torch.float32)
    def test_requires_cuda(self, device, dtype):
        pass

instantiate_device_type_tests(TestConditionalFeature, globals())
```

**When to skip:** Only skip when the feature is intentionally not supported (see [skip_patterns.md](skip_patterns.md)).

---

### Step 6: Register at Module Scope

At the end of your test file, add the instantiation call:

```python
if __name__ == "__main__":
    # Instantiate device-specific test classes
    instantiate_device_type_tests(TestMyOp, globals())
    
    # Run all tests
    run_tests()
```

This **must** be at module scope (not inside a function or class).

---

### Step 7: Run Locally

**Build the extension (one-time):**

```powershell
cd test/cpp_extensions/open_registration_extension
python setup.py build_ext --inplace
```

**Run a single test:**

```powershell
# From repo root
python -m pytest test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_ops.py::TestMyOp::test_add_cpu -v

# Or with unittest
python -m unittest test.cpp_extensions.open_registration_extension.torch_openreg.tests.test_ops.TestMyOp.test_add_cpu
```

**Run all your tests:**

```powershell
python -m pytest test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_ops.py::TestMyOp -v
```

**Run with verbose output:**

```powershell
python -m pytest test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_ops.py -v --tb=short
```

---

### Step 8: Interpret Results

**All tests pass:**
```
test_add_cpu_float32 PASSED
test_add_cuda_float32 PASSED
test_add_privateuse1_float32 PASSED
```
✅ Ready to commit!

**Some tests fail:**
```
test_add_privateuse1_float32 FAILED
AssertionError: expected device type openreg, got cpu
```
See [failure_interpretation.md](failure_interpretation.md) to debug.

**Expected failures (skip works):**
```
test_gpu_only_cpu SKIPPED (reason: test runs on GPU only)
test_gpu_only_cuda PASSED
```
✅ Skips working as expected.

---

### Step 9: Commit and Create PR

**Before pushing:**

1. Run all OpenReg tests to ensure no regressions:
   ```powershell
   python -m pytest test/cpp_extensions/open_registration_extension/torch_openreg/tests/ -q
   ```

2. Format and lint your code:
   ```powershell
   python -m black test_ops.py  # If using Black formatter
   python -m flake8 test_ops.py  # If using Flake8
   ```

3. Run a final sanity check:
   ```powershell
   python -c "
   import sys
   sys.path.insert(0, 'test/cpp_extensions/open_registration_extension')
   from torch_openreg.tests import test_ops
   print('✅ Import successful')
   "
   ```

**Commit message:**

```
Add test_add to OpenReg ops tests

- Tests basic addition operation on all devices and dtypes
- Parametrized over CPU, CUDA, and OpenReg backends
- Validates device preservation and numerical correctness

Fixes #1234
Related to: OpenReg Testing Patterns (issue #169597)
```

**Create PR:**

Reference the parent issue and link to related documentation:
```
Adds a comprehensive test for the add operation.

See [test_instantiation.md](docs/openreg/test_instantiation.md) for background on parametrization.
See [failure_interpretation.md](docs/openreg/failure_interpretation.md) for debugging failing tests.

Closes #1234
```

---

## Minimal Test Example

Here's a complete minimal example ready to use:

```python
# File: test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_my_new_op.py

import torch
from torch.testing._internal.common_utils import TestCase, run_tests, dtypes
from torch.testing._internal.common_device_type import instantiate_device_type_tests


class TestMyNewOp(TestCase):
    """Tests for a new operation."""
    
    @dtypes(torch.float32, torch.float64)
    def test_my_op_basic(self, device, dtype):
        """Test basic operation with multiple dtypes."""
        x = torch.randn(2, 2, dtype=dtype, device=device)
        y = torch.randn(2, 2, dtype=dtype, device=device)
        
        # Your operation here
        result = x + y
        
        # Compute expected result
        expected = x.cpu() + y.cpu()
        
        # Assert correctness
        self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-6))
    
    def test_my_op_shape_preservation(self, device):
        """Test that shape is preserved."""
        input_shape = (3, 4, 5)
        x = torch.randn(input_shape, device=device)
        y = torch.randn(input_shape, device=device)
        
        result = x + y
        
        self.assertEqual(result.shape, torch.Size(input_shape))


if __name__ == "__main__":
    instantiate_device_type_tests(TestMyNewOp, globals())
    run_tests()
```

Save this file, run it locally, and you'll automatically get tests for CPU, CUDA, and OpenReg (if available).

---

## Common Pitfalls to Avoid

| ❌ Don't | ✅ Do |
|----------|-------|
| Hard-code `device="cpu"` | Use the `device` parameter |
| Skip without a good reason | Only skip when feature is intentionally unsupported |
| Use loose tolerances everywhere | Use tight tolerances, document when they need relaxing |
| Forget `instantiate_device_type_tests()` call | Always add it at module scope |
| Test only one dtype | Use `@dtypes()` to cover multiple dtypes |
| Ignore device type in assertions | Always check device using `self.device` or `device` parameter |

---

## Troubleshooting

**Problem:** Test does not run (test method not discovered)

**Solution:** Make sure your test class inherits from `TestCase` and method name starts with `test_`.

**Problem:** Import error for OpenReg extension

**Solution:** Build the extension first:
```powershell
cd test/cpp_extensions/open_registration_extension
python setup.py build_ext --inplace
```

**Problem:** Test passes on CPU but fails on OpenReg

**Solution:** See [failure_interpretation.md](failure_interpretation.md) for debugging guidance.

**Problem:** My parameter is not being passed to the test method

**Solution:** Check the order of parameters. Device must come before dtype:
```python
@dtypes(torch.float32)
def test_foo(self, device, dtype):  # Correct order
    pass
```

---

## Summary

1. **Pick a test location** based on what you're testing
2. **Choose `TestCase` + `instantiate_device_type_tests()`** for device-parametrized tests
3. **Add `@dtypes()` decorator** if you need multiple data types
4. **Write the test body** following the template
5. **Add skips** only for intentionally unsupported cases
6. **Run locally** with pytest or unittest
7. **Commit and create a PR** with clear description
8. **Reference documentation** in your commit message

See [test_instantiation.md](test_instantiation.md) for background on how parametrization works, and [failure_interpretation.md](failure_interpretation.md) for debugging.
