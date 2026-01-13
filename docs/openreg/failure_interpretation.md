# Failure Interpretation Guide

## Overview

When OpenReg tests fail, it's not always clear whether the failure is a real bug, a missing feature, or an unsupported capability. This guide helps you categorize failures, understand root causes, and decide the appropriate action.

---

## Running OpenReg Tests Locally

### Prerequisites

**Windows:**
```
Visual Studio Build Tools (MSVC)
CMake >= 3.10
Ninja build system (pip install ninja)
Python dev headers (python-dev)
```

**Setup:**
```powershell
# From the PyTorch repo root

# 1. Install build dependencies
python -m pip install -r requirements-build.txt

# 2. Build PyTorch (optional; may already be built)
python -m pip install -e .

# 3. Build the OpenReg extension
cd test/cpp_extensions/open_registration_extension
python setup.py build_ext --inplace
```

### Running a Single Test

```powershell
# From repo root, run a specific test file
python -m pytest test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_device.py::TestDevice::test_device_count -v

# Or use unittest directly
python -m unittest test.cpp_extensions.open_registration_extension.torch_openreg.tests.test_device.TestDevice.test_device_count
```

### Running a Subset of OpenReg Tests

```powershell
# All device tests
python -m pytest test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_device.py -v

# All ops tests
python -m pytest test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_ops.py -v

# Tests matching a pattern (e.g., containing "copy")
python -m pytest test/cpp_extensions/open_registration_extension/torch_openreg/tests/ -k "copy" -v
```

### Enabling Verbose Output and Stack Traces

```powershell
# Show C++ stack traces for native errors
$env:TORCH_SHOW_CPP_STACKTRACES = "1"
python -m pytest test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_device.py::TestDevice::test_invalid_device_index -v

# Show local variables in failures
python -m pytest --showlocals test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_device.py -v
```

---

## Common Failure Categories and Triage

### Category 1: Import and Build Errors

**Symptom:**
```
ImportError: cannot import name 'torch_openreg' from 'torch' (...)
ModuleNotFoundError: No module named 'torch_openreg._C'
```

**Root Causes:**
1. OpenReg extension not built (`torch_openreg._C` missing)
2. DLL/shared library not on PATH
3. OpenReg runtime DLL not found or incompatible

**Triage Steps:**

```powershell
# Step 1: Check if extension is built
ls test/cpp_extensions/open_registration_extension/torch_openreg/

# Should show: __init__.py, lib/ (containing .dll or .so)

# Step 2: Rebuild if missing
cd test/cpp_extensions/open_registration_extension
python setup.py build_ext --inplace

# Step 3: Check Python can import it
python -c "import torch_openreg; print(torch_openreg.__file__)"

# Step 4: If still failing, check for missing runtime DLL
# On Windows, use Dependency Walker on the .pyd file
# or check the loader output:
$env:PATH += ";test/cpp_extensions/open_registration_extension/torch_openreg/lib"
python -c "import torch_openreg._C"
```

**Action:** Fix the build or DLL path. See [operator_coverage.md#backend-maturity-stages](operator_coverage.md) for expected setup.

---

### Category 2: Assertion Mismatches (Wrong Expected Values)

**Symptom:**
```
AssertionError: 2 != 999

test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_device.py:15: in test_device_count
    self.assertEqual(count, 2)
E   AssertionError: 2 != 999
```

**Root Causes:**
1. Test expectation is incorrect (test bug)
2. Backend behavior differs from expectation (feature gap or config issue)
3. Test is checking the wrong thing (test logic bug)

**Triage Steps:**

```python
# Step 1: Print the actual value to understand
def test_device_count(self):
    count = torch.accelerator.device_count()
    print(f"Actual device count: {count}")  # Debug print
    self.assertEqual(count, 2)  # Expected value

# Step 2: Check if it matches your setup
# If you have N OpenReg devices configured, count should equal N

# Step 3: Check if the test was recently changed
# git log --oneline test_device.py | head -5
# git diff HEAD~1 test_device.py  # See recent changes
```

**Action:**
- If the actual value is correct: **update the test expectation.**
- If the actual value is wrong: **this is a real bug in the backend.** File an issue and fix the underlying code.
- If the test logic is wrong: **fix the test.**

---

### Category 3: Missing Kernel or Unsupported Operation

**Symptom:**
```
RuntimeError: add not implemented for device type openreg

test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_ops.py:42: in test_add
    result = x + y
E   RuntimeError: add not implemented for device type openreg
```

**Root Causes:**
1. The operation kernel is not registered for the device
2. The operation is intentionally not supported for this device
3. The device dispatch system is not correctly configured

**Triage Steps:**

```python
# Step 1: Check if operation is supposed to be supported
# See docs/openreg/operator_coverage.md for expected op support

# Step 2: Check if the kernel is registered
# In the C++ code, search for REGISTER_OPENREG_KERNEL(add)
# or equivalent dispatch registration

# Step 3: Verify the operation is in the correct dispatch namespace
# OpenReg uses PRIVATEUSE1 dispatch key; check if kernel is registered there

# Step 4: Enable verbose dispatch tracing
$env:TORCH_DISPATCH_VERBOSE = "1"
python -m pytest test/cpp_extensions/.../test_ops.py::test_add -v
```

**Action:**
- If the operation should be supported: **implement the kernel** and register it.
- If the operation is intentionally unsupported: **add a skip** (see [skip_patterns.md](skip_patterns.md)).
- If dispatch is misconfigured: **fix the dispatch registration.**

---

### Category 4: Numerical Mismatch (Wrong Answer)

**Symptom:**
```
AssertionError: Tensor values do not match
expected ≈ [1.5, 2.5, 3.5]
actual   ≈ [1.4, 2.6, 3.5]

test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_ops.py:85: in test_numeric_output
    self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-6))
E   AssertionError: False
```

**Root Causes:**
1. Numerical precision issue (rounding, accumulation order)
2. Wrong algorithm or formula in the kernel
3. Uninitialized memory or undefined behavior
4. Tolerance too strict for the operation

**Triage Steps:**

```python
# Step 1: Check if the mismatch is within expected tolerance
# Different implementations (CPU, CUDA, OpenReg) may have different rounding

result = x + y  # OpenReg
expected = x.cpu() + y.cpu()  # CPU reference
diff = (result - expected).abs().max()
print(f"Max difference: {diff}")

# Step 2: Compare against a known reference
# Run the same computation on CPU and compare
# If CPU also differs, the issue is in the test setup

# Step 3: Check the tolerance
# Some ops (e.g., reductions) accumulate error; need higher tolerance
# For float32: usually rtol=1e-4 to 1e-5 is reasonable
# For float64: usually rtol=1e-10 to 1e-8 is reasonable

# Step 4: Check if the kernel implementation is correct
# Review the C++ code for the operation
# Look for integer overflow, incorrect formula, or uninitialized variables
```

**Action:**
- If tolerance is too strict: **relax it** (but document why).
- If algorithm is wrong: **fix the kernel implementation.**
- If the difference is tiny and within float precision: **increase tolerance.**
- If the difference is large: **investigate the kernel code and fix the bug.**

---

### Category 5: Device Type Mismatch (Wrong Device in Output)

**Symptom:**
```
AssertionError: device type mismatch
expected device type: openreg
actual device type: cpu

test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_ops.py:52: in test_result_device
    self.assertEqual(result.device.type, "openreg")
E   AssertionError: Expected device type openreg, got cpu
```

**Root Causes:**
1. Kernel returns result on wrong device (implementation bug)
2. Temporary computation moved to CPU inadvertently
3. Device promotion rules are incorrect

**Triage Steps:**

```python
# Step 1: Trace which operation is on the wrong device
def test_result_device(self, device, dtype):
    x = torch.randn(2, 2, dtype=dtype, device=device)
    print(f"x device: {x.device}")  # Should be openreg
    
    y = torch.randn(2, 2, dtype=dtype, device=device)
    print(f"y device: {y.device}")  # Should be openreg
    
    result = x + y
    print(f"result device: {result.device}")  # Should be openreg
    
    self.assertEqual(result.device.type, "openreg")

# Step 2: Check if intermediate operations are moving to CPU
# Some operations (e.g., item(), numpy()) move to CPU
# Ensure your computation stays on device

# Step 3: Check the kernel dispatch
# Make sure the kernel is registered for the device
# and not falling back to CPU due to missing registration
```

**Action:**
- If an operation is moving to CPU: **fix the kernel to return the result on the original device.**
- If dispatch is falling back to CPU: **register the kernel for the device.**

---

### Category 6: Dtype Mismatch (Wrong Data Type)

**Symptom:**
```
RuntimeError: expected scalar type float32 but found float64

test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_ops.py:72: in test_dtype_preserves
    result = my_op(x)
E   RuntimeError: expected scalar type float32 but found float64
```

**Root Causes:**
1. Kernel does not support the input dtype
2. Implicit type promotion or conversion happening
3. Dtype dispatch is not configured correctly

**Triage Steps:**

```python
# Step 1: Check what dtypes the operation supports
# In the test, see which dtypes are passed
# @dtypes(torch.float32, torch.float64)  # These dtypes are tested
# def test_my_op(self, device, dtype):

# Step 2: Check if kernel handles that dtype
# In C++ code, look for dtype dispatch or instantiation
// Example: if only float32 is implemented, float64 will fail

# Step 3: Check for implicit conversions
# Some operations may convert input to a different dtype
# Verify that conversion is intentional (e.g., int64 key for embedding)

# Step 4: If error is a constraint, add to skip list
# (See skip_patterns.md for how to skip dtype combinations)
```

**Action:**
- If dtype should be supported: **implement or register the kernel for that dtype.**
- If dtype is intentionally unsupported: **add a skip** (see [skip_patterns.md](skip_patterns.md)).

---

### Category 7: Native C++ Runtime Error (TORCH_CHECK or Assertion)

**Symptom:**
```
RuntimeError: Dimension out of range (expected to be in range of [-3, 2], but got 5)

test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_ops.py:95: in test_dim_check
    result = x.reshape(-1)
E   RuntimeError: Dimension out of range (expected to be in range of [-3, 2], but got 5)

  File "torch/_C/__init__.pyi", line unknown in <unknown>
  (Detailed C++ stack will follow if TORCH_SHOW_CPP_STACKTRACES=1)
```

**Root Causes:**
1. Test is passing invalid input (test bug)
2. C++ kernel validation is too strict or incorrect
3. Dimension promotion or interpretation is wrong

**Triage Steps:**

```powershell
# Step 1: Enable C++ stack traces
$env:TORCH_SHOW_CPP_STACKTRACES = "1"
python -m pytest test/cpp_extensions/.../test_ops.py::test_dim_check -v

# This will print a full C++ stack trace showing where the error originated

# Step 2: Check the error message
# The error "Dimension out of range" suggests the dimension is invalid
# Verify the test is passing valid dimensions

# Step 3: Check the C++ validation logic
# Look in the kernel for TORCH_CHECK or assertions
# Example in aten/src/ATen/native/...:
# TORCH_CHECK(dim >= -ndim && dim < ndim, "...")

# Step 4: Add debug output to the C++ code
// In the kernel:
std::cout << "Received dim: " << dim << ", ndim: " << ndim << std::endl;
TORCH_CHECK(dim >= -ndim && dim < ndim, "...");
```

**Action:**
- If test input is invalid: **fix the test.**
- If validation is wrong: **fix the C++ validation logic.**
- If error message is unhelpful: **improve the TORCH_CHECK message** (include actual values).

---

### Category 8: Timeout or Hang

**Symptom:**
```
<Test appears to hang; no output after 5 minutes>
TIMEOUT: Test did not complete within 30 seconds
```

**Root Causes:**
1. Infinite loop in the kernel
2. Deadlock in stream/event synchronization
3. Blocking operation waiting for unavailable resource
4. Test creating too much work

**Triage Steps:**

```python
# Step 1: Add timeouts and try to narrow down which operation hangs
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out")

signal.signal(signal.SIGALRM, timeout_handler)

def test_op_that_hangs(self, device, dtype):
    signal.alarm(5)  # 5 second timeout
    try:
        result = x.some_op()
    finally:
        signal.alarm(0)  # Cancel timeout

# Step 2: Run under a debugger
# python -m pdb -c "b test_op_that_hangs" -c "c" test_ops.py
# Then interrupt (Ctrl+C) and inspect the stack

# Step 3: Check for blocking operations
# Look for stream.synchronize(), event.wait(), or blocking I/O

# Step 4: Try a smaller workload
# Instead of processing 1000 tensors, try 1
x = torch.randn(2, 2, device=device)  # Small tensor
result = x.some_op()
```

**Action:**
- If loop is infinite: **fix the kernel loop condition.**
- If synchronization is deadlocking: **check stream/event ordering.**
- If workload is too large: **reduce test size or add memory/time limits.**

---

## Decision Tree: What Should I Do?

```
Failure occurred
│
├─ Build/Import Error?
│  └─> Rebuild or fix DLL path (Category 1)
│
├─ Wrong Expected Value?
│  ├─> Test expectation wrong? Update test (Category 2)
│  └─> Actual value wrong? Fix backend code (Category 2)
│
├─ Operation Not Implemented?
│  ├─> Should be supported? Implement kernel (Category 3)
│  └─> Intentionally unsupported? Add skip (Category 3)
│
├─ Wrong Answer (Numerical)?
│  ├─> Tolerance too strict? Relax tolerance (Category 4)
│  ├─> Algorithm wrong? Fix kernel (Category 4)
│  └─> Precision accumulation? Document and increase tolerance (Category 4)
│
├─ Wrong Device in Output?
│  └─> Fix kernel to return on correct device (Category 5)
│
├─ Wrong Dtype?
│  ├─> Should be supported? Implement dtype (Category 6)
│  └─> Intentionally unsupported? Add skip (Category 6)
│
├─ C++ Runtime Error?
│  ├─> Invalid input? Fix test (Category 7)
│  └─> Wrong validation? Fix kernel (Category 7)
│
└─ Timeout or Hang?
   └─> Debug and fix infinite loop or deadlock (Category 8)
```

---

## When to Skip vs. When to Fix

### ✅ Appropriate to Skip

- Operation is intentionally not supported on this backend
- Feature requires hardware support not available on this device
- Complex feature with limited resources (will be implemented later)
- Known limitation that's documented elsewhere

**Example Skip:**
```python
skip_if(
    backend_device_match(["cpu"]),
    "CUDA synchronization not available on CPU"
)
```

### ❌ Inappropriate to Skip

- Missing kernel that should be trivial to implement
- Test bug that masks a real issue
- Temporary workaround without a plan to fix it
- Using skip to hide performance/precision problems

---

## Summary

1. **Categorize the failure** using the categories above
2. **Run the test locally** with debug output and stack traces
3. **Inspect the relevant code** (test or kernel)
4. **Decide:** fix the code, update the test, or add a skip
5. **Test locally** to verify the fix
6. **Commit and reference** the failure category in your commit message

See [adding_tests.md](adding_tests.md) for how to add tests, and [skip_patterns.md](skip_patterns.md) for how to properly skip unsupported cases.
