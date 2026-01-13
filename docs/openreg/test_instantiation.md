# Test Instantiation Guide

## Overview

OpenReg tests are designed to be **highly parameterized** — a single test definition automatically expands into many concrete test cases, covering multiple device types, data types (dtypes), and operators. This guide explains the machinery behind that expansion and why the design improves testing.

## Why Parameterized Tests?

Imagine writing a test for the `add` operation. Without parameterization, you might write separate test functions for CPU/float32, CPU/float64, CUDA/float32, etc. — hundreds of nearly identical tests. Parameterization lets you write **one generic test** and automatically instantiate versions for all combinations:

```python
# ONE generic test template
def test_add(self, device, dtype):
    x = torch.randn(3, 3, dtype=dtype, device=device)
    y = torch.randn(3, 3, dtype=dtype, device=device)
    result = x + y
    expected = x.cpu().numpy() + y.cpu().numpy()
    self.assertTrue(torch.allclose(result, expected))

# Becomes MANY concrete tests at module load time:
# test_add_cpu_float32, test_add_cpu_float64, test_add_cuda_float32, ...
```

Benefits:
- **DRY (Don't Repeat Yourself):** One test body, many configurations.
- **Consistency:** All device/dtype combinations use the same logic.
- **Maintainability:** Fix the test once; fixes apply everywhere.
- **Coverage:** Easy to ensure all combinations are tested.

---

## Key Concepts

### 1. DeviceTypeTestBase

`DeviceTypeTestBase` is the base class for device-specialized test cases. It provides:

- **Device context:** The test knows its device type (CPU, CUDA, OpenReg, etc.)
- **Precision helpers:** Methods to compare results with appropriate tolerances
- **Setup/teardown:** Device-specific initialization and cleanup
- **Utility methods:** Device queries, memory management, stream handling

**Location:** [`torch/testing/_internal/common_device_type.py`](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_device_type.py)

**Usage in OpenReg tests:**

OpenReg tests typically inherit from `TestCase` and rely on the test instantiation framework to create device-specific subclasses. The key function `instantiate_device_type_tests()` is called at module scope to generate `CPUTestBase`-derived and device-specific classes.

Example from [`test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_device.py`](../../../test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_device.py):

```python
class TestDevice(TestCase):
    def test_device_count(self):
        """Test device count query"""
        count = torch.accelerator.device_count()
        self.assertEqual(count, 2)
```

At module scope, `instantiate_device_type_tests(TestDevice, globals())` transforms this into:
- `TestDeviceCPU` with `test_device_count_cpu()`
- `TestDevicePrivateUse1` (if OpenReg available) with `test_device_count_privateuse1()`

Each instantiated test receives the device string and can use it in assertions.

### 2. Parametrization Decorators

#### `@dtypes(*dtypes)` 

Parametrizes a test over multiple data types. The decorator expands a single test into multiple tests, one per dtype.

```python
from torch.testing._internal.common_utils import dtypes

@dtypes(torch.float32, torch.float64, torch.int32)
def test_my_op(self, device, dtype):
    x = torch.randn(2, 2, dtype=dtype, device=device)
    # Test runs 3x: once for each dtype
```

**Result (if CPU/CUDA devices available):**
- `test_my_op_cpu_float32`, `test_my_op_cpu_float64`, `test_my_op_cpu_int32`
- `test_my_op_cuda_float32`, `test_my_op_cuda_float64`, `test_my_op_cuda_int32`

#### `@ops(op_db)` (Advanced)

Parametrizes a test over a database of operator metadata objects (`OpInfo`). Each `OpInfo` describes an operator, its valid dtypes, devices, and special handling.

```python
from torch.testing._internal.common_methods_invocations import op_db

@ops(op_db)
def test_all_ops(self, device, dtype, op):
    # op is an OpInfo object with fields:
    #   op.name, op.dtypes, op.gradcheck_dtypes, op.supports_out, ...
    pass
```

This expands into hundreds of tests if `op_db` contains hundreds of operators.

**Location:** Decorator defined in [`torch/testing/_internal/common_device_type.py`](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_device_type.py)

### 3. OpInfo Objects

`OpInfo` is a metadata object that describes an operator and how it should be tested.

**Key Fields:**

| Field | Purpose |
|-------|---------|
| `name` | String name of the operation (e.g., `"add"`) |
| `dtypes` | Tuple of supported dtypes (e.g., `(torch.float32, torch.float64)`) |
| `dtypes_args` | Dtypes valid for specific arguments (for ops with mixed-dtype signatures) |
| `supports_out` | Whether the op supports an `out=` argument |
| `supports_inplace` | Whether the op has an in-place variant |
| `supports_autograd` | Whether the op is differentiable |
| `decompose_fn` | Optional decomposition (for testing operator fusion/lowering) |
| `skips` | List of `SkipInfo` objects indicating which device/dtype combinations to skip |

**Location:** [`torch/testing/_internal/common_methods_invocations.py`](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_methods_invocations.py)

**Example OpInfo:**

```python
OpInfo(
    name="add",
    dtypes=all_types_and(torch.float16),
    dtypes_args=partial_dtype_specifier(arg_dtypes={"alpha": (torch.int64,)}),
    supports_out=True,
    supports_inplace=True,
    supports_autograd=True,
    skips=(
        SkipInfo("CUDA", "dtype mismatch", {}),  # Skip on CUDA
        SkipInfo("PrivateUse1", "Not yet supported", {}),  # Skip on OpenReg
    ),
)
```

The test framework reads these metadata and:
1. Only instantiates tests for supported dtypes
2. Calls the test with the correct device and dtype
3. Skips combinations marked in the `skips` field

---

## How Tests Expand: A Detailed Example

### Starting with a Template

```python
# File: test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_ops.py

class TestOps(TestCase):
    
    @dtypes(torch.float32, torch.float64)
    def test_add(self, device, dtype):
        """Generic test template — accepts device and dtype parameters"""
        x = torch.randn(3, 3, dtype=dtype, device=device)
        y = torch.randn(3, 3, dtype=dtype, device=device)
        result = x + y
        self.assertEqual(result.device.type, device.split(":")[0])

if __name__ == "__main__":
    # This call transforms TestOps at module load time
    instantiate_device_type_tests(TestOps, globals())
    run_tests()
```

### After Instantiation

The `instantiate_device_type_tests()` function:

1. **Reads the template class** `TestOps`
2. **Identifies available devices** (CPU, CUDA, OpenReg if available)
3. **Creates device-specific subclasses:**
   ```python
   class TestOpsCPU(CPUTestBase):
       def test_add_cpu_float32(self):
           test_add(self, "cpu", torch.float32)
       
       def test_add_cpu_float64(self):
           test_add(self, "cpu", torch.float64)

   class TestOpsCUDA(CUDATestBase):
       def test_add_cuda_float32(self):
           test_add(self, "cuda:0", torch.float32)
       
       def test_add_cuda_float64(self):
           test_add(self, "cuda:0", torch.float64)

   class TestOpsPrivateUse1(PrivateUse1TestBase):
       def test_add_privateuse1_float32(self):
           test_add(self, "openreg:0", torch.float32)
       
       def test_add_privateuse1_float64(self):
           test_add(self, "openreg:0", torch.float64)
   ```

4. **Replaces the original** `TestOps` with the instantiated classes in `globals()`

### Result

**Before instantiation:** No runnable tests (template not recognized by test runners)

**After instantiation:** 6 concrete, runnable tests:
- `TestOpsCPU.test_add_cpu_float32`
- `TestOpsCPU.test_add_cpu_float64`
- `TestOpsCUDA.test_add_cuda_float32`
- `TestOpsCUDA.test_add_cuda_float64`
- `TestOpsPrivateUse1.test_add_privateuse1_float32`
- `TestOpsPrivateUse1.test_add_privateuse1_float64`

Each test runs independently with its own device and dtype.

---

## Expansion Flow (Code Path)

### 1. Test Loading

```
Python imports test file
  ↓
Test framework detects `instantiate_device_type_tests()` call
  ↓
Loop over available devices (CPU, CUDA, PrivateUse1, ...)
  ↓
For each device, create a subclass inheriting from device-specific base
  (e.g., CPUTestBase, CUDATestBase, PrivateUse1TestBase)
  ↓
For each method in template class:
  - If it has @dtypes, @ops, or other parametrization:
    - Expand it into multiple methods (one per dtype/op combo)
  - If it has no parametrization:
    - Copy it as-is (but rename to include device suffix)
  ↓
Register new classes in globals()
  ↓
Remove original template class from globals()
```

### 2. Key Functions

| Function | File | Role |
|----------|------|------|
| `instantiate_device_type_tests()` | `common_device_type.py` | Main orchestrator; creates device-specific subclasses |
| `_TestParametrizer` | `common_utils.py` | Base class for decorators (@dtypes, @ops, etc.) |
| `@dtypes` decorator | `common_device_type.py` | Marks a test for dtype parametrization |
| `@ops` decorator | `common_device_type.py` | Marks a test for operator parametrization |
| `DeviceTypeTestBase` | `common_device_type.py` | Base class with device context, precision helpers |
| `CPUTestBase`, `CUDATestBase`, etc. | `common_device_type.py` | Device-specific bases (inherit from `DeviceTypeTestBase`) |

---

## Common Mistakes and Pitfalls

### ❌ Mistake 1: Forgetting `instantiate_device_type_tests()`

```python
# WRONG — tests will not run
class TestMyOps(TestCase):
    def test_foo(self, device):
        pass

# No instantiate_device_type_tests() call!
if __name__ == "__main__":
    run_tests()
```

**Fix:** Add the call at module scope:

```python
instantiate_device_type_tests(TestMyOps, globals())
if __name__ == "__main__":
    run_tests()
```

### ❌ Mistake 2: Wrong Signature for Parametrized Tests

```python
# WRONG — the device and dtype parameters are required
@dtypes(torch.float32, torch.float64)
def test_foo(self):  # Missing 'device' and 'dtype' parameters!
    pass
```

**Fix:** Include all parameters in order: `device`, then `dtype`:

```python
@dtypes(torch.float32, torch.float64)
def test_foo(self, device, dtype):
    pass
```

### ❌ Mistake 3: Using Hard-Coded Device Strings

```python
# WRONG — test only runs on CPU, ignoring the 'device' parameter
@dtypes(torch.float32)
def test_foo(self, device, dtype):
    x = torch.randn(2, 2, device="cpu")  # Hard-coded!
    # ...
```

**Fix:** Use the `device` parameter:

```python
@dtypes(torch.float32)
def test_foo(self, device, dtype):
    x = torch.randn(2, 2, device=device)  # Use parameter!
    # ...
```

### ❌ Mistake 4: Ignoring Device Differences in Precision

```python
# WRONG — CPU and GPU have different precision characteristics
def test_numeric_output(self, device, dtype):
    result = some_compute(device=device)
    expected = cpu_reference(device="cpu")
    self.assertEqual(result, expected)  # Will fail on GPU!
```

**Fix:** Use device-aware comparison with tolerances:

```python
def test_numeric_output(self, device, dtype):
    result = some_compute(device=device)
    expected = cpu_reference(device="cpu")
    # Use assertAlmostEqual or torch.allclose with rtol/atol
    self.assertTrue(torch.allclose(result, expected, rtol=1e-5, atol=1e-6))
```

---

## Summary

- **Parameterized tests reduce duplication** by expressing one test logic that expands into many device/dtype combinations.
- **`DeviceTypeTestBase`** provides device context and precision helpers.
- **`@dtypes`, `@ops`, and other decorators** mark tests for parametrization.
- **`instantiate_device_type_tests()`** is called at module scope and transforms template classes into device-specific, runnable test classes.
- **Each instantiated test** has a unique name (e.g., `test_foo_cpu_float32`) and runs independently.
- **Common pitfalls** include forgetting the instantiation call, wrong signatures, hard-coded devices, and precision issues.

See [adding_tests.md](adding_tests.md) for a step-by-step workflow to add new tests.
