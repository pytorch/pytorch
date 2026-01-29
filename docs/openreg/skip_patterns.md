# Skip Patterns Reference

## Overview

Test skips are used to exclude specific test cases (device/dtype combinations, operations, etc.) from running. While skips are sometimes necessary, they must be used **deliberately and transparently**. This guide explains when to skip, which decorators to use, and how to write clear skip reasons.

---

## When Skips Are Appropriate

✅ **DO skip when:**

1. **Operation is intentionally unsupported** on this backend
   ```python
   @skip_if(backend_device_match("openreg"), "CUDA fused kernels not available on OpenReg")
   def test_fused_adam(self, device, dtype):
       pass
   ```

2. **Backend does not support a specific dtype**
   ```python
   @skip_if(backend_dtype_match("openreg", torch.bfloat16), "bfloat16 not supported on OpenReg")
   def test_bfloat16_op(self, device, dtype):
       pass
   ```

3. **Feature requires unavailable hardware or dependencies**
   ```python
   @skip_if(torch.cuda.is_available() == False, "CUDA required for this feature")
   def test_cuda_only(self, device, dtype):
       pass
   ```

4. **Test is flaky or hardware-dependent (with a plan to fix)**
   ```python
   # TODO: Fix timing issue in test_concurrent_streams (issue #12345)
   @skipIfRunningOn("openreg")
   def test_concurrent_streams(self, device, dtype):
       pass
   ```

---

## When Skips Are Inappropriate

❌ **DO NOT skip when:**

1. **Hiding a test that should fail**
   ```python
   # WRONG: Don't hide broken tests
   @skipAlways("This test is broken")
   def test_something(self):
       pass
   
   # RIGHT: Fix the test or file an issue
   def test_something(self):
       # TODO: Fix issue #999 before enabling this test
       pass
   ```

2. **Avoiding a required operation**
   ```python
   # WRONG: Don't skip core operations
   @skip_if(backend_device_match("openreg"), "add is complicated")
   def test_add(self, device, dtype):
       pass
   
   # RIGHT: Implement the operation or document why it's not supported
   def test_add(self, device, dtype):
       pass
   ```

3. **Papering over precision/performance issues**
   ```python
   # WRONG: Skip to avoid fixing a real bug
   @skip_if(backend_device_match("openreg"), "Results don't match CPU")
   def test_numeric_output(self, device, dtype):
       pass
   
   # RIGHT: Fix the numerical issue or document the expected tolerance
   def test_numeric_output(self, device, dtype):
       result = ...
       expected = ...
       self.assertTrue(torch.allclose(result, expected, rtol=1e-4, atol=1e-5))
   ```

4. **Convenience (avoiding fixing a failing test)**
   ```python
   # WRONG: Don't skip just because fixing is inconvenient
   @skipIfRunningOn("openreg")
   def test_dtype_promotion(self, device, dtype):
       pass
   
   # RIGHT: Investigate and fix the issue
   def test_dtype_promotion(self, device, dtype):
       # ... debug and fix ...
       pass
   ```

---

## Approved Skip Decorators

### 1. `@skipIf(condition, reason)` (Generic)

Skip if a boolean condition is true.

```python
from torch.testing._internal.common_utils import skipIf

@skipIf(not torch.cuda.is_available(), "CUDA not available")
def test_cuda_feature(self, device, dtype):
    pass

@skipIf(torch.version.cuda is None, "CUDA not installed")
def test_requires_cuda(self, device, dtype):
    pass
```

### 2. `@skipIfRunningOn(device_type)` (Device-Based)

Skip for a specific device type.

```python
from torch.testing._internal.common_utils import skipIfRunningOn

@skipIfRunningOn("cpu")
def test_gpu_only(self, device, dtype):
    pass

@skipIfRunningOn("openreg")
def test_skip_openreg(self, device, dtype):
    # This test will not run on OpenReg
    pass
```

### 3. `@skipIfTorchDynamo()` (Dynamic Compilation)

Skip for dynamic tensor compilation (TorchDynamo).

```python
from torch.testing._internal.common_utils import skipIfTorchDynamo

@skipIfTorchDynamo()
def test_graph_unreliable(self, device, dtype):
    # This test is not compatible with torch.compile
    pass
```

### 4. `@dtypesIfRunningOn(devices, dtypes)` / Custom Skip

Create a device/dtype-specific skip using the `@dtypes` pattern with conditional logic.

```python
from torch.testing._internal.common_device_type import dtypes

# Skip specific dtype on specific device
@dtypes(torch.float32, torch.float64)
def test_numeric(self, device, dtype):
    if device == "openreg" and dtype == torch.float16:
        self.skipTest("float16 not supported on OpenReg")
    # ... test logic ...
    pass
```

### 5. `@skipIfTorchDynamo()` + Custom Message

Combine decorators for more control.

```python
from torch.testing._internal.common_utils import skipIfRunningOn, skipIf

@skipIfRunningOn("meta")
@skipIf(torch.cuda.is_available() == False, "Requires CUDA")
def test_advanced_feature(self, device, dtype):
    pass
```

---

## How to Write a Good Skip Reason

### ✅ Good Skip Reasons

```python
# Clear, specific, and actionable
@skipIfRunningOn("openreg")
def test_async_work(self, device, dtype):
    # ✅ GOOD: Explains WHAT and WHY
    # "Async work not implemented on OpenReg; see issue #1234"
    pass

# Links to issue
@skipIf(
    not torch.cuda.is_available(),
    "CUDA required; feature only available on NVIDIA GPUs (see RFC-0045)"
)
def test_cuda_feature(self, device, dtype):
    pass

# References timeline
@skipIfRunningOn("openreg")
def test_sparse_ops(self, device, dtype):
    # Sparse ops coming in OpenReg v2.0 (March 2024)
    pass
```

### ❌ Bad Skip Reasons

```python
# Too vague
@skipIfRunningOn("openreg")
def test_something(self, device, dtype):
    # ❌ BAD: "Not working" doesn't explain why or when it will be fixed
    pass

# Misleading
@skipIfRunningOn("openreg")
def test_add(self, device, dtype):
    # ❌ BAD: "Not important" hides a missing feature
    pass

# No actionable info
@skipIf(True, "Disabled")
def test_something(self, device, dtype):
    # ❌ BAD: No context; can't determine if this should ever run
    pass
```

---

## Audit Your Skips

Periodically audit skip lists to ensure they're still valid:

```python
# Run tests with skip info
python -m pytest test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_ops.py -v --tb=short -q

# Search for all skips in a file
grep -n "skipIf\|skipIfRunningOn" test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_ops.py

# Check if a skip is still needed (comment it out temporarily and run the test)
# @skipIfRunningOn("openreg")  # TODO: Check if still needed (issue #1234)
# def test_something(self, device, dtype):
```

---

## OpInfo-Based Skips

For operator-based tests (using the `@ops` decorator), skips are defined in the `OpInfo` object:

```python
from torch.testing._internal.common_methods_invocations import OpInfo, SkipInfo

OpInfo(
    name="add",
    dtypes=all_types_and(torch.float16),
    supports_out=True,
    skips=(
        # Skip add on CPU for int8 (pretend reason for example)
        SkipInfo("CPU", "int8 not supported", {"dtype": torch.int8}),
        
        # Skip add on OpenReg for all dtypes (not yet implemented)
        SkipInfo("PrivateUse1", "add not yet implemented on OpenReg"),
        
        # Skip on CUDA with a specific dtype
        SkipInfo("CUDA", "Precision issue on V100 GPUs", {"dtype": torch.float16}),
    ),
)
```

Each `SkipInfo` entry should include:
- **Device:** Which device type (e.g., "CPU", "CUDA", "PrivateUse1")
- **Reason:** Clear, specific skip reason
- **Kwargs (optional):** Filter by dtype, dtype_args, etc.

---

## Summary

- **Skips are transparency tools:** They document intentional limitations, not workarounds for bugs.
- **Use approved decorators:** `@skipIf`, `@skipIfRunningOn`, `@skipIfTorchDynamo`.
- **Write clear reasons:** Include WHAT is skipped, WHY, and link to issues/timeline.
- **Audit regularly:** Remove stale skips and update reasons as features are implemented.
- **Document exceptions:** If you skip a core operation, explain why in a comment or issue.

See [adding_tests.md](adding_tests.md) for how to add tests with appropriate skips, and [failure_interpretation.md](failure_interpretation.md) for how to categorize failures and decide whether to skip or fix.
