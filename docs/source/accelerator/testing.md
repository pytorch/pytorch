# Test Suite Reuse

## Background

PyTorch maintains a large, comprehensive test suite covering operators, modules, framework functionality, and device behavior.
Tests are categorized according to their hardware requirements:

- **Generic tests** — validate framework logic independent of accelerator support.
- **Device-generic tests** — validate behavior expected to be consistent across accelerator backends.
- **Device-specific tests** — rely on capabilities or behavior specific to a particular device type.

Among these, **device-generic tests define the baseline functionality that PyTorch expects accelerator backends to support**.

Instead of maintaining an independent test suite, out-of-tree (OOT) backends can reuse upstream tests to validate compatibility with PyTorch's backend requirements and expected behavior.

PyTorch provides two mechanisms to support this workflow:

1. **Hardware classification** — categorize tests by hardware requirements and filter applicable tests at runtime.
2. **Test reuse configuration** — adapt upstream tests to an OOT backend's capabilities without modifying upstream tests, including limiting unsupported cases, handling expected differences, and gradually expanding coverage.

This document describes the available infrastructure and recommended workflow for OOT backends to reuse PyTorch's upstream tests.



## Test Reuse Strategy

OOT backends should focus on `ACCELERATOR` tests — these define the common functionality expected across all backends. The full classification scheme is listed below for reference:

| Classification                 | Meaning                                      |
| ------------------------------ | -------------------------------------------- |
| `GENERIC`                      | No accelerator needed — pure framework logic |
| `ACCELERATOR`                  | Expected to run on all accelerator backends  |
| `CPU` / `CUDA` / `MPS` / `XPU` | Specific device type (use sparingly)         |


### Generic Tests

Generic tests validate framework functionality that does not depend on accelerator support. These tests are already exercised by PyTorch's own CPU CI across all supported architectures, so OOT backends do not need to run them.

### Accelerator Tests

Accelerator tests represent the common functionality PyTorch expects from accelerator backends. OOT backends should aim to pass as many of these as possible:

```bash
python test/run_test.py --hw-classification ACCELERATOR
```

However, backend feature coverage may differ during bring-up. Common gaps include:

- Missing operator implementations.
- Partial dtype support for operators.
- Backend-specific numerical behavior or precision differences.

OOT backends can adapt upstream tests through the configuration options described in [Test Reuse Configuration](#test-reuse-configuration) below, rather than forking and modifying the tests.

### Device-specific Tests

Device-specific tests target functionality or behavior tied to a particular device type (e.g. `CPU`, `CUDA`, `MPS`, `XPU`). These tests use device-specific APIs and should not be run by OOT backends.



## Test Reuse Configuration

OOT backends can adjust test coverage through the following configuration options.
These options allow backends to start with supported functionality and gradually expand coverage as support improves.

The following configuration options are currently supported:

| Configuration              | Type              | Purpose                                                   |
| -------------------------- | ----------------- | --------------------------------------------------------- |
| `op_allowlist`             | `Collection[str]` | Limit `@ops` tests to a supported operator subset         |
| `op_overrides`             | `dict`            | Customize `@ops` test behavior per-operator               |
| `test_exclusions`          | `dict`            | Exclude unsupported test classes, methods, or dtype variants |
| `bypass_device_restrictions` | `bool`          | Override legacy `@onlyCUDA` / `@onlyOn` guards             |

All configurations except `bypass_device_restrictions` are set through `PrivateUse1TestBase.set_test_configs()` (inherited from [`DeviceTypeTestBase`](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_device_type.py)). Each call to `set_test_configs()` fully replaces the existing configuration: omitted keyword arguments are reset to `None`. Therefore, provide all desired configurations in a single call. `bypass_device_restrictions` is configured separately as a class attribute on the test subclass.


### Restricting Test Scope

Test scope restriction allows OOT backends to reuse upstream tests selectively based on their supported functionality. Unsupported cases can be excluded through `op_allowlist` and `test_exclusions`.

#### `op_allowlist`

Limit `@ops`-parametrized tests to supported operators. Values match `OpInfo.full_name` (e.g. `"add.Tensor"`, `"linalg.norm"`):

```python
PrivateUse1TestBase.set_test_configs(
    op_allowlist=("add.Tensor", "sub.Tensor", "mul.Tensor")
)
```

Ops not in the allowlist produce no test variants.

#### `test_exclusions`

Exclude incompatible test classes, methods, or dtype variants during device test generation:

```python
PrivateUse1TestBase.set_test_configs(
    test_exclusions={
        "TestCUDAGraphs": "*",                       # whole class
        "TestTensorIndexing": ["test_index_put"],    # specific methods
        "TestDtypeAware": {                          # dtype-level
            "test_dtype_filter": {"dtypes": [torch.float32]},
        },
    }
)
```

For dtype-level exclusions, a tuple variant `(float64, int32)` is excluded if **any** of its component dtypes appears in the excluded list.



### Adjusting Test Behavior

#### `op_overrides`

Customize `@ops` tests per-operator without modifying upstream `OpInfo` definitions, for example to relax precision requirements:

```python
import torch
from torch.testing._internal.common_device_type import PrivateUse1TestBase, precisionOverride
from torch.testing._internal.opinfo.core import DecorateInfo

PrivateUse1TestBase.set_test_configs(
    op_overrides={
        "op_precision": [DecorateInfo(precisionOverride({torch.float32: 1e-2}))],
    }
)
```

Overrides are applied on top of existing `OpInfo` decorators. When both `op_allowlist` and `op_overrides` are set, `op_allowlist` filters first, then `op_overrides` is applied to the remaining operators.



### Expanding Test Scope

#### `bypass_device_restrictions`

Some upstream tests may be applicable to an OOT backend but blocked by legacy device guards such as `@onlyCUDA` or `@onlyOn`. Enable `bypass_device_restrictions = True` directly on the test class to run them:

```python
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from test_prims import TestPrims

class TestPrimsOpenReg(TestPrims):
    bypass_device_restrictions = True

instantiate_device_type_tests(TestPrimsOpenReg, globals(), only_for="openreg")
```

> note: `bypass_device_restrictions` is a short-term workaround for OOT backends. Once you confirm that a test works on your backend device, please consider upstreaming the fix by removing unnecessary legacy guards or opening an issue to discuss the required changes.


## Bring-up Workflow

A recommended order for adopting PyTorch tests in a new OOT backend.

### 0. Verify backend registration

Confirm that your OOT backend is properly registered and visible through the accelerator API:

```python
import torch

# Verify that the accelerator API sees your backend
assert torch.accelerator.is_available()
assert torch.accelerator.device_count() > 0

# Check the registered PrivateUse1 backend name
print(f"PrivateUse1 backend name: {torch._C._get_privateuse1_backend_name()}")

# Check the current accelerator backend
backend = torch.accelerator.current_accelerator().type
print(f"Current accelerator backend: {backend}")

# Make sure this matches the backend name registered by your OOT backend
assert backend == "my_backend"  # replace with your backend name

# Verify tensor creation uses your backend
t = torch.tensor([1, 2, 3], device=backend)
assert t.device.type == backend
print(f"Tensor device: {t.device}")
```

### 1. Run accelerator tests to assess coverage

Run tests classified as `ACCELERATOR` to evaluate initial backend coverage and identify tests that require additional configuration:

```bash
# Runs all tests with ACCELERATOR classification across the test suite
python test/run_test.py --hw-classification ACCELERATOR
```

During early bring-up, you could start with a single test file and gradually expand the scope. For example:

```bash
python test/run_test.py -i test_torch --hw-classification ACCELERATOR
```

### 2. Restrict unsupported ops

If some `@ops`-parametrized tests fail because your backend does not yet implement certain operators, you can use `op_allowlist` to limit tests to your supported subset.

For example, suppose a test subclass `TestTorchDeviceType` which has tests parametrized over the full operator database:

```python
from torch.testing._internal.common_device_type import ops, instantiate_device_type_tests
from torch.testing._internal.opinfo.core import op_db

class TestFoo(TestCase):
    @ops(op_db)
    def test_bar(self, device, dtype, op):
        ...
```

If only `add`, `mul`, `sin`, and `cos` are implemented on your backend, `test_bar` will fail for every other op. Use `op_allowlist` to run it only for those four:

```python
PrivateUse1TestBase.set_test_configs(
    op_allowlist=("add.Tensor", "mul.Tensor", "sin", "cos"),
)
```

### 3. Handle backend differences

If a test passes functionally but produces slightly different numerical results (e.g. precision differences), you can use `op_overrides` to relax tolerances.

For example, suppose an upstream test checks `div` with a tight tolerance that your backend cannot match on `float32`:

```python
class TestFoo(TestCase):
    @ops([op for op in op_db if op.name == "div"])
    def test_div(self, device, dtype, op):
        x = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
        expected = x / x
        actual = op(x, x)
        self.assertEqual(actual, expected)  # defaults: atol=1e-5, rtol=1e-5
        # Fails on your backend: actual differs beyond the default tolerance
```

Use `op_overrides` to widen the tolerance for this specific test:

```python
from torch.testing._internal.common_device_type import PrivateUse1TestBase, precisionOverride
from torch.testing._internal.opinfo.core import DecorateInfo

PrivateUse1TestBase.set_test_configs(
    op_overrides={
        "div.Tensor": [
            DecorateInfo(
                precisionOverride({torch.float32: 1e-2}),
                "TestFoo", "test_div",
            ),
        ],
    },
)
```

### 4. Exclude incompatible tests

If some test classes or methods cover functionality your backend does not yet support, you can use `test_exclusions` to skip them.

For example, suppose a test class has methods your backend does not yet support:

```python
class TestTensorOps(TestCase):

    def test_basic(self, device):
        ...

    def test_advanced_indexing(self, device):
        # Your backend does not support this indexing pattern yet
        ...
```

Use `test_exclusions` to skip `test_advanced_indexing` while still running `test_basic`:

```python
PrivateUse1TestBase.set_test_configs(
    test_exclusions={
        "TestTensorOps": ["test_advanced_indexing"],
    },
)
```


> If a reusable test is blocked by legacy device guards (such as `@onlyCUDA`), set `bypass_device_restrictions = True` on your subclass when the backend supports the required functionality.

For a complete runnable example, see [test_testing.py](https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_testing.py).
