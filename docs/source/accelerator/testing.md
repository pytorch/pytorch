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

OOT backends should adopt tests incrementally based on their current level of support. Tests can be selected by category using the `--hw-classification` flag:

| Classification                 | Meaning                                      |
| ------------------------------ | -------------------------------------------- |
| `GENERIC`                      | No accelerator needed — pure framework logic |
| `DEVICE_GENERIC`               | Expected to run on all accelerator backends  |
| `CPU` / `CUDA` / `MPS` / `XPU` | Specific device type (use sparingly)         |


### Generic Tests

Generic tests validate framework functionality that does not depend on accelerator support. OOT backends can run these tests directly in their PyTorch environment:

```bash
python test/run_test.py --hw-classification GENERIC
```

### Device-generic Tests

Device-generic tests represent the common functionality PyTorch expects from accelerator backends. OOT backends should aim to pass as many of these as possible:

```bash
python test/run_test.py --hw-classification DEVICE_GENERIC
```

However, backend feature coverage may differ during bring-up. Common gaps include:

- Missing operator implementations.
- Partial dtype support for operators.
- Backend-specific numerical behavior or precision differences.

OOT backends can adapt upstream tests through the configuration options described in [Test Reuse Configuration](#test-reuse-configuration) below, rather than forking and modifying the tests.

### Device-specific Tests

Device-specific tests target functionality or behavior tied to a particular device type (e.g. `CPU`, `CUDA`, `MPS`, `XPU`):

```bash
python test/run_test.py --hw-classification CUDA
```

If an OOT backend provides equivalent functionality, applicable tests may be reused through the normal device test instantiation flow:

```bash
python test/run_test.py -i test_torch
```

Some tests may still be blocked by legacy device guards such as `@onlyCUDA` or `@onlyOn(["cuda"])`. In this case, [`bypass_device_restrictions`](#bypass_device_restrictions) can be enabled to run the tests on the OOT backend device.



## Test Reuse Configuration

OOT backends can adjust test coverage through the following configuration options.
These options allow backends to start with supported functionality and gradually expand coverage as support improves.



### Restricting Test Scope

Test scope restriction allows OOT backends to reuse upstream tests selectively based on their supported functionality. Unsupported cases can be excluded and test behavior can be customized through the following configurations:

| Configuration     | Type              | Purpose                                                      |
| ----------------- | ----------------- | ------------------------------------------------------------ |
| `op_allowlist`    | `Collection[str]` | Limit `@ops` tests to a supported operator subset            |
| `test_exclusions` | `dict`            | Exclude unsupported test classes, methods, or dtype variants |

> These configurations can be set through `set_test_configs()`. Calling `set_test_configs()` without arguments resets all configurations to `None`.

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



## Bring-up Workflow

A recommended order for adopting PyTorch tests in a new OOT backend:

1. **Identify reusable tests** — use `--hw-classification GENERIC DEVICE_GENERIC` to find tests that may be reused.
2. **Restrict unsupported ops** — use `op_allowlist` to run only supported operators during early bring-up.
3. **Handle backend differences** — add `op_overrides` for expected failures, skips, and precision differences.
4. **Exclude incompatible tests** — use `test_exclusions` only for functionality that cannot reasonably be reused.

> If a reusable test is blocked by legacy device guards (such as `onlyCUDA`), set `bypass_device_restrictions = True` when the backend supports the required functionality.

For a complete runnable example, see [test_testing.py][OpenReg test_testing].

[OpenReg test_testing]: https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_testing.py
