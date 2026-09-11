# Test Suite Reuse

## Background

PyTorch has a large upstream test suite. OOT backends can reuse these tests directly to ensure quality. Two mechanisms make this possible: **test classification** and **dynamic test configuration**. Without requiring changes to upstream tests.

## Core Mechanisms

### Test Classification

Test classification labels each test class (`GENERIC` / `ACCELERATOR` / device-specific), enabling runtime filtering via `--hw-classification` (supported by both `unittest` and `pytest`).


| Classification                 | Test scope                                                |
| ------------------------------ | --------------------------------------------------------- |
| `GENERIC`                      | Validates framework functionality that is independent of accelerator support. |
| `ACCELERATOR`                  | Validates common functionality expected from accelerator backends.  |
| Device-specific (`CPU`, `CUDA`, `XPU`, `MPS`) | Validates functionality or behavior specific to a particular backend.          |


**For OOT backends, upstream test reuse should primarily focus on `ACCELERATOR` tests.** `GENERIC` tests are already covered by upstream CPU CI. Device-specific tests target built-in backends. OOT backends typically maintain their own device-specific tests downstream.


**Running upstream `ACCELERATOR` tests**

Configure the following environment variables before running the upstream test suite:

- `PYTORCH_TESTING_DEVICE_FOR_CUSTOM` adds the specified custom device type(s) for testing.
- `PYTORCH_TESTING_DEVICE_ONLY_FOR` limits testing to the specified device type(s).

```bash
export PYTORCH_TESTING_DEVICE_FOR_CUSTOM=<your_backend>
export PYTORCH_TESTING_DEVICE_ONLY_FOR=<your_backend>
# Run the full upstream ACCELERATOR test suite
python test/run_test.py --hw-classification ACCELERATOR
```

### Dynamic Test Configuration

`ACCELERATOR` tests cover a broad range of accelerator functionality. During backend bring-up, OOT backends often add support for accelerator features incrementally, so some tests may not yet be applicable. Common cases include:

- Missing operator implementations.
- Partial dtype support for operators.
- Backend-specific numerical behavior or precision differences.

Dynamic Test Configuration provides a set of configuration options that allow OOT backends to **control upstream tests reuse** without modifying the tests themselves:


| Option | What it does | When to use |
|---|---|---|
| `op_allowlist` | Operator test allowlist | Early bring-up |
| `test_exclusions` | Exclude class/method/dtype-level variants | Unsupported features |
| `op_overrides` | Per-operator decorator overrides | Specific ops need special handling |
| `bypass_device_restrictions` | Skip `@onlyCUDA` and similar guards | Temporary workaround |

OOT backends configure all options except `bypass_device_restrictions` through `PrivateUse1TestBase.set_test_configs()` (inherited from [`DeviceTypeTestBase`](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_device_type.py)). 

Each call to `set_test_configs()` fully replaces the existing configuration: omitted keyword arguments are reset to `None`. Therefore, provide all desired configurations in a single call. 

`bypass_device_restrictions` is configured separately as a class attribute on the test subclass.

#### `op_allowlist`

Limit `@ops`-parametrized tests to supported operators. Values match `OpInfo.full_name` (e.g. `"add.Tensor"`, `"linalg.norm"`):

```python
PrivateUse1TestBase.set_test_configs(
    op_allowlist=("add.Tensor", "sub.Tensor", "mul.Tensor")
)
```

Ops not in the allowlist produce no test variants.

#### `test_exclusions`

Exclude incompatible test classes, methods, or dtype variants:

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

#### `op_overrides`

Customize `@ops` tests per-operator, for example to relax precision:

```python
import torch
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
    }
)
```

Overrides are applied on top of existing `OpInfo` decorators.

#### `bypass_device_restrictions`

Some upstream tests are blocked by legacy device guards like `@onlyCUDA`. Set `bypass_device_restrictions = True` on the test class to run them:

```python
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from test_prims import TestPrims

class TestPrimsOpenReg(TestPrims):
    bypass_device_restrictions = True

instantiate_device_type_tests(TestPrimsOpenReg, globals(), only_for="openreg")
```

> This is a temporary workaround for OOT backends. Once you confirm that a test works on your backend device, please consider upstreaming the fix by removing unnecessary legacy guards or opening an issue to discuss the required changes.


**Example** — a complete configuration combining multiple options:

```python
import torch
from torch.testing._internal.common_device_type import (
    PrivateUse1TestBase,
    instantiate_device_type_tests,
    precisionOverride,
)
from torch.testing._internal.opinfo.core import DecorateInfo
from test_ops import TestOps

PrivateUse1TestBase.set_test_configs(
    op_allowlist=("add.Tensor", "mul.Tensor", "sin", "cos"),
    test_exclusions={
        "TestTensorIndexing": ["test_index_put"],
    },
    op_overrides={
        "div.Tensor": [
            DecorateInfo(
                precisionOverride({torch.float32: 1e-2}),
                "TestOps", "test_div",
            ),
        ],
    },
)

instantiate_device_type_tests(TestOps, globals(), only_for="openreg")
```


## Bring-up Workflow

A step-by-step workflow for bringing a new OOT backend from initial bring-up to passing upstream `ACCELERATOR` tests.

### 1. Verify backend registration

Confirm that your OOT backend is properly registered, recognized by the accelerator API, and can be used to create tensors:

```python
import torch

# Verify that the accelerator API sees your backend
assert torch.accelerator.is_available()
assert torch.accelerator.device_count() > 0

# Verify the registered PrivateUse1 backend name
assert torch._C._get_privateuse1_backend_name() == "<your_backend>"

# Verify the current accelerator
backend = torch.accelerator.current_accelerator().type
assert backend == "<your_backend>"

# Verify tensor creation
t = torch.tensor([1, 2, 3], device=backend)
assert t.device.type == backend
```

### 2. Configure environment variables

Configure the upstream test infrastructure to run only on your backend:

```bash
export PYTORCH_TESTING_DEVICE_FOR_CUSTOM=<your_backend>
export PYTORCH_TESTING_DEVICE_ONLY_FOR=<your_backend>
```

### 3. Run accelerator tests to assess coverage

Run tests classified as `ACCELERATOR` to evaluate initial backend coverage and identify unsupported functionality and tests that require configuration:

```bash
# Runs all tests with ACCELERATOR classification across the test suite
python test/run_test.py --hw-classification ACCELERATOR
```

During early bring-up, start with a single test file and gradually expand the scope. For example:

```bash
python test/run_test.py -i test_torch --hw-classification ACCELERATOR
```

For tests that do not yet pass, analyze the failures and choose the appropriate configuration described in Steps 4–6.

### 4. Restrict to supported ops

Use `op_allowlist` if required operators are not yet implemented:

```python
PrivateUse1TestBase.set_test_configs(
    op_allowlist=("add.Tensor", "mul.Tensor", "sin", "cos"),
)
```

### 5. Handle precision differences

Use `op_overrides` for backend-specific numerical or precision differences:

```python
PrivateUse1TestBase.set_test_configs(
    op_allowlist=("add.Tensor", "div.Tensor"),
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

### 6. Exclude unsupported features

Use `test_exclusions` for functionality or dtypes that are not yet supported:

```python
PrivateUse1TestBase.set_test_configs(
    op_allowlist=("add.Tensor", "mul.Tensor"),
    test_exclusions={
        "TestTensorOps": ["test_advanced_indexing"],
    },
)
```

### 7. Gradually expand coverage

As support improves, gradually remove entries from `op_allowlist`, `test_exclusions`, and `op_overrides` where they are no longer needed.

For a complete runnable configuration example, see [test_testing.py](https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_testing.py).

## Suggestions

- Connect your downstream repo to upstream CI via CRCR (see [CI Integration](ci.md)).
