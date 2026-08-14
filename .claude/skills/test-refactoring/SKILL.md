---
name: test-refactoring
description: Make PyTorch tests device-agnostic so they run on all accelerator backends (CUDA, XPU, MPS, HPU, PrivateUse1/OpenReg). Use when refactoring test files to remove CUDA-specific code, when user mentions "device-agnostic", "device-generic", "decouple from CUDA", "test refactoring", "out-of-tree backend" test support, or @onlyAccelerator/@skipOps/hw_classification migrations.
---

# PyTorch Test Refactoring (Device-Agnostic Tests)

This skill guides refactoring PyTorch tests to be device-agnostic, based on patterns from the PyTorch Test Refactoring project (https://github.com/orgs/pytorch/projects/154, 144 completed items).

## Goal

Tests must run on **all registered accelerator backends** without modification. Any hardcoded CUDA/XPU assumption is a bug for out-of-tree (OOT) backends like PrivateUse1/OpenReg.

## When to use this skill

- Refactoring a test file that contains `.cuda()`, `torch.cuda.*`, `@onlyCUDA`, `@requires_cuda`, `TEST_CUDA`, or `only_for=("cpu", "cuda")`
- Making a test file device-agnostic / device-generic for out-of-tree backends
- PR titles like "[Test] Make X device-agnostic", "[Testcase Refactoring] Decouple X from CUDA-specific"
- Adding `hw_classification` to test classes
- Migrating op skips from `op_db` to `@skipOps`

## Workflow

1. **Find CUDA-specific patterns**: search the file for `cuda`, `TEST_CUDA`, `onlyCUDA`, `requires_cuda`, `GPU_TYPE`, `HAS_CUDA`, `only_for`, `torch.accelerator` misuse.
2. **Classify each test**: device-generic (works everywhere), accelerator-only (needs any GPU), or truly device-specific (tests cuDNN/NCCL/CUDA-only behavior).
3. **Apply the transformations below** in order.
4. **Validate**: run the file on CPU at minimum; run specific test classes only, never the whole suite.

## Transformation Patterns

### 1. Remove `only_for` from `instantiate_device_type_tests`

```python
# Before
instantiate_device_type_tests(TestFoo, globals(), only_for=("cpu", "cuda"))

# After
instantiate_device_type_tests(TestFoo, globals())
```

Reference: #184661. Keep `allow_mps=True`/`allow_xpu=True` only if those backends were previously excluded by default.

### 2. `@onlyCUDA` / `@onlyOn(["cuda", "xpu"])` → `@onlyAccelerator`

For tests that need *any* accelerator but not CPU:

```python
# Before
@onlyCUDA
def test_foo(self, device):

# After
from torch.testing._internal.common_device_type import onlyAccelerator
@onlyAccelerator
def test_foo(self, device):
```

References: #183587 (8 occurrences in test_linalg), #183874, #184162. `onlyAccelerator` skips on CPU and meta (see [common_device_type.py](../../torch/testing/_internal/common_device_type.py)).

When generalizing dtypes after switching to `@onlyAccelerator`, add per-backend dtype decorators, e.g. `@dtypesIfMPS(torch.float16)` alongside `@dtypes(torch.bfloat16, torch.float16)` (#183587).

### 3. `.cuda()` / hardcoded `"cuda"` device → `device` param

```python
# Before
m.cuda()
x = torch.zeros(n).cuda()
cpu_result = torch.matmul(a.cpu(), b.cpu()).cuda().half()

# After
m.to(device)
x = torch.zeros(n, device=device)
cpu_result = torch.matmul(a.cpu(), b.cpu()).to(device=device, dtype=torch.half)
```

Reference: #183587. Inside `instantiate_device_type_tests` classes, always use the injected `device` argument.

### 4. `torch.cuda.is_available()` runtime guards → decorators or `torch.accelerator`

```python
# Before
def test_foo(self):
    if not torch.cuda.is_available():
        return

# After (preferred: decorator)
@onlyAccelerator
def test_foo(self, device):

# After (generic runtime check)
if torch.accelerator.is_available():
    ...
```

Reference: #183874. For accelerator device type: `torch.accelerator.current_accelerator().type` (e.g. #189825 replaced `torch.cuda.is_available()` in fx/test_common_passes.py).

### 5. Device module APIs → `torch.get_device_module`

```python
# Before
torch.cuda.manual_seed_all(seed)
torch.cuda.reset_peak_memory_stats()

# After
torch.get_device_module(device_type).manual_seed_all(seed)
torch.get_device_module(self.device_type).reset_peak_memory_stats()
```

References: #190172, #190188 (ONNX test commons). For memory stats prefer `torch.accelerator` API where available.

### 6. Remove module-level hardcoded device detection

```python
# Before (module level)
device_type = acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"

# After: delete the module-level variable; use the injected `device` param in tests.
# Non-device-type test classes: call torch.accelerator.current_accelerator() inline.
```

Reference: #184192. Replace `device_type == "xpu"` with `torch.device(device).type == 'xpu'`; `device.startswith("cuda")` with `torch.device(device).type == 'cuda'`.

### 7. Wrong-device / cross-device error tests → `_get_other_device` helper

```python
# Before
if torch.cuda.is_available():
    wrong_device = 'cpu' if self.device_type != 'cpu' else 'cuda'

# After (add helper to the class)
def _get_other_device(self, dtype=None):
    if self.device_type != 'cpu':
        return 'cpu'
    if torch.accelerator.is_available():
        other = torch.accelerator.current_accelerator().type
        if other == 'mps':  # backend lacking the checked behavior
            return None
        return other
    return None

wrong_device = self._get_other_device(dtype=dtype)
if wrong_device is not None:
    ...
```

Reference: #183588 (test_linalg error-path tests).

### 8. In-body guards that silently exclude backends → explicit skip decorators

A hardcoded `torch.cuda.is_available() or torch.xpu.is_available()` body guard silently hides MPS/HPU/PrivateUse1 coverage. Declare exclusions explicitly:

```python
# Before
if torch.cuda.is_available() or torch.xpu.is_available():
    devices.append(device)

# After
@skipMPS
def test_foo(self, device):
    if torch.accelerator.is_available():
        devices.append(device)
```

Available: `skipMPS`, `skipMPSIf`, `skipCUDAIf`, `skipXPU`, `skipHPU`, `skipPRIVATEUSE1`, `skipCUDAIfNoMagma`, `skipCUDAIfRocm`, etc. Explicit skips make coverage gaps visible; hidden guards do not.

### 9. CPU-only tests → `@onlyCPU`

```python
# Before
def test_state_dict(self):

# After
@onlyCPU
def test_state_dict(self, device):
```

Reference: #183874.

### 10. Extract device-specific tests into a separate class

When a test genuinely requires CUDA/cuDNN, move it out of the generic class:

```python
# Before: @unittest.skipIf(not TEST_CUDNN, ...) methods inside TestConvolutionNN

# After: new class at the end of the file
class TestConvolutionNNCUDA(TestCase):
    @unittest.skipIf(not TEST_CUDNN, "CUDNN not available")
    def test_cudnn_not_mutate_stride(self):
        ...  # may keep .cuda() here; class is CUDA-only
```

References: #184242, #184243 (TestConvolutionNNCUDA), #183586 (TestLinalgCUDA), #182433 (TestProfilerCUDA), #188090/#188210 (split CUPTI monitor tests into test_cupti_monitor.py). Generic class gets `instantiate_device_type_tests`; the extracted class does not.

### 11. Split mixed test classes into cohesive per-concern classes

Large classes mixing device-generic and device-specific tests should be split, extracting shared helpers into a mixin:

```python
class _SubclassCompileCheckMixin:
    def _compile_check(self, fn, inps, ...): ...

class SubclassTests(_SubclassCompileCheckMixin, torch._dynamo.test_case.TestCase):
    ...
```

Reference: #185238 (dynamo/test_subclasses.py).

### 12. Migrate op_db skips to `@skipOps` on the test method

Device-agnostic skips should live on the test, not buried in OpInfo definitions:

```python
# Before (in common_methods_invocations.py op_db entry)
skips=(DecorateInfo(unittest.skip("..."), "TestBwdGradients", "test_fn_grad", device_type="cuda", dtypes=(torch.float64,)),)

# After (on the test in test_ops_gradients.py)
@skipOps(
    (
        OpSkip("op_name", "variant", device_type=("cuda",), dtypes=(torch.float64,), expected_failure=False),
    )
)
def test_fn_grad(self, device, dtype, op):
```

References: #184355 [3/N] TestBwdGradients, #184356 [4/N] TestFwdGradients, #184678 [5/N] TestMeta; infrastructure: #177256/#183541 (unified skipOps), #178565 (DecorateInfo accepts device_type lists). See `skipOps` in [common_device_type.py](../../torch/testing/_internal/common_device_type.py).

### 13. Multi-GPU requirements → `skip_if_lt_x_gpu`

```python
# Before
@unittest.skipIf(torch.cuda.device_count() < 2, "needs 2 GPUs")

# After
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
@skip_if_lt_x_gpu(2)
def test_foo(self):
```

References: #189686, #189687, #189693, #189694, #190177–#190180 (DCP e2e migrations). Note it checks CUDA/HPU/XPU; PrivateUse1 support tracked separately (#190662 adds `at_least_x_gpu` support).

### 14. Distributed accelerator tests → `requires_accelerator_dist_backend` + `distributed_backend()` hook

```python
# Before
@skip_if_lt_x_gpu(2)
@requires_nccl()
def test_allreduce(self):

# After
@requires_accelerator_dist_backend(["nccl", "xccl", "hccl"])
def test_allreduce(self):
    backend = self.distributed_backend()  # resolves per device type
```

References: #190152, #190153, #184181 (`distributed_backend()` hook on `DeviceTypeTestBase`, defaults to `dist.get_default_backend_for_device(cls.device_type)`), #189860 (detect registered accelerator backends in distributed helpers), #190182 (decouple hardcoded device types in distributed test infra).

### 15. Add `hw_classification` to test classes

Classify every test class so `--hw-classification` filtering works:

```python
from torch.testing._internal.common_utils import HardwareClassification

class TestLinalg(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR   # instantiated via instantiate_device_type_tests

class TestFoo(TestCase):
    hw_classification = HardwareClassification.GENERIC       # device-agnostic framework logic, no device param

class TestConvolutionNNCUDA(TestCase):
    hw_classification = HardwareClassification.CUDA          # device-specific class (replaces @onlyCUDA at class level)
```

References: #186918 (infrastructure in common_utils.py), #190508/#190991 (rollout), #190181 (remove nightly fallbacks). Categories: `GENERIC`, `ACCELERATOR`, `CPU`, `CUDA`, `MPS`, `XPU`.

### 16. Other useful decorators/hooks

- `@onlyNativeDeviceTypes` — CPU, CUDA, Meta, PrivateUse1 only
- `@onlyCUDAAndPRIVATEUSE1` — transitional
- `@largeTensorTest("40GB")` — with device arg for accelerator
- `has_sufficient_memory()` hook on `DeviceTypeTestBase` (#184648)
- `test_exclusions` class attribute for dtype-aware per-class exclusions (#185489, #180820)
- `op_allowlist` registry for OOT backends reusing test suites (#181703)
- `@dtypesIfPRIVATEUSE1`, `@dtypesIfMPS`, `@dtypesIfCUDA` for per-backend dtype sets

## Inductor/Dynamo test specifics

- `GPU_TYPE` and `HAS_CUDA_AND_TRITON` come from `torch.testing._internal.inductor_utils`; when decoupling from CUDA, gate on triton availability for the current accelerator instead of `torch.cuda.is_available()` (#190507 test_auto_chunker, #190660 test_compiled_autograd, #190668 test_autoheuristic, #180328/#180344/#180374 dynamo tests).
- `torch._inductor.test_case.TestCase` device-parametrized variants exist for dynamo/inductor tests (#181497/#181498 ReproTestsDevice / ReproTestsAllDevices pattern: move tests into device-parameterized class, keep CPU-only ones separate).

## Common mistakes to avoid

1. **Editing `build/` generated files** — never; edit the test source.
2. **Keeping `cuda.is_available()` guards "just in case"** — they silently exclude new backends; use skip decorators instead.
3. **Blanket `@onlyAccelerator` on tests using CUDA-only APIs** (cuDNN flags, `torch.backends.cuda.*`, NCCL) — extract to a device-specific class instead.
4. **MPS dtype gaps** — after `@onlyCUDA → @onlyAccelerator`, MPS may fail on missing dtypes; add `@dtypesIfMPS` or `@skipMPS` deliberately.
5. **Running the full test suite to validate** — run only the touched classes, e.g. `python test/test_linalg.py TestLinalgCPU.test_norm_fused_type_promotion_cpu_float32`.
6. **Forgetting `del scope[...]` behavior** — after `instantiate_device_type_tests`, only instantiated names like `test_foo_cpu` are runnable; use `-k test_foo` to match all.

## Validation checklist

- [ ] File has no remaining `torch.cuda.*`, `.cuda()`, `TEST_CUDA` outside extracted device-specific classes
- [ ] `instantiate_device_type_tests` has no `only_for` unless justified
- [ ] Every test class has `hw_classification` where the rollout requires it
- [ ] New skips use decorators, not in-body guards
- [ ] CPU run passes: `python test/<file>.py -k <ClassName>CPU`
- [ ] If available, verify with OpenReg: `python test/run_test.py --openreg <file>` or run the torch_openreg test suite

## Reference PRs (merged, project #154)

| Pattern | PRs |
|---|---|
| only_for removal | #184661 |
| onlyCUDA → onlyAccelerator | #183587, #183874, #184162, #188305 |
| torch.accelerator migration | #183588, #184192, #189825, #185266 |
| get_device_module | #190172, #190188 |
| Extract CUDA class | #184242, #184243, #183586, #182433, #188090 |
| Split classes | #185238, #181497, #181498 |
| skipOps migration | #184355, #184356, #184678, #183541, #178565 |
| skip_if_lt_x_gpu | #189686, #189687, #190177–#190180 |
| Distributed backend-agnostic | #184181, #190152, #190153, #189860, #190182, #190309 |
| hw_classification | #186918, #190508, #190991, #190181 |
| Device exclusions infra | #180820, #185489, #184648 |
| OpInfo-based suites | #176717, #176593, #185699, #185881 |
