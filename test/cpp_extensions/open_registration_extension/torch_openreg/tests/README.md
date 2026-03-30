# OpenReg Testing Patterns

This document describes the standardized testing patterns used in the OpenReg test suite and how they relate to PyTorch's broader device-generic testing infrastructure.

## Overview

OpenReg tests live in this directory as individual `test_*.py` modules. Each module focuses on a specific feature area (device management, operators, memory, profiling, etc.) and follows PyTorch's unittest-based conventions.

### Test files

| File | Area |
|------|------|
| `test_device.py` | Device APIs, context managers, fork safety |
| `test_ops.py` | Factories, copy, ops, SDPA, quantization, fallback, custom autograd |
| `test_memory.py` | Allocator, leaks, pin memory, multi-device allocation |
| `test_storage.py` | Pinned storage, serialization, `torch.save` / `map_location` |
| `test_profiler.py` | Autograd profiler, `record_function` |
| `test_streams.py` | `torch.Stream`, accelerator stream APIs |
| `test_event.py` | `torch.Event` |
| `test_rng.py` | Generator, RNG state |
| `test_utils.py` | DLPack |
| `test_misc.py` | PrivateUse1 rename/registration, legacy types |
| `test_autocast.py` | Autocast / AMP |
| `test_autograd.py` | Autograd integration |

## Quick start for new contributors

1. **Install OpenReg** (from the extension root, one level above this directory):
   ```bash
   cd test/cpp_extensions/open_registration_extension/torch_openreg
   python -m pip install --no-build-isolation -e .
   ```
2. **Run a single test** to verify your setup:
   ```bash
   python tests/test_device.py TestDevice.test_device_count
   ```
3. **Run an entire file:**
   ```bash
   python tests/test_ops.py
   ```
4. **Suggested reading order** if you are new to OpenReg:
   - Start with the [parent README](../README.md) for project goals and directory layout.
   - Read `test_device.py` — it is the simplest file and covers the most fundamental integration (device registration, context managers).
   - Read `test_ops.py` — it covers the core operator patterns (factory, copy, fallback, SDPA, custom autograd).
   - From there, explore other `test_*.py` files based on the feature you are working on (see the [file table](#test-files) above).

## Why tests are structured this way

OpenReg is not a production backend — it is a **minimalist reference implementation** whose purpose is to verify that every integration point between PyTorch and a new backend works correctly (see the [Design Principles](../README.md#design-principles) in the parent README). The test structure follows directly from this goal:

**Per-feature modules map to integration capabilities.** Each `test_*.py` file corresponds to a category of backend integration (device management, operators, memory, streams, profiling, etc.). This makes it easy for backend authors to see which capabilities exist, run only the subset they are working on, and identify exactly which integration path is broken when a test fails. A new backend author can treat the file list as a checklist of features to implement.

**Plain unittest with hard-coded device strings (not `instantiate_device_type_tests`).** The tests in this directory are *OpenReg-specific*. They verify that the `openreg` backend correctly implements each integration hook, not that PyTorch operators produce correct numerics across all devices. Device-generic correctness testing is handled separately by PyTorch's mainline suite (which *does* use `instantiate_device_type_tests` and automatically includes `openreg` via `PrivateUse1TestBase` — see below). Keeping these two concerns separate means:
- Backend authors get fast, focused feedback on their integration work.
- Failures in this directory always point to the backend integration, never to PyTorch core.
- Tests stay minimal and readable, matching OpenReg's "just right" philosophy.

**Skips document the gap between the current implementation and full integration.** Every `@unittest.skip` or `@skipIfTorchDynamo` annotation carries a reason string. These are not just test metadata — they are the living record of what is not yet implemented. Reviewing the skips gives maintainers a quick picture of platform readiness without running the suite.

## Test instantiation

### OpenReg-specific tests (this directory)

Tests here use **plain unittest-style classes** that extend `TestCase` (or `NNTestCase` for neural-network tests). The device string `"openreg"` is hard-coded in each test. Every file ends with:

```python
import torch
from torch.testing._internal.common_utils import run_tests, TestCase

class TestFeature(TestCase):
    def test_something(self):
        x = torch.empty(3, device="openreg")
        self.assertEqual(x.device.type, "openreg")

if __name__ == "__main__":
    run_tests()
```

There is no call to `instantiate_device_type_tests` in this directory — each test explicitly targets the `openreg` device.

### PyTorch device-generic tests (mainline test suite)

PyTorch's mainline test suite (e.g. `test/test_ops.py`, `test/test_torch.py`) uses a **template-based** system where test classes are written once and then *instantiated* per device type. OpenReg plugs into this automatically via the `PrivateUse1TestBase` infrastructure.

**How it works:**

1. A template class defines tests that accept a `device` parameter:

```python
from torch.testing._internal.common_device_type import instantiate_device_type_tests

class TestCommon(TestCase):
    def test_foo(self, device):
        x = torch.randn(3, device=device)
        ...

instantiate_device_type_tests(TestCommon, globals())
```

2. `instantiate_device_type_tests` replaces `TestCommon` with device-specific classes like `TestCommonCPU`, `TestCommonCUDA`, and — when a PrivateUse1 backend is available — `TestCommonOPENREG` (the suffix is `device_type.upper()`).

3. Each test is renamed with a device suffix (e.g. `test_foo_openreg`) and receives the device string at runtime.

4. `PrivateUse1TestBase` (in `torch/testing/_internal/common_device_type.py`) detects the registered backend name via `torch._C._get_privateuse1_backend_name()` and sets up the device module and primary device automatically.

**OpInfo-driven tests** further parametrize over operators *and* dtypes:

```python
from torch.testing._internal.common_device_type import ops
from torch.testing._internal.common_methods_invocations import op_db

@ops(op_db)
def test_op(self, device, dtype, op):
    ...
```

Each entry in `op_db` is an `OpInfo` (defined in `torch.testing._internal.opinfo.core`) that declares `supported_dtypes` per device type. For a PrivateUse1 backend, the infrastructure maps `"privateuse1"` to the actual backend name when querying supported dtypes.

**When do OpenReg tests run through the mainline suite?** Whenever OpenReg is installed and `torch.openreg.is_available()` returns `True`, the mainline device-generic tests automatically include the `openreg` device. This is controlled by `is_privateuse1_backend_available()` in `common_utils.py`, which dynamically looks up the registered privateuse1 backend name via `torch._C._get_privateuse1_backend_name()`, retrieves the corresponding module (e.g., `torch.openreg`), and calls its `is_available()`. This mechanism is not specific to OpenReg — it works for any privateuse1 backend. In `common_device_type.py`, `instantiate_device_type_tests` appends `PrivateUse1TestBase` to the device test bases when this check passes, so device-generic tests automatically get an `openreg` variant alongside `cpu` and `cuda`.

## Skip and expected failure patterns

The test suite uses several mechanisms to skip tests that cannot run in certain environments. Choose the narrowest mechanism that fits your situation.

### `@unittest.skip(reason)`

Unconditionally skip a test or an entire class. Use when a feature is **not yet implemented** and the test is a placeholder for future work.

```python
@unittest.skip("openreg backend does not implement per-device RNG yet")
def test_generator_state(self):
    ...
```

Skipping an entire class:

```python
@unittest.skip("Skipping all quantization tests for openreg backend")
class TestQuantizationExtended(TestCase):
    ...
```

### `@skipIfTorchDynamo(msg)`

Skip when running under TorchDynamo (e.g. dynamo-wrapped test modes). Use when a test exercises functionality that Dynamo does not yet trace correctly for OpenReg.

```python
from torch.testing._internal.common_utils import skipIfTorchDynamo

@skipIfTorchDynamo("unsupported aten.is_pinned.default")
def test_pin_memory(self):
    ...
```

### `@skipIfWindows(msg=...)` and `@skipIfMPS`

Platform-specific skips for tests that rely on OS features (e.g. `fork`) or should not run on certain accelerators.

```python
from torch.testing._internal.common_utils import skipIfWindows

@skipIfWindows(msg="Fork not available on Windows")
def test_device_poison_fork(self):
    ...
```

### `@unittest.skipIf(condition, reason)`

Conditional skip based on a runtime check (library versions, device count, etc.).

```python
@unittest.skipIf(numpy.__version__ < "1.25", "requires numpy >= 1.25")
def test_open_device_numpy_serialization(self):
    ...
```

### `self.skipTest(reason)`

Runtime skip inside a test body, for conditions that can only be evaluated after setup.

```python
def test_profiler_multiple_devices(self):
    if torch.openreg.device_count() < 2:
        self.skipTest("requires at least 2 openreg devices")
    ...
```

### Stacking decorators

Multiple skip decorators can be stacked when a test has several exclusion conditions:

```python
@skipIfMPS
@skipIfWindows()
@skipIfTorchDynamo()
def test_autograd_init(self):
    ...
```

### Dtype restrictions (device-generic tests)

When using `instantiate_device_type_tests`, dtype-level restrictions use the `@dtypes` / `@dtypesIfPRIVATEUSE1` decorators:

```python
from torch.testing._internal.common_device_type import dtypes, dtypesIfPRIVATEUSE1

@dtypesIfPRIVATEUSE1(torch.float32, torch.float16)
def test_some_op(self, device, dtype):
    ...
```

`OpInfo.supported_dtypes` handles this automatically for OpInfo-driven tests — the framework only instantiates dtype combinations that the operator actually supports on each device.

### Note on `expectedFailure`

`unittest.expectedFailure` is **not currently used** in the OpenReg test suite. If a test is known to fail, prefer `@unittest.skip` with a reason string explaining when the skip can be removed. Use `expectedFailure` only when you want CI to detect *if the failure goes away* (i.e. as an alert that the skip can be removed), since `expectedFailure` will fail the test if it starts passing.

## Common testing patterns

This section documents recurring patterns used across the OpenReg test files. Refer to these when writing new tests.

### Assertion methods

`self.assertEqual` is the default for tensor comparisons (handles tolerance automatically). The suite also uses:

| Method | When to use | Example |
|--------|-------------|---------|
| `assertTrue` / `assertFalse` | Boolean conditions, `torch.all(...)` checks | `self.assertTrue(torch.all(x == 2))` |
| `assertRaisesRegex` | Verify an operation raises a specific exception | See [Error assertions](#error-assertions) below |
| `assertWarnsRegex` | Verify a warning is emitted | `with self.assertWarnsRegex(UserWarning, "In openreg autocast"):` |
| `assertIn` | Check membership (e.g. profiler event names) | `self.assertIn("traceEvents", trace_data)` |
| `assertIsNone` / `assertIsNotNone` | Null checks on device index, grad, etc. | `self.assertIsNone(device.index)` |
| `assertGreater` / `assertLess` | Ordering (timing bounds, counts) | `self.assertLess(elapsed, 60.0)` |
| `assertIsInstance` | Type checks on profiler output | `self.assertIsInstance(table, str)` |

For comparing tensor values element-wise, prefer `self.assertEqual` (tolerance-aware) over `torch.allclose` in new telests.

### Error assertions

Use `assertRaisesRegex` to verify that an operation raises the expected exception:

```python
with self.assertRaisesRegex(ValueError, "Both events must be created"):
    event1.elapsed_time(event2)
```

Use `assertWarnsRegex` for expected warnings (e.g. autocast dtype warnings):

```python
with self.assertWarnsRegex(UserWarning, "In openreg autocast"):
    with torch.amp.autocast(device_type="openreg", dtype=torch.bfloat16):
        ...
```

### `setUp` and `tearDown` for stateful tests

Tests that modify global device state or test for memory leaks use fixtures to ensure clean state:

```python
class TestMultiDeviceAllocation(TestCase):
    def setUp(self):
        self.device_count = torch.openreg.device_count()
        self.assertEqual(self.device_count, 2, "This test requires 2 OpenReg devices")

    def tearDown(self):
        torch.openreg.set_device(0)
```

For allocator and leak tests, `setUp` calls `gc.collect()` (and optionally `time.sleep`) to ensure the heap is settled before measuring:

```python
def setUp(self):
    gc.collect()
    time.sleep(0.1)
```

### Device control APIs

Tests use two device-control surfaces. Prefer `torch.accelerator` for device-agnostic code and `torch.openreg` for backend-specific APIs:

| API | When to use | Example |
|-----|-------------|---------|
| `torch.accelerator.device_index(i)` | Device-agnostic context manager | `with torch.accelerator.device_index(1): ...` |
| `torch.accelerator.set_device_index(i)` | Set current device generically | `torch.accelerator.set_device_index(0)` |
| `torch.openreg.set_device(i)` | Backend-specific device switch | `torch.openreg.set_device(0)` |
| `torch.openreg.device_count()` | Query device count | `if torch.openreg.device_count() < 2: self.skipTest(...)` |
| `torch.openreg.init()` | Explicit backend init (fork safety tests) | `torch.openreg.init()` |

### Serialization testing

Serialization tests in `test_storage.py` use several PyTorch-specific utilities:

```python
import tempfile
from torch.serialization import safe_globals

# Round-trip via torch.save / torch.load
with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "model.pt")
    torch.save(state_dict, path)
    loaded = torch.load(path, map_location="openreg", weights_only=True)

# When loading objects that require class allowlisting:
with safe_globals([numpy.dtype]):
    loaded = torch.load(path, weights_only=True)
```

Use `unittest.mock.patch.object` to test code paths that depend on internal PyTorch flags:

```python
with unittest.mock.patch.object(torch._C, "_has_storage", return_value=False):
    ...
```

### Multi-process fork testing

`test_device.py` tests fork safety by spawning a child process and communicating via a `Queue`:

```python
import multiprocessing

# Parent initializes
torch.openreg.init()

def child(q):
    try:
        torch.openreg.init()
    except Exception as e:
        q.put(e)

ctx = multiprocessing.get_context("fork")
q = ctx.Queue()
p = ctx.Process(target=child, args=(q,))
p.start()
p.join()

assert not q.empty()  # child must have raised
exc = q.get()
# Expect RuntimeError about re-initialization in forked subprocess
```

This pattern verifies that the backend detects and **rejects** re-initialization after `fork`, expecting a `RuntimeError` when a forked child attempts to call `torch.openreg.init()` after the parent has already initialized.

### Autocast testing

`test_autocast.py` tests AMP integration. Use `torch.amp.autocast` (the current API) rather than the legacy `torch.autocast`:

```python
with torch.amp.autocast(device_type="openreg", dtype=torch.float16):
    result = torch.mm(a, b)
self.assertEqual(result.dtype, torch.float16)
```

### Profiler testing

`test_profiler.py` uses the autograd profiler and inspects the resulting events:

```python
from torch.autograd.profiler import profile as autograd_profile
from torch.profiler import record_function

with autograd_profile(use_device="openreg") as prof:
    with record_function("openreg_custom_operation"):
        x = torch.randn(10, 10, device="openreg")
        x @ y

events = prof.function_events
event_names = [e.name for e in events]
self.assertTrue(any("openreg_custom_operation" in name for name in event_names))
```

For chrome trace export validation:

```python
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    trace_file = f.name

try:
    prof.export_chrome_trace(trace_file)
    with open(trace_file) as f:
        trace_data = json.load(f)
    if isinstance(trace_data, dict):
        self.assertIn("traceEvents", trace_data)
finally:
    os.remove(trace_file)
```

## Adding new tests

### Step 1: Choose the right file

Find the existing `test_*.py` that matches the feature area. If no file is appropriate, create a new `test_<area>.py` following the same structure.

### Step 2: Choose the right base class

| Base class | When to use |
|------------|-------------|
| `TestCase` (from `torch.testing._internal.common_utils`) | Default for most tests |
| `NNTestCase` (from `torch.testing._internal.common_nn`) | Tests involving `nn.Module`, SDPA, or neural network layers |

### Step 3: Write the test

```python
# Owner(s): ["module: PrivateUse1"]

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestNewFeature(TestCase):
    def test_my_feature(self):
        x = torch.randn(4, device="openreg")
        y = torch.randn(4, device="openreg")
        z = x + y
        self.assertEqual(z.device.type, "openreg")
        self.assertEqual(z.cpu(), x.cpu() + y.cpu())


if __name__ == "__main__":
    run_tests()
```

Key conventions:
- The first line must be `# Owner(s): ["module: PrivateUse1"]`.
- Import `run_tests` and `TestCase` from `torch.testing._internal.common_utils`.
- End the file with `if __name__ == "__main__": run_tests()`.
- Use `self.assertEqual` for tensor comparisons (it handles tolerance).
- Use the hard-coded device string `"openreg"` — these are OpenReg-specific tests, not device-generic templates.

### Step 4: Add skip decorators if needed

If the test depends on features not yet implemented:

```python
@unittest.skip("requires feature X which is not yet implemented")
def test_future_feature(self):
    ...
```

If the test is incompatible with TorchDynamo:

```python
@skipIfTorchDynamo("reason for incompatibility")
def test_dynamo_incompatible(self):
    ...
```

### Step 5: Run the test

```bash
python test_<area>.py TestClassName.test_method_name
```

Or run the entire file:

```bash
python test_<area>.py
```

### Contributing device-generic tests

If the test is not OpenReg-specific and should run on **all** device types (CPU, CUDA, OpenReg, etc.), add it to the appropriate mainline test file (e.g. `test/test_torch.py` or `test/test_ops.py`) using the template pattern:

```python
class TestSomething(TestCase):
    def test_feature(self, device):
        x = torch.randn(3, device=device)
        ...

instantiate_device_type_tests(TestSomething, globals())
```

The test will automatically run on `openreg` when the backend is installed, via `PrivateUse1TestBase`.

## Interpreting failures

When a test fails, the failure's location tells you which integration area is affected (the test file name maps to a capability — see the table in [Overview](#test-files)). The error message then narrows the problem further.

### No kernel registered

```
RuntimeError: Could not run 'aten::some_op' with arguments from the 'PrivateUse1' backend.
```

**What it means:** An operator was dispatched to the `openreg` device but no kernel is registered for it.

**How to fix:**
1. Decide whether to implement the kernel natively or fall back to CPU.
2. For a native kernel, add it in `csrc/aten/native/` and register it via `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)` in `csrc/aten/OpenRegExtra.cpp` (or `OpenRegMinimal.cpp` for core ops). See the existing `abs_stub` for the STUB form or `empty.memory_format` for the direct form.
3. For a CPU fallback, register a per-operator fallback like `sub.Tensor` in `OpenRegMinimal.cpp`, or rely on the global `wrapper_cpu_fallback` if the op only needs CPU semantics.

### Device mismatch

```
AssertionError: torch.device('openreg:0') != torch.device('cpu')
```

**What it means:** A tensor ended up on the wrong device. This typically happens when `copy_` between devices is not handled, or a factory op returns a CPU tensor instead of an `openreg` tensor.

**How to fix:**
1. Check that `copy_` handles the failing device direction (CPU → openreg, openreg → CPU, openreg → openreg). The implementation lives in `csrc/aten/native/Minimal.cpp`.
2. For factory ops (`empty`, `zeros`, etc.), verify the device argument is propagated through to the allocator.

### Numerical mismatch

```
AssertionError: tensor values differ (max abs diff: 1.5e-3)
```

**What it means:** The operator implementation produces different results from the CPU reference.

**How to fix:**
1. Isolate the op: create a small tensor on CPU, copy to `openreg`, run the op, copy back, and compare:
   ```python
   x_cpu = torch.randn(4)
   y_cpu = torch.abs(x_cpu)
   y_openreg = torch.abs(x_cpu.to("openreg")).cpu()
   torch.testing.assert_close(y_cpu, y_openreg)
   ```
2. If the op falls back to CPU (via `wrapper_cpu_fallback`), the numerics should match exactly. A mismatch here usually means data was corrupted during the device-to-host or host-to-device copy — check the allocator and `copy_` implementation.
3. If the op has a native kernel, compare its implementation against the CPU reference in the PyTorch source tree (`aten/src/ATen/native/`).

### Skipped tests

```
SKIP: "openreg backend does not implement per-device RNG yet"
```

**What it means:** The test was intentionally disabled because the underlying feature is not yet implemented. Skipped tests are **not failures** — they document known gaps.

**How to use skips for readiness evaluation:** Collect all skips to see what remains unimplemented:

```bash
grep -rn "@unittest.skip\|self.skipTest" test_*.py
```

Each skip reason explains what is missing. Once you implement the feature, remove the skip and verify the test passes.

### Dynamo-related skips

Tests decorated with `@skipIfTorchDynamo` are skipped only when the test suite is run under TorchDynamo. If you see these skips in CI but the test passes when run directly, the issue is in Dynamo's tracing of that op for the `openreg` device, not in your backend implementation.

### Runtime environment failures

Tests that call `self.skipTest(...)` at runtime (e.g. checking device count) skip because the test environment doesn't meet prerequisites. For example, multi-device profiler tests require at least 2 `openreg` devices. These are not backend bugs — they reflect the test machine configuration.

## Evaluating platform readiness

Maintainers can use the test suite as a structured readiness assessment for a backend. The test files are organized by integration capability, and each file maps to a tier of functionality:

### Capability tiers

| Tier | Files | What it proves |
|------|-------|----------------|
| **1 — Foundational** | `test_device.py`, `test_ops.py`, `test_memory.py` | The backend can register a device, create tensors, move data between CPU and device, and run basic ops. Without these, nothing else works. |
| **2 — Core infrastructure** | `test_storage.py`, `test_streams.py`, `test_event.py`, `test_rng.py` | Serialization, async execution, synchronization, and random number generation work. Required for training loops. |
| **3 — Ecosystem integration** | `test_autograd.py`, `test_autocast.py`, `test_profiler.py`, `test_utils.py` | Autograd, AMP, profiling, and interop (DLPack) are functional. Required for production use. |
| **4 — Completeness** | `test_misc.py` | Legacy APIs, rename registration, and backward-compatibility shims. |

A backend is considered **minimally viable** when all Tier 1 tests pass (no failures, skips are acceptable for unimplemented-but-documented gaps). **Production ready** means Tiers 1-3 pass.

### Generating a readiness summary

Run the full suite and count results per file:

```bash
for f in test_*.py; do
    echo "=== $(basename $f) ==="
    python "$f" 2>&1 | tail -2
done
```

This gives a two-line summary per file: the `Ran N tests in X.XXXs` line followed by the result (e.g. `OK`, `OK (skipped=N)`, or `FAILED (failures=N, errors=M)`).

To see all known gaps at a glance:

```bash
grep -rn "@unittest.skip\|self.skipTest" test_*.py
```

Each line shows a skip reason that explains what feature is missing and can serve as a work item for closing the gap.
