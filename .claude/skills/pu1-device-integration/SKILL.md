---
name: pu1-device-integration
description: Guide third-party hardware vendors to integrate a new accelerator device into PyTorch via the PrivateUse1 (PU1) out-of-tree backend mechanism. Use when adding a new hardware device to PyTorch, implementing a PU1 backend, or working with out-of-tree device extensions.
---

# PrivateUse1 (PU1) Third-Party Device Integration Guide

This skill walks through integrating a new accelerator device into PyTorch using the `PrivateUse1` dispatch key — PyTorch's official out-of-tree backend slot. The canonical reference implementation is `test/cpp_extensions/open_registration_extension/torch_openreg/`.

## Architecture Overview

A PU1 backend consists of:
1. **C++ device runtime library** — device/memory/stream management, analogous to the CUDA Runtime
2. **PU1 hooks** — C++ class that implements `PrivateUse1HooksInterface`, connecting your runtime to PyTorch internals
3. **Operator registrations** — `TORCH_LIBRARY_IMPL` blocks that dispatch `aten` ops to your kernels (with CPU fallback for unimplemented ops)
4. **Python module** — device-level Python API (`device_count`, `set_device`, etc.) and registration calls that wire everything together
5. **Package** — CMake + setuptools build, with optional autoload via entry_points

## Step 1: Implement the C++ Device Runtime

Your runtime library must expose at minimum:

| Category | Required APIs |
|---|---|
| Device management | `GetDeviceCount`, `SetDevice`, `GetDevice`, `DeviceSynchronize` |
| Memory management | `Malloc`, `Free`, `MallocHost` (pinned), `FreeHost`, `Memcpy`, `MemcpyAsync`, `PointerGetAttributes` |
| Streams | `StreamCreate`, `StreamDestroy`, `StreamSynchronize`, `StreamWaitEvent` |
| Events | `EventCreate`, `EventDestroy`, `EventRecord`, `EventSynchronize`, `EventElapsedTime` |

Build this as a shared library (e.g., `libmydevice.so`) via CMake. See `third_party/openreg/` for a reference implementation that simulates CUDA behavior on CPU using `mmap`/`mprotect` and pthreads.

## Step 2: Implement `PrivateUse1HooksInterface`

Create a class that inherits from `at::PrivateUse1HooksInterface` and override all relevant methods. Then register it at static-init time.

**File:** `csrc/runtime/MyDeviceHooks.cpp`

```cpp
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include "mydevice_runtime.h"  // your device runtime header

struct MyDeviceHooksInterface : public at::PrivateUse1HooksInterface {
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return mydevice_get_device_count() > 0; }

  const at::Generator& getDefaultGenerator(c10::DeviceIndex idx) const override {
    return at::GetGeneratorForPrivateuse1(idx);
  }

  at::Device getDeviceFromPtr(void* data) const override {
    // Use your runtime's pointer attribute query
    mydevice_ptr_attrs attrs;
    mydevice_pointer_get_attributes(&attrs, data);
    return at::Device(at::kPrivateUse1, attrs.device);
  }

  bool isPinnedPtr(const void* data) const override {
    mydevice_ptr_attrs attrs;
    mydevice_pointer_get_attributes(&attrs, const_cast<void*>(data));
    return attrs.is_pinned;
  }

  at::Allocator* getPinnedMemoryAllocator() const override {
    return mydevice_pinned_allocator();
  }

  bool hasPrimaryContext(c10::DeviceIndex idx) const override {
    return mydevice_has_context(idx);
  }
};

// Static registration — runs before main()
static bool register_hook [[maybe_unused]] = []() {
  at::RegisterPrivateUse1HooksInterface(new MyDeviceHooksInterface());
  return true;
}();
```

## Step 3: Register the Minimum Required Operators

These aten ops **must** be implemented — they underpin tensor allocation and memory layout:

**File:** `csrc/aten/MyDeviceOps.cpp`

```cpp
#include <torch/library.h>
#include <ATen/native/CPUFallback.h>

namespace at::mydevice {

// Required: tensor allocation
at::Tensor empty_memory_format(
    c10::IntArrayRef size,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    std::optional<c10::MemoryFormat> memory_format) {
  // Allocate memory on your device, return a Tensor backed by it
  auto storage = /* mydevice_malloc(nbytes) */;
  return at::detail::make_tensor<at::TensorImpl>(
      std::move(storage), at::DispatchKey::PrivateUse1, caffe2::TypeMeta::Make<float>());
}

at::Tensor empty_strided(
    c10::IntArrayRef size, c10::IntArrayRef stride,
    std::optional<c10::ScalarType> dtype, std::optional<c10::Layout> layout,
    std::optional<c10::Device> device, std::optional<bool> pin_memory) {
  // Similar to empty_memory_format but with explicit strides
}

// Also implement: as_strided, resize_, _reshape_alias

// CPU fallback — delegates unimplemented ops to CPU via data copy
static void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  at::native::cpu_fallback(op, stack);
}

// Register the minimum required ops
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", empty_memory_format);
  m.impl("empty_strided",       empty_strided);
  m.impl("as_strided",          as_strided);
  m.impl("resize_",             resize_);
  m.impl("_reshape_alias",      _reshape_alias);
}

// Global CPU fallback for everything else
TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

} // namespace at::mydevice
```

To register a native kernel for a specific op (instead of falling back to CPU):

```cpp
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", mydevice_add_kernel);
  m.impl("mm",         mydevice_matmul_kernel);
}
```

To force a single op to always use CPU fallback (overriding a global kernel):

```cpp
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("some_op", torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}
```

## Step 4: Python Module Setup

Create `torch_mydevice/mydevice/__init__.py` that mirrors `torch.cuda`'s API surface:

```python
import torch
import torch_mydevice._C  # your compiled C extension

class device:
    def __init__(self, device):
        self.idx = torch.accelerator._get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = torch_mydevice._C._exchangeDevice(self.idx)

    def __exit__(self, *args):
        torch_mydevice._C._set_device(self.prev_idx)
        return False

def is_available() -> bool:
    return torch_mydevice._C._get_device_count() > 0

def device_count() -> int:
    return torch_mydevice._C._get_device_count()

def current_device() -> int:
    return torch_mydevice._C._get_device()

def set_device(device: int) -> None:
    if device >= 0:
        torch_mydevice._C._set_device(device)

def synchronize(device=None) -> None:
    torch_mydevice._C._synchronize(device)
```

Then in the top-level `torch_mydevice/__init__.py`, wire everything into PyTorch:

```python
import torch
import torch_mydevice._C
import torch_mydevice.mydevice

# 1. Rename the PU1 slot to your device name (do this exactly once, at import)
torch.utils.rename_privateuse1_backend("mydevice")

# 2. Register the Python device module (makes torch.mydevice work)
torch._register_device_module("mydevice", torch_mydevice.mydevice)

# 3. Auto-generate Tensor methods (.mydevice(), .is_mydevice, etc.)
torch.utils.generate_methods_for_privateuse1_backend(for_storage=True)
```

After this, users can do:
```python
import torch_mydevice
t = torch.tensor([1.0]).mydevice()  # moves tensor to your device
t2 = torch.empty(3, device="mydevice")
```

## Step 5: Implement Python-Exposed C++ Bindings

Use pybind11 (via `torch/extension.h`) to expose device management functions. The minimum set needed by the Python module above:

```cpp
// torch_mydevice/csrc/Module.cpp
#include <torch/extension.h>

PYBIND11_MODULE(_C, m) {
  m.def("_get_device_count", []() { return mydevice_get_device_count(); });
  m.def("_get_device",       []() { return mydevice_get_device(); });
  m.def("_set_device",       [](int idx) { mydevice_set_device(idx); return idx; });
  m.def("_exchangeDevice",   [](int idx) {
    int prev = mydevice_get_device();
    mydevice_set_device(idx);
    return prev;
  });
  m.def("_synchronize",      [](std::optional<int> device) {
    mydevice_synchronize();
  });
  m.def("_init",             []() { mydevice_init(); });
}
```

## Step 6: CMake Build Setup

Structure your `CMakeLists.txt` to:
1. Build `libmydevice.so` (your runtime)
2. Build `libtorch_mydevice_bindings.so` (PyTorch extension linking against PyTorch headers)
3. Generate `_C.so` (Python stub via `stub.c` that `dlopen`s the bindings library)

Key CMake patterns:

```cmake
find_package(Torch REQUIRED HINTS ${PYTORCH_INSTALL_DIR})

# Your runtime library
add_library(mydevice SHARED csrc/runtime/device.cpp csrc/runtime/memory.cpp)
target_include_directories(mydevice PUBLIC include/)

# PyTorch bindings (includes hooks, operator registrations, pybind module)
add_library(torch_mydevice_bindings SHARED
  csrc/runtime/MyDeviceHooks.cpp
  csrc/aten/MyDeviceOps.cpp
  torch_mydevice/csrc/Module.cpp
)
target_link_libraries(torch_mydevice_bindings
  PUBLIC ${TORCH_LIBRARIES} mydevice
)
target_compile_options(torch_mydevice_bindings
  PUBLIC ${TORCH_CXX_FLAGS}
)
```

The `stub.c` pattern (used by OpenReg) lazy-loads the bindings library via `dlopen` to avoid circular import issues at Python import time.

## Step 7: Package and Autoload

In `setup.py` (or `pyproject.toml`), register an entry point so PyTorch can auto-import your backend:

```python
setup(
    name="torch_mydevice",
    packages=find_packages(),
    ext_modules=[
        Extension(
            name="torch_mydevice._C",
            sources=["torch_mydevice/csrc/stub.c"],
            language="c",
            libraries=["torch_mydevice_bindings"],
            library_dirs=[...],
        )
    ],
    entry_points={
        "torch.backends": [
            "torch_mydevice = torch_mydevice:_autoload",
        ],
    },
)
```

In `torch_mydevice/__init__.py`:
```python
def _autoload():
    # Called by PyTorch's backend autoloader — no-op is fine
    pass
```

Set `TORCH_DEVICE_BACKEND_AUTOLOAD=1` (or rely on the default) to enable automatic import of all installed `torch.backends` entry points when `torch` is first imported.

## Step 8: Testing

Use `instantiate_device_type_tests` to run PyTorch's existing op tests against your device:

```python
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.common_device_type import instantiate_device_type_tests

class TestMyDevice(TestCase):
    def test_tensor_creation(self, device):
        t = torch.empty(3, 4, device=device)
        self.assertEqual(t.device.type, "mydevice")
        self.assertEqual(t.shape, (3, 4))

    def test_add_fallback(self, device):
        a = torch.tensor([1.0, 2.0], device=device)
        b = torch.tensor([3.0, 4.0], device=device)
        c = a + b
        self.assertEqual(c.cpu(), torch.tensor([4.0, 6.0]))

instantiate_device_type_tests(TestMyDevice, globals(), only_for="mydevice")

if __name__ == "__main__":
    run_tests()
```

To validate the full op suite against your device, run existing PyTorch tests with `--device mydevice`.

## Optional Integrations

### AMP (Automatic Mixed Precision)

Register supported dtypes for autocast:

```python
# torch_mydevice/mydevice/amp/__init__.py
def get_amp_supported_dtype():
    return [torch.float16, torch.bfloat16]
```

```cpp
// In your module init or hooks
torch::autocast::register_autocast_cache_and_dtypes(
    c10::DeviceType::PrivateUse1, {at::kHalf, at::kBFloat16});
```

### Random Number Generation

Register a custom RNG generator:

```python
from torch.testing._internal.common_utils import run_tests
torch._register_device_module("mydevice", module_with_rng)
```

See `torch_openreg/torch_openreg/openreg/random.py` for the Python side and `at::GeneratorForPrivateuse1` for the C++ side.

### Distributed (Process Groups)

```python
torch.distributed.Backend.register_backend(
    "mycomm",
    my_create_process_group_fn,
    devices=["mydevice"],
)
```

### Profiler Integration

Implement the profiler observer interface to get profiler traces. See `torch/csrc/profiler/standalone/privateuse1_observer.h` for the hooks to implement.

## Common Pitfalls

- **`rename_privateuse1_backend` must be called before any tensor is created on your device.** Call it in the top-level `__init__.py`, early.
- **`TORCH_LIBRARY_IMPL` blocks with the same key in the same TU will conflict.** Split into separate translation units or use a single registration block per key.
- **CPU fallback copies data to CPU, runs the op, copies back.** This is correct for testing but will be slow in production — implement native kernels for hot paths.
- **The stub/dlopen pattern avoids circular imports** when `torch` itself tries to autoload your backend during `import torch`. Set `TORCH_DEVICE_BACKEND_AUTOLOAD=0` in `setup.py`'s `get_pytorch_dir()` to avoid this during build.
- **`generate_methods_for_privateuse1_backend(for_storage=True)`** must be called to enable `Tensor.mydevice()`, `Tensor.is_mydevice`, and `Storage.mydevice()`. Call once at import time.

## Key Reference Files

| Path | Purpose |
|---|---|
| `test/cpp_extensions/open_registration_extension/torch_openreg/` | Complete reference implementation |
| `aten/src/ATen/detail/PrivateUse1HooksInterface.h` | Hooks interface to implement |
| `aten/src/ATen/core/GeneratorForPrivateuseone.h` | RNG generator helpers |
| `torch/csrc/profiler/standalone/privateuse1_profiler.h` | Profiler integration hooks |
| `torch/testing/_internal/common_device_type.py` | `instantiate_device_type_tests` for device-generic tests |
