# Memory Allocator

## Background

Memory allocators manage device memory allocation and deallocation for tensors. Accelerator backends provide custom allocators by implementing `c10::DeviceAllocator` and registering them via `REGISTER_ALLOCATOR`. PyTorch's runtime calls these allocators in contexts where the Python GIL or internal locks may already be held, which imposes constraints on what the allocator can safely do.

These constraints are documented for CUDA in [`c10/cuda/CUDACachingAllocator.h`][CUDACachingAllocator.h] but apply equally to any `PrivateUse1` backend. This page documents the constraints and the patterns backends use to work within them.

## Design

### Allocator Interface

Accelerator vendors implement `c10::DeviceAllocator` (which extends `c10::Allocator`) and register it for `PrivateUse1`:

| Method          | Description                                                                 |
| :---            | :---                                                                        |
| `allocate()`    | Allocate device memory and return a `DataPtr` with an associated deleter    |
| `raw_deleter()` | Return the deleter function pointer for raw allocations                     |
| `copy_data()`   | Copy data between two allocations owned by this allocator                   |

Additional methods from `c10::DeviceAllocator` support caching, stream recording, and memory statistics. See the [OpenReg allocator][OpenRegDeviceAllocator.h] for a complete implementation.

### GIL and Lock Constraints

The GIL may already be held by the thread that calls into the allocator. This happens during:

- Tensor construction from Python (`torch.empty(..., device='my_device')`)
- Garbage collection of tensors (the destructor runs during `tp_dealloc`)
- `torch.cuda.empty_cache()` equivalents called from Python

Because the GIL is already held, `allocate()`, `raw_deleter()`, and any allocator callbacks (e.g. `AllocatorTraceTracker`) **must not** attempt to acquire it. Doing so deadlocks the process.

```{note}
From [`c10/cuda/CUDACachingAllocator.h`][CUDACachingAllocator.h]:

> *"Python's GIL may be held when calling the allocator so it is unsafe to try to acquire the GIL in this callback."*
```

Concretely, this means:

- **No Python API calls** -- do not call `PyObject_*`, `Py_INCREF`/`Py_DECREF`, or any function that internally acquires the GIL.
- **No `pybind11::gil_scoped_acquire`** -- this is the most common way this constraint is accidentally violated.
- **No Python-side callbacks** -- do not invoke registered Python callables from within the allocator path.

When the allocator holds its own internal lock (e.g. a per-device mutex), any other locks acquired while that lock is held must obey a strict order. Since the GIL can be held *before* the allocator lock, the allocator lock must always be ordered *after* the GIL:

```
GIL -> allocator lock    OK  (this is the order PyTorch uses)
allocator lock -> GIL    DEADLOCK
```

Use `std::recursive_mutex` if the allocator's internal helpers may re-enter through the same lock.

### Constraints Summary

| Allocator method             | GIL state   | Safe operations                                                | Must avoid                                    |
| :---                         | :---        | :---                                                           | :---                                          |
| `allocate()`                 | May be held | Atomic counters, C++ allocations, lock internal mutex          | Acquiring GIL, Python API calls               |
| `raw_deleter()` / destructor | May be held | Queue pointer for deferred free, update C-side stats           | Acquiring GIL, recursive device runtime calls |
| `AllocatorTraceTracker`      | May be held | Logging, C-side bookkeeping                                    | Acquiring GIL, locks ordered before the GIL   |
| `getDeviceFromPtr()`         | May be held | C-side map lookup, thread-local fallback                       | Acquiring GIL, Python API calls               |
| First tensor use (Python)    | Held        | Associate metadata, allocate real device memory                | (no restrictions)                             |
| Periodic cleanup (Python)    | Held        | Drain deferred frees, clean up Python metadata                 | (no restrictions)                             |

### Patterns for Backends with Python-Side State

Some backends maintain Python-side metadata per allocation (e.g. a Python dict mapping pointers to device-specific objects). Since `allocate()` cannot touch Python, these backends use the following patterns:

**Deferred registration.** In `allocate()`, assign a unique C integer ID as the "data pointer" using an `std::atomic<uint64_t>` counter. On first tensor use (Python side, GIL held), lazily associate the ID with real device memory and Python-side metadata via a custom `__torch_function__` override or a wrapper around the first kernel dispatch. In the destructor, remove the C-side ID mapping and queue the Python-side cleanup for later.

**Thread-local device tracking.** The accelerator hook `getDeviceFromPtr(void*)` must also be GIL-free. Backends that cannot determine the device from a raw pointer alone (because the "pointer" is a synthetic ID) can maintain a thread-local variable tracking the current device index. Since `set_device()` is always called before allocations on a specific device, the thread-local value is reliable as a fallback.

**Destructor batching.** When Python's garbage collector runs, it can trigger a cascade of tensor destructors. If each deleter does nontrivial work (acquiring locks, calling into device runtimes, performing IPC cleanup), the recursive GC calls can overflow the C stack. Instead of freeing immediately, queue pointers for deferred cleanup and drain the queue at a safe point -- for example, at the start of the next allocation, or in a periodic cleanup called from Python.

## Implementation

For illustration, OpenReg (Open Registration) is a PyTorch integration example that fills the gap for out-of-tree accelerator backend integration. Its allocator implementation ([`OpenRegDeviceAllocator.h/cpp`][OpenRegDeviceAllocator.h]) demonstrates how to build a GIL-safe allocator.

### Allocator Class

The allocator class extends `c10::DeviceAllocator` and manages per-device allocators with a global pointer-to-device mapping:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegDeviceAllocator.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG DEVICE ALLOCATOR CLASS
    :end-before: LITERALINCLUDE END: OPENREG DEVICE ALLOCATOR CLASS
    :linenos:
```

Key design choices:

- `std::recursive_mutex` prevents deadlocks when internal helpers re-enter
- `ska::flat_hash_map` tracks pointer-to-device mappings entirely in C++ (no Python)
- Per-device `DeviceMemoryAllocator` instances isolate per-device statistics

### Allocation

The `allocate()` method queries the current device, delegates to the per-device allocator, and tracks the pointer-to-device mapping. No Python calls are made:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegDeviceAllocator.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG ALLOCATE
    :end-before: LITERALINCLUDE END: OPENREG ALLOCATE
    :linenos:
```

### Deleter

The deleter is a plain C function that delegates to `freeMemory()`. It is GIL-free and safe to call from garbage collection:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegDeviceAllocator.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG DELETER
    :end-before: LITERALINCLUDE END: OPENREG DELETER
    :linenos:
```

### Registration

Register the allocator for `PrivateUse1` using `REGISTER_ALLOCATOR`:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegDeviceAllocator.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG REGISTER ALLOCATOR
    :end-before: LITERALINCLUDE END: OPENREG REGISTER ALLOCATOR
    :linenos:
```

```{seealso}
- {ref}`CUDA custom allocator documentation <cuda-memory-custom-allocator>` for the CUDA-specific pluggable allocator API
- [Accelerator hooks](hooks.md) for the full hooks interface including `getPinnedMemoryAllocator()`
```

[CUDACachingAllocator.h]: https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.h "CUDACachingAllocator.h"
[OpenRegDeviceAllocator.h]: https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegDeviceAllocator.h "OpenRegDeviceAllocator.h"
