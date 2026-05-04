# Distributed Training Integration

## Background

Distributed training allows accelerators to scale workloads across multiple devices and nodes by coordinating collective communication (e.g., allreduce, broadcast, allgather) through a [ProcessGroup](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroup.hpp) backend. PyTorch ships built-in backends such as [NCCL](https://developer.nvidia.com/nccl) for CUDA and [Gloo](https://github.com/facebookincubator/gloo) for CPU, but the framework exposes a registration mechanism that allows out-of-tree accelerator vendors to plug in their own collective communication library without modifying upstream code.

The integration surface can be broken down into three layers:

1. **C++ Backend implementation** — A subclass of [`c10d::Backend`][Backend.hpp] that implements the collective, point-to-point, and synchronization operations.
2. **Python bindings** — Expose the C++ backend and a factory function to Python via [pybind11](https://pybind11.readthedocs.io/).
3. **Backend registration** — Register the backend with [`torch.distributed.Backend.register_backend()`][distributed_c10d.py] so that `init_process_group` can discover and instantiate it.

```{note}
The `torch_openreg` reference implementation ships a minimal backend called **OCCL** (OpenReg Collective Communications Library) that demonstrates the full integration. All code examples in this chapter reference the OCCL implementation.
```

## Design

### Registration API

The primary entry point for OOT backend registration is [`Backend.register_backend()`][distributed_c10d.py]:

| Parameter       | Type                    | Description                                                                                                        |
| :---            | :---                    | :---                                                                                                               |
| `name`          | `str`                   | Backend name, e.g. `"occl"`. Must match the value passed to `init_process_group(backend=...)`.                     |
| `func`          | `Callable`              | Factory function that creates a backend instance (see signature below).                                            |
| `extended_api`  | `bool`                  | If `True`, the factory receives a `_DistributedBackendOptions` object instead of individual arguments.             |
| `devices`       | `str \| list[str] \| None` | Device types supported by this backend, e.g. `["openreg"]`. Populates the device-to-backend mapping.           |

When `devices` is specified, the backend is automatically associated with those device types. This means `init_process_group()` can resolve the correct backend when the user passes a `device_id` argument without explicitly naming a backend.

### Factory Function Signature

The factory function receives different arguments depending on `extended_api`:

| Mode              | Signature                                                                         |
| :---              | :---                                                                              |
| Standard (default) | `func(store: Store, rank: int, world_size: int, timeout: timedelta) -> Backend`  |
| Extended API       | `func(dist_backend_opts: _DistributedBackendOptions, backend_options) -> Backend` |

The standard mode is sufficient for most backends. The extended API provides additional context such as `group_id` and `global_ranks_in_group`.

### Collective Operations

The [`c10d::Backend`][Backend.hpp] base class defines virtual methods for all collective, point-to-point, and synchronization operations. The table below lists the operations an OOT backend must implement:

| Category        | Operations                                                                                     |
| :---            | :---                                                                                           |
| **Collective**  | `broadcast`, `allreduce`, `allreduce_coalesced`, `reduce`, `allgather`, `_allgather_base`, `allgather_coalesced`, `allgather_into_tensor_coalesced`, `gather`, `scatter`, `reduce_scatter`, `_reduce_scatter_base`, `reduce_scatter_tensor_coalesced`, `alltoall_base` |
| **Point-to-Point** | `send`, `recv`, `recvAnysource`                                                             |
| **Synchronization** | `barrier`                                                                                   |

Each operation returns a [`c10::intrusive_ptr<Work>`][Work.hpp] that represents the asynchronous operation. For backends with synchronous operations, the `Work` object can be immediately completed.

### Optional Capabilities

Backends can advertise optional capabilities by overriding the following methods:

| Method                    | Default  | Description                                           |
| :---                      | :---     | :---                                                  |
| `supportsSplitting()`     | `false`  | Process group splitting support                       |
| `supportsCoalescing()`    | `false`  | Coalesced collective operations                       |

## Implementation

This section walks through the complete integration, using the OCCL reference implementation as an example. The implementation follows three steps:

1. Implement the C++ backend
2. Create Python bindings
3. Register the backend in Python

### Step 1: Implement the C++ Backend

Create a class that inherits from `c10d::Backend` and implements the required collective operations. The backend must also define:

- A **`Work` subclass** that tracks asynchronous operation state
- An **`Options` subclass** (inheriting from `Backend::Options`) for backend-specific configuration
- A **factory function** to construct the backend

#### Work Object

The `Work` subclass manages the lifecycle of an asynchronous collective operation. For a minimal (synchronous) implementation, the work can be completed immediately in its constructor:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/distributed/c10d/ProcessGroupOCCL.hpp
    :language: c++
    :start-after: class DummyWork : public Work {
    :end-at: };
    :dedent: 3
    :linenos:
    :caption: ProcessGroupOCCL::DummyWork declaration (ProcessGroupOCCL.hpp)
```

For production backends, `Work` typically wraps an asynchronous handle from the vendor's communication library (e.g., a stream event or request handle), and `wait()` blocks until the operation completes on the device.

#### Backend Class

The backend class inherits from `c10d::Backend` and overrides all collective operations listed in the [Design section](#collective-operations). Each method should validate that input tensors reside on the expected device type (e.g., `PrivateUse1`) and then dispatch to the vendor's communication library. Key implementation details:

- **`getBackendName()`** must return the same string used during Python registration (e.g., `"occl"`).
- **Input validation** — Each collective should verify tensor device types. The OCCL reference uses `CHECK_TENSOR` and `CHECK_TENSOR_LIST` macros for this.
- **Return value** — All collectives return a `c10::intrusive_ptr<Work>`.

See [`ProcessGroupOCCL.hpp`][ProcessGroupOCCL.hpp] and [`ProcessGroupOCCL.cpp`][ProcessGroupOCCL.cpp] for the full reference implementation.

#### Factory Function

The factory function is the bridge between Python registration and C++ instantiation:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/distributed/c10d/ProcessGroupOCCL.cpp
    :language: c++
    :lines: 359-365
    :linenos:
    :lineno-start: 359
    :caption: createProcessGroupOCCL factory (ProcessGroupOCCL.cpp)
```

The factory receives a `Store` (for cross-rank coordination during initialization), `rank`, `world_size`, and `timeout`. Production backends typically use the store for bootstrapping (e.g., exchanging connection info).

### Step 2: Python Bindings

Expose the backend class and factory function to Python using pybind11. This is done in the extension module's initialization:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp
    :language: c++
    :lines: 127-140
    :linenos:
    :lineno-start: 127
    :caption: Python bindings for OCCL (Module.cpp)
```

Important considerations:

- The `py::class_` template must list `c10d::Backend` as a base class and use `c10::intrusive_ptr` as the holder, so that PyTorch recognizes the backend in its internal registry.
- The factory function signature must match the expected pattern: `(Store, rank, world_size, timeout) -> Backend`.
- Guard the bindings with `#if USE_DISTRIBUTED` to handle builds where distributed is disabled.

### Step 3: Register the Backend in Python

In the extension package's `__init__.py`, register the backend with `torch.distributed`:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/__init__.py
    :language: python
    :lines: 20-26
    :linenos:
    :caption: Backend registration (torch_openreg/__init__.py)
```

The call to `Backend.register_backend()` does the following:

1. Adds `"occl"` to `Backend.backend_list`, making it a recognized backend name.
2. Maps `"openreg"` device type to the `"occl"` backend in `Backend.default_device_backend_map`.
3. Stores the factory function so that `init_process_group()` can call it when `backend="occl"` is specified.

## Usage

After registration, the backend integrates seamlessly with `torch.distributed`:

```python
import torch
import torch.distributed as dist

# Import triggers autoload, which registers the "occl" backend
import torch_openreg

# Initialize process group — OCCL is auto-selected for openreg devices
dist.init_process_group(
    backend="occl",
    init_method="env://",
    world_size=2,
    rank=0,
)

# Use standard distributed APIs
tensor = torch.randn(4, device="openreg")
dist.all_reduce(tensor)

dist.destroy_process_group()
```

Alternatively, the backend name can be omitted if a `device_id` is provided — PyTorch resolves the backend from the device-to-backend mapping:

```python
dist.init_process_group(
    device_id=torch.device("openreg:0"),
    init_method="env://",
    world_size=2,
    rank=0,
)
```

### Multi-device Backend Strings

PyTorch supports specifying different backends for different device types in a single process group using the `"device:backend"` format:

```python
dist.init_process_group(
    backend="cpu:gloo,openreg:occl",
    init_method="env://",
    world_size=2,
    rank=0,
)
```

## Testing

Key testing considerations:

- Verify that the backend appears in `dist.Backend.backend_list` after import.
- Confirm that `init_process_group` / `destroy_process_group` succeeds.
- Test that collective operations accept tensors on the registered device and return completed `Work` objects.
- Use `MultiProcessTestCase` from `torch.testing._internal.common_distributed` for multi-process test execution.

See the [OCCL test suite](https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/tests/test_distributed.py) for a reference example.

[Backend.hpp]: https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/Backend.hpp "Backend.hpp"
[Work.hpp]: https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/Work.hpp "Work.hpp"
[distributed_c10d.py]: https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py "distributed_c10d.py"
[ProcessGroupOCCL.hpp]: https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/distributed/c10d/ProcessGroupOCCL.hpp "ProcessGroupOCCL.hpp"
[ProcessGroupOCCL.cpp]: https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/distributed/c10d/ProcessGroupOCCL.cpp "ProcessGroupOCCL.cpp"
[Module.cpp]: https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp "Module.cpp"
