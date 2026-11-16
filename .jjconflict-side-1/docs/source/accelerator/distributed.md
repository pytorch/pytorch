# Distributed Training Integration

## Background

Distributed training allows accelerators to scale workloads across multiple devices and nodes by coordinating collective communication (e.g., allreduce, broadcast, allgather) through a [ProcessGroup](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroup.hpp) backend. PyTorch ships built-in backends such as [NCCL](https://developer.nvidia.com/nccl) for CUDA and [Gloo](https://github.com/facebookincubator/gloo) for CPU, but the framework exposes a registration mechanism that allows out-of-tree accelerator vendors to plug in their own collective communication library without modifying upstream code.

The integration surface can be broken down into three layers:

1. **C++ Backend implementation** – A subclass of [`c10d::Backend`][Backend.hpp] that implements the collective, point-to-point, and synchronization operations.
2. **Python bindings** – Expose the C++ backend class to Python via [pybind11](https://pybind11.readthedocs.io/).
3. **Backend registration** – Register the backend with [`torch.distributed.Backend.register_backend()`][distributed_c10d.py] so that `init_process_group` can discover and instantiate it.

```{note}
[OpenReg](https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg) (`torch_openreg`) is PyTorch's official reference implementation for out-of-tree accelerator integration. It ships a minimal distributed backend called **OCCL** (OpenReg Collective Communications Library) that demonstrates the full `ProcessGroup` integration. All code examples in this chapter reference the OCCL implementation.
```

## Before You Start

This guide covers **ProcessGroup backend integration** only -- how to register a custom collective communication backend with `torch.distributed`. It does not cover full-stack integration with higher-level APIs such as DDP, FSDP, or other distributed training strategies.

Before following this guide, make sure you have:

- **An importable `torch_xxx` extension package** that registers your device via `PrivateUse1`. See the earlier chapters in this guide for device registration, operators, and runtime hooks.
- **A collective communication library (CCL)** that provides implementations of basic collectives such as `allreduce` and `broadcast` for your device. The CCL can be vendor-provided (e.g., NCCL for NVIDIA, HCCL for Huawei) or a custom implementation.

## Design

This section describes the interfaces and concepts involved in backend registration.

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

### Backend Operations

The [`c10d::Backend`][Backend.hpp] base class defines virtual methods for collective, point-to-point, and synchronization operations. Each operation returns a [`c10::intrusive_ptr<Work>`][Work.hpp] that represents the asynchronous operation. For backends with synchronous operations, the `Work` object can be immediately completed.

#### Minimal Required Operations

To get a working backend that supports basic distributed training, implement the following operations at minimum:

| Category        | Operations                                      |
| :---            | :---                                             |
| **Collective**  | `broadcast`, `allreduce`, `allgather`, `reduce_scatter` |
| **Synchronization** | `barrier`                                   |

These cover the core communication patterns used by DDP and other common distributed workflows.

#### Extended Operations

For broader compatibility with advanced distributed strategies (e.g., FSDP, model parallelism, pipeline parallelism), implement the full set of operations:

| Category        | Operations                                      |
| :---            | :---                                             |
| **Collective**  | `allreduce_coalesced`, `reduce`, `_allgather_base`, `allgather_coalesced`, `allgather_into_tensor_coalesced`, `gather`, `scatter`, `_reduce_scatter_base`, `reduce_scatter_tensor_coalesced`, `alltoall_base` |
| **Point-to-Point** | `send`, `recv`, `recvAnysource`              |

See [`Backend.hpp`][Backend.hpp] for the full list of virtual methods and their signatures.

### Optional Capabilities

Backends can advertise optional capabilities by overriding the following methods:

| Method                    | Default  | Description                                           |
| :---                      | :---     | :---                                                  |
| `supportsSplitting()`     | `false`  | Process group splitting support                       |
| `supportsCoalescing()`    | `false`  | Coalesced collective operations                       |

## Implementation

This section walks through the concrete steps to implement and register a backend, using the OCCL reference implementation as an example. The implementation follows three steps:

1. Implement the C++ backend
2. Create Python bindings
3. Register the backend in Python

### Step 1: Implement the C++ Backend

Create a class that inherits from `c10d::Backend` and implements the required collective operations. The backend must also define:

- A **`Work` subclass** that tracks asynchronous operation state
- An **`Options` subclass** (inheriting from `Backend::Options`) for backend-specific configuration

#### Work Object

The `Work` subclass manages the lifecycle of an asynchronous collective operation. For a minimal (synchronous) implementation, the work can be completed immediately in its constructor:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/distributed/c10d/ProcessGroupOCCL.hpp
    :language: c++
    :start-after: LITERALINCLUDE START: OCCL DUMMYWORK
    :end-before: LITERALINCLUDE END: OCCL DUMMYWORK
    :linenos:
    :caption: ProcessGroupOCCL::DummyWork declaration (ProcessGroupOCCL.hpp)
```

For production backends, `Work` typically wraps an asynchronous handle from the vendor's communication library (e.g., a stream event or request handle), and `wait()` blocks until the operation completes on the device.

#### Backend Class

The backend class inherits from `c10d::Backend` and overrides the collective operations. Each method should validate that input tensors reside on the expected device type (e.g., `PrivateUse1`) and then dispatch to the vendor's communication library. Key implementation details:

- **`getBackendName()`** must return the same string used during Python registration (e.g., `"occl"`).
- **Input validation** – Each collective should verify tensor device types. The OCCL reference uses `CHECK_TENSOR` and `CHECK_TENSOR_LIST` macros for this.
- **Return value** – All collectives return a `c10::intrusive_ptr<Work>`.

See [`ProcessGroupOCCL.hpp`][ProcessGroupOCCL.hpp] and [`ProcessGroupOCCL.cpp`][ProcessGroupOCCL.cpp] for the full reference implementation.

### Step 2: Python Bindings

Expose the backend class to Python using pybind11. The OCCL reference places bindings in a dedicated [`init.cpp`][init.cpp] file, separate from the main extension module, and calls `initProcessGroupBindings()` from the module's entry point:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/distributed/init.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OCCL PYBIND
    :end-before: LITERALINCLUDE END: OCCL PYBIND
    :linenos:
    :caption: Python bindings for OCCL (init.cpp)
```

Important considerations:

- The `py::class_` template must list `c10d::Backend` as a base class and use `c10::intrusive_ptr` as the holder, so that PyTorch recognizes the backend in its internal registry.
- The constructor is exposed directly via `py::init` with a lambda that forwards to the C++ constructor. This avoids the need for a separate factory function.
- Guard the bindings with `#if USE_DISTRIBUTED` to handle builds where distributed is disabled.

### Step 3: Register the Backend in Python

In the extension package's `__init__.py`, register the backend with `torch.distributed`:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/__init__.py
    :language: python
    :start-after: LITERALINCLUDE START: OCCL BACKEND REGISTRATION
    :end-before: LITERALINCLUDE END: OCCL BACKEND REGISTRATION
    :linenos:
    :caption: Backend registration (torch_openreg/__init__.py)
```

The Python side imports the pybind11-exposed `ProcessGroupOCCL` class and wraps it in a thin factory function that matches the signature expected by `register_backend()`. The call to `Backend.register_backend()` does the following:

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

# Initialize process group – OCCL is auto-selected for openreg devices
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

Alternatively, the backend name can be omitted if a `device_id` is provided – PyTorch resolves the backend from the device-to-backend mapping:

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
[init.cpp]: https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/distributed/init.cpp "init.cpp"
