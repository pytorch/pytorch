# Device Guard

## Background

The Device Guard abstraction in PyTorch provides RAII-based device and stream management, allowing code to temporarily switch device contexts and automatically restore the original device upon exiting a scope. This is essential for operations that need to ensure they execute on a specific device regardless of the current global device state.

For custom accelerators registered via the OpenReg framework, implementing a DeviceGuardImpl enables seamless integration with PyTorch's device management infrastructure. The `OpenRegGuardImpl` class implements `c10::impl::DeviceGuardImplInterface` for the `PrivateUse1` dispatch key, providing the following key capabilities:

- **Device Context Management**: Save, switch, and restore the current device index
- **Stream Management**: Create, query, and switch between computation streams
- **Event Synchronization**: Record events on streams and synchronize execution
- **Device Synchronization**: Wait for all operations on a device or stream to complete

By registering `OpenRegGuardImpl` with PyTorch, user code can use device context managers like `torch.accelerator.device_index()` and operations automatically handle device switching through the guard's RAII semantics.

## Design

The `OpenRegGuardImpl` implements the `c10::impl::DeviceGuardImplInterface` and provides three main categories of functionality:

### Device Management

Device management enables switching between different accelerator devices and querying device information. This forms the foundation for device context management in PyTorch.

| Functionality               | Description                                                      | Application Scenario                                                             |
| --------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Device Query and Switch** | `exchangeDevice`, `setDevice`, `getDevice`, `uncheckedSetDevice` | Change current device for operations, save/restore device context in RAII guards |
| **Device Count**            | `deviceCount` (noexcept)                                         | Query number of available devices; must return 0 on errors rather than throwing  |
| **Device Synchronization**  | `synchronizeDevice`                                              | Wait for all operations on a device to complete                                  |

### Stream Management

Streams enable asynchronous execution of operations on accelerator devices. Multiple streams allow concurrent execution of independent operations on the same device.

| Functionality              | Description                                                                                  | Application Scenario                                                 |
| -------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Stream Creation/Access** | `getStream`, `getDefaultStream`, `getNewStream`, `getStreamFromGlobalPool`, `exchangeStream` | Create and manage computation streams for asynchronous execution     |
| **Stream Synchronization** | `queryStream`, `synchronizeStream`                                                           | Check stream completion status, wait for stream operations to finish |

### Event Management

Events provide synchronization primitives for coordinating execution across streams and measuring execution time. They mark specific points in stream execution that can be waited on or timed.

| Functionality             | Description                      | Application Scenario                                                              |
| ------------------------- | -------------------------------- | --------------------------------------------------------------------------------- |
| **Event Recording**       | `record`                         | Mark a point in stream execution for later synchronization or timing              |
| **Event Blocking**        | `block`                          | Make one stream wait for an event from another stream (inter-stream dependencies) |
| **Event Synchronization** | `queryEvent`, `synchronizeEvent` | Check event completion, wait for event to complete                                |
| **Event Lifecycle**       | `destroyEvent`                   | Free event resources when no longer needed                                        |
| **Event Timing**          | `elapsedTime`                    | Measure time between two events for performance profiling                         |

**Design Principles:**

- All device parameters must be validated to ensure they have `DeviceType::PrivateUse1`
- Operations that access device-specific resources (streams, events) must temporarily switch to that device and restore the original device afterward to maintain consistent global state
- The `deviceCount()` method must be noexcept and return 0 on fatal errors (required by C10 contract)
- Event objects are created lazily on first `record()` call with flags determined by `EventFlag` parameter
- Streams and events are thin wrappers around OpenReg runtime types, requiring conversions via `OpenRegStream` and `orEvent_t`

## Implementation

The DeviceGuard operates entirely at the C++ level with no direct Python implementation. PyTorch's C++ runtime uses the guard internally when managing device contexts.

### Python Side

The DeviceGuard has no Python-side implementation. Instead, Python APIs like `torch.accelerator.device_index()` provide user-facing device context management that internally triggers C++ guard usage through PyTorch's dispatch system.

Example user code that indirectly uses the guard:

```python
import torch

# Context manager for device switching - uses DeviceGuard internally
with torch.accelerator.device_index(1):
    # All operations here run on device 1
    tensor = torch.randn(100, device='openreg:1')
    result = tensor * 2
# Original device automatically restored when exiting context
```

### C++ Side

The guard implementation is registered using the `C10_REGISTER_GUARD_IMPL` macro:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GUARD REGISTRATION
    :end-before: LITERALINCLUDE END: OPENREG GUARD REGISTRATION
    :linenos:
```

The `record` method demonstrates the key implementation pattern used throughout the guard: device validation, temporary device switching, operation execution, and automatic device restoration:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GUARD EVENT RECORD
    :end-before: LITERALINCLUDE END: OPENREG GUARD EVENT RECORD
    :linenos:
```

**Implementation Notes:**

- **Device Context Management**: All operations save the current device via `orGetDevice()`, switch to the target device with `orSetDevice()`, perform the operation, and restore the original device
- **Type Safety**: Device parameters are validated using `TORCH_CHECK(d.is_privateuseone(), ...)` to ensure only `PrivateUse1` devices are processed
- **Error Handling**: OpenReg runtime calls are wrapped with `OPENREG_CHECK()` to convert runtime errors into PyTorch exceptions
- **Lazy Event Creation**: Events are created on-demand during the first `record()` call, with creation flags determined by the `EventFlag` parameter
- **Stream Integration**: The guard uses `OpenRegStream` wrapper objects to convert between `c10::Stream` and native OpenReg stream handles

## Integration

The DeviceGuard is invoked automatically by PyTorch's device management infrastructure. When users interact with device context APIs, the guard ensures proper device switching and restoration. Below is a layer-by-layer flow showing how a device context switch propagates through the system.

### Layer 1: User Code

User code uses the `torch.accelerator.device_index()` context manager to temporarily switch to device 1:

```python
import torch

# Switch to device 1 within a context
with torch.accelerator.device_index(1):
    tensor = torch.randn(100, device='openreg:1')
    # Operations execute on device 1
# Automatically restored to original device
```

**Responsibility**: Provide high-level device context management API for users

### Layer 2: Python `torch.accelerator` Module

The `device_index` context manager class:

```{eval-rst}
.. literalinclude:: ../../../torch/accelerator/__init__.py
    :language: python
    :lines: 249-267
    :linenos:
```

**Responsibility**: Implement context manager that calls C++ device exchange functions

### Layer 3: PyTorch C++ Accelerator API

The `InlineDeviceGuard` constructor demonstrates RAII-based device management:

```{eval-rst}
.. literalinclude:: ../../../c10/core/impl/InlineDeviceGuard.h
    :language: cpp
    :lines: 71-77
    :linenos:
```

**Responsibility**: RAII guard that automatically exchanges devices on construction and restores on destruction

### Layer 4: DeviceGuardImplInterface Dispatch

The `DeviceGuardImplInterface` provides the abstract interface that dispatches to device-specific implementations. Key virtual methods from `c10/core/impl/DeviceGuardImplInterface.h`:

```{eval-rst}
.. literalinclude:: ../../../c10/core/impl/DeviceGuardImplInterface.h
    :language: cpp
    :lines: 61-73
    :linenos:
```

**Responsibility**: Define interface contract for device guards; dispatch to registered implementation based on device type

### Layer 5: OpenRegGuardImpl Registration

The guard implementation is registered for `PrivateUse1` device type:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GUARD REGISTRATION
    :end-before: LITERALINCLUDE END: OPENREG GUARD REGISTRATION
    :linenos:
```

**Responsibility**: Register OpenRegGuardImpl as the device guard handler for PrivateUse1 devices

### Layer 6: OpenRegGuardImpl Device Management

The actual device switching implementation:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GUARD DEVICE MANAGEMENT
    :end-before: LITERALINCLUDE END: OPENREG GUARD DEVICE MANAGEMENT
    :linenos:
```

**Responsibility**: Validate device type is PrivateUse1, delegate to OpenReg runtime functions (`orGetDevice()`, `orSetDevice()`) to perform actual device switching