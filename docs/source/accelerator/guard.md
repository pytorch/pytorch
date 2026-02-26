# Guard

## Background

The Device Guard abstraction in PyTorch provides RAII-based device and stream management, allowing code to temporarily switch device contexts and automatically restore the original device upon exiting a scope. This is essential for operations that need to ensure they execute on a specific device regardless of the current global device state.

For custom accelerators, a `CustomDeviceGuardImpl` that implements `c10::impl::DeviceGuardImplInterface` enables seamless integration with PyTorch's device management infrastructure. For example, the OpenReg (Open Registration) integration example in PyTorch provides an `OpenRegGuardImpl` for the `PrivateUse1` dispatch key, with the following capabilities:

- **Device Management**: Save, switch, and restore the current device index
- **Stream Management**: Create, query, and switch between computation streams
- **Event Management**: Record events on streams and synchronize execution

By registering `OpenRegGuardImpl` with PyTorch, user code can use device context managers like `torch.accelerator.device_index()`, and operations automatically handle device switching via the guard's RAII semantics.

## Design
The guard interface class [`c10::impl::DeviceGuardImplInterface`][DeviceGuardImplInterface] provides three main categories of functionality:

### Device Management

Device management enables switching between different accelerator devices and querying device information. This forms the foundation for device context management in PyTorch.

| Functionality               | Description                                                      | Application Scenario                                                             |
| --------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Device Query and Switch** | `exchangeDevice`, `setDevice`, `getDevice`, `uncheckedSetDevice` | Change the current device for operations; save/restore device context in RAII guards |
| **Device Count**            | `deviceCount` (noexcept)                                         | Query number of available devices; must return 0 on errors rather than throwing  |
| **Device Synchronization**  | `synchronizeDevice`                                              | Wait for all operations on a device to complete                                  |

### Stream Management

Streams enable asynchronous execution of operations on accelerator devices. Multiple streams allow concurrent execution of independent operations on the same device.

| Functionality              | Description                                                                                  | Application Scenario                                                 |
| -------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Stream Creation/Access** | `getStream`, `getDefaultStream`, `getNewStream`, `getStreamFromGlobalPool`, `exchangeStream` | Create and manage compute streams for asynchronous execution         |
| **Stream Synchronization** | `queryStream`, `synchronizeStream`                                                           | Check stream completion status, wait for stream operations to finish |

### Event Management

Events provide synchronization primitives for coordinating execution across streams and measuring execution time. They mark specific points in stream execution that can be awaited or timed.

| Functionality             | Description                      | Application Scenario                                                              |
| ------------------------- | -------------------------------- | --------------------------------------------------------------------------------- |
| **Event Recording**       | `record`                         | Mark a point in stream execution for later synchronization or timing              |
| **Event Blocking**        | `block`                          | Make one stream wait for an event from another stream (inter-stream dependencies) |
| **Event Synchronization** | `queryEvent`, `synchronizeEvent` | Check event completion, wait for event to complete                                |
| **Event Lifecycle**       | `destroyEvent`                   | Free event resources when no longer needed                                        |
| **Event Timing**          | `elapsedTime`                    | Measure time between two events for performance profiling                         |

## Implementation

- {ref}`Device Guard Implementation <device-guard>`
- Stream Guard Implementation (Upcoming)
- Event Guard Implementation (Upcoming)

[OpenReg Guard]: https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.h "OpenReg Guard"
[DeviceGuardImplInterface]: https://github.com/pytorch/pytorch/blob/main/c10/core/impl/DeviceGuardImplInterface.h "DeviceGuardImplInterface"
