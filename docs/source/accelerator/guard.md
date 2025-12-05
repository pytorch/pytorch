# Guard

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

{ref}`Implementation <device-guard>`

Device management enables switching between different accelerator devices and querying device information. This forms the foundation for device context management in PyTorch.

| Functionality               | Description                                                      | Application Scenario                                                             |
| --------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **Device Query and Switch** | `exchangeDevice`, `setDevice`, `getDevice`, `uncheckedSetDevice` | Change current device for operations, save/restore device context in RAII guards |
| **Device Count**            | `deviceCount` (noexcept)                                         | Query number of available devices; must return 0 on errors rather than throwing  |
| **Device Synchronization**  | `synchronizeDevice`                                              | Wait for all operations on a device to complete                                  |

### Stream Management

Implementation (documentation upcoming, currently can see code in [`OpenRegGuard.h`][OpenReg Guard] for reference)

Streams enable asynchronous execution of operations on accelerator devices. Multiple streams allow concurrent execution of independent operations on the same device.

| Functionality              | Description                                                                                  | Application Scenario                                                 |
| -------------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Stream Creation/Access** | `getStream`, `getDefaultStream`, `getNewStream`, `getStreamFromGlobalPool`, `exchangeStream` | Create and manage computation streams for asynchronous execution     |
| **Stream Synchronization** | `queryStream`, `synchronizeStream`                                                           | Check stream completion status, wait for stream operations to finish |

### Event Management

Implementation (documentation upcoming, currently can see code in [`OpenRegGuard.h`][OpenReg Guard] for reference)

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

[OpenReg Guard]: https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.h "OpenReg Guard"
