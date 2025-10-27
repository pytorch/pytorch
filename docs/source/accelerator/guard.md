# Device Guard

## Background

The Device Guard abstraction in PyTorch centralizes device and stream management so code can be written generically against multiple device backends. A DeviceGuardImpl implementation encapsulates how to query and change the current device, create and query streams/events, and synchronize work for a particular device type.

The OpenReg backend provides an implementation of this interface for the `PrivateUse1` dispatch key, enabling code to treat OpenReg devices like other built-in backends while delegating backend-specific calls (e.g., `set_device`, `orStreamWaitEvent`) to the OpenReg runtime.

Key responsibilities of `OpenRegGuardImpl` include:

- Managing the current OpenReg device index.
- Creating, querying, and synchronizing streams and events.
- Mapping C10 `Stream`/`Event` values to the OpenReg runtime equivalents.
- Ensuring calls that temporarily change the device restore the original device when finished.

## Design

`OpenRegGuardImpl` implements `c10::impl::DeviceGuardImplInterface` and registers itself for `DeviceType::PrivateUse1`. Its `static_type` is therefore `DeviceType::PrivateUse1`.

The implementation follows the standard DeviceGuard contract:

- **device type**: return the handled `DeviceType`.
- `exchangeDevice` / `setDevice` / `getDevice` / `uncheckedSetDevice`: change or query the current device index.
- **stream management**: `getStream`, `getDefaultStream`, `getNewStream`, `getStreamFromGlobalPool`, exchangeStream.
- **device count**: return number of available OpenReg devices (must not throw; return 0 on driver errors).
- **event lifecycle and timing**: `create/destroy/record/block/query/synchronize events` and compute `elapsedTime`.
- **stream and device synchronizations**: `queryStream`, `synchronizeStream`, `synchronizeDevice`.

Design notes and invariants:

- Device indices are validated to be `PrivateUse1` devices where appropriate and calls use `TORCH_CHECK` for precondition enforcement.
- Many backend calls temporarily change the active device to the stream/event device index and restore the previous device at the end of the operation to ensure the global thread-local device state remains consistent.
- Stream and event objects are thin wrappers around OpenReg runtime types; conversions occur via `OpenRegStream` and `orEvent_t` casts.
- Event API honors `EventFlag` decisions (e.g., `PYTORCH_DEFAULT` vs `BACKEND_DEFAULT`) and maps them to OpenReg event creation flags.

## Implementation

### C++ Integration

The `OpenRegGuardImpl` class provides the core device guard implementation for OpenReg devices, inheriting from `c10::impl::DeviceGuardImplInterface` and handling the `PrivateUse1` device type.

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG DEVICE MGMT GUARD IMPL EXAMPLE
    :end-before: LITERALINCLUDE END: OPENREG DEVICE MGMT GUARD IMPL EXAMPLE
    :linenos:
```

The guard implementation must be registered with PyTorch's dispatch system using the `C10_REGISTER_GUARD_IMPL` macro:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GUARD REGISTRATION
    :end-before: LITERALINCLUDE END: OPENREG GUARD REGISTRATION
    :linenos:
```

The implementation provides device management through methods that validate device types and delegate to OpenReg runtime functions:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.h
    :language: c++
    :start-after: LITERALINCLUDE START: GUARD DEVICE MANAGEMENT
    :end-before: LITERALINCLUDE END: GUARD DEVICE MANAGEMENT
    :linenos:
```

### Event Management

Event recording demonstrates the key patterns used throughout the guard implementation: device switching with automatic restoration, type conversions between C10 and OpenReg types, and error checking:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.h
    :language: c++
    :start-after: LITERALINCLUDE START: GUARD EVENT RECORD
    :end-before: LITERALINCLUDE END: GUARD EVENT RECORD
    :linenos:
```

### Key Implementation Patterns

The OpenReg guard implementation follows several important patterns:

- **Device Context Management**: All operations that interact with OpenReg runtime save the current device, switch to the target device, perform the operation, and restore the original device.

- **Type Safety**: Device parameters are validated using `TORCH_CHECK(d.is_privateuseone(), ...)` to ensure only `PrivateUse1` devices are processed.

- **Error Handling**: OpenReg runtime calls are wrapped with `OPENREG_CHECK(...)` to convert runtime errors into PyTorch exceptions.

- **Lazy Event Creation**: Events are created on-demand during the first `record()` call, with creation flags determined by the `EventFlag` parameter.

- **Stream Integration**: The guard uses `OpenRegStream` wrapper objects to convert between `c10::Stream` and native OpenReg stream handles.

## Runtime behavior and examples

### Device Management

The guard provides several methods for device management:

- **`exchangeDevice(Device d)`**: Sets the current device and returns the previous device, validating that the device is `PrivateUse1`.
- **`setDevice(Device d)`**: Sets the current device after validation.
- **`getDevice()`**: Returns the current device as a `Device` object.
- **`uncheckedSetDevice(Device d)`**: Sets the device without error checking (safe for destructors).
- **`deviceCount()`**: Returns the number of available OpenReg devices (must not throw).

### Stream Management

Stream operations return `c10::Stream` objects that wrap OpenReg streams:

- **`getStream(Device d)`**: Gets the current stream for a device.
- **`getDefaultStream(Device d)`**: Gets the default stream for a device.
- **`getNewStream(Device d, int priority)`**: Creates a new stream from the pool with specified priority.
- **`getStreamFromGlobalPool(Device d, bool isHighPriority)`**: Gets a stream from the global pool.
- **`exchangeStream(Stream s)`**: Sets a stream as current and returns the previous stream.

### Event Lifecycle

Event operations handle timing and synchronization:

- **`record(void** event, const Stream& stream, DeviceIndex device_index, EventFlag flag)`**: Records an event on a stream, creating the event if null.
- **`block(void* event, const Stream& stream)`**: Makes a stream wait for an event to complete.
- **`destroyEvent(void* event, DeviceIndex device_index)`**: Destroys an event and frees resources.
- **`queryEvent(void* event)`**: Returns true if the event has completed or was never scheduled.
- **`synchronizeEvent(void* event)`**: Blocks until the event completes.
- **`elapsedTime(void* event1, void* event2, DeviceIndex device_index)`**: Computes elapsed time between two events.

### Synchronization

Synchronization methods for streams and devices:

- **`queryStream(const Stream& stream)`**: Returns true if all work on the stream has completed.
- **`synchronizeStream(const Stream& stream)`**: Blocks until all work on the stream completes.
- **`synchronizeDevice(DeviceIndex device_index)`**: Blocks until all work on the device completes.

## Error handling and safety

- The guard uses `TORCH_CHECK` for precondition validation (e.g., device type checks and matching device indices for events/streams).
- Backend calls are wrapped with `OPENREG_CHECK` so that underlying runtime failures are surfaced as exceptions consistent with other backends.
- `deviceCount()` is noexcept and must return 0 on fatal errors rather than throwing — this is an important contract for the C10 guard layer.