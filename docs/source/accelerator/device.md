# Device Management

## Background

Device management is a foundational component of PyTorch's accelerator support system, enabling PyTorch to interact with different hardware accelerators in a unified manner. For custom accelerators like OpenReg, device management provides the essential infrastructure for:

- **Device Discovery and Enumeration**: Identifying how many accelerator devices are available in the system
- **Device Context Management**: Tracking and switching between different devices during computation
- **Error Handling and Validation**: Ensuring device operations are safe and providing meaningful error messages
- **Integration with PyTorch Ecosystem**: Enabling device-aware features like tensors, streams, events, and automatic mixed precision

The OpenReg device management implementation (`OpenRegFunctions.h/cpp`) serves as an adapter layer that wraps the underlying third-party OpenReg runtime and exposes a PyTorch-compatible C++ API. This module provides simple, well-tested functions that higher layers (streams, events, generators, Python bindings) rely on to manage device state throughout the backend.

## Design

The device management system must support the following basic functionalities:

| Functionality            | Description                                               | Application Scenarios                                                                                          |
| ------------------------ | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Device Count Query**   | Query the total number of available devices in the system | - Application initialization<br>- Multi-device workload distribution<br>- Validating device indices before use |
| **Current Device Query** | Get the currently active device for the calling thread    | - Debugging and logging<br>- Determining tensor placement<br>- Guard implementations                           |
| **Set Device**           | Change the active device for subsequent operations        | - Switching context between devices<br>- Initializing specific device resources<br>- Multi-GPU training loops  |
| **Exchange Device**      | Atomically swap device and return the previous device     | - Implementing device guards<br>- Temporarily switching device context<br>- RAII-based device management       |

These core functionalities form the foundation upon which more complex features (streams, events, memory management) are built. Each function must handle errors gracefully and integrate with PyTorch's existing error reporting infrastructure.

## Implementation

This section demonstrates how to implement the device management functionality using `set_device` as an example. The implementation follows a layered approach: low-level C++ wrappers, high-level C++ API, Python bindings, and finally Python user-facing API.

### C++ Side

The C++ implementation consists of two layers: a thin wrapper around the third-party runtime and a public API with error handling.

The `SetDevice` function wraps the OpenReg runtime calls and adds optimization (avoiding redundant sets):

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegFunctions.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG SET DEVICE FUNCTION
    :end-before: LITERALINCLUDE END: OPENREG SET DEVICE FUNCTION
    :linenos:
```

**Key implementation points:**

1. **Error Checking**: All OpenReg runtime calls are wrapped with `OPENREG_CHECK`, which converts error codes to exceptions
2. **Optimization**: Avoids redundant device switching by checking if the target device is already current
3. **Bounds Validation**: The public `set_device()` function calls `check_device_index()` to validate the device index is within valid range
4. **Caching**: `device_count()` uses static initialization to cache the device count, avoiding repeated system calls

### Python Side

The Python implementation provides a user-friendly API that internally calls the C++ bindings. The implementation in `torch_openreg/openreg/__init__.py` wraps the C extension (for example, `torch.openreg.set_device()`):

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/__init__.py
    :language: python
    :start-after: LITERALINCLUDE START: PYTHON SET DEVICE FUNCTION
    :end-before: LITERALINCLUDE END: PYTHON SET DEVICE FUNCTION
    :linenos:
```

## Integration

### Python

Device management functions must be exposed from C++ to Python to enable user access. The following example shows how `set_device` is exposed through Python bindings:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: MODULE SET DEVICE HELPER
    :end-before: LITERALINCLUDE END: MODULE SET DEVICE HELPER
    :linenos:
```


The complete list of device management APIs exposed to Python:

| C++ Binding Function | Python API                                                                       | Description                                  | Application Scenarios                                                                 |
| -------------------- | -------------------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------- |
| `_getDeviceCount`    | `torch_openreg._C._get_device_count()`<br>`torch_openreg.openreg.device_count()` | Returns the total number of devices          | - Application startup<br>- Multi-device configuration<br>- Device availability checks |
| `_getDevice`         | `torch_openreg._C._get_device()`<br>`torch_openreg.openreg.current_device()`     | Returns the current active device index      | - Debugging<br>- Logging device context<br>- Conditional device operations            |
| `_setDevice`         | `torch_openreg._C._set_device(idx)`<br>`torch_openreg.openreg.set_device(idx)`   | Sets the active device                       | - Multi-GPU training<br>- Device context switching<br>- Explicit device placement     |
| `_exchangeDevice`    | `torch_openreg._C._exchangeDevice(idx)`                                          | Atomically swaps device and returns previous | - Guard context managers<br>- Temporary device switches<br>- RAII patterns            |

### Guard

Device guards are essential for managing device context in PyTorch. They enable automatic, exception-safe device switching through RAII (Resource Acquisition Is Initialization) patterns. Guards are used extensively throughout PyTorch internals and user code.

The OpenReg backend integrates device management into `OpenRegGuardImpl`, which implements the `DeviceGuardImplInterface`. Below shows the key device management methods in the guard implementation:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG DEVICE MGMT GUARD IMPL EXAMPLE
    :end-before: LITERALINCLUDE END: OPENREG DEVICE MGMT GUARD IMPL EXAMPLE
    :linenos:
```

**Key guard implementation points:**

1. **Device Type Enforcement**: The guard enforces that it only manages `PrivateUse1` devices through compile-time constants and runtime checks
2. **Device Management Methods**:
   - `exchangeDevice()`: Atomically swaps to a new device and returns the previous device, used for RAII patterns
   - `getDevice()`: Queries the current active device by calling `current_device()`
   - `setDevice()`: Sets the active device with type validation
3. **Type Safety**: All methods validate that the device type is `PrivateUse1` using `is_privateuseone()` checks
4. **Integration**: Methods directly call the device management functions (`ExchangeDevice`, `current_device`, `set_device`) we implemented earlier

The guard is registered with PyTorch using the macro shown below:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG GUARD REGISTRATION
    :end-before: LITERALINCLUDE END: OPENREG GUARD REGISTRATION
    :linenos:
```

This registration makes the guard implementation available to PyTorch's device management system for the `PrivateUse1` device type.