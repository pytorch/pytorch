# Device Management

## Background

Device management handles basic operations like querying how many devices are available and switching between them. Accelerator backends need to wrap their device runtime's APIs and expose them to PyTorch.

The OpenReg implementation ([`OpenRegFunctions.h/cpp`][OpenReg Device Management]) shows how to wrap a third-party runtime. These functions are used throughout the backend - by streams, events, generators, and Python bindings.

## Design

Accelerator vendors need to implement these core functions:

| Function Name             | Description                                                      | Application Scenarios                                                                                          |
| ------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `device_count()`          | Query the total number of available devices in the system        | - Application initialization<br>- Multi-device workload distribution<br>- Validating device indices before use |
| `current_device()`        | Get the currently active device for the calling thread           | - Debugging and logging<br>- Determining tensor placement<br>- Guard implementations                           |
| `set_device()`            | Change the active device for subsequent operations               | - Switching context between devices<br>- Initializing specific device resources<br>- Multi-GPU training loops  |
| `exchange_device()`       | Atomically swap device and return the previous device            | - Implementing device guards<br>- Temporarily switching device context<br>- RAII-based device management       |
| `maybe_exchange_device()` | Conditionally exchange device only if the index is valid (-1 OK) | - Safe device switching with optional indices<br>- Guard implementations with nullable device values           |

These functions are building blocks for more complex features like streams, events, and memory management. Make sure to validate inputs and handle errors properly.

## Implementation

This section shows how to implement device management using `set_device` as an example. The implementation requires:
1. C++ wrappers around the device runtime
2. Python bindings to expose the C++ functions
3. User-friendly Python APIs

### C++ Side

Wrap the device runtime's API and add error handling. The `SetDevice` function shows this pattern:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegFunctions.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG SetDevice FUNCTION
    :end-before: LITERALINCLUDE END: OPENREG SetDevice FUNCTION
    :linenos:
```
```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegFunctions.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG set_device FUNCTION
    :end-before: LITERALINCLUDE END: OPENREG set_device FUNCTION
    :linenos:
```

### Binding

Expose the C++ functions to Python using pybind11:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: MODULE SET DEVICE HELPER
    :end-before: LITERALINCLUDE END: MODULE SET DEVICE HELPER
    :linenos:
```
```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG MODULE METHODS
    :end-before: LITERALINCLUDE END: OPENREG MODULE METHODS
    :linenos:
    :emphasize-lines: 5
```

### Python Side

Wrap the C++ bindings with user-friendly Python functions:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/__init__.py
    :language: python
    :start-after: LITERALINCLUDE START: PYTHON SET DEVICE FUNCTION
    :end-before: LITERALINCLUDE END: PYTHON SET DEVICE FUNCTION
    :linenos:
```

Here's the complete mapping from C++ to Python:

| C++ Binding Function | C++ Binding API (pybind11)               | Python User API                  | Description                                  |
| -------------------- | ---------------------------------------- | -------------------------------- | -------------------------------------------- |
| `_getDeviceCount`    | `torch_openreg._C._get_device_count()`   | `torch.openreg.device_count()`   | Returns the total number of devices          |
| `_getDevice`         | `torch_openreg._C._get_device()`         | `torch.openreg.current_device()` | Returns the current active device index      |
| `_setDevice`         | `torch_openreg._C._set_device(idx)`      | `torch.openreg.set_device(idx)`  | Sets the active device                       |
| `_exchangeDevice`    | `torch_openreg._C._exchange_device(idx)` | N/A (internal use only)          | Atomically swaps device and returns previous |

## Guard

Device guards provide automatic device switching with exception safety. They're similar to lock guards in C++ - they switch device on construction and restore it on destruction.

Implement `DeviceGuardImplInterface` to integrate with PyTorch's guard system:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG DEVICE MGMT GUARD IMPL EXAMPLE
    :end-before: LITERALINCLUDE END: OPENREG DEVICE MGMT GUARD IMPL EXAMPLE
    :linenos:
```

**What needs to be implemented:**

1. **exchangeDevice()**: Switch to a new device and return the old one (used by guard constructors)
2. **getDevice()**: Get the current device
3. **setDevice()**: Set the active device
4. **Type checking**: Validate that device type matches the backend

This makes the guard available to PyTorch for the `PrivateUse1` device type. Users can then use standard PyTorch device guards with the custom backend.

[OpenReg Device Management]: https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegFunctions.cpp "OpenReg Device Management"