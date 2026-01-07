# Device Management

## Background

Device management covers basics such as querying how many devices are available and switching between them. Accelerator backends wrap their device‑runtime APIs and expose them to PyTorch.

## Design

Accelerator vendors should implement these core functions:

| Function name             | Description                                                      | Application scenarios                                                                                          |
| ------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `device_count()`          | Query the total number of available devices in the system        | - Application initialization<br>- Multi-device workload distribution<br>- Validating device indices before use |
| `current_device()`        | Get the currently active device for the calling thread           | - Debugging and logging<br>- Determining tensor placement<br>- Guard implementations                           |
| `set_device()`            | Change the active device for subsequent operations               | - Switching context between devices<br>- Initializing specific device resources<br>- Multi-GPU training loops  |
| `exchange_device()`       | Atomically swap device and return the previous device            | - Implementing device guards<br>- Temporarily switching device context<br>- RAII-based device management       |
| `maybe_exchange_device()` | Conditionally exchange device only if the index is valid (−1 allowed) | - Safe device switching with optional indices<br>- Guard implementations with nullable device values           |

These functions are the building blocks for streams, events, and memory management. Validate inputs and handle errors properly.

## Implementation

This section illustrates device management using `set_device` as an example. The implementation requires:
1. C++ wrappers around the device runtime
2. Python bindings to expose the C++ functions
3. User-friendly Python APIs

For illustration, OpenReg (Open Registration) is a PyTorch integration example that fills the gap for out‑of‑tree accelerator backend integration. Its implementation ([`OpenRegFunctions.h/cpp`][OpenReg Device Management]) demonstrates how to wrap a third‑party runtime cleanly. These functions are reused across the backend—for streams, events, generators, and Python bindings.


### C++ side

Wrap the device‑runtime API and add error handling. The `SetDevice` function shows this pattern:

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

### Bindings

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

### Python side

Wrap the C++ bindings with user-friendly Python functions:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/openreg/__init__.py
    :language: python
    :start-after: LITERALINCLUDE START: PYTHON SET DEVICE FUNCTION
    :end-before: LITERALINCLUDE END: PYTHON SET DEVICE FUNCTION
    :linenos:
```

Here's the complete mapping from C++ to Python:

| C++ binding function | C++ binding API (pybind11)               | Python user API                  | Description                                  |
| -------------------- | ---------------------------------------- | -------------------------------- | -------------------------------------------- |
| `_getDeviceCount`    | `torch_openreg._C._get_device_count()`   | `torch.openreg.device_count()`   | Returns the total number of devices          |
| `_getDevice`         | `torch_openreg._C._get_device()`         | `torch.openreg.current_device()` | Returns the current active device index      |
| `_setDevice`         | `torch_openreg._C._set_device(idx)`      | `torch.openreg.set_device(idx)`  | Sets the active device                       |
| `_exchangeDevice`    | `torch_openreg._C._exchange_device(idx)` | N/A (internal use only)          | Atomically swaps device and returns previous |

(device-guard)=

## Guard

Device guards provide automatic device switching with exception safety. They’re similar to C++ lock guards—they switch devices on construction and restore on destruction.

Implement `DeviceGuardImplInterface` to integrate with PyTorch's guard system:

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegGuard.h
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG ALL DEVICE GUARD IMPL
    :end-before: LITERALINCLUDE END: OPENREG ALL DEVICE GUARD IMPL
    :linenos:
```

This makes the guard available in PyTorch for the `PrivateUse1` device type; users can then use standard PyTorch device guards with the custom backend.

[OpenReg Device Management]: https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegFunctions.cpp "OpenReg Device Management"
