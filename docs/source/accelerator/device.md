# Device Management

## Background

`OpenRegFunctions.h / cpp` is the small runtime module that adapts the underlying OpenReg simulated device runtime to the PyTorch device management APIs. It wraps low-level OpenReg (third-party) calls and exposes a concise, safe C++ API for device queries and management used across the backend (generator initialization, stream/event creation, module bindings, and tests).

Key responsibilities:

- Querying device count and current device
- Setting and exchanging the current device
- Normalizing errors and integrating with PyTorch's `TORCH_CHECK` and warning infrastructure

This module provides simple, well-tested functions that higher layers (C++ runtime and Python bindings) call to manage the simulated device state.

## Design and API

The runtime exposes a small API in `OpenRegFunctions.h` which is implemented in `OpenRegFunctions.cpp`.

- `DeviceIndex device_count() noexcept` — returns the number of available OpenReg devices. The implementation caches the device count on first call and never throws.
- `DeviceIndex current_device()` — returns the currently selected device index. Uses OpenReg API and performs error checks.
- `void set_device(DeviceIndex device)` — set the current device (with error checks).
- `DeviceIndex ExchangeDevice(DeviceIndex device)` — atomically (from the caller's perspective) set a device and return the previous device.

These routines translate the lower-level OpenReg runtime errors into the project's error handling (via `OPENREG_CHECK` / `TORCH_CHECK`) and provide a stable surface for other runtime pieces to rely on.

## Implementation

### C++ Integration

The implementation wraps the third-party OpenReg functions (`orGetDeviceCount`, `orGetDevice`, `orSetDevice`) and adds caching and error handling. For example, the `SetDevice` functions is shown below:
```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/csrc/runtime/OpenRegFunctions.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: OPENREG SET DEVICE FUNCTION
    :end-before: LITERALINCLUDE END: OPENREG SET DEVICE FUNCTION
    :linenos:
```

Important implementation notes:

- `device_count()` uses a function-local `static int count = [](){ ... }();` pattern. This ensures the number of devices is queried and validated once at program startup/access time, avoiding repeated system calls.
- Errors from the third-party runtime are wrapped with `OPENREG_CHECK` (a macro defined in this extension) which converts them to exceptions that play nicely with PyTorch's error handling.
- `device_count()` catches exceptions and emits a warning (`TORCH_WARN`) instead of failing the process, returning 0 devices when the runtime is not available.

### Python Integration

The C++ functions are consumed by the Python extension (`torch_openreg._C`) which exposes a handful of small helpers used by the Python package and tests, such as `_get_device_count`, `_get_device`, `_set_device`, and `_exchangeDevice`.

Bindings are implemented in `Module.cpp`. See the relevant snippet below which shows how the default generator and device helpers obtain and return values to Python.

```{eval-rst}
.. literalinclude:: ../../../test/cpp_extensions/open_registration_extension/torch_openreg/torch_openreg/csrc/Module.cpp
    :language: c++
    :start-after: LITERALINCLUDE START: MODULE SET DEVICE HELPER
    :end-before: LITERALINCLUDE END: MODULE SET DEVICE HELPER
    :linenos:
```

From Python, the binding provides small functions (via the `_C` extension) that are used by `torch_openreg`'s Python package. Example usages:

- `torch_openreg._C._get_device_count()` — returns the number of devices
- `torch_openreg._C._get_device()` — returns the current device index
- `torch_openreg._C._set_device(idx)` — set the active device
- `torch_openreg._C._exchangeDevice(idx)` — swap the device and get previous index