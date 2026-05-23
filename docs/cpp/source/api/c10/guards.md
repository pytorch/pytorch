---
myst:
  html_meta:
    description: Device and stream guards in PyTorch C++ — RAII guards for managing current device and stream context.
    keywords: PyTorch, C++, DeviceGuard, StreamGuard, RAII, device management
---

# Device Guards

C10 provides device-agnostic RAII guards for managing the current device
context. These guards work across all backends (CUDA, XPU, etc.) and
automatically restore the previous device when they go out of scope.

For backend-specific guards, see {doc}`../cuda/guards` and {doc}`../xpu/index`.

## DeviceGuard

```{doxygenclass} c10::DeviceGuard
:members:
:undoc-members:
```

**Example:**

```cpp
#include <c10/core/DeviceGuard.h>

{
    c10::DeviceGuard guard(c10::Device(c10::kCUDA, 1));
    // All operations here run on CUDA device 1
}
// Previous device is restored
```

## OptionalDeviceGuard

```{doxygenclass} c10::OptionalDeviceGuard
:members:
:undoc-members:
```

**Example:**

```cpp
#include <c10/core/DeviceGuard.h>

c10::OptionalDeviceGuard guard;
if (use_gpu) {
    guard.reset_device(c10::Device(c10::kCUDA, 0));
}
// Guard only restores device if it was set
```
