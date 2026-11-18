---
myst:
  html_meta:
    description: XPU utility functions in PyTorch C++ — device count, properties, and stream management for Intel GPUs.
    keywords: PyTorch, C++, XPU, Intel GPU, device_count, utilities
---

# XPU Utility Functions

High-level utility functions for querying and managing XPU devices.

## Device Management

```{doxygenfunction} torch::xpu::device_count
```

```{doxygenfunction} torch::xpu::is_available
```

```{doxygenfunction} torch::xpu::synchronize
```

**Example:**

```cpp
#include <torch/torch.h>

if (torch::xpu::is_available()) {
    size_t num_devices = torch::xpu::device_count();
    std::cout << "Found " << num_devices << " XPU device(s)" << std::endl;

    // Synchronize all streams on device 0
    torch::xpu::synchronize(0);
}
```

## Random Number Generation

```{doxygenfunction} torch::xpu::manual_seed
```

```{doxygenfunction} torch::xpu::manual_seed_all
```

**Example:**

```cpp
// Set seed for reproducibility on current XPU device
torch::xpu::manual_seed(42);

// Set seed for all XPU devices
torch::xpu::manual_seed_all(42);
```
