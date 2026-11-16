---
myst:
  html_meta:
    description: Device and DeviceType in PyTorch C++ — c10::Device for specifying CPU, CUDA, XPU, and other backends.
    keywords: PyTorch, C++, Device, DeviceType, CPU, CUDA, XPU, MPS, c10
---

# Device and DeviceType

PyTorch provides device abstractions for writing code that works across
CPU, CUDA, and other backends.

## Device

```{doxygenstruct} c10::Device
:members:
:undoc-members:
```

**Example:**

```cpp
c10::Device cpu_device(c10::kCPU);
c10::Device cuda_device(c10::kCUDA, 0);  // CUDA device 0

if (cuda_device.is_cuda()) {
    std::cout << "Using CUDA device " << cuda_device.index() << std::endl;
}
```

## DeviceType

```{cpp:enum-class} c10::DeviceType

Enumeration of supported device types.
```

```{cpp:enumerator} CPU = 0

CPU device.
```

```{cpp:enumerator} CUDA = 1

NVIDIA CUDA GPU.
```

```{cpp:enumerator} HIP = 6

AMD HIP GPU.
```

```{cpp:enumerator} XLA = 9

XLA / TPU.
```

```{cpp:enumerator} Vulkan = 10

Vulkan GPU.
```

```{cpp:enumerator} Metal = 11

Apple Metal GPU.
```

```{cpp:enumerator} XPU = 12

Intel XPU GPU.
```

```{cpp:enumerator} MPS = 13

Apple Metal Performance Shaders.
```

```{cpp:enumerator} Meta = 14

Meta tensors (shape only, no data).
```

```{cpp:enumerator} HPU = 15

Habana HPU.
```

```{cpp:enumerator} Lazy = 17

Lazy tensors.
```

```{cpp:enumerator} IPU = 18

Graphcore IPU.
```

```{cpp:enumerator} MTIA = 19

Meta training and inference accelerator.
```

```{cpp:enumerator} PrivateUse1 = 20

Custom backend registered via `c10::register_privateuse1_backend()`.
```

Convenience constants:

- `c10::kCPU`, `c10::kCUDA`, `c10::kHIP`
- `c10::kXLA`, `c10::kVulkan`, `c10::kMetal`
- `c10::kXPU`, `c10::kMPS`, `c10::kMeta`
- `c10::kHPU`, `c10::kLazy`, `c10::kIPU`, `c10::kMTIA`
- `c10::kPrivateUse1`
