---
myst:
  html_meta:
    description: XPU streams in PyTorch C++ — XPUStream for asynchronous Intel GPU execution.
    keywords: PyTorch, C++, XPU, XPUStream, Intel GPU, stream, asynchronous
---

# XPU Streams

XPU streams provide a mechanism for asynchronous execution of operations
on Intel GPUs. Like CUDA streams, operations queued to the same stream execute
in order, while operations on different streams can execute concurrently.

## XPUStream

```{doxygenclass} c10::xpu::XPUStream
:members:
:undoc-members:
```

**Example:**

```cpp
#include <c10/xpu/XPUStream.h>

// Get the current XPU stream
auto stream = c10::xpu::getCurrentXPUStream();

// Create a new stream from the pool
auto new_stream = c10::xpu::getStreamFromPool();

// Synchronize
stream.synchronize();
```

## Acquiring XPU Streams

```{doxygenfunction} c10::xpu::getCurrentXPUStream
```

```{doxygenfunction} c10::xpu::setCurrentXPUStream
```

```{doxygenfunction} c10::xpu::getStreamFromPool(const bool isHighPriority, DeviceIndex device)
```

## Stream Synchronization

```{doxygenfunction} c10::xpu::syncStreamsOnDevice
```
