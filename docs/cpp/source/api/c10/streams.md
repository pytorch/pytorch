---
myst:
  html_meta:
    description: Stream API in PyTorch C++ — c10::Stream for asynchronous execution on devices.
    keywords: PyTorch, C++, Stream, c10::Stream, asynchronous, device
---

# Streams

`c10::Stream` is the device-agnostic base stream class. It provides a
common interface for working with streams across different backends
(CUDA, XPU, etc.).

For backend-specific stream APIs, see {doc}`../cuda/streams` and {doc}`../xpu/index`.

## Stream

```{doxygenclass} c10::Stream
:members:
:undoc-members:
```

**Example:**

```cpp
#include <c10/core/Stream.h>

// Streams are typically obtained from backend-specific APIs
auto cuda_stream = c10::cuda::getCurrentCUDAStream();

// c10::Stream provides the common interface
c10::Device device = cuda_stream.device();
c10::DeviceType type = cuda_stream.device_type();
```
