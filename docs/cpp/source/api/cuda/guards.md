---
myst:
  html_meta:
    description: CUDA device and stream guards in PyTorch C++ — CUDAGuard, CUDAStreamGuard, and CUDAMultiStreamGuard.
    keywords: PyTorch, C++, CUDA, CUDAGuard, CUDAStreamGuard, device guard, multi-GPU
---

# CUDA Guards

CUDA guards are RAII wrappers that set a CUDA device or stream as the current
context and automatically restore the previous context when the guard goes
out of scope.

## CUDAGuard

```{doxygenstruct} c10::cuda::CUDAGuard
:members:
:undoc-members:
```

**Example:**

```cpp
#include <c10/cuda/CUDAGuard.h>

{
    c10::cuda::CUDAGuard guard(1);  // Switch to device 1
    // All CUDA operations here run on device 1
    auto tensor = torch::zeros({2, 2}, torch::device(torch::kCUDA));
}
// Previous device is restored
```

## CUDAStreamGuard

```{doxygenstruct} c10::cuda::CUDAStreamGuard
:members:
:undoc-members:
```

**Example:**

```cpp
#include <c10/cuda/CUDAGuard.h>

auto stream = c10::cuda::getStreamFromPool();
{
    c10::cuda::CUDAStreamGuard guard(stream);
    // Operations here use the specified stream
}
// Previous stream is restored
```

## OptionalCUDAGuard

```{doxygenstruct} c10::cuda::OptionalCUDAGuard
:members:
:undoc-members:
```

**Example:**

```cpp
c10::cuda::OptionalCUDAGuard guard;
if (use_cuda) {
    guard.set_device(0);
}
// Guard only switches device if set_device was called
```

## OptionalCUDAStreamGuard

```{doxygenstruct} c10::cuda::OptionalCUDAStreamGuard
:members:
:undoc-members:
```

## CUDAMultiStreamGuard

```{doxygenstruct} c10::cuda::CUDAMultiStreamGuard
:members:
:undoc-members:
```

**Example:**

```cpp
at::cuda::CUDAStream stream0 = at::cuda::getStreamFromPool(false, 0);
at::cuda::CUDAStream stream1 = at::cuda::getStreamFromPool(false, 1);

{
    at::cuda::CUDAMultiStreamGuard multi_guard({stream0, stream1});
    // stream0 is current on device 0, stream1 on device 1
}
// Both streams restored
```
