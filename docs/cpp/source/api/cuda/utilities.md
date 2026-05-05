---
myst:
  html_meta:
    description: CUDA utility functions in PyTorch C++ — device properties, cuBLAS/cuDNN handles, and stream management.
    keywords: PyTorch, C++, CUDA, device_count, cuBLAS, cuDNN, cuSPARSE, TensorDescriptor
---

# CUDA Utility Functions

PyTorch provides utility functions for querying and managing CUDA devices,
streams, and library handles.

## Device Management

```{doxygenfunction} torch::cuda::device_count
```

```{cpp:function} int c10::cuda::current_device()

Returns the index of the current CUDA device.
```

**Example:**

```cpp
#include <c10/cuda/CUDAFunctions.h>

// Check available devices
int num_devices = c10::cuda::device_count();

// Get current device
int current = c10::cuda::current_device();
```

## Device Properties

```{doxygenfunction} at::cuda::getCurrentDeviceProperties
```

```{doxygenfunction} at::cuda::getDeviceProperties
```

```{doxygenfunction} at::cuda::canDeviceAccessPeer
```

```{doxygenfunction} at::cuda::warp_size
```

**Example:**

```cpp
#include <ATen/cuda/CUDAContext.h>

// Query properties of the current device
cudaDeviceProp* props = at::cuda::getCurrentDeviceProperties();
std::cout << "Device: " << props->name << std::endl;
std::cout << "Compute capability: " << props->major << "." << props->minor << std::endl;

// Query a specific device
cudaDeviceProp* dev1_props = at::cuda::getDeviceProperties(1);

// Check peer access
bool can_access = at::cuda::canDeviceAccessPeer(0, 1);
```

## Library Handles

These functions return handles for CUDA math libraries on the current device
and stream. They are primarily useful when writing custom CUDA kernels that
call cuBLAS, cuSPARSE, or cuSOLVER directly.

```{doxygenfunction} at::cuda::getCurrentCUDABlasHandle
```

```{doxygenfunction} at::cuda::getCurrentCUDABlasLtHandle
```

```{doxygenfunction} at::cuda::getCurrentCUDASparseHandle
```

```{doxygenfunction} at::cuda::getCurrentCUDASolverDnHandle
```

**Example:**

```cpp
#include <ATen/cuda/CUDAContext.h>

// Get cuBLAS handle for current device/stream
cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

// Get cuSPARSE handle
cusparseHandle_t sparse_handle = at::cuda::getCurrentCUDASparseHandle();
```

## cuDNN Descriptors

When writing custom kernels that call cuDNN directly, PyTorch provides RAII
wrapper classes for cuDNN descriptors. These are defined in
`ATen/cudnn/Descriptors.h`.

### Descriptor (Base Class)

```{doxygenclass} at::native::Descriptor
:members:
:undoc-members:
```

A generic RAII wrapper for cuDNN descriptor types. Descriptors default
construct to a `nullptr` and are initialized on first use via `mut_desc()`.
Use `desc()` for read-only access.

### TensorDescriptor

```{doxygenclass} at::native::TensorDescriptor
:members:
:undoc-members:
```

Wraps `cudnnTensorDescriptor_t`. Supports padding lower-dimensional tensors
to meet cuDNN broadcasting requirements (see `pad` parameter).

**Example:**

```cpp
#include <ATen/cudnn/Descriptors.h>

at::Tensor input = torch::randn({32, 3, 224, 224}, torch::kCUDA);
at::native::TensorDescriptor desc(input);
cudnnTensorDescriptor_t raw = desc.desc();
```

### FilterDescriptor

```{doxygenclass} at::native::FilterDescriptor
:members:
:undoc-members:
```

Wraps `cudnnFilterDescriptor_t` for convolution filter weights.

### ConvolutionDescriptor

```{doxygenstruct} at::native::ConvolutionDescriptor
:members:
:undoc-members:
```

Wraps `cudnnConvolutionDescriptor_t`. Configures padding, stride, dilation,
groups, and math type (TF32, tensor ops) for convolution operations.

### RNNDataDescriptor

```{doxygenclass} at::native::RNNDataDescriptor
:members:
:undoc-members:
```

Wraps `cudnnRNNDataDescriptor_t` for variable-length sequence data.

### DropoutDescriptor

```{doxygenstruct} at::native::DropoutDescriptor
:members:
:undoc-members:
```

Wraps `cudnnDropoutDescriptor_t`. Manages RNG state for cuDNN dropout.

### ActivationDescriptor

```{doxygenstruct} at::native::ActivationDescriptor
:members:
:undoc-members:
```

Wraps `cudnnActivationDescriptor_t`.

### SpatialTransformerDescriptor

```{doxygenstruct} at::native::SpatialTransformerDescriptor
:members:
:undoc-members:
```

### CTCLossDescriptor

```{doxygenstruct} at::native::CTCLossDescriptor
:members:
:undoc-members:
```

## Stream Management

```{doxygenfunction} c10::cuda::getDefaultCUDAStream
```

```{doxygenfunction} c10::cuda::getCurrentCUDAStream
```

```{doxygenfunction} c10::cuda::setCurrentCUDAStream
```

```{doxygenfunction} c10::cuda::getStreamFromPool(const bool isHighPriority, DeviceIndex device)
```

```{doxygenfunction} c10::cuda::getStreamFromExternal
```

**Example:**

```cpp
#include <c10/cuda/CUDAStream.h>

// Create and set custom stream
auto stream = c10::cuda::getStreamFromPool();
c10::cuda::setCurrentCUDAStream(stream);

// Get default stream
auto default_stream = c10::cuda::getDefaultCUDAStream();

// Wrap an externally created CUDA stream
cudaStream_t ext_stream;
cudaStreamCreate(&ext_stream);
auto wrapped = c10::cuda::getStreamFromExternal(ext_stream, /*device_index=*/0);
```
