---
myst:
  html_meta:
    description: Stable ABI utilities in PyTorch C++ — version checking and compatibility helpers.
    keywords: PyTorch, C++, stable ABI, utilities, version, compatibility
---

# Utilities

The stable API provides various utility functions and types for working with
tensors and CUDA operations.

## DeviceGuard Class

```{doxygenclass} torch::stable::accelerator::DeviceGuard
:members:
:undoc-members:
```

```{doxygenfunction} torch::stable::accelerator::getCurrentDeviceIndex
```

**Example:**

```cpp
{
    torch::stable::accelerator::DeviceGuard guard(1);
    // Operations here run on device 1
}
// Previous device is restored
```

## Stream

```{doxygenclass} torch::stable::accelerator::Stream
:members:
:undoc-members:
```

## Stream Utilities

For CUDA stream access, we currently recommend the ABI stable C shim API. This
will be improved in a future release with a more ergonomic wrapper.

### Getting the Current CUDA Stream

To obtain the current `cudaStream_t` for use in CUDA kernels:

```cpp
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/headeronly/util/shim_utils.h>

// For now, we rely on the ABI stable C shim API to get the current CUDA stream.
void* stream_ptr = nullptr;
TORCH_ERROR_CODE_CHECK(
    aoti_torch_get_current_cuda_stream(tensor.get_device_index(), &stream_ptr));
cudaStream_t stream = static_cast<cudaStream_t>(stream_ptr);

// Now you can use 'stream' in your CUDA kernel launches
my_kernel<<<blocks, threads, 0, stream>>>(args...);
```

```{note}

The `TORCH_ERROR_CODE_CHECK` macro is required when using C shim APIs
to properly check error codes and throw appropriate exceptions.
```

## CUDA Error Checking Macros

These macros provide stable ABI equivalents for CUDA error checking.
They wrap CUDA API calls and kernel launches, providing detailed error
messages using PyTorch's error formatting.

### STD_CUDA_CHECK

```{c:macro} STD_CUDA_CHECK(EXPR)

Checks the result of a CUDA API call and throws an exception on error.
Users of this macro are expected to include `cuda_runtime.h`.

**Example:**

```cpp
STD_CUDA_CHECK(cudaMalloc(&ptr, size));
STD_CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
```

Minimum compatible version: PyTorch 2.10.
```
### STD_CUDA_KERNEL_LAUNCH_CHECK

```{c:macro} STD_CUDA_KERNEL_LAUNCH_CHECK()

Checks for errors from the most recent CUDA kernel launch. Equivalent to
`STD_CUDA_CHECK(cudaGetLastError())`.

**Example:**

```cpp
my_kernel<<<blocks, threads, 0, stream>>>(args...);
STD_CUDA_KERNEL_LAUNCH_CHECK();
```

Minimum compatible version: PyTorch 2.10.
```
## Header-Only Utilities

The `torch::headeronly` namespace provides header-only versions of common
PyTorch types and utilities. These can be used without linking against libtorch
at all! This portability makes them ideal for maintaining binary compatibility
across PyTorch versions.

### Error Checking

`STD_TORCH_CHECK` is a header-only macro for runtime assertions:

```cpp
#include <torch/headeronly/util/Exception.h>

STD_TORCH_CHECK(condition, "Error message with ", variable, " interpolation");
```

Wherever you used `TORCH_CHECK` before, you can replace usage with `STD_TORCH_CHECK`
to remove the need to link against libtorch. The only difference is that when the
condition check fails, `TORCH_CHECK` throws a fancier `c10::Error` while
`STD_TORCH_CHECK` throws a `std::runtime_error`.

### Core Types

The following `c10::` types are available as header-only versions under
`torch::headeronly::`:

- `torch::headeronly::ScalarType` - Tensor data types (Float, Double, Int, etc.)
- `torch::headeronly::DeviceType` - Device types (CPU, CUDA, etc.)
- `torch::headeronly::MemoryFormat` - Memory layout formats (Contiguous, ChannelsLast, etc.)
- `torch::headeronly::Layout` - Tensor layouts (Strided, Sparse, etc.)

```cpp
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/MemoryFormat.h>
#include <torch/headeronly/core/Layout.h>

auto dtype = torch::headeronly::ScalarType::Float;
auto device_type = torch::headeronly::DeviceType::CUDA;
auto memory_format = torch::headeronly::MemoryFormat::Contiguous;
auto layout = torch::headeronly::Layout::Strided;
```

### TensorAccessor

`TensorAccessor` provides efficient, bounds-checked access to tensor data.
You can construct one from a stable tensor's data pointer, sizes, and strides:

```cpp
#include <torch/headeronly/core/TensorAccessor.h>

// Create a TensorAccessor for a 2D float tensor
auto sizes = tensor.sizes();
auto strides = tensor.strides();
torch::headeronly::TensorAccessor<float, 2> accessor(
    static_cast<float*>(tensor.mutable_data_ptr()),
    sizes.data(),
    strides.data());

// Access elements
float value = accessor[i][j];
```

### Dispatch Macros

Header-only dispatch macros (THO = Torch Header Only) are available for
dtype dispatching:

```cpp
#include <torch/headeronly/core/Dispatch_v2.h>

THO_DISPATCH_V2(
   tensor.scalar_type(),  // will be resolved as scalar_t
   "my_kernel",
   AT_WRAP(([&]() {
   // code to specialize with scalar_t
   // scalar_t is the resolved C++ type (e.g. float, double)
   auto* data = static_cast<scalar_t*>(tensor.mutable_data_ptr());
   Scalar s(*data);
   })),
   AT_EXPAND(AT_ALL_TYPES),
   AT_EXPAND(AT_COMPLEX_TYPES),
   torch::headeronly::ScalarType::Half,
   // as many type arguments as needed
);
```

`THO_DISPATCH_V2` works the same way as `AT_DISPATCH_V2` (see
`ATen/Dispatch_v2.h`) but does not require linking against libtorch.
As a result, whereas `AT_DISPATCH_V2` would have thrown `c10::NotImplementedError`
for unimplemented paths, `THO_DISPATCH_V2` will throw `std::runtime_error`.

For ease of use, we've also migrated the below AT_* macros representing
collections of types to be header-only and thus have no dependency on libtorch:

- `AT_FLOATING_TYPES`
- `AT_INTEGRAL_TYPES`
- `AT_INTEGRAL_TYPES_V2`
- `AT_ALL_TYPES`
- `AT_COMPLEX_TYPES`
- `AT_ALL_TYPES_AND_COMPLEX`
- `AT_FLOAT8_TYPES`
- `AT_BAREBONES_UNSIGNED_TYPES`
- `AT_QINT_TYPES`

If your extension uses our older AT_DISPATCH version 1 infrastructure,
you can also migrate to a header-only libtorch-free world without upgrading
everything to version 2.

`THO_DISPATCH_SWITCH` and `THO_DISPATCH_CASE` are the header-only
equivalents of `AT_DISPATCH_SWITCH` and `AT_DISPATCH_CASE`. Similarly,
the only user-visible difference is the exception type on an unhandled dtype,
where the `AT_` version throws a `c10::NotImplementedError` and the `THO_`
version throws a `std::runtime_error`.

The migration is pretty mechanical:

- `AT_DISPATCH_SWITCH` → `THO_DISPATCH_SWITCH`
- `AT_DISPATCH_CASE` → `THO_DISPATCH_CASE`
- `AT_PRIVATE_CASE_TYPE_USING_HINT` → `THO_PRIVATE_CASE_TYPE_USING_HINT`
- `at::ScalarType::X` → `torch::headeronly::ScalarType::X`

```cpp
// ---- Before (requires linking against libtorch) ----
#include <torch/all.h>

#define MY_DISPATCH_CASE_FLOATING_TYPES(...)            \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define MY_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME,                    \
                     MY_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
```

```cpp
// ---- After (header-only, no libtorch dependency) ----
#include <torch/headeronly/core/Dispatch.h>

#define MY_DISPATCH_CASE_FLOATING_TYPES(...)                          \
  THO_DISPATCH_CASE(torch::headeronly::ScalarType::Float, __VA_ARGS__) \
  THO_DISPATCH_CASE(torch::headeronly::ScalarType::Half, __VA_ARGS__)  \
  THO_DISPATCH_CASE(torch::headeronly::ScalarType::BFloat16, __VA_ARGS__)

#define MY_DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  THO_DISPATCH_SWITCH(TYPE, NAME,                   \
                      MY_DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
```

For the complete list of header-only APIs, see `torch/header_only_apis.txt`
in the PyTorch source tree.

## Parallelization Utilities

```{doxygenfunction} torch::stable::parallel_for
```

```{doxygenfunction} torch::stable::get_num_threads
```
