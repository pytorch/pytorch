---
myst:
  html_meta:
    description: Stable ABI operator API in PyTorch C++ — StableLibrary and boxed kernel registration.
    keywords: PyTorch, C++, stable ABI, StableLibrary, operator, boxed kernel
---

# Stable Operators

The stable API provides tensor operations that maintain binary compatibility
across PyTorch versions.

## Tensor Class

```{doxygenclass} torch::stable::Tensor
:members:
:undoc-members:
```

**Example:**

```cpp
torch::stable::Tensor tensor = torch::stable::empty({3, 4}, ...);
float* data = tensor.data_ptr<float>();
auto shape = tensor.sizes();
```

## Device Class

```{doxygenclass} torch::stable::Device
:members:
:undoc-members:
```

**Example:**

```cpp
torch::stable::Device cpu_device(torch::headeronly::DeviceType::CPU);
torch::stable::Device cuda_device(torch::headeronly::DeviceType::CUDA, 0);
```

## Tensor Creation

```{doxygenfunction} torch::stable::empty
```

```{doxygenfunction} torch::stable::empty_like
```

```{doxygenfunction} torch::stable::new_empty(const torch::stable::Tensor &self, torch::headeronly::IntHeaderOnlyArrayRef size, std::optional<torch::headeronly::ScalarType> dtype, std::optional<torch::headeronly::Layout> layout, std::optional<torch::stable::Device> device, std::optional<bool> pin_memory)
```

```{doxygenfunction} torch::stable::new_zeros(const torch::stable::Tensor &self, torch::headeronly::IntHeaderOnlyArrayRef size, std::optional<torch::headeronly::ScalarType> dtype, std::optional<torch::headeronly::Layout> layout, std::optional<torch::stable::Device> device, std::optional<bool> pin_memory)
```

```{doxygenfunction} torch::stable::full
```

```{doxygenfunction} torch::stable::from_blob(void *data, torch::headeronly::IntHeaderOnlyArrayRef sizes, torch::headeronly::IntHeaderOnlyArrayRef strides, torch::stable::Device device, torch::headeronly::ScalarType dtype, int64_t storage_offset, torch::headeronly::Layout layout)
```

**Example:**

```cpp
auto tensor = torch::stable::empty(
    {3, 4},
    torch::headeronly::ScalarType::Float,
    torch::headeronly::Layout::Strided,
    torch::stable::Device(torch::headeronly::DeviceType::CUDA, 0),
    false,
    torch::headeronly::MemoryFormat::Contiguous);
```

## Tensor Manipulation

```{doxygenfunction} torch::stable::clone
```

```{doxygenfunction} torch::stable::contiguous
```

```{doxygenfunction} torch::stable::reshape
```

```{doxygenfunction} torch::stable::view
```

```{doxygenfunction} torch::stable::flatten
```

```{doxygenfunction} torch::stable::squeeze
```

```{doxygenfunction} torch::stable::unsqueeze
```

```{doxygenfunction} torch::stable::transpose
```

```{doxygenfunction} torch::stable::select
```

```{doxygenfunction} torch::stable::narrow
```

```{doxygenfunction} torch::stable::pad
```

## Device and Type Conversion

```{doxygenfunction} torch::stable::to(const torch::stable::Tensor &self, std::optional<torch::headeronly::ScalarType> dtype, std::optional<torch::headeronly::Layout> layout, std::optional<torch::stable::Device> device, std::optional<bool> pin_memory, bool non_blocking, bool copy, std::optional<torch::headeronly::MemoryFormat> memory_format)
```

```{doxygenfunction} torch::stable::to(const torch::stable::Tensor &self, torch::stable::Device device, bool non_blocking, bool copy)
```

## In-place Operations

```{doxygenfunction} torch::stable::fill_
```

```{doxygenfunction} torch::stable::zero_
```

```{doxygenfunction} torch::stable::copy_
```

## Mathematical Operations

```{doxygenfunction} torch::stable::matmul
```

```{doxygenfunction} torch::stable::amax(const torch::stable::Tensor &self, int64_t dim, bool keepdim)
```

```{doxygenfunction} torch::stable::amax(const torch::stable::Tensor &self, torch::headeronly::IntHeaderOnlyArrayRef dims, bool keepdim)
```

```{doxygenfunction} torch::stable::sum
```

```{doxygenfunction} torch::stable::sum_out
```

```{doxygenfunction} torch::stable::subtract
```
