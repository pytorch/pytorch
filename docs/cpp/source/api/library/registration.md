---
myst:
  html_meta:
    description: Operator registration in PyTorch C++ — TORCH_LIBRARY, TORCH_LIBRARY_IMPL for custom operators.
    keywords: PyTorch, C++, TORCH_LIBRARY, operator registration, dispatch, custom op
---

# Operator Registration

The library API provides macros and classes for registering custom operators
with PyTorch's dispatcher.

## Macros

### TORCH_LIBRARY

```{doxygendefine} TORCH_LIBRARY
```

**Example:**

```cpp
TORCH_LIBRARY(myops, m) {
  m.def("add(Tensor self, Tensor other) -> Tensor", &add_impl);
  m.def("mul(Tensor self, Tensor other) -> Tensor");
  m.impl("mul", torch::kCPU, &mul_cpu_impl);
  m.impl("mul", torch::kCUDA, &mul_cuda_impl);
}
```

### TORCH_LIBRARY_IMPL

```{doxygendefine} TORCH_LIBRARY_IMPL
```

**Example:**

```cpp
TORCH_LIBRARY_IMPL(myops, XLA, m) {
  m.impl("mul", &mul_xla_impl);
}
```

### TORCH_LIBRARY_FRAGMENT

```{doxygendefine} TORCH_LIBRARY_FRAGMENT
```

**Example:**

```cpp
// In file1.cpp
TORCH_LIBRARY(myops, m) {
  m.def("add(Tensor self, Tensor other) -> Tensor", &add_impl);
}

// In file2.cpp
TORCH_LIBRARY_FRAGMENT(myops, m) {
  m.def("mul(Tensor self, Tensor other) -> Tensor", &mul_impl);
}
```

## Classes

### Library

```{doxygenclass} torch::Library
```

**Example:**

```cpp
TORCH_LIBRARY(myops, m) {
  // Define with implementation
  m.def("add(Tensor self, Tensor other) -> Tensor", &add_impl);

  // Define schema only
  m.def("mul(Tensor self, Tensor other) -> Tensor");

  // Provide backend-specific implementations
  m.impl("mul", torch::kCPU, &mul_cpu_impl);
  m.impl("mul", torch::kCUDA, &mul_cuda_impl);
}
```

### CppFunction

```{doxygenclass} torch::CppFunction
:members:
:no-link:
```

### OrderedDict

```{doxygenclass} torch::OrderedDict
:members:
:undoc-members:
```

## Functions

The library API provides builder methods on the `Library` class for registering
operators. See the `Library` class documentation above for the full API including
`def()`, `impl()`, and `fallback()` methods.

## Dispatch Keys

Common dispatch keys used with `torch::dispatch()`:

- `torch::kCPU` - CPU backend
- `torch::kCUDA` - CUDA backend
- `torch::kAutograd` - Autograd backend
- `torch::kMeta` - Meta tensor backend
