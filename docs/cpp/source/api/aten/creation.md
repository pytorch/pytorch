---
myst:
  html_meta:
    description: Tensor creation functions in PyTorch C++ — zeros, ones, randn, arange, from_blob, and more.
    keywords: PyTorch, C++, tensor creation, zeros, ones, randn, arange, from_blob
---

# Tensor Creation

Factory functions create new tensors with different initialization patterns.
All factory functions follow a general schema:

```cpp
torch::<function-name>(<function-specific-options>, <sizes>, <tensor-options>)
```

## Available Factory Functions

- `torch::zeros` - Tensor filled with zeros
- `torch::ones` - Tensor filled with ones
- `torch::empty` - Uninitialized tensor
- `torch::full` - Tensor filled with a single value
- `torch::rand` - Uniform random values on [0, 1)
- `torch::randn` - Standard normal distribution
- `torch::randint` - Random integers in a range
- `torch::arange` - Sequence of integers
- `torch::linspace` - Linearly spaced values
- `torch::logspace` - Logarithmically spaced values
- `torch::eye` - Identity matrix
- `torch::randperm` - Random permutation of integers

## Specifying a Size

Functions that do not require specific arguments can be invoked with just a
size. For example, the following line creates a vector with 5 components:

```cpp
torch::Tensor tensor = torch::ones(5);
```

An `IntArrayRef` is constructed by specifying the size along each dimension in
curly braces. For example, `{2, 3}` for a matrix with two rows and three
columns, `{3, 4, 5}` for a three-dimensional tensor:

```cpp
torch::Tensor tensor = torch::randn({3, 4, 5});
assert(tensor.sizes() == std::vector<int64_t>{3, 4, 5});
```

You can also pass an `std::vector<int64_t>` instead of curly braces.
Use `tensor.size(i)` to access a single dimension.

## Passing Function-Specific Parameters

Some factory functions accept additional parameters. For example, `randint`
takes an upper bound on the value for the integers it generates:

```cpp
torch::Tensor tensor = torch::randint(/*high=*/10, {5, 5});

// With a lower bound
torch::Tensor tensor = torch::randint(/*low=*/3, /*high=*/10, {5, 5});
```

```{tip}

The size always follows the function-specific arguments.
```

```{attention}

Some functions like `arange` do not need a size at all, since it is fully
determined by the function-specific arguments (the range bounds).
```

## Configuring Properties with TensorOptions

`TensorOptions` configures the data type, layout, device, and
`requires_grad` of a new tensor. The construction axes are:

- `dtype`: the data type of the elements (e.g. `kFloat32`, `kInt64`)
- `layout`: either `kStrided` (dense) or `kSparse`
- `device`: a compute device (e.g. `kCPU`, `kCUDA`)
- `requires_grad`: whether to track gradients

Allowed values:

- `dtype`: `kUInt8`, `kInt8`, `kInt16`, `kInt32`, `kInt64`,
  `kFloat32`, `kFloat64`
- `layout`: `kStrided`, `kSparse`
- `device`: `kCPU`, or `kCUDA` (with an optional device index)
- `requires_grad`: `true` or `false`

```{tip}

Rust-style shorthands exist for dtypes, like `kF32` instead of
`kFloat32`. See
[torch/types.h](https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/types.h)
for the full list.
```

Here is an example of creating a `TensorOptions` object:

```cpp
auto options =
  torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 1)
    .requires_grad(true);

torch::Tensor tensor = torch::full({3, 4}, /*value=*/123, options);

assert(tensor.dtype() == torch::kFloat32);
assert(tensor.layout() == torch::kStrided);
assert(tensor.device().type() == torch::kCUDA);
assert(tensor.device().index() == 1);
assert(tensor.requires_grad());
```

**Defaults:** Any axis you omit takes its default value: `kFloat32` for dtype,
`kStrided` for layout, `kCPU` for device, and `false` for
`requires_grad`. This means you can omit `TensorOptions` entirely:

```cpp
// A 32-bit float, strided, CPU tensor that does not require a gradient.
torch::Tensor tensor = torch::randn({3, 4});
```

**Shorthand syntax:** For each axis there is a free function in the `torch::`
namespace (`torch::dtype()`, `torch::device()`, `torch::layout()`,
`torch::requires_grad()`). Each returns a `TensorOptions` object that can
be further refined with builder methods:

```cpp
// These are equivalent:
torch::ones(10, torch::TensorOptions().dtype(torch::kFloat32))
torch::ones(10, torch::dtype(torch::kFloat32))

// Chaining:
torch::ones(10, torch::dtype(torch::kFloat32).layout(torch::kStrided))
```

**Implicit construction:** `TensorOptions` is implicitly constructible from
individual values, so when only one axis differs from the default you can
write:

```cpp
torch::ones(10, torch::kFloat32)
```

Putting it all together, a C++ tensor creation call mirrors the Python
equivalent closely:

```python
# Python
torch.randn(3, 4, dtype=torch.float32, device=torch.device('cuda', 1), requires_grad=True)
```

```cpp
// C++
torch::randn({3, 4}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 1).requires_grad(true))
```

## Using Externally Created Data

If you already have tensor data allocated in memory (CPU or CUDA), use
`from_blob` to view that memory as a `Tensor`:

```cpp
float data[] = {1, 2, 3, 4, 5, 6};
torch::Tensor tensor = torch::from_blob(data, {2, 3});
```

```{note}

Tensors created with `from_blob` cannot be resized because ATen does not
own the memory.
```

## Tensor Conversion

Use `to()` to convert tensors between dtypes and devices. The conversion
creates a new tensor and does not occur in-place:

```cpp
torch::Tensor source = torch::randn({2, 3}, torch::kInt64);

// Convert dtype
torch::Tensor float_tensor = source.to(torch::kFloat32);

// Move to GPU (default CUDA device)
torch::Tensor gpu_tensor = float_tensor.to(torch::kCUDA);

// Specific GPU device
torch::Tensor gpu1_tensor = float_tensor.to(torch::Device(torch::kCUDA, 1));

// Async copy
torch::Tensor async_tensor = gpu_tensor.to(torch::kCPU, /*non_blocking=*/true);
```

```{attention}

The result of the conversion is a new tensor pointing to new memory,
unrelated to the source tensor.
```

## Scalars and Zero-Dimensional Tensors

`Scalar` represents a single dynamically-typed number. Like a `Tensor`,
`Scalar` is dynamically typed and can hold any of ATen's number types.
Scalars can be implicitly constructed from C++ number types:

```cpp
namespace torch {
Tensor addmm(Scalar beta, const Tensor & self,
             Scalar alpha, const Tensor & mat1,
             const Tensor & mat2);
Scalar sum(const Tensor & self);
} // namespace torch

// Usage
torch::Tensor a = ...;
torch::Tensor b = ...;
torch::Tensor c = ...;
torch::Tensor r = torch::addmm(1.0, a, .5, b, c);
```

Zero-dimensional tensors hold a single value and can reference elements in
larger tensors:

```cpp
torch::Tensor matrix = torch::rand({10, 20});
matrix[1][2] = 4;  // matrix[1][2] is a zero-dimensional tensor
```
