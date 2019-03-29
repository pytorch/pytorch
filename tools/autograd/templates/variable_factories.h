#pragma once

// ${generated_comment}

#include <ATen/ATen.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/tracer.h>

#include <functional>
#include <initializer_list>
#include <utility>

namespace torch {

#define TENSOR(T, S, _1)                                                   \
  inline at::Tensor tensor(                                                \
      at::ArrayRef<T> values, const at::TensorOptions& options) {          \
    at::Tensor result =                                                    \
        at::tensor(values, at::TensorOptions(options).is_variable(false)); \
    return autograd::make_variable(result, options.requires_grad());       \
  }                                                                        \
  inline at::Tensor tensor(                                                \
      std::initializer_list<T> values, const at::TensorOptions& options) { \
    return torch::tensor(at::ArrayRef<T>(values), options);                \
  }                                                                        \
  inline at::Tensor tensor(T value, const at::TensorOptions& options) {    \
    return torch::tensor(at::ArrayRef<T>(value), options);                 \
  }                                                                        \
  inline at::Tensor tensor(at::ArrayRef<T> values) {                       \
    return torch::tensor(std::move(values), at::dtype(at::k##S));          \
  }                                                                        \
  inline at::Tensor tensor(std::initializer_list<T> values) {              \
    return torch::tensor(at::ArrayRef<T>(values));                         \
  }                                                                        \
  inline at::Tensor tensor(T value) {                                      \
    return torch::tensor(at::ArrayRef<T>(value));                          \
  }
AT_FORALL_SCALAR_TYPES_EXCEPT_HALF(TENSOR)
#undef TENSOR

/// A generic deleter function.
using Deleter = std::function<void(void*)>;

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor, `strides` the
/// stride in each dimension. The `deleter` function (a
/// `std::function<void(void*)>`) will be called on the `data` when the Tensor
/// data would normally be deallocated. The `TensorOptions` specify additional
/// configuration options for the returned tensor, such as what type to
/// interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const Deleter& deleter,
    const at::TensorOptions& options = at::TensorOptions()) {
  at::Tensor tensor =
      at::from_blob(data, sizes, strides, deleter, options.is_variable(false));
  return autograd::make_variable(tensor, options.requires_grad());
}

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor, `strides` the
/// stride in each dimension. The `TensorOptions`
/// specify additional configuration options for the returned tensor, such as
/// what type to interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    const at::TensorOptions& options = at::TensorOptions()) {
  return torch::from_blob(
      data,
      sizes,
      strides,
      /*deleter=*/[](void*) {},
      options);
}

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor. The `deleter`
/// (a `std::function<void(void*)>`) function will be called on the `data` when
/// the Tensor data would normally be deallocated. The `TensorOptions` specify
/// additional configuration options for the returned tensor, such as what type
/// to interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    const Deleter& deleter,
    const at::TensorOptions& options = at::TensorOptions()) {
  at::Tensor tensor =
      at::from_blob(data, sizes, deleter, options.is_variable(false));
  return autograd::make_variable(tensor, options.requires_grad());
}

/// Exposes the given `data` as a `Tensor` without taking ownership of the
/// original data. `sizes` should specify the shape of the tensor. The
/// `TensorOptions` specify additional configuration options for the returned
/// tensor, such as what type to interpret the `data` as.
inline at::Tensor from_blob(
    void* data,
    at::IntArrayRef sizes,
    const at::TensorOptions& options = at::TensorOptions()) {
  return torch::from_blob(data, sizes, /*deleter=*/[](void*) {}, options);
}

${function_definitions}

} // namespace torch
