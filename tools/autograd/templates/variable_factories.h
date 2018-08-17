#pragma once

// ${generated_comment}

#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>
#include <ATen/core/ArrayRef.h>

#include <functional>
#include <initializer_list>
#include <utility>

namespace torch {

#define TENSOR(T, S, _1)                                                    \
  inline at::Tensor tensor(                                                 \
      at::ArrayRef<T> values, const at::TensorOptions& options) {           \
    at::Tensor result = at::tensor(values, options.discard_runtime_type()); \
    return autograd::make_variable(result, options.requires_grad());        \
  }                                                                         \
  inline at::Tensor tensor(                                                 \
      std::initializer_list<T> values, const at::TensorOptions& options) {  \
    return torch::tensor(at::ArrayRef<T>(values), options);                 \
  }                                                                         \
  inline at::Tensor tensor(T value, const at::TensorOptions& options) {     \
    return torch::tensor(at::ArrayRef<T>(value), options);                  \
  }                                                                         \
  inline at::Tensor tensor(at::ArrayRef<T> values) {                        \
    return torch::tensor(std::move(values), at::dtype(at::k##S));           \
  }                                                                         \
  inline at::Tensor tensor(std::initializer_list<T> values) {               \
    return torch::tensor(at::ArrayRef<T>(values));                          \
  }                                                                         \
  inline at::Tensor tensor(T value) {                                       \
    return torch::tensor(at::ArrayRef<T>(value));                           \
  }
AT_FORALL_SCALAR_TYPES_EXCEPT_HALF(TENSOR)
#undef TENSOR

inline at::Tensor from_blob(
    void* data,
    at::IntList sizes,
    const std::function<void(void*)>& deleter,
    const at::TensorOptions& options = {}) {
  at::Tensor tensor =
      at::from_blob(data, sizes, deleter, options.discard_runtime_type());
  return autograd::make_variable(
      tensor, /*requires_grad=*/options.requires_grad());
}

inline at::Tensor from_blob(
    void* data,
    at::IntList sizes,
    const at::TensorOptions& options = {}) {
  return torch::from_blob(data, sizes, /*deleter=*/[](void*) {}, options);
}

${function_definitions}

} // namespace torch
