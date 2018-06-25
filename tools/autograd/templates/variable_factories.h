#pragma once

// ${generated_comment}

#include <torch/csrc/autograd/variable.h>

#include <ATen/ATen.h>
#include <ATen/ArrayRef.h>

#include <initializer_list>
#include <utility>

namespace torch {

#define TENSOR(T, S, _1)                                                    \
  inline Tensor tensor(                                                     \
      at::ArrayRef<T> values, const at::TensorOptions& options) {           \
    at::Tensor result = at::tensor(values, options.discard_runtime_type()); \
    return autograd::make_variable(result, options.requires_grad());        \
  }                                                                         \
  inline Tensor tensor(                                                     \
      std::initializer_list<T> values, const at::TensorOptions& options) {  \
    return torch::tensor(at::ArrayRef<T>(values), options);                 \
  }                                                                         \
  inline Tensor tensor(T value, const at::TensorOptions& options) {         \
    return torch::tensor(at::ArrayRef<T>(value), options);                  \
  }                                                                         \
  inline Tensor tensor(at::ArrayRef<T> values) {                            \
    return torch::tensor(std::move(values), at::dtype(at::k##S));           \
  }                                                                         \
  inline Tensor tensor(std::initializer_list<T> values) {                   \
    return torch::tensor(at::ArrayRef<T>(values));                          \
  }                                                                         \
  inline Tensor tensor(T value) {                                           \
    return torch::tensor(at::ArrayRef<T>(value));                           \
  }
AT_FORALL_SCALAR_TYPES_EXCEPT_HALF(TENSOR)
#undef TENSOR

${function_definitions}
} // namespace torch
