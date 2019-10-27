#pragma once

// ${generated_comment}

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/util/ArrayRef.h>
#include <c10/core/MemoryFormat.h>
#include <ATen/core/EnableNamedTensor.h>
#include <torch/csrc/api/include/torch/detail/TensorDataContainer.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/jit/tracer.h>
#include <torch/csrc/jit/ir.h>

#include <functional>
#include <initializer_list>
#include <utility>

#ifdef BUILD_NAMEDTENSOR
using at::DimnameList;
#endif

namespace torch {

/// NOTE: Currently `torch::tensor(...)` doesn't support mixed data types
/// (i.e. `torch::tensor({{bool, 2.0}})` doesn't work). We might be able to
/// support it in the future by iterating over all sub-lists to find
/// the largest data type that can represent all of the elements, or by using
/// variadic templates.
inline at::Tensor tensor(detail::TensorDataContainer init_list_tensor, const at::TensorOptions& options = {}) {
  return autograd::make_variable(
    init_list_tensor.convert_to_tensor(options.has_dtype() ? options : options.dtype(init_list_tensor.scalar_type())),
    options.requires_grad());
}

/// A generic deleter function.
using Deleter = std::function<void(void*)>;
using at::MemoryFormat;

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
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::from_blob(data, sizes, strides, deleter, options.is_variable(false));
  })();
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
  at::Tensor tensor = ([&]() {
    at::AutoNonVariableTypeMode non_var_type_mode(true);
    return at::from_blob(data, sizes, deleter, options.is_variable(false));
  })();
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
