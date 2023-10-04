#pragma once

#include <ATen/Context.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/csrc/jit/runtime/vararg_functions.h>

namespace torch {
namespace jit {

inline void noop(Stack& n) {}

int64_t normalizeIndex(int64_t idx, int64_t list_size);

// reference function THPVariable_to in python_variable_methods.cpp
static C10_UNUSED at::Tensor to_dispatch(
    at::Tensor self,
    c10::optional<at::Device> device,
    c10::optional<at::ScalarType> scalarType,
    bool non_blocking,
    bool copy) {
  if (device && device->is_cuda()) {
    at::globalContext().lazyInitCUDA();
  }
  if (!device && !scalarType && !copy) {
    return self;
  } else if (!device) {
    return self.to(*scalarType, non_blocking, copy);
  } else if (!scalarType) {
    return self.to(*device, non_blocking, copy);
  } else {
    return self.to(*device, *scalarType, non_blocking, copy);
  }
}

// Convert the tensor pointed to by \p data to a nested list. \p dim is the
// number of dimensions in the tensor and \p cur_dim is the dimension being
// processed by the current invocation. \p ty is the expected output IR type of
// the operation. \p is the scalar type of \p data. \p sizes and \p strides are
// the sizes and strides of the tensor operand and \p element_size is the size
// in bytes of one tensor element.
IValue tensorToListRecursive(
    char* data,
    int64_t cur_dim,
    int64_t num_tensor_dims,
    at::TypePtr ty,
    at::ScalarType scalar_ty,
    at::IntArrayRef sizes,
    at::IntArrayRef strides,
    size_t element_size);

} // namespace jit
} // namespace torch
