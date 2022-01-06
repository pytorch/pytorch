#pragma once

#include <torch/csrc/lazy/core/tensor.h>

// Certain tensor operations can be expressed in terms of other tensor
// operations. Add their implementations here instead of the main torch::lazy::LazyTensor
// class.

namespace torch_lazy_tensors {
namespace tensor_ops {

torch::lazy::LazyTensor Select(const torch::lazy::LazyTensor& input, int64_t dim, int64_t index);

}  // namespace tensor_ops
}  // namespace torch_lazy_tensors
