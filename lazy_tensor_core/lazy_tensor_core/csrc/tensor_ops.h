#pragma once

#include "lazy_tensor_core/csrc/tensor.h"

// Certain tensor operations can be expressed in terms of other tensor
// operations. Add their implementations here instead of the main LazyTensor
// class.

namespace torch_lazy_tensors {
namespace tensor_ops {

LazyTensor Select(const LazyTensor& input, int64_t dim, int64_t index);

}  // namespace tensor_ops
}  // namespace torch_lazy_tensors
