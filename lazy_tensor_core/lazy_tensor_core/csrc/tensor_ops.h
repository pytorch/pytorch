#pragma once

#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensor_core/csrc/tensor.h"

// Certain tensor operations can be expressed in terms of other tensor
// operations. Add their implementations here instead of the main LazyTensor
// class.

namespace torch_lazy_tensors {
namespace tensor_ops {

LazyTensor Cross(const LazyTensor& input, const LazyTensor& other,
                 c10::optional<int64_t> dim);

LazyTensor Select(const LazyTensor& input, int64_t dim, int64_t index);

}  // namespace tensor_ops
}  // namespace torch_lazy_tensors
