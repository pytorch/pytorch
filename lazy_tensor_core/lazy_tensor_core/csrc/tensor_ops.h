#pragma once

#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensor_core/csrc/tensor.h"

// Certain tensor operations can be expressed in terms of other tensor
// operations. Add their implementations here instead of the main LazyTensor
// class.

namespace torch_lazy_tensors {
namespace tensor_ops {

LazyTensor Cross(const LazyTensor& input, const LazyTensor& other,
                 c10::optional<lazy_tensors::int64> dim);

LazyTensor MakeMatrixWithDiagonal(const LazyTensor& input,
                                  lazy_tensors::int64 diagonal);


LazyTensor Select(const LazyTensor& input, lazy_tensors::int64 dim,
                  lazy_tensors::int64 index);

}  // namespace tensor_ops
}  // namespace torch_lazy_tensors
