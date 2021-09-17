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

LazyTensor KlDivBackward(const LazyTensor& grad_output, const LazyTensor& input,
                         const LazyTensor& target, ReductionMode reduction,
                         bool log_target);

LazyTensor MakeMatrixWithDiagonal(const LazyTensor& input,
                                  lazy_tensors::int64 diagonal);

LazyTensor SmoothL1Loss(const LazyTensor& input, const LazyTensor& target,
                        ReductionMode reduction, double beta);

LazyTensor SmoothL1LossBackward(const LazyTensor& grad_output,
                                const LazyTensor& input,
                                const LazyTensor& target,
                                ReductionMode reduction, double beta);

LazyTensor Softplus(const LazyTensor& input, const at::Scalar& beta,
                    const at::Scalar& threshold);

LazyTensor SoftplusBackward(const LazyTensor& grad_output,
                            const LazyTensor& input, const at::Scalar& beta,
                            const at::Scalar& threshold,
                            const LazyTensor& output);

LazyTensor Select(const LazyTensor& input, lazy_tensors::int64 dim,
                  lazy_tensors::int64 index);

LazyTensor EmbeddingDenseBackward(const LazyTensor& grad_output,
                                  const LazyTensor& indices,
                                  lazy_tensors::int64 num_weights,
                                  lazy_tensors::int64 padding_idx,
                                  bool scale_grad_by_freq);

}  // namespace tensor_ops
}  // namespace torch_lazy_tensors
