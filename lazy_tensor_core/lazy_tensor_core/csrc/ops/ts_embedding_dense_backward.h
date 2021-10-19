#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class TSEmbeddingDenseBackward : public TsNode {
 public:
  TSEmbeddingDenseBackward(const torch::lazy::Value& grad_output, const torch::lazy::Value& indices,
                           lazy_tensors::int64 num_weights,
                           lazy_tensors::int64 padding_idx,
                           bool scale_grad_by_freq);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  lazy_tensors::int64 num_weights() const { return num_weights_; }

  lazy_tensors::int64 padding_idx() const { return padding_idx_; }

  bool scale_grad_by_freq() const { return scale_grad_by_freq_; }

 private:
  lazy_tensors::int64 num_weights_;
  lazy_tensors::int64 padding_idx_;
  bool scale_grad_by_freq_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
