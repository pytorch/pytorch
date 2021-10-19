#pragma once

#include <c10/util/Optional.h>

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"
#include "lazy_tensor_core/csrc/reduction.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class NllLoss2dBackward : public TsNode {
 public:
  NllLoss2dBackward(const torch::lazy::Value& grad_output, const torch::lazy::Value& logits,
                    const torch::lazy::Value& labels, const c10::optional<torch::lazy::Value>& weight,
                    const c10::optional<torch::lazy::Value>& total_weight,
                    ReductionMode reduction, int ignore_index);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  ReductionMode reduction() const { return reduction_; }

  int ignore_index() const { return ignore_index_; }

 private:
  ReductionMode reduction_;
  int ignore_index_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
