#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"
#include "lazy_tensor_core/csrc/reduction.h"
#include "lazy_tensors/types.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class L1LossBackward : public TsNode {
 public:
  L1LossBackward(const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
                 const torch::lazy::Value& target, ReductionMode reduction);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  ReductionMode reduction() const { return reduction_; }

 private:
  ReductionMode reduction_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
