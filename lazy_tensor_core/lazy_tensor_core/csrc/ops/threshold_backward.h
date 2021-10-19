#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class ThresholdBackward : public TsNode {
 public:
  ThresholdBackward(const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
                    float threshold);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  float threshold() const { return threshold_; }

 private:
  float threshold_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
