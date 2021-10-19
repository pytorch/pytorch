#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class LinearInterpolation : public TsNode {
 public:
  LinearInterpolation(const torch::lazy::Value& value, const torch::lazy::Value& new_value, double alpha);

  std::string ToString() const override;

  NodePtr Clone(OpList operands) const override;

  double alpha() const { return alpha_; }

 private:
  double alpha_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
