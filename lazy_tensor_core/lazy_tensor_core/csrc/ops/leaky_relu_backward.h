#pragma once

#include <c10/core/Scalar.h>

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class LeakyReluBackward : public TsNode {
 public:
  LeakyReluBackward(const torch::lazy::Value& grad_output, const torch::lazy::Value& input,
                    double negative_slope, bool self_is_result = false);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  double negative_slope() const { return negative_slope_; }

  bool self_is_result() const { return self_is_result_; }

 private:
  double negative_slope_;
  bool self_is_result_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
