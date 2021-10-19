#pragma once

#include "lazy_tensor_core/csrc/ts_backend/TsNode.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class TSNativeBatchNormForward : public TsNode {
 public:
  TSNativeBatchNormForward(const torch::lazy::Value& input, const torch::lazy::Value& weight,
                           const torch::lazy::Value& bias, const torch::lazy::Value& running_mean,
                           const torch::lazy::Value& running_var, bool training,
                           double momentum, double eps);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  bool training() const { return training_; }

  double momentum() const { return momentum_; }

  double eps() const { return eps_; }

 private:
  bool training_;
  double momentum_;
  double eps_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
