#pragma once

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

class NativeBatchNormForward : public Node {
 public:
  NativeBatchNormForward(const Value& input, const Value& weight,
                         const Value& bias, const Value& running_mean,
                         const Value& running_var, bool training, double eps);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  bool training() const { return training_; }

  double eps() const { return eps_; }

 private:
  bool training_;
  double eps_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
