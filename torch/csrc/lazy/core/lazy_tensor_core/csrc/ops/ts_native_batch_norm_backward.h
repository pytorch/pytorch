#pragma once

#include <array>

#include "lazy_tensor_core/csrc/ir.h"

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// Node for the backward batch norm operator.
class TSNativeBatchNormBackward : public Node {
 public:
  TSNativeBatchNormBackward(const Value& grad_out, const Value& input,
                            const Value& weight, const Value& running_mean,
                            const Value& running_var, const Value& save_mean,
                            const Value& save_invstd, bool training, double eps,
                            std::array<bool, 3> output_mask);

  TSNativeBatchNormBackward(const Value& grad_out, const Value& input,
                            const Value& weight, const Value& save_mean,
                            const Value& save_invstd, bool training, double eps,
                            std::array<bool, 3> output_mask);

  NodePtr Clone(OpList operands) const override;

  std::string ToString() const override;

  bool training() const { return training_; }

  double eps() const { return eps_; }

  const std::array<bool, 3>& output_mask() const { return output_mask_; }

 private:
  bool training_;
  double eps_;
  std::array<bool, 3> output_mask_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_lazy_tensors
