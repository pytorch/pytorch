#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

#include <array>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// Node for the backward batch norm operator.
class TSNativeBatchNormBackward : public torch::lazy::TsNode {
 public:
  TSNativeBatchNormBackward(const torch::lazy::Value& grad_out, const torch::lazy::Value& input,
                            const torch::lazy::Value& weight, const torch::lazy::Value& running_mean,
                            const torch::lazy::Value& running_var, const torch::lazy::Value& save_mean,
                            const torch::lazy::Value& save_invstd, bool training, double eps,
                            std::array<bool, 3> output_mask);

  TSNativeBatchNormBackward(const torch::lazy::Value& grad_out, const torch::lazy::Value& input,
                            const torch::lazy::Value& weight, const torch::lazy::Value& save_mean,
                            const torch::lazy::Value& save_invstd, bool training, double eps,
                            std::array<bool, 3> output_mask);

  bool Equal(const torch::lazy::Value& grad_out,
             const torch::lazy::Value& input, const torch::lazy::Value& weight,
             const torch::lazy::Value& running_mean,
             const torch::lazy::Value& running_var,
             const torch::lazy::Value& save_mean,
             const torch::lazy::Value& save_invstd, bool training, double eps,
             std::array<bool, 3> output_mask) const {
    size_t i = 0;
    return (operand(i++) == grad_out && operand(i++) == input &&
            operand(i++) == weight && operand(i++) == running_mean &&
            operand(i++) == running_var && operand(i++) == save_mean &&
            operand(i++) == save_invstd && training_ == training &&
            eps_ == eps && output_mask_ == output_mask);
  }

  bool Equal(const torch::lazy::Value& grad_out,
             const torch::lazy::Value& input, const torch::lazy::Value& weight,
             const torch::lazy::Value& save_mean,
             const torch::lazy::Value& save_invstd, bool training, double eps,
             std::array<bool, 3> output_mask) const {
    size_t i = 0;
    return (operand(i++) == grad_out && operand(i++) == input &&
            operand(i++) == weight && operand(i++) == save_mean &&
            operand(i++) == save_invstd && training_ == training &&
            eps_ == eps && output_mask_ == output_mask);
  }

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
