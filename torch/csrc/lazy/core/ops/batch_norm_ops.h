#pragma once

#include <torch/csrc/lazy/backend/backend_node.h>

namespace torch {
namespace lazy {

// Node for the backward batch norm operator.
class NativeBatchNormBackward : public torch::lazy::BackendNode {
 public:
  NativeBatchNormBackward(const torch::lazy::Value& grad_out, const torch::lazy::Value& input,
                            const torch::lazy::Value& weight, const torch::lazy::Value& running_mean,
                            const torch::lazy::Value& running_var, const torch::lazy::Value& save_mean,
                            const torch::lazy::Value& save_invstd, bool training, double eps,
                            std::array<bool, 3> output_mask);

  NativeBatchNormBackward(const torch::lazy::Value& grad_out, const torch::lazy::Value& input,
                            const torch::lazy::Value& weight, const torch::lazy::Value& save_mean,
                            const torch::lazy::Value& save_invstd, bool training, double eps,
                            std::array<bool, 3> output_mask);

  std::string ToString() const override;

  bool training() const { return training_; }

  double eps() const { return eps_; }

  const std::array<bool, 3>& output_mask() const { return output_mask_; }

 private:
  bool training_;
  double eps_;
  std::array<bool, 3> output_mask_;
};

class NativeBatchNormForward : public torch::lazy::BackendNode {
 public:
  NativeBatchNormForward(const torch::lazy::Value& input, const torch::lazy::Value& weight,
                           const torch::lazy::Value& bias, const torch::lazy::Value& running_mean,
                           const torch::lazy::Value& running_var, bool training,
                           double momentum, double eps);

  std::string ToString() const override;

  bool training() const { return training_; }

  double momentum() const { return momentum_; }

  double eps() const { return eps_; }

 private:
  bool training_;
  double momentum_;
  double eps_;
};

}  // namespace lazy
}  // namespace torch
