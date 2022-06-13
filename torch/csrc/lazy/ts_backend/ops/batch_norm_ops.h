#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch {
namespace lazy {

// Node for the backward batch norm operator.
class TSNativeBatchNormBackward : public torch::lazy::TsNode {
 public:
  static OpKind ClassOpKind() {
    return OpKind(at::aten::native_batch_norm_backward);
  }

  TSNativeBatchNormBackward(
      const torch::lazy::Value& grad_out,
      const torch::lazy::Value& input,
      const torch::lazy::Value& weight,
      const torch::lazy::Value& running_mean,
      const torch::lazy::Value& running_var,
      const torch::lazy::Value& save_mean,
      const torch::lazy::Value& save_invstd,
      bool training,
      double eps,
      std::array<bool, 3> output_mask);

  TSNativeBatchNormBackward(
      const torch::lazy::Value& grad_out,
      const torch::lazy::Value& input,
      const torch::lazy::Value& weight,
      const torch::lazy::Value& save_mean,
      const torch::lazy::Value& save_invstd,
      bool training,
      double eps,
      std::array<bool, 3> output_mask);

  bool CanBeReused(
      const torch::lazy::Value& grad_out,
      const torch::lazy::Value& input,
      const torch::lazy::Value& weight,
      const torch::lazy::Value& running_mean,
      const torch::lazy::Value& running_var,
      const torch::lazy::Value& save_mean,
      const torch::lazy::Value& save_invstd,
      bool training,
      double eps,
      std::array<bool, 3> output_mask) const {
    size_t i = 0;
    return (
        operand(i++) == grad_out && operand(i++) == input &&
        operand(i++) == weight && operand(i++) == running_mean &&
        operand(i++) == running_var && operand(i++) == save_mean &&
        operand(i++) == save_invstd && training_ == training && eps_ == eps &&
        output_mask_ == output_mask);
  }

  bool CanBeReused(
      const torch::lazy::Value& grad_out,
      const torch::lazy::Value& input,
      const torch::lazy::Value& weight,
      const torch::lazy::Value& save_mean,
      const torch::lazy::Value& save_invstd,
      bool training,
      double eps,
      std::array<bool, 3> output_mask) const {
    size_t i = 0;
    return (
        operand(i++) == grad_out && operand(i++) == input &&
        operand(i++) == weight && operand(i++) == save_mean &&
        operand(i++) == save_invstd && training_ == training && eps_ == eps &&
        output_mask_ == output_mask);
  }

  std::string ToString() const override;

  bool training() const {
    return training_;
  }

  double eps() const {
    return eps_;
  }

  const std::array<bool, 3>& output_mask() const {
    return output_mask_;
  }

  TSOpVector Lower(
      std::shared_ptr<torch::jit::GraphFunction> function,
      TSLoweringContext* loctx) const override;

 private:
  bool training_;
  double eps_;
  std::array<bool, 3> output_mask_;
};

class TSNativeBatchNormForward : public torch::lazy::TsNode {
 public:
  static OpKind ClassOpKind() {
    return OpKind(at::aten::native_batch_norm);
  }

  TSNativeBatchNormForward(
      const torch::lazy::Value& input,
      const torch::lazy::Value& weight,
      const torch::lazy::Value& bias,
      const torch::lazy::Value& running_mean,
      const torch::lazy::Value& running_var,
      bool training,
      double momentum,
      double eps);

  bool CanBeReused(
      const torch::lazy::Value& input,
      const torch::lazy::Value& weight,
      const torch::lazy::Value& bias,
      const torch::lazy::Value& running_mean,
      const torch::lazy::Value& running_var,
      bool training,
      double momentum,
      double eps) const {
    size_t i = 0;
    return (
        operand(i++) == input && operand(i++) == weight &&
        operand(i++) == bias && operand(i++) == running_mean &&
        operand(i++) == running_var && training_ == training &&
        momentum_ == momentum && eps == eps_);
  }

  std::string ToString() const override;

  bool training() const {
    return training_;
  }

  double momentum() const {
    return momentum_;
  }

  double eps() const {
    return eps_;
  }

  TSOpVector Lower(
      std::shared_ptr<torch::jit::GraphFunction> function,
      TSLoweringContext* loctx) const override;

 private:
  bool training_;
  double momentum_;
  double eps_;
};

} // namespace lazy
} // namespace torch
