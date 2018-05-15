#pragma once

#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <cstdint>

namespace torch { namespace nn {
class BatchNorm : public torch::nn::CloneableModule<BatchNorm> {
 public:
  explicit BatchNorm(int64_t features);

  void reset() override;

  variable_list forward(variable_list) override;

  TORCH_PARAMETER(int64_t, features);
  TORCH_PARAMETER(bool, affine) = true;
  TORCH_PARAMETER(bool, stateful) = false;
  TORCH_PARAMETER(double, eps) = 1e-5;
  TORCH_PARAMETER(double, momentum) = 0.1;

 private:
  Variable weights_;
  Variable bias_;
  Variable running_mean_;
  Variable running_variance_;
};
}} // namespace torch::nn
