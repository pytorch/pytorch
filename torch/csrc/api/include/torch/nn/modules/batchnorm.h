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

  TORCH_ATTR(int64_t, features);
  TORCH_ATTR(bool, affine) = true;
  TORCH_ATTR(bool, stateful) = false;
  TORCH_ATTR(double, eps) = 1e-5;
  TORCH_ATTR(double, momentum) = 0.1;
  TORCH_ATTR(Variable, weight);
  TORCH_ATTR(Variable, bias);
  TORCH_ATTR(Variable, running_mean);
  TORCH_ATTR(Variable, running_variance);
};
}} // namespace torch::nn
