#pragma once

#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <cstdint>

namespace torch { namespace nn {
class BatchNorm : public torch::nn::CloneableModule<BatchNorm> {
 public:
  explicit BatchNorm(uint32_t num_features);

  TORCH_AUTOGRAD_KWARG(BatchNorm, double, eps, 1e-5, 1e-5)
  TORCH_AUTOGRAD_KWARG(BatchNorm, double, momentum, 0.1, 0.1)
  TORCH_AUTOGRAD_KWARG(BatchNorm, bool, affine, true, true)
  TORCH_AUTOGRAD_KWARG(BatchNorm, bool, stateful, false, true)

  variable_list forward(variable_list) override;

  Variable weight;
  Variable bias;
  Variable running_mean;
  Variable running_var;

 protected:
  uint32_t num_features_;
};
}} // namespace torch::nn
