#pragma once

#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <cstdint>

namespace torch { namespace nn {
class BatchNorm : public torch::nn::CloneableModule<BatchNorm> {
 public:
  BatchNorm(uint32_t num_features) : num_features_(num_features) {}

  AUTOGRAD_KWARG(BatchNorm, double, eps, 1e-5, 1e-5)
  AUTOGRAD_KWARG(BatchNorm, double, momentum, 0.1, 0.1)
  AUTOGRAD_KWARG(BatchNorm, bool, affine, true, true)
  AUTOGRAD_KWARG(BatchNorm, bool, stateful, false, true)

  void reset_parameters() override;
  variable_list forward(variable_list) override;
  void initialize_parameters() override;

  Variable weight;
  Variable bias;
  Variable running_mean;
  Variable running_var;

 protected:
  uint32_t num_features_;
};
}} // namespace torch::nn
