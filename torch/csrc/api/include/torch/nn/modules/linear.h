#pragma once

#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <cstdint>

namespace torch { namespace nn {

class Linear : public torch::nn::CloneableModule<Linear> {
 public:
  Linear(uint32_t nin, uint32_t nout);

  variable_list forward(variable_list) override;
  TORCH_AUTOGRAD_KWARG(Linear, bool, no_bias, false, true);

  uint32_t nin, nout;
  Variable weight, bias;
};

}} // namespace torch::nn
