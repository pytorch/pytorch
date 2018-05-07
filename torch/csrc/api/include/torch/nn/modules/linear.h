#pragma once

#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <cstdint>

namespace torch { namespace nn {

class Linear : public torch::nn::CloneableModule<Linear> {
 public:
  Linear(uint32_t nin, uint32_t nout) : nin(nin), nout(nout) {}

  variable_list forward(variable_list) override;
  void reset_parameters() override;
  void initialize_parameters() override;
  AUTOGRAD_KWARG(Linear, bool, no_bias, false, true);

  Variable weight, bias;
  uint32_t nin, nout;
};

}} // namespace torch::nn
