#pragma once

#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <ATen/optional.h>

#include <cstdint>

namespace torch { namespace nn {

class Linear : public torch::nn::CloneableModule<Linear> {
 public:
  Linear(uint32_t nin, uint32_t nout, bool with_bias = true);

  variable_list forward(variable_list) override;

  uint32_t nin, nout;
  Variable weight;
  at::optional<Variable> bias;
};

}} // namespace torch::nn
