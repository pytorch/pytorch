#pragma once

#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <cstdint>

namespace torch {
namespace nn {

class Linear : public torch::nn::CloneableModule<Linear> {
 public:
  Linear(size_t features_in, size_t features_out);

  void reset() override;

  std::vector<Variable> forward(std::vector<Variable>);

  TORCH_ATTR(int64_t, in);
  TORCH_ATTR(int64_t, out);
  TORCH_ATTR(bool, with_bias) = true;
  TORCH_ATTR(Variable, weight);
  TORCH_ATTR(Variable, bias);
};

} // namespace nn
} // namespace torch
