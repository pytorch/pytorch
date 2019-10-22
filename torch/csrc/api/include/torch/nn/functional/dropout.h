#pragma once

#include <torch/nn/options/dropout.h>

namespace torch {
namespace nn {
namespace functional {

using AlphaDropoutOptions = DropoutOptions;

inline Tensor alpha_dropout(const Tensor& input, const AlphaDropoutOptions& options = {}, bool training = false) {
  TORCH_CHECK(options.p() >= 0, "Dropout rate must not be less than zero");
  TORCH_CHECK(options.p() <= 1, "Dropout rate must not be greater than one");
  
  return torch::alpha_dropout(input, options.p(), training);
}
} // namespace functional
} // namespace nn
} // namespace torch
