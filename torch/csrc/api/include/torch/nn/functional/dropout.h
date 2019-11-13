#pragma once

#include <torch/nn/options/dropout.h>

namespace torch {
namespace nn {
namespace functional {

using AlphaDropoutOptions = DropoutOptions;

inline Tensor alpha_dropout(Tensor input, const AlphaDropoutOptions& options = {}, bool training = false) {
  TORCH_CHECK(
    options.p() >= 0 && options.p() <= 1,
    "dropout probability has to be between 0 and 1, but got ", options.p()
  );
  
  return options.inplace() ? torch::alpha_dropout_(input, options.p(), training) : torch::alpha_dropout(input, options.p(), training);
}
} // namespace functional
} // namespace nn
} // namespace torch
