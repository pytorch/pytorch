#pragma once

#include <torch/nn/options/padding.h>

namespace torch {
namespace nn{
namespace functional {

inline Tensor reflection_pad1d(const Tensor& input, const ReflectionPad1dOptions& options) {
  return torch::reflection_pad1d(input, options.padding());
}

} // namespace functional
} // namespace nn
} // namespace torch
