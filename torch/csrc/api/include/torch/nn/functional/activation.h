#pragma once

#include <torch/nn/options/activation.h>
#include <torch/types.h>

namespace torch {
namespace nn{
namespace functional {

inline Tensor elu(Tensor& input, const ELUOptions& options = {}) {
  if (options.inplace()) {
    return torch::elu_(input, options.alpha());
  } else {
    return torch::elu(input, options.alpha());
  }
}

inline Tensor selu(Tensor& input, const SELUOptions& options = {}) {
  if (options.inplace()) {
    return torch::selu_(input);
  } else {
    return torch::selu(input);
  }
}

inline Tensor hardshrink(const Tensor& input,
                         const HardshrinkOptions& options = {}) {
  return torch::hardshrink(input, options.lambda());
}

inline Tensor hardtanh(Tensor& input, const HardtanhOptions& options = {}) {
  if (options.inplace()) {
    return torch::hardtanh_(input, options.min_val(), options.max_val());
  } else {
    return torch::hardtanh(input, options.min_val(), options.max_val());
  }
}

inline Tensor leaky_relu(Tensor& input, const LeakyReLUOptions& options = {}) {
  if (options.inplace()) {
    return torch::leaky_relu_(input, options.negative_slope());
  } else {
    return torch::leaky_relu(input, options.negative_slope());
  }
}

inline Tensor logsigmoid(const Tensor& input) {
  return torch::log_sigmoid(input);
}

inline Tensor softmax(const Tensor& input, const SoftmaxOptions& options,
                      c10::optional<torch::Dtype> dtype = c10::nullopt) {
  int64_t dim = options.dim();
  Tensor ret;

  if (dtype == c10::nullopt) {
    ret = input.softmax(dim);
  } else {
    ret = input.softmax(dim, dtype);
  }

  return ret;
}

inline Tensor prelu(const Tensor& input, const Tensor& weight) {
  return torch::prelu(input, weight);
}

} // namespace functional
} // namespace nn
} // namespace torch
