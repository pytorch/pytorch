#pragma once

#include <torch/nn/options/activation.h>
#include <torch/types.h>

namespace torch {
namespace nn{
namespace functional {

inline Tensor elu(Tensor& input, const ELUOptions& options) {
  if (options.inplace()) {
    return torch::elu_(input, options.alpha());
  } else {
    return torch::elu(input, options.alpha());
  }
}

inline Tensor hardshrink(const Tensor& input,
                         const HardshrinkOptions& options) {
  return torch::hardshrink(input, options.lambda());
}

inline Tensor hardtanh(Tensor& input, const HardtanhOptions& options) {
  if (options.inplace()) {
    return torch::hardtanh_(input, options.min_val(), options.max_val());
  } else {
    return torch::hardtanh(input, options.min_val(), options.max_val());
  }
}

inline Tensor leaky_relu(Tensor& input, const LeakyReLUOptions& options) {
  if (options.inplace()) {
    return torch::leaky_relu_(input, options.negative_slope());
  } else {
    return torch::leaky_relu(input, options.negative_slope());
  }
}

inline Tensor logsigmoid(const Tensor& input) {
  return torch::log_sigmoid(input);
}

inline Tensor gumbel_softmax(const Tensor& logits, const GumbelSoftmaxOptions& options = {}) {
    auto gumbels = -torch::empty_like(logits).exponential_().log();  // ~Gumbel(0,1)
    gumbels = (logits + gumbels) / options.tau();  // ~Gumbel(logits, tau)
    auto y_soft = gumbels.softmax(options.dim());

    torch::Tensor ret;
    if (options.hard()) {
        // Straight through.
        auto index = std::get<1>(y_soft.max(options.dim(), /*keepdim=*/true));
        auto y_hard = torch::zeros_like(logits).scatter_(options.dim(), index, 1.0);
        ret = y_hard - y_soft.detach() + y_soft;
    } else {
        ret = y_soft;
    }
    return ret;
}

} // namespace functional
} // namespace nn
} // namespace torch
