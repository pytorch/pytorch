#pragma once

#include <torch/nn/options/activation.h>
#include <torch/types.h>

namespace torch {
namespace nn{
namespace functional {

inline Tensor elu(Tensor& input, const ELUFuncOptions& options = {}) {
  if (options.inplace()) {
    return torch::elu_(input, options.alpha());
  } else {
    return torch::elu(input, options.alpha());
  }
}

inline Tensor selu(Tensor& input, const SELUFuncOptions& options = {}) {
  if (options.inplace()) {
    return torch::selu_(input);
  } else {
    return torch::selu(input);
  }
}

inline Tensor hardshrink(const Tensor& input,
                         const HardshrinkFuncOptions& options = {}) {
  return torch::hardshrink(input, options.lambda());
}

inline Tensor hardtanh(Tensor& input, const HardtanhFuncOptions& options = {}) {
  if (options.inplace()) {
    return torch::hardtanh_(input, options.min_val(), options.max_val());
  } else {
    return torch::hardtanh(input, options.min_val(), options.max_val());
  }
}

inline Tensor leaky_relu(Tensor& input, const LeakyReLUFuncOptions& options = {}) {
  if (options.inplace()) {
    return torch::leaky_relu_(input, options.negative_slope());
  } else {
    return torch::leaky_relu(input, options.negative_slope());
  }
}

inline Tensor logsigmoid(const Tensor& input) {
  return torch::log_sigmoid(input);
}

inline Tensor gumbel_softmax(const Tensor& logits, const GumbelSoftmaxFuncOptions& options = {}) {
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

inline Tensor softmax(const Tensor& input, const SoftmaxFuncOptions& options) {
  int64_t dim = options.dim();
  Tensor ret;

  if (dtype == c10::nullopt) {
    ret = input.softmax(dim);
  } else {
    ret = input.softmax(dim, dtype);
  }

  return ret;
}

inline Tensor softmin(const Tensor& input, const SoftminFuncOptions& options) {
  int64_t dim = options.dim();
  Tensor ret;

  if (dtype == c10::nullopt) {
    ret = (-input).softmax(dim);
  } else {
    ret = (-input).softmax(dim, dtype);
  }

  return ret;
}

inline Tensor log_softmax(const Tensor& input, const LogSoftmaxFuncOptions& options) {
  int64_t dim = options.dim();
  Tensor ret;

  if (dtype == c10::nullopt) {
    ret = input.log_softmax(dim);
  } else {
    ret = input.log_softmax(dim, dtype);
  }

  return ret;
}

inline Tensor gelu(const Tensor& input) {
  return torch::gelu(input);
}

inline Tensor prelu(const Tensor& input, const Tensor& weight) {
  return torch::prelu(input, weight);
}

inline Tensor relu(Tensor& input, const ReLUFuncOptions& options = {}) {
  if (options.inplace()) {
    return torch::relu_(input);
  } else {
    return torch::relu(input);
  }
}

inline Tensor relu6(Tensor& input, const ReLU6FuncOptions& options = {}) {
  return hardtanh(input,
    HardtanhOptions().min_val(0).max_val(6).inplace(options.inplace()));
}

inline Tensor rrelu(Tensor& input, const RReLUFuncOptions& options = {}) {
  if (options.inplace()) {
    return torch::rrelu_(input, options.lower(), options.upper(), training);
  } else {
    return torch::rrelu(input, options.lower(), options.upper(), training);
  }
}

inline Tensor celu(Tensor& input, const CELUFuncOptions& options = {}) {
  if (options.inplace()) {
    return torch::celu_(input, options.alpha());
  } else {
    return torch::celu(input, options.alpha());
  }
}

inline Tensor softplus(const Tensor& input,
                       const SoftplusFuncOptions& options = {}) {
  return torch::softplus(input, options.beta(), options.threshold());
}

inline Tensor softshrink(const Tensor& input,
                         const SoftshrinkFuncOptions& options = {}) {
  return torch::softshrink(input, options.lambda());
}

inline Tensor softsign(const Tensor& input) {
  return input / (input.abs() + 1);
}

inline Tensor tanhshrink(const Tensor& input) {
  return input - input.tanh();
}

inline Tensor threshold(Tensor& input, const ThresholdFuncOptions& options) {
  if (options.inplace()) {
    return torch::threshold_(input, options.threshold(), options.value());
  } else {
    return torch::threshold(input, options.threshold(), options.value());
  }
}

} // namespace functional
} // namespace nn
} // namespace torch
