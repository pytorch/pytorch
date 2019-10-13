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

inline Tensor softmin(const Tensor& input, const SoftminOptions& options,
                      c10::optional<torch::Dtype> dtype = c10::nullopt) {
  int64_t dim = options.dim();
  Tensor ret;

  if (dtype == c10::nullopt) {
    ret = (-input).softmax(dim);
  } else {
    ret = (-input).softmax(dim, dtype);
  }

  return ret;
}

inline Tensor log_softmax(const Tensor& input, const LogSoftmaxOptions& options,
                          c10::optional<torch::Dtype> dtype = c10::nullopt) {
  int64_t dim = options.dim();
  Tensor ret;

  if (dtype == c10::nullopt) {
    ret = input.log_softmax(dim);
  } else {
    ret = input.log_softmax(dim, dtype);
  }

  return ret;
}

inline Tensor prelu(const Tensor& input, const Tensor& weight) {
  return torch::prelu(input, weight);
}

inline Tensor relu(Tensor& input, const ReLUOptions& options = {}) {
  if (options.inplace()) {
    return torch::relu_(input);
  } else {
    return torch::relu(input);
  }
}

inline Tensor relu6(Tensor& input, const ReLU6Options& options = {}) {
  return hardtanh(input,
    HardtanhOptions().min_val(0).max_val(6).inplace(options.inplace()));
}

inline Tensor rrelu(Tensor& input, const RReLUOptions& options = {},
                    bool training = false) {
  if (options.inplace()) {
    return torch::rrelu_(input, options.lower(), options.upper(), training);
  } else {
    return torch::rrelu(input, options.lower(), options.upper(), training);
  }
}

inline Tensor celu(Tensor& input, const CELUOptions& options = {}) {
  if (options.inplace()) {
    return torch::celu_(input, options.alpha());
  } else {
    return torch::celu(input, options.alpha());
  }
}

inline Tensor softplus(const Tensor& input,
                       const SoftplusOptions& options = {}) {
  return torch::softplus(input, options.beta(), options.threshold());
}

inline Tensor softshrink(const Tensor& input,
                         const SoftshrinkOptions& options = {}) {
  return torch::softshrink(input, options.lambda());
}

inline Tensor softsign(const Tensor& input) {
  return input / (input.abs() + 1);
}

inline Tensor tanhshrink(const Tensor& input) {
  return input - input.tanh();
}

inline Tensor threshold(Tensor& input, const ThresholdOptions& options) {
  if (options.inplace()) {
    return torch::threshold_(input, options.threshold(), options.value());
  } else {
    return torch::threshold(input, options.threshold(), options.value());
  }
}

} // namespace functional
} // namespace nn
} // namespace torch
