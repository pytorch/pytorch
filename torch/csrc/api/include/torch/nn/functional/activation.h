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

inline int _get_softmax_dim(const std::string& name, int ndim) {
  TORCH_WARN("Implicit dimension choice for ", name, " has been deprecated. "
             "Change the call to include dim=X as an argument.");
  int ret;
  if (ndim == 0 || ndim == 1 || ndim == 3) {
    ret = 0;
  } else {
    ret = 1;
  }
  return ret;
}

inline Tensor softmax(const Tensor& input, const SoftmaxOptions& options) {
  int dim = options.dim();
  torch::Dtype dtype = options.dtype();
  Tensor ret;

  if (dim == -1) {
    dim = _get_softmax_dim("softmax", input.dim());
  }
  if (dtype == torch::Dtype::Undefined) {
    ret = input.softmax(dim);
  } else {
    ret = input.softmax(dim, dtype);
  }

  return ret;
}

} // namespace functional
} // namespace nn
} // namespace torch
