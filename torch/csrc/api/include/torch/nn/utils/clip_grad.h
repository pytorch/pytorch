#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace nn {
namespace utils {

// Clips gradient norm of a vector of Tensors.
// See
// https://pytorch.org/docs/stable/nn.html?highlight=clip_grad_norm#torch.nn.utils.clip_grad_norm_
// for more details about this module.
inline float clip_grad_norm_(
    std::vector<Tensor>& parameters,
    float max_norm,
    float norm_type = 2.0) {
  std::vector<Tensor> params_with_grad;

  for (const auto& param : parameters) {
    auto& grad = param.grad();
    if (grad.defined()) {
      params_with_grad.push_back(param);
    }
  }
  float total_norm = 0.0;
  if (norm_type == std::numeric_limits<float>::infinity()) {
    for (const auto& param : params_with_grad) {
      auto param_max = param.grad().data().abs().max().item().toFloat();
      if (param_max > total_norm) {
        total_norm = param_max;
      }
    }
  } else {
    for (const auto& param : params_with_grad) {
      auto param_norm = param.grad().data().norm(norm_type);
      total_norm += std::pow(param_norm.item().toFloat(), norm_type);
    }
    total_norm = std::pow(total_norm, 1.0 / norm_type);
  }

  auto clip_coef = max_norm / (total_norm + 1e-6);
  if (clip_coef < 1) {
    for (auto& param : params_with_grad) {
      param.grad().data().mul_(clip_coef);
    }
  }
  return total_norm;
}

// A wrapper around clip_grad_norm_ that allows us to call the function with a
// single Tensor.
inline float clip_grad_norm_(
    Tensor& parameters,
    float max_norm,
    float norm_type = 2.0) {
  std::vector<Tensor> params = {parameters};
  return clip_grad_norm_(params, max_norm, norm_type);
}

} // namespace utils
} // namespace nn
} // namespace torch
