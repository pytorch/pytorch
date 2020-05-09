#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace nn {
namespace utils {

// Clips gradient norm of a vector of Tensors.
// See
// https://pytorch.org/docs/stable/nn.html?highlight=clip_grad_norm#torch.nn.utils.clip_grad_norm_
// for more details about this module.
inline double clip_grad_norm_(
    std::vector<Tensor> parameters,
    double max_norm,
    double norm_type = 2.0) {
  std::vector<Tensor> params_with_grad;

  for (const auto& param : parameters) {
    auto& grad = param.grad();
    if (grad.defined()) {
      params_with_grad.push_back(param);
    }
  }
  double total_norm = 0.0;
  if (norm_type == std::numeric_limits<double>::infinity()) {
    for (const auto& param : params_with_grad) {
      auto param_max = param.grad().data().abs().max().item().toDouble();
      if (param_max > total_norm) {
        total_norm = param_max;
      }
    }
  } else {
    for (const auto& param : params_with_grad) {
      auto param_norm = param.grad().data().norm(norm_type);
      total_norm += std::pow(param_norm.item().toDouble(), norm_type);
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
// braced-init-list of Tensors.
inline double clip_grad_norm_(
    std::initializer_list<Tensor> parameters,
    double max_norm,
    double norm_type = 2.0) {
  return clip_grad_norm_(std::vector<Tensor>(parameters), max_norm, norm_type);
}

// A wrapper around clip_grad_norm_ that allows us to call the function with a
// single Tensor.
inline double clip_grad_norm_(
    Tensor parameter,
    double max_norm,
    double norm_type = 2.0) {
  std::vector<Tensor> params = {parameter};
  return clip_grad_norm_(params, max_norm, norm_type);
}

// Clips gradient of an iterable of parameters at specified value.
// Gradients are modified in-place.
// See https://pytorch.org/docs/stable/nn.html#clip-grad-value
// for more details about this module.
inline void clip_grad_value_(
    std::vector<Tensor> parameters,
    double clip_value) {

  for (const auto& param : parameters) {
    if (param.grad().defined()) {
      param.grad().data().clamp_(-clip_value, clip_value);
    }
  }
}

// A wrapper around clip_grad_value_ that allows us to call the function with a
// braced-init-list of Tensors.
inline void clip_grad_value_(std::initializer_list<Tensor> parameters, double clip_value) {
  clip_grad_value_(std::vector<Tensor>(parameters), clip_value);
}

// A wrapper around clip_grad_value_ that allows us to call the function with a
// single Tensor.
inline void clip_grad_value_(Tensor parameter, double clip_value) {
  std::vector<Tensor> params = {parameter};
  clip_grad_value_(params, clip_value);
}

} // namespace utils
} // namespace nn
} // namespace torch
