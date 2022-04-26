#pragma once

#include <torch/csrc/Export.h>

namespace torch {
namespace nn {
namespace utils {

// Clips gradient norm of a vector of Tensors.
// See
// https://pytorch.org/docs/stable/nn.html?highlight=clip_grad_norm#torch.nn.utils.clip_grad_norm_
// for more details about this module.
//
// Difference with the python version: unlike the python version, even when skipping the finiteness
// checks (error_if_nonfinite = false), this function will introduce a device <=> CPU
// synchronization (for devices where that makes sense!) in order to return a CPU-side `double`.
// This C++ version therefore cannot be run fully asynchronously w.r.t. the device of the
// gradients.
inline double clip_grad_norm_(
    std::vector<Tensor> parameters,
    double max_norm,
    double norm_type = 2.0,
    bool error_if_nonfinite = false) {
  std::vector<Tensor> params_with_grad;

  for (const auto& param : parameters) {
    auto& grad = param.grad();
    if (grad.defined()) {
      params_with_grad.push_back(param);
    }
  }

  if (params_with_grad.empty()) {
    return 0.0;
  }

  Tensor total_norm_tensor;
  if (norm_type == std::numeric_limits<double>::infinity()) {
    std::vector<Tensor> norms;
    norms.reserve(params_with_grad.size());

    for (const auto& param : params_with_grad) {
      norms.emplace_back(param.grad().data().abs().max());
    }
    total_norm_tensor = (norms.size() == 1) ? norms[0] : torch::max(torch::stack(norms));
  } else if (norm_type == 0) {
    total_norm_tensor = torch::full({}, static_cast<double>(params_with_grad.size()));
  } else {
    std::vector<Tensor> norms;
    norms.reserve(params_with_grad.size());

    for (const auto& param : params_with_grad) {
      norms.emplace_back(param.grad().data().norm(norm_type));
    }
    total_norm_tensor = (norms.size() == 1) ? norms[0] : torch::stack(norms).norm(norm_type);
  }

  // When possible (ie when skipping the finiteness check), we avoid synchronizing the CPU and the
  // gradients' device until the very end to preserve async execution on the device.
  // When checking for finite-ness, this optional ensures we only sync once.
  c10::optional<double> total_norm = c10::nullopt;
  if (error_if_nonfinite) {
    total_norm = total_norm_tensor.item().toDouble();
    TORCH_CHECK(std::isfinite(*total_norm),
      "The total norm of order ", norm_type, " for gradients from `parameters` ",
      "is non-finite, so it cannot be clipped. To disable this error and scale ",
      "the gradients with the non-finite norm anyway, set ",
      "`error_if_nonfinite=false`");
  }

  auto clip_coef =  max_norm / (total_norm_tensor + 1e-6);
  auto clip_coef_clamped = torch::clamp(clip_coef, c10::nullopt /* min */, 1.0 /* max */);
  for (auto& param : params_with_grad) {
    param.grad().data().mul_(clip_coef_clamped);
  }

  if (!total_norm.has_value()) {
    total_norm = total_norm_tensor.item().toDouble();
  }
  return *total_norm;
}

// A wrapper around clip_grad_norm_ that allows us to call the function with a
// braced-init-list of Tensors.
inline double clip_grad_norm_(
    std::initializer_list<Tensor> parameters,
    double max_norm,
    double norm_type = 2.0,
    bool error_if_nonfinite = false) {
  return clip_grad_norm_(std::vector<Tensor>(parameters), max_norm, norm_type, error_if_nonfinite);
}

// A wrapper around clip_grad_norm_ that allows us to call the function with a
// single Tensor.
inline double clip_grad_norm_(
    Tensor parameter,
    double max_norm,
    double norm_type = 2.0,
    bool error_if_nonfinite = false) {
  std::vector<Tensor> params = {parameter};
  return clip_grad_norm_(params, max_norm, norm_type, error_if_nonfinite);
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
