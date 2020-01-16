#pragma once

#include <torch/nn/options/dropout.h>

namespace torch {
namespace nn {
namespace functional {

namespace detail {

inline Tensor dropout(Tensor& input, double p, bool training, bool inplace) {
  TORCH_CHECK(
    p >= 0. && p <= 1.,
    "dropout probability has to be between 0 and 1, but got ",
    p);
  if (inplace) {
    return torch::dropout_(input, p, training);
  } else {
    return torch::dropout(input, p, training);
  }
}

} // namespace detail

inline Tensor dropout(Tensor input,
    const DropoutFuncOptions& options = {}) {
  return detail::dropout(
      input, options.p(), options.training(), options.inplace());
}

// ============================================================================

namespace detail {

inline Tensor dropout2d(Tensor& input, double p, bool training, bool inplace) {
  TORCH_CHECK(
    p >= 0. && p <= 1.,
    "dropout probability has to be between 0 and 1, but got ",
    p);
  if (inplace) {
    return torch::feature_dropout_(input, p, training);
  } else {
    return torch::feature_dropout(input, p, training);
  }
}

} // namespace detail

inline Tensor dropout2d(Tensor input,
    const Dropout2dFuncOptions& options = {}) {
  return detail::dropout2d(
      input, options.p(), options.training(), options.inplace());
}

// ============================================================================

namespace detail {

inline Tensor dropout3d(Tensor& input, double p, bool training, bool inplace) {
  TORCH_CHECK(
    p >= 0. && p <= 1.,
    "dropout probability has to be between 0 and 1, but got ",
    p);
  if (inplace) {
    return torch::feature_dropout_(input, p, training);
  } else {
    return torch::feature_dropout(input, p, training);
  }
}

} // namespace detail

inline Tensor dropout3d(Tensor input,
    const Dropout3dFuncOptions& options = {}) {
  return detail::dropout3d(
      input, options.p(), options.training(), options.inplace());
}

// ============================================================================

namespace detail {

inline Tensor alpha_dropout(Tensor input, double p, bool training, bool inplace) {
  if (p < 0. || p > 1.) {
    TORCH_CHECK(false, "dropout probability has to be between 0 and 1, but got ", p);
  }
  return inplace ? torch::alpha_dropout_(input, p, training) : torch::alpha_dropout(input, p, training);
}

} // namespace detail

inline Tensor alpha_dropout(Tensor input, const AlphaDropoutFuncOptions& options = {}) {
  return detail::alpha_dropout(input, options.p(), options.training(), options.inplace());
}

// ============================================================================

namespace detail {

inline Tensor feature_alpha_dropout(Tensor input, double p, bool training, bool inplace) {
  if (p < 0. || p > 1.) {
    TORCH_CHECK(false, "dropout probability has to be between 0 and 1, but got ", p);
  }
  return inplace ? torch::feature_alpha_dropout_(input, p, training) : torch::feature_alpha_dropout(input, p, training);
}

} // namespace detail

inline Tensor feature_alpha_dropout(Tensor input, const FeatureAlphaDropoutFuncOptions& options = {}) {
  return detail::feature_alpha_dropout(input, options.p(), options.training(), options.inplace());
}

} // namespace functional
} // namespace nn
} // namespace torch
