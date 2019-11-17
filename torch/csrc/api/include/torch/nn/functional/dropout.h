#pragma once

#include <torch/nn/options/dropout.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor alpha_dropout(Tensor input, const AlphaDropoutOptions& options = {}, bool training = false) {
  TORCH_CHECK(
    options.p() >= 0 && options.p() <= 1,
    "dropout probability has to be between 0 and 1, but got ", options.p()
  );
  
  return options.inplace() ? torch::alpha_dropout_(input, options.p(), training) : torch::alpha_dropout(input, options.p(), training);
}

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

inline Tensor dropout(Tensor& input,
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

inline Tensor dropout2d(Tensor& input,
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

inline Tensor dropout3d(Tensor& input,
    const Dropout3dFuncOptions& options = {}) {
  return detail::dropout3d(
      input, options.p(), options.training(), options.inplace());
}

} // namespace functional
} // namespace nn
} // namespace torch
