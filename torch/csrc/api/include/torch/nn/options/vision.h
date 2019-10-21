#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>
#include <ATen/native/GridSampler.h>

namespace torch {

using torch::native::detail::GridSamplerInterpolation;
using torch::native::detail::GridSamplerPadding;

namespace nn {

/// Options for a Grid Sample module.
struct TORCH_API GridSampleOptions {
  /// interpolation mode to calculate output values. Default: Bilinear
  TORCH_ARG(torch::GridSamplerInterpolation, mode) = torch::GridSamplerInterpolation::Bilinear;
  /// padding mode for outside grid values. Default: Zeros
  TORCH_ARG(torch::GridSamplerPadding, padding_mode) = torch::GridSamplerPadding::Zeros;
  /// Specifies perspective to pixel as point. Default: true
  TORCH_ARG(c10::optional<bool>, align_corners) = c10::nullopt;
};

} // namespace nn
} // namespace torch