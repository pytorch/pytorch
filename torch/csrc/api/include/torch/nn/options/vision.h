#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/types.h>

namespace torch {
namespace nn {
namespace functional {

/// Options for a Grid Sample module.
struct TORCH_API GridSampleFuncOptions {
  /// interpolation mode to calculate output values. Default: Bilinear
  TORCH_ARG(std::string, mode) = "bilinear";
  /// padding mode for outside grid values. Default: Zeros
  TORCH_ARG(std::string, padding_mode) = "zeros";
  /// Specifies perspective to pixel as point. Default: false
  TORCH_ARG(c10::optional<bool>, align_corners) = c10::nullopt;
};

} // namespace functional
} // namespace nn
} // namespace torch
