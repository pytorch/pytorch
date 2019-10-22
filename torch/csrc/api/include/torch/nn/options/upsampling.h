#pragma once

#include <torch/arg.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/expanding_array.h>
#include <torch/types.h>

#include <vector>

namespace torch {

enum class Interpolation {
  Nearest,
  Linear,
  Bilinear,
  Bicubic,
  Trilinear,
  Area
};

namespace nn {

/// Options for a `D`-dimensional interpolate functional.
struct TORCH_API InterpolationOptions {
  /// output spatial sizes.
  TORCH_ARG(c10::optional<std::vector<int64_t>>, size) = c10::nullopt;

  /// multiplier for spatial size.
  TORCH_ARG(c10::optional<std::vector<double>>, scale_factor) = c10::nullopt;

  /// algorithm used for upsampling. Default: ``nearest``
  TORCH_ARG(Interpolation, mode) = Interpolation::Nearest;

  /// if `true`, the corner pixels of the input and output tensors are
  /// aligned, and thus preserving the values at those pixels. This only has
  /// effect when :attr:`mode` is `Linear`, `Bilinear`, or
  /// `Trilinear`. Default: false
  TORCH_ARG(c10::optional<bool>, align_corners) = c10::nullopt;
};

/// Options for a `D`-dimensional Upsample module.
using UpsampleOptions = InterpolationOptions;

} // namespace nn
} // namespace torch
