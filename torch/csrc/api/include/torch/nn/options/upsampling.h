#pragma once

#include <c10/util/variant.h>
#include <torch/arg.h>
#include <torch/enum.h>
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/expanding_array.h>
#include <torch/types.h>

#include <vector>

namespace torch {
namespace nn {

/// Options for a `D`-dimensional interpolate functional.
struct TORCH_API InterpolateOptions {
  /// output spatial sizes.
  TORCH_ARG(std::vector<int64_t>, size) = {};

  /// multiplier for spatial size.
  TORCH_ARG(std::vector<double>, scale_factor) = {};

  /// algorithm used for upsampling. Default: ``nearest``
  typedef c10::variant<
      enumtype::kNearest,
      enumtype::kLinear,
      enumtype::kBilinear,
      enumtype::kBicubic,
      enumtype::kTrilinear,
      enumtype::kArea> mode_t;
  TORCH_ARG(mode_t, mode) = torch::kNearest;

  /// if `true`, the corner pixels of the input and output tensors are
  /// aligned, and thus preserving the values at those pixels. This only has
  /// effect when :attr:`mode` is `Linear`, `Bilinear`, or
  /// `Trilinear`. Default: false
  TORCH_ARG(c10::optional<bool>, align_corners) = c10::nullopt;
};

/// Options for a `D`-dimensional Upsample module.
using UpsampleOptions = InterpolateOptions;

} // namespace nn
} // namespace torch
