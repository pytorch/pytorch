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

/// Options for a `D`-dimensional Upsample module.
struct TORCH_API UpsampleOptions {
  /// output spatial sizes.
  TORCH_ARG(std::vector<int64_t>, size) = {};

  /// multiplier for spatial size.
  TORCH_ARG(std::vector<double>, scale_factor) = {};

  /// the upsampling algorithm: one of "nearest", "linear", "bilinear",
  /// "bicubic" and "trilinear". Default: "nearest"
  typedef c10::variant<
      enumtype::kNearest,
      enumtype::kLinear,
      enumtype::kBilinear,
      enumtype::kBicubic,
      enumtype::kTrilinear> mode_t;
  TORCH_ARG(mode_t, mode) = torch::kNearest;

  /// if "True", the corner pixels of the input and output tensors are
  /// aligned, and thus preserving the values at those pixels. This only has
  /// effect when :attr:`mode` is "linear", "bilinear", or
  /// "trilinear". Default: "False"
  TORCH_ARG(c10::optional<bool>, align_corners) = c10::nullopt;
};

namespace functional {

/// Options for a `D`-dimensional interpolate functional.
struct TORCH_API InterpolateFuncOptions {
  typedef c10::variant<
      enumtype::kNearest,
      enumtype::kLinear,
      enumtype::kBilinear,
      enumtype::kBicubic,
      enumtype::kTrilinear,
      enumtype::kArea> mode_t;

  /// output spatial sizes.
  TORCH_ARG(std::vector<int64_t>, size) = {};

  /// multiplier for spatial size.
  TORCH_ARG(std::vector<double>, scale_factor) = {};

  /// the upsampling algorithm: one of "nearest", "linear", "bilinear",
  /// "bicubic", "trilinear", and "area". Default: "nearest"
  TORCH_ARG(mode_t, mode) = torch::kNearest;

  /// Geometrically, we consider the pixels of the input and output as squares
  /// rather than points. If set to "True", the input and output tensors are
  /// aligned by the center points of their corner pixels, preserving the values
  /// at the corner pixels. If set to "False", the input and output tensors
  /// are aligned by the corner points of their corner pixels, and the
  /// interpolation uses edge value padding for out-of-boundary values, making
  /// this operation *independent* of input size when :attr:`scale_factor` is
  /// kept the same. This only has an effect when :attr:`mode` is "linear",
  /// "bilinear", "bicubic" or "trilinear". Default: "False"
  TORCH_ARG(c10::optional<bool>, align_corners) = c10::nullopt;
};

} // namespace functional

} // namespace nn
} // namespace torch
