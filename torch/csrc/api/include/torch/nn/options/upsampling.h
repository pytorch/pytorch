#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/enum.h>
#include <torch/expanding_array.h>
#include <torch/types.h>

#include <vector>

namespace torch {
namespace nn {

/// Options for the `Upsample` module.
///
/// Example:
/// ```
/// Upsample
/// model(UpsampleOptions().scale_factor(std::vector<double>({3})).mode(torch::kLinear).align_corners(false));
/// ```
struct TORCH_API UpsampleOptions {
  /// output spatial sizes.
  TORCH_ARG(c10::optional<std::vector<int64_t>>, size) = c10::nullopt;

  /// multiplier for spatial size.
  TORCH_ARG(c10::optional<std::vector<double>>, scale_factor) = c10::nullopt;

  /// the upsampling algorithm: one of "nearest", "linear", "bilinear",
  /// "bicubic" and "trilinear". Default: "nearest"
  typedef std::variant<
      enumtype::kNearest,
      enumtype::kLinear,
      enumtype::kBilinear,
      enumtype::kBicubic,
      enumtype::kTrilinear>
      mode_t;
  TORCH_ARG(mode_t, mode) = torch::kNearest;

  /// if "True", the corner pixels of the input and output tensors are
  /// aligned, and thus preserving the values at those pixels. This only has
  /// effect when :attr:`mode` is "linear", "bilinear", "bicubic", or
  /// "trilinear". Default: "False"
  TORCH_ARG(c10::optional<bool>, align_corners) = c10::nullopt;
};

namespace functional {

/// Options for `torch::nn::functional::interpolate`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::interpolate(input,
/// F::InterpolateFuncOptions().size(std::vector<int64_t>({4})).mode(torch::kNearest));
/// ```
struct TORCH_API InterpolateFuncOptions {
  typedef std::variant<
      enumtype::kNearest,
      enumtype::kLinear,
      enumtype::kBilinear,
      enumtype::kBicubic,
      enumtype::kTrilinear,
      enumtype::kArea,
      enumtype::kNearestExact>
      mode_t;

  /// output spatial sizes.
  TORCH_ARG(c10::optional<std::vector<int64_t>>, size) = c10::nullopt;

  /// multiplier for spatial size.
  TORCH_ARG(c10::optional<std::vector<double>>, scale_factor) = c10::nullopt;

  /// the upsampling algorithm: one of "nearest", "linear", "bilinear",
  /// "bicubic", "trilinear", "area", "nearest-exact". Default: "nearest"
  TORCH_ARG(mode_t, mode) = torch::kNearest;

  /// Geometrically, we consider the pixels of the input and output as squares
  /// rather than points. If set to "True", the input and output tensors are
  /// aligned by the center points of their corner pixels, preserving the values
  /// at the corner pixels. If set to "False", the input and output tensors
  /// are aligned by the corner points of their corner pixels, and the
  /// interpolation uses edge value padding for out-of-boundary values, making
  /// this operation *independent* of input size when `scale_factor` is
  /// kept the same.  It is *required* when interpolating mode is "linear",
  /// "bilinear", "bicubic" or "trilinear". Default: "False"
  TORCH_ARG(c10::optional<bool>, align_corners) = c10::nullopt;

  /// recompute the scale_factor for use in the
  /// interpolation calculation.  When `scale_factor` is passed as a parameter,
  /// it is used to compute the `output_size`.  If `recompute_scale_factor` is
  /// `true` or not specified, a new `scale_factor` will be computed based on
  /// the output and input sizes for use in the interpolation computation (i.e.
  /// the computation will be identical to if the computed `output_size` were
  /// passed-in explicitly).  Otherwise, the passed-in `scale_factor` will be
  /// used in the interpolation computation.  Note that when `scale_factor` is
  /// floating-point, the recomputed scale_factor may differ from the one passed
  /// in due to rounding and precision issues.
  TORCH_ARG(c10::optional<bool>, recompute_scale_factor) = c10::nullopt;

  /// flag to apply anti-aliasing. Using anti-alias
  /// option together with :attr:`align_corners` equals "False", interpolation
  /// result would match Pillow result for downsampling operation. Supported
  /// modes: "bilinear". Default: "False".
  TORCH_ARG(bool, antialias) = false;
};

} // namespace functional

} // namespace nn
} // namespace torch
