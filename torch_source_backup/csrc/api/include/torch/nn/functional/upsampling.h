#pragma once

#include <c10/util/irange.h>
#include <torch/nn/functional/pooling.h>
#include <torch/nn/options/upsampling.h>

#include <cmath>
#include <utility>

namespace torch::nn::functional {

inline std::vector<int64_t> _interp_output_size(
    int64_t dim,
    std::tuple<
        Tensor,
        std::optional<std::vector<int64_t>>,
        std::optional<std::vector<double>>,
        std::optional<bool>> closed_over_args) {
  auto [input, size, scale_factor, recompute_scale_factor] =
      std::move(closed_over_args);
  if (size == std::nullopt && scale_factor == std::nullopt) {
    TORCH_CHECK(false, "either size or scale_factor should be defined");
  }
  if (size != std::nullopt && scale_factor != std::nullopt) {
    TORCH_CHECK(false, "only one of size or scale_factor should be defined");
  }
  if (scale_factor != std::nullopt) {
    if (static_cast<int64_t>(scale_factor.value().size()) != dim) {
      TORCH_CHECK(
          false,
          "scale_factor shape must match input shape. ",
          "Input is ",
          dim,
          "D, scale_factor size is ",
          torch::ArrayRef<double>(*scale_factor));
    }
  }
  if (size != std::nullopt) {
    return *size;
  }

  TORCH_INTERNAL_ASSERT(scale_factor != std::nullopt);
  auto scale_factors = *scale_factor;

  if (recompute_scale_factor == std::nullopt) {
    // only warn when the scales have floating values since
    // the result for ints is the same with/without recompute_scale_factor
    bool is_float_scale_factor = false;
    for (double scale : scale_factors) {
      is_float_scale_factor = floor(scale) != scale;
      if (is_float_scale_factor) {
        break;
      }
    }
    if (is_float_scale_factor) {
      TORCH_WARN(
          "The default behavior for interpolate/upsample with float scale_factor changed "
          "in 1.6.0 to align with other frameworks/libraries, and uses scale_factor directly, "
          "instead of relying on the computed output size. "
          "If you wish to keep the old behavior, please set recompute_scale_factor=True. "
          "See the documentation of nn.Upsample for details. ");
    }
  }

  std::vector<int64_t> ret;
  for (const auto i : c10::irange(dim)) {
    ret.emplace_back(static_cast<int64_t>(
        floor(static_cast<double>(input.size(i + 2)) * scale_factors[i])));
  }
  return ret;
}

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor interpolate(
    const Tensor& input,
    const std::optional<std::vector<int64_t>>& size,
    const std::optional<std::vector<double>>& scale_factor,
    InterpolateFuncOptions::mode_t mode,
    std::optional<bool> align_corners,
    std::optional<bool> recompute_scale_factor,
    bool antialias) {
  if (std::holds_alternative<enumtype::kNearest>(mode) ||
      std::get_if<enumtype::kArea>(&mode)) {
    if (align_corners != std::nullopt) {
      TORCH_CHECK(
          false,
          "align_corners option can only be set with the "
          "interpolating modes: linear | bilinear | bicubic | trilinear");
    }
  } else {
    if (align_corners == std::nullopt) {
      TORCH_WARN(
          "Default upsampling behavior when mode=",
          enumtype::get_enum_name(mode),
          " is changed "
          "to align_corners=False since 0.4.0. Please specify "
          "align_corners=True if the old behavior is desired. "
          "See the documentation of nn.Upsample for details.");
      align_corners = false;
    }
  }

  TORCH_CHECK(
      input.dim() >= 3 && input.dim() <= 5,
      "Input Error: Only 3D, 4D and 5D input Tensors supported "
      "(got ",
      input.dim(),
      "D) for the modes: nearest | linear | bilinear | bicubic | trilinear "
      "(got ",
      enumtype::get_enum_name(mode),
      ")");

  auto scale_factor_len = input.dim() - 2;
  std::vector<std::optional<double>> scale_factor_list(
      scale_factor_len, std::nullopt);
  if (scale_factor != std::nullopt && !recompute_scale_factor.value_or(false)) {
    auto _scale_factor_repeated = *scale_factor;
    scale_factor_list = {};
    for (const auto& elem : _scale_factor_repeated) {
      scale_factor_list.emplace_back(elem);
    }
  }

  if (antialias &&
      !(input.dim() == 4 &&
        (std::get_if<enumtype::kBilinear>(&mode) ||
         std::get_if<enumtype::kBicubic>(&mode)))) {
    TORCH_CHECK(
        false,
        "Anti-alias option is only supported for bilinear and bicubic modes");
  }

  auto closed_over_args =
      std::make_tuple(input, size, scale_factor, recompute_scale_factor);
  if (input.dim() == 3 && std::get_if<enumtype::kNearest>(&mode)) {
    return torch::upsample_nearest1d(
        input,
        _interp_output_size(1, std::move(closed_over_args)),
        scale_factor_list.at(0));
  } else if (input.dim() == 4 && std::get_if<enumtype::kNearest>(&mode)) {
    return torch::upsample_nearest2d(
        input,
        _interp_output_size(2, std::move(closed_over_args)),
        scale_factor_list.at(0),
        scale_factor_list.at(1));
  } else if (input.dim() == 5 && std::get_if<enumtype::kNearest>(&mode)) {
    return torch::upsample_nearest3d(
        input,
        _interp_output_size(3, std::move(closed_over_args)),
        scale_factor_list.at(0),
        scale_factor_list.at(1),
        scale_factor_list.at(2));
  } else if (input.dim() == 3 && std::get_if<enumtype::kNearestExact>(&mode)) {
    return torch::_upsample_nearest_exact1d(
        input,
        _interp_output_size(1, std::move(closed_over_args)),
        scale_factor_list.at(0));
  } else if (input.dim() == 4 && std::get_if<enumtype::kNearestExact>(&mode)) {
    return torch::_upsample_nearest_exact2d(
        input,
        _interp_output_size(2, std::move(closed_over_args)),
        scale_factor_list.at(0),
        scale_factor_list.at(1));
  } else if (input.dim() == 5 && std::get_if<enumtype::kNearestExact>(&mode)) {
    return torch::_upsample_nearest_exact3d(
        input,
        _interp_output_size(3, std::move(closed_over_args)),
        scale_factor_list.at(0),
        scale_factor_list.at(1),
        scale_factor_list.at(2));
  } else if (input.dim() == 3 && std::get_if<enumtype::kArea>(&mode)) {
    return detail::adaptive_avg_pool1d(
        input, _interp_output_size(1, std::move(closed_over_args)));
  } else if (input.dim() == 4 && std::get_if<enumtype::kArea>(&mode)) {
    return detail::adaptive_avg_pool2d(
        input, _interp_output_size(2, std::move(closed_over_args)));
  } else if (input.dim() == 5 && std::get_if<enumtype::kArea>(&mode)) {
    return detail::adaptive_avg_pool3d(
        input, _interp_output_size(3, std::move(closed_over_args)));
  } else if (input.dim() == 3 && std::get_if<enumtype::kLinear>(&mode)) {
    TORCH_CHECK(
        align_corners != std::nullopt, "align_corners should be specified.");
    return torch::upsample_linear1d(
        input,
        _interp_output_size(1, std::move(closed_over_args)),
        *align_corners,
        scale_factor_list.at(0));
  } else if (input.dim() == 3 && std::get_if<enumtype::kBilinear>(&mode)) {
    TORCH_CHECK(false, "Got 3D input, but bilinear mode needs 4D input");
  } else if (input.dim() == 3 && std::get_if<enumtype::kTrilinear>(&mode)) {
    TORCH_CHECK(false, "Got 3D input, but trilinear mode needs 5D input");
  } else if (input.dim() == 4 && std::get_if<enumtype::kLinear>(&mode)) {
    TORCH_CHECK(false, "Got 4D input, but linear mode needs 3D input");
  } else if (input.dim() == 4 && std::get_if<enumtype::kBilinear>(&mode)) {
    TORCH_CHECK(
        align_corners != std::nullopt, "align_corners should be specified.");
    if (antialias) {
      return torch::_upsample_bilinear2d_aa(
          input,
          _interp_output_size(2, std::move(closed_over_args)),
          *align_corners,
          scale_factor_list.at(0),
          scale_factor_list.at(1));
    }
    return torch::upsample_bilinear2d(
        input,
        _interp_output_size(2, std::move(closed_over_args)),
        *align_corners,
        scale_factor_list.at(0),
        scale_factor_list.at(1));
  } else if (input.dim() == 4 && std::get_if<enumtype::kTrilinear>(&mode)) {
    TORCH_CHECK(false, "Got 4D input, but trilinear mode needs 5D input");
  } else if (input.dim() == 5 && std::get_if<enumtype::kLinear>(&mode)) {
    TORCH_CHECK(false, "Got 5D input, but linear mode needs 3D input");
  } else if (input.dim() == 5 && std::get_if<enumtype::kBilinear>(&mode)) {
    TORCH_CHECK(false, "Got 5D input, but bilinear mode needs 4D input");
  } else if (input.dim() == 5 && std::get_if<enumtype::kTrilinear>(&mode)) {
    TORCH_CHECK(
        align_corners != std::nullopt, "align_corners should be specified.");
    return torch::upsample_trilinear3d(
        input,
        _interp_output_size(3, std::move(closed_over_args)),
        *align_corners,
        scale_factor_list.at(0),
        scale_factor_list.at(1),
        scale_factor_list.at(2));
  } else if (input.dim() == 4 && std::get_if<enumtype::kBicubic>(&mode)) {
    TORCH_CHECK(
        align_corners != std::nullopt, "align_corners should be specified.");
    if (antialias) {
      return torch::_upsample_bicubic2d_aa(
          input,
          _interp_output_size(2, std::move(closed_over_args)),
          *align_corners,
          scale_factor_list.at(0),
          scale_factor_list.at(1));
    }
    return torch::upsample_bicubic2d(
        input,
        _interp_output_size(2, std::move(closed_over_args)),
        *align_corners,
        scale_factor_list.at(0),
        scale_factor_list.at(1));
  } else {
    TORCH_CHECK(
        false,
        "Input Error: Only 3D, 4D and 5D input Tensors supported "
        "(got ",
        input.dim(),
        "D) for the modes: nearest | linear | bilinear | bicubic | trilinear "
        "(got ",
        enumtype::get_enum_name(mode),
        ")");
  }
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.interpolate
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::InterpolateFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::interpolate(input,
/// F::InterpolateFuncOptions().size({4}).mode(torch::kNearest));
/// ```
inline Tensor interpolate(
    const Tensor& input,
    const InterpolateFuncOptions& options = {}) {
  return detail::interpolate(
      input,
      options.size(),
      options.scale_factor(),
      options.mode(),
      options.align_corners(),
      options.recompute_scale_factor(),
      options.antialias());
}

} // namespace torch::nn::functional
