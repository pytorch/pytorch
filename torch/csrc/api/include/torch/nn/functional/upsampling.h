#pragma once

#include <torch/nn/functional/pooling.h>
#include <torch/nn/options/upsampling.h>

#include <cmath>

namespace torch {
namespace nn {
namespace functional {

inline Tensor interpolate(const Tensor& input, InterpolateFuncOptions options) {
  auto _check_size_scale_factor = [options](size_t dim) {
    if (options.size().empty() && options.scale_factor().empty()) {
      TORCH_CHECK(false, "either size or scale_factor should be defined");
    }
    if (!options.size().empty() && !options.scale_factor().empty()) {
      TORCH_CHECK(false, "only one of size or scale_factor should be defined");
    }
    if (!options.scale_factor().empty() &&
        options.scale_factor().size() != dim) {
      TORCH_CHECK(
          false,
          "scale_factor shape must match input shape. "
          "Input is ", dim, "D, scale_factor size is ",
          options.scale_factor().size());
    }
  };

  auto _output_size = [input, options, _check_size_scale_factor](size_t dim) {
    _check_size_scale_factor(dim);
    if (!options.size().empty()) {
      return options.size();
    }
    auto scale_factors = options.scale_factor();

    std::vector<int64_t> sizes;
    for (size_t i = 0; i < dim; ++i) {
      sizes.push_back(static_cast<int64_t>(std::floor(
          static_cast<double>(input.size(i + 2)) * scale_factors[i])));
    }
    return sizes;
  };

  if (c10::get_if<enumtype::kNearest>(&options.mode()) ||
      c10::get_if<enumtype::kArea>(&options.mode())) {
    if (options.align_corners() != c10::nullopt) {
      TORCH_CHECK(
          false,
          "align_corners option can only be set with the "
          "interpolating modes: linear | bilinear | bicubic | trilinear");
    }
  } else {
    if (options.align_corners() == c10::nullopt) {
      TORCH_WARN(
          "Default upsampling behavior when mode is linear, bilinear, bicubic, "
          "or trilinear, has changed to align_corners=False since 0.4.0. "
          "Please specify align_corners=True if the old behavior is desired. "
          "See the documentation of nn.Upsample for details.");
      options.align_corners(false);
    }
  }

  if (input.dim() == 3 && c10::get_if<enumtype::kNearest>(&options.mode())) {
    return torch::upsample_nearest1d(input, _output_size(1));
  } else if (input.dim() == 4 && c10::get_if<enumtype::kNearest>(&options.mode())) {
    return torch::upsample_nearest2d(input, _output_size(2));
  } else if (input.dim() == 5 && c10::get_if<enumtype::kNearest>(&options.mode())) {
    return torch::upsample_nearest3d(input, _output_size(3));
  } else if (input.dim() == 3 && c10::get_if<enumtype::kArea>(&options.mode())) {
    return adaptive_avg_pool1d(input, _output_size(1));
  } else if (input.dim() == 4 && c10::get_if<enumtype::kArea>(&options.mode())) {
    return adaptive_avg_pool2d(input, _output_size(2));
  } else if (input.dim() == 5 && c10::get_if<enumtype::kArea>(&options.mode())) {
    return adaptive_avg_pool3d(input, _output_size(3));
  } else if (input.dim() == 3 && c10::get_if<enumtype::kLinear>(&options.mode())) {
    return torch::upsample_linear1d(input, _output_size(1), *options.align_corners());
  } else if (input.dim() == 3 && c10::get_if<enumtype::kBilinear>(&options.mode())) {
    TORCH_CHECK(false, "Got 3D input, but bilinear mode needs 4D input");
  } else if (input.dim() == 3 && c10::get_if<enumtype::kTrilinear>(&options.mode())) {
    TORCH_CHECK(false, "Got 3D input, but trilinear mode needs 5D input");
  } else if (input.dim() == 4 && c10::get_if<enumtype::kLinear>(&options.mode())) {
    TORCH_CHECK(false, "Got 4D input, but linear mode needs 3D input");
  } else if (input.dim() == 4 && c10::get_if<enumtype::kBilinear>(&options.mode())) {
    return torch::upsample_bilinear2d(input, _output_size(2), *options.align_corners());
  } else if (input.dim() == 4 && c10::get_if<enumtype::kTrilinear>(&options.mode())) {
    TORCH_CHECK(false, "Got 4D input, but trilinear mode needs 5D input");
  } else if (input.dim() == 5 && c10::get_if<enumtype::kLinear>(&options.mode())) {
    TORCH_CHECK(false, "Got 5D input, but linear mode needs 3D input");
  } else if (input.dim() == 5 && c10::get_if<enumtype::kBilinear>(&options.mode())) {
    TORCH_CHECK(false, "Got 5D input, but bilinear mode needs 4D input");
  } else if (input.dim() == 5 && c10::get_if<enumtype::kTrilinear>(&options.mode())) {
    return torch::upsample_trilinear3d(input, _output_size(3), *options.align_corners());
  } else if (input.dim() == 4 && c10::get_if<enumtype::kBicubic>(&options.mode())) {
    return torch::upsample_bicubic2d(input, _output_size(2), *options.align_corners());
  } else {
    TORCH_CHECK(
        false,
        "Input Error: Only 3D, 4D and 5D input Tensors supported "
        "(got ", input.dim(), "D) for the modes: nearest | linear | bilinear | bicubic | trilinear "
        "(got ", enumtype::get_enum_name(options.mode()), ")");
  }
}

} // namespace functional
} // namespace nn
} // namespace torch
