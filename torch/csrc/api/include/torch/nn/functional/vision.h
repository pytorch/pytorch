#pragma once

#include <torch/nn/options/vision.h>
#include <torch/types.h>

namespace torch {
namespace nn {
namespace functional {

inline Tensor affine_grid(
    const Tensor& theta,
    const IntArrayRef& size,
    bool align_corners = false) {
  // enforce floating point dtype on theta
  TORCH_CHECK(
      theta.is_floating_point(),
      "Expected theta to have floating point type, but got ",
      theta.dtype());

  // check that shapes and sizes match
  if (size.size() == 4) {
    TORCH_CHECK(
        theta.dim() == 3 && theta.size(-2) == 2 && theta.size(-1) == 3,
        "Expected a batch of 2D affine matrices of shape Nx2x3 for size ",
        size,
        ". Got ",
        theta.sizes(),
        ".");
  } else if (size.size() == 5) {
    TORCH_CHECK(
        theta.dim() == 3 && theta.size(-2) == 3 && theta.size(-1) == 4,
        "Expected a batch of 3D affine matrices of shape Nx3x4 for size ",
        size,
        ". Got ",
        theta.sizes(),
        ".");
  } else {
    TORCH_CHECK(
        false,
        "affine_grid only supports 4D and 5D sizes, ",
        "for 2D and 3D affine transforms, respectively. ",
        "Got size ",
        size);
  }

  if (*std::min_element(size.begin(), size.end()) <= 0) {
    TORCH_CHECK(false, "Expected non-zero, positive output size. Got ", size);
  }

  return torch::affine_grid_generator(theta, size, align_corners);
}

// ============================================================================

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace detail {
inline Tensor grid_sample(
    const Tensor& input,
    const Tensor& grid,
    GridSampleFuncOptions::mode_t mode,
    GridSampleFuncOptions::padding_mode_t padding_mode,
    c10::optional<bool> align_corners) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t mode_enum, padding_mode_enum;

  if (std::holds_alternative<enumtype::kBilinear>(mode)) {
    mode_enum = 0;
  } else if (std::holds_alternative<enumtype::kNearest>(mode)) {
    mode_enum = 1;
  } else { /// mode == 'bicubic'
    mode_enum = 2;
  }

  if (std::holds_alternative<enumtype::kZeros>(padding_mode)) {
    padding_mode_enum = 0;
  } else if (std::holds_alternative<enumtype::kBorder>(padding_mode)) {
    padding_mode_enum = 1;
  } else { /// padding_mode == 'reflection'
    padding_mode_enum = 2;
  }

  if (!align_corners.has_value()) {
    TORCH_WARN(
        "Default grid_sample and affine_grid behavior has changed ",
        "to align_corners=False since 1.3.0. Please specify ",
        "align_corners=True if the old behavior is desired. ",
        "See the documentation of grid_sample for details.");
    align_corners = false;
  }

  return torch::grid_sampler(
      input, grid, mode_enum, padding_mode_enum, align_corners.value());
}
} // namespace detail
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

/// See
/// https://pytorch.org/docs/main/nn.functional.html#torch.nn.functional.grid_sample
/// about the exact behavior of this functional.
///
/// See the documentation for `torch::nn::functional::GridSampleFuncOptions`
/// class to learn what optional arguments are supported for this functional.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::grid_sample(input, grid,
/// F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true));
/// ```
inline Tensor grid_sample(
    const Tensor& input,
    const Tensor& grid,
    const GridSampleFuncOptions& options = {}) {
  return detail::grid_sample(
      input,
      grid,
      options.mode(),
      options.padding_mode(),
      options.align_corners());
}

} // namespace functional
} // namespace nn
} // namespace torch
