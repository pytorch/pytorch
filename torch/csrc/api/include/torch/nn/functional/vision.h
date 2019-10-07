#pragma once

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
        theta.sizes(), ".");
  } else if (size.size() == 5) {
    TORCH_CHECK(
        theta.dim() == 3 && theta.size(-2) == 3 && theta.size(-1) == 4,
        "Expected a batch of 3D affine matrices of shape Nx3x4 for size ",
        size,
        ". Got ",
        theta.sizes(), ".");
  } else {
    TORCH_CHECK(
        false,
        "affine_grid only supports 4D and 5D sizes, ",
        "for 2D and 3D affine transforms, respectively. ",
        "Got size ", size);
  }

  if (*std::min_element(size.begin(), size.end()) <= 0) {
    TORCH_CHECK(false, "Expected non-zero, positive output size. Got ", size);
  }

  return torch::affine_grid_generator(theta, size, align_corners);
}

} // namespace functional
} // namespace nn
} // namespace torch
