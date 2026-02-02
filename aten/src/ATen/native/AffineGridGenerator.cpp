#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorOperators.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/affine_grid_generator_backward_native.h>
#include <ATen/ops/affine_grid_generator_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/linspace.h>
#include <ATen/ops/tensor.h>
#endif

namespace at::native {

static at::Tensor linspace_from_neg_one(const Tensor& grid, int64_t num_steps,
                                 bool align_corners) {
  if (num_steps <= 1) {
    return at::tensor(0, grid.options());
  }
  auto range = at::linspace(-1, 1, num_steps, grid.options());
  if (!align_corners) {
    range = range * (num_steps - 1) / num_steps;
  }
  return range;
}

// Optimized affine grid generator for 4D inputs.
// Instead of materializing a full (N, H, W, 3) base grid and doing bmm,
// we exploit the structure of the affine transformation to use broadcasting.
//
// The affine transformation is:
//   grid[n, h, w, :] = theta[n, :, 0] * x[w] + theta[n, :, 1] * y[h] + theta[n, :, 2]
//
// This reduces complexity from O(N * H * W * 3) to O(H + W) for base grid creation,
// and the final computation uses efficient broadcasting instead of explicit bmm.
//
// See: https://github.com/pytorch/pytorch/issues/174045
static Tensor affine_grid_generator_4D(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    bool align_corners) {
  // Create 1D linspaces - O(H + W) instead of O(N * H * W * 3)
  auto lin_h = linspace_from_neg_one(theta, H, align_corners).view({1, H, 1, 1});
  auto lin_w = linspace_from_neg_one(theta, W, align_corners).view({1, 1, W, 1});

  // theta has shape (N, 2, 3)
  // theta[:, :, 0] = weights for x (linspace along width)
  // theta[:, :, 1] = weights for y (linspace along height)
  // theta[:, :, 2] = translation (constant term)
  auto theta_w = theta.select(2, 0).view({N, 1, 1, 2});  // (N, 1, 1, 2)
  auto theta_h = theta.select(2, 1).view({N, 1, 1, 2});  // (N, 1, 1, 2)
  auto theta_c = theta.select(2, 2).view({N, 1, 1, 2});  // (N, 1, 1, 2)

  // Compute grid using broadcasting: O(N * 2 * (H + W)) instead of O(N * H * W * 3 * 2)
  // grid = theta_c + theta_h * lin_h + theta_w * lin_w
  // Broadcasting: (N, 1, 1, 2) + (N, 1, 1, 2) * (1, H, 1, 1) + (N, 1, 1, 2) * (1, 1, W, 1)
  // -> (N, H, W, 2)
  return theta_c + theta_h * lin_h + theta_w * lin_w;
}

// Optimized affine grid generator for 5D inputs.
// Same optimization as 4D but for volumetric data.
// Reduces complexity from O(N * D * H * W * 4) to O(D + H + W).
//
// See: https://github.com/pytorch/pytorch/issues/174045
static Tensor affine_grid_generator_5D(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    bool align_corners) {
  // Create 1D linspaces - O(D + H + W) instead of O(N * D * H * W * 4)
  auto lin_d = linspace_from_neg_one(theta, D, align_corners).view({1, D, 1, 1, 1});
  auto lin_h = linspace_from_neg_one(theta, H, align_corners).view({1, 1, H, 1, 1});
  auto lin_w = linspace_from_neg_one(theta, W, align_corners).view({1, 1, 1, W, 1});

  // theta has shape (N, 3, 4)
  // theta[:, :, 0] = weights for x (linspace along width)
  // theta[:, :, 1] = weights for y (linspace along height)
  // theta[:, :, 2] = weights for z (linspace along depth)
  // theta[:, :, 3] = translation (constant term)
  auto theta_w = theta.select(2, 0).view({N, 1, 1, 1, 3});  // (N, 1, 1, 1, 3)
  auto theta_h = theta.select(2, 1).view({N, 1, 1, 1, 3});  // (N, 1, 1, 1, 3)
  auto theta_d = theta.select(2, 2).view({N, 1, 1, 1, 3});  // (N, 1, 1, 1, 3)
  auto theta_c = theta.select(2, 3).view({N, 1, 1, 1, 3});  // (N, 1, 1, 1, 3)

  // Compute grid using broadcasting: O(N * 3 * (D + H + W)) instead of O(N * D * H * W * 4 * 3)
  // grid = theta_c + theta_d * lin_d + theta_h * lin_h + theta_w * lin_w
  // -> (N, D, H, W, 3)
  return theta_c + theta_d * lin_d + theta_h * lin_h + theta_w * lin_w;
}

// Legacy make_base_grid functions - still needed for backward pass
static Tensor make_base_grid_4D(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    bool align_corners) {
  auto base_grid = at::empty({N, H, W, 3}, theta.options());

  base_grid.select(-1, 0).copy_(linspace_from_neg_one(theta, W, align_corners));
  base_grid.select(-1, 1).copy_(linspace_from_neg_one(theta, H, align_corners).unsqueeze_(-1));
  base_grid.select(-1, 2).fill_(1);

  return base_grid;
}

static Tensor make_base_grid_5D(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    bool align_corners) {
  auto base_grid = at::empty({N, D, H, W, 4}, theta.options());

  base_grid.select(-1, 0).copy_(linspace_from_neg_one(theta, W, align_corners));
  base_grid.select(-1, 1).copy_(linspace_from_neg_one(theta, H, align_corners).unsqueeze_(-1));
  base_grid.select(-1, 2).copy_(linspace_from_neg_one(theta, D, align_corners).unsqueeze_(-1).unsqueeze_(-1));
  base_grid.select(-1, 3).fill_(1);

  return base_grid;
}

Tensor affine_grid_generator(const Tensor& theta, IntArrayRef size, bool align_corners) {
  TORCH_CHECK(
      size.size() == 4 || size.size() == 5,
      "AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.");
  if (size.size() == 4) {
    return affine_grid_generator_4D(
        theta, size[0], size[1], size[2], size[3], align_corners);
  } else {
    return affine_grid_generator_5D(
        theta, size[0], size[1], size[2], size[3], size[4], align_corners);
  }
}

static Tensor affine_grid_generator_4D_backward(
    const Tensor& grad_grid,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    bool align_corners) {
  auto base_grid = make_base_grid_4D(grad_grid, N, C, H, W, align_corners);
  AT_ASSERT(grad_grid.sizes() == IntArrayRef({N, H, W, 2}));
  auto grad_theta = base_grid.view({N, H * W, 3})
                        .transpose(1, 2)
                        .bmm(grad_grid.reshape({N, H * W, 2}));
  return grad_theta.transpose(1, 2);
}

static Tensor affine_grid_generator_5D_backward(
    const Tensor& grad_grid,
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    bool align_corners) {
  auto base_grid = make_base_grid_5D(grad_grid, N, C, D, H, W, align_corners);
  AT_ASSERT(grad_grid.sizes() == IntArrayRef({N, D, H, W, 3}));
  auto grad_theta = base_grid.view({N, D * H * W, 4})
                        .transpose(1, 2)
                        .bmm(grad_grid.reshape({N, D * H * W, 3}));
  return grad_theta.transpose(1, 2);
}

Tensor affine_grid_generator_backward(const Tensor& grad, IntArrayRef size, bool align_corners) {
  TORCH_CHECK(
      size.size() == 4 || size.size() == 5,
      "AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.");
  if (size.size() == 4) {
    return affine_grid_generator_4D_backward(
        grad, size[0], size[1], size[2], size[3], align_corners);
  } else {
    return affine_grid_generator_5D_backward(
        grad, size[0], size[1], size[2], size[3], size[4], align_corners);
  }
}

}  // namespace at::native
