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

namespace at { namespace native {

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

static Tensor affine_grid_generator_4D(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    bool align_corners) {
  Tensor base_grid = make_base_grid_4D(theta, N, C, H, W, align_corners);
  auto grid = base_grid.view({N, H * W, 3}).bmm(theta.transpose(1, 2));
  return grid.view({N, H, W, 2});
}

static Tensor affine_grid_generator_5D(
    const Tensor& theta,
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    bool align_corners) {
  Tensor base_grid = make_base_grid_5D(theta, N, C, D, H, W, align_corners);
  auto grid = base_grid.view({N, D * H * W, 4}).bmm(theta.transpose(1, 2));
  return grid.view({N, D, H, W, 3});
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
                        .bmm(grad_grid.view({N, H * W, 2}));
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
                        .bmm(grad_grid.view({N, D * H * W, 3}));
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

}}  // namespace at::native
