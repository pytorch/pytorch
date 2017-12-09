#pragma once

#include <ATen/ATen.h>

namespace at { namespace native {

Tensor cudnn_affine_grid_generator_forward(
    const Tensor& theta,
    int64_t N, int64_t C, int64_t H, int64_t W);

Tensor cudnn_affine_grid_generator_backward(
    const Tensor& grad_theta,
    int64_t N, int64_t C, int64_t H, int64_t W);

}}
