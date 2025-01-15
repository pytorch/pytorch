#pragma once

#include <ATen/Tensor.h>
#include <array>

namespace at {
namespace native {

std::tuple<Tensor, Tensor> grid_sampler_2d_backward_mps(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    std::array<bool, 2> output_mask);

} // namespace native
} // namespace at
