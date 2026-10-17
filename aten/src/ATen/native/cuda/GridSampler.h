#pragma once
#include <array>
#include <cstdint>

namespace at {
class TensorBase;
}

namespace at::native {

void launch_grid_sampler_2d_forward_kernel(
    const TensorBase &output, const TensorBase &input, const TensorBase &grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners);

void launch_grid_sampler_3d_forward_kernel(
    const TensorBase &output, const TensorBase &input, const TensorBase &grid,
    int64_t interpolation_mode, int64_t padding_mode, bool align_corners);

void launch_grid_sampler_2d_backward_kernel(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase &grad_output, const TensorBase &input,
    const TensorBase &grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool, 2> output_mask);

#ifdef USE_ROCM
// True when launch_grid_sampler_2d_backward_kernel would select the
// channel-parallel kernel given a channels-last grad_input, so the caller
// accumulates into a channels-last scratch instead of the returned gradient.
bool rocm_grid_sampler_2d_backward_use_channels_last_scratch(
    const TensorBase &input, const TensorBase &grid);
#endif

void launch_grid_sampler_3d_backward_kernel(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase &grad_output, const TensorBase &input,
    const TensorBase &grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool, 2> output_mask);

}  // namespace at::native
