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
// Kernel selection for launch_grid_sampler_2d_backward_kernel. The caller
// allocates grad_input from `scratch` (a channels-last buffer copied into the
// contiguous result) and the launcher selects the kernel from `channel_lane`,
// so both follow one set of rules.
struct RocmGridSampler2dBackwardDispatch {
  bool channel_lane = false;
  bool scratch = false;
};
RocmGridSampler2dBackwardDispatch rocm_grid_sampler_2d_backward_dispatch(
    const TensorBase &input, const TensorBase &grid, bool input_requires_grad);
#endif

void launch_grid_sampler_3d_backward_kernel(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase &grad_output, const TensorBase &input,
    const TensorBase &grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool, 2> output_mask);

}  // namespace at::native
