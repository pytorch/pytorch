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
// Kernel selection for the 2D backward, made once per call. The caller
// allocates grad_input from `scratch` (a channels-last buffer copied into the
// contiguous result) and passes the same decision to
// launch_grid_sampler_2d_backward_kernel_rocm, which selects the kernel from
// `channel_lane`.
struct RocmGridSampler2dBackwardDispatch {
  bool channel_lane = false;
  bool scratch = false;
};
RocmGridSampler2dBackwardDispatch rocm_grid_sampler_2d_backward_dispatch(
    const TensorBase &input, const TensorBase &grid, bool input_requires_grad);

void launch_grid_sampler_2d_backward_kernel_rocm(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase &grad_output, const TensorBase &input,
    const TensorBase &grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool, 2> output_mask,
    const RocmGridSampler2dBackwardDispatch &dispatch);
#endif

void launch_grid_sampler_3d_backward_kernel(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase &grad_output, const TensorBase &input,
    const TensorBase &grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool, 2> output_mask);

}  // namespace at::native
