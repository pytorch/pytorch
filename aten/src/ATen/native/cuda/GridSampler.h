#pragma once
#include <array>
#include <cstdint>
#ifdef USE_ROCM
#include <ATen/core/TensorBase.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <limits>
#endif

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

// Lanes per block of the channel-parallel kernel: a power of two, which its
// tree reduction requires, capped at its 256-thread launch bound.
inline int64_t rocm_grid_sampler_2d_backward_lanes(int64_t num_channels) {
  int64_t lanes = 1;
  while (lanes < num_channels && lanes < 256) {
    lanes <<= 1;
  }
  return lanes;
}

// Whether the channel-parallel kernel can index in 32 bits. Besides the
// tensor offsets, each lane's channel counter advances by the lane count past
// its last channel, so C - 1 + lanes must also fit.
inline bool rocm_grid_sampler_2d_backward_channel_lane_use_32bit_index(
    const TensorBase &input, const TensorBase &grid,
    const TensorBase &grad_output) {
  const int64_t C = input.size(1);
  return C <= std::numeric_limits<int32_t>::max() -
          (rocm_grid_sampler_2d_backward_lanes(C) - 1) &&
      canUse32BitIndexMath(input) && canUse32BitIndexMath(grid) &&
      canUse32BitIndexMath(grad_output);
}

#endif
void launch_grid_sampler_3d_backward_kernel(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase &grad_output, const TensorBase &input,
    const TensorBase &grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool, 2> output_mask);

}  // namespace at::native
