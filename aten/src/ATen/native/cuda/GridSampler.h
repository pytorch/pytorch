#pragma once
#include <array>
#include <cstdint>
#ifdef USE_ROCM
#include <ATen/core/TensorBase.h>
#include <ATen/native/CanUse32BitIndexMath.h>
#include <c10/util/llvmMathExtras.h>
#include <algorithm>
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
// Maximum lanes per block of the ROCm channel-parallel 2D backward kernel,
// which is also its launch bound.
constexpr int kRocmGridSampler2dMaxLanes = 256;

// Kernel selection for the 2D backward, made once per call by
// grid_sampler_2d_backward_cuda (GridSampler.cpp). When `channel_lane` is
// set, the caller launches launch_grid_sampler_2d_backward_channel_lane
// instead of launch_grid_sampler_2d_backward_kernel. When `scratch` is also
// set, the caller passes it a channels-last gradient buffer and copies that
// into the contiguous result after the launch.
struct RocmGridSampler2dBackwardDispatch {
  bool channel_lane = false;
  bool scratch = false;
};
RocmGridSampler2dBackwardDispatch rocm_grid_sampler_2d_backward_dispatch(
    const TensorBase &input, const TensorBase &grid,
    int64_t interpolation_mode, bool input_requires_grad);

// Launches the ROCm channel-parallel kernel for a call that
// rocm_grid_sampler_2d_backward_dispatch selected it for; it neither repeats
// that selection nor rereads the override. The caller supplies the selected
// gradient buffer (the channels-last scratch when `scratch` is set) and owns
// the copy out of it. Like launch_grid_sampler_2d_backward_kernel, this
// validates the arguments and raises the nondeterminism alert before the
// launch.
void launch_grid_sampler_2d_backward_channel_lane(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase &grad_output, const TensorBase &input,
    const TensorBase &grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool, 2> output_mask);

// Lanes per block of the channel-parallel kernel: a power of two, which its
// tree reduction requires, capped at its launch bound.
inline int64_t rocm_grid_sampler_2d_backward_lanes(int64_t num_channels) {
  // Clamp first: PowerOf2Ceil(0) is 0, and C = 0 still needs one lane.
  return static_cast<int64_t>(c10::llvm::PowerOf2Ceil(
      static_cast<uint64_t>(std::clamp<int64_t>(
          num_channels, 1, kRocmGridSampler2dMaxLanes))));
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
