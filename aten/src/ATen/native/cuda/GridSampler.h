#pragma once
#include <array>
#include <cstdint>
#ifdef USE_ROCM
#include <limits>
#endif

namespace at {
class TensorBase;
}

namespace at::native {

#ifdef USE_ROCM
// Note [ROCm grid_sampler_2d backward channel-lane eligibility]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// The ROCm 2D backward kernel maps one block per (n, h, w) with lanes striding
// over channels, instead of one thread per (n, h, w) looping over channels. That
// pays off only when there are enough channels to fill a wave, and it pays off
// most when the channel dimension is innermost, because then both the
// grad_output/input gathers and the grad_input atomics are contiguous across the
// wave.
//
// THREE call sites must agree on this decision, and the predicate must therefore
// capture *every* reason the channel-lane kernel might not run:
//   * grid_sampler_2d_backward_cuda in GridSampler.cpp, which picks grad_input's
//     memory format;
//   * launch_grid_sampler_2d_backward_kernel in GridSampler.cu, which picks the
//     kernel;
//   * grid_sampler_2d_backward_meta in torch/_meta_registrations.py, which must
//     advertise the strides the other two produce.
//
// Keeping any condition in only one of them is a correctness bug, not just an
// inconsistency. Allocating a channels-last grad_input and then falling back to
// the generic kernel hands channels-last spatial strides (sH == W * C) to the
// shared safe_add_2d, whose products are `int` -- see Note [ROCm grid_sampler_2d
// backward wide offset arithmetic] in GridSampler.cu. Only the channel-lane
// kernel uses the widened helpers, so that combination can overflow into an
// out-of-bounds atomic. The launch-extent test below therefore lives here rather
// than at the dispatch site.
//
// Channel thresholds come from a measured sweep on gfx1250 (C in 1..32 across
// three shape families, both layouts). Channels-last regresses only at C == 1
// and is >= 1.6x from C == 4. Contiguous is noisy through C == 12, only reaches
// 1.07x at C == 16, and is a comfortable >= 1.34x from C == 32.

// Lanes per block: next power of two >= C (the kernel's tree reduction requires
// a power of two), capped at the 256-thread launch bound.
inline int64_t rocm_grid_sampler_2d_backward_lanes(int64_t num_channels) {
  int64_t lanes = 1;
  while (lanes < num_channels && lanes < 256) {
    lanes <<= 1;
  }
  return lanes;
}

inline bool rocm_grid_sampler_2d_backward_use_channel_lane(
    int64_t num_channels, bool channels_last, int64_t nblocks_nhw) {
  const bool enough_channels =
      channels_last ? (num_channels >= 4) : (num_channels >= 32);
  if (!enough_channels || nblocks_nhw <= 0) {
    return false;
  }
  // One block per (n, h, w), so the grid dimension must be addressable...
  if (nblocks_nhw > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    return false;
  }
  // ...and HIP additionally bounds the global work size, gridDim.x * blockDim.x.
  // See the same bound in UpSampleNearest2d.cu.
  constexpr int64_t kHipMaxGlobalWorkSize = 4294967295LL;  // UINT32_MAX
  return nblocks_nhw <=
      kHipMaxGlobalWorkSize / rocm_grid_sampler_2d_backward_lanes(num_channels);
}
#endif  // USE_ROCM

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

void launch_grid_sampler_3d_backward_kernel(
    const TensorBase &grad_input, const TensorBase &grad_grid,
    const TensorBase &grad_output, const TensorBase &input,
    const TensorBase &grid, int64_t interpolation_mode, int64_t padding_mode,
    bool align_corners, std::array<bool, 2> output_mask);

}  // namespace at::native
