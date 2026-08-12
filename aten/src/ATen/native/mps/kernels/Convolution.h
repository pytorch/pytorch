#pragma once
#include <c10/metal/common.h>

// Eight adjacent outputs reuse each loaded depthwise weight while retaining
// enough threads to saturate the GPU for the measured Conv1d workloads.
C10_METAL_CONSTEXPR int32_t conv1d_dw_outputs_per_thread = 8;
#define CONV1D_DW_OUTPUTS_PER_THREAD_STR "8"

// The strides express NCL or NLC storage; swap_grid flips the (x, y) thread
// axes so the stride-1 storage axis stays along x for coalescing. The vec8
// variants assume NCL and ignore the stride fields.
struct Conv1dDwParams {
  int32_t input_channels;
  int32_t input_length;
  int32_t output_length;
  int32_t batch_size;
  int32_t kernel_size;
  int32_t stride;
  int32_t padding;
  int32_t dilation;
  int32_t channel_multiplier;
  int32_t in_channel_stride;
  int32_t in_pos_stride;
  int32_t out_channel_stride;
  int32_t out_pos_stride;
  bool swap_grid;
  bool has_bias;
};

// Source-width shape hint the conv1d conv3d_mpp specializations are compiled
// with (see CONV3D_MPP_SRCW in Convolution.metal); activations up to this
// length dispatch without host-side splitting. Oversizing the hint measured
// free, so the value is plain headroom; length * channels still has to stay
// inside the int32 addressing the eligibility gates enforce.
C10_METAL_CONSTEXPR int32_t conv1d_mpp_src_width_hint = 1 << 22;

// A region is an interval along the output length, not a 3D tensor subregion.
// Direct SGEMM supports both NCL and NLC activation storage.
//
// Example (groups=1): input [N, C_in, L] = [1, 2, 5] and weights
// [C_out, C_in, K] = [3, 2, 3], with stride=1, padding=1, dilation=1,
// produce output [N, C_out, outW] = [1, 3, 5].
// Logically, each zero-padded [C_in, K] patch flattens to C_in*K=6:
// W[3, 6] @ patch[6, 1] -> output column [3, 1], one value per weight row.
// This is only the GEMM view; the direct kernel does not materialize im2col.
// It accumulates valid W_j[3, 2] @ X_j[2, out_cols] products while indexing
// the NCL or NLC activation directly.
//
// Along the output length:
//   output 0 reads input [0, 2) with weight taps [1, 3);
//   outputs [1, 4) read inputs [0, 3), [1, 4), [2, 5) with taps [0, 3);
//   output 4 reads input [3, 5) with weight taps [0, 2).
// The middle region is {out_col0=1, out_cols=3, in_col0=0, taps=3,
// w_tap0=0, tile0=1}; the left-edge region occupies grid-y tile 0.
struct Conv1dSgemmRegion {
  // Index of the region's first output position.
  int32_t out_col0;
  // Number of consecutive output positions in the region.
  int32_t out_cols;
  // Input position used by out_col0 with weight tap w_tap0.
  int32_t in_col0;
  // Number of weight taps used per output in this region.
  int32_t taps;
  // Index of the first weight tap used.
  int32_t w_tap0;
  // Grid-y index assigned to the region's first output tile.
  int32_t tile0;
};

C10_METAL_CONSTEXPR int32_t conv1d_sgemm_max_regions = 16;

struct Conv1dSgemmParams {
  int32_t C_in;
  int32_t C_out;
  int32_t L;
  // Full output length, used when advancing between channels or batches.
  int32_t outW_total;
  // Input-position spacing between adjacent weight taps.
  int32_t dilation;
  int32_t groups;
  int32_t region_count;
  bool has_bias;
  ::c10::metal::array<Conv1dSgemmRegion, conv1d_sgemm_max_regions> regions;
};

// Source element strides of the OIDHW weight view (may be non-contiguous).
struct ConvWeightPermuteParams {
  uint32_t output_channels;
  uint32_t input_channels_per_group;
  uint32_t kernel_height;
  uint32_t kernel_width;
  uint32_t output_channel_stride;
  uint32_t input_channel_stride;
  uint32_t depth_stride;
  uint32_t height_stride;
  uint32_t width_stride;
};

struct Conv2DParams {
  int32_t N;
  int32_t C_in;
  int32_t C_out;
  int32_t H;
  int32_t W;
  int32_t outH;
  int32_t outW;
  int32_t kH;
  int32_t kW;
  int32_t sH;
  int32_t sW;
  int32_t padH;
  int32_t padW;
  int32_t dH;
  int32_t dW;
  int32_t C_in_per_group;
  int32_t C_out_per_group;
  bool has_bias;
};

// Shared between Convolution.metal and operations/Convolution.mm. conv3d_mpp
// bakes kernel, stride, and dilation into template args; conv3d_simd reads
// them from here.
struct Conv3DParams {
  Conv2DParams conv2d;
  int32_t D;
  int32_t outD;
  int32_t kD;
  int32_t sD;
  int32_t padD;
  int32_t dD;
  bool out_ncdhw;
};
