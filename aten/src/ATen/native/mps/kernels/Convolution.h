#pragma once
#include <c10/metal/common.h>

// Eight adjacent outputs reuse each loaded depthwise weight while retaining
// enough threads to saturate the GPU for the measured Conv1d workloads.
C10_METAL_CONSTEXPR int32_t conv1d_dw_outputs_per_thread = 8;
#define CONV1D_DW_OUTPUTS_PER_THREAD_STR "8"

struct Conv1dDwParams {
  int32_t input_channels;
  int32_t input_length;
  int32_t output_length;
  int32_t batch_size;
  int32_t kernel_size;
  int32_t stride;
  int32_t padding;
  int32_t dilation;
  bool has_bias;
};

// Source element strides of the OIDHW weight view (may be non-contiguous).
struct ConvWeightPermuteParams {
  int32_t output_channels;
  int32_t input_channels_per_group;
  int32_t kernel_height;
  int32_t kernel_width;
  int32_t output_channel_stride;
  int32_t input_channel_stride;
  int32_t depth_stride;
  int32_t height_stride;
  int32_t width_stride;
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
