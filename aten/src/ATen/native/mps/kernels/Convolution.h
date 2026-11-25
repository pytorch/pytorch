#pragma once
#include <c10/metal/common.h>

// Outputs each conv1d_depthwise thread keeps in registers, so every weight
// tap is loaded once per 8 outputs.
C10_METAL_CONSTEXPR int32_t kConv1dDepthwiseOutputsPerThread = 8;

// MPS Graph runs undilated depthwise convolutions with strides of up to
// kMPSGraphDepthwiseMaxStride, and a product of filter taps and strides of up
// to kMPSGraphDepthwiseMaxTaps, with a kernel that keeps a threadgroup's input
// tile in threadgroup memory and does not check its size. The output is wrong
// once a tile exceeds kMPSGraphDepthwiseTileBytes, and on GPUs older than the
// Apple9 family (M1, M2) also when it takes exactly that many bytes.
// kMPSGraphDepthwiseTileOutputs bounds the output columns, output rows and
// images it gives one threadgroup. channels_last inputs of at least
// kMPSGraphDepthwiseUntiledChannels channels run on a kernel without tiles.
C10_METAL_CONSTEXPR int32_t kMPSGraphDepthwiseMaxStride = 2;
C10_METAL_CONSTEXPR int32_t kMPSGraphDepthwiseMaxTaps = 1024;
C10_METAL_CONSTEXPR int32_t kMPSGraphDepthwiseTileBytes = 65536;
C10_METAL_CONSTEXPR int32_t kMPSGraphDepthwiseUntiledChannels = 4;
C10_METAL_CONSTEXPR ::c10::metal::array<::c10::metal::array<int32_t, 3>, 5>
    kMPSGraphDepthwiseTileOutputs = {
        {{2, 2, 32}, {4, 2, 31}, {2, 4, 31}, {16, 2, 16}, {2, 16, 16}}};

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
