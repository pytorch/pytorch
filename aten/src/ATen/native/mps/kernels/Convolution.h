#pragma once
#include <c10/metal/common.h>

struct Conv1dDwParams {
  int32_t C;
  int32_t L;
  int32_t LO;
  int32_t NB;
  int32_t K;
  int32_t S;
  int32_t P;
  int32_t D;
  bool has_bias;
};

// src element strides of the OIDHW weight view (may be non-contiguous)
struct ConvWeightPermuteParams {
  int32_t O;
  int32_t CG;
  int32_t KH;
  int32_t KW;
  int32_t SO;
  int32_t SC;
  int32_t SD;
  int32_t SH;
  int32_t SW;
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
