#pragma once
#include <c10/metal/common.h>

// Shared between Convolution.metal and operations/Convolution.mm. conv3d_mpp
// bakes K*/S*/D* into template args; conv3d_simd reads them from here.
struct Conv3dDims {
  int C, H, W, O;
  int HO, WO, NB;
  int PADX, PADY;
  int CG, OG, OGT;
  int D, DO, PADZ;
  int KD, KH, KW;
  int SZ, SY, SX;
  int DZ, DY, DX;
  int HAS_BIAS, OUT_NCDHW;
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
