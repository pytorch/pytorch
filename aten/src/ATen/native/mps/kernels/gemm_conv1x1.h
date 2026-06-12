#pragma once
// gemm_conv1x1.h - 1x1 convolution expressed as GEMM for the very-thin-N path
// (N <= 64 underfills a matmul2d tile). Metal-4 only. KCONST (== K) is a compile-time
// template param, so the host selects conv only when K matches a precompiled value.
#if __METAL_VERSION__ >= 400
#include <ATen/native/mps/kernels/gemm_common.h>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_cooperative_tensor>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

namespace at_gemm {

template <typename IN_T, typename OUT_T, int BMW, int BNO, int NSG, int KCONST>
kernel void conv1x1_gemm(
    device IN_T* A [[buffer(0)]],
    device IN_T* B [[buffer(1)]],
    device OUT_T* C [[buffer(2)]],
    constant ConvDims& gP [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  int gM = gP.M, gN = gP.N, gK = gP.K; // gK == KCONST (host gates on the match)
  tensor<device IN_T, dextents<int32_t, 4>, tensor_inline> tA(
      A, dextents<int32_t, 4>(gK, gM, 1, 1));
  tensor<device IN_T, dextents<int32_t, 4>, tensor_inline> tW(
      B, dextents<int32_t, 4>(gN, gK, 1, 1));
  tensor<device OUT_T, dextents<int32_t, 4>, tensor_inline> tC(
      C, dextents<int32_t, 4>(gN, gM, 1, 1));

  constexpr auto desc = convolution2d_descriptor(
      int4(BNO, BMW, 1, 1),
      int4(KCONST, BMW, 1, 1),
      int2(1, 1),
      convolution2d_activation_layout::nhwc,
      convolution2d_weights_layout::hwio,
      int2(1, 1),
      int2(1, 1),
      1,
      true,
      convolution2d_descriptor::mode::multiply);
  convolution2d<desc, execution_simdgroups<NSG>> op;

  int tiles_o = (gN + BNO - 1) / BNO;
  int tiles_w = (gM + BMW - 1) / BMW;
  if (int(tgid.x) >= tiles_o || int(tgid.y) >= tiles_w) {
    return;
  }
  int o_off = int(tgid.x) * BNO;
  int w_off = int(tgid.y) * BMW;

  auto mA = tA.slice(0, w_off, 0, 0);
  auto mW = tW.slice(o_off, 0, 0, 0);
  auto mC = tC.slice(o_off, w_off, 0, 0);
  auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mW), float>();
  op.run(mA, mW, cT);
  auto cO = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mW), OUT_T>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
    cO[i] = (OUT_T)cT[i];
  }
  cO.store(mC);
}

} // namespace at_gemm
#endif // __METAL_VERSION__ >= 400
