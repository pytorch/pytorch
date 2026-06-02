#pragma once
// gemm_m5_tensor.h - the PRIMARY GEMM backend: mpp::tensor_ops::matmul2d on the
// Apple M5 tensor unit. Metal-4 only (cooperative tensors), so the whole file is
// guarded by __METAL_VERSION__ >= 400 and only populates kernels_40.metallib.
//
// Fully-templated port of metalBLAS m5_tensor.h, restricted to the packed,
// untransposed case (the dispatcher routes transposed/strided inputs to the m5
// manual kernel). MN-alignment is a runtime check (interior tiles take the
// static-extent slice; partial edge tiles take a dynamic slice + validity mask).
#if __METAL_VERSION__ >= 400
#include <ATen/native/mps/kernels/gemm_common.h>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_cooperative_tensor>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

namespace at_gemm {

template <
    typename IN_T,
    typename OUT_T,
    int BM,
    int BN,
    int NSG,
    bool RELAXED,
    GemmEpilogue EPI,
    bool BATCHED>
kernel void m5_tensor_gemm(
    device IN_T* A [[buffer(0)]],
    device IN_T* B [[buffer(1)]],
    device OUT_T* C [[buffer(2)]],
    constant GemmDimsPacked& gP [[buffer(3)]],
    device const OUT_T* self [[buffer(4)]],
    constant ::c10::metal::array<::c10::metal::opmath_t<OUT_T>, 2>& alpha_beta
    [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  using op_t = ::c10::metal::opmath_t<OUT_T>;
  int gM = gP.M, gN = gP.N, gK = gP.K;

  if IF_CONSTEXPR (BATCHED) {
    int b = int(tgid.z);
    A += (size_t)b * gP.batch_a;
    B += (size_t)b * gP.batch_b;
    C += (size_t)b * gP.batch_c;
    if IF_CONSTEXPR (EPI == GemmEpilogue::AlphaBeta) {
      self += (size_t)b * gP.batch_self;
    }
  }

  // Tensor views from raw pointers; extent order is (cols, rows) for row-major.
  tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tA(
      A, dextents<int32_t, 2>(gK, gM));
  tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tB(
      B, dextents<int32_t, 2>(gN, gK));
  tensor<device OUT_T, dextents<int32_t, 2>, tensor_inline> tC(
      C, dextents<int32_t, 2>(gN, gM));

  // K is a runtime extent: the descriptor's K field is int, and the
  // dynamic_extent sentinel narrows to -1 (its documented "dynamic" encoding).
  constexpr auto desc = matmul2d_descriptor(
      BM,
      BN,
      static_cast<int>(dynamic_extent),
      /*transpose_a=*/false,
      /*transpose_b=*/false,
      RELAXED,
      matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<NSG>> op;

  int tiles_m = (gM + BM - 1) / BM;
  int tiles_n = (gN + BN - 1) / BN;
  int tgx = int(tgid.x);
  int tgy = int(tgid.y);
  if (tgx >= tiles_n || tgy >= tiles_m) {
    return;
  }
  int m_off = tgy * BM;
  int n_off = tgx * BN;

  op_t alpha = alpha_beta[0];
  op_t beta = alpha_beta[1];

  // Interior tiles are exactly BM x BN and fully in-bounds: static-extent slices
  // let matmul2d skip per-tile edge predication.
  bool inside = (m_off + BM <= gM) && (n_off + BN <= gN);
  if (inside) {
    auto mA = tA.template slice<dynamic_extent, BM>(0, m_off);
    auto mB = tB.template slice<BN, dynamic_extent>(n_off, 0);
    auto mC = tC.template slice<BN, BM>(n_off, m_off);
    auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    op.run(mA, mB, cT);
    auto cO = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), OUT_T>();
    if IF_CONSTEXPR (EPI == GemmEpilogue::None) {
      for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        cO[i] = (OUT_T)cT[i];
      }
    } else {
      uint16_t e = 0;
      for (auto it = cT.begin(); it != cT.end(); ++it, ++e) {
        auto idx = it.get_multidimensional_index(); // [col, row]
        int r = m_off + int(idx[1]);
        int c = n_off + int(idx[0]);
        cO[e] = apply_epilogue<EPI, OUT_T, float>(
            cT[e], r, c, self, gP.self_r, gP.self_c, alpha, beta);
      }
    }
    cO.store(mC);
  } else {
    // Partial edge tile: dynamic slice with per-element validity mask.
    auto mA = tA.slice(0, m_off);
    auto mB = tB.slice(n_off, 0);
    auto mC = tC.slice(n_off, m_off);
    auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    op.run(mA, mB, cT);
    auto cO = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), OUT_T>();
    uint16_t e = 0;
    for (auto it = cT.begin(); it != cT.end(); ++it, ++e) {
      if (!cT.is_valid_element(e)) {
        continue;
      }
      if IF_CONSTEXPR (EPI == GemmEpilogue::None) {
        cO[e] = (OUT_T)cT[e];
      } else {
        auto idx = it.get_multidimensional_index();
        int r = m_off + int(idx[1]);
        int c = n_off + int(idx[0]);
        cO[e] = apply_epilogue<EPI, OUT_T, float>(
            cT[e], r, c, self, gP.self_r, gP.self_c, alpha, beta);
      }
    }
    cO.store(mC);
  }
}

} // namespace at_gemm
#endif // __METAL_VERSION__ >= 400
