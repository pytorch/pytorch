#pragma once
// gemm_m5_tensor.h - the PRIMARY GEMM backend: mpp::tensor_ops::matmul2d on the
// Apple tensor unit. Metal-4 only (cooperative tensors), so the whole file is
// guarded by __METAL_VERSION__ >= 400 and only populates kernels_40.metallib.
//
// Fully-templated port of metalBLAS mpp_tensor.h. Handles packed, transposed
// (column-major) and arbitrary-leading-dim operands through strided tensor_inline
// views (inner stride 1, outer stride = the leading dim) plus the matmul2d
// transpose flags - so col-major weights, [::k]-strided and transposed inputs all
// ride this path without a materialized copy. Untransposed interior tiles take a
// static-extent slice (no per-tile edge predication); transposed and partial-edge
// tiles take a dynamic slice + per-element validity mask.
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
    bool TRANS_A,
    bool TRANS_B,
    bool RELAXED,
    GemmEpilogue EPI,
    bool BATCHED>
kernel void m5_tensor_gemm(
    device IN_T* A [[buffer(0)]],
    device IN_T* B [[buffer(1)]],
    device OUT_T* C [[buffer(2)]],
    constant GemmDimsStrided& gP [[buffer(3)]],
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

  // Strided tensor views: extent order is (cols, rows) for row-major; transposed
  // operands swap the extents and flip the descriptor flag. The {1, ld} stride
  // (unit inner, ld outer) absorbs any leading-dim / column-major view.
  auto eA = TRANS_A ? dextents<int32_t, 2>(gM, gK) : dextents<int32_t, 2>(gK, gM);
  auto eB = TRANS_B ? dextents<int32_t, 2>(gK, gN) : dextents<int32_t, 2>(gN, gK);
  tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tA(
      A, eA, array<int32_t, 2>{1, gP.lda});
  tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tB(
      B, eB, array<int32_t, 2>{1, gP.ldb});
  tensor<device OUT_T, dextents<int32_t, 2>, tensor_inline> tC(
      C, dextents<int32_t, 2>(gN, gM), array<int32_t, 2>{1, gP.ldc});

  // K is a runtime extent: the descriptor's K field is int, and the
  // dynamic_extent sentinel narrows to -1 (its documented "dynamic" encoding).
  constexpr auto desc = matmul2d_descriptor(
      BM,
      BN,
      static_cast<int>(dynamic_extent),
      TRANS_A,
      TRANS_B,
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
  bool inside = (m_off + BM <= gM) && (n_off + BN <= gN);

  if IF_CONSTEXPR (!TRANS_A && !TRANS_B) {
    // Untransposed fast path: static-extent slices mark interior tiles exactly
    // BM x BN and in-bounds, dropping dynamic-slice edge predication.
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
  } else {
    // Transposed / strided operands: matmul2d reads them through the transpose
    // flags; a dynamic slice + validity mask handles any tile.
    auto mA = TRANS_A ? tA.slice(m_off, 0) : tA.slice(0, m_off);
    auto mB = TRANS_B ? tB.slice(0, n_off) : tB.slice(n_off, 0);
    auto mC = tC.slice(n_off, m_off);
    auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    op.run(mA, mB, cT);
    auto cO = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), OUT_T>();
    uint16_t e = 0;
    for (auto it = cT.begin(); it != cT.end(); ++it, ++e) {
      if (!inside && !cT.is_valid_element(e)) {
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
