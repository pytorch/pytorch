#pragma once
// gemm_splitk.h - split-K matmul2d GEMM + reduction pass (deep-K, few-output-tile
// shapes where a single matmul2d underfills the GPU but splitting K across G
// threadgroups per tile fills it). Metal-4 only.
//
// KCHUNK is a RUNTIME extent: the AOT build cannot template on the data-dependent
// K / G, so chunk g offsets the A/B base pointers to its K-range and views exactly
// `kchunk` columns with the descriptor's K = dynamic_extent. Inputs are packed
// (lda == K, ldb == N, ldc == N).
//
// EVERY chunk (including chunk 0) writes its partial to an fp32 plane; the reduce
// sums all G planes in fp32 and stores OUT_T once. (Writing chunk 0 directly to the
// OUT_T output - as metalBLAS does to save a plane - rounds it to bf16 before the
// reduction, which survives catastrophic cancellation and blows up the relative
// error at near-zero outputs. Keeping every chunk in fp32 matches m5_tensor.)
#if __METAL_VERSION__ >= 400
#include <ATen/native/mps/kernels/gemm_common.h>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_cooperative_tensor>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
using namespace mpp::tensor_ops;

namespace at_gemm {

// tgid.z = K-chunk g; writes fp32 partial plane g of Cp. K range [g*kchunk, +kc).
template <typename IN_T, int BM, int BN, int NSG, bool RELAXED>
kernel void splitk_gemm(
    device IN_T* A [[buffer(0)]],
    device IN_T* B [[buffer(1)]],
    device float* Cp [[buffer(2)]],
    constant SplitKDims& gP [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  int gM = gP.M, gN = gP.N, gK = gP.K, kchunk = gP.kchunk;
  int g = int(tgid.z);
  int k0 = g * kchunk;
  if (k0 >= gK) {
    return;
  }
  int kc = min(kchunk, gK - k0);
  int tiles_n = (gN + BN - 1) / BN;
  int tgx = int(tgid.x), tgy = int(tgid.y);
  if (tgx >= tiles_n) {
    return;
  }
  int m_off = tgy * BM;
  int n_off = tgx * BN;

  // Views offset to the chunk's K-range (inner stride 1; leading dim = full K / N).
  device IN_T* Ag = A + (size_t)k0; // A packed (M, K): elem(m, k) = A[m*gK + k]
  device IN_T* Bg = B + (size_t)k0 * gN; // B packed (K, N): elem(k, n) = B[k*gN + n]
  device float* Cg = Cp + (size_t)g * gM * gN; // this chunk's fp32 plane
  tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tA(
      Ag, dextents<int32_t, 2>(kc, gM), array<int32_t, 2>{1, gK});
  tensor<device IN_T, dextents<int32_t, 2>, tensor_inline> tB(
      Bg, dextents<int32_t, 2>(gN, kc), array<int32_t, 2>{1, gN});
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tCp(
      Cg, dextents<int32_t, 2>(gN, gM), array<int32_t, 2>{1, gN});

  constexpr auto desc = matmul2d_descriptor(
      BM, BN, static_cast<int>(dynamic_extent), false, false, RELAXED,
      matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<NSG>> op;

  // Interior tiles take static-extent slices (matmul2d needs the BM/BN extents to
  // place the store; a plain dynamic slice mis-sizes it); edge tiles fall to a
  // dynamic slice (which clamps the store to the in-bounds region).
  const bool inside = (m_off + BM <= gM) && (n_off + BN <= gN);
  if (inside) {
    auto mA = tA.template slice<dynamic_extent, BM>(0, m_off);
    auto mB = tB.template slice<BN, dynamic_extent>(n_off, 0);
    auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    op.run(mA, mB, cT);
    auto mC = tCp.template slice<BN, BM>(n_off, m_off);
    cT.store(mC);
  } else {
    auto mA = tA.slice(0, m_off);
    auto mB = tB.slice(n_off, 0);
    auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    op.run(mA, mB, cT);
    auto mC = tCp.slice(n_off, m_off);
    cT.store(mC);
  }
}

// C[i] = (OUT_T) sum_{p < planes} Cp[p, i], one thread per output element.
template <typename OUT_T>
kernel void splitk_reduce(
    device const float* Cp [[buffer(0)]],
    device OUT_T* C [[buffer(1)]],
    constant SplitKReduceDims& gP [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  int n = gP.n, planes = gP.planes;
  int i = int(gid);
  if (i >= n) {
    return;
  }
  float s = 0.0f;
  size_t off = (size_t)i;
  for (int p = 0; p < planes; ++p) {
    s += Cp[off];
    off += (size_t)n;
  }
  C[i] = (OUT_T)s;
}

} // namespace at_gemm
#endif // __METAL_VERSION__ >= 400
