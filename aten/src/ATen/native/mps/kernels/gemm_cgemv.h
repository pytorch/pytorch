#pragma once
// gemm_cgemv.h - native interleaved-complex GEMV for the rank-1 complex cases
// (M==1 / N==1). Reads the matrix once as interleaved C2 (vs the four-real-GEMM
// decomposition's repeated real/imag plane reads), accumulating in float2 for
// both complex dtypes. The epilogue applies the complex alpha*acc + beta*bias
// (complex addmm), fused into the same pass.
//
// Fully-templated port of metalBLAS cgemv_t.h / cgemv_nt.h: C2 (float2 / half2),
// R (output real scalar), NWARPS and epilogue are template params. Reuses GemvDims
// (n, K, ld, xs, self_r, self_c); the bias is indexed at the output position, so
// the broadcast step is self_c for cgemv_t (columns) and self_r for cgemv_nt (rows).
#include <ATen/native/mps/kernels/gemm_common.h>
#include <metal_stdlib>

using namespace metal;

namespace at_gemm {

// Complex epilogue: out = alpha*acc + beta*bias as a complex multiply. The beta
// term is dropped at runtime when beta == 0 so an uninitialized bias NaN can't
// reach the result (mirrors apply_epilogue). ab = {alpha_re, alpha_im, beta_re,
// beta_im}.
template <GemmEpilogue EPI, typename C2, typename R>
inline C2 apply_cepilogue(
    float2 acc,
    int bidx,
    device const C2* bias,
    ::c10::metal::array<float, 4> ab) {
  if IF_CONSTEXPR (EPI == GemmEpilogue::None) {
    return C2((R)acc.x, (R)acc.y);
  } else {
    float outr = ab[0] * acc.x - ab[1] * acc.y;
    float outi = ab[0] * acc.y + ab[1] * acc.x;
    if (ab[2] != 0.0f || ab[3] != 0.0f) {
      C2 b = bias[bidx];
      outr += ab[2] * (float)b.x - ab[3] * (float)b.y;
      outi += ab[2] * (float)b.y + ab[3] * (float)b.x;
    }
    return C2((R)outr, (R)outi);
  }
}

// y = x @ B, B is (K, N) row-major complex. Lanes own one column; NWARPS
// simdgroups split K and reduce the per-warp partials in threadgroup memory.
template <typename C2, typename R, int NWARPS, GemmEpilogue EPI>
kernel void cgemv_t(
    device const C2* B [[buffer(0)]],
    device const C2* x [[buffer(1)]],
    device C2* y [[buffer(2)]],
    constant GemvDims& gP [[buffer(3)]],
    device const C2* self [[buffer(4)]],
    constant ::c10::metal::array<float, 4>& ab [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BLOCK_N = 32;
  const int gN = gP.n, gK = gP.K, gLdb = gP.ld, gXs = gP.xs;
  threadgroup float2 partials[NWARPS][BLOCK_N];

  const int col0 = int(tgid.x) * BLOCK_N;
  const int n = col0 + int(lane);

  const int k_per_warp = (gK + NWARPS - 1) / NWARPS;
  const int k_start = int(sgid) * k_per_warp;
  const int k_end = min(gK, k_start + k_per_warp);

  float2 acc = float2(0);
  if (n < gN) {
    for (int k = k_start; k < k_end; ++k) {
      C2 b = B[k * gLdb + n];
      C2 xk = x[k * gXs];
      acc.x += (float)b.x * (float)xk.x - (float)b.y * (float)xk.y;
      acc.y += (float)b.x * (float)xk.y + (float)b.y * (float)xk.x;
    }
  }
  partials[sgid][lane] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgid == 0) {
    float2 s = float2(0);
#pragma unroll
    for (int w = 0; w < NWARPS; ++w) {
      s += partials[w][lane];
    }
    if (n < gN) {
      y[n] = apply_cepilogue<EPI, C2, R>(s, n * gP.self_c, self, ab);
    }
  }
}

// y = A @ x, A is (M, K) row-major complex. Each simdgroup owns one row; lanes
// stride K (coalesced C2 loads) and reduce with simd_sum (float accum, so the
// integer-overload issue that forced gemv_nt onto threadgroup reduction does not
// apply here; row is simdgroup-uniform, so the early return cannot deadlock).
template <typename C2, typename R, int NWARPS, GemmEpilogue EPI>
kernel void cgemv_nt(
    device const C2* A [[buffer(0)]],
    device const C2* x [[buffer(1)]],
    device C2* y [[buffer(2)]],
    constant GemvDims& gP [[buffer(3)]],
    device const C2* self [[buffer(4)]],
    constant ::c10::metal::array<float, 4>& ab [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int gM = gP.n, gK = gP.K, gLda = gP.ld, gXs = gP.xs;
  const int row = int(tgid.x) * NWARPS + int(sgid);
  if (row >= gM) {
    return;
  }
  const device C2* Arow = A + (size_t)row * gLda;
  float2 acc = float2(0);
  for (int k = int(lane); k < gK; k += 32) {
    C2 a = Arow[k];
    C2 xk = x[k * gXs];
    acc.x += (float)a.x * (float)xk.x - (float)a.y * (float)xk.y;
    acc.y += (float)a.x * (float)xk.y + (float)a.y * (float)xk.x;
  }
  acc.x = simd_sum(acc.x);
  acc.y = simd_sum(acc.y);
  if (lane == 0) {
    y[row] = apply_cepilogue<EPI, C2, R>(acc, row * gP.self_r, self, ab);
  }
}

} // namespace at_gemm
