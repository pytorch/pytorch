#pragma once
// gemm_gemv.h - bandwidth-bound GEMV kernels for the rank-1 (M==1 / N==1) cases,
// which the tiled GEMM kernels handle correctly but inefficiently (a 32x32+ tile
// is ~97% masked for a vector). Real dtypes (float/half/bfloat) and integers share
// these (ACC_T = opmath_t<DT>: float for fp/bf16, int/long for integers).
//
// Fully-templated port of metalBLAS gemv_t.h / gemv_nt.h: VEC / NWARPS / epilogue
// and (gemv_nt) the threadgroup-reduce variant for int64 are template params.
// Reuses apply_epilogue, with the addend indexed at the output position.
#include <ATen/native/mps/kernels/gemm_common.h>
#include <metal_stdlib>

using namespace metal;

namespace at_gemm {

// y = x @ B, B is (K, N) row-major. Lanes own VEC columns (a coalesced line);
// NWARPS simdgroups split K and reduce in threadgroup memory.
template <typename DT, int NWARPS, int VEC, GemmEpilogue EPI>
kernel void gemv_t(
    device const DT* B [[buffer(0)]],
    device const DT* x [[buffer(1)]],
    device DT* y [[buffer(2)]],
    constant GemvDims& gP [[buffer(3)]],
    device const DT* self [[buffer(4)]],
    constant ::c10::metal::array<::c10::metal::opmath_t<DT>, 2>& alpha_beta
    [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  using ACC_T = ::c10::metal::opmath_t<DT>;
  using Vec = GemmVec<DT, VEC>;
  constexpr int BLOCK_N = 32 * VEC;
  const int gN = gP.n, gK = gP.K, gLdb = gP.ld, gXs = gP.xs;
  threadgroup ACC_T partials[NWARPS][BLOCK_N];

  const int col0 = int(tgid.x) * BLOCK_N;
  const int n0 = col0 + int(lane) * VEC;

  const int k_per_warp = (gK + NWARPS - 1) / NWARPS;
  const int k_start = int(sgid) * k_per_warp;
  const int k_end = min(gK, k_start + k_per_warp);

  ACC_T acc[VEC];
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    acc[i] = (ACC_T)0;
  }

  const bool full = (n0 + VEC) <= gN;
  if (full) {
    int k = k_start;
    for (; k + 4 <= k_end; k += 4) {
      Vec b0 = *((const device Vec*)(&B[(k + 0) * gLdb + n0]));
      Vec b1 = *((const device Vec*)(&B[(k + 1) * gLdb + n0]));
      Vec b2 = *((const device Vec*)(&B[(k + 2) * gLdb + n0]));
      Vec b3 = *((const device Vec*)(&B[(k + 3) * gLdb + n0]));
      ACC_T x0 = (ACC_T)x[(k + 0) * gXs], x1 = (ACC_T)x[(k + 1) * gXs];
      ACC_T x2 = (ACC_T)x[(k + 2) * gXs], x3 = (ACC_T)x[(k + 3) * gXs];
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        acc[i] += (ACC_T)b0.v[i] * x0;
        acc[i] += (ACC_T)b1.v[i] * x1;
        acc[i] += (ACC_T)b2.v[i] * x2;
        acc[i] += (ACC_T)b3.v[i] * x3;
      }
    }
    for (; k < k_end; ++k) {
      Vec bv = *((const device Vec*)(&B[k * gLdb + n0]));
      ACC_T xk = (ACC_T)x[k * gXs];
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        acc[i] += (ACC_T)bv.v[i] * xk;
      }
    }
  } else {
    for (int k = k_start; k < k_end; ++k) {
      ACC_T xk = (ACC_T)x[k * gXs];
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        int n = n0 + i;
        if (n < gN) {
          acc[i] += (ACC_T)B[k * gLdb + n] * xk;
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    partials[sgid][int(lane) * VEC + i] = acc[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgid == 0) {
    ACC_T alpha = alpha_beta[0];
    ACC_T beta = alpha_beta[1];
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      int cc = int(lane) * VEC + i;
      ACC_T s = (ACC_T)0;
#pragma unroll
      for (int w = 0; w < NWARPS; ++w) {
        s += partials[w][cc];
      }
      int n = col0 + cc;
      if (n < gN) {
        y[n] = apply_epilogue<EPI, DT, ACC_T>(
            s, /*r=*/0, /*c=*/n, self, gP.self_r, gP.self_c, alpha, beta);
      }
    }
  }
}

// y = A @ x, A is (M, K) row-major. Each simdgroup owns one row; lanes stride K
// (VEC-wide coalesced loads) and reduce their partials via threadgroup memory.
// (simd_sum is avoided: it has no integer overload, and under metal3.1 IF_CONSTEXPR
// is a runtime if, so a simd_sum branch would still compile for integer dtypes.)
template <typename DT, int NWARPS, int VEC, GemmEpilogue EPI>
kernel void gemv_nt(
    device const DT* A [[buffer(0)]],
    device const DT* x [[buffer(1)]],
    device DT* y [[buffer(2)]],
    constant GemvDims& gP [[buffer(3)]],
    device const DT* self [[buffer(4)]],
    constant ::c10::metal::array<::c10::metal::opmath_t<DT>, 2>& alpha_beta
    [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  using ACC_T = ::c10::metal::opmath_t<DT>;
  using Vec = GemmVec<DT, VEC>;
  const int gM = gP.n, gK = gP.K, gLda = gP.ld, gXs = gP.xs;
  threadgroup ACC_T part[NWARPS][32];
  const int K_STRIDE = 32 * VEC;

  int row = int(tgid.x) * NWARPS + int(sgid);
  if (row >= gM) {
    return;
  }
  const device DT* Arow = &A[row * gLda];
  ACC_T acc = (ACC_T)0;
  int k = int(lane) * VEC;
  for (; k + 3 * K_STRIDE + VEC <= gK; k += 4 * K_STRIDE) {
    Vec a0 = *((const device Vec*)(&Arow[k + 0 * K_STRIDE]));
    Vec a1 = *((const device Vec*)(&Arow[k + 1 * K_STRIDE]));
    Vec a2 = *((const device Vec*)(&Arow[k + 2 * K_STRIDE]));
    Vec a3 = *((const device Vec*)(&Arow[k + 3 * K_STRIDE]));
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      acc += (ACC_T)a0.v[i] * (ACC_T)x[(k + 0 * K_STRIDE + i) * gXs];
      acc += (ACC_T)a1.v[i] * (ACC_T)x[(k + 1 * K_STRIDE + i) * gXs];
      acc += (ACC_T)a2.v[i] * (ACC_T)x[(k + 2 * K_STRIDE + i) * gXs];
      acc += (ACC_T)a3.v[i] * (ACC_T)x[(k + 3 * K_STRIDE + i) * gXs];
    }
  }
  for (; k + VEC <= gK; k += K_STRIDE) {
    Vec av = *((const device Vec*)(&Arow[k]));
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      acc += (ACC_T)av.v[i] * (ACC_T)x[(k + i) * gXs];
    }
  }
  if (lane == 0) {
    int kk = (gK / VEC) * VEC;
    for (; kk < gK; ++kk) {
      acc += (ACC_T)Arow[kk] * (ACC_T)x[kk * gXs];
    }
  }

  ACC_T alpha = alpha_beta[0];
  ACC_T beta = alpha_beta[1];
  // simdgroup_barrier (not threadgroup_barrier): the out-of-range-row `return`
  // above is divergent across simdgroups and would deadlock a threadgroup barrier.
  simdgroup_barrier(mem_flags::mem_threadgroup);
  part[sgid][lane] = acc;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) {
    ACC_T s = (ACC_T)0;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
      s += part[sgid][i];
    }
    y[row] = apply_epilogue<EPI, DT, ACC_T>(
        s, /*r=*/row, /*c=*/0, self, gP.self_r, gP.self_c, alpha, beta);
  }
}

} // namespace at_gemm
