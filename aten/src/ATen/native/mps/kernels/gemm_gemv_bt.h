#pragma once
// gemm_gemv_bt.h - thin-M batched GEMV (Y = X @ B, M=2..16): streams B once,
// keeps the M dot-products in registers. Port of metalBLAS gemv_bt.h (half/bf16).
#include <ATen/native/mps/kernels/gemm_common.h>
#include <metal_stdlib>

using namespace metal;

namespace at_gemm {

// Derived NWARPS (must match GemmMetal.mm): caps the row-major partials tile
// (NWARPS*MROWS*32*VEC floats) within threadgroup memory.
constexpr int gemv_bt_nwarps(int mrows, int vec) {
  return (mrows * vec <= 24) ? 8 : 4;
}

// Row-major B: lanes own VEC columns (BLOCK_N=32*VEC line), NWARPS split K and
// reduce in threadgroup memory. X read scalar. MROWS is a padded capacity.
template <typename DT, int MROWS, int VEC, GemmEpilogue EPI>
kernel void gemv_bt(
    device const DT* B [[buffer(0)]],
    device const DT* X [[buffer(1)]],
    device DT* Y [[buffer(2)]],
    constant GemvBtDims& gP [[buffer(3)]],
    device const DT* self [[buffer(4)]],
    constant ::c10::metal::array<::c10::metal::opmath_t<DT>, 2>& alpha_beta
    [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  using ACC_T = ::c10::metal::opmath_t<DT>;
  using Vec = GemmVec<DT, VEC>;
  constexpr int NWARPS = gemv_bt_nwarps(MROWS, VEC);
  constexpr int BLOCK_N = 32 * VEC;
  threadgroup ACC_T partials[NWARPS][MROWS][BLOCK_N];

  const int gM = gP.M, gN = gP.N, gK = gP.K;
  const int gLdb = gP.ldb, gLdx = gP.ldx, gLdy = gP.ldy;
  B += (int64_t)tgid.z * gP.batch_b;
  X += (int64_t)tgid.z * gP.batch_x;
  Y += (int64_t)tgid.z * gP.batch_y;
  self += (int64_t)tgid.z * gP.batch_self;

  // Clamped per-row X base pointers: padding rows (m >= gM) alias the last valid
  // row so their loads stay in-bounds; their results are dropped at the store.
  const device DT* Xr[MROWS];
#pragma unroll
  for (int m = 0; m < MROWS; ++m) {
    const int mm = (m < gM) ? m : (gM - 1);
    Xr[m] = X + (int64_t)mm * gLdx;
  }

  const int col0 = int(tgid.x) * BLOCK_N;
  const int n0 = col0 + int(lane) * VEC;
  const int k_per_warp = (gK + NWARPS - 1) / NWARPS;
  const int k_start = int(sgid) * k_per_warp;
  const int k_end = min(gK, k_start + k_per_warp);

  ACC_T acc[MROWS][VEC];
#pragma unroll
  for (int m = 0; m < MROWS; ++m) {
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      acc[m][i] = (ACC_T)0;
    }
  }

  const bool full = (n0 + VEC) <= gN;
  if (full) {
    int k = k_start;
    for (; k + 4 <= k_end; k += 4) {
      Vec b0 = *((const device Vec*)(&B[(k + 0) * gLdb + n0]));
      Vec b1 = *((const device Vec*)(&B[(k + 1) * gLdb + n0]));
      Vec b2 = *((const device Vec*)(&B[(k + 2) * gLdb + n0]));
      Vec b3 = *((const device Vec*)(&B[(k + 3) * gLdb + n0]));
#pragma unroll
      for (int m = 0; m < MROWS; ++m) {
        ACC_T x0 = (ACC_T)Xr[m][k + 0], x1 = (ACC_T)Xr[m][k + 1];
        ACC_T x2 = (ACC_T)Xr[m][k + 2], x3 = (ACC_T)Xr[m][k + 3];
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          acc[m][i] += (ACC_T)b0.v[i] * x0;
          acc[m][i] += (ACC_T)b1.v[i] * x1;
          acc[m][i] += (ACC_T)b2.v[i] * x2;
          acc[m][i] += (ACC_T)b3.v[i] * x3;
        }
      }
    }
    for (; k < k_end; ++k) {
      Vec bv = *((const device Vec*)(&B[k * gLdb + n0]));
#pragma unroll
      for (int m = 0; m < MROWS; ++m) {
        ACC_T xk = (ACC_T)Xr[m][k];
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          acc[m][i] += (ACC_T)bv.v[i] * xk;
        }
      }
    }
  } else {
    // Edge block: scalar with per-column bounds for non-VEC-aligned N.
    for (int k = k_start; k < k_end; ++k) {
#pragma unroll
      for (int m = 0; m < MROWS; ++m) {
        ACC_T xk = (ACC_T)Xr[m][k];
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          const int n = n0 + i;
          if (n < gN) {
            acc[m][i] += (ACC_T)B[k * gLdb + n] * xk;
          }
        }
      }
    }
  }

#pragma unroll
  for (int m = 0; m < MROWS; ++m) {
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      partials[sgid][m][int(lane) * VEC + i] = acc[m][i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // First warp aggregates the NWARPS K-partials and writes Y for every real row.
  if (sgid == 0) {
    const ACC_T alpha = alpha_beta[0];
    const ACC_T beta = alpha_beta[1];
#pragma unroll
    for (int m = 0; m < MROWS; ++m) {
      if (m >= gM) {
        continue;
      }
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        const int cc = int(lane) * VEC + i;
        const int n = col0 + cc;
        if (n < gN) {
          ACC_T s = (ACC_T)0;
#pragma unroll
          for (int w = 0; w < NWARPS; ++w) {
            s += partials[w][m][cc];
          }
          Y[m * gLdy + n] = apply_epilogue<EPI, DT, ACC_T>(
              s, m, n, self, gP.self_r, gP.self_c, alpha, beta);
        }
      }
    }
  }
}

// Column-major B (x @ W.t()): a warp reduces one column over K via simd_sum.
// NCOLS>1 reuses each X row across NCOLS columns. X and B vectorize over K.
template <typename DT, int MROWS, int VEC, int NCOLS, GemmEpilogue EPI>
kernel void gemv_bt_t(
    device const DT* B [[buffer(0)]],
    device const DT* X [[buffer(1)]],
    device DT* Y [[buffer(2)]],
    constant GemvBtDims& gP [[buffer(3)]],
    device const DT* self [[buffer(4)]],
    constant ::c10::metal::array<::c10::metal::opmath_t<DT>, 2>& alpha_beta
    [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  using ACC_T = ::c10::metal::opmath_t<DT>;
  using Vec = GemmVec<DT, VEC>;
  constexpr int NWARPS = gemv_bt_nwarps(MROWS, VEC);
  constexpr int KS = 32 * VEC;

  const int gM = gP.M, gN = gP.N, gK = gP.K;
  const int gLdb = gP.ldb, gLdx = gP.ldx, gLdy = gP.ldy;
  B += (int64_t)tgid.z * gP.batch_b;
  X += (int64_t)tgid.z * gP.batch_x;
  Y += (int64_t)tgid.z * gP.batch_y;
  self += (int64_t)tgid.z * gP.batch_self;
  const ACC_T alpha = alpha_beta[0];
  const ACC_T beta = alpha_beta[1];

  const device DT* Xr[MROWS];
#pragma unroll
  for (int m = 0; m < MROWS; ++m) {
    const int mm = (m < gM) ? m : (gM - 1);
    Xr[m] = X + (int64_t)mm * gLdx;
  }

  if (NCOLS == 1) {
    const int n = int(tgid.x) * NWARPS + int(sgid);
    if (n >= gN) {
      return; // warp-uniform: no threadgroup barrier on this path
    }
    const device DT* Bn = &B[(int64_t)n * gLdb]; // column n, K-contiguous
    ACC_T acc[MROWS];
#pragma unroll
    for (int m = 0; m < MROWS; ++m) {
      acc[m] = (ACC_T)0;
    }
    int k = int(lane) * VEC;
    for (; k + 3 * KS + VEC <= gK; k += 4 * KS) {
      Vec w0 = *((const device Vec*)(&Bn[k]));
      Vec w1 = *((const device Vec*)(&Bn[k + KS]));
      Vec w2 = *((const device Vec*)(&Bn[k + 2 * KS]));
      Vec w3 = *((const device Vec*)(&Bn[k + 3 * KS]));
#pragma unroll
      for (int m = 0; m < MROWS; ++m) {
        Vec x0 = *((const device Vec*)(&Xr[m][k]));
        Vec x1 = *((const device Vec*)(&Xr[m][k + KS]));
        Vec x2 = *((const device Vec*)(&Xr[m][k + 2 * KS]));
        Vec x3 = *((const device Vec*)(&Xr[m][k + 3 * KS]));
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          acc[m] += (ACC_T)w0.v[i] * (ACC_T)x0.v[i] + (ACC_T)w1.v[i] * (ACC_T)x1.v[i] +
              (ACC_T)w2.v[i] * (ACC_T)x2.v[i] + (ACC_T)w3.v[i] * (ACC_T)x3.v[i];
        }
      }
    }
    for (; k + VEC <= gK; k += KS) {
      Vec wv = *((const device Vec*)(&Bn[k]));
#pragma unroll
      for (int m = 0; m < MROWS; ++m) {
        Vec xv = *((const device Vec*)(&Xr[m][k]));
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          acc[m] += (ACC_T)wv.v[i] * (ACC_T)xv.v[i];
        }
      }
    }
    if (lane == 0) { // scalar K%VEC tail; simd_sum folds it in
      int kk = (gK / VEC) * VEC;
      for (; kk < gK; ++kk) {
        ACC_T wk = (ACC_T)Bn[kk];
#pragma unroll
        for (int m = 0; m < MROWS; ++m) {
          acc[m] += wk * (ACC_T)Xr[m][kk];
        }
      }
    }
#pragma unroll
    for (int m = 0; m < MROWS; ++m) {
      ACC_T s = simd_sum(acc[m]);
      if (lane == 0 && m < gM) {
        Y[m * gLdy + n] = apply_epilogue<EPI, DT, ACC_T>(
            s, m, n, self, gP.self_r, gP.self_c, alpha, beta);
      }
    }
  } else {
    // Register-blocked over output columns: one X load feeds NCOLS columns.
    const int nbase = (int(tgid.x) * NWARPS + int(sgid)) * NCOLS;
    if (nbase >= gN) {
      return;
    }
    const int ncv = min(NCOLS, gN - nbase); // valid columns in this block (edge)
    ACC_T acc[MROWS][NCOLS];
#pragma unroll
    for (int m = 0; m < MROWS; ++m) {
#pragma unroll
      for (int c = 0; c < NCOLS; ++c) {
        acc[m][c] = (ACC_T)0;
      }
    }
    for (int k = int(lane) * VEC; k + VEC <= gK; k += KS) {
      Vec w[NCOLS];
#pragma unroll
      for (int c = 0; c < NCOLS; ++c) {
        if (c < ncv) {
          w[c] = *((const device Vec*)(&B[(int64_t)(nbase + c) * gLdb + k]));
        }
      }
#pragma unroll
      for (int m = 0; m < MROWS; ++m) {
        Vec xv = *((const device Vec*)(&Xr[m][k]));
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) {
#pragma unroll
          for (int i = 0; i < VEC; ++i) {
            acc[m][c] += (ACC_T)w[c].v[i] * (ACC_T)xv.v[i];
          }
        }
      }
    }
    if (lane == 0) { // scalar K%VEC tail
      int kk = (gK / VEC) * VEC;
      for (; kk < gK; ++kk) {
#pragma unroll
        for (int m = 0; m < MROWS; ++m) {
          ACC_T xk = (ACC_T)Xr[m][kk];
#pragma unroll
          for (int c = 0; c < NCOLS; ++c) {
            if (c < ncv) {
              acc[m][c] += (ACC_T)B[(int64_t)(nbase + c) * gLdb + kk] * xk;
            }
          }
        }
      }
    }
#pragma unroll
    for (int m = 0; m < MROWS; ++m) {
#pragma unroll
      for (int c = 0; c < NCOLS; ++c) {
        ACC_T s = simd_sum(acc[m][c]);
        if (lane == 0 && c < ncv && m < gM) {
          const int n = nbase + c;
          Y[m * gLdy + n] = apply_epilogue<EPI, DT, ACC_T>(
              s, m, n, self, gP.self_r, gP.self_c, alpha, beta);
        }
      }
    }
  }
}

} // namespace at_gemm
