#pragma once
// gemm_int.h - register-tiled integer GEMM. The simdgroup_matrix unit and the M5
// tensor unit are float-only, so integer matmul is done with plain MACs: each
// thread owns a TM x TN micro-tile, accumulating in ACC_T (>= OUT_T width) and
// truncating to OUT_T on store. Two's-complement wrap makes accumulate-wide-then-
// truncate bit-exact with torch's narrow wrapping (mod 2^n is associative), so
// ACC_T = opmath_t<DT> (int for <=32-bit, long for 64-bit) is always sufficient.
//
// Fully-templated port of metalBLAS int_gemm.h: tile sizes / transpose / epilogue
// / batch are template params; bounds are runtime checks. Shares GemmDimsStrided
// and apply_epilogue with simd_gemm so the host binding path is identical.
#include <ATen/native/mps/kernels/gemm_common.h>
#include <metal_stdlib>

using namespace metal;

namespace at_gemm {

template <
    typename DT,
    int BM,
    int BN,
    int BK,
    int TX,
    int TY,
    bool TRANS_A,
    bool TRANS_B,
    GemmEpilogue EPI,
    bool BATCHED>
kernel void int_gemm(
    device const DT* A [[buffer(0)]],
    device const DT* B [[buffer(1)]],
    device DT* C [[buffer(2)]],
    constant GemmDimsStrided& gP [[buffer(3)]],
    device const DT* self [[buffer(4)]],
    constant ::c10::metal::array<::c10::metal::opmath_t<DT>, 2>& alpha_beta
    [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  using ACC_T = ::c10::metal::opmath_t<DT>;
  constexpr int TGP_SIZE = TX * TY;
  constexpr int TM = BM / TY; // rows each thread owns
  constexpr int TN = BN / TX; // cols each thread owns

  static_assert(BM % TY == 0, "BM must be a multiple of TY");
  static_assert(BN % TX == 0, "BN must be a multiple of TX");

  if IF_CONSTEXPR (BATCHED) {
    int b = int(tgid.z);
    A += (size_t)b * gP.batch_a;
    B += (size_t)b * gP.batch_b;
    C += (size_t)b * gP.batch_c;
    if IF_CONSTEXPR (EPI == GemmEpilogue::AlphaBeta) {
      self += (size_t)b * gP.batch_self;
    }
  }

  const int gM = gP.M, gN = gP.N, gK = gP.K;
  const int gLda = gP.lda, gLdb = gP.ldb, gLdc = gP.ldc;

  // As stored transposed [BK][BM] (a contiguous BM slab per k); Bs [BK][BN].
  // Out-of-bounds loads zero-fill so the compute loop is branch-free over BK.
  threadgroup DT As[BK * BM];
  threadgroup DT Bs[BK * BN];

  const int m_block = int(tgid.y) * BM;
  const int n_block = int(tgid.x) * BN;

  const int ty = int(tid) / TX; // this thread's row group
  const int tx = int(tid) % TX; // this thread's col group
  const int row0 = ty * TM;
  const int col0 = tx * TN;

  ACC_T acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      acc[i][j] = (ACC_T)0;
    }
  }

  const int ktiles = (gK + BK - 1) / BK;
  for (int kt = 0; kt < ktiles; ++kt) {
    const int k_base = kt * BK;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooperative load of the A tile into As[k*BM + m] (transposed).
    for (int pos = int(tid); pos < BM * BK; pos += TGP_SIZE) {
      int m = pos % BM;
      int k = pos / BM;
      int gm = m_block + m;
      int gk = k_base + k;
      DT v = (DT)0;
      if (gm < gM && gk < gK) {
        if IF_CONSTEXPR (TRANS_A) {
          v = A[gk * gLda + gm]; // A col-major: elem(m,k) = A[k*lda + m]
        } else {
          v = A[gm * gLda + gk]; // A row-major: elem(m,k) = A[m*lda + k]
        }
      }
      As[k * BM + m] = v;
    }
    // Cooperative load of the B tile into Bs[k*BN + n].
    for (int pos = int(tid); pos < BK * BN; pos += TGP_SIZE) {
      int n = pos % BN;
      int k = pos / BN;
      int gk = k_base + k;
      int gn = n_block + n;
      DT v = (DT)0;
      if (gk < gK && gn < gN) {
        if IF_CONSTEXPR (TRANS_B) {
          v = B[gn * gLdb + gk]; // B col-major: elem(k,n) = B[n*ldb + k]
        } else {
          v = B[gk * gLdb + gn]; // B row-major: elem(k,n) = B[k*ldb + n]
        }
      }
      Bs[k * BN + n] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      DT av[TM];
      DT bv[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        av[i] = As[kk * BM + row0 + i];
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        bv[j] = Bs[kk * BN + col0 + j];
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] += (ACC_T)av[i] * (ACC_T)bv[j];
        }
      }
    }
  }

  ACC_T alpha = alpha_beta[0];
  ACC_T beta = alpha_beta[1];

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int gm = m_block + row0 + i;
    if (gm >= gM) {
      continue;
    }
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int gn = n_block + col0 + j;
      if (gn < gN) {
        C[gm * gLdc + gn] = apply_epilogue<EPI, DT, ACC_T>(
            acc[i][j], gm, gn, self, gP.self_r, gP.self_c, alpha, beta);
      }
    }
  }
}

} // namespace at_gemm
