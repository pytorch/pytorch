#pragma once
// gemm_simd.h - portable simdgroup_matrix<T,8,8> tiled GEMM (Metal 3; no tensor
// unit). The universal fallback used on macOS < 26 / pre-Apple10 GPUs, and for
// transposed / strided inputs everywhere. Fully templated port of metalBLAS
// simd_gemm.h: tile sizes / transpose / epilogue / batch are template params;
// MN- and K-alignment are handled by runtime bounds checks (no template axis).
#include <ATen/native/mps/kernels/gemm_common.h>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

namespace at_gemm {

template <
    typename IN_T,
    int BM,
    int BK,
    int VEC,
    int LDA_TGP,
    int TGP_SIZE,
    bool TRANS_A>
static inline void simd_load_A(
    threadgroup IN_T* As,
    const device IN_T* A,
    int lda,
    int M,
    int a_row0,
    int a_col0,
    int tid,
    int kbound) {
  using Vec = GemmVec<IN_T, VEC>;
  constexpr int A_TCOLS = BK / VEC;
  constexpr int A_ROW_STEP = TGP_SIZE / A_TCOLS;
  int local_row0 = tid / A_TCOLS;
  int local_col0 = (tid % A_TCOLS) * VEC;
#pragma unroll
  for (int r = 0; r < BM; r += A_ROW_STEP) {
    int rl = local_row0 + r;
    if (rl >= BM) {
      break;
    }
    Vec acc;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      acc.v[i] = (IN_T)0;
    }
    if IF_CONSTEXPR (TRANS_A) {
      // A stored K x M (lda = stride along K): element (m, k) is A[k*lda + m].
      int gm = a_row0 + rl;
      bool m_ok = gm < M;
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        int gk = a_col0 + local_col0 + i;
        acc.v[i] = (m_ok && gk < kbound) ? A[gk * lda + gm] : (IN_T)0;
      }
    } else {
      bool m_ok = (a_row0 + rl) < M;
      int gc_k0 = a_col0 + local_col0;
      bool k_full = (gc_k0 + VEC) <= kbound;
      // VecF load needs lda VEC-aligned; else an unaligned load corrupts.
      if (m_ok && k_full && (lda % VEC) == 0) {
        acc = *((const device Vec*)(&A[(a_row0 + rl) * lda + gc_k0]));
      } else {
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          int gk = gc_k0 + i;
          acc.v[i] = (m_ok && gk < kbound) ? A[(a_row0 + rl) * lda + gk] : (IN_T)0;
        }
      }
    }
    *((threadgroup Vec*)(&As[rl * LDA_TGP + local_col0])) = acc;
  }
}

template <
    typename IN_T,
    int BN,
    int BK,
    int VEC,
    int LDB_TGP,
    int TGP_SIZE,
    bool TRANS_B>
static inline void simd_load_B(
    threadgroup IN_T* Bs,
    const device IN_T* B,
    int ldb,
    int N,
    int b_row0,
    int b_col0,
    int tid,
    int kbound) {
  using Vec = GemmVec<IN_T, VEC>;
  constexpr int B_TCOLS = BN / VEC;
  constexpr int B_ROW_STEP = TGP_SIZE / B_TCOLS;
  int local_row0 = tid / B_TCOLS;
  int local_col0 = (tid % B_TCOLS) * VEC;
  int n_global = b_col0 + local_col0;
#pragma unroll
  for (int r = 0; r < BK; r += B_ROW_STEP) {
    int rl = local_row0 + r;
    if (rl >= BK) {
      break;
    }
    Vec acc;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      acc.v[i] = (IN_T)0;
    }
    int gk = b_row0 + rl;
    if IF_CONSTEXPR (TRANS_B) {
      if (gk < kbound) {
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          int gn = n_global + i;
          acc.v[i] = gn < N ? B[gn * ldb + gk] : (IN_T)0;
        }
      }
    } else {
      bool n_full = (n_global + VEC) <= N;
      if (gk < kbound && n_full && (ldb % VEC) == 0) {
        acc = *((const device Vec*)(&B[gk * ldb + n_global]));
      } else if (gk < kbound) {
#pragma unroll
        for (int i = 0; i < VEC; ++i) {
          int gn = n_global + i;
          acc.v[i] = gn < N ? B[gk * ldb + gn] : (IN_T)0;
        }
      }
    }
    *((threadgroup Vec*)(&Bs[rl * LDB_TGP + local_col0])) = acc;
  }
}

template <
    typename IN_T,
    typename OUT_T,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool TRANS_A,
    bool TRANS_B,
    GemmEpilogue EPI,
    bool BATCHED>
kernel void simd_gemm(
    device const IN_T* A [[buffer(0)]],
    device const IN_T* B [[buffer(1)]],
    device OUT_T* C [[buffer(2)]],
    constant GemmDimsStrided& gP [[buffer(3)]],
    device const OUT_T* self [[buffer(4)]],
    constant ::c10::metal::array<::c10::metal::opmath_t<OUT_T>, 2>& alpha_beta
    [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  using ACC_T = float;
  constexpr int SG_SIZE = 32;
  constexpr int TGP_SIZE = WM * WN * SG_SIZE;
  constexpr int WT_M = BM / WM;
  constexpr int WT_N = BN / WN;
  constexpr int TM = WT_M / 8;
  constexpr int TN = WT_N / 8;
  constexpr int PAD = 16 / sizeof(IN_T);
  constexpr int LDA_TGP = BK + PAD;
  constexpr int LDB_TGP = BN + PAD;
  constexpr int VEC = 16 / sizeof(IN_T);

  static_assert(BM % (8 * WM) == 0, "BM must be a multiple of 8*WM");
  static_assert(BN % (8 * WN) == 0, "BN must be a multiple of 8*WN");
  static_assert(BK % 8 == 0, "BK must be a multiple of 8");
  static_assert(BK % VEC == 0, "BK must be divisible by VEC");
  static_assert(BN % VEC == 0, "BN must be divisible by VEC");

  int gM = gP.M, gN = gP.N, gK = gP.K;
  int gLda = gP.lda, gLdb = gP.ldb, gLdc = gP.ldc;

  if IF_CONSTEXPR (BATCHED) {
    int b = int(tgid.z);
    A += (size_t)b * gP.batch_a;
    B += (size_t)b * gP.batch_b;
    C += (size_t)b * gP.batch_c;
    if IF_CONSTEXPR (EPI == GemmEpilogue::AlphaBeta) {
      self += (size_t)b * gP.batch_self;
    }
  }

  threadgroup IN_T As[BM * LDA_TGP];
  threadgroup IN_T Bs[BK * LDB_TGP];

  int tid = int(sgid) * SG_SIZE + int(lane);

  int tiles_m = (gM + BM - 1) / BM;
  int tiles_n = (gN + BN - 1) / BN;
  int sw = gP.swizzle_log;
  int sw_mask = (1 << sw) - 1;
  int tgy = (int(tgid.y) << sw) | (int(tgid.x) & sw_mask);
  int tgx = int(tgid.x) >> sw;
  if (tgx >= tiles_n || tgy >= tiles_m) {
    return;
  }

  int m_block = tgy * BM;
  int n_block = tgx * BN;

  int warp_row = int(sgid) / WN;
  int warp_col = int(sgid) % WN;
  int warp_m = warp_row * WT_M;
  int warp_n = warp_col * WT_N;

  simdgroup_matrix<ACC_T, 8, 8> Cfrag[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      Cfrag[i][j] = simdgroup_matrix<ACC_T, 8, 8>(0);
    }
  }

  int k_tiles_full = gK / BK;
  int k_tail = gK - k_tiles_full * BK;

  for (int kt = 0; kt < k_tiles_full; ++kt) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simd_load_A<IN_T, BM, BK, VEC, LDA_TGP, TGP_SIZE, TRANS_A>(
        As, A, gLda, gM, m_block, kt * BK, tid, (kt + 1) * BK);
    simd_load_B<IN_T, BN, BK, VEC, LDB_TGP, TGP_SIZE, TRANS_B>(
        Bs, B, gLdb, gN, kt * BK, n_block, tid, (kt + 1) * BK);
    threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_matrix<IN_T, 8, 8> Afrag[TM];
      simdgroup_matrix<IN_T, 8, 8> Bfrag[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        simdgroup_load(Afrag[i], &As[(warp_m + i * 8) * LDA_TGP + kk], LDA_TGP);
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        simdgroup_load(Bfrag[j], &Bs[kk * LDB_TGP + warp_n + j * 8], LDB_TGP);
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          simdgroup_multiply_accumulate(
              Cfrag[i][j], Afrag[i], Bfrag[j], Cfrag[i][j]);
        }
      }
    }
  }

  if (k_tail > 0) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < BM * LDA_TGP; i += TGP_SIZE) {
      As[i] = (IN_T)0;
    }
    for (int i = tid; i < BK * LDB_TGP; i += TGP_SIZE) {
      Bs[i] = (IN_T)0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simd_load_A<IN_T, BM, BK, VEC, LDA_TGP, TGP_SIZE, TRANS_A>(
        As, A, gLda, gM, m_block, k_tiles_full * BK, tid, gK);
    simd_load_B<IN_T, BN, BK, VEC, LDB_TGP, TGP_SIZE, TRANS_B>(
        Bs, B, gLdb, gN, k_tiles_full * BK, n_block, tid, gK);
    threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll
    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_matrix<IN_T, 8, 8> Afrag[TM];
      simdgroup_matrix<IN_T, 8, 8> Bfrag[TN];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        simdgroup_load(Afrag[i], &As[(warp_m + i * 8) * LDA_TGP + kk], LDA_TGP);
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        simdgroup_load(Bfrag[j], &Bs[kk * LDB_TGP + warp_n + j * 8], LDB_TGP);
      }
#pragma unroll
      for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          simdgroup_multiply_accumulate(
              Cfrag[i][j], Afrag[i], Bfrag[j], Cfrag[i][j]);
        }
      }
    }
  }

  // (row, col) each lane owns within an 8x8 simdgroup_matrix.
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  ::c10::metal::opmath_t<OUT_T> alpha = alpha_beta[0];
  ::c10::metal::opmath_t<OUT_T> beta = alpha_beta[1];

#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      int row = m_block + warp_m + i * 8 + fm;
      int col = n_block + warp_n + j * 8 + fn;
      ACC_T te0 = Cfrag[i][j].thread_elements()[0];
      ACC_T te1 = Cfrag[i][j].thread_elements()[1];
      if (row < gM && col + 0 < gN) {
        C[row * gLdc + col + 0] = apply_epilogue<EPI, OUT_T, ACC_T>(
            te0, row, col + 0, self, gP.self_r, gP.self_c, alpha, beta);
      }
      if (row < gM && col + 1 < gN) {
        C[row * gLdc + col + 1] = apply_epilogue<EPI, OUT_T, ACC_T>(
            te1, row, col + 1, self, gP.self_r, gP.self_c, alpha, beta);
      }
    }
  }
}

} // namespace at_gemm
