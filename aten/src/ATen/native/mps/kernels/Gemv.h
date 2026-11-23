// Rank-1 GEMV kernels (M==1 xor N==1): gemv_t (output along matrix columns,
// simdgroups split K), gemv_t2d (gemv_t with a 2D lane layout and 16-byte
// loads) and gemv_nt (one simdgroup per output row), each instantiated over
// the MB_GEMV_* config matrix for float/half/bfloat.
#pragma once

template <typename DT, int VEC, bool XC, typename IDX>
inline ::c10::metal::aligned_vector<DT, VEC> load_x(device const DT* x, IDX k, IDX xs) {
  using Vec = ::c10::metal::aligned_vector<DT, VEC>;
  if IF_CONSTEXPR (XC) {
    return *((const device Vec*)(&x[k]));
  } else {
    Vec r;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      r.v[i] = x[(k + i) * xs];
    }
    return r;
  }
}

template <typename DT, int VEC, bool STRIDED, typename IDX>
inline ::c10::metal::aligned_vector<DT, VEC> load_matrix(device const DT* matrix, IDX offset, IDX stride) {
  using Vec = ::c10::metal::aligned_vector<DT, VEC>;
  if IF_CONSTEXPR (!STRIDED) {
    return *((const device Vec*)(&matrix[offset]));
  } else {
    Vec result;
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      result.v[i] = matrix[(offset + i) * stride];
    }
    return result;
  }
}

template <typename DT, bool STRIDED, typename IDX>
inline DT load_matrix_element(device const DT* matrix, IDX offset, IDX stride) {
  if IF_CONSTEXPR (STRIDED) {
    return matrix[offset * stride];
  } else {
    return matrix[offset];
  }
}

// Epilogue applied to one output element, cast to OUT_T. beta==0 must not read
// bias (may be uninitialized/NaN; matches addmm semantics).
template <GemmEpilogue EPI, typename OUT_T, typename ACC_T, typename IDX>
inline OUT_T apply_epilogue(ACC_T acc,
                            IDX r,
                            IDX c,
                            device const OUT_T* bias,
                            IDX bias_r,
                            IDX bias_c,
                            float alpha,
                            float beta) {
  using op_t = ::c10::metal::opmath_t<OUT_T>;
  op_t v = static_cast<op_t>(acc);
  if IF_CONSTEXPR (EPI == GemmEpilogue::Bias) {
    v = alpha * v;
    if (beta != op_t(0)) {
      v += beta * static_cast<op_t>(bias[r * bias_r + c * bias_c]);
    }
  }
  return static_cast<OUT_T>(v);
}

// y = x @ B, B is (K, N) row-major. Lanes own VEC columns (a coalesced line);
// NSIMD simdgroups split K and reduce in threadgroup memory.
template <typename DT, int NSIMD, int VEC, GemmEpilogue EPI, bool STRIDED = false, typename IDX = int>
kernel void gemv_t(device const DT* B [[buffer(0)]],
                   device const DT* x [[buffer(1)]],
                   device DT* y [[buffer(2)]],
                   constant GemvDims& gP [[buffer(3)]],
                   device const DT* bias [[buffer(4)]],
                   constant ::c10::metal::array<float, 2>& alpha_beta [[buffer(5)]],
                   uint3 tgid [[threadgroup_position_in_grid]],
                   uint sgid [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
  using ACC_T = ::c10::metal::opmath_t<DT>;
  using Vec = ::c10::metal::aligned_vector<DT, VEC>;
  constexpr int BLOCK_N = 32 * VEC;
  const IDX gN = IDX(gP.n), gK = IDX(gP.K), gLdb = IDX(gP.ld);
  const IDX gMs = IDX(gP.ms), gXs = IDX(gP.xs);
  // +1 pad: the reduce reads partials[lane][cc] down a column; an odd row
  // stride keeps those 32 accesses on distinct banks.
  threadgroup ACC_T partials[NSIMD][BLOCK_N + 1];

  const IDX col0 = IDX(tgid.x) * BLOCK_N;
  const IDX n0 = col0 + IDX(lane) * VEC;

  const IDX k_per_simd = (gK + NSIMD - 1) / NSIMD;
  const IDX k_start = IDX(sgid) * k_per_simd;
  const IDX k_end = min(gK, k_start + k_per_simd);

  ACC_T acc[VEC];
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    acc[i] = (ACC_T)0;
  }

  const bool full = (n0 + VEC) <= gN;
  if (full) {
    IDX k = k_start;
    for (; k + 4 <= k_end; k += 4) {
      Vec b0 = load_matrix<DT, VEC, STRIDED>(&B[(k + 0) * gLdb], n0, gMs);
      Vec b1 = load_matrix<DT, VEC, STRIDED>(&B[(k + 1) * gLdb], n0, gMs);
      Vec b2 = load_matrix<DT, VEC, STRIDED>(&B[(k + 2) * gLdb], n0, gMs);
      Vec b3 = load_matrix<DT, VEC, STRIDED>(&B[(k + 3) * gLdb], n0, gMs);
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
      Vec bv = load_matrix<DT, VEC, STRIDED>(&B[k * gLdb], n0, gMs);
      ACC_T xk = (ACC_T)x[k * gXs];
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        acc[i] += (ACC_T)bv.v[i] * xk;
      }
    }
  } else {
    for (IDX k = k_start; k < k_end; ++k) {
      ACC_T xk = (ACC_T)x[k * gXs];
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        IDX n = n0 + i;
        if (n < gN) {
          acc[i] += (ACC_T)load_matrix_element<DT, STRIDED>(&B[k * gLdb], n, gMs) * xk;
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    partials[sgid][int(lane) * VEC + i] = acc[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Cross-simdgroup reduce: every simdgroup owns BLOCK_N/NSIMD columns and
  // simd_sums the NSIMD partials of each (lanes >= NSIMD contribute zero).
  const ACC_T alpha = alpha_beta[0];
  const ACC_T beta = alpha_beta[1];
  for (int cc = int(sgid); cc < BLOCK_N; cc += NSIMD) {
    ACC_T v = int(lane) < NSIMD ? partials[lane][cc] : (ACC_T)0;
    v = simd_sum(v);
    const IDX n = col0 + cc;
    if (lane == 0 && n < gN) {
      y[n] = apply_epilogue<EPI, DT, ACC_T>(v, IDX(0), n, bias, IDX(gP.bias_r), IDX(gP.bias_c), alpha, beta);
    }
  }
}

// gemv_t with a 2D lane layout (KQ k-sublanes x C=32/KQ column groups), each
// lane loading a full 16-byte vector: up to 8x fewer/wider loads, which holds
// up under reduced GPU clocks. Requires ld and offset VEC-aligned (host
// checks).
template <typename DT, int NSIMD, int KQ, GemmEpilogue EPI, typename IDX = int>
kernel void gemv_t2d(device const DT* B [[buffer(0)]],
                     device const DT* x [[buffer(1)]],
                     device DT* y [[buffer(2)]],
                     constant GemvDims& gP [[buffer(3)]],
                     device const DT* bias [[buffer(4)]],
                     constant ::c10::metal::array<float, 2>& alpha_beta [[buffer(5)]],
                     uint3 tgid [[threadgroup_position_in_grid]],
                     uint sgid [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  using ACC_T = ::c10::metal::opmath_t<DT>;
  constexpr int VEC = 16 / sizeof(DT);
  constexpr int C = 32 / KQ;
  constexpr int BLOCK_N = C * VEC;
  using Vec = ::c10::metal::aligned_vector<DT, VEC>;
  threadgroup ACC_T partials[NSIMD][BLOCK_N + 1];
  const IDX gN = IDX(gP.n), gK = IDX(gP.K), gLdb = IDX(gP.ld), gXs = IDX(gP.xs);
  const IDX col0 = IDX(tgid.x) * BLOCK_N;
  const int kq = int(lane) / C;
  const int cl = int(lane) % C;
  const IDX n0 = col0 + cl * VEC;
  const IDX kps = (gK + NSIMD - 1) / NSIMD;
  const IDX k_start = IDX(sgid) * kps;
  const IDX k_end = min(gK, k_start + kps);
  ACC_T acc[VEC];
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    acc[i] = (ACC_T)0;
  }
  if ((n0 + VEC) <= gN) {
    IDX k = k_start + kq;
    for (; k + 3 * KQ < k_end; k += 4 * KQ) {
      Vec b0 = *((const device Vec*)(&B[(k + 0 * KQ) * gLdb + n0]));
      Vec b1 = *((const device Vec*)(&B[(k + 1 * KQ) * gLdb + n0]));
      Vec b2 = *((const device Vec*)(&B[(k + 2 * KQ) * gLdb + n0]));
      Vec b3 = *((const device Vec*)(&B[(k + 3 * KQ) * gLdb + n0]));
      ACC_T x0 = (ACC_T)x[(k + 0 * KQ) * gXs];
      ACC_T x1 = (ACC_T)x[(k + 1 * KQ) * gXs];
      ACC_T x2 = (ACC_T)x[(k + 2 * KQ) * gXs];
      ACC_T x3 = (ACC_T)x[(k + 3 * KQ) * gXs];
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        acc[i] += (ACC_T)b0.v[i] * x0;
        acc[i] += (ACC_T)b1.v[i] * x1;
        acc[i] += (ACC_T)b2.v[i] * x2;
        acc[i] += (ACC_T)b3.v[i] * x3;
      }
    }
    for (; k < k_end; k += KQ) {
      Vec bv = *((const device Vec*)(&B[k * gLdb + n0]));
      ACC_T xk = (ACC_T)x[k * gXs];
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        acc[i] += (ACC_T)bv.v[i] * xk;
      }
    }
  } else {
    for (IDX k = k_start + kq; k < k_end; k += KQ) {
      ACC_T xk = (ACC_T)x[k * gXs];
#pragma unroll
      for (int i = 0; i < VEC; ++i) {
        IDX n = n0 + i;
        if (n < gN) {
          acc[i] += (ACC_T)B[k * gLdb + n] * xk;
        }
      }
    }
  }
  // Fold the KQ k-sublanes of each column group (lane stride C).
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    for (uint off = 16; off >= uint(C); off >>= 1) {
      acc[i] += simd_shuffle_down(acc[i], off);
    }
  }
  if (kq == 0) {
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      partials[sgid][cl * VEC + i] = acc[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const ACC_T alpha = alpha_beta[0];
  const ACC_T beta = alpha_beta[1];
  for (int cc = int(sgid); cc < BLOCK_N; cc += NSIMD) {
    ACC_T v = int(lane) < NSIMD ? partials[lane][cc] : (ACC_T)0;
    v = simd_sum(v);
    const IDX n = col0 + cc;
    if (lane == 0 && n < gN) {
      y[n] = apply_epilogue<EPI, DT, ACC_T>(v, IDX(0), n, bias, IDX(gP.bias_r), IDX(gP.bias_c), alpha, beta);
    }
  }
}

// y = A @ x, A is (M, K) row-major. Each simdgroup owns one row; lanes stride
// K and reduce with simd_sum. XC: x is unit-stride and VEC-aligned.
template <typename DT, int NSIMD, int VEC, GemmEpilogue EPI, bool XC, bool STRIDED = false, typename IDX = int>
kernel void gemv_nt(device const DT* A [[buffer(0)]],
                    device const DT* x [[buffer(1)]],
                    device DT* y [[buffer(2)]],
                    constant GemvDims& gP [[buffer(3)]],
                    device const DT* bias [[buffer(4)]],
                    constant ::c10::metal::array<float, 2>& alpha_beta [[buffer(5)]],
                    uint3 tgid [[threadgroup_position_in_grid]],
                    uint sgid [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  using ACC_T = ::c10::metal::opmath_t<DT>;
  using Vec = ::c10::metal::aligned_vector<DT, VEC>;
  const IDX gM = IDX(gP.n), gK = IDX(gP.K), gLda = IDX(gP.ld);
  const IDX gMs = IDX(gP.ms), gXs = IDX(gP.xs);
  const int K_STRIDE = 32 * VEC;

  const IDX row = IDX(tgid.x) * NSIMD + IDX(sgid);
  if (row >= gM) {
    return;
  }
  const device DT* Arow = &A[row * gLda];
  ACC_T acc = (ACC_T)0;
  IDX k = IDX(lane) * VEC;
  for (; k + 3 * K_STRIDE + VEC <= gK; k += 4 * K_STRIDE) {
    Vec x0 = load_x<DT, VEC, XC>(x, k + 0 * K_STRIDE, gXs);
    Vec x1 = load_x<DT, VEC, XC>(x, k + 1 * K_STRIDE, gXs);
    Vec x2 = load_x<DT, VEC, XC>(x, k + 2 * K_STRIDE, gXs);
    Vec x3 = load_x<DT, VEC, XC>(x, k + 3 * K_STRIDE, gXs);
    Vec a0 = load_matrix<DT, VEC, STRIDED>(Arow, k + 0 * K_STRIDE, gMs);
    Vec a1 = load_matrix<DT, VEC, STRIDED>(Arow, k + 1 * K_STRIDE, gMs);
    Vec a2 = load_matrix<DT, VEC, STRIDED>(Arow, k + 2 * K_STRIDE, gMs);
    Vec a3 = load_matrix<DT, VEC, STRIDED>(Arow, k + 3 * K_STRIDE, gMs);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      acc += (ACC_T)a0.v[i] * (ACC_T)x0.v[i];
      acc += (ACC_T)a1.v[i] * (ACC_T)x1.v[i];
      acc += (ACC_T)a2.v[i] * (ACC_T)x2.v[i];
      acc += (ACC_T)a3.v[i] * (ACC_T)x3.v[i];
    }
  }
  for (; k + VEC <= gK; k += K_STRIDE) {
    Vec xv = load_x<DT, VEC, XC>(x, k, gXs);
    Vec av = load_matrix<DT, VEC, STRIDED>(Arow, k, gMs);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
      acc += (ACC_T)av.v[i] * (ACC_T)xv.v[i];
    }
  }
  if (lane == 0) {
    for (IDX kk = (gK / VEC) * VEC; kk < gK; ++kk) {
      acc += (ACC_T)load_matrix_element<DT, STRIDED>(Arow, kk, gMs) * (ACC_T)x[kk * gXs];
    }
  }

  const ACC_T s = simd_sum(acc);
  if (lane == 0) {
    y[row] = apply_epilogue<EPI, DT, ACC_T>(
        s, row, IDX(0), bias, IDX(gP.bias_r), IDX(gP.bias_c), alpha_beta[0], alpha_beta[1]);
  }
}

// Explicit instantiations (host_name = dispatch key): exactly the configs the
// GemvPolicy can emit, closed under clamp_vec (vec halved when the matrix is
// not vec-aligned) and the T2D->Standard misalignment fallback.
#define MB_GEMV_T(DT, NSIMD, VEC, EN, EV, SN, SV)                                       \
  template[[host_name("gemv_t_" #DT "_" #NSIMD "_" #VEC "_" #EN SN)]] kernel void       \
  gemv_t<DT, NSIMD, VEC, GemmEpilogue::EV, SV>(device const DT*,                        \
                                               device const DT*,                        \
                                               device DT*,                              \
                                               constant GemvDims&,                      \
                                               device const DT*,                        \
                                               constant ::c10::metal::array<float, 2>&, \
                                               uint3,                                   \
                                               uint,                                    \
                                               uint);
#define MB_GEMV_T_E(DT, NSIMD, VEC)                       \
  MB_GEMV_T(DT, NSIMD, VEC, none, None, "", false)        \
  MB_GEMV_T(DT, NSIMD, VEC, ab, Bias, "", false)          \
  MB_GEMV_T(DT, NSIMD, VEC, none, None, "_strided", true) \
  MB_GEMV_T(DT, NSIMD, VEC, ab, Bias, "_strided", true)

#define MB_GEMV_T2D(DT, NSIMD, KQ, EN, EV)                                                       \
  template [[host_name("gemv_t2d_" #DT "_" #NSIMD "_" #KQ "_" #EN)]]                             \
  kernel void gemv_t2d<DT, NSIMD, KQ, GemmEpilogue::EV>(device const DT*,                        \
                                                        device const DT*,                        \
                                                        device DT*,                              \
                                                        constant GemvDims&,                      \
                                                        device const DT*,                        \
                                                        constant ::c10::metal::array<float, 2>&, \
                                                        uint3,                                   \
                                                        uint,                                    \
                                                        uint);
#define MB_GEMV_T2D_E(DT, NSIMD, KQ)     \
  MB_GEMV_T2D(DT, NSIMD, KQ, none, None) \
  MB_GEMV_T2D(DT, NSIMD, KQ, ab, Bias)

#define MB_GEMV_NT(DT, NSIMD, VEC, EN, EV, XN, XV, SN, SV)                                   \
  template[[host_name("gemv_nt_" #DT "_" #NSIMD "_" #VEC "_" #EN "_" #XN SN)]] kernel void   \
  gemv_nt<DT, NSIMD, VEC, GemmEpilogue::EV, XV, SV>(device const DT*,                        \
                                                    device const DT*,                        \
                                                    device DT*,                              \
                                                    constant GemvDims&,                      \
                                                    device const DT*,                        \
                                                    constant ::c10::metal::array<float, 2>&, \
                                                    uint3,                                   \
                                                    uint,                                    \
                                                    uint);
// VEC==1: a vector x load degenerates to a scalar one, strided-x only.
#define MB_GEMV_NT_E1(DT, NSIMD)                                    \
  MB_GEMV_NT(DT, NSIMD, 1, none, None, xs, false, "", false)        \
  MB_GEMV_NT(DT, NSIMD, 1, ab, Bias, xs, false, "", false)          \
  MB_GEMV_NT(DT, NSIMD, 1, none, None, xs, false, "_strided", true) \
  MB_GEMV_NT(DT, NSIMD, 1, ab, Bias, xs, false, "_strided", true)

#define MB_GEMV_NT_EV(DT, NSIMD, VEC)                                 \
  MB_GEMV_NT(DT, NSIMD, VEC, none, None, xs, false, "", false)        \
  MB_GEMV_NT(DT, NSIMD, VEC, ab, Bias, xs, false, "", false)          \
  MB_GEMV_NT(DT, NSIMD, VEC, none, None, xc, true, "", false)         \
  MB_GEMV_NT(DT, NSIMD, VEC, ab, Bias, xc, true, "", false)           \
  MB_GEMV_NT(DT, NSIMD, VEC, none, None, xs, false, "_strided", true) \
  MB_GEMV_NT(DT, NSIMD, VEC, ab, Bias, xs, false, "_strided", true)   \
  MB_GEMV_NT(DT, NSIMD, VEC, none, None, xc, true, "_strided", true)  \
  MB_GEMV_NT(DT, NSIMD, VEC, ab, Bias, xc, true, "_strided", true)

// fp32 t: nsimd tiers {4, 8, 16, 32} at vec 2 (clamped to 1), scalar columns
// {32, 1}; nt: nsimd {4, 16} at vec 4 (clamped to 2/1); t2d at kq=4.
#define MB_GEMV_FLOAT(DT)  \
  MB_GEMV_T_E(DT, 4, 1)    \
  MB_GEMV_T_E(DT, 8, 1)    \
  MB_GEMV_T_E(DT, 16, 1)   \
  MB_GEMV_T_E(DT, 32, 1)   \
  MB_GEMV_T_E(DT, 4, 2)    \
  MB_GEMV_T_E(DT, 8, 2)    \
  MB_GEMV_T_E(DT, 16, 2)   \
  MB_GEMV_T_E(DT, 32, 2)   \
  MB_GEMV_NT_E1(DT, 4)     \
  MB_GEMV_NT_E1(DT, 16)    \
  MB_GEMV_NT_EV(DT, 4, 2)  \
  MB_GEMV_NT_EV(DT, 16, 2) \
  MB_GEMV_NT_EV(DT, 4, 4)  \
  MB_GEMV_NT_EV(DT, 16, 4) \
  MB_GEMV_T2D_E(DT, 16, 4)

// half/bf16 t: nsimd {16, 32} at vec 2 (clamped to 1); nt: nsimd {4, 8} at
// vec 8 (clamped to 4/2/1); t2d at kq=8.
#define MB_GEMV_LP(DT)    \
  MB_GEMV_T_E(DT, 16, 1)  \
  MB_GEMV_T_E(DT, 32, 1)  \
  MB_GEMV_T_E(DT, 16, 2)  \
  MB_GEMV_T_E(DT, 32, 2)  \
  MB_GEMV_NT_E1(DT, 4)    \
  MB_GEMV_NT_E1(DT, 8)    \
  MB_GEMV_NT_EV(DT, 4, 2) \
  MB_GEMV_NT_EV(DT, 4, 4) \
  MB_GEMV_NT_EV(DT, 4, 8) \
  MB_GEMV_NT_EV(DT, 8, 2) \
  MB_GEMV_NT_EV(DT, 8, 4) \
  MB_GEMV_NT_EV(DT, 8, 8) \
  MB_GEMV_T2D_E(DT, 16, 8)

MB_GEMV_FLOAT(float)
MB_GEMV_LP(half)
MB_GEMV_LP(bfloat)

// 64-bit-index variants for operands whose element offsets overflow int32
// (matrix numel > 2^31, or huge view strides). Such matrices are DRAM-bound,
// so the host skips the policy and uses one fixed config per side, closed
// under the alignment clamps: gemv_t {16, 2}, gemv_nt {8, 8} (fp32 {8, 4});
// no t2d.
#define MB_GEMV_T_I64(DT, NSIMD, VEC, EN, EV, SN, SV)                                         \
  template[[host_name("gemv_t_" #DT "_" #NSIMD "_" #VEC "_" #EN SN "_i64")]] kernel void      \
  gemv_t<DT, NSIMD, VEC, GemmEpilogue::EV, SV, long>(device const DT*,                        \
                                                     device const DT*,                        \
                                                     device DT*,                              \
                                                     constant GemvDims&,                      \
                                                     device const DT*,                        \
                                                     constant ::c10::metal::array<float, 2>&, \
                                                     uint3,                                   \
                                                     uint,                                    \
                                                     uint);
#define MB_GEMV_T_I64_E(DT, NSIMD, VEC)                       \
  MB_GEMV_T_I64(DT, NSIMD, VEC, none, None, "", false)        \
  MB_GEMV_T_I64(DT, NSIMD, VEC, ab, Bias, "", false)          \
  MB_GEMV_T_I64(DT, NSIMD, VEC, none, None, "_strided", true) \
  MB_GEMV_T_I64(DT, NSIMD, VEC, ab, Bias, "_strided", true)

#define MB_GEMV_NT_I64(DT, NSIMD, VEC, EN, EV, XN, XV, SN, SV)                                     \
  template[[host_name("gemv_nt_" #DT "_" #NSIMD "_" #VEC "_" #EN "_" #XN SN "_i64")]] kernel void  \
  gemv_nt<DT, NSIMD, VEC, GemmEpilogue::EV, XV, SV, long>(device const DT*,                        \
                                                          device const DT*,                        \
                                                          device DT*,                              \
                                                          constant GemvDims&,                      \
                                                          device const DT*,                        \
                                                          constant ::c10::metal::array<float, 2>&, \
                                                          uint3,                                   \
                                                          uint,                                    \
                                                          uint);
#define MB_GEMV_NT_I64_E1(DT, NSIMD)                                    \
  MB_GEMV_NT_I64(DT, NSIMD, 1, none, None, xs, false, "", false)        \
  MB_GEMV_NT_I64(DT, NSIMD, 1, ab, Bias, xs, false, "", false)          \
  MB_GEMV_NT_I64(DT, NSIMD, 1, none, None, xs, false, "_strided", true) \
  MB_GEMV_NT_I64(DT, NSIMD, 1, ab, Bias, xs, false, "_strided", true)
#define MB_GEMV_NT_I64_EV(DT, NSIMD, VEC)                                 \
  MB_GEMV_NT_I64(DT, NSIMD, VEC, none, None, xs, false, "", false)        \
  MB_GEMV_NT_I64(DT, NSIMD, VEC, ab, Bias, xs, false, "", false)          \
  MB_GEMV_NT_I64(DT, NSIMD, VEC, none, None, xc, true, "", false)         \
  MB_GEMV_NT_I64(DT, NSIMD, VEC, ab, Bias, xc, true, "", false)           \
  MB_GEMV_NT_I64(DT, NSIMD, VEC, none, None, xs, false, "_strided", true) \
  MB_GEMV_NT_I64(DT, NSIMD, VEC, ab, Bias, xs, false, "_strided", true)   \
  MB_GEMV_NT_I64(DT, NSIMD, VEC, none, None, xc, true, "_strided", true)  \
  MB_GEMV_NT_I64(DT, NSIMD, VEC, ab, Bias, xc, true, "_strided", true)

#define MB_GEMV_I64_LP(DT)    \
  MB_GEMV_T_I64_E(DT, 16, 2)  \
  MB_GEMV_T_I64_E(DT, 16, 1)  \
  MB_GEMV_NT_I64_EV(DT, 8, 8) \
  MB_GEMV_NT_I64_EV(DT, 8, 4) \
  MB_GEMV_NT_I64_EV(DT, 8, 2) \
  MB_GEMV_NT_I64_E1(DT, 8)

MB_GEMV_I64_LP(half)
MB_GEMV_I64_LP(bfloat)
MB_GEMV_T_I64_E(float, 16, 2)
MB_GEMV_T_I64_E(float, 16, 1)
MB_GEMV_NT_I64_EV(float, 8, 4)
MB_GEMV_NT_I64_EV(float, 8, 2)
MB_GEMV_NT_I64_E1(float, 8)
