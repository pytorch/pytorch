#include <ATen/native/mps/kernels/LinearAlgebra.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

// LU factorization with partial pivoting (mirrors LAPACK sgetrf), in place on a
// row-major fp32 (B, M, N) buffer. The host (lu_factor_panel_encode in
// LU.mm) drives a blocked right-looking schedule built from:
//   factorPanelLU / luStream* -> sgetf2 (unblocked panel, isamax pivoting)
//   laswpGatherLU             -> slaswp (apply a block's row interchanges)
//   trsmPanelLU               -> strsm  (unit-lower triangular solve)
//   gemmLU / gemmSimdLU       -> sgemm  (Schur update A22 -= L21 * U12)
//   transposeInPlaceLU        -> row-major factor to column-major LU output
// Buffer slots: (0) A in/out, (1) pivots (1-based), (2) info, (3) dims{M,N},
// (4) per-kernel params, (5) window descriptor, (6) streaming scratch.
// The LU kernels are templated over the element type: float (sgetrf) and
// float2/complex64 (cgetrf, host_name suffix _c64). Pivot magnitude is fabs
// for real and cabs1 = |re| + |im| for complex (LAPACK icamax); the complex
// elimination math routes through c10::metal::mul/div so the float path keeps
// its exact fma instruction sequence. Plain LU is purely algebraic, so the
// complex path uses NO conjugation anywhere.
//
// Type dispatch below uses `if IF_CONSTEXPR`, which lowers to a plain runtime
// `if` on Metal 3 (see c10/metal/common.h), so both arms are parsed and must
// type-check for every T. Where an arm is well-formed for only one element
// type -- the reciprocal, the pivot magnitude, and the float4 vectorized
// stores -- it lives in an overload instead of a branch.
inline float luPivotMag(float v) {
  return ::metal::fabs(v);
}
inline float luPivotMag(float2 v) {
  // cabs1 = |re| + |im| (LAPACK icamax)
  const float2 a = ::metal::precise::abs(v);
  return a.x + a.y;
}
inline float luRecip(float v) {
  return 1.0f / v;
}
inline float2 luRecip(float2 v) {
  return c10::metal::div(float2(1.0f, 0.0f), v);
}

// 4-wide vectorized store. The complex overloads are never selected at run
// time (the callers' vec4/aligned guards are false for complex), but they must
// exist so the call type-checks when T is float2.
inline void luStore4(device float* dst, thread const float* src) {
  *(device float4*)dst = float4(src[0], src[1], src[2], src[3]);
}
inline void luStore4(device float2* dst, thread const float2* src) {
  for (short i = 0; i < 4; i++) {
    dst[i] = src[i];
  }
}
inline void luStore4(device float* dst, threadgroup const float* src) {
  *(device float4*)dst = float4(src[0], src[1], src[2], src[3]);
}
inline void luStore4(device float2* dst, threadgroup const float2* src) {
  for (short i = 0; i < 4; i++) {
    dst[i] = src[i];
  }
}

// Unblocked 32-wide panel factor; each thread owns R rows, W = 32/R columns.
template <typename T, short R, short W>
kernel void factorPanelLU(
    device T* A [[buffer(0)]],
    device int* pivots [[buffer(1)]],
    device int* info [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    constant uint4& params [[buffer(4)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]],
    uint warp_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint tid = tid3.x;
  const uint G = tpg.x;
  const uint M = dims.x;
  const uint N = dims.y;
  const uint minMN = min(M, N);
  const uint d0 = params.x;
  const uint H = M - d0;
  const uint nb = min(uint(W), minMN - d0);
  device T* Ab = A + ulong(bid.x) * M * N;
  device int* pv = pivots + ulong(bid.x) * minMN;

  if (d0 == 0 && tid == 0) {
    info[bid.x] = 0;
  }

  threadgroup T pivBuf[W];
  threadgroup T rowJBuf[W];
  threadgroup float wval[32];
  threadgroup uint widx[32];
  threadgroup uint sPiv[1];

  T row[R][W];
  const bool vec4 =
      !c10::metal::is_complex_v<T> && ((N % 4u) == 0) && (nb == W);
#pragma unroll
  for (short r = 0; r < R; r++) {
    const uint lr = tid + uint(r) * G;
    if (lr < H) {
      device const T* src = Ab + ulong(d0 + lr) * N + d0;
      if IF_CONSTEXPR (c10::metal::is_complex_v<T>) {
#pragma unroll
        for (short c = 0; c < W; c++) {
          row[r][c] = (uint(c) < nb) ? src[c] : T(0.0f);
        }
      } else if (vec4) {
#pragma unroll
        for (short c = 0; c < W; c += 4) {
          const float4 v = *(device const float4*)(src + c);
          for (short ci = 0; ci < 4; ci++) {
            row[r][c + ci] = v[ci];
          }
        }
      } else {
#pragma unroll
        for (short c = 0; c < W; c++) {
          row[r][c] = (uint(c) < nb) ? src[c] : 0.0f;
        }
      }
    }
  }

  const uint nwarps = G / c10::metal::simdgroup_size;
  for (uint j = 0; j < nb; j++) {
    // local first-max over owned rows, then two-level argmax reduction with
    // smallest-index tiebreak (matches LAPACK isamax)
    float bv = -1.0f;
    uint bi = 0xffffffffu;
#pragma unroll
    for (short r = 0; r < R; r++) {
      const uint lr = tid + uint(r) * G;
      if (lr < H && lr >= j) {
        const float v = luPivotMag(row[r][j]);
        if (v > bv) {
          bv = v;
          bi = lr;
        }
      }
    }
    const float mv = simd_max(bv);
    const uint mi = simd_min((bv == mv) ? bi : 0xffffffffu);
    if (lane == 0) {
      wval[warp_id] = mv;
      widx[warp_id] = mi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp_id == 0) {
      const float v2 = (lane < nwarps) ? wval[lane] : -1.0f;
      const uint i2 = (lane < nwarps) ? widx[lane] : 0xffffffffu;
      const float m2 = simd_max(v2);
      uint p2 = simd_min((v2 == m2) ? i2 : 0xffffffffu);
      if (lane == 0) {
        if (p2 == 0xffffffffu) { // all-NaN column: pivot on j, NaN spreads
          p2 = j;
        }
        sPiv[0] = p2;
        pv[d0 + j] = int(d0 + p2 + 1); // 1-based like LAPACK
        if (m2 == 0.0f && info[bid.x] == 0) {
          info[bid.x] = int(d0 + j + 1);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint p = sPiv[0];

    // swap full rows j <-> p through smem; pivBuf doubles as the U row j
    // broadcast for the rank-1 update
    if (tid == j) {
#pragma unroll
      for (short c = 0; c < W; c++) {
        rowJBuf[c] = row[0][c];
      }
    }
#pragma unroll
    for (short r = 0; r < R; r++) {
      if (tid + uint(r) * G == p) {
#pragma unroll
        for (short c = 0; c < W; c++) {
          pivBuf[c] = row[r][c];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p != j) {
      if (tid == j) {
#pragma unroll
        for (short c = 0; c < W; c++) {
          row[0][c] = pivBuf[c];
        }
      }
#pragma unroll
      for (short r = 0; r < R; r++) {
        if (tid + uint(r) * G == p) {
#pragma unroll
          for (short c = 0; c < W; c++) {
            row[r][c] = rowJBuf[c];
          }
        }
      }
    }

    const T upiv = pivBuf[j];
    if (luPivotMag(upiv) != 0.0f) {
      const T rp = luRecip(upiv);
      // batch the smem loads ahead of the fma burst (in-order pipe)
      T uc[W];
#pragma unroll
      for (short c = 0; c < W; c++) {
        uc[c] = pivBuf[c];
      }
#pragma unroll
      for (short r = 0; r < R; r++) {
        const uint lr = tid + uint(r) * G;
        if (lr < H && lr > j) {
          const T l = c10::metal::mul(row[r][j], rp);
          row[r][j] = l;
#pragma unroll
          for (short c = 0; c < W; c++) {
            if (uint(c) > j) {
              if IF_CONSTEXPR (c10::metal::is_complex_v<T>) {
                row[r][c] -= c10::metal::mul(l, uc[c]);
              } else {
                row[r][c] = fma(-l, uc[c], row[r][c]);
              }
            }
          }
        }
      }
    }
  }

#pragma unroll
  for (short r = 0; r < R; r++) {
    const uint lr = tid + uint(r) * G;
    if (lr < H) {
      device T* dst = Ab + ulong(d0 + lr) * N + d0;
      if (vec4) {
#pragma unroll
        for (short c = 0; c < W; c += 4) {
          luStore4(dst + c, &row[r][c]);
        }
      } else {
#pragma unroll
        for (short c = 0; c < W; c++) {
          if (uint(c) < nb) {
            dst[c] = row[r][c];
          }
        }
      }
    }
  }
}

#define INSTANTIATE_FACTOR_PANEL_LU(T, R, W)                \
  template [[host_name("factorPanelLU_" #T "_" #R "_" #W)]] \
  kernel void factorPanelLU<T, R, W>(                       \
      device T * A [[buffer(0)]],                           \
      device int* pivots [[buffer(1)]],                     \
      device int* info [[buffer(2)]],                       \
      constant uint2& dims [[buffer(3)]],                   \
      constant uint4& params [[buffer(4)]],                 \
      uint3 tid3 [[thread_position_in_threadgroup]],        \
      uint3 bid [[threadgroup_position_in_grid]],           \
      uint3 tpg [[threads_per_threadgroup]],                \
      uint warp_id [[simdgroup_index_in_threadgroup]],      \
      uint lane [[thread_index_in_simdgroup]]);

INSTANTIATE_FACTOR_PANEL_LU(float, 1, 32)
INSTANTIATE_FACTOR_PANEL_LU(float, 2, 16)
INSTANTIATE_FACTOR_PANEL_LU(float, 4, 8)
INSTANTIATE_FACTOR_PANEL_LU(float2, 1, 32)
INSTANTIATE_FACTOR_PANEL_LU(float2, 2, 16)
INSTANTIATE_FACTOR_PANEL_LU(float2, 4, 8)

// Streaming panel factorization for tall panels (H > kStreamMinRows): factor
// one column at a time across many threadgroups when the register-resident
// factorPanelLU no longer fits. luStreamUpdate applies column j's rank-1 update
// over all rows and writes each threadgroup's local argmax partial to scratch;
// luStreamPivot then reduces those partials to the global pivot for column j.
template <typename T>
[[max_total_threads_per_threadgroup(kLUStreamNT)]]
kernel void luStreamUpdate(
    device T* A [[buffer(0)]],
    constant uint2& dims [[buffer(3)]],
    constant uint4& params [[buffer(4)]], // d0, j, RPT, searchOnly
    device LUStreamScratch<T>* scratch [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint warp_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint M = dims.x;
  const uint N = dims.y;
  const uint d0 = params.x;
  const uint j = params.y;
  const uint RPT = params.z;
  const bool searchOnly = params.w != 0;
  const uint minMN = min(M, N);
  const uint nb = min(32u, minMN - d0);
  const uint H = M - d0;
  device T* Ab = A + ulong(tgid.y) * M * N;
  device auto& scr = scratch[tgid.y];

  const uint rowStart = searchOnly ? j : j + 1;
  const uint sc = searchOnly ? j : j + 1; // column searched for next pivot
  const uint base = rowStart + (tgid.x * kLUStreamWarpsPerTG + warp_id) * RPT;

  T uc = T(0.0f);
  T rp = T(0.0f);
  bool doUpdate = false;
  if (!searchOnly) {
    const T upiv = scr.uRow[j];
    doUpdate = luPivotMag(upiv) != 0.0f;
    rp = doUpdate ? luRecip(upiv) : T(0.0f);
    uc = (lane < nb) ? scr.uRow[lane] : T(0.0f);
  }

  float bv = -1.0f;
  uint bi = 0xffffffffu;
  const bool active = lane >= j && lane < nb;
  for (uint r = 0; r < RPT; r++) {
    const uint lr = base + r;
    if (lr >= H) {
      break;
    }
    device T* rowp = Ab + ulong(d0 + lr) * N + d0;
    T v = active ? rowp[lane] : T(0.0f);
    if (doUpdate) {
      const T l = c10::metal::mul(simd_broadcast(v, ushort(j)), rp);
      if (lane == uint(j)) {
        v = l;
      } else if (lane > j && lane < nb) {
        if IF_CONSTEXPR (c10::metal::is_complex_v<T>) {
          v -= c10::metal::mul(l, uc);
        } else {
          v = fma(-l, uc, v);
        }
      }
      if (active) {
        rowp[lane] = v;
      }
    }
    if (lane == sc && sc < nb) {
      const float av = luPivotMag(v);
      if (av > bv) {
        bv = av;
        bi = lr;
      }
    }
  }
  threadgroup float wv[kLUStreamWarpsPerTG];
  threadgroup uint wi[kLUStreamWarpsPerTG];
  const float mv = simd_max(bv);
  const uint mi = simd_min((bv == mv) ? bi : 0xffffffffu);
  if (lane == 0) {
    wv[warp_id] = mv;
    wi[warp_id] = mi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (warp_id == 0) {
    const float v2 = (lane < kLUStreamWarpsPerTG) ? wv[lane] : -1.0f;
    const uint i2 = (lane < kLUStreamWarpsPerTG) ? wi[lane] : 0xffffffffu;
    const float m2 = simd_max(v2);
    const uint p2 = simd_min((v2 == m2) ? i2 : 0xffffffffu);
    if (lane == 0) {
      scr.vpart[tgid.x] = m2;
      scr.ipart[tgid.x] = p2;
    }
  }
}

#define INSTANTIATE_LU_STREAM_UPDATE(T)                  \
  template [[host_name("luStreamUpdate_" #T)]]           \
  kernel void luStreamUpdate<T>(                         \
      device T * A [[buffer(0)]],                        \
      constant uint2 & dims [[buffer(3)]],               \
      constant uint4 & params [[buffer(4)]],             \
      device LUStreamScratch<T> * scratch [[buffer(6)]], \
      uint3 tgid [[threadgroup_position_in_grid]],       \
      uint warp_id [[simdgroup_index_in_threadgroup]],   \
      uint lane [[thread_index_in_simdgroup]]);

INSTANTIATE_LU_STREAM_UPDATE(float)
INSTANTIATE_LU_STREAM_UPDATE(float2)

// Reduce luStreamUpdate's per-threadgroup argmax partials to the global pivot
// for column j, record it (1-based, like LAPACK), swap the pivot row, and
// broadcast the resulting U row back to scratch for the next update.
template <typename T>
[[max_total_threads_per_threadgroup(kLUStreamNT)]]
kernel void luStreamPivot(
    device T* A [[buffer(0)]],
    device int* pivots [[buffer(1)]],
    device int* info [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    constant uint4& params [[buffer(4)]], // d0, j, npart
    device LUStreamScratch<T>* scratch [[buffer(6)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint warp_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint M = dims.x;
  const uint N = dims.y;
  const uint d0 = params.x;
  const uint j = params.y;
  const uint npart = params.z;
  const uint minMN = min(M, N);
  const uint nb = min(32u, minMN - d0);
  const uint tid = tid3.x; // threadgroup of kLUStreamNT threads
  device T* Ab = A + ulong(tgid.x) * M * N;
  device int* pv = pivots + ulong(tgid.x) * minMN;
  device auto& scr = scratch[tgid.x];

  if (d0 == 0 && j == 0 && tid == 0) {
    info[tgid.x] = 0;
  }
  threadgroup float wv[kLUStreamWarpsPerTG];
  threadgroup uint wi[kLUStreamWarpsPerTG];
  threadgroup uint sPiv[1];

  // first-max semantics: equal partials resolve to the smaller global row
  float bv = -1.0f;
  uint bi = 0xffffffffu;
  for (uint i = tid; i < npart; i += kLUStreamNT) {
    const float v = scr.vpart[i];
    const uint ix = scr.ipart[i];
    if (v > bv || (v == bv && ix < bi)) {
      bv = v;
      bi = ix;
    }
  }
  const float mv = simd_max(bv);
  const uint mi = simd_min((bv == mv) ? bi : 0xffffffffu);
  if (lane == 0) {
    wv[warp_id] = mv;
    wi[warp_id] = mi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (warp_id == 0) {
    const float v2 = (lane < kLUStreamWarpsPerTG) ? wv[lane] : -1.0f;
    const uint i2 = (lane < kLUStreamWarpsPerTG) ? wi[lane] : 0xffffffffu;
    const float m2 = simd_max(v2);
    uint p2 = simd_min((v2 == m2) ? i2 : 0xffffffffu);
    if (lane == 0) {
      if (p2 == 0xffffffffu) { // all-NaN column: pivot on j, NaN spreads
        p2 = j;
      }
      sPiv[0] = p2;
      pv[d0 + j] = int(d0 + p2 + 1); // 1-based like LAPACK
      if (m2 == 0.0f && info[tgid.x] == 0) {
        info[tgid.x] = int(d0 + j + 1);
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    const uint p = sPiv[0];
    if (lane < nb) {
      device T* rj = Ab + ulong(d0 + j) * N + d0 + lane;
      T vj = *rj;
      if (p != j) {
        device T* rp2 = Ab + ulong(d0 + p) * N + d0 + lane;
        const T vp = *rp2;
        *rj = vp;
        *rp2 = vj;
        vj = vp;
      }
      scr.uRow[lane] = vj;
    }
  }
}

#define INSTANTIATE_LU_STREAM_PIVOT(T)                  \
  template [[host_name("luStreamPivot_" #T)]]           \
  kernel void luStreamPivot<T>(                         \
      device T * A [[buffer(0)]],                       \
      device int* pivots [[buffer(1)]],                 \
      device int* info [[buffer(2)]],                   \
      constant uint2& dims [[buffer(3)]],               \
      constant uint4& params [[buffer(4)]],             \
      device LUStreamScratch<T>* scratch [[buffer(6)]], \
      uint3 tid3 [[thread_position_in_threadgroup]],    \
      uint3 tgid [[threadgroup_position_in_grid]],      \
      uint warp_id [[simdgroup_index_in_threadgroup]],  \
      uint lane [[thread_index_in_simdgroup]]);

INSTANTIATE_LU_STREAM_PIVOT(float)
INSTANTIATE_LU_STREAM_PIVOT(float2)

// slaswp: apply a block's pivot interchanges as one staged gather/scatter
// through threadgroup memory, not nb sequential row swaps. CW is the staged
// column-chunk width: 64 floats, 32 float2 (the 64x64 float2 tile would be
// 32KB and exceed the threadgroup memory budget with the index arrays).
template <typename T, short CW>
kernel void laswpGatherLU(
    device T* A [[buffer(0)]],
    device const int* pivots [[buffer(1)]],
    constant uint2& dims [[buffer(3)]],
    constant uint4& params [[buffer(4)]],
    constant uint4& w [[buffer(5)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]],
    uint warp_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint M = dims.x;
  const uint N = dims.y;
  const uint d0 = params.x;
  const uint nb = params.y; // <= 32
  const uint W0 = w.y - w.x;
  const uint W = W0 + (w.w - w.z);
  const uint tid = tid3.x;
  const uint G = tpg.x;
  device T* Ab = A + ulong(tgid.y) * M * N;
  device const int* pvt = pivots + ulong(tgid.y) * min(M, N) + d0;

  threadgroup uint rowIds[64]; // global row of each slot
  threadgroup uint src[64]; // slot whose staged data this slot receives
  threadgroup uint counts[1];
  threadgroup T stage[64][CW];

  if (warp_id == 0) {
    // pivots are stored 1-based (LAPACK convention)
    const uint myp = (lane < nb) ? uint(pvt[lane]) - 1 : 0xffffffffu;
    const bool outb = (lane < nb) && (myp >= d0 + nb);
    // dedup out-of-band pivot rows; first occurrence keeps
    bool keep = outb;
    for (ushort t = 0; t < 32; t++) {
      const uint pt = simd_broadcast(myp, t);
      if (outb && uint(t) < lane && pt == myp) {
        keep = false;
      }
    }
    const uint pre = simd_prefix_exclusive_sum(keep ? 1u : 0u);
    if (lane < nb) {
      rowIds[lane] = d0 + lane;
    }
    if (keep) {
      rowIds[nb + pre] = myp;
    }
    const uint nextras = simd_sum(keep ? 1u : 0u);
    if (lane == 0) {
      counts[0] = nb + nextras;
    }
    src[lane] = lane;
    src[lane + 32] = lane + 32;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // simulate the swap sequence on slot indices; extras located by ballot
    const uint exRow =
        (nb + lane < counts[0]) ? rowIds[nb + lane] : 0xffffffffu;
    for (uint s = 0; s < nb; s++) {
      const uint p2 = simd_broadcast(myp, ushort(s));
      uint slotp;
      if (p2 < d0 + nb) {
        slotp = p2 - d0;
      } else {
        slotp = nb + simd_min((exRow == p2) ? lane : 0xffffffffu);
      }
      if (lane == 0 && slotp != s) {
        const uint t2 = src[s];
        src[s] = src[slotp];
        src[slotp] = t2;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint nlist = counts[0];
  const uint vbase = tgid.x * CW;
  const bool aligned = (N % 4u) == 0;

  for (uint i = tid; i < nlist * (CW / 4); i += G) {
    const uint r = i / (CW / 4);
    const uint q = (i % (CW / 4)) * 4;
    const uint v = vbase + q;
    if (v >= W) {
      continue;
    }
    const uint c = (v < W0) ? (w.x + v) : (w.z + (v - W0));
    const uint cnt = min(4u, W - v);
    device const T* sp = Ab + ulong(rowIds[r]) * N + c;
    if IF_CONSTEXPR (c10::metal::is_complex_v<T>) {
      for (uint e = 0; e < cnt; e++) {
        stage[r][q + e] = sp[e];
      }
    } else if (cnt == 4 && aligned) {
      const float4 t = *(device const float4*)sp;
      stage[r][q + 0] = t.x;
      stage[r][q + 1] = t.y;
      stage[r][q + 2] = t.z;
      stage[r][q + 3] = t.w;
    } else {
      for (uint e = 0; e < cnt; e++) {
        stage[r][q + e] = sp[e];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < nlist * (CW / 4); i += G) {
    const uint r = i / (CW / 4);
    const uint sr = src[r];
    if (sr == r) {
      continue;
    }
    const uint q = (i % (CW / 4)) * 4;
    const uint v = vbase + q;
    if (v >= W) {
      continue;
    }
    const uint c = (v < W0) ? (w.x + v) : (w.z + (v - W0));
    const uint cnt = min(4u, W - v);
    device T* dp = Ab + ulong(rowIds[r]) * N + c;
    if (cnt == 4 && aligned) {
      luStore4(dp, &stage[sr][q]);
    } else {
      for (uint e = 0; e < cnt; e++) {
        dp[e] = stage[sr][q + e];
      }
    }
  }
}

#define INSTANTIATE_LASWP_GATHER_LU(T, CW)             \
  template [[host_name("laswpGatherLU_" #T)]]          \
  kernel void laswpGatherLU<T, CW>(                    \
      device T * A [[buffer(0)]],                      \
      device const int* pivots [[buffer(1)]],          \
      constant uint2& dims [[buffer(3)]],              \
      constant uint4& params [[buffer(4)]],            \
      constant uint4& w [[buffer(5)]],                 \
      uint3 tid3 [[thread_position_in_threadgroup]],   \
      uint3 tgid [[threadgroup_position_in_grid]],     \
      uint3 tpg [[threads_per_threadgroup]],           \
      uint warp_id [[simdgroup_index_in_threadgroup]], \
      uint lane [[thread_index_in_simdgroup]]);

INSTANTIATE_LASWP_GATHER_LU(float, 64)
INSTANTIATE_LASWP_GATHER_LU(float2, 32)

// strsm: solve unit-lower L*X = B for the panel's off-diagonal block, with L
// staged in threadgroup memory and one thread per column of B.
template <typename T, short TS>
kernel void trsmPanelLU(
    device T* A [[buffer(0)]],
    constant uint2& dims [[buffer(3)]],
    constant uint4& params [[buffer(4)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]]) {
  const uint M = dims.x;
  const uint N = dims.y;
  const uint d0 = params.x;
  const uint cs = params.y;
  const uint ce = params.z;
  const uint nr = params.w;
  const uint tid = tid3.x;
  const uint G = tpg.x;
  device T* Ab = A + ulong(tgid.x) * M * N;

  threadgroup T L[TS][TS + 1];
  if IF_CONSTEXPR (c10::metal::is_complex_v<T>) {
    // zero-pad the ragged block so the unrolled solve below stays a no-op
    // past nr (scalar loads; no float4 path for complex)
    for (uint i = tid; i < TS * TS; i += G) {
      const uint r = i / TS;
      const uint c = i % TS;
      L[r][c] = (r < nr && c < nr) ? Ab[ulong(d0 + r) * N + d0 + c] : T(0.0f);
    }
  } else if ((N % 4u) == 0 && nr == TS) {
    for (uint i = tid; i < TS * TS / 4; i += G) {
      const uint r = i / (TS / 4);
      const uint c = (i % (TS / 4)) * 4;
      const float4 v = *(device const float4*)(Ab + ulong(d0 + r) * N + d0 + c);
      L[r][c + 0] = v.x;
      L[r][c + 1] = v.y;
      L[r][c + 2] = v.z;
      L[r][c + 3] = v.w;
    }
  } else {
    // zero-pad the ragged block so the unrolled solve below stays a no-op
    // past nr
    for (uint i = tid; i < TS * TS; i += G) {
      const uint r = i / TS;
      const uint c = i % TS;
      L[r][c] = (r < nr && c < nr) ? Ab[ulong(d0 + r) * N + d0 + c] : 0.0f;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint col = cs + tgid.y * G + tid;
  if (col >= ce) {
    return;
  }
  T x[TS];
#pragma unroll
  for (short r = 0; r < TS; r++) {
    x[r] = (uint(r) < nr) ? Ab[ulong(d0 + r) * N + col] : T(0.0f);
  }
#pragma unroll
  for (short c = 0; c < TS; c++) {
    const T xc = x[c];
    if IF_CONSTEXPR (c10::metal::is_complex_v<T>) {
      // no dcol register staging for complex: it would double the per-thread
      // register bytes; read L directly instead
#pragma unroll
      for (short i = 0; i < TS; i++) {
        if (i > c) {
          x[i] -= c10::metal::mul(xc, L[i][c]);
        }
      }
    } else {
      // batch the column loads ahead of the fma burst (in-order pipe)
      T dcol[TS];
#pragma unroll
      for (short i = 0; i < TS; i++) {
        dcol[i] = L[i][c];
      }
#pragma unroll
      for (short i = 0; i < TS; i++) {
        if (i > c) {
          x[i] = fma(-xc, dcol[i], x[i]);
        }
      }
    }
  }
#pragma unroll
  for (short r = 0; r < TS; r++) {
    if (uint(r) < nr) {
      Ab[ulong(d0 + r) * N + col] = x[r];
    }
  }
}

#define INSTANTIATE_TRSM_PANEL_LU(T, TS)             \
  template [[host_name("trsmPanelLU_" #T "_" #TS)]]  \
  kernel void trsmPanelLU<T, TS>(                    \
      device T * A [[buffer(0)]],                    \
      constant uint2 & dims [[buffer(3)]],           \
      constant uint4 & params [[buffer(4)]],         \
      uint3 tid3 [[thread_position_in_threadgroup]], \
      uint3 tgid [[threadgroup_position_in_grid]],   \
      uint3 tpg [[threads_per_threadgroup]]);

INSTANTIATE_TRSM_PANEL_LU(float, 8)
INSTANTIATE_TRSM_PANEL_LU(float, 16)
INSTANTIATE_TRSM_PANEL_LU(float, 32)
INSTANTIATE_TRSM_PANEL_LU(float2, 8)
INSTANTIATE_TRSM_PANEL_LU(float2, 16)
INSTANTIATE_TRSM_PANEL_LU(float2, 32)

// In-place square transpose so the row-major factor matches the column-major LU
// view; tiles with tj < ti are produced by their mirror.
template <typename T>
kernel void transposeInPlaceLU(
    device T* A [[buffer(0)]],
    constant uint2& dims [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  const uint N = dims.y;
  const uint ti = tgid.y * 32;
  const uint tj = tgid.x * 32;
  if (tj < ti) {
    return;
  }
  device T* Ab = A + ulong(tgid.z) * N * N;
  threadgroup T ta[32][33];
  threadgroup T tb[32][33];
  const uint lx = tid3.x; // 0..31
  const uint ly = tid3.y; // 0..7

  for (uint r = ly; r < 32; r += 8) {
    if (ti + r < N && tj + lx < N) {
      ta[r][lx] = Ab[ulong(ti + r) * N + tj + lx];
    }
    if (tj + r < N && ti + lx < N) {
      tb[r][lx] = Ab[ulong(tj + r) * N + ti + lx];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = ly; r < 32; r += 8) {
    if (ti + r < N && tj + lx < N) {
      Ab[ulong(ti + r) * N + tj + lx] = tb[lx][r];
    }
    if (tj + r < N && ti + lx < N) {
      Ab[ulong(tj + r) * N + ti + lx] = ta[lx][r];
    }
  }
}

#define INSTANTIATE_TRANSPOSE_IN_PLACE_LU(T)         \
  template [[host_name("transposeInPlaceLU_" #T)]]   \
  kernel void transposeInPlaceLU<T>(                 \
      device T * A [[buffer(0)]],                    \
      constant uint2 & dims [[buffer(3)]],           \
      uint3 tid3 [[thread_position_in_threadgroup]], \
      uint3 tgid [[threadgroup_position_in_grid]]);

INSTANTIATE_TRANSPOSE_IN_PLACE_LU(float)
INSTANTIATE_TRANSPOSE_IN_PLACE_LU(float2)

// Schur update A -= A@A for complex64: simdgroup_matrix and MPP matmul2d are
// float-only hardware paths, so complex uses this threadgroup-tiled scalar
// kernel (c10::metal::mul complex multiply, zero-padded ragged edges).
// addmm_ cannot be used instead: it does not produce the correct result when
// all of its inputs alias the same tensor, which is exactly this in-place
// update where L21, U12 and the destination are windows of one buffer.
// Window convention matches gemmSimdLU/gemmLU: params = {rs, re, cs, ce},
// w = {kc, kw, 0, 0}; C = A[rs:re, cs:ce] -= A[rs:re, kc:kc+kw] @
// A[kc:kc+kw, cs:ce]. In-place safe: C rows/cols are disjoint from the
// L21 columns and U12 rows, and each thread only RMWs its own element.
template <typename T>
kernel void gemmTiledLU(
    device T* A [[buffer(0)]],
    constant uint2& dims [[buffer(3)]],
    constant uint4& params [[buffer(4)]],
    constant uint4& w [[buffer(5)]],
    uint3 tid3 [[thread_position_in_threadgroup]], // 16 x 16 x 1
    uint3 tgid [[threadgroup_position_in_grid]]) { // (col tile, row tile, B)
  const uint M = dims.x;
  const uint N = dims.y;
  const uint rs = params.x;
  const uint re = params.y;
  const uint cs = params.z;
  const uint ce = params.w;
  const uint kc = w.x;
  const uint kw = w.y;
  (void)M;

  device T* Ab = A + ulong(tgid.z) * M * N;

  const uint i = rs + tgid.y * 16 + tid3.y; // C row
  const uint jc = cs + tgid.x * 16 + tid3.x; // C col

  threadgroup T Atile[16][16];
  threadgroup T Btile[16][16];

  T sum = T(0.0f);
  const uint numTiles = (kw + 15) / 16;
  for (uint t = 0; t < numTiles; ++t) {
    const uint kA = kc + t * 16 + tid3.x; // column into L21
    const uint kB = kc + t * 16 + tid3.y; // row into U12
    Atile[tid3.y][tid3.x] =
        (i < re && kA < kc + kw) ? Ab[ulong(i) * N + kA] : T(0.0f);
    Btile[tid3.y][tid3.x] =
        (kB < kc + kw && jc < ce) ? Ab[ulong(kB) * N + jc] : T(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 16; ++kk) {
      sum += c10::metal::mul(Atile[tid3.y][kk], Btile[kk][tid3.x]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (i < re && jc < ce) {
    Ab[ulong(i) * N + jc] -= sum;
  }
}

#define INSTANTIATE_GEMM_TILED_LU(T)                 \
  template [[host_name("gemmTiledLU_" #T)]]          \
  kernel void gemmTiledLU<T>(                        \
      device T * A [[buffer(0)]],                    \
      constant uint2 & dims [[buffer(3)]],           \
      constant uint4 & params [[buffer(4)]],         \
      constant uint4 & w [[buffer(5)]],              \
      uint3 tid3 [[thread_position_in_threadgroup]], \
      uint3 tgid [[threadgroup_position_in_grid]]);

INSTANTIATE_GEMM_TILED_LU(float2)

// Schur-complement trailing update C -= A*B (sgemm) via simdgroup matmul;
// fallback used when matmul2d is unavailable (cf. gemmLU).
kernel void gemmSimdLU(
    device float* A [[buffer(0)]],
    constant uint2& dims [[buffer(3)]],
    constant uint4& win [[buffer(4)]],
    constant uint4& kwin [[buffer(5)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint warp_id [[simdgroup_index_in_threadgroup]]) {
  const uint M = dims.x;
  const uint N = dims.y;
  const uint rs = win.x;
  const uint Tm = win.y - rs;
  const uint cs = win.z;
  const uint Tn = win.w - cs;
  const uint kc = kwin.x;
  const uint kw = kwin.y;
  const uint ro = tgid.y * 32;
  const uint co = tgid.x * 64;
  const uint tid = tid3.x;
  device float* Ab = A + ulong(tgid.z) * M * N;

  threadgroup float As[32][17];
  threadgroup float Bs[16][65];
  threadgroup float Cs[32][65];

  simdgroup_float8x8 acc[8];
#pragma unroll
  for (short f = 0; f < 8; f++) {
    acc[f] = simdgroup_float8x8(0.0f);
  }

  for (uint k0 = 0; k0 < kw; k0 += 16) {
    for (uint i = tid; i < 32 * 16; i += 128) {
      const uint r = i / 16;
      const uint c = i % 16;
      const bool ok = (ro + r < Tm) && (k0 + c < kw);
      As[r][c] = ok ? Ab[ulong(rs + ro + r) * N + kc + k0 + c] : 0.0f;
    }
    for (uint i = tid; i < 16 * 64; i += 128) {
      const uint r = i / 64;
      const uint c = i % 64;
      const bool ok = (k0 + r < kw) && (co + c < Tn);
      Bs[r][c] = ok ? Ab[ulong(kc + k0 + r) * N + cs + co + c] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < 16; kk += 8) {
      simdgroup_float8x8 a;
      simdgroup_load(a, &As[8 * warp_id][kk], 17);
#pragma unroll
      for (short f = 0; f < 8; f++) {
        simdgroup_float8x8 b;
        simdgroup_load(b, &Bs[kk][8 * f], 65);
        simdgroup_multiply_accumulate(acc[f], a, b, acc[f]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

#pragma unroll
  for (short f = 0; f < 8; f++) {
    simdgroup_store(acc[f], &Cs[8 * warp_id][8 * f], 65);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 32 * 64; i += 128) {
    const uint r = i / 64;
    const uint c = i % 64;
    if (ro + r < Tm && co + c < Tn) {
      device float* p = Ab + ulong(rs + ro + r) * N + cs + co + c;
      *p = *p - Cs[r][c];
    }
  }
}

#if C10_METAL_HAS_MPP

// Same Schur update C -= A*B (sgemm) as gemmSimdLU, but via MetalPerformance-
// Primitives matmul2d (macOS 26.2+, gated by lu_has_matmul2d()).
template <int BM, int BN, int NSG>
kernel void gemmLU(
    device float* A [[buffer(0)]],
    constant uint2& dims [[buffer(3)]],
    constant uint4& win [[buffer(4)]],
    constant uint4& kwin [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  const int gN = int(dims.y);
  const int rs = int(win.x);
  const int Tm = int(win.y) - rs;
  const int cs = int(win.z);
  const int Tn = int(win.w) - cs;
  const int kc = int(kwin.x);
  const int K = int(kwin.y);
  const int ro = int(tgid.y) * BM;
  const int co = int(tgid.x) * BN;
  device float* Ab = A + ulong(tgid.z) * ulong(dims.x) * ulong(dims.y);

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      BM,
      BN,
      static_cast<int>(dynamic_extent),
      false,
      false,
      false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroups<NSG>> op;

  device float* aP = Ab + rs * gN + kc;
  device float* bP = Ab + kc * gN + cs;
  device float* cP = Ab + rs * gN + cs;
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tA(
      aP, dextents<int32_t, 2>(K, Tm), array<int32_t, 2>{1, gN});
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tB(
      bP, dextents<int32_t, 2>(Tn, K), array<int32_t, 2>{1, gN});
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(
      cP, dextents<int32_t, 2>(Tn, Tm), array<int32_t, 2>{1, gN});

  auto schur = [&](auto mA, auto mB, auto mC, bool checkValid) {
    auto cT = op.template get_destination_cooperative_tensor<
        decltype(mA),
        decltype(mB),
        float>();
    op.run(mA, mB, cT);
    uint16_t e = 0;
    for (auto it = cT.begin(); it != cT.end(); ++it, ++e) {
      if (checkValid && !cT.is_valid_element(e)) {
        continue;
      }
      auto idx = it.get_multidimensional_index();
      const int r = ro + int(idx[1]);
      const int c = co + int(idx[0]);
      cT[e] = cP[r * gN + c] - cT[e];
    }
    cT.store(mC);
  };

  const bool inside = (ro + BM <= Tm) && (co + BN <= Tn);
  if (inside) {
    schur(
        tA.template slice<dynamic_extent, BM>(0, ro),
        tB.template slice<BN, dynamic_extent>(co, 0),
        tC.template slice<BN, BM>(co, ro),
        false);
  } else {
    schur(tA.slice(0, ro), tB.slice(co, 0), tC.slice(co, ro), true);
  }
}

#define INSTANTIATE_GEMM_LU(BM, BN, NSG)                 \
  template [[host_name("gemmLU_" #BM "_" #BN "_" #NSG)]] \
  kernel void gemmLU<BM, BN, NSG>(                       \
      device float* A [[buffer(0)]],                     \
      constant uint2& dims [[buffer(3)]],                \
      constant uint4& win [[buffer(4)]],                 \
      constant uint4& kwin [[buffer(5)]],                \
      uint3 tgid [[threadgroup_position_in_grid]]);

INSTANTIATE_GEMM_LU(64, 64, 4)
INSTANTIATE_GEMM_LU(32, 64, 2)

#endif // C10_METAL_HAS_MPP

template <typename T, bool upper, bool unit, short TS>
kernel void trsmDiagSolveLU(
    device T* A [[buffer(0)]],
    constant uint2& dims [[buffer(3)]],
    constant uint4& params [[buffer(4)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]]) {
  const uint M = dims.x;
  const uint N = dims.y;
  const uint d0 = params.x;
  const uint cs = params.y;
  const uint ce = params.z;
  const uint nr = params.w;
  const uint tid = tid3.x;
  const uint G = tpg.x;
  device T* Ab = A + ulong(tgid.x) * M * N;

  threadgroup T Td[TS][TS + 1];
  for (uint i = tid; i < TS * TS; i += G) {
    const uint r = i / TS;
    const uint c = i % TS;
    if IF_CONSTEXPR (c10::metal::is_complex_v<T>) {
      // pad with zeros; the divide below is guarded on c < nr so the zero
      // diagonal in the padding never produces a 0/0 NaN (no complex one)
      Td[r][c] = (r < nr && c < nr) ? Ab[ulong(d0 + r) * N + d0 + c] : T(0.0f);
    } else {
      Td[r][c] = (r < nr && c < nr) ? Ab[ulong(d0 + r) * N + d0 + c]
                                    : (r == c ? T(1.0f) : T(0.0f));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint col = cs + tgid.y * G + tid;
  if (col >= ce) {
    return;
  }
  T x[TS];
#pragma unroll
  for (short r = 0; r < TS; r++) {
    x[r] = (uint(r) < nr) ? Ab[ulong(d0 + r) * N + col] : T(0.0f);
  }
  if IF_CONSTEXPR (c10::metal::is_complex_v<T>) {
    // no dcol register staging for complex: it would double the per-thread
    // register bytes; read Td directly instead (cf. trsmPanelLU)
    if (!upper) {
#pragma unroll
      for (short c = 0; c < TS; c++) {
        const T xc =
            (unit || uint(c) >= nr) ? x[c] : c10::metal::div(x[c], Td[c][c]);
        x[c] = xc;
#pragma unroll
        for (short i = 0; i < TS; i++) {
          if (i > c) {
            x[i] -= c10::metal::mul(xc, Td[i][c]);
          }
        }
      }
    } else {
#pragma unroll
      for (short c = TS - 1; c >= 0; c--) {
        const T xc =
            (unit || uint(c) >= nr) ? x[c] : c10::metal::div(x[c], Td[c][c]);
        x[c] = xc;
#pragma unroll
        for (short i = 0; i < TS; i++) {
          if (i < c) {
            x[i] -= c10::metal::mul(xc, Td[i][c]);
          }
        }
      }
    }
  } else if (!upper) {
#pragma unroll
    for (short c = 0; c < TS; c++) {
      T dcol[TS];
#pragma unroll
      for (short i = 0; i < TS; i++) {
        dcol[i] = Td[i][c];
      }
      const T xc = unit ? x[c] : x[c] / Td[c][c];
      x[c] = xc;
#pragma unroll
      for (short i = 0; i < TS; i++) {
        if (i > c) {
          x[i] = fma(-xc, dcol[i], x[i]);
        }
      }
    }
  } else {
#pragma unroll
    for (short c = TS - 1; c >= 0; c--) {
      T dcol[TS];
#pragma unroll
      for (short i = 0; i < TS; i++) {
        dcol[i] = Td[i][c];
      }
      const T xc = unit ? x[c] : x[c] / Td[c][c];
      x[c] = xc;
#pragma unroll
      for (short i = 0; i < TS; i++) {
        if (i < c) {
          x[i] = fma(-xc, dcol[i], x[i]);
        }
      }
    }
  }
#pragma unroll
  for (short r = 0; r < TS; r++) {
    if (uint(r) < nr) {
      Ab[ulong(d0 + r) * N + col] = x[r];
    }
  }
}

#define INSTANTIATE_TRSM_DIAG_SOLVE(T, UP, UN, SUFF)      \
  template [[host_name("trsmDiagSolveLU_" #T "_" #SUFF)]] \
  kernel void trsmDiagSolveLU<T, UP, UN, 32>(             \
      device T * A [[buffer(0)]],                         \
      constant uint2 & dims [[buffer(3)]],                \
      constant uint4 & params [[buffer(4)]],              \
      uint3 tid3 [[thread_position_in_threadgroup]],      \
      uint3 tgid [[threadgroup_position_in_grid]],        \
      uint3 tpg [[threads_per_threadgroup]]);

INSTANTIATE_TRSM_DIAG_SOLVE(float, false, true, lower_unit)
INSTANTIATE_TRSM_DIAG_SOLVE(float, true, false, upper_nonunit)
INSTANTIATE_TRSM_DIAG_SOLVE(float, false, false, lower_nonunit)
INSTANTIATE_TRSM_DIAG_SOLVE(float, true, true, upper_unit)
INSTANTIATE_TRSM_DIAG_SOLVE(float2, false, true, lower_unit)
INSTANTIATE_TRSM_DIAG_SOLVE(float2, true, false, upper_nonunit)
INSTANTIATE_TRSM_DIAG_SOLVE(float2, false, false, lower_nonunit)
INSTANTIATE_TRSM_DIAG_SOLVE(float2, true, true, upper_unit)

template <typename T>
kernel void luApplyPivotsRHS(
    device T* A [[buffer(0)]],
    device const int* pivots [[buffer(1)]],
    constant uint2& dims [[buffer(3)]],
    constant uint4& params [[buffer(4)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]]) {
  const uint M = dims.x;
  const uint N = dims.y;
  const uint coff = params.x;
  const uint k = params.y;
  const uint npiv = params.z;
  const uint inverse = params.w;
  const uint tid = tid3.x;
  const uint G = tpg.x;
  device T* Ab = A + ulong(tgid.x) * M * N;
  device const int* pv = pivots + ulong(tgid.x) * npiv;

  for (uint s = 0; s < npiv; s++) {
    const uint i = inverse ? (npiv - 1 - s) : s;
    const uint p = uint(pv[i] - 1);
    if (p != i) {
      for (uint col = tid; col < k; col += G) {
        const uint cc = coff + col;
        const T t = Ab[ulong(i) * N + cc];
        Ab[ulong(i) * N + cc] = Ab[ulong(p) * N + cc];
        Ab[ulong(p) * N + cc] = t;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

#define INSTANTIATE_LU_APPLY_PIVOTS_RHS(T)           \
  template [[host_name("luApplyPivotsRHS_" #T)]]     \
  kernel void luApplyPivotsRHS<T>(                   \
      device T * A [[buffer(0)]],                    \
      device const int* pivots [[buffer(1)]],        \
      constant uint2& dims [[buffer(3)]],            \
      constant uint4& params [[buffer(4)]],          \
      uint3 tid3 [[thread_position_in_threadgroup]], \
      uint3 tgid [[threadgroup_position_in_grid]],   \
      uint3 tpg [[threads_per_threadgroup]]);

INSTANTIATE_LU_APPLY_PIVOTS_RHS(float)
INSTANTIATE_LU_APPLY_PIVOTS_RHS(float2)

// Small-matrix inverse (luInvSmall): one thread per matrix, the matrix and
// its inverse live in registers. Every array index must stay a compile-time
// constant after unrolling, so row swaps scan for r == p instead of indexing
// a[p] directly.
template <short N>
kernel void luInvSmall(
    device const float* A [[buffer(0)]],
    device float* X [[buffer(1)]],
    device int* info [[buffer(2)]],
    constant LUSmallInvParams<>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  device const float* Ab = A + long(tid) * params.A_bstride;
  device float* Xb = X + long(tid) * params.X_bstride;

  float a[N][N];
  float x[N][N];
#pragma unroll
  for (short r = 0; r < N; r++) {
#pragma unroll
    for (short c = 0; c < N; c++) {
      a[r][c] = Ab[long(r) * params.A_rstride + long(c) * params.A_cstride];
      x[r][c] = (r == c) ? 1.0f : 0.0f;
    }
  }

  int inf = 0;
#pragma unroll
  for (short j = 0; j < N; j++) {
    float bv = -1.0f;
    short p = j;
#pragma unroll
    for (short r = 0; r < N; r++) {
      if (r >= j) {
        const float v = luPivotMag(a[r][j]);
        if (v > bv) {
          bv = v;
          p = r;
        }
      }
    }
    if (bv == 0.0f && inf == 0) {
      inf = j + 1;
    }
#pragma unroll
    for (short r = 0; r < N; r++) {
      if (r > j && r == p) {
#pragma unroll
        for (short c = 0; c < N; c++) {
          const float ta = a[j][c];
          a[j][c] = a[r][c];
          a[r][c] = ta;
          const float tx = x[j][c];
          x[j][c] = x[r][c];
          x[r][c] = tx;
        }
      }
    }
    if (bv != 0.0f) {
      const float rp = luRecip(a[j][j]);
#pragma unroll
      for (short r = 0; r < N; r++) {
        if (r > j) {
          const float l = a[r][j] * rp;
#pragma unroll
          for (short c = 0; c < N; c++) {
            if (c > j) {
              a[r][c] = fma(-l, a[j][c], a[r][c]);
            }
            x[r][c] = fma(-l, x[j][c], x[r][c]);
          }
        }
      }
    }
  }
#pragma unroll
  for (short c = N - 1; c >= 0; c--) {
#pragma unroll
    for (short k = 0; k < N; k++) {
      x[c][k] = x[c][k] / a[c][c];
    }
#pragma unroll
    for (short i = 0; i < N; i++) {
      if (i < c) {
#pragma unroll
        for (short k = 0; k < N; k++) {
          x[i][k] = fma(-a[i][c], x[c][k], x[i][k]);
        }
      }
    }
  }
  info[tid] = inf;

#pragma unroll
  for (short r = 0; r < N; r++) {
#pragma unroll
    for (short c = 0; c < N; c++) {
      Xb[long(r) * params.X_rstride + long(c) * params.X_cstride] = x[r][c];
    }
  }
}

#define INSTANTIATE_LU_INV_SMALL(N)                      \
  template [[host_name("luInvSmall_" #N)]]               \
  kernel void luInvSmall<N>(                             \
      device const float* A [[buffer(0)]],               \
      device float* X [[buffer(1)]],                     \
      device int* info [[buffer(2)]],                    \
      constant LUSmallInvParams<>& params [[buffer(3)]], \
      uint tid [[thread_position_in_grid]]);

// exact sizes 1..kLUSmallInvMax picked by lu_inv_small_encode
INSTANTIATE_LU_INV_SMALL(1)
INSTANTIATE_LU_INV_SMALL(2)
INSTANTIATE_LU_INV_SMALL(3)
INSTANTIATE_LU_INV_SMALL(4)
static_assert(kLUSmallInvMax == 4, "update luInvSmall instantiations");

kernel void applyPivots(
    device float* P [[buffer(0)]],
    device const int* pivots [[buffer(1)]],
    constant uint& R [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]]) {
  uint tx = tid.x;
  uint group_size = tpg.x * tpg.y;
  uint batch_idx = bid.x;

  for (int i = static_cast<int>(K) - 1; i >= 0; i--) {
    int pivot = pivots[batch_idx * K + i];
    if (pivot == i) {
      // no swap needed
      continue;
    }

    for (uint j = tx * 4; j < R; j += group_size * 4) {
      uint elementsRemaining = R - j;

      // if we can use float4 or not
      if (elementsRemaining < 4) {
        for (uint e = 0; e < elementsRemaining; e++) {
          float row_i_value = P[batch_idx * R * R + i * R + (j + e)];
          float pivot_row_value = P[batch_idx * R * R + pivot * R + (j + e)];

          P[batch_idx * R * R + i * R + (j + e)] = pivot_row_value;
          P[batch_idx * R * R + pivot * R + (j + e)] = row_i_value;
        }
      } else {
        // vectorized load/stores
        device float4* rowIPtr =
            reinterpret_cast<device float4*>(&P[batch_idx * R * R + i * R + j]);
        device float4* pivotPtr = reinterpret_cast<device float4*>(
            &P[batch_idx * R * R + pivot * R + j]);

        float4 row_i_val = *rowIPtr;
        float4 pivot_val = *pivotPtr;

        *rowIPtr = pivot_val;
        *pivotPtr = row_i_val;
      }
    }
    // barrier here so different threads do not rush after each other
    // swapping rows for the next iteration while
    // some threads are swapping the current one
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <typename TO, typename TI>
kernel void unpack_pivots(
    device TO* perm [[buffer(0)]],
    device const TI* pivots [[buffer(1)]],
    constant UnpackPivotsParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
  auto perm_batch_stride = params.perm_batch_stride;
  auto pivots_batch_stride = params.pivots_batch_stride;
  auto dim_size = params.dim_size;

  perm += perm_batch_stride * tid;
  pivots += pivots_batch_stride * tid;

  for (uint32_t i = 0; i < dim_size; i++) {
    auto j = pivots[i] - 1;
    auto perm_j = perm[j];
    perm[j] = perm[i];
    perm[i] = perm_j;
  }
}

#define REGISTER_UNPACK_PIVOTS(TO, TI)                   \
  template [[host_name("unpack_pivots_" #TO "_" #TI)]]   \
  kernel void unpack_pivots<TO, TI>(                     \
      device TO * perm [[buffer(0)]],                    \
      device const TI* pivots [[buffer(1)]],             \
      constant UnpackPivotsParams& params [[buffer(2)]], \
      uint tid [[thread_position_in_grid]]);

REGISTER_UNPACK_PIVOTS(int, int);
REGISTER_UNPACK_PIVOTS(int, long);
REGISTER_UNPACK_PIVOTS(long, int);
REGISTER_UNPACK_PIVOTS(long, long);
