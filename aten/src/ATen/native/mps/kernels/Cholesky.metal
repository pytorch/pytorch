#include <ATen/native/mps/kernels/LinearAlgebra.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

template <bool col_major, typename T>
inline device T& get_ref(device T* A, uint row, uint col, uint N) {
  if (col_major) {
    return A[row * N + col];
  }
  return A[row + col * N];
}

template <typename T>
inline int factor_tile32_warp(
    threadgroup T (&tile)[32][33],
    threadgroup T (&col)[32],
    uint n,
    uint lane) {
  T row[32];
  for (uint c = 0; c < 32; c++) {
    row[c] = tile[lane][c];
  }
  int ret = 0;
  for (uint kk = 0; kk < n; kk++) {
    float dsq = simd_broadcast(c10::metal::cast_to<float>(row[kk]), ushort(kk));
    if (!(dsq > 0.0f)) {
      ret = int(kk) + 1;
      break;
    }
    float rs = rsqrt(dsq);
    T l = (lane == kk) ? c10::metal::cast_to<T>(dsq * rs) : row[kk] * rs;
    col[lane] = l;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    T ccol[32];
#pragma unroll
    for (uint i = 0; i < 32; i++) {
      ccol[i] = col[i];
    }
    if (lane >= kk) {
      row[kk] = l;
      if (lane > kk) {
#pragma unroll
        for (uint i = 0; i < 32; i++) {
          if (i > kk) {
            row[i] = c10::metal::fma(-l, c10::metal::conj(ccol[i]), row[i]);
          }
        }
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lane < n) {
    for (uint c = 0; c < 32; c++) {
      tile[lane][c] = row[c];
    }
    col[lane] =
        c10::metal::cast_to<T>(1.0f / c10::metal::cast_to<float>(row[lane]));
  }
  return ret;
}

// Solves row * D^-T in registers against the factored 32x32 diagonal block
// `diagT` (rdiag[c] = 1 / diagT[c][c]) by right-looking column elimination:
// the critical path per column is one multiply plus the fma into the next
// column; the remaining unrolled updates pipeline behind it.
inline void trsm_row32(
    thread float (&row)[32],
    threadgroup float (&diagT)[32][33],
    threadgroup float (&rdiag)[32]) {
  for (uint c = 0; c < 32; c++) {
    // batch the column loads ahead of the fma burst; an interleaved
    // load/fma sequence stalls the in-order pipe on every smem access
    float dcol[32];
#pragma unroll
    for (uint i = 0; i < 32; i++) {
      dcol[i] = diagT[i][c];
    }
    float xc = row[c] * rdiag[c];
    row[c] = xc;
#pragma unroll
    for (uint i = 0; i < 32; i++) {
      if (i > c) {
        row[i] = fma(-xc, dcol[i], row[i]);
      }
    }
  }
}

template <bool upper, typename T>
kernel void factorDiagonalBlock(
    device T* A [[buffer(0)]],
    device int* info [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& NB [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]]) {
  uint tx = tid.x;
  uint ty = tid.y;
  uint linear_tid = ty * tpg.x + tx;
  uint group_size = tpg.x * tpg.y;

  const uint actSize = min(N - k * NB, NB);
  const uint batch_offset = bid.x * N * N;
  const uint row0 = k * NB;
  const uint col0 = k * NB;

  threadgroup T tile[32][33];
  threadgroup T col[32];
  threadgroup float scratch[1];
  const uint tileSize = actSize * actSize;

  for (uint i = linear_tid; i < tileSize; i += group_size) {
    uint r = i / actSize;
    uint c = i % actSize;
    tile[r][c] = get_ref<upper>(A + batch_offset, row0 + r, col0 + c, N);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // simdgroups are linear chunks of the linearized threadgroup, so threads
  // 0..31 form simdgroup 0
  if (linear_tid < 32) {
    int f = factor_tile32_warp(tile, col, actSize, linear_tid);
    if (linear_tid == 0) {
      scratch[0] = float(f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int fail = int(scratch[0]);
  if (fail != 0) {
    // first failure wins; report the global leading-minor index like LAPACK
    if (linear_tid == 0 && info[bid.x] == 0) {
      info[bid.x] = int(row0) + fail;
    }
    return;
  }

  for (uint i = linear_tid; i < tileSize; i += group_size) {
    uint r = i / actSize;
    uint c = i % actSize;
    get_ref<upper>(A + batch_offset, row0 + r, col0 + c, N) = tile[r][c];
  }
}

#define INSTANTIATE_FACTOR_DIAGONAL_BLOCK(SUFF, UPPER, DTYPE)    \
  template [[host_name("factorDiagonalBlock" #SUFF "_" #DTYPE)]] \
  kernel void factorDiagonalBlock<UPPER, DTYPE>(                 \
      device DTYPE * A [[buffer(0)]],                            \
      device int* info [[buffer(1)]],                            \
      constant uint& N [[buffer(2)]],                            \
      constant uint& NB [[buffer(3)]],                           \
      constant uint& k [[buffer(4)]],                            \
      uint3 tid [[thread_position_in_threadgroup]],              \
      uint3 bid [[threadgroup_position_in_grid]],                \
      uint3 tpg [[threads_per_threadgroup]]);

INSTANTIATE_FACTOR_DIAGONAL_BLOCK(U, true, float)
INSTANTIATE_FACTOR_DIAGONAL_BLOCK(L, false, float)
INSTANTIATE_FACTOR_DIAGONAL_BLOCK(U, true, float2)
INSTANTIATE_FACTOR_DIAGONAL_BLOCK(L, false, float2)

inline float trsm_dot(
    threadgroup const float* x,
    threadgroup const float* y,
    uint n) {
  float4 sum4 = float4(0.0);
  uint p = 0;
  for (; p + 4 <= n; p += 4) {
    float4 x4 = float4(x[p], x[p + 1], x[p + 2], x[p + 3]);
    float4 y4 = float4(y[p], y[p + 1], y[p + 2], y[p + 3]);
    sum4 = fma(x4, y4, sum4);
  }
  float sum = sum4.x + sum4.y + sum4.z + sum4.w;
  for (; p < n; p++) {
    sum = fma(x[p], y[p], sum);
  }
  return sum;
}

inline float2 trsm_dot(
    threadgroup const float2* x,
    threadgroup const float2* y,
    uint n) {
  float2 sum = float2(0.0);
  for (uint p = 0; p < n; p++) {
    sum = c10::metal::fma(x[p], c10::metal::conj(y[p]), sum);
  }
  return sum;
}

template <bool upper, typename T>
kernel void applyTRSM(
    device T* A [[buffer(0)]],
    constant uint& N [[buffer(2)]],
    constant uint& NB [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]]) {
  // Thread indexing
  const uint tx = tid.x;
  const uint ty = tid.y;
  const uint linear_tid = ty * tpg.x + tx;
  const uint group_size = tpg.x * tpg.y;
  const uint b = tgid.x;
  const uint idxJ = tgid.y;

  // Size calculations
  const uint actSize_k = min(int32_t(N - k * NB), int32_t(NB));
  const uint j = (k + 1) + idxJ;
  const uint row0 = j * NB;
  const uint col0 = k * NB;
  const uint actSize_j = min((int)(N - row0), (int)NB);
  const uint batch_offset = b * N * N;

  // Early exit conditions
  if (actSize_k == 0 || j >= (N + NB - 1) / NB || j == k || actSize_j == 0) {
    return;
  }

  threadgroup T diag[32 * 32];
  threadgroup T target[32 * 32];

  for (uint i = linear_tid; i < actSize_k * actSize_k; i += group_size) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    diag[i] = get_ref<upper>(A + batch_offset, k * NB + r, k * NB + c, N);
  }
  for (uint i = linear_tid; i < actSize_j * actSize_k; i += group_size) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    target[i] = get_ref<upper>(A + batch_offset, row0 + r, col0 + c, N);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

// forward substitution with loop unrolling and vectorization
#pragma unroll 4
  for (uint col = 0; col < actSize_k; col++) {
    float diag_val = c10::metal::cast_to<float>(diag[col * actSize_k + col]);
    diag_val = (fabs(diag_val) < 1e-6f) ? copysign(1e-6f, diag_val) : diag_val;

    // multiple rows per thread
    for (uint row = linear_tid; row < actSize_j; row += group_size) {
      T sum = target[row * actSize_k + col] -
          trsm_dot(target + row * actSize_k, diag + col * actSize_k, col);
      target[row * actSize_k + col] = sum / diag_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // results back to global memory
  for (uint i = linear_tid; i < actSize_j * actSize_k; i += group_size) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    get_ref<upper>(A + batch_offset, row0 + r, col0 + c, N) = target[i];
  }
}

#define INSTANTIATE_APPLY_TRSM(SUFF, UPPER, DTYPE)     \
  template [[host_name("applyTRSM" #SUFF "_" #DTYPE)]] \
  kernel void applyTRSM<UPPER, DTYPE>(                 \
      device DTYPE * A [[buffer(0)]],                  \
      constant uint & N [[buffer(2)]],                 \
      constant uint & NB [[buffer(3)]],                \
      constant uint & k [[buffer(4)]],                 \
      uint3 tid [[thread_position_in_threadgroup]],    \
      uint3 tgid [[threadgroup_position_in_grid]],     \
      uint3 tpg [[threads_per_threadgroup]]);

INSTANTIATE_APPLY_TRSM(U, true, float)
INSTANTIATE_APPLY_TRSM(L, false, float)
INSTANTIATE_APPLY_TRSM(U, true, float2)
INSTANTIATE_APPLY_TRSM(L, false, float2)

template <bool upper>
inline void syrk_simdgroup_tile(
    device float* A,
    uint batch_offset,
    uint N,
    uint NB,
    uint k,
    uint row0,
    uint col0,
    uint actSize_h,
    uint actSize_j,
    uint actSize_k,
    bool diag_tile,
    uint warp_id,
    uint simdGroupsPerThreadgroup) {
  simdgroup_matrix<float, 8, 8> negative_identity =
      simdgroup_matrix<float, 8, 8>(-1.0);
  simdgroup_matrix<float, 8, 8> Prod;
  simdgroup_matrix<float, 8, 8> Afrag;
  simdgroup_matrix<float, 8, 8> Bfrag;

  uint numSbX = actSize_h / 8; // How many 8-wide blocks
  uint numSbY = actSize_j / 8; // How many 8-tall blocks
  uint totalSubBlocks = numSbX * numSbY;

  for (uint sb = warp_id; sb < totalSubBlocks; sb += simdGroupsPerThreadgroup) {
    uint sb_y = (sb / numSbX) * 8;
    uint sb_x = (sb % numSbX) * 8;

    // Skip elements that are below diagonal if j == h
    if (diag_tile && sb_y < sb_x) {
      continue;
    }

    // Same logic to load/store Cfrag, Afrag, Bfrag...
    simdgroup_matrix<float, 8, 8> Cfrag;
    simdgroup_load(
        Cfrag,
        &get_ref<upper>(A + batch_offset, row0 + sb_y, col0 + sb_x, N),
        N,
        0,
        !upper);

    for (uint kk = 0; kk < actSize_k; kk += 8) {
      simdgroup_load(
          Afrag,
          &get_ref<upper>(A + batch_offset, row0 + sb_y, k * NB + kk, N),
          N,
          0,
          !upper);
      simdgroup_load(
          Bfrag,
          &get_ref<upper>(A + batch_offset, col0 + sb_x, k * NB + kk, N),
          N,
          /* matrix_origin = */ 0,
          /* transpose = */ upper);

      simdgroup_multiply(Prod, Afrag, Bfrag);
      simdgroup_multiply_accumulate(Cfrag, Prod, negative_identity, Cfrag);
    }

    simdgroup_store(
        Cfrag,
        &get_ref<upper>(A + batch_offset, row0 + sb_y, col0 + sb_x, N),
        N,
        0,
        !upper);
  }
}

// never called; satisfies the complex instantiation (no if constexpr in
// Metal 3)
template <bool upper>
inline void syrk_simdgroup_tile(
    device float2*,
    uint,
    uint,
    uint,
    uint,
    uint,
    uint,
    uint,
    uint,
    uint,
    bool,
    uint,
    uint) {}

template <bool upper, typename T>
kernel void applySYRK(
    device T* A [[buffer(0)]],
    constant uint& N [[buffer(2)]],
    constant uint& NB [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]],
    uint warp_id [[simdgroup_index_in_threadgroup]]) {
  const uint tx = tid.x;
  const uint ty = tid.y;
  const uint simdGroupsPerThreadgroup = (tpg.x * tpg.y + 31) / 32;
  const uint b = tgid.x;
  const uint pairID = tgid.y;

  const uint jRel = (uint)((-1.0 + sqrt(1.0 + 8.0 * float(pairID))) / 2.0);
  const uint hRel = pairID - ((jRel * (jRel + 1)) >> 1);

  const uint startJ = (k + 1);
  const uint j = startJ + jRel;
  const uint h = startJ + hRel;

  const uint row0 = j * NB;
  const uint col0 = h * NB;

  const uint actSize_k = min(int32_t(N - k * NB), int32_t(NB));
  const uint actSize_j = min((uint)(N - row0), NB);
  const uint actSize_h = min((uint)(N - col0), NB);

  const uint batch_offset = b * N * N;

  if (actSize_j == 0 || actSize_h == 0 || actSize_k == 0) {
    return;
  }

  // Check if dimensions are multiples of 8
  // so we can use simdoup matrices
  const bool use_simdgroup = !c10::metal::is_complex_v<T> &&
      (actSize_j % 8 == 0) && (actSize_h % 8 == 0) && (actSize_k % 8 == 0);

  if (use_simdgroup) {
    syrk_simdgroup_tile<upper>(
        A,
        batch_offset,
        N,
        NB,
        k,
        row0,
        col0,
        actSize_h,
        actSize_j,
        actSize_k,
        j == h,
        warp_id,
        simdGroupsPerThreadgroup);
  } else {
    // Fallback for non-multiple-of-8 dimensions
    threadgroup T sum_accumulator[32 * 32];
    for (uint y = ty; y < actSize_j; y += tpg.y) {
      for (uint x = tx; x < actSize_h; x += tpg.x) {
        // since we use this for accumulator, better to set it to 0.0
        // to avoid random values
        sum_accumulator[y * tpg.x + x] = T(0);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint y = ty; y < actSize_j; y += tpg.y) {
      for (uint x = tx; x < actSize_h; x += tpg.x) {
        if (j == h && y < x) {
          continue;
        }

        T sum = T(0);
        for (uint i = 0; i < actSize_k; i++) {
          T a_val = get_ref<upper>(A + batch_offset, row0 + y, k * NB + i, N);
          T b_val = get_ref<upper>(A + batch_offset, col0 + x, k * NB + i, N);
          sum = c10::metal::fma(a_val, c10::metal::conj(b_val), sum);
        }
        sum_accumulator[y * tpg.x + x] += sum;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint y = ty; y < actSize_j; y += tpg.y) {
      for (uint x = tx; x < actSize_h; x += tpg.x) {
        get_ref<upper>(A + batch_offset, row0 + y, col0 + x, N) -=
            sum_accumulator[y * tpg.x + x];
      }
    }
  }
}

#define INSTANTIATE_APPLY_SYRK(SUFF, UPPER, DTYPE)     \
  template [[host_name("applySYRK" #SUFF "_" #DTYPE)]] \
  kernel void applySYRK<UPPER, DTYPE>(                 \
      device DTYPE * A [[buffer(0)]],                  \
      constant uint & N [[buffer(2)]],                 \
      constant uint & NB [[buffer(3)]],                \
      constant uint & k [[buffer(4)]],                 \
      uint3 tid [[thread_position_in_threadgroup]],    \
      uint3 tgid [[threadgroup_position_in_grid]],     \
      uint3 tpg [[threads_per_threadgroup]],           \
      uint sgitg [[simdgroup_index_in_threadgroup]]);

INSTANTIATE_APPLY_SYRK(U, true, float)
INSTANTIATE_APPLY_SYRK(L, false, float)
INSTANTIATE_APPLY_SYRK(U, true, float2)
INSTANTIATE_APPLY_SYRK(L, false, float2)

template <bool upper>
kernel void factorDiagonalPanel(
    device float* A [[buffer(0)]],
    device int* info [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& NB [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]],
    uint warp_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint GROUP = 96; // 3 simdgroups
  const uint tid = tid3.x;
  const uint c0 = k * NB;
  const uint actPanel = min(N - c0, NB);
  const uint S = (actPanel + 31) / 32;
  device float* Ab = A + ulong(bid.x) * N * N;

  // lower tiles of the block; tile (a, b), a >= b, lives at a*(a+1)/2 + b
  threadgroup float blk[6][32][33];
  threadgroup float rdiag[32];
  threadgroup float scratch[1];

  if (k == 0 && tid == 0) {
    info[bid.x] = 0;
  }

  const bool full = (actPanel == NB) && (N % 4 == 0);
  if (full) {
    // float4 over the contiguous device dim; smem stores are scalar (the
    // 33-padded rows are not 16B aligned)
    for (uint i = tid; i < 6 * 256; i += GROUP) {
      const uint t = i / 256;
      const uint e = i % 256;
      const uint a = (t < 1) ? 0 : (t < 3 ? 1 : 2);
      const uint b = t - a * (a + 1) / 2;
      const uint fast = (e & 7) * 4; // contiguous-dim offset
      const uint slow = e >> 3;
      const uint r = upper ? slow : fast;
      const uint c = upper ? fast : slow;
      float4 v = *(device const float4*)(&get_ref<upper>(
          Ab, c0 + 32 * a + r, c0 + 32 * b + c, N));
      if (upper) {
        blk[t][r][c + 0] = v.x;
        blk[t][r][c + 1] = v.y;
        blk[t][r][c + 2] = v.z;
        blk[t][r][c + 3] = v.w;
      } else {
        blk[t][r + 0][c] = v.x;
        blk[t][r + 1][c] = v.y;
        blk[t][r + 2][c] = v.z;
        blk[t][r + 3][c] = v.w;
      }
    }
  } else {
    for (uint i = tid; i < 6 * 1024; i += GROUP) {
      const uint t = i / 1024;
      const uint e = i % 1024;
      const uint a = (t < 1) ? 0 : (t < 3 ? 1 : 2);
      const uint b = t - a * (a + 1) / 2;
      const uint r = upper ? e >> 5 : e & 31;
      const uint c = upper ? e & 31 : e >> 5;
      if (32 * a + r < actPanel && 32 * b + c < actPanel) {
        blk[t][r][c] = get_ref<upper>(Ab, c0 + 32 * a + r, c0 + 32 * b + c, N);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint s = 0; s < S; s++) {
    const uint actS = min(32u, actPanel - 32 * s);
    const uint m = S - 1 - s;
    threadgroup float(&diagT)[32][33] = blk[s * (s + 1) / 2 + s];

    if (warp_id == 0) {
      int f = factor_tile32_warp(diagT, rdiag, actS, lane);
      if (lane == 0) {
        scratch[0] = float(f);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int fail = int(scratch[0]);
    if (fail != 0) {
      // bail without writing the block back; the result is undefined once
      // info is set
      if (tid == 0 && info[bid.x] == 0) {
        info[bid.x] = int(c0 + 32 * s) + fail;
      }
      return;
    }

    // TRSM the strips below within the block, one simdgroup per strip
    if (warp_id < m) {
      const uint rt = s + 1 + warp_id;
      const uint actR = min(32u, actPanel - 32 * rt);
      threadgroup float(&st)[32][33] = blk[rt * (rt + 1) / 2 + s];
      if (lane < actR) {
        float row[32];
        for (uint c = 0; c < 32; c++) {
          row[c] = st[lane][c];
        }
        trsm_row32(row, diagT, rdiag);
        for (uint c = 0; c < 32; c++) {
          st[lane][c] = row[c];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // rank-32 update of the remaining tiles, one simdgroup per (rt, ht) pair
    const uint nPairs = m * (m + 1) / 2;
    if (warp_id < nPairs) {
      const uint jRel = (warp_id == 0) ? 0 : 1;
      const uint hRel = (warp_id == 2) ? 1 : 0;
      const uint rt = s + 1 + jRel;
      const uint ht = s + 1 + hRel;
      const uint actR = min(32u, actPanel - 32 * rt);
      const uint actH = min(32u, actPanel - 32 * ht);
      threadgroup float(&stR)[32][33] = blk[rt * (rt + 1) / 2 + s];
      threadgroup float(&stH)[32][33] = blk[ht * (ht + 1) / 2 + s];
      threadgroup float(&tg)[32][33] = blk[rt * (rt + 1) / 2 + ht];

      if (actR == 32 && actH == 32) {
        simdgroup_float8x8 negI = simdgroup_float8x8(-1.0f);
        for (uint f = 0; f < 16; f++) {
          uint fy = (f / 4) * 8;
          uint fx = (f % 4) * 8;
          if (rt == ht && fy < fx) {
            continue;
          }
          simdgroup_float8x8 C, Af, Bf, P;
          simdgroup_load(C, &tg[fy][fx], 33);
          for (uint kk = 0; kk < 32; kk += 8) {
            simdgroup_load(Af, &stR[fy][kk], 33);
            simdgroup_load(Bf, &stH[fx][kk], 33, 0, /*transpose=*/true);
            simdgroup_multiply(P, Af, Bf);
            simdgroup_multiply_accumulate(C, P, negI, C);
          }
          simdgroup_store(C, &tg[fy][fx], 33);
        }
      } else {
        // ragged last-panel tiles: scalar fallback
        for (uint idx = lane; idx < actR * actH; idx += 32) {
          uint y = idx / actH;
          uint x = idx % actH;
          if (rt == ht && y < x) {
            continue;
          }
          float sum = 0.0f;
          for (uint kk = 0; kk < 32; kk++) {
            sum = fma(stR[y][kk], stH[x][kk], sum);
          }
          tg[y][x] -= sum;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (full) {
    for (uint i = tid; i < 6 * 256; i += GROUP) {
      const uint t = i / 256;
      const uint e = i % 256;
      const uint a = (t < 1) ? 0 : (t < 3 ? 1 : 2);
      const uint b = t - a * (a + 1) / 2;
      const uint fast = (e & 7) * 4;
      const uint slow = e >> 3;
      const uint r = upper ? slow : fast;
      const uint c = upper ? fast : slow;
      float4 v;
      if (upper) {
        v = float4(
            blk[t][r][c], blk[t][r][c + 1], blk[t][r][c + 2], blk[t][r][c + 3]);
      } else {
        v = float4(
            blk[t][r][c], blk[t][r + 1][c], blk[t][r + 2][c], blk[t][r + 3][c]);
      }
      *(device float4*)(&get_ref<upper>(
          Ab, c0 + 32 * a + r, c0 + 32 * b + c, N)) = v;
    }
  } else {
    for (uint i = tid; i < 6 * 1024; i += GROUP) {
      const uint t = i / 1024;
      const uint e = i % 1024;
      const uint a = (t < 1) ? 0 : (t < 3 ? 1 : 2);
      const uint b = t - a * (a + 1) / 2;
      const uint r = upper ? e >> 5 : e & 31;
      const uint c = upper ? e & 31 : e >> 5;
      if (32 * a + r < actPanel && 32 * b + c < actPanel) {
        get_ref<upper>(Ab, c0 + 32 * a + r, c0 + 32 * b + c, N) = blk[t][r][c];
      }
    }
  }
}

#define INSTANTIATE_FACTOR_DIAGONAL_PANEL(SUFF, UPPER) \
  template [[host_name("factorDiagonalPanel" #SUFF)]]  \
  kernel void factorDiagonalPanel<UPPER>(              \
      device float* A [[buffer(0)]],                   \
      device int* info [[buffer(1)]],                  \
      constant uint& N [[buffer(2)]],                  \
      constant uint& NB [[buffer(3)]],                 \
      constant uint& k [[buffer(4)]],                  \
      uint3 tid3 [[thread_position_in_threadgroup]],   \
      uint3 bid [[threadgroup_position_in_grid]],      \
      uint warp_id [[simdgroup_index_in_threadgroup]], \
      uint lane [[thread_index_in_simdgroup]]);

INSTANTIATE_FACTOR_DIAGONAL_PANEL(U, true)
INSTANTIATE_FACTOR_DIAGONAL_PANEL(L, false)

template <bool upper>
kernel void applyPanelTRSM(
    device float* A [[buffer(0)]],
    constant uint& N [[buffer(2)]],
    constant uint& NB [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint warp_id [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint GROUP = 128; // 4 simdgroups
  const uint tid = tid3.x;
  const uint c0 = k * NB;
  const uint r0 = (k + 1) * NB + tgid.y * 32;
  const uint actR = min(32u, N - r0);
  device float* Ab = A + ulong(tgid.x) * N * N;

  threadgroup float strip[32][97];
  threadgroup float diagT[32][33];
  threadgroup float dT[32][33];
  threadgroup float rdiag[32];

  const bool full = (actR == 32) && (N % 4 == 0);
  if (full) {
    for (uint i = tid; i < 768; i += GROUP) {
      uint r, c;
      if (upper) {
        r = i / 24;
        c = (i % 24) * 4;
      } else {
        r = (i & 7) * 4;
        c = i >> 3;
      }
      float4 v =
          *(device const float4*)(&get_ref<upper>(Ab, r0 + r, c0 + c, N));
      if (upper) {
        strip[r][c + 0] = v.x;
        strip[r][c + 1] = v.y;
        strip[r][c + 2] = v.z;
        strip[r][c + 3] = v.w;
      } else {
        strip[r + 0][c] = v.x;
        strip[r + 1][c] = v.y;
        strip[r + 2][c] = v.z;
        strip[r + 3][c] = v.w;
      }
    }
  } else {
    for (uint i = tid; i < actR * 96; i += GROUP) {
      uint r = upper ? i / 96 : i % actR;
      uint c = upper ? i % 96 : i / actR;
      strip[r][c] = get_ref<upper>(Ab, r0 + r, c0 + c, N);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  simdgroup_float8x8 negI = simdgroup_float8x8(-1.0f);
  const bool aligned4 = (N % 4 == 0);
  for (uint s = 0; s < NB / 32; s++) {
    const uint d0 = c0 + 32 * s;

    if (aligned4) {
      for (uint i = tid; i < 256; i += GROUP) {
        uint fast = (i & 7) * 4;
        uint slow = i >> 3;
        uint r = upper ? slow : fast;
        uint c = upper ? fast : slow;
        float4 v =
            *(device const float4*)(&get_ref<upper>(Ab, d0 + r, d0 + c, N));
        if (upper) {
          diagT[r][c + 0] = v.x;
          diagT[r][c + 1] = v.y;
          diagT[r][c + 2] = v.z;
          diagT[r][c + 3] = v.w;
        } else {
          diagT[r + 0][c] = v.x;
          diagT[r + 1][c] = v.y;
          diagT[r + 2][c] = v.z;
          diagT[r + 3][c] = v.w;
        }
      }
    } else {
      for (uint i = tid; i < 1024; i += GROUP) {
        uint r = upper ? i >> 5 : i & 31;
        uint c = upper ? i & 31 : i >> 5;
        diagT[r][c] = get_ref<upper>(Ab, d0 + r, d0 + c, N);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // strip[:, s] -= strip[:, t < s] @ D(s, t)^T; simdgroup w owns fragment
    // row 8w of the 32x32 target, with each D(s, t) staged through smem
    if (s > 0) {
      const uint fy = warp_id * 8;
      simdgroup_float8x8 C[4];
      for (uint j = 0; j < 4; j++) {
        simdgroup_load(C[j], &strip[fy][32 * s + 8 * j], 97);
      }
      for (uint t = 0; t < s; t++) {
        const uint ct = c0 + 32 * t;
        if (aligned4) {
          for (uint i = tid; i < 256; i += GROUP) {
            uint fast = (i & 7) * 4;
            uint slow = i >> 3;
            uint r = upper ? slow : fast;
            uint c = upper ? fast : slow;
            float4 v =
                *(device const float4*)(&get_ref<upper>(Ab, d0 + r, ct + c, N));
            if (upper) {
              dT[r][c + 0] = v.x;
              dT[r][c + 1] = v.y;
              dT[r][c + 2] = v.z;
              dT[r][c + 3] = v.w;
            } else {
              dT[r + 0][c] = v.x;
              dT[r + 1][c] = v.y;
              dT[r + 2][c] = v.z;
              dT[r + 3][c] = v.w;
            }
          }
        } else {
          for (uint i = tid; i < 1024; i += GROUP) {
            uint r = upper ? i >> 5 : i & 31;
            uint c = upper ? i & 31 : i >> 5;
            dT[r][c] = get_ref<upper>(Ab, d0 + r, ct + c, N);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kf = 0; kf < 32; kf += 8) {
          simdgroup_float8x8 Af, P;
          simdgroup_load(Af, &strip[fy][32 * t + kf], 97);
          for (uint j = 0; j < 4; j++) {
            simdgroup_float8x8 Bf;
            simdgroup_load(Bf, &dT[8 * j][kf], 33, 0, /*transpose=*/true);
            simdgroup_multiply(P, Af, Bf);
            simdgroup_multiply_accumulate(C[j], P, negI, C[j]);
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      for (uint j = 0; j < 4; j++) {
        simdgroup_store(C[j], &strip[fy][32 * s + 8 * j], 97);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (warp_id == 0) {
      rdiag[lane] = 1.0f / diagT[lane][lane];
      simdgroup_barrier(mem_flags::mem_threadgroup);
      if (lane < actR) {
        float row[32];
        for (uint c = 0; c < 32; c++) {
          row[c] = strip[lane][32 * s + c];
        }
        trsm_row32(row, diagT, rdiag);
        for (uint c = 0; c < 32; c++) {
          strip[lane][32 * s + c] = row[c];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (full) {
    for (uint i = tid; i < 768; i += GROUP) {
      uint r, c;
      if (upper) {
        r = i / 24;
        c = (i % 24) * 4;
      } else {
        r = (i & 7) * 4;
        c = i >> 3;
      }
      float4 v;
      if (upper) {
        v = float4(
            strip[r][c], strip[r][c + 1], strip[r][c + 2], strip[r][c + 3]);
      } else {
        v = float4(
            strip[r][c], strip[r + 1][c], strip[r + 2][c], strip[r + 3][c]);
      }
      *(device float4*)(&get_ref<upper>(Ab, r0 + r, c0 + c, N)) = v;
    }
  } else {
    for (uint i = tid; i < actR * 96; i += GROUP) {
      uint r = upper ? i / 96 : i % actR;
      uint c = upper ? i % 96 : i / actR;
      get_ref<upper>(Ab, r0 + r, c0 + c, N) = strip[r][c];
    }
  }
}

#define INSTANTIATE_APPLY_PANEL_TRSM(SUFF, UPPER)      \
  template [[host_name("applyPanelTRSM" #SUFF)]]       \
  kernel void applyPanelTRSM<UPPER>(                   \
      device float* A [[buffer(0)]],                   \
      constant uint& N [[buffer(2)]],                  \
      constant uint& NB [[buffer(3)]],                 \
      constant uint& k [[buffer(4)]],                  \
      uint3 tid3 [[thread_position_in_threadgroup]],   \
      uint3 tgid [[threadgroup_position_in_grid]],     \
      uint warp_id [[simdgroup_index_in_threadgroup]], \
      uint lane [[thread_index_in_simdgroup]]);

INSTANTIATE_APPLY_PANEL_TRSM(U, true)
INSTANTIATE_APPLY_PANEL_TRSM(L, false)

#if C10_METAL_HAS_MPP

template <bool upper, int BM, int BN, int NSG>
kernel void applySYRKTrailing(
    device float* A [[buffer(0)]],
    constant uint& N [[buffer(2)]],
    constant uint& NB [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  const int gN = int(N);
  const int o = int((k + 1) * NB);
  const int pc = int(k * NB);
  const int T = gN - o;
  const int K = int(NB);
  const int ro = int(tgid.y) * BM;
  const int co = int(tgid.x) * BN;
  if (ro + BM <= co) {
    return;
  }
  device float* Ab = A + ulong(tgid.z) * ulong(N) * ulong(N);

  constexpr auto desc = upper
      ? mpp::tensor_ops::matmul2d_descriptor(
            BM,
            BN,
            static_cast<int>(dynamic_extent),
            false,
            true,
            false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply)
      : mpp::tensor_ops::matmul2d_descriptor(
            BN,
            BM,
            static_cast<int>(dynamic_extent),
            true,
            false,
            false,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<desc, execution_simdgroups<NSG>> op;

  device float* panel = upper ? (Ab + o * gN + pc) : (Ab + o + pc * gN);
  device float* trail = upper ? (Ab + o * gN + o) : (Ab + o + o * gN);
  const auto eP =
      upper ? dextents<int32_t, 2>(K, T) : dextents<int32_t, 2>(T, K);
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tA(
      panel, eP, array<int32_t, 2>{1, gN});
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tB(
      panel, eP, array<int32_t, 2>{1, gN});
  tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(
      trail, dextents<int32_t, 2>(T, T), array<int32_t, 2>{1, gN});

  const bool inside = (ro + BM <= T) && (co + BN <= T);

  if (inside) {
    if (upper) {
      auto mA = tA.template slice<dynamic_extent, BM>(0, ro);
      auto mB = tB.template slice<dynamic_extent, BN>(0, co);
      auto mC = tC.template slice<BN, BM>(co, ro);
      auto cT = op.template get_destination_cooperative_tensor<
          decltype(mA),
          decltype(mB),
          float>();
      op.run(mA, mB, cT);
      uint16_t e = 0;
      for (auto it = cT.begin(); it != cT.end(); ++it, ++e) {
        auto idx = it.get_multidimensional_index();
        int r = ro + int(idx[1]);
        int c = co + int(idx[0]);
        cT[e] = get_ref<upper>(Ab, o + r, o + c, N) - cT[e];
      }
      cT.store(mC);
    } else {
      auto mA = tA.template slice<BN, dynamic_extent>(co, 0);
      auto mB = tB.template slice<BM, dynamic_extent>(ro, 0);
      auto mC = tC.template slice<BM, BN>(ro, co);
      auto cT = op.template get_destination_cooperative_tensor<
          decltype(mA),
          decltype(mB),
          float>();
      op.run(mA, mB, cT);
      uint16_t e = 0;
      for (auto it = cT.begin(); it != cT.end(); ++it, ++e) {
        auto idx = it.get_multidimensional_index();
        int r = ro + int(idx[0]);
        int c = co + int(idx[1]);
        cT[e] = get_ref<upper>(Ab, o + r, o + c, N) - cT[e];
      }
      cT.store(mC);
    }
  } else {
    if (upper) {
      auto mA = tA.slice(0, ro);
      auto mB = tB.slice(0, co);
      auto mC = tC.slice(co, ro);
      auto cT = op.template get_destination_cooperative_tensor<
          decltype(mA),
          decltype(mB),
          float>();
      op.run(mA, mB, cT);
      uint16_t e = 0;
      for (auto it = cT.begin(); it != cT.end(); ++it, ++e) {
        if (!cT.is_valid_element(e)) {
          continue;
        }
        auto idx = it.get_multidimensional_index();
        int r = ro + int(idx[1]);
        int c = co + int(idx[0]);
        cT[e] = get_ref<upper>(Ab, o + r, o + c, N) - cT[e];
      }
      cT.store(mC);
    } else {
      auto mA = tA.slice(co, 0);
      auto mB = tB.slice(ro, 0);
      auto mC = tC.slice(ro, co);
      auto cT = op.template get_destination_cooperative_tensor<
          decltype(mA),
          decltype(mB),
          float>();
      op.run(mA, mB, cT);
      uint16_t e = 0;
      for (auto it = cT.begin(); it != cT.end(); ++it, ++e) {
        if (!cT.is_valid_element(e)) {
          continue;
        }
        auto idx = it.get_multidimensional_index();
        int r = ro + int(idx[0]);
        int c = co + int(idx[1]);
        cT[e] = get_ref<upper>(Ab, o + r, o + c, N) - cT[e];
      }
      cT.store(mC);
    }
  }
}

#define INSTANTIATE_SYRK_TRAILING(SUFF, UPPER, BM, BN, NSG)      \
  template [[host_name("applySYRKTrailing" #SUFF "_" #BM "_" #BN \
                       "_" #NSG)]] kernel void                   \
  applySYRKTrailing<UPPER, BM, BN, NSG>(                         \
      device float* A [[buffer(0)]],                             \
      constant uint& N [[buffer(2)]],                            \
      constant uint& NB [[buffer(3)]],                           \
      constant uint& k [[buffer(4)]],                            \
      uint3 tgid [[threadgroup_position_in_grid]]);

INSTANTIATE_SYRK_TRAILING(U, true, 64, 64, 4)
INSTANTIATE_SYRK_TRAILING(L, false, 64, 64, 4)
INSTANTIATE_SYRK_TRAILING(U, true, 32, 64, 2)
INSTANTIATE_SYRK_TRAILING(L, false, 32, 64, 2)
INSTANTIATE_SYRK_TRAILING(U, true, 32, 128, 4)
INSTANTIATE_SYRK_TRAILING(L, false, 32, 128, 4)

#endif // C10_METAL_HAS_MPP
