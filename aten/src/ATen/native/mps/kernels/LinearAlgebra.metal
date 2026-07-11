#include <ATen/native/mps/kernels/LinearAlgebra.h>
#include <c10/metal/common.h>
#include <c10/metal/reduction_utils.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
constant uint TILE_DIM = 16;

template <typename T>
inline c10::metal::opmath_t<T> matmul_inner(
    constant T* mat1Data,
    constant T* mat2Data,
    constant array<ulong2, 3>& strides,
    constant uint3& sizes,
    threadgroup c10::metal::opmath_t<T> A_tile[TILE_DIM][TILE_DIM],
    threadgroup c10::metal::opmath_t<T> B_tile[TILE_DIM][TILE_DIM],
    uint2 tid,
    uint2 thread_id) {
  using TA = c10::metal::opmath_t<T>;
  TA sum = 0;

  uint numTiles = (sizes.y + TILE_DIM - 1) / TILE_DIM;
  for (uint t = 0; t < numTiles; t++) {
    uint tiledCol = t * TILE_DIM + tid.x;
    if (thread_id.y < sizes.x && tiledCol < sizes.y) {
      A_tile[tid.y][tid.x] = static_cast<TA>(
          mat1Data[thread_id.y * strides[0].x + tiledCol * strides[0].y]);
    } else {
      A_tile[tid.y][tid.x] = 0;
    }

    uint tiledRow = t * TILE_DIM + tid.y;
    if (tiledRow < sizes.y && thread_id.x < sizes.z) {
      B_tile[tid.y][tid.x] = static_cast<TA>(
          mat2Data[tiledRow * strides[1].x + thread_id.x * strides[1].y]);
    } else {
      B_tile[tid.y][tid.x] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < TILE_DIM; k++) {
      sum += c10::metal::mul(A_tile[tid.y][k], B_tile[k][tid.x]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  return sum;
}

template <typename T, uint N>
inline c10::metal::opmath_t<T> batched_matmul_inner(
    constant T* mat1Data,
    constant T* mat2Data,
    uint batch,
    constant array<ulong, N>& strides,
    constant uint4& sizes,
    threadgroup c10::metal::opmath_t<T> A_tile[TILE_DIM][TILE_DIM],
    threadgroup c10::metal::opmath_t<T> B_tile[TILE_DIM][TILE_DIM],
    uint3 tid,
    uint row,
    uint col) {
  using TA = c10::metal::opmath_t<T>;
  TA sum = 0;

  // Compute batch offsets
  uint batch1Offset = batch * strides[2];
  uint batch2Offset = batch * strides[5];

  uint numTiles = (sizes.y + TILE_DIM - 1) / TILE_DIM;
  for (uint t = 0; t < numTiles; t++) {
    uint tiledCol = t * TILE_DIM + tid.x;
    if (row < sizes.x && tiledCol < sizes.y) {
      A_tile[tid.y][tid.x] = static_cast<TA>(
          mat1Data[batch1Offset + row * strides[1] + tiledCol * strides[0]]);
    } else {
      A_tile[tid.y][tid.x] = 0;
    }

    uint tiledRow = t * TILE_DIM + tid.y;
    if (tiledRow < sizes.y && col < sizes.z) {
      B_tile[tid.y][tid.x] = static_cast<TA>(
          mat2Data[batch2Offset + tiledRow * strides[4] + col * strides[3]]);
    } else {
      B_tile[tid.y][tid.x] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < TILE_DIM; k++) {
      sum += c10::metal::mul(A_tile[tid.y][k], B_tile[k][tid.x]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  return sum;
}

template <typename T>
kernel void matmul(
    constant T* mat1Data [[buffer(0)]],
    constant T* mat2Data [[buffer(1)]],
    device T* outputData [[buffer(2)]],
    constant array<ulong2, 3>& strides [[buffer(3)]],
    constant uint3& sizes [[buffer(4)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 thread_id [[thread_position_in_grid]]) {
  threadgroup c10::metal::opmath_t<T> A_tile[TILE_DIM][TILE_DIM];
  threadgroup c10::metal::opmath_t<T> B_tile[TILE_DIM][TILE_DIM];

  auto sum = matmul_inner(
      mat1Data, mat2Data, strides, sizes, A_tile, B_tile, tid, thread_id);
  if (thread_id.y < sizes.x && thread_id.x < sizes.z) {
    outputData[thread_id.y * strides[2].x + thread_id.x * strides[2].y] =
        static_cast<T>(sum);
  }
}

template <typename T>
kernel void addmm(
    constant T* mat1Data [[buffer(0)]],
    constant T* mat2Data [[buffer(1)]],
    device T* outputData [[buffer(2)]],
    constant T* biasData [[buffer(3)]],
    constant array<c10::metal::opmath_t<T>, 2>& alpha_beta [[buffer(4)]],
    constant array<ulong2, 4>& strides [[buffer(5)]],
    constant uint3& sizes [[buffer(6)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 thread_id [[thread_position_in_grid]]) {
  threadgroup c10::metal::opmath_t<T> A_tile[TILE_DIM][TILE_DIM];
  threadgroup c10::metal::opmath_t<T> B_tile[TILE_DIM][TILE_DIM];

  auto sum = matmul_inner<T>(
      mat1Data,
      mat2Data,
      reinterpret_cast<constant array<ulong2, 3>&>(strides),
      sizes,
      A_tile,
      B_tile,
      tid,
      thread_id);
  if (thread_id.y < sizes.x && thread_id.x < sizes.z) {
    using TA = c10::metal::opmath_t<T>;
    auto bias = static_cast<TA>(
        biasData[thread_id.y * strides[3].x + thread_id.x * strides[3].y]);
    outputData[thread_id.y * strides[2].x + thread_id.x * strides[2].y] =
        static_cast<T>(
            c10::metal::mul(alpha_beta[0], sum) +
            c10::metal::mul(alpha_beta[1], bias));
  }
}

template <typename T>
kernel void naive_bmm(
    constant T* mat1Data [[buffer(0)]],
    constant T* mat2Data [[buffer(1)]],
    device T* outputData [[buffer(2)]],
    constant array<ulong, 9>& strides [[buffer(3)]],
    constant uint4& sizes [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  uint batch = group_id.z;
  uint col = group_id.x * TILE_DIM + tid.x;
  uint row = group_id.y * TILE_DIM + tid.y;

  threadgroup c10::metal::opmath_t<T> A_tile[TILE_DIM][TILE_DIM];
  threadgroup c10::metal::opmath_t<T> B_tile[TILE_DIM][TILE_DIM];

  auto sum = batched_matmul_inner<T, 9>(
      mat1Data, mat2Data, batch, strides, sizes, A_tile, B_tile, tid, row, col);

  if (row < sizes.x && col < sizes.z) {
    outputData[batch * strides[8] + col * strides[6] + row * strides[7]] =
        static_cast<T>(sum);
  }
}

template <typename T>
kernel void naive_baddbmm(
    constant T* mat1Data [[buffer(0)]],
    constant T* mat2Data [[buffer(1)]],
    device T* outputData [[buffer(2)]],
    constant T* biasData [[buffer(3)]],
    constant array<c10::metal::opmath_t<T>, 2>& alpha_beta [[buffer(4)]],
    constant array<ulong, 12>& strides [[buffer(5)]],
    constant uint4& sizes [[buffer(6)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  uint batch = group_id.z;
  uint col = group_id.x * TILE_DIM + tid.x;
  uint row = group_id.y * TILE_DIM + tid.y;

  threadgroup c10::metal::opmath_t<T> A_tile[TILE_DIM][TILE_DIM];
  threadgroup c10::metal::opmath_t<T> B_tile[TILE_DIM][TILE_DIM];

  auto sum = batched_matmul_inner<T, 12>(
      mat1Data, mat2Data, batch, strides, sizes, A_tile, B_tile, tid, row, col);

  if (row < sizes.x && col < sizes.z) {
    using TA = c10::metal::opmath_t<T>;
    uint biasOffset = batch * strides[11];
    auto bias = static_cast<TA>(
        biasData[biasOffset + row * strides[10] + col * strides[9]]);
    outputData[batch * strides[8] + col * strides[6] + row * strides[7]] =
        static_cast<T>(
            c10::metal::mul(alpha_beta[0], sum) +
            c10::metal::mul(alpha_beta[1], bias));
  }
}

template <typename T>
kernel void naive_addbmm(
    constant T* mat1Data [[buffer(0)]],
    constant T* mat2Data [[buffer(1)]],
    device T* outputData [[buffer(2)]],
    constant T* biasData [[buffer(3)]],
    constant array<c10::metal::opmath_t<T>, 2>& alpha_beta [[buffer(4)]],
    constant array<ulong, 12>& strides [[buffer(5)]],
    constant uint4& sizes [[buffer(6)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
  uint col = group_id.x * TILE_DIM + tid.x;
  uint row = group_id.y * TILE_DIM + tid.y;

  c10::metal::opmath_t<T> sum = 0;

  threadgroup c10::metal::opmath_t<T> A_tile[TILE_DIM][TILE_DIM];
  threadgroup c10::metal::opmath_t<T> B_tile[TILE_DIM][TILE_DIM];

  // Iterate through all batches and accumulate
  for (uint batch = 0; batch < sizes.w; batch++) {
    sum += batched_matmul_inner<T, 12>(
        mat1Data,
        mat2Data,
        batch,
        strides,
        sizes,
        A_tile,
        B_tile,
        tid,
        row,
        col);
  }

  if (row < sizes.x && col < sizes.z) {
    using TA = c10::metal::opmath_t<T>;
    auto bias = static_cast<TA>(biasData[row * strides[10] + col * strides[9]]);
    outputData[row * strides[7] + col * strides[6]] = static_cast<T>(
        c10::metal::mul(alpha_beta[0], sum) +
        c10::metal::mul(alpha_beta[1], bias));
  }
}

template <bool col_major>
inline device float& get_ref(device float* A, uint row, uint col, uint N);

template <>
inline device float& get_ref<true>(
    device float* A,
    uint row,
    uint col,
    uint N) {
  return A[row * N + col];
}

template <>
inline device float& get_ref<false>(
    device float* A,
    uint row,
    uint col,
    uint N) {
  return A[row + col * N];
}

inline int factor_tile32_warp(
    threadgroup float (&tile)[32][33],
    threadgroup float (&col)[32],
    uint n,
    uint lane) {
  float row[32];
  for (uint c = 0; c < 32; c++) {
    row[c] = tile[lane][c];
  }
  int ret = 0;
  for (uint kk = 0; kk < n; kk++) {
    float dsq = simd_broadcast(row[kk], ushort(kk));
    if (!(dsq > 0.0f)) {
      ret = int(kk) + 1;
      break;
    }
    float rs = rsqrt(dsq);
    float l = (lane == kk) ? dsq * rs : row[kk] * rs;
    col[lane] = l;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float ccol[32];
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
            row[i] = fma(-l, ccol[i], row[i]);
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
    col[lane] = 1.0f / row[lane];
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

template <bool upper>
kernel void factorDiagonalBlock(
    device float* A [[buffer(0)]],
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

  threadgroup float tile[32][33];
  threadgroup float col[32];
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

template [[host_name("factorDiagonalBlockU")]]
kernel void factorDiagonalBlock<true>(
    device float* A [[buffer(0)]],
    device int* info [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& NB [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]]);

template [[host_name("factorDiagonalBlockL")]]
kernel void factorDiagonalBlock<false>(
    device float* A [[buffer(0)]],
    device int* info [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& NB [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 bid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]]);

template <bool upper>
kernel void applyTRSM(
    device float* A [[buffer(0)]],
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

  threadgroup float diag[32 * 32];
  threadgroup float target[32 * 32];

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
    float diag_val = diag[col * actSize_k + col];
    diag_val = (fabs(diag_val) < 1e-6f) ? copysign(1e-6f, diag_val) : diag_val;

    // multiple rows per thread
    for (uint row = linear_tid; row < actSize_j; row += group_size) {
      float sum = target[row * actSize_k + col];
      // vectorized accumulation
      float4 sum4 = float4(0.0);
      uint p = 0;
      for (; p + 4 <= col; p += 4) {
        float4 target4 = float4(
            target[row * actSize_k + p],
            target[row * actSize_k + p + 1],
            target[row * actSize_k + p + 2],
            target[row * actSize_k + p + 3]);
        float4 diag4 = float4(
            diag[col * actSize_k + p],
            diag[col * actSize_k + p + 1],
            diag[col * actSize_k + p + 2],
            diag[col * actSize_k + p + 3]);
        sum4 = fma(target4, -diag4, sum4);
      }
      sum += sum4.x + sum4.y + sum4.z + sum4.w;

      // remaining elements
      for (; p < col; p++) {
        sum = fma(target[row * actSize_k + p], -diag[col * actSize_k + p], sum);
      }
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

template [[host_name("applyTRSMU")]]
kernel void applyTRSM<true>(
    device float* A [[buffer(0)]],
    constant uint& N [[buffer(2)]],
    constant uint& NB [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]]);

template [[host_name("applyTRSML")]]
kernel void applyTRSM<false>(
    device float* A [[buffer(0)]],
    constant uint& N [[buffer(2)]],
    constant uint& NB [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]]);

template <bool upper>
kernel void applySYRK(
    device float* A [[buffer(0)]],
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
  bool use_simdgroup =
      (actSize_j % 8 == 0) && (actSize_h % 8 == 0) && (actSize_k % 8 == 0);

  if (use_simdgroup) {
    simdgroup_matrix<float, 8, 8> negative_identity =
        simdgroup_matrix<float, 8, 8>(-1.0);
    simdgroup_matrix<float, 8, 8> Prod;
    simdgroup_matrix<float, 8, 8> Afrag;
    simdgroup_matrix<float, 8, 8> Bfrag;

    uint numSbX = actSize_h / 8; // How many 8-wide blocks
    uint numSbY = actSize_j / 8; // How many 8-tall blocks
    uint totalSubBlocks = numSbX * numSbY;

    for (uint sb = warp_id; sb < totalSubBlocks;
         sb += simdGroupsPerThreadgroup) {
      uint sb_y = (sb / numSbX) * 8;
      uint sb_x = (sb % numSbX) * 8;

      // Skip elements that are below diagonal if j == h
      if (j == h && sb_y < sb_x) {
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
  } else {
    // Fallback for non-multiple-of-8 dimensions
    threadgroup float sum_accumulator[32 * 32];
    for (uint y = ty; y < actSize_j; y += tpg.y) {
      for (uint x = tx; x < actSize_h; x += tpg.x) {
        // since we use this for accumulator, better to set it to 0.0
        // to avoid random values
        sum_accumulator[y * tpg.x + x] = 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint y = ty; y < actSize_j; y += tpg.y) {
      for (uint x = tx; x < actSize_h; x += tpg.x) {
        if (j == h && y < x) {
          continue;
        }

        float sum = 0.0f;
        for (uint i = 0; i < actSize_k; i++) {
          float a_val =
              get_ref<upper>(A + batch_offset, row0 + y, k * NB + i, N);
          float b_val =
              get_ref<upper>(A + batch_offset, col0 + x, k * NB + i, N);
          sum = fma(a_val, b_val, sum);
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

template [[host_name("applySYRKU")]]
kernel void applySYRK<true>(
    device float* A [[buffer(0)]],
    constant uint& N [[buffer(2)]],
    constant uint& NB [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]);

template [[host_name("applySYRKL")]]
kernel void applySYRK<false>(
    device float* A [[buffer(0)]],
    constant uint& N [[buffer(2)]],
    constant uint& NB [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]);

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

#if __METAL_VERSION__ >= 400 && \
    __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

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

#endif // __METAL_VERSION__ >= 400 && MetalPerformancePrimitives

// LU factorization with partial pivoting (mirrors LAPACK sgetrf), in place on a
// row-major fp32 (B, M, N) buffer. The host (lu_factor_panel_encode in
// LinearAlgebra.mm) drives a blocked right-looking schedule built from:
//   factorPanelLU / luStream* -> sgetf2 (unblocked panel, isamax pivoting)
//   laswpGatherLU             -> slaswp (apply a block's row interchanges)
//   trsmPanelLU               -> strsm  (unit-lower triangular solve)
//   gemmLU / gemmSimdLU       -> sgemm  (Schur update A22 -= L21 * U12)
//   transposeInPlaceLU        -> row-major factor to column-major LU output
// Buffer slots: (0) A in/out, (1) pivots (1-based), (2) info, (3) dims{M,N},
// (4) per-kernel params, (5) window descriptor, (6) streaming scratch.
// Unblocked 32-wide panel factor; each thread owns R rows, W = 32/R columns.
template <short R, short W>
kernel void factorPanelLU(
    device float* A [[buffer(0)]],
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
  device float* Ab = A + ulong(bid.x) * M * N;
  device int* pv = pivots + ulong(bid.x) * minMN;

  if (d0 == 0 && tid == 0) {
    info[bid.x] = 0;
  }

  threadgroup float pivBuf[W];
  threadgroup float rowJBuf[W];
  threadgroup float wval[32];
  threadgroup uint widx[32];
  threadgroup uint sPiv[1];

  float row[R][W];
  const bool vec4 = ((N % 4u) == 0) && (nb == W);
#pragma unroll
  for (short r = 0; r < R; r++) {
    const uint lr = tid + uint(r) * G;
    if (lr < H) {
      device const float* src = Ab + ulong(d0 + lr) * N + d0;
      if (vec4) {
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
        const float v = fabs(row[r][j]);
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

    const float upiv = pivBuf[j];
    if (upiv != 0.0f) {
      const float rp = 1.0f / upiv;
      // batch the smem loads ahead of the fma burst (in-order pipe)
      float uc[W];
#pragma unroll
      for (short c = 0; c < W; c++) {
        uc[c] = pivBuf[c];
      }
#pragma unroll
      for (short r = 0; r < R; r++) {
        const uint lr = tid + uint(r) * G;
        if (lr < H && lr > j) {
          const float l = row[r][j] * rp;
          row[r][j] = l;
#pragma unroll
          for (short c = 0; c < W; c++) {
            if (uint(c) > j) {
              row[r][c] = fma(-l, uc[c], row[r][c]);
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
      device float* dst = Ab + ulong(d0 + lr) * N + d0;
      if (vec4) {
#pragma unroll
        for (short c = 0; c < W; c += 4) {
          *(device float4*)(dst + c) =
              float4(row[r][c], row[r][c + 1], row[r][c + 2], row[r][c + 3]);
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

#define INSTANTIATE_FACTOR_PANEL_LU(R, W)              \
  template [[host_name("factorPanelLU_" #R "_" #W)]]   \
  kernel void factorPanelLU<R, W>(                     \
      device float* A [[buffer(0)]],                   \
      device int* pivots [[buffer(1)]],                \
      device int* info [[buffer(2)]],                  \
      constant uint2& dims [[buffer(3)]],              \
      constant uint4& params [[buffer(4)]],            \
      uint3 tid3 [[thread_position_in_threadgroup]],   \
      uint3 bid [[threadgroup_position_in_grid]],      \
      uint3 tpg [[threads_per_threadgroup]],           \
      uint warp_id [[simdgroup_index_in_threadgroup]], \
      uint lane [[thread_index_in_simdgroup]]);

INSTANTIATE_FACTOR_PANEL_LU(1, 32)
INSTANTIATE_FACTOR_PANEL_LU(2, 16)
INSTANTIATE_FACTOR_PANEL_LU(4, 8)

// Streaming panel factorization for tall panels (H > kStreamMinRows): factor
// one column at a time across many threadgroups when the register-resident
// factorPanelLU no longer fits. luStreamUpdate applies column j's rank-1 update
// over all rows and writes each threadgroup's local argmax partial to scratch;
// luStreamPivot then reduces those partials to the global pivot for column j.
[[max_total_threads_per_threadgroup(kLUStreamNT)]]
kernel void luStreamUpdate(
    device float* A [[buffer(0)]],
    constant uint2& dims [[buffer(3)]],
    constant uint4& params [[buffer(4)]], // d0, j, RPT, searchOnly
    device float* scratch [[buffer(6)]],
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
  device float* Ab = A + ulong(tgid.y) * M * N;
  device float* scr = scratch + ulong(tgid.y) * (2 * kLUStreamNT + 32);
  device float* uRow = scr + 2 * kLUStreamNT;

  const uint rowStart = searchOnly ? j : j + 1;
  const uint sc = searchOnly ? j : j + 1; // column searched for next pivot
  const uint base = rowStart + (tgid.x * kLUStreamWarpsPerTG + warp_id) * RPT;

  float uc = 0.0f;
  float rp = 0.0f;
  bool doUpdate = false;
  if (!searchOnly) {
    const float upiv = uRow[j];
    doUpdate = upiv != 0.0f;
    rp = doUpdate ? (1.0f / upiv) : 0.0f;
    uc = (lane < nb) ? uRow[lane] : 0.0f;
  }

  float bv = -1.0f;
  uint bi = 0xffffffffu;
  const bool active = lane >= j && lane < nb;
  for (uint r = 0; r < RPT; r++) {
    const uint lr = base + r;
    if (lr >= H) {
      break;
    }
    device float* rowp = Ab + ulong(d0 + lr) * N + d0;
    float v = active ? rowp[lane] : 0.0f;
    if (doUpdate) {
      const float l = simd_broadcast(v, ushort(j)) * rp;
      if (lane == uint(j)) {
        v = l;
      } else if (lane > j && lane < nb) {
        v = fma(-l, uc, v);
      }
      if (active) {
        rowp[lane] = v;
      }
    }
    if (lane == sc && sc < nb) {
      const float av = fabs(v);
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
      scr[tgid.x] = m2;
      ((device uint*)(scr + kLUStreamNT))[tgid.x] = p2;
    }
  }
}

// Reduce luStreamUpdate's per-threadgroup argmax partials to the global pivot
// for column j, record it (1-based, like LAPACK), swap the pivot row, and
// broadcast the resulting U row back to scratch for the next update.
[[max_total_threads_per_threadgroup(kLUStreamNT)]]
kernel void luStreamPivot(
    device float* A [[buffer(0)]],
    device int* pivots [[buffer(1)]],
    device int* info [[buffer(2)]],
    constant uint2& dims [[buffer(3)]],
    constant uint4& params [[buffer(4)]], // d0, j, npart
    device float* scratch [[buffer(6)]],
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
  device float* Ab = A + ulong(tgid.x) * M * N;
  device int* pv = pivots + ulong(tgid.x) * minMN;
  device float* scr = scratch + ulong(tgid.x) * (2 * kLUStreamNT + 32);
  device const uint* sidx = (device const uint*)(scr + kLUStreamNT);
  device float* uRow = scr + 2 * kLUStreamNT;

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
    const float v = scr[i];
    const uint ix = sidx[i];
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
      device float* rj = Ab + ulong(d0 + j) * N + d0 + lane;
      float vj = *rj;
      if (p != j) {
        device float* rp2 = Ab + ulong(d0 + p) * N + d0 + lane;
        const float vp = *rp2;
        *rj = vp;
        *rp2 = vj;
        vj = vp;
      }
      uRow[lane] = vj;
    }
  }
}

// slaswp: apply a block's pivot interchanges as one staged gather/scatter
// through threadgroup memory, not nb sequential row swaps.
kernel void laswpGatherLU(
    device float* A [[buffer(0)]],
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
  device float* Ab = A + ulong(tgid.y) * M * N;
  device const int* pvt = pivots + ulong(tgid.y) * min(M, N) + d0;

  threadgroup uint rowIds[64]; // global row of each slot
  threadgroup uint src[64]; // slot whose staged data this slot receives
  threadgroup uint counts[1];
  threadgroup float stage[64][64];

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
  const uint vbase = tgid.x * 64;
  const bool aligned = (N % 4u) == 0;

  for (uint i = tid; i < nlist * 16; i += G) {
    const uint r = i / 16;
    const uint q = (i % 16) * 4;
    const uint v = vbase + q;
    if (v >= W) {
      continue;
    }
    const uint c = (v < W0) ? (w.x + v) : (w.z + (v - W0));
    const uint cnt = min(4u, W - v);
    device const float* sp = Ab + ulong(rowIds[r]) * N + c;
    if (cnt == 4 && aligned) {
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

  for (uint i = tid; i < nlist * 16; i += G) {
    const uint r = i / 16;
    const uint sr = src[r];
    if (sr == r) {
      continue;
    }
    const uint q = (i % 16) * 4;
    const uint v = vbase + q;
    if (v >= W) {
      continue;
    }
    const uint c = (v < W0) ? (w.x + v) : (w.z + (v - W0));
    const uint cnt = min(4u, W - v);
    device float* dp = Ab + ulong(rowIds[r]) * N + c;
    if (cnt == 4 && aligned) {
      *(device float4*)dp = float4(
          stage[sr][q], stage[sr][q + 1], stage[sr][q + 2], stage[sr][q + 3]);
    } else {
      for (uint e = 0; e < cnt; e++) {
        dp[e] = stage[sr][q + e];
      }
    }
  }
}

// strsm: solve unit-lower L*X = B for the panel's off-diagonal block, with L
// staged in threadgroup memory and one thread per column of B.
template <short TS>
kernel void trsmPanelLU(
    device float* A [[buffer(0)]],
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
  device float* Ab = A + ulong(tgid.x) * M * N;

  threadgroup float L[TS][TS + 1];
  if ((N % 4u) == 0 && nr == TS) {
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
  float x[TS];
#pragma unroll
  for (short r = 0; r < TS; r++) {
    x[r] = (uint(r) < nr) ? Ab[ulong(d0 + r) * N + col] : 0.0f;
  }
#pragma unroll
  for (short c = 0; c < TS; c++) {
    // batch the column loads ahead of the fma burst (in-order pipe)
    float dcol[TS];
#pragma unroll
    for (short i = 0; i < TS; i++) {
      dcol[i] = L[i][c];
    }
    const float xc = x[c];
#pragma unroll
    for (short i = 0; i < TS; i++) {
      if (i > c) {
        x[i] = fma(-xc, dcol[i], x[i]);
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

#define INSTANTIATE_TRSM_PANEL_LU(TS)                \
  template [[host_name("trsmPanelLU_" #TS)]]         \
  kernel void trsmPanelLU<TS>(                       \
      device float* A [[buffer(0)]],                 \
      constant uint2& dims [[buffer(3)]],            \
      constant uint4& params [[buffer(4)]],          \
      uint3 tid3 [[thread_position_in_threadgroup]], \
      uint3 tgid [[threadgroup_position_in_grid]],   \
      uint3 tpg [[threads_per_threadgroup]]);

INSTANTIATE_TRSM_PANEL_LU(8)
INSTANTIATE_TRSM_PANEL_LU(16)
INSTANTIATE_TRSM_PANEL_LU(32)

// In-place square transpose so the row-major factor matches the column-major LU
// view; tiles with tj < ti are produced by their mirror.
kernel void transposeInPlaceLU(
    device float* A [[buffer(0)]],
    constant uint2& dims [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]) {
  const uint N = dims.y;
  const uint ti = tgid.y * 32;
  const uint tj = tgid.x * 32;
  if (tj < ti) {
    return;
  }
  device float* Ab = A + ulong(tgid.z) * N * N;
  threadgroup float ta[32][33];
  threadgroup float tb[32][33];
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

#if __METAL_VERSION__ >= 400 && \
    __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

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

#endif // __METAL_VERSION__ >= 400 && MetalPerformancePrimitives

template <bool upper, bool unit, short TS>
kernel void trsmDiagSolveLU(
    device float* A [[buffer(0)]],
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
  device float* Ab = A + ulong(tgid.x) * M * N;

  threadgroup float T[TS][TS + 1];
  for (uint i = tid; i < TS * TS; i += G) {
    const uint r = i / TS;
    const uint c = i % TS;
    T[r][c] = (r < nr && c < nr) ? Ab[ulong(d0 + r) * N + d0 + c]
                                 : (r == c ? 1.0f : 0.0f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint col = cs + tgid.y * G + tid;
  if (col >= ce) {
    return;
  }
  float x[TS];
#pragma unroll
  for (short r = 0; r < TS; r++) {
    x[r] = (uint(r) < nr) ? Ab[ulong(d0 + r) * N + col] : 0.0f;
  }
  if (!upper) {
#pragma unroll
    for (short c = 0; c < TS; c++) {
      float dcol[TS];
#pragma unroll
      for (short i = 0; i < TS; i++) {
        dcol[i] = T[i][c];
      }
      const float xc = unit ? x[c] : x[c] / T[c][c];
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
      float dcol[TS];
#pragma unroll
      for (short i = 0; i < TS; i++) {
        dcol[i] = T[i][c];
      }
      const float xc = unit ? x[c] : x[c] / T[c][c];
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

#define INSTANTIATE_TRSM_DIAG_SOLVE(UP, UN, SUFF)    \
  template [[host_name("trsmDiagSolveLU_" #SUFF)]]   \
  kernel void trsmDiagSolveLU<UP, UN, 32>(           \
      device float* A [[buffer(0)]],                 \
      constant uint2& dims [[buffer(3)]],            \
      constant uint4& params [[buffer(4)]],          \
      uint3 tid3 [[thread_position_in_threadgroup]], \
      uint3 tgid [[threadgroup_position_in_grid]],   \
      uint3 tpg [[threads_per_threadgroup]]);

INSTANTIATE_TRSM_DIAG_SOLVE(false, true, lower_unit)
INSTANTIATE_TRSM_DIAG_SOLVE(true, false, upper_nonunit)
INSTANTIATE_TRSM_DIAG_SOLVE(false, false, lower_nonunit)
INSTANTIATE_TRSM_DIAG_SOLVE(true, true, upper_unit)

kernel void luApplyPivotsRHS(
    device float* A [[buffer(0)]],
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
  device float* Ab = A + ulong(tgid.x) * M * N;
  device const int* pv = pivots + ulong(tgid.x) * npiv;

  for (uint s = 0; s < npiv; s++) {
    const uint i = inverse ? (npiv - 1 - s) : s;
    const uint p = uint(pv[i] - 1);
    if (p != i) {
      for (uint col = tid; col < k; col += G) {
        const uint cc = coff + col;
        const float t = Ab[ulong(i) * N + cc];
        Ab[ulong(i) * N + cc] = Ab[ulong(p) * N + cc];
        Ab[ulong(p) * N + cc] = t;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

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

template <typename T>
static T bool_to_float(bool b) {
  return static_cast<T>(b);
}

template <>
half2 bool_to_float(bool b) {
  return half2(b ? 1 : 0, 0);
}

template <>
float2 bool_to_float(bool b) {
  return float2(b ? 1 : 0, 0);
}

template <typename T>
static T calc_H_irc(
    device T* A,
    uint32_t A_stride_r,
    uint32_t A_stride_c,
    constant T* tau,
    uint32_t tau_stride,
    uint32_t r,
    uint32_t c,
    uint32_t i) {
  T I_val = bool_to_float<T>(r == c);
  T tau_val = tau[i * tau_stride];

  T A_ci = c10::metal::conj(A[c * A_stride_r + i * A_stride_c]);
  T A_ri = A[r * A_stride_r + i * A_stride_c];

  T c_eq_i = bool_to_float<T>(c == i);
  T r_eq_i = bool_to_float<T>(r == i);

  T A_ci_ = (c > i) ? A_ci : c_eq_i;
  T A_ri_ = (r > i) ? A_ri : r_eq_i;

  return I_val - c10::metal::mul(tau_val, c10::metal::mul(A_ci_, A_ri_));
}

// Calculate (A @ B)[r, c], the element in the r-th row and c-th column of the
// result of matrix multiplying A and B together. A and B must be size m-by-m
// and have the same strides. The formula for this operation, written in Python
// syntax, is:
//   (A @ B)[r, c] = A[r, :].dot(B[:, c])
template <typename T>
static T calc_matmul_rc(
    device T* A,
    device T* B,
    uint32_t stride_r,
    uint32_t stride_c,
    uint32_t m,
    uint32_t r,
    uint32_t c) {
  T AB_rc = 0;
  auto A_row_offset = r * stride_r;
  auto B_col_offset = c * stride_c;

  uint32_t A_col_offset = 0;
  uint32_t B_row_offset = 0;

  for (uint32_t j = 0; j < m;
       j++, A_col_offset += stride_c, B_row_offset += stride_r) {
    AB_rc += c10::metal::mul(
        A[A_row_offset + A_col_offset], B[B_row_offset + B_col_offset]);
  }
  return AB_rc;
}

template <typename T>
kernel void orgqr(
    device T* A [[buffer(0)]],
    constant T* tau [[buffer(1)]],
    device T* H [[buffer(2)]],
    device T* H_prod [[buffer(3)]],
    device T* H_prod_work [[buffer(4)]],
    constant OrgqrParams<>& params [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]]) {
  constant auto& A_strides = params.A_strides;
  constant auto& tau_strides = params.tau_strides;
  constant auto& H_strides = params.H_strides;
  constant auto& H_sizes = params.H_sizes;

  auto num_batch_dims = params.num_batch_dims;
  auto m = params.m;
  auto m2 = params.m2;
  auto n = params.n;
  auto k = params.k;

  auto batch_idx = tgid;

  // Find the matrices for this thread's batch index
  uint32_t A_offset = 0;
  uint32_t tau_offset = 0;
  uint32_t H_offset = 0;

  for (auto dim = num_batch_dims - 1; dim >= 0; dim--) {
    auto dim_size = H_sizes[dim];
    auto dim_idx = batch_idx % dim_size;

    A_offset += dim_idx * A_strides[dim];
    tau_offset += dim_idx * tau_strides[dim];
    H_offset += dim_idx * H_strides[dim];

    batch_idx /= dim_size;
  }

  A += A_offset;
  tau += tau_offset;
  H += H_offset;
  H_prod += H_offset;
  H_prod_work += H_offset;

  auto A_stride_r = A_strides[num_batch_dims];
  auto A_stride_c = A_strides[num_batch_dims + 1];
  auto tau_stride = tau_strides[num_batch_dims];
  auto H_stride_r = H_strides[num_batch_dims];
  auto H_stride_c = H_strides[num_batch_dims + 1];

  for (uint32_t i = 0; i < k; i++) {
    // Calculate and write H_i
    for (auto matrix_idx = tid; matrix_idx < m2; matrix_idx += tptg) {
      auto r = matrix_idx / m;
      auto c = matrix_idx % m;
      T H_irc = calc_H_irc(A, A_stride_r, A_stride_c, tau, tau_stride, r, c, i);

      if (i == 0) {
        H_prod[r * H_stride_r + c * H_stride_c] = H_irc;
      } else {
        H[r * H_stride_r + c * H_stride_c] = H_irc;
      }
    }

    if (i > 0) {
      // Need this sync because the below matmul requires all threads to finish
      // writing their entries to `H_prod` and `H`.
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Calculate H_prod @ H_i, and write result to H_prod_work
      for (auto matrix_idx = tid; matrix_idx < m2; matrix_idx += tptg) {
        auto r = matrix_idx / m;
        auto c = matrix_idx % m;

        T H_prod_0_to_i_rc =
            calc_matmul_rc(H_prod, H, H_stride_r, H_stride_c, m, r, c);

        H_prod_work[r * H_stride_r + c * H_stride_c] = H_prod_0_to_i_rc;
      }

      // Need this sync because the above matmul uses the current values in
      // `H_prod`, and we don't want to overwrite those until all threads are
      // finished using them.
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Copy H_prod_work into H_prod
      for (auto matrix_idx = tid; matrix_idx < m2; matrix_idx += tptg) {
        auto r = matrix_idx / m;
        auto c = matrix_idx % m;
        H_prod[r * H_stride_r + c * H_stride_c] =
            H_prod_work[r * H_stride_r + c * H_stride_c];
      }
    }
  }

  for (auto matrix_idx = tid; matrix_idx < m2; matrix_idx += tptg) {
    auto r = matrix_idx / m;
    auto c = matrix_idx % m;
    if (c < n) {
      A[r * A_stride_r + c * A_stride_c] =
          H_prod[r * H_stride_r + c * H_stride_c];
    }
  }
}

template <typename TO, typename TI>
kernel void unpack_pivots(
    device TO* perm [[buffer(0)]],
    constant TI* pivots [[buffer(1)]],
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

template <typename T>
kernel void linalg_qr_householder(
    device T* A [[buffer(0)]],
    device T* Q [[buffer(1)]],
    device T* R [[buffer(2)]],
    device int* info [[buffer(3)]],
    constant QrParams& params [[buffer(4)]],
    device T* v_work [[buffer(5)]],
    uint3 thread_pos [[thread_position_in_threadgroup]],
    uint3 tpg [[threads_per_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]]) {
  using opmath_t = c10::metal::opmath_t<T>;

  const uint32_t tid = thread_pos.x;
  const uint32_t group_size = tpg.x;
  const uint32_t m = params.m;
  const uint32_t n = params.n;

  // Batch indexing
  const uint32_t batch_idx = tg_pos.x;
  const uint32_t A_stride = m * n;
  const uint32_t Q_stride = m * m;
  const uint32_t R_stride = m * n;
  const uint32_t v_stride = m;

  device T* A_batch = A + batch_idx * A_stride;
  device T* Q_batch = Q + batch_idx * Q_stride;
  device T* R_batch = R + batch_idx * R_stride;
  device T* v_batch = v_work + batch_idx * v_stride;

  constexpr auto kMaxThreadsPerThreadgroup = 1024;
  constexpr auto kMaxSIMDGroups =
      kMaxThreadsPerThreadgroup / c10::metal::simdgroup_size;

  threadgroup opmath_t scratch[kMaxSIMDGroups];
  threadgroup opmath_t tau_shared;

  // initialize Q = Identity (m x m)
  for (uint32_t i = tid; i < m * m; i += group_size) {
    Q_batch[i] = static_cast<T>((i / m == i % m) ? 1.0 : 0.0);
  }

  // initialize R = A (m x n)
  for (uint32_t i = tid; i < m * n; i += group_size) {
    R_batch[i] = A_batch[i];
  }
  threadgroup_barrier(mem_flags::mem_device);

  for (uint32_t k = 0; k < min(m, n); k++) {
    // Step 1: compute norm of R[k:m, k] and copy to v_batch
    opmath_t norm_sq = 0.0;
    for (uint32_t i = k + tid; i < m; i += group_size) {
      T r_ik = R_batch[i * n + k];
      v_batch[i] = r_ik;
      const auto val = static_cast<opmath_t>(r_ik);
      norm_sq = fma(val, val, norm_sq);
    }
    const auto norm = precise::sqrt(
        c10::metal::threadgroup_sum(scratch, norm_sq, tid, group_size));

    // scale norm_eps by matrix dimension to handle accumulated error
    const auto norm_eps = numeric_limits<opmath_t>::epsilon() * m;
    constexpr auto tau_eps = numeric_limits<opmath_t>::epsilon();

    // Step 2: compute Householder vector and tau
    if (tid == 0) {
      // LAPACK convention: skip reflection for last row to preserve natural
      // sign When k == m - 1, there's only one element in the column, so
      // reflection would just flip its sign. Instead, preserve whatever value
      // emerged from prior transformations to match LAPACK's behavior.
      if (fabs(norm) < norm_eps || k == m - 1) {
        tau_shared = 0.0;
      } else {
        opmath_t alpha = static_cast<opmath_t>(v_batch[k]);
        opmath_t sign_alpha = (alpha >= 0.0) ? 1.0 : -1.0;
        opmath_t beta = sign_alpha * norm;
        opmath_t u1 = alpha + beta;

        tau_shared = 1.0 + fabs(alpha) / norm;

        v_batch[k] = static_cast<T>(1.0); // always 1 by construction
        for (uint32_t i = k + 1; i < m; i++) {
          v_batch[i] = static_cast<T>(static_cast<opmath_t>(v_batch[i]) / u1);
        }

        R_batch[k * n + k] = static_cast<T>(-beta);
      }
    }
    threadgroup_barrier(mem_flags::mem_device);

    const auto tau = tau_shared;
    if (tau < tau_eps)
      continue;

    // (zero out column k below diagonal)
    for (uint32_t i = k + 1 + tid; i < m; i += group_size) {
      R_batch[i * n + k] = static_cast<T>(0.0);
    }

    // Step 3: apply reflection to trailing columns of R
    // Parallelize across columns: each SIMD group (32 threads) handles one
    // column
    uint32_t simd_lane = tid % c10::metal::simdgroup_size;
    uint32_t simd_group_id = tid / c10::metal::simdgroup_size;
    uint32_t num_simd_groups = group_size / c10::metal::simdgroup_size;

    for (uint32_t j_base = k + 1; j_base < n; j_base += num_simd_groups) {
      uint32_t j = j_base + simd_group_id;
      if (j < n) {
        // Each SIMD group computes dot product for its column
        // Use SIMD reduction within the group
        opmath_t dot = 0.0;
        for (uint32_t i = k + simd_lane; i < m; i += 32) {
          opmath_t v_i = static_cast<opmath_t>(v_batch[i]);
          opmath_t r_ij = static_cast<opmath_t>(R_batch[i * n + j]);
          dot = fma(v_i, r_ij, dot);
        }
        opmath_t vt_col = simd_sum(dot);
        opmath_t factor = tau * vt_col;

        // Update column
        for (uint32_t i = k + simd_lane; i < m; i += 32) {
          opmath_t v_i = static_cast<opmath_t>(v_batch[i]);
          opmath_t r_ij = static_cast<opmath_t>(R_batch[i * n + j]);
          R_batch[i * n + j] = static_cast<T>(r_ij - v_i * factor);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Step 4: accumulate Q = Q * H_k
    // each SIMD group handles one row
    for (uint32_t i_base = 0; i_base < m; i_base += num_simd_groups) {
      uint32_t i = i_base + simd_group_id;
      if (i < m) {
        opmath_t dot = 0.0;
        for (uint32_t j = k + simd_lane; j < m; j += 32) {
          opmath_t v_j = static_cast<opmath_t>(v_batch[j]);
          opmath_t q_ij = static_cast<opmath_t>(Q_batch[i * m + j]);
          dot = fma(q_ij, v_j, dot);
        }
        opmath_t row_v = simd_sum(dot);
        opmath_t factor = tau * row_v;

        // Update row
        for (uint32_t j = k + simd_lane; j < m; j += 32) {
          opmath_t v_j = static_cast<opmath_t>(v_batch[j]);
          opmath_t q_ij = static_cast<opmath_t>(Q_batch[i * m + j]);
          Q_batch[i * m + j] = static_cast<T>(q_ij - v_j * factor);
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_device);
  }

  if (tid == 0) {
    info[0] = 0;
  }
}

#define REGISTER_QR(T)                                \
  template [[host_name("linalg_qr_householder_" #T)]] \
  kernel void linalg_qr_householder<T>(               \
      device T * A [[buffer(0)]],                     \
      device T * Q [[buffer(1)]],                     \
      device T * R [[buffer(2)]],                     \
      device int* info [[buffer(3)]],                 \
      constant QrParams& params [[buffer(4)]],        \
      device T* v_work [[buffer(5)]],                 \
      uint3 tid [[thread_position_in_threadgroup]],   \
      uint3 tpg [[threads_per_threadgroup]],          \
      uint3 tg_pos [[threadgroup_position_in_grid]]);

REGISTER_QR(float);

#define INSTANTIATE_MM_OPS(DTYPE)                                           \
  template [[host_name("matmul_" #DTYPE)]] kernel void matmul<DTYPE>(       \
      constant DTYPE * mat1Data [[buffer(0)]],                              \
      constant DTYPE * mat2Data [[buffer(1)]],                              \
      device DTYPE * outputData [[buffer(2)]],                              \
      constant array<ulong2, 3> & strides [[buffer(3)]],                    \
      constant uint3 & sizes [[buffer(4)]],                                 \
      uint2 tid [[thread_position_in_threadgroup]],                         \
      uint2 group_id [[threadgroup_position_in_grid]]);                     \
  template [[host_name("naive_bmm_" #DTYPE)]] kernel void naive_bmm<DTYPE>( \
      constant DTYPE * mat1Data [[buffer(0)]],                              \
      constant DTYPE * mat2Data [[buffer(1)]],                              \
      device DTYPE * outputData [[buffer(2)]],                              \
      constant array<ulong, 9> & strides [[buffer(3)]],                     \
      constant uint4 & sizes [[buffer(4)]],                                 \
      uint3 tid [[thread_position_in_threadgroup]],                         \
      uint3 group_id [[threadgroup_position_in_grid]]);                     \
  template [[host_name("addmm_" #DTYPE)]] kernel void addmm<DTYPE>(         \
      constant DTYPE * mat1Data [[buffer(0)]],                              \
      constant DTYPE * mat2Data [[buffer(1)]],                              \
      device DTYPE * outputData [[buffer(2)]],                              \
      constant DTYPE * biasData [[buffer(3)]],                              \
      constant array<c10::metal::opmath_t<DTYPE>, 2> &                      \
          alpha_beta [[buffer(4)]],                                         \
      constant array<ulong2, 4> & strides [[buffer(5)]],                    \
      constant uint3 & sizes [[buffer(6)]],                                 \
      uint2 tid [[thread_position_in_threadgroup]],                         \
      uint2 group_id [[threadgroup_position_in_grid]]);                     \
  template [[host_name("naive_baddbmm_" #DTYPE)]]                           \
  kernel void naive_baddbmm<DTYPE>(                                         \
      constant DTYPE * mat1Data [[buffer(0)]],                              \
      constant DTYPE * mat2Data [[buffer(1)]],                              \
      device DTYPE * outputData [[buffer(2)]],                              \
      constant DTYPE * biasData [[buffer(3)]],                              \
      constant array<c10::metal::opmath_t<DTYPE>, 2> &                      \
          alpha_beta [[buffer(4)]],                                         \
      constant array<ulong, 12> & strides [[buffer(5)]],                    \
      constant uint4 & sizes [[buffer(6)]],                                 \
      uint3 tid [[thread_position_in_threadgroup]],                         \
      uint3 group_id [[threadgroup_position_in_grid]]);                     \
  template [[host_name("naive_addbmm_" #DTYPE)]]                            \
  kernel void naive_addbmm<DTYPE>(                                          \
      constant DTYPE * mat1Data [[buffer(0)]],                              \
      constant DTYPE * mat2Data [[buffer(1)]],                              \
      device DTYPE * outputData [[buffer(2)]],                              \
      constant DTYPE * biasData [[buffer(3)]],                              \
      constant array<c10::metal::opmath_t<DTYPE>, 2> &                      \
          alpha_beta [[buffer(4)]],                                         \
      constant array<ulong, 12> & strides [[buffer(5)]],                    \
      constant uint4 & sizes [[buffer(6)]],                                 \
      uint3 tid [[thread_position_in_threadgroup]],                         \
      uint3 group_id [[threadgroup_position_in_grid]])

INSTANTIATE_MM_OPS(float);
INSTANTIATE_MM_OPS(half);
INSTANTIATE_MM_OPS(bfloat);

// Complex MM
INSTANTIATE_MM_OPS(float2);
INSTANTIATE_MM_OPS(half2);

// Integral MM
INSTANTIATE_MM_OPS(long);
INSTANTIATE_MM_OPS(int);
INSTANTIATE_MM_OPS(short);
INSTANTIATE_MM_OPS(char);
INSTANTIATE_MM_OPS(uchar);

#define REGISTER_ORGQR(T)                            \
  template [[host_name("orgqr_" #T)]]                \
  kernel void orgqr<T>(                              \
      device T * A [[buffer(0)]],                    \
      constant T * tau [[buffer(1)]],                \
      device T * H [[buffer(2)]],                    \
      device T * H_prod [[buffer(3)]],               \
      device T * H_prod_work [[buffer(4)]],          \
      constant OrgqrParams<> & params [[buffer(5)]], \
      uint tid [[thread_position_in_threadgroup]],   \
      uint tptg [[threads_per_threadgroup]],         \
      uint tgid [[threadgroup_position_in_grid]]);

REGISTER_ORGQR(float);
REGISTER_ORGQR(half);
REGISTER_ORGQR(bfloat);
REGISTER_ORGQR(float2);
REGISTER_ORGQR(half2);

#define REGISTER_UNPACK_PIVOTS(TO, TI)                    \
  template [[host_name("unpack_pivots_" #TO "_" #TI)]]    \
  kernel void unpack_pivots<TO, TI>(                      \
      device TO * perm [[buffer(0)]],                     \
      constant TI * pivots [[buffer(1)]],                 \
      constant UnpackPivotsParams & params [[buffer(2)]], \
      uint tid [[thread_position_in_grid]]);

REGISTER_UNPACK_PIVOTS(int, int);
REGISTER_UNPACK_PIVOTS(int, long);
REGISTER_UNPACK_PIVOTS(long, int);
REGISTER_UNPACK_PIVOTS(long, long);

template <typename T>
struct svd_real {
  using type = T;
};
template <>
struct svd_real<float2> {
  using type = float;
};
template <typename T>
using svd_real_t = typename svd_real<T>::type;

inline float svd_abs2(float z) {
  return z * z;
}
inline float svd_abs2(float2 z) {
  return z.x * z.x + z.y * z.y;
}
inline float svd_conjmul(float a, float b) {
  return a * b;
}
inline float2 svd_conjmul(float2 a, float2 b) {
  return float2(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
}
inline float svd_conj(float z) {
  return z;
}
inline float2 svd_conj(float2 z) {
  return float2(z.x, -z.y);
}
inline float svd_mul(float a, float b) {
  return a * b;
}
inline float2 svd_mul(float2 a, float2 b) {
  return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
inline float svd_simd_sum(float v) {
  return c10::metal::simd_sum(v);
}
inline float2 svd_simd_sum(float2 v) {
  return float2(c10::metal::simd_sum(v.x), c10::metal::simd_sum(v.y));
}
inline float svd_one(float) {
  return 1.0f;
}
inline float2 svd_one(float2) {
  return float2(1.0f, 0.0f);
}
inline float svd_real_part(float z) {
  return z;
}
inline float svd_real_part(float2 z) {
  return z.x;
}
// NB: float2(x) -> (x,x), so build real T explicitly.
inline float svd_from_real(float, float x) {
  return x;
}
inline float2 svd_from_real(float2, float x) {
  return float2(x, 0.0f);
}

template <typename T>
kernel void svd_jacobi(
    device const T* A [[buffer(0)]],
    device T* U [[buffer(1)]],
    device svd_real_t<T>* S [[buffer(2)]],
    device T* V [[buffer(3)]],
    device T* Vacc [[buffer(4)]], // rotation accumulator when V not staged
    device int* info [[buffer(5)]],
    constant SvdParams& params [[buffer(6)]],
    threadgroup T* Atg [[threadgroup(0)]],
    threadgroup T* Vtg [[threadgroup(1)]],
    uint3 thread_pos [[thread_position_in_threadgroup]],
    uint3 tpg [[threads_per_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint32_t tid = thread_pos.x;
  const uint32_t group_size = tpg.x;
  const uint32_t m = params.m;
  const uint32_t n = params.n;
  const uint32_t batch_idx = tg_pos.x;
  const uint32_t kSimd = c10::metal::simdgroup_size;
  const uint32_t num_sg = group_size / kSimd;

  device const T* A_b = A + batch_idx * m * n;
  device T* U_b = U + batch_idx * params.u_bstride;
  device T* V_b = V + batch_idx * params.v_bstride;
  device T* Vacc_b = Vacc + batch_idx * n * n;

  // Stage A column-major so each lane's row access is contiguous.
  for (uint32_t idx = tid; idx < m * n; idx += group_size) {
    uint32_t row = idx / n, col = idx % n;
    Atg[col * m + row] = A_b[idx];
  }
  if (params.compute_uv) {
    if (params.stage_v) {
      for (uint32_t i = tid; i < n * n; i += group_size) {
        uint32_t row = i / n, col = i % n;
        // NB: float2(1.0) broadcasts to (1,1); use svd_one()/T(0) for a real
        // 1/0.
        Vtg[col * n + row] = (row == col) ? svd_one(T(0)) : T(0);
      }
    } else {
      for (uint32_t i = tid; i < n * n; i += group_size) {
        Vacc_b[i] = (i / n == i % n) ? svd_one(T(0)) : T(0);
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

  constexpr auto eps = numeric_limits<float>::epsilon();
  // Concurrent SIMD-groups flag "I rotated"; a plain flag races, so use an
  // atomic.
  threadgroup atomic_uint any_rotation;

  // Round-robin tournament pairing (closed-form circle method): pad to even ne;
  // each sweep is ne-1 rounds of ne/2 disjoint pairs; index >= n is phantom.
  const uint32_t ne = n + (n & 1u);
  const uint32_t n_pairs = ne / 2;

  uint32_t sweep = 0;
  for (; sweep < params.max_sweeps; ++sweep) {
    if (tid == 0) {
      atomic_store_explicit(&any_rotation, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint32_t round = 0; round < ne - 1; ++round) {
      for (uint32_t k = simd_group; k < n_pairs; k += num_sg) {
        uint32_t p = (k == 0) ? 0u : ((k - 1 + round) % (ne - 1)) + 1u;
        uint32_t kq = ne - 1 - k;
        uint32_t q = (kq == 0) ? 0u : ((kq - 1 + round) % (ne - 1)) + 1u;
        bool act = !(p >= n || q >= n);
        if (act && p > q) {
          uint32_t tmp = p;
          p = q;
          q = tmp;
        }

        threadgroup T* colP = Atg + p * m;
        threadgroup T* colQ = Atg + q * m;
        float app = 0, aqq = 0;
        T apq_acc = T(0);
        if (act) {
          for (uint32_t i = simd_lane; i < m; i += kSimd) {
            T vp = colP[i];
            T vq = colQ[i];
            app += svd_abs2(vp);
            aqq += svd_abs2(vq);
            apq_acc += svd_conjmul(vp, vq);
          }
        }
        app = c10::metal::simd_sum(app);
        aqq = c10::metal::simd_sum(aqq);
        apq_acc = svd_simd_sum(apq_acc);

        if (!act) {
          continue;
        }
        float apq_abs = precise::sqrt(svd_abs2(apq_acc));
        float off = precise::sqrt(app * aqq);
        if (off < eps || apq_abs <= params.tol * off) {
          continue;
        }
        if (simd_lane == 0) {
          atomic_store_explicit(&any_rotation, 1u, memory_order_relaxed);
        }
        T phi = (apq_abs > 0) ? (apq_acc * (1.0f / apq_abs)) : svd_one(T(0));
        float tau = (aqq - app) / (2 * apq_abs);
        float t = (tau >= 0 ? 1.0f : -1.0f) /
            (fabs(tau) + precise::sqrt(1 + tau * tau));
        float c = 1 / precise::sqrt(1 + t * t);
        float s = c * t;
        T cphi = svd_conj(phi);
        for (uint32_t i = simd_lane; i < m; i += kSimd) {
          T vp = colP[i];
          T vq = colQ[i];
          colP[i] = c * vp - svd_mul(cphi, s * vq);
          colQ[i] = svd_mul(phi, s * vp) + c * vq;
        }
        if (params.compute_uv) {
          if (params.stage_v) {
            threadgroup T* vP = Vtg + p * n;
            threadgroup T* vQ = Vtg + q * n;
            for (uint32_t i = simd_lane; i < n; i += kSimd) {
              T vp = vP[i];
              T vq = vQ[i];
              vP[i] = c * vp - svd_mul(cphi, s * vq);
              vQ[i] = svd_mul(phi, s * vp) + c * vq;
            }
          } else {
            device T* vP = Vacc_b + p * n;
            device T* vQ = Vacc_b + q * n;
            for (uint32_t i = simd_lane; i < n; i += kSimd) {
              T vp = vP[i];
              T vq = vQ[i];
              vP[i] = c * vp - svd_mul(cphi, s * vq);
              vQ[i] = svd_mul(phi, s * vp) + c * vq;
            }
          }
        }
      }
      threadgroup_barrier(
          params.stage_v
              ? mem_flags::mem_threadgroup
              : (mem_flags::mem_threadgroup | mem_flags::mem_device));
    }

    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    threadgroup uint32_t do_break = 0;
    if (tid == 0) {
      do_break =
          atomic_load_explicit(&any_rotation, memory_order_relaxed) == 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (do_break) {
      break;
    }
  }

  // n <= 90 (host staging gate); 96 gives headroom.
  threadgroup float sig[96];
  threadgroup uint32_t ord[96];
  for (uint32_t j = simd_group; j < n; j += num_sg) {
    threadgroup T* colj = Atg + j * m;
    float norm_sq = 0;
    for (uint32_t i = simd_lane; i < m; i += kSimd) {
      norm_sq += svd_abs2(colj[i]);
    }
    float sigma = precise::sqrt(c10::metal::simd_sum(norm_sq));
    if (simd_lane == 0) {
      sig[j] = sigma;
      ord[j] = j;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (tid == 0) {
    for (uint32_t a = 0; a < n; ++a) {
      uint32_t best = a;
      for (uint32_t b = a + 1; b < n; ++b) {
        if (sig[ord[b]] > sig[ord[best]])
          best = b;
      }
      uint32_t tmp = ord[a];
      ord[a] = ord[best];
      ord[best] = tmp;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Emit column j from source ord[j]. Transposed run swaps left/right targets;
  // right vectors written as Vh rows are conjugated (Vh = V^H), left vectors
  // not.
  for (uint32_t j = simd_group; j < n; j += num_sg) {
    uint32_t src = ord[j];
    float sigma = sig[src];
    if (simd_lane == 0) {
      S[batch_idx * n + j] = sigma;
    }
    float inv = sigma > eps ? (1 / sigma) : 0.0f;
    threadgroup T* colsrc = Atg + src * m;
    if (params.transposed == 0u) {
      for (uint32_t i = simd_lane; i < m; i += kSimd) {
        U_b[j * params.u_ld + i] = inv * colsrc[i];
      }
      if (params.compute_uv) {
        threadgroup T* vsrc = Vtg + src * n;
        for (uint32_t c = simd_lane; c < n; c += kSimd) {
          T v = params.stage_v ? vsrc[c] : Vacc_b[src * n + c];
          V_b[c * params.v_ld + j] = svd_conj(v);
        }
      }
    } else {
      for (uint32_t i = simd_lane; i < m; i += kSimd) {
        V_b[i * params.v_ld + j] = svd_conj(inv * colsrc[i]);
      }
      if (params.compute_uv) {
        threadgroup T* vsrc = Vtg + src * n;
        for (uint32_t c = simd_lane; c < n; c += kSimd) {
          U_b[j * params.u_ld + c] =
              params.stage_v ? vsrc[c] : Vacc_b[src * n + c];
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device);

  if (tid == 0) {
    // NaN/Inf never triggers a rotation, so flag info to raise like the CPU
    // path.
    bool nonfinite = false;
    for (uint32_t j = 0; j < n; ++j) {
      if (!isfinite(sig[j])) {
        nonfinite = true;
        break;
      }
    }
    info[batch_idx] = (nonfinite || sweep >= params.max_sweeps)
        ? static_cast<int>(sweep + 1)
        : 0;
  }
}

#define REGISTER_SVD_JACOBI(T)                             \
  template [[host_name("svd_jacobi_" #T)]]                 \
  kernel void svd_jacobi<T>(                               \
      device const T* A [[buffer(0)]],                     \
      device T* U [[buffer(1)]],                           \
      device svd_real_t<T>* S [[buffer(2)]],               \
      device T* V [[buffer(3)]],                           \
      device T* Vacc [[buffer(4)]],                        \
      device int* info [[buffer(5)]],                      \
      constant SvdParams& params [[buffer(6)]],            \
      threadgroup T* Atg [[threadgroup(0)]],               \
      threadgroup T* Vtg [[threadgroup(1)]],               \
      uint3 thread_pos [[thread_position_in_threadgroup]], \
      uint3 tpg [[threads_per_threadgroup]],               \
      uint3 tg_pos [[threadgroup_position_in_grid]],       \
      uint simd_lane [[thread_index_in_simdgroup]],        \
      uint simd_group [[simdgroup_index_in_threadgroup]]);

REGISTER_SVD_JACOBI(float);
REGISTER_SVD_JACOBI(float2);

template <typename T>
kernel void eigh_jacobi(
    device T* A [[buffer(0)]],
    device svd_real_t<T>* W [[buffer(1)]],
    device T* Q [[buffer(2)]],
    device int* info [[buffer(3)]],
    constant EighParams& params [[buffer(4)]],
    threadgroup T* Atg [[threadgroup(0)]],
    threadgroup T* Qtg [[threadgroup(1)]],
    uint3 thread_pos [[thread_position_in_threadgroup]],
    uint3 tpg [[threads_per_threadgroup]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  const uint32_t tid = thread_pos.x;
  const uint32_t group_size = tpg.x;
  const uint32_t n = params.n;
  const uint32_t batch_idx = tg_pos.x;
  const uint32_t kSimd = c10::metal::simdgroup_size;
  const uint32_t num_sg = group_size / kSimd;
  const bool compute_v = params.compute_v != 0u;

  device T* A_b = A + batch_idx * n * n;
  device T* Q_b = Q + batch_idx * n * n;

  // Stage A into Atg, symmetrizing from the selected UPLO triangle (input may
  // be non-Hermitian otherwise); two-sided Jacobi needs an exactly Hermitian
  // matrix.
  const bool upper = params.upper != 0u;
  for (uint32_t i = tid; i < n * n; i += group_size) {
    uint32_t row = i % n, col = i / n;
    if (row == col) {
      Atg[i] = svd_from_real(T(0), svd_real_part(A_b[i]));
    } else {
      bool in_upper = row < col;
      if (in_upper == upper) {
        Atg[i] = A_b[i];
      } else {
        Atg[i] = svd_conj(A_b[col + row * n]);
      }
    }
  }
  if (compute_v) {
    for (uint32_t i = tid; i < n * n; i += group_size) {
      uint32_t row = i % n, col = i / n;
      Qtg[i] = (row == col) ? svd_one(T(0)) : T(0);
    }
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

  threadgroup float cbuf[48];
  threadgroup T sbuf[48];
  threadgroup uint32_t pbuf[48], qbuf[48];
  // Concurrent SIMD-groups flag "I rotated"; a plain flag races, so use an
  // atomic.
  threadgroup atomic_uint any_rotation;

  const uint32_t ne = n + (n & 1u);
  const uint32_t n_pairs = ne / 2;

  threadgroup float red_diag[16];
  threadgroup float red_off[16];

  uint32_t sweep = 0;
  for (; sweep < params.max_sweeps; ++sweep) {
    if (tid == 0) {
      atomic_store_explicit(&any_rotation, 0u, memory_order_relaxed);
    }
    {
      float ld = 0.0f;
      float lo = 0.0f;
      for (uint32_t i = tid; i < n * n; i += group_size) {
        uint32_t row = i % n, col = i / n;
        float a2 = svd_abs2(Atg[i]);
        if (row == col) {
          ld = max(ld, a2);
        } else {
          lo = max(lo, a2);
        }
      }
      ld = c10::metal::simd_max(ld);
      lo = c10::metal::simd_max(lo);
      if (simd_lane == 0) {
        red_diag[simd_group] = ld;
        red_off[simd_group] = lo;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float g2 = 0.0f;
    float o2 = 0.0f;
    for (uint32_t s = 0; s < num_sg; ++s) {
      g2 = max(g2, red_diag[s]);
      o2 = max(o2, red_off[s]);
    }
    const float gscale = precise::sqrt(g2);
    if (o2 <= params.tol * params.tol * g2) {
      break;
    }

    for (uint32_t round = 0; round < ne - 1; ++round) {
      for (uint32_t k = simd_group; k < n_pairs; k += num_sg) {
        uint32_t p = (k == 0) ? 0u : ((k - 1 + round) % (ne - 1)) + 1u;
        uint32_t kq = ne - 1 - k;
        uint32_t q = (kq == 0) ? 0u : ((kq - 1 + round) % (ne - 1)) + 1u;
        bool act = !(p >= n || q >= n || p == q);
        if (act && p > q) {
          uint32_t t = p;
          p = q;
          q = t;
        }
        if (!act) {
          if (simd_lane == 0) {
            pbuf[k] = n;
            qbuf[k] = n;
          }
          continue;
        }
        float app = svd_real_part(Atg[p * n + p]);
        float aqq = svd_real_part(Atg[q * n + q]);
        T apq = Atg[q * n + p];
        float apq_abs = precise::sqrt(svd_abs2(apq));
        float off = precise::sqrt(::metal::fabs(app * aqq));
        float c = 1.0f;
        T s = T(0);
        float thresh = max(params.tol * off, params.tol * gscale);
        bool rotate = apq_abs > thresh + 1e-30f;
        if (rotate) {
          if (simd_lane == 0) {
            atomic_store_explicit(&any_rotation, 1u, memory_order_relaxed);
          }
          T phi = apq * (1.0f / apq_abs);
          float tau = (aqq - app) / (2.0f * apq_abs);
          float t = (tau >= 0 ? 1.0f : -1.0f) /
              (fabs(tau) + precise::sqrt(1.0f + tau * tau));
          c = 1.0f / precise::sqrt(1.0f + t * t);
          float sreal = c * t;
          s = svd_mul(phi, svd_from_real(T(0), sreal));
        }
        if (simd_lane == 0) {
          cbuf[k] = c;
          sbuf[k] = s;
          pbuf[k] = rotate ? p : n;
          qbuf[k] = q;
        }
        if (!rotate) {
          continue;
        }
        T cs = svd_conj(s);
        threadgroup T* colP = Atg + p * n;
        threadgroup T* colQ = Atg + q * n;
        for (uint32_t i = simd_lane; i < n; i += kSimd) {
          T ap = colP[i], aq = colQ[i];
          colP[i] = c * ap - svd_mul(cs, aq);
          colQ[i] = svd_mul(s, ap) + c * aq;
        }
        if (compute_v) {
          threadgroup T* qP = Qtg + p * n;
          threadgroup T* qQ = Qtg + q * n;
          for (uint32_t i = simd_lane; i < n; i += kSimd) {
            T qp = qP[i], qq = qQ[i];
            qP[i] = c * qp - svd_mul(cs, qq);
            qQ[i] = svd_mul(s, qp) + c * qq;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (uint32_t k = simd_group; k < n_pairs; k += num_sg) {
        uint32_t p = pbuf[k], q = qbuf[k];
        if (p >= n) {
          continue;
        }
        float c = cbuf[k];
        T s = sbuf[k];
        T cs = svd_conj(s);
        for (uint32_t col = simd_lane; col < n; col += kSimd) {
          T ap = Atg[col * n + p], aq = Atg[col * n + q];
          Atg[col * n + p] = c * ap - svd_mul(s, aq);
          Atg[col * n + q] = svd_mul(cs, ap) + c * aq;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (atomic_load_explicit(&any_rotation, memory_order_relaxed) == 0u) {
      break;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  threadgroup float wv[96];
  threadgroup uint32_t ord[96];
  for (uint32_t j = simd_group; j < n; j += num_sg) {
    if (simd_lane == 0) {
      wv[j] = svd_real_part(Atg[j * n + j]);
      ord[j] = j;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint32_t j = simd_group; j < n; j += num_sg) {
    float vj = wv[j];
    uint32_t cnt = 0;
    for (uint32_t k = simd_lane; k < n; k += kSimd) {
      float vk = wv[k];
      cnt += (vk < vj || (vk == vj && k < j)) ? 1u : 0u;
    }
    cnt = c10::metal::simd_sum(cnt);
    if (simd_lane == 0) {
      ord[cnt] = j;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint32_t j = simd_group; j < n; j += num_sg) {
    uint32_t src = ord[j];
    if (simd_lane == 0) {
      W[batch_idx * n + j] = wv[src];
    }
    if (compute_v) {
      threadgroup T* qs = Qtg + src * n;
      for (uint32_t i = simd_lane; i < n; i += kSimd) {
        Q_b[j * n + i] = qs[i];
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_device);

  if (tid == 0) {
    // NaN/Inf never triggers a rotation, so flag info to raise like the CPU
    // path.
    bool nonfinite = false;
    for (uint32_t j = 0; j < n; ++j) {
      if (!isfinite(wv[j])) {
        nonfinite = true;
        break;
      }
    }
    info[batch_idx] = (nonfinite || sweep >= params.max_sweeps)
        ? static_cast<int>(sweep + 1)
        : 0;
  }
}

#define REGISTER_EIGH_JACOBI(T)                            \
  template [[host_name("eigh_jacobi_" #T)]]                \
  kernel void eigh_jacobi<T>(                              \
      device T * A [[buffer(0)]],                          \
      device svd_real_t<T> * W [[buffer(1)]],              \
      device T * Q [[buffer(2)]],                          \
      device int* info [[buffer(3)]],                      \
      constant EighParams& params [[buffer(4)]],           \
      threadgroup T* Atg [[threadgroup(0)]],               \
      threadgroup T* Qtg [[threadgroup(1)]],               \
      uint3 thread_pos [[thread_position_in_threadgroup]], \
      uint3 tpg [[threads_per_threadgroup]],               \
      uint3 tg_pos [[threadgroup_position_in_grid]],       \
      uint simd_lane [[thread_index_in_simdgroup]],        \
      uint simd_group [[simdgroup_index_in_threadgroup]]);

REGISTER_EIGH_JACOBI(float);
REGISTER_EIGH_JACOBI(float2);
