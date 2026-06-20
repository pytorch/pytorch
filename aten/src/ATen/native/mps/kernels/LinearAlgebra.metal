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

inline float blockReduceSum(
    threadgroup float* sharedScratch,
    float val,
    uint linear_tid) {
  float simd_result = simd_sum(val);
  // each warp's first index should write the result to consecutive
  // ids in sharedScratch buffer
  if (linear_tid % 32 == 0) {
    sharedScratch[linear_tid / 32] = simd_result;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // final reduction across first warp
  if (linear_tid < 8) { // 256/32 = 8 simdgroups
    float sum = sharedScratch[linear_tid];
    sum = simd_sum(sum);
    sharedScratch[0] = sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return sharedScratch[0];
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
  threadgroup float reduceScratch[8];
  const uint tileSize = actSize * actSize;

  for (uint i = linear_tid; i < tileSize; i += group_size) {
    uint r = i / actSize;
    uint c = i % actSize;
    tile[r][c] = get_ref<upper>(A + batch_offset, row0 + r, col0 + c, N);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma unroll 4
  for (uint kk = 0; kk < actSize; kk++) {
    float diagElt = 0.0f;
    if (kk > 0) {
      float4 partialSum4 = float4(0.0f);
      uint i = linear_tid * 4;
      // vectorized reduce
      for (; i + 4 <= kk; i += group_size * 4) {
        float4 val4;
        val4.x = (i < kk) ? tile[kk][i] : 0.0f;
        val4.y = (i + 1 < kk) ? tile[kk][i + 1] : 0.0f;
        val4.z = (i + 2 < kk) ? tile[kk][i + 2] : 0.0f;
        val4.w = (i + 3 < kk) ? tile[kk][i + 3] : 0.0f;

        partialSum4 = fma(val4, val4, partialSum4);
      }

      float partialSum =
          partialSum4.x + partialSum4.y + partialSum4.z + partialSum4.w;

      // remaining elements
      for (i = linear_tid + (kk / 4) * 4; i < kk; i += group_size) {
        float val = tile[kk][i];
        partialSum = fma(val, val, partialSum);
      }
      diagElt = blockReduceSum(reduceScratch, partialSum, linear_tid);
    }

    if (linear_tid == 0) {
      float diagVal = tile[kk][kk] - diagElt;
      if (!(diagVal > 0.0f)) {
        info[bid.x] = kk + 1;
        return;
      }
      tile[kk][kk] = sqrt(diagVal);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float pivot = tile[kk][kk];

    for (uint j = kk + 1 + linear_tid; j < actSize; j += group_size) {
      float4 partialSum4 = float4(0.0f);
      uint i = 0;

      // 4 elements at a time
      for (; i + 4 <= kk; i += 4) {
        float4 row4 =
            float4(tile[j][i], tile[j][i + 1], tile[j][i + 2], tile[j][i + 3]);
        float4 diag4 = float4(
            tile[kk][i], tile[kk][i + 1], tile[kk][i + 2], tile[kk][i + 3]);
        partialSum4 = fma(row4, diag4, partialSum4);
      }
      float partialSum =
          partialSum4.x + partialSum4.y + partialSum4.z + partialSum4.w;
      // remaining elements
      for (; i < kk; i++) {
        partialSum = fma(tile[j][i], tile[kk][i], partialSum);
      }
      float val = tile[j][kk];
      val -= partialSum;
      val /= pivot;
      tile[j][kk] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
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
