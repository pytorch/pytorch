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
          row[r][c + 0] = v.x;
          row[r][c + 1] = v.y;
          row[r][c + 2] = v.z;
          row[r][c + 3] = v.w;
        }
      } else {
#pragma unroll
        for (short c = 0; c < W; c++) {
          row[r][c] = (uint(c) < nb) ? src[c] : 0.0f;
        }
      }
    }
  }

  const uint nwarps = G / 32;
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

constant constexpr uint kLUStreamNT = 256;

kernel void luStreamUpdate(
    device float* A [[buffer(0)]],
    constant uint2& dims [[buffer(3)]],
    constant uint4& params [[buffer(4)]], // d0, j, RPT, searchOnly
    device float* scratch [[buffer(6)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
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
  const uint base = rowStart + (tgid.x * 8 + warp_id) * RPT;

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
  threadgroup float wv[8];
  threadgroup uint wi[8];
  const float mv = simd_max(bv);
  const uint mi = simd_min((bv == mv) ? bi : 0xffffffffu);
  if (lane == 0) {
    wv[warp_id] = mv;
    wi[warp_id] = mi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (warp_id == 0) {
    const float v2 = (lane < 8) ? wv[lane] : -1.0f;
    const uint i2 = (lane < 8) ? wi[lane] : 0xffffffffu;
    const float m2 = simd_max(v2);
    const uint p2 = simd_min((v2 == m2) ? i2 : 0xffffffffu);
    if (lane == 0) {
      scr[tgid.x] = m2;
      ((device uint*)(scr + kLUStreamNT))[tgid.x] = p2;
    }
  }
}

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
  const uint tid = tid3.x; // G = 256
  device float* Ab = A + ulong(tgid.x) * M * N;
  device int* pv = pivots + ulong(tgid.x) * minMN;
  device float* scr = scratch + ulong(tgid.x) * (2 * kLUStreamNT + 32);
  device const uint* sidx = (device const uint*)(scr + kLUStreamNT);
  device float* uRow = scr + 2 * kLUStreamNT;

  if (d0 == 0 && j == 0 && tid == 0) {
    info[tgid.x] = 0;
  }
  threadgroup float wv[8];
  threadgroup uint wi[8];
  threadgroup uint sPiv[1];

  // first-max semantics: equal partials resolve to the smaller global row
  float bv = -1.0f;
  uint bi = 0xffffffffu;
  for (uint i = tid; i < npart; i += 256) {
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
    const float v2 = (lane < 8) ? wv[lane] : -1.0f;
    const uint i2 = (lane < 8) ? wi[lane] : 0xffffffffu;
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

  const bool inside = (ro + BM <= Tm) && (co + BN <= Tn);
  if (inside) {
    auto mA = tA.template slice<dynamic_extent, BM>(0, ro);
    auto mB = tB.template slice<BN, dynamic_extent>(co, 0);
    auto mC = tC.template slice<BN, BM>(co, ro);
    auto cT = op.template get_destination_cooperative_tensor<
        decltype(mA),
        decltype(mB),
        float>();
    op.run(mA, mB, cT);
    uint16_t e = 0;
    for (auto it = cT.begin(); it != cT.end(); ++it, ++e) {
      auto idx = it.get_multidimensional_index();
      const int r = ro + int(idx[1]);
      const int c = co + int(idx[0]);
      cT[e] = cP[r * gN + c] - cT[e];
    }
    cT.store(mC);
  } else {
    auto mA = tA.slice(0, ro);
    auto mB = tB.slice(co, 0);
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
      const int r = ro + int(idx[1]);
      const int c = co + int(idx[0]);
      cT[e] = cP[r * gN + c] - cT[e];
    }
    cT.store(mC);
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
    const auto norm = ::metal::precise::sqrt(
        c10::metal::threadgroup_sum(scratch, norm_sq, tid, group_size));

    // scale norm_eps by matrix dimension to handle accumulated error
    const auto norm_eps = ::metal::numeric_limits<opmath_t>::epsilon() * m;
    const auto tau_eps = ::metal::numeric_limits<opmath_t>::epsilon();

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
