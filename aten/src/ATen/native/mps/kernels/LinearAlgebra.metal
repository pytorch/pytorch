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
    threadgroup T A_tile[TILE_DIM][TILE_DIM],
    threadgroup T B_tile[TILE_DIM][TILE_DIM],
    uint2 tid,
    uint2 thread_id) {
  c10::metal::opmath_t<T> sum = 0;

  uint numTiles = (sizes.y + TILE_DIM - 1) / TILE_DIM;
  for (uint t = 0; t < numTiles; t++) {
    uint tiledCol = t * TILE_DIM + tid.x;
    if (thread_id.y < sizes.x && tiledCol < sizes.y) {
      A_tile[tid.y][tid.x] =
          mat1Data[thread_id.y * strides[0].x + tiledCol * strides[0].y];
    } else {
      A_tile[tid.y][tid.x] = 0;
    }

    uint tiledRow = t * TILE_DIM + tid.y;
    if (tiledRow < sizes.y && thread_id.x < sizes.z) {
      B_tile[tid.y][tid.x] =
          mat2Data[tiledRow * strides[1].x + thread_id.x * strides[1].y];
    } else {
      B_tile[tid.y][tid.x] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < TILE_DIM; k++) {
      sum += A_tile[tid.y][k] * B_tile[k][tid.x];
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
  threadgroup T A_tile[TILE_DIM][TILE_DIM];
  threadgroup T B_tile[TILE_DIM][TILE_DIM];

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
  threadgroup T A_tile[TILE_DIM][TILE_DIM];
  threadgroup T B_tile[TILE_DIM][TILE_DIM];

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
    auto bias =
        biasData[thread_id.y * strides[3].x + thread_id.x * strides[3].y];
    outputData[thread_id.y * strides[2].x + thread_id.x * strides[2].y] =
        static_cast<T>(alpha_beta[0] * sum + alpha_beta[1] * bias);
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

  c10::metal::opmath_t<T> sum = 0;

  threadgroup T A_tile[TILE_DIM][TILE_DIM];
  threadgroup T B_tile[TILE_DIM][TILE_DIM];

  // batch offsets for both matrices
  uint batch1Offset = batch * strides[2];
  uint batch2Offset = batch * strides[5];

  uint numTiles = (sizes.y + TILE_DIM - 1) / TILE_DIM;
  for (uint t = 0; t < numTiles; t++) {
    uint tiledCol = t * TILE_DIM + tid.x;
    if (row < sizes.x && tiledCol < sizes.y) {
      A_tile[tid.y][tid.x] =
          mat1Data[batch1Offset + row * strides[1] + tiledCol * strides[0]];
    } else {
      A_tile[tid.y][tid.x] = 0;
    }

    uint tiledRow = t * TILE_DIM + tid.y;
    if (tiledRow < sizes.y && col < sizes.z) {
      B_tile[tid.y][tid.x] =
          mat2Data[batch2Offset + tiledRow * strides[4] + col * strides[3]];
    } else {
      B_tile[tid.y][tid.x] = 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < TILE_DIM; k++) {
      sum += A_tile[tid.y][k] * B_tile[k][tid.x];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (row < sizes.x && col < sizes.z) {
    outputData[batch * strides[8] + col * strides[6] + row * strides[7]] =
        static_cast<T>(sum);
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
      if (diagVal <= 0.0f) {
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
    uint sgitg [[simdgroup_index_in_threadgroup]]) {
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
    uint warp_id = sgitg;

    simdgroup_matrix<float, 8, 8> negative_identity =
        simdgroup_matrix<float, 8, 8>(-1.0);
    simdgroup_matrix<float, 8, 8> identity = simdgroup_matrix<float, 8, 8>(1.0);
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
        simdgroup_multiply(Prod, Prod, negative_identity);
        simdgroup_multiply_accumulate(Cfrag, Cfrag, identity, Prod);
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
      uint2 group_id [[threadgroup_position_in_grid]])

INSTANTIATE_MM_OPS(float);
INSTANTIATE_MM_OPS(half);
INSTANTIATE_MM_OPS(bfloat);

// Integral MM
INSTANTIATE_MM_OPS(long);
INSTANTIATE_MM_OPS(int);
INSTANTIATE_MM_OPS(short);
INSTANTIATE_MM_OPS(char);
INSTANTIATE_MM_OPS(uchar);
