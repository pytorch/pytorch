#include <metal_array>
#include <metal_stdlib>

using namespace metal;
template <typename T>
T dot_product(constant T* v1, constant T* v2, ulong2 strides, uint32_t size) {
  T rc = T(0.0);
  for (uint32_t i = 0; i < size; ++i) {
    rc += v1[i * strides.x] * v2[i * strides.y];
  }
  return rc;
}

template <typename T>
kernel void naive_matmul(
    constant T* mat1Data [[buffer(0)]],
    constant T* mat2Data [[buffer(1)]],
    device T* outputData [[buffer(2)]],
    constant array<ulong2, 3>& strides [[buffer(3)]],
    constant uint3& sizes [[buffer(4)]],
    uint thread_index [[thread_position_in_grid]]) {
  uint y = thread_index / sizes.x;
  uint x = thread_index % sizes.x;
  if (x >= sizes.x || y >= sizes.z) {
    return;
  }
  auto rc = dot_product(
      mat1Data + x * strides[0].x,
      mat2Data + y * strides[1].y,
      ulong2(strides[0].y, strides[1].x),
      sizes.y);
  outputData[x * strides[2].x + y * strides[2].y] = rc;
}

// ------------------------------------
// Helper structs to pack N, NB and similar vars
// ------------------------------------
struct BlockParams {
  uint N; // Full matrix size
  uint NB; // block size
  uint k; // block index
  uint activeNB; // block size for partial blocks
  uint batch_size; // total number of batches
  uint batch_stride; // stride between matrices in batch
};

struct TRSMParams {
  uint N;
  uint NB;
  uint k;
  uint j;
  uint activeNB_k;
  uint activeNB_j;
  uint batch_size;
  uint batch_stride;
};

struct UpdateParams {
  uint N;
  uint NB;
  uint k;
  uint j;
  uint h;
  uint activeNB_k;
  uint activeNB_j;
  uint activeNB_h;
  uint batch_size;
  uint batch_stride;
};

inline float blockReduceSum(
    threadgroup float* sharedScratch,
    float val,
    uint tid,
    uint tpg) {
  // Store this thread's partial sum in shared memory.
  sharedScratch[tid] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduce in powers of 2.
  for (uint offset = tpg >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      sharedScratch[tid] += sharedScratch[tid + offset];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // The reduced sum is now in sharedScratch[0].
  return sharedScratch[0];
}

kernel void factorDiagonalBlock(
    device float* A [[buffer(0)]],
    constant BlockParams& sizes [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint tpg [[threads_per_threadgroup]]) {
  if (bid >= sizes.batch_size)
    return;

  const uint N = sizes.N;
  const uint NB = sizes.NB;
  const uint k = sizes.k;
  const uint actSize = sizes.activeNB;
  const uint batch_offset = bid * sizes.batch_stride;

  const uint row0 = k * NB;
  const uint col0 = k * NB;

  threadgroup float tile[32][33];
  threadgroup float reduceScratch[1024];
  const uint tileSize = actSize * actSize;
  //-----------------------------------
  // Load block from global to shared
  //-----------------------------------
  for (uint i = tid; i < tileSize; i += tpg) {
    uint r = i / actSize;
    uint c = i % actSize;
    tile[r][c] = A[batch_offset + (row0 + r) * N + (col0 + c)];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  //------------------------------------------------------
  // Factorization: for kk in [0..actSize-1]
  // We do:
  //  1) tile[kk][kk] = √(tile[kk][kk] - Σ tile[kk][i]^2)
  //  2) tile[j][kk]  = (tile[j][kk] - Σ tile[j][i]*tile[kk][i]) / tile[kk][kk]
  //------------------------------------------------------
  for (uint kk = 0; kk < actSize; kk++) {
    //
    // 1) Diagonal update in parallel
    //
    //    diagVal = tile[kk][kk] - ∑(tile[kk][i]^2, i=0..kk-1)
    //    tile[kk][kk] = sqrt(diagVal)
    //
    float diagElt = 0.0f;
    if (kk > 0) {
      // Each thread accumulates partial sums for tile[kk][i]^2
      // over i in [0..kk-1], stepping by tpg
      float partialSum = 0.0f;
      for (uint i = tid; i < kk; i += tpg) {
        float val = tile[kk][i];
        partialSum += val * val;
      }
      // Do a parallel sum across the thread group.
      diagElt = blockReduceSum(reduceScratch, partialSum, tid, tpg);
    }

    if (tid == 0) {
      float diagVal = tile[kk][kk] - diagElt;
      tile[kk][kk] = sqrt(diagVal);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float pivot = tile[kk][kk];
    //
    // 2) Off-diagonal update, for j in [kk+1..actSize-1]
    //
    //    tile[j][kk] = ( tile[j][kk] - ∑ tile[j][i]*tile[kk][i], i=0..kk-1 ) /
    //    pivot
    //
    // We can process each row j in parallel. Each thread picks up
    // multiple rows j. For each row, we do a parallel reduction of
    // the sum over i in [0..kk-1].
    for (uint j = kk + 1 + tid; j < actSize; j += tpg) {
      // 2a) partial sum of tile[j][i]*tile[kk][i]
      //     for i in [0..kk-1].
      float partialSum = 0.0f;
      for (uint i = 0; i < kk; i++) {
        partialSum += tile[j][i] * tile[kk][i];
      }

      // 2b) tile[j][kk] = (tile[j][kk] - partialSum) / pivot
      float val = tile[j][kk];
      val -= partialSum;
      val /= pivot;
      tile[j][kk] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // from shared to global
  for (uint i = tid; i < tileSize; i += tpg) {
    uint r = i / actSize;
    uint c = i % actSize;
    A[batch_offset + (row0 + r) * N + (col0 + c)] = tile[r][c];
  }
}

kernel void applyTRSM(
    device float* A [[buffer(0)]],
    constant TRSMParams& sizes [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint tpg [[threads_per_threadgroup]]) {
  if (bid >= sizes.batch_size)
    return;

  const uint N = sizes.N;
  const uint NB = sizes.NB;
  const uint k = sizes.k;
  const uint j = sizes.j;
  const uint actSize_k = sizes.activeNB_k;
  const uint actSize_j = sizes.activeNB_j;
  const uint batch_offset = bid * sizes.batch_stride;

  if (j == k || actSize_k == 0 || actSize_j == 0) {
    return;
  }

  uint row0 = j * NB;
  uint col0 = k * NB;

  threadgroup float diag[32 * 32];
  threadgroup float target[32 * 32];

  for (uint i = tid; i < actSize_k * actSize_k; i += tpg) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    diag[i] = A[batch_offset + (k * NB + r) * N + (k * NB + c)];
  }
  for (uint i = tid; i < actSize_j * actSize_k; i += tpg) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    target[i] = A[batch_offset + (row0 + r) * N + (col0 + c)];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint col = 0; col < actSize_k; col++) {
    float diag_val = diag[col * actSize_k + col];
    for (uint row = tid; row < actSize_j; row += tpg) {
      float sumVal = target[row * actSize_k + col];
      for (uint p = 0; p < col; p++) {
        sumVal -= target[row * actSize_k + p] * diag[col * actSize_k + p];
      }
      target[row * actSize_k + col] = sumVal / diag_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint i = tid; i < actSize_j * actSize_k; i += tpg) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    A[batch_offset + (row0 + r) * N + (col0 + c)] = target[i];
  }
}

kernel void applySYRK(
    device float* A [[buffer(0)]],
    constant UpdateParams& sizes [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint tpg [[threads_per_threadgroup]]) {
  if (bid >= sizes.batch_size)
    return;

  const uint N = sizes.N;
  const uint NB = sizes.NB;
  const uint k = sizes.k;
  const uint j = sizes.j;
  const uint h = sizes.h;
  const uint actSize_k = sizes.activeNB_k;
  const uint actSize_j = sizes.activeNB_j;
  const uint actSize_h = sizes.activeNB_h;
  const uint batch_offset = bid * sizes.batch_stride;

  if (actSize_j == 0 || actSize_h == 0 || actSize_k == 0) {
    return;
  }

  uint row0 = j * NB;
  uint col0 = h * NB;

  threadgroup float left[32 * 32];
  threadgroup float right[32 * 32];
  threadgroup float tile[32 * 32];

  for (uint i = tid; i < actSize_j * actSize_k; i += tpg) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    left[i] = A[batch_offset + (j * NB + r) * N + (k * NB + c)];
  }
  for (uint i = tid; i < actSize_h * actSize_k; i += tpg) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    right[i] = A[batch_offset + (h * NB + r) * N + (k * NB + c)];
  }
  for (uint i = tid; i < actSize_j * actSize_h; i += tpg) {
    uint r = i / actSize_h;
    uint c = i % actSize_h;
    tile[i] = A[batch_offset + (row0 + r) * N + (col0 + c)];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint idx = tid; idx < actSize_j * actSize_h; idx += tpg) {
    uint r = idx / actSize_h;
    uint c = idx % actSize_h;

    // If j == h, only process the lower-triangular portion
    // i.e., only apply the update if r >= c
    if (j == h && r < c) {
      continue;
    }
    float sumVal = tile[idx];
    for (uint p = 0; p < actSize_k; p++) {
      sumVal -= left[r * actSize_k + p] * right[c * actSize_k + p];
    }
    tile[idx] = sumVal;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid; i < actSize_j * actSize_h; i += tpg) {
    uint r = i / actSize_h;
    uint c = i % actSize_h;
    A[batch_offset + (row0 + r) * N + (col0 + c)] = tile[i];
  }
}

#define INSTANTIATE_NAIVE_MM(DTYPE)                          \
  template [[host_name("naive_matmul_" #DTYPE)]] kernel void \
  naive_matmul<DTYPE>(                                       \
      constant DTYPE * mat1Data [[buffer(0)]],               \
      constant DTYPE * mat2Data [[buffer(1)]],               \
      device DTYPE * outputData [[buffer(2)]],               \
      constant array<ulong2, 3> & strides [[buffer(3)]],     \
      constant uint3 & sizes [[buffer(4)]],                  \
      uint thread_index [[thread_position_in_grid]])

INSTANTIATE_NAIVE_MM(float);
INSTANTIATE_NAIVE_MM(half);
#if __METAL_VERSION__ >= 310
INSTANTIATE_NAIVE_MM(bfloat);
#endif
