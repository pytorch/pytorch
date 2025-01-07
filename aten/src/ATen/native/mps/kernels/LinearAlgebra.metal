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
  uint k; // current diagonal block
  uint startJ; // first block to process, i.e. k+1
  uint nBlocksJ; // total number of j blocks in this step
  uint activeNB_k;
  uint batch_size;
  uint batch_stride;
};

struct SYRKBatchedParams {
  uint N;
  uint NB;
  uint k; // current diagonal block index
  uint startJ; // typically k+1
  uint nBlocksJ; // how many j-blocks total in [startJ..numBlocks)
  uint activeNB_k; // size of block 'k'
  uint batch_size;
  uint batch_stride;
  uint nPairs; // total # of (j, h) pairs
};

template <typename T>
inline T sqrt_cast(T x) {
#if __METAL_VERSION__ >= 310
  if (is_same_v<T, bfloat16>) {
    return T(sqrt(float(x)));
  }
#endif
  if (is_same_v<T, half>) {
    return T(sqrt(float(x)));
  }
  return T(sqrt(x));
}

template <typename T>
inline T blockReduceSum(
    threadgroup T* sharedScratch,
    T val,
    uint tid,
    uint tpg) {
  sharedScratch[tid] = val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint offset = tpg >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      sharedScratch[tid] += sharedScratch[tid + offset];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  return sharedScratch[0];
}

template <typename T>
kernel void factorDiagonalBlock(
    device T* A [[buffer(0)]],
    constant BlockParams& sizes [[buffer(1)]],
    device int* success [[buffer(2)]],
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

  threadgroup T tile[32][33];
  threadgroup T reduceScratch[1024];
  const uint tileSize = actSize * actSize;

  for (uint i = tid; i < tileSize; i += tpg) {
    uint r = i / actSize;
    uint c = i % actSize;
    tile[r][c] = A[batch_offset + (row0 + r) * N + (col0 + c)];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint kk = 0; kk < actSize; kk++) {
    T diagElt = T(0);
    if (kk > 0) {
      T partialSum = T(0);
      for (uint i = tid; i < kk; i += tpg) {
        T val = tile[kk][i];
        partialSum += val * val;
      }
      diagElt = blockReduceSum(reduceScratch, partialSum, tid, tpg);
    }

    if (tid == 0) {
      T diagVal = tile[kk][kk] - diagElt;
      // Check for positive definiteness
      if (diagVal <= T(0)) {
        success[bid] = 0; // matrix is not positive definite
        return;
      }
      tile[kk][kk] = sqrt_cast(diagVal);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    T pivot = tile[kk][kk];

    for (uint j = kk + 1 + tid; j < actSize; j += tpg) {
      T partialSum = T(0);
      for (uint i = 0; i < kk; i++) {
        partialSum += tile[j][i] * tile[kk][i];
      }

      T val = tile[j][kk];
      val -= partialSum;
      val /= pivot;
      tile[j][kk] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint i = tid; i < tileSize; i += tpg) {
    uint r = i / actSize;
    uint c = i % actSize;
    A[batch_offset + (row0 + r) * N + (col0 + c)] = tile[r][c];
  }

  if (tid == 0) {
    success[bid] &= 1; // Mark successful completion
  }
}

template <typename T>
kernel void applyTRSM(
    device T* A [[buffer(0)]],
    constant TRSMParams& sizes [[buffer(1)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]]) {
  uint b = tgid.x;
  uint idxJ = tgid.y;
  if (b >= sizes.batch_size) {
    return;
  }

  uint j = sizes.startJ + idxJ;

  const uint N = sizes.N;
  const uint NB = sizes.NB;
  const uint k = sizes.k;
  const uint actSize_k = sizes.activeNB_k;
  const uint batch_offset = b * sizes.batch_stride;

  uint row0 = j * NB;
  uint col0 = k * NB;

  uint actSize_j = (uint)min((int)(N - row0), (int)NB);
  if (actSize_k == 0 || actSize_j == 0) {
    return;
  }
  if (j == k) {
    return;
  }

  threadgroup T diag[32 * 32];
  threadgroup T target[32 * 32];

  for (uint i = tid.x; i < actSize_k * actSize_k; i += tpg.x) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    diag[i] = A[batch_offset + (k * NB + r) * N + (k * NB + c)];
  }
  for (uint i = tid.x; i < actSize_j * actSize_k; i += tpg.x) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    target[i] = A[batch_offset + (row0 + r) * N + (col0 + c)];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint col = 0; col < actSize_k; col++) {
    T diag_val = diag[col * actSize_k + col];
    for (uint row = tid.x; row < actSize_j; row += tpg.x) {
      T sumVal = target[row * actSize_k + col];
      for (uint p = 0; p < col; p++) {
        sumVal -= target[row * actSize_k + p] * diag[col * actSize_k + p];
      }
      target[row * actSize_k + col] = sumVal / diag_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint i = tid.x; i < actSize_j * actSize_k; i += tpg.x) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    A[batch_offset + (row0 + r) * N + (col0 + c)] = target[i];
  }
}

template <typename T>
kernel void applySYRK(
    device T* A [[buffer(0)]],
    constant SYRKBatchedParams& sizes [[buffer(1)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]]) {
  uint b = tgid.x;
  uint pairID = tgid.y;
  if (b >= sizes.batch_size)
    return;

  uint jRel = (-1 + sqrt(1 + 8 * float(pairID))) / 2;
  uint hRel = pairID - (jRel * (jRel + 1) >> 1);

  uint j = sizes.startJ + jRel;
  uint h = sizes.startJ + hRel;
  uint row0 = j * sizes.NB;
  uint col0 = h * sizes.NB;

  const uint N = sizes.N;
  const uint NB = sizes.NB;
  const uint actSize_k = sizes.activeNB_k;
  const uint actSize_j = min((uint)(N - row0), NB);
  const uint actSize_h = min((uint)(N - col0), NB);
  const uint batch_offset = b * sizes.batch_stride;

  if (actSize_j == 0 || actSize_h == 0 || actSize_k == 0)
    return;

  threadgroup T left[32 * 32];
  threadgroup T right_t[32 * 32];
  threadgroup T tile[32 * 32];

  const uint threads = min(tpg.x, actSize_j * actSize_k);

  for (uint i = tid.x; i < actSize_j * actSize_k; i += threads) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    left[r * actSize_k + c] =
        A[batch_offset + (j * NB + r) * N + (sizes.k * NB + c)];
  }

  for (uint i = tid.x; i < actSize_h * actSize_k; i += threads) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    right_t[c * actSize_h + r] =
        A[batch_offset + (h * NB + r) * N + (sizes.k * NB + c)];
  }

  for (uint i = tid.x; i < actSize_j * actSize_h; i += threads) {
    uint r = i / actSize_h;
    uint c = i % actSize_h;
    tile[r * actSize_h + c] = A[batch_offset + (row0 + r) * N + (col0 + c)];
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint idx = tid.x; idx < actSize_j * actSize_h; idx += threads) {
    uint r = idx / actSize_h;
    uint c = idx % actSize_h;

    if ((j == h) && (r < c))
      continue;

    uint tile_idx = r * actSize_h + c;
    T sum = tile[tile_idx];

    uint left_row = r * actSize_k;
    uint right_col = c;

    uint k = 0;
    T sum4[4] = {T(0), T(0), T(0), T(0)};

    for (; k + 4 <= actSize_k; k += 4) {
      T left4[4] = {
          left[left_row + k],
          left[left_row + k + 1],
          left[left_row + k + 2],
          left[left_row + k + 3]};

      T right4[4] = {
          right_t[(k + 0) * actSize_h + right_col],
          right_t[(k + 1) * actSize_h + right_col],
          right_t[(k + 2) * actSize_h + right_col],
          right_t[(k + 3) * actSize_h + right_col]};

      for (uint i = 0; i < 4; ++i) {
        sum4[i] += left4[i] * right4[i];
      }
    }

    sum -= sum4[0] + sum4[1] + sum4[2] + sum4[3];

    for (; k < actSize_k; k++) {
      sum -= left[left_row + k] * right_t[k * actSize_h + right_col];
    }

    tile[tile_idx] = sum;
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint i = tid.x; i < actSize_j * actSize_h; i += threads) {
    uint r = i / actSize_h;
    uint c = i % actSize_h;
    A[batch_offset + (row0 + r) * N + (col0 + c)] = tile[r * actSize_h + c];
  }
}

#define INSTANTIATE_FACTOR_DIAGONAL(DTYPE)                          \
  template [[host_name("factorDiagonalBlock_" #DTYPE)]] kernel void \
  factorDiagonalBlock<DTYPE>(                                       \
      device DTYPE * A [[buffer(0)]],                               \
      constant BlockParams & sizes [[buffer(1)]],                   \
      device int* success [[buffer(2)]],                            \
      uint tid [[thread_position_in_threadgroup]],                  \
      uint bid [[threadgroup_position_in_grid]],                    \
      uint tpg [[threads_per_threadgroup]])

#define INSTANTIATE_TRSM(DTYPE)                                             \
  template [[host_name("applyTRSM_" #DTYPE)]] kernel void applyTRSM<DTYPE>( \
      device DTYPE * A [[buffer(0)]],                                       \
      constant TRSMParams & sizes [[buffer(1)]],                            \
      uint3 tid [[thread_position_in_threadgroup]],                         \
      uint3 tgid [[threadgroup_position_in_grid]],                          \
      uint3 tpg [[threads_per_threadgroup]])

#define INSTANTIATE_SYRK(DTYPE)                                             \
  template [[host_name("applySYRK_" #DTYPE)]] kernel void applySYRK<DTYPE>( \
      device DTYPE * A [[buffer(0)]],                                       \
      constant SYRKBatchedParams & sizes [[buffer(1)]],                     \
      uint3 tid [[thread_position_in_threadgroup]],                         \
      uint3 tgid [[threadgroup_position_in_grid]],                          \
      uint3 tpg [[threads_per_threadgroup]])

INSTANTIATE_FACTOR_DIAGONAL(float);
INSTANTIATE_FACTOR_DIAGONAL(half);
INSTANTIATE_TRSM(float);
INSTANTIATE_TRSM(half);
INSTANTIATE_SYRK(float);
INSTANTIATE_SYRK(half);

#if __METAL_VERSION__ >= 310
INSTANTIATE_FACTOR_DIAGONAL(bfloat);
INSTANTIATE_TRSM(bfloat);
INSTANTIATE_SYRK(bfloat);
#endif

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
