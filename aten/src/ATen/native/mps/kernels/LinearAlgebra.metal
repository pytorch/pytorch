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
};

struct TRSMParams {
  uint N;
  uint NB;
  uint k;
  uint j;
  uint activeNB_k;
  uint activeNB_j;
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
};

kernel void factorDiagonalBlock(
    device float* A [[buffer(0)]],
    constant BlockParams& sizes [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]) {
  const uint N = sizes.N;
  const uint NB = sizes.NB;
  const uint k = sizes.k;
  const uint actSize = sizes.activeNB; // real block size for this diagonal

  uint row0 = k * NB;
  uint col0 = k * NB;

  threadgroup float tile[32 * 32];
  for (uint i = tid; i < 32 * 32; i += tpg) {
    tile[i] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // 1) Load tile from global A into tile
  uint tileSize = actSize * actSize;
  for (uint i = tid; i < tileSize; i += tpg) {
    uint r = i / actSize;
    uint c = i % actSize;
    tile[i] = A[(row0 + r) * N + (col0 + c)];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2) unblocked Cholesky on tile
  for (uint kk = 0; kk < actSize; kk++) {
    if (tid == 0) {
      float sumVal = tile[kk * actSize + kk];
      for (uint i = 0; i < kk; i++) {
        float val = tile[kk * actSize + i];
        sumVal -= val * val;
      }
      tile[kk * actSize + kk] = sqrt(sumVal);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // compute L[j,kk] for j=kk+1..actSize-1
    float diagVal = tile[kk * actSize + kk];
    for (uint j = kk + 1 + tid; j < actSize; j += tpg) {
      float sumVal = tile[j * actSize + kk];
      for (uint i = 0; i < kk; i++) {
        sumVal -= tile[j * actSize + i] * tile[kk * actSize + i];
      }
      tile[j * actSize + kk] = sumVal / diagVal;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // 3) store tile back into global A
  for (uint i = tid; i < tileSize; i += tpg) {
    uint r = i / actSize;
    uint c = i % actSize;
    A[(row0 + r) * N + (col0 + c)] = tile[i];
  }
}

//-----------------------------------------------
// TRSM for blocks below the diagonal(part of cholesky decomposition)
//-----------------------------------------------
kernel void applyTRSM(
    device float* A [[buffer(0)]],
    constant TRSMParams& sizes [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]) {
  const uint N = sizes.N;
  const uint NB = sizes.NB;
  const uint k = sizes.k;
  const uint j = sizes.j;
  const uint actSize_k = sizes.activeNB_k; // actual block size for (k,k)
  const uint actSize_j = sizes.activeNB_j; // actual block size for (j,k)

  // If j==k, the block is diagonal, so nothing to do in TRSM:
  if (j == k || actSize_k == 0 || actSize_j == 0) {
    return;
  }

  uint row0 = j * NB;
  uint col0 = k * NB;

  // shared memory for diagonal block (k,k) and target block (j,k)
  threadgroup float diag[32 * 32];
  threadgroup float target[32 * 32];
  for (uint i = tid; i < 32 * 32; i += tpg) {
    diag[i] = 0.0f;
    target[i] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 1) Load diag block A_{k,k} of size actSize_k x actSize_k
  for (uint i = tid; i < actSize_k * actSize_k; i += tpg) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    diag[i] = A[(k * NB + r) * N + (k * NB + c)];
  }
  // 2) Load target block A_{j,k} of size actSize_j x actSize_k
  for (uint i = tid; i < actSize_j * actSize_k; i += tpg) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    target[i] = A[(row0 + r) * N + (col0 + c)];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 3) Solve target = target * inv(diag)^T. We'll treat diag as an L matrix
  for (uint col = 0; col < actSize_k; col++) {
    float diag_val = diag[col * actSize_k + col];
    // Each thread does different rows in [0..actSize_j)
    for (uint row = tid; row < actSize_j; row += tpg) {
      float sumVal = target[row * actSize_k + col];
      for (uint p = 0; p < col; p++) {
        sumVal -= target[row * actSize_k + p] * diag[col * actSize_k + p];
      }
      target[row * actSize_k + col] = sumVal / diag_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // 4) Store updated block A_{j,k}
  for (uint i = tid; i < actSize_j * actSize_k; i += tpg) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    A[(row0 + r) * N + (col0 + c)] = target[i];
  }
}

//-----------------------------------------------
// SYRK/GEMM to update trailing blocks
//-----------------------------------------------
// For (j,h) with j>k, h>=j, we do A_{j,h} -= A_{j,k} * (A_{h,k})^T

kernel void applySYRK(
    device float* A [[buffer(0)]],
    constant UpdateParams& sizes [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]) {
  const uint N = sizes.N;
  const uint NB = sizes.NB;
  const uint k = sizes.k;
  const uint j = sizes.j;
  const uint h = sizes.h;
  const uint actSize_k = sizes.activeNB_k;
  const uint actSize_j = sizes.activeNB_j;
  const uint actSize_h = sizes.activeNB_h;
  // If there's no valid dimension, skip
  if (actSize_j == 0 || actSize_h == 0 || actSize_k == 0) {
    return;
  }

  uint row0 = j * NB;
  uint col0 = h * NB;

  // We'll load A_{j,k} (left), A_{h,k} (right), and A_{j,h} (tile)
  threadgroup float left[32 * 32]; // j x k
  threadgroup float right[32 * 32]; // h x k
  threadgroup float tile[32 * 32]; // j x h

  for (uint i = tid; i < 32 * 32; i += tpg) {
    left[i] = 0.0f;
    right[i] = 0.0f;
    tile[i] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 1) Load blocks:
  // left  = A_{j,k} of size actSize_j x actSize_k
  // right = A_{h,k} of size actSize_h x actSize_k
  // tile  = A_{j,h} of size actSize_j x actSize_h
  for (uint i = tid; i < actSize_j * actSize_k; i += tpg) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    left[i] = A[(j * NB + r) * N + (k * NB + c)];
  }
  for (uint i = tid; i < actSize_h * actSize_k; i += tpg) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    right[i] = A[(h * NB + r) * N + (k * NB + c)];
  }
  for (uint i = tid; i < actSize_j * actSize_h; i += tpg) {
    uint r = i / actSize_h;
    uint c = i % actSize_h;
    tile[i] = A[(row0 + r) * N + (col0 + c)];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 2) Update tile = tile - (left * right^T)
  //    left:  (actSize_j x actSize_k)
  //    right: (actSize_h x actSize_k), but we do right^T => (actSize_k x
  //    actSize_h) tile:  (actSize_j x actSize_h)
  for (uint idx = tid; idx < actSize_j * actSize_h; idx += tpg) {
    uint r = idx / actSize_h;
    uint c = idx % actSize_h;
    float sumVal = tile[idx];
    for (uint p = 0; p < actSize_k; p++) {
      sumVal -= left[r * actSize_k + p] * right[c * actSize_k + p];
    }
    tile[idx] = sumVal;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // 3) Store the updated tile A_{j,h}
  for (uint i = tid; i < actSize_j * actSize_h; i += tpg) {
    uint r = i / actSize_h;
    uint c = i % actSize_h;
    A[(row0 + r) * N + (col0 + c)] = tile[i];
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
