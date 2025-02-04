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

inline float blockReduceSum(
    threadgroup float* sharedScratch,
    float val,
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

kernel void factorDiagonalBlock(
    device float* A [[buffer(0)]],
    device int* success [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& NB [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint tpg [[threads_per_threadgroup]]) {
  const uint actSize = min(N - k * NB, NB); // uint64 before NB
  const uint batch_offset = bid * N * N;

  const uint row0 = k * NB;
  const uint col0 = k * NB;

  threadgroup float tile[32][33];
  threadgroup float reduceScratch[256];
  const uint tileSize = actSize * actSize;

  for (uint i = tid; i < tileSize; i += tpg) {
    uint r = i / actSize;
    uint c = i % actSize;
    tile[r][c] = A[batch_offset + (row0 + r) * N + (col0 + c)];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint kk = 0; kk < actSize; kk++) {
    float diagElt = 0.0f;
    if (kk > 0) {
      float partialSum = 0.0f;
      for (uint i = tid; i < kk; i += tpg) {
        float val = tile[kk][i];
        partialSum = fma(val, val, partialSum);
      }
      diagElt = blockReduceSum(reduceScratch, partialSum, tid, tpg);
    }

    if (tid == 0) {
      float diagVal = tile[kk][kk] - diagElt;
      // Check for positive definiteness
      if (diagVal <= 0.0f) {
        success[bid] = 0; // matrix is not positive definite
        return;
      }
      tile[kk][kk] = sqrt(diagVal);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float pivot = tile[kk][kk];

    for (uint j = kk + 1 + tid; j < actSize; j += tpg) {
      float partialSum = 0.0f;
      for (uint i = 0; i < kk; i++) {
        partialSum = fma(tile[j][i], tile[kk][i], partialSum);
      }

      float val = tile[j][kk];
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
}

kernel void applyTRSM(
    device float* A [[buffer(0)]],
    constant uint& N [[buffer(2)]],
    constant uint& NB [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]]) {
  uint b = tgid.x;
  uint idxJ = tgid.y;

  const uint actSize_k = uint(min(int64_t(N - k * NB), int64_t(NB)));
  const uint batch_offset = b * N * N;
  const uint j = (k + 1) + idxJ;

  uint row0 = j * NB;
  uint col0 = k * NB;

  uint actSize_j = (uint)min((int)(N - row0), (int)NB);
  if (actSize_k == 0 || actSize_j == 0) {
    return;
  }
  if (j == k) {
    return;
  }

  threadgroup float diag[32 * 32];
  threadgroup float target[32 * 32];

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
    float diag_val = diag[col * actSize_k + col];
    if (abs(diag_val) < 1e-6f) {
      diag_val = (diag_val < 0.0f) ? -1e-6f : 1e-6f;
    }

    for (uint row = tid.x; row < actSize_j; row += tpg.x) {
      float sum = target[row * actSize_k + col];

      // kahan sum
      float c = 0.0f;
      for (uint p = 0; p < col; p++) {
        float y = -target[row * actSize_k + p] * diag[col * actSize_k + p] - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
      }

      target[row * actSize_k + col] = sum / diag_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint i = tid.x; i < actSize_j * actSize_k; i += tpg.x) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    A[batch_offset + (row0 + r) * N + (col0 + c)] = target[i];
  }
}

kernel void applySYRK(
    device float* A [[buffer(0)]],
    constant uint& N [[buffer(2)]],
    constant uint& NB [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpg [[threads_per_threadgroup]]) {
  uint b = tgid.x;
  uint pairID = tgid.y;

  uint jRel = (-1 + sqrt(1 + 8 * float(pairID))) / 2;
  uint hRel = pairID - (jRel * (jRel + 1) >> 1);

  const uint startJ = (k + 1);
  uint j = startJ + jRel;
  uint h = startJ + hRel;
  uint row0 = j * NB;
  uint col0 = h * NB;

  const uint actSize_k = uint(min(int64_t(N - k * NB), int64_t(NB)));
  const uint actSize_j = min((uint)(N - row0), NB);
  const uint actSize_h = min((uint)(N - col0), NB);
  const uint batch_offset = b * N * N;

  if (actSize_j == 0 || actSize_h == 0 || actSize_k == 0)
    return;

  threadgroup float left[32 * 33];
  threadgroup float right_t[32 * 33];
  threadgroup float tile[32 * 33];

  const uint threads = min(tpg.x, actSize_j * actSize_k);

  for (uint i = tid.x; i < actSize_j * actSize_k; i += threads) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    left[r * actSize_k + c] = A[batch_offset + (j * NB + r) * N + (k * NB + c)];
  }

  for (uint i = tid.x; i < actSize_h * actSize_k; i += threads) {
    uint r = i / actSize_k;
    uint c = i % actSize_k;
    right_t[c * actSize_h + r] =
        A[batch_offset + (h * NB + r) * N + (k * NB + c)];
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
    float sum = tile[tile_idx];

    uint left_row = r * actSize_k;
    uint right_col = c;

    uint k = 0;
    float4 sum4 = {0.0f, 0.0f, 0.0f, 0.0f};

    for (; k + 4 <= actSize_k; k += 4) {
      float4 left4 = {
          left[left_row + k],
          left[left_row + k + 1],
          left[left_row + k + 2],
          left[left_row + k + 3]};

      float4 right4 = {
          right_t[(k + 0) * actSize_h + right_col],
          right_t[(k + 1) * actSize_h + right_col],
          right_t[(k + 2) * actSize_h + right_col],
          right_t[(k + 3) * actSize_h + right_col]};

      sum4 = fma(left4, right4, sum4);
    }

    sum -= dot(sum4, 1.0);

    for (; k < actSize_k; k++) {
      sum = fma(-left[left_row + k], right_t[k * actSize_h + right_col], sum);
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
