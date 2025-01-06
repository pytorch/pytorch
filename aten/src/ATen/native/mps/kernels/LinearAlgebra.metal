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
    uint tgid [[threadgroup_position_in_grid]],
    uint tpg [[threads_per_threadgroup]]) {
  // decode which batch and which j block from tgid:
  // total threadgroups across all batches and all j's
  // tgid goes from [0 .. batch_size * nBlocksJ - 1]
  // so:
  //   b = tgid / nBlocksJ (which batch)
  //   idxJ = tgid % nBlocksJ (which j block within that step)
  uint b = tgid / sizes.nBlocksJ;
  uint idxJ = tgid % sizes.nBlocksJ;
  if (b >= sizes.batch_size) {
    return;
  }

  // compute the actual j block index
  uint j = sizes.startJ + idxJ;

  const uint N = sizes.N;
  const uint NB = sizes.NB;
  const uint k = sizes.k;
  const uint actSize_k = sizes.activeNB_k;
  const uint batch_offset = b * sizes.batch_stride;

  uint row0 = j * NB; // row offset
  uint col0 = k * NB; // column offset

  // how many rows in the j block, might be smaller if j is near the end
  uint actSize_j = (uint)min((int)(N - row0), (int)NB);
  if (actSize_k == 0 || actSize_j == 0) {
    return;
  }
  if (j == k) {
    return; // no-op if same block
  }

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
    constant SYRKBatchedParams& sizes [[buffer(1)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tpg [[threads_per_threadgroup]]) {
    
    // rarly exit check for batch size
    uint b = tgid / sizes.nPairs;
    if (b >= sizes.batch_size) return;
    uint pairID = tgid % sizes.nPairs;
    
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
    
    if (actSize_j == 0 || actSize_h == 0 || actSize_k == 0) return;
    
    threadgroup float left[32 * 32];
    threadgroup float right[32 * 32];
    threadgroup float tile[32 * 32];
    
    const uint threads = min(tpg, actSize_j * actSize_k);
    
    // Load left matrix
    for (uint i = tid; i < actSize_j * actSize_k; i += threads) {
        uint r = i / actSize_k;
        uint c = i % actSize_k;
        left[r * actSize_k + c] = A[batch_offset + (j * NB + r) * N + (sizes.k * NB + c)];
    }
    
    // Load right matrix
    for (uint i = tid; i < actSize_h * actSize_k; i += threads) {
        uint r = i / actSize_k;
        uint c = i % actSize_k;
        right[r * actSize_k + c] = A[batch_offset + (h * NB + r) * N + (sizes.k * NB + c)];
    }
    
    // Load tile
    for (uint i = tid; i < actSize_j * actSize_h; i += threads) {
        uint r = i / actSize_h;
        uint c = i % actSize_h;
        tile[r * actSize_h + c] = A[batch_offset + (row0 + r) * N + (col0 + c)];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Optimized SYRK computation with manual unrolling
    for (uint idx = tid; idx < actSize_j * actSize_h; idx += threads) {
        uint r = idx / actSize_h;
        uint c = idx % actSize_h;
        
        if ((j == h) && (r < c)) continue;
        
        uint tile_idx = r * actSize_h + c;
        float sum = tile[tile_idx];
        
        uint left_row = r * actSize_k;
        uint right_row = c * actSize_k;
        
        // Process 4 elements at a time
        uint k = 0;
        for (; k + 4 <= actSize_k; k += 4) {
            sum -= left[left_row + k] * right[right_row + k]
                 + left[left_row + k + 1] * right[right_row + k + 1]
                 + left[left_row + k + 2] * right[right_row + k + 2]
                 + left[left_row + k + 3] * right[right_row + k + 3];
        }
        
        // Handle remaining elements
        for (; k < actSize_k; k++) {
            sum -= left[left_row + k] * right[right_row + k];
        }
        
        tile[tile_idx] = sum;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
  
    for (uint i = tid; i < actSize_j * actSize_h; i += threads) {
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
