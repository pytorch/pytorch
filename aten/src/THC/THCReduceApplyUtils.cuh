#ifndef THC_REDUCE_APPLY_UTILS_INC
#define THC_REDUCE_APPLY_UTILS_INC

#include <cuda.h>
#include <assert.h>
#include "THCGeneral.h"
#include "THCTensor.h"
#include "THCDeviceUtils.cuh"
#include "THCTensorInfo.cuh"

// Enum that indicates whether tensor arguments are read/write or
// read-only
enum TensorArgType { ReadWrite, ReadOnly };

template <typename IndexType>
__device__ __forceinline__ IndexType getLinearBlockId() {
  return blockIdx.z * gridDim.y * gridDim.x +
    blockIdx.y * gridDim.x +
    blockIdx.x;
}

// Reduce N values concurrently, i.e. suppose N = 2, and there are 4 threads:
// (1, 2), (3, 4), (5, 6), (7, 8), then the return in threadVals for thread 0
// is (1 + 3 + 5 + 7, 2 + 4 + 6 + 8) = (16, 20)
template <typename T, typename ReduceOp, int N>
__device__ void reduceNValuesInBlock(T *smem,
                             T threadVals[N],
                             int numVals,
                             ReduceOp reduceOp,
                             T init) {
  if (numVals == 0) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      threadVals[i] = init;
    }
    return;
  }

  // We store each of the N values contiguously, so if N = 2, all values for
  // the first threadVal for each thread in the block are stored followed by
  // all of the values for the second threadVal for each thread in the block
  if (threadIdx.x < numVals) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      smem[i * numVals + threadIdx.x] = threadVals[i];
    }
  }
  __syncthreads();

  // Number of lanes in the final reduction --> this is used to determine
  // where to put the outputs of each of the n things we are reducing. If
  // nLP = 32, then we have the 32 outputs for the first threadVal,
  // followed by the 32 outputs for the second threadVal, etc.
  int numLanesParticipating = min(numVals, warpSize);

  if (numVals > warpSize && ((threadIdx.x / warpSize) == 0 )) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      threadVals[i] = threadIdx.x < numVals ? threadVals[i] : init;
    }

    for (int i = warpSize + threadIdx.x; i < numVals; i += warpSize) {
#pragma unroll
      for (int j = 0; j < N; ++j) {
        threadVals[j] = reduceOp(threadVals[j], smem[j * numVals + i]);
      }
    }

#pragma unroll
    for (int i = 0; i < N; ++i) {
      smem[i * numLanesParticipating + threadIdx.x] = threadVals[i];
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    if (numLanesParticipating == 32) {
#pragma unroll
      for (int i = 0; i < N; ++i) {
#pragma unroll
        for (int j = 1; j < 32; ++j) {
          threadVals[i] = reduceOp(threadVals[i], smem[i * 32 + j]);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        for (int j = 1; j < numLanesParticipating; ++j) {
          threadVals[i] = reduceOp(threadVals[i], smem[i * numVals + j]);
        }
      }
    }
  }
}

// Block-wide reduction in shared memory helper; only threadIdx.x == 0 will
// return the reduced value
template <typename T, typename ReduceOp>
__device__ T reduceBlock(T* smem,
                         int numVals,
                         T threadVal,
                         ReduceOp reduceOp,
                         T init) {
  reduceNValuesInBlock<T, ReduceOp, 1>(smem, &threadVal, numVals, reduceOp, init);
  return threadVal;
}


// Block-wide reduction where each thread locally reduces N
// values before letting a single warp take over - assumes
// threadVals is in registers, not shared memory
template <typename T, typename ReduceOp, int N>
__device__ T reduceBlockWithNThreadLocalReductions(T *smem,
                         T threadVals[N],
                         int numVals,
                         ReduceOp reduceOp,
                         T init) {
  int offset = threadIdx.x * N;
  T local = offset < numVals ? threadVals[0] : init;

#pragma unroll
  for (int i = 1; i < N; ++i) {
    ++offset;
    T next = offset < numVals ? threadVals[i] : init;
    local = reduceOp(local, next);
  }

  return reduceBlock<T, ReduceOp>(smem, blockDim.x < numVals ? blockDim.x : numVals, local, reduceOp, init);
}

// Make sure the given tensor doesn't have too many dimensions
void THCCheckTensorDims(THCState* state, THCudaTensor* tensor, int arg);

// Produces a grid with at least one point per tile
THC_API bool THC_getGridFromTiles(ptrdiff_t gridTiles, dim3& grid);

#endif // THC_REDUCE_APPLY_UTILS_INC
