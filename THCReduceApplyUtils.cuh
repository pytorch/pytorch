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

// Block-wide reduction in shared memory helper; only threadIdx.x == 0 will
// return the reduced value
template <typename T, typename ReduceOp>
__device__ T reduceBlock(T* smem,
                         int numVals,
                         T threadVal,
                         ReduceOp reduceOp,
                         T init) {
  if (numVals == 0) {
    return init;
  }

  if (threadIdx.x < numVals) {
    smem[threadIdx.x] = threadVal;
  }

  // First warp will perform reductions across warps
  __syncthreads();
  if ((threadIdx.x / warpSize) == 0) {
    T r = threadIdx.x < numVals ? smem[threadIdx.x] : init;

    for (int i = warpSize + threadIdx.x; i < numVals; i += warpSize) {
      r = reduceOp(r, smem[i]);
    }

    smem[threadIdx.x] = r;
  }

  // First thread will perform reductions across the block
  __syncthreads();

  T r = init;
  if (threadIdx.x == 0) {
    r = smem[0];

    int numLanesParticipating = min(numVals, warpSize);

    if (numLanesParticipating == 32) {
      // Unroll for warpSize == 32 and numVals >= 32
#pragma unroll
      for (int i = 1; i < 32; ++i) {
        r = reduceOp(r, smem[i]);
      }
    } else {
      for (int i = 1; i < numLanesParticipating; ++i) {
        r = reduceOp(r, smem[i]);
      }
    }
  }

  return r;
}

// Make sure the given tensor doesn't have too many dimensions
void THCCheckTensorDims(THCState* state, THCudaTensor* tensor, int arg);

// Produces a grid with at least one point per tile
THC_API bool THC_getGridFromTiles(long gridTiles, dim3& grid);

#endif // THC_REDUCE_APPLY_UTILS_INC
