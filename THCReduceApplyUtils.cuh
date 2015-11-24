#ifndef THC_REDUCE_APPLY_UTILS_INC
#define THC_REDUCE_APPLY_UTILS_INC

#include <cuda.h>
#include <assert.h>
#include "THGeneral.h"
#include "THCGeneral.h"
#include "THCTensor.h"
#include "THCDeviceUtils.cuh"

// Maximum number of dimensions allowed for cutorch
#define MAX_CUTORCH_DIMS 25

// Warning string for tensor arguments that are too large or have too
// many dimensions
#define CUTORCH_STR(X) #X
#define CUTORCH_DIM_WARNING "tensor too large or too many (>" \
  CUTORCH_STR(MAX_CUTORCH_DIMS) ") dimensions"

// Enum that indicates whether tensor arguments are read/write or
// read-only
enum TensorArgType { ReadWrite, ReadOnly };

// Copy operator for the pointwise apply kernel
template <typename T>
struct CopyOp {
  __device__ __forceinline__ void operator()(T* dst, T* src) {
#if __CUDA_ARCH__ >= 350
    *dst = __ldg(src);
#else
    *dst = *src;
#endif
  }
};

// CUDA kernel argument that defines tensor layout
template <typename IndexType>
struct TensorInfo {
  // Extracts size/stride information for the kernel.
  // The optional `reduceDim` indicates a reduction dimension for the
  // given tensor, so that the resulting size for this dimension will be 1.
  TensorInfo(THCState* state, THCudaTensor* t, int reduceDim = -1);

  // Collapses all runs of successive dimensions if the size/strides
  // match up within the run and there are no holes between the
  // dimensions.
  // If excludeDim is set (not -1), then excludeDim will not be
  // collapsed with any other dimension.
  // Function returns the new dimension index that excludeDim maps to,
  // since the collapsed dimensions are <= the input dimensions.
  int collapseDims(int excludeDim = -1);

  // Contiguous tensors of more than one dimension are collapsed down
  // to one tensor
  __host__ __device__ inline bool isContiguous() const {
    return (dims == 1 && strides[0] == 1);
  }

  float* data;
  IndexType sizes[MAX_CUTORCH_DIMS];
  IndexType strides[MAX_CUTORCH_DIMS];
  int dims;
};

template <typename IndexType>
TensorInfo<IndexType>::TensorInfo(THCState* state,
                                  THCudaTensor* t,
                                  int reduceDim) {
  data = THCudaTensor_data(state, t);
  dims = THCudaTensor_nDimension(state, t);
  assert(dims <= MAX_CUTORCH_DIMS);

  for (int i = 0; i < dims; ++i) {
    sizes[i] = THCudaTensor_size(state, t, i);
    strides[i] = THCudaTensor_stride(state, t, i);
  }

  assert(reduceDim == -1 || (reduceDim < dims && reduceDim >= 0));

  if (reduceDim != -1) {
    sizes[reduceDim] = 1;
  }
}

template <typename IndexType>
int
TensorInfo<IndexType>::collapseDims(int excludeDim) {
  // Find the innermost dimension not of size 1, since dimensions of size 1 are
  // collapsible.
  int firstNonOneDim = -1;

  for (int i = dims - 1; i >= 0; --i) {
    if (i == excludeDim) {
      // We cannot collapse this dimension, even if it is size 1
      firstNonOneDim = i;
      break;
    }

    if (sizes[i] != 1) {
      firstNonOneDim = i;
      break;
    }
  }

  // Special case: if all dimensions are of size 1, then this is a
  // single-point tensor that we still have to operate on. Reduce to a
  // single point.
  if (firstNonOneDim == -1) {
    assert(excludeDim == -1);

    dims = 1;
    sizes[0] = 1;
    strides[0] = 1;

    // Everything effectively got collapsed into this dimension
    return 0;
  }

  // Count the number of successive dimensions that can be collapsed, from
  // innermost to outermost.
  int numCollapsed = 0;

  // Skip the leading size 1 dims
  numCollapsed += dims - 1 - firstNonOneDim;

  // We perform one pass through to determine how many dimensions we
  // can collapse, before calculating the actual size of the collapsed
  // dimensions.
  // size/strideInner are the size/strides of the previous inner
  // non-collapsible dim we encounter.
  long sizeInner = sizes[firstNonOneDim];
  long strideInner = strides[firstNonOneDim];

  for (int i = firstNonOneDim - 1; i >= 0; --i) {
    long sizeOuter = sizes[i];
    long strideOuter = strides[i];

    // Don't collapse this dimension if we want to exclude it from
    // collapsing.
    // Since this code is attempting to collapse a subsequent
    // dimension (i) with the preceding dimension (i + 1), we can only
    // perform collapsing if the preceding dimension can be collapsed
    // (i.e., not excludeDim)
    if ((excludeDim != i) && (excludeDim != i + 1)) {
      // The next outermost dimension can be skipped if size 1
      if (sizeOuter == 1) {
        ++numCollapsed;
        continue;
      }

      // If the next outermost dimension is contiguous with the
      // previous non-collapsed one, collapse it
      if (strideOuter == strideInner * sizeInner) {
        ++numCollapsed;

        // This is the run of collapsed dimensions' size
        sizeInner = sizeInner * sizeOuter;
        continue;
      }
    }

    // Otherwise, this new outer dimension at `i` cannot be collapsed
    // because it is excluded from collapsing, or it is not contiguous
    // with the previous inner dimension.
    sizeInner = sizeOuter;
    strideInner = strideOuter;
  }

  // This will be our new size/stride and dimension.
  IndexType newSizes[MAX_CUTORCH_DIMS];
  IndexType newStrides[MAX_CUTORCH_DIMS];

  assert(numCollapsed < dims);
  int newDims = dims - numCollapsed;

  // We return the index of the excluded dimension that is excluded
  // from being collapsed here.
  int returnDim = -1;

  // We perform a second pass through the dimensions to actually
  // calculate the size of the collapsed dimensions.
  int collapsedIndex = dims - numCollapsed - 1;
  newSizes[collapsedIndex] = sizes[firstNonOneDim];
  newStrides[collapsedIndex] = strides[firstNonOneDim];

  if (firstNonOneDim == excludeDim) {
    returnDim = collapsedIndex;
  }

  for (int i = firstNonOneDim - 1; i >= 0; --i) {
    long sizeOuter = sizes[i];
    long strideOuter = strides[i];

    if ((excludeDim != i) && (excludeDim != i + 1)) {
      if (sizeOuter == 1) {
        // skip
        continue;
      }

      if (strideOuter == newSizes[collapsedIndex] * newStrides[collapsedIndex]) {
        // collapse
        newSizes[collapsedIndex] *= sizeOuter;
        continue;
      }
    }

    // Otherwise, strides don't match, or dim `i` is excluded from
    // collapsing.
    --collapsedIndex;
    assert(collapsedIndex >= 0);
    assert(collapsedIndex < newDims);
    newSizes[collapsedIndex] = sizeOuter;
    newStrides[collapsedIndex] = strideOuter;

    if (excludeDim == i) {
      returnDim = collapsedIndex;
    }
  }

  // We must have filled all the dimensions we're looking for
  assert(collapsedIndex == 0);
  assert((excludeDim == -1) || (returnDim != -1));

  dims = newDims;

  for (int i = 0; i < dims; ++i) {
    sizes[i] = newSizes[i];
    strides[i] = newStrides[i];
  }

  // After collapsing, the original `excludeDim` may have been
  // renumbered to this new `returnDim`, since some dimensions could
  // have been collapsed.
  return returnDim;
}

// Translate a linear index for the apply to a float* offset;
// specialized on `Dims` to reduce nvcc compilation time
template <typename IndexType, int Dims>
struct IndexToOffset {
  static __host__ __device__ IndexType get(
    IndexType linearId,
    const TensorInfo<IndexType>& info) {
    IndexType offset = 0;

    // Use static dims
    for (int i = Dims - 1; i >= 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;

      if (i > 0) {
        linearId /= info.sizes[i];
      }
    }

    return offset;
  }
};

template <typename IndexType>
struct IndexToOffset<IndexType, -2> {
  static __forceinline__ __host__ __device__ IndexType
    get(IndexType linearId, const TensorInfo<IndexType>& info) {
    return linearId;
  }
};

template <typename IndexType>
struct IndexToOffset<IndexType, -1> {
  static __forceinline__ __host__ __device__ IndexType
    get(IndexType linearId, const TensorInfo<IndexType>& info) {
    IndexType offset = 0;

    // Use dynamic dims
    for (int i = info.dims - 1; i >= 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;

      linearId /= info.sizes[i];
    }

    return offset;
  }
};

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

// Returns true if all linear ID -> offset math can be performed using 32 bit
// unsigned math, which is faster than 64 bit math
THC_API bool THC_canUse32BitIndexMath(THCState* state, THCudaTensor* t);

// Produces a grid with at least one point per tile
THC_API bool THC_getGridFromTiles(long gridTiles, dim3& grid);

// Determines if the given tensor has overlapping data points (i.e.,
// is there more than one index into the tensor that references the
// same piece of data)?
THC_API bool THC_overlappingIndices(THCState* state, THCudaTensor* t);

#endif // THC_REDUCE_APPLY_UTILS_INC
