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
  // Successive dimensions can be collapsed if the size/strides match
  // up and thus there are no holes between the dimensions. This is used
  // to reduce the complexity of the problem.
  // The optional `reduceDim` indicates a reduction dimension for the
  // given tensor, so that the output size for this dimension will be 1.
  TensorInfo(THCState* state, THCudaTensor* t, int reduceDim = -1);

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
                                  int reduceDim)
    : data(NULL), dims(0) {
  int origDims = THCudaTensor_nDimension(state, t);
  assert(origDims <= MAX_CUTORCH_DIMS);
  assert(reduceDim < origDims);

  data = THCudaTensor_data(state, t);

  // Count the number of successive dimensions that can be collapsed, from
  // innermost to outermost.
  int numCollapsed = 0;

  // Find the innermost dimension not of size 1, since dimensions of size 1 are
  // collapsible.
  int firstNonOneDim = -1;

  for (int i = origDims - 1; i >= 0; --i) {
    if (THCudaTensor_size(state, t, i) != 1 && i != reduceDim) {
      firstNonOneDim = i;
      break;
    }
  }

  // Special case: if all dimensions are of size 1, then this is a
  // single-point tensor that we still have to operate on. Reduce to a
  // single point.
  if (firstNonOneDim == -1) {
    dims = 1;
    sizes[0] = 1;
    strides[0] = 1;
    return;
  }

  // Skip the leading size 1 dims
  numCollapsed += origDims - 1 - firstNonOneDim;

  // Now, to determine the other collapsible dims. These are the size/strides
  // of the previous inner non-collapsible dim we encounter.
  long sizeInner = THCudaTensor_size(state, t, firstNonOneDim);
  long strideInner = THCudaTensor_stride(state, t, firstNonOneDim);

  for (int i = firstNonOneDim - 1; i >= 0; --i) {
    long sizeOuter = (i == reduceDim) ? 1 : THCudaTensor_size(state, t, i);
    long strideOuter = THCudaTensor_stride(state, t, i);

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

    // Otherwise, this new outer dimension at `i` cannot be collapsed
    // and is different from the previous.
    sizeInner = sizeOuter;
    strideInner = strideOuter;
  }

  assert(numCollapsed < origDims);
  dims = origDims - numCollapsed;

  // Determine the sizes of the collapsed dimensions.
  int collapsedIndex = origDims - numCollapsed - 1;
  sizes[collapsedIndex] = THCudaTensor_size(state, t, firstNonOneDim);
  strides[collapsedIndex] = THCudaTensor_stride(state, t, firstNonOneDim);

  for (int i = firstNonOneDim - 1; i >= 0; --i) {
    long sizeOuter = (i == reduceDim) ? 1 : THCudaTensor_size(state, t, i);
    long strideOuter = THCudaTensor_stride(state, t, i);

    if (sizeOuter == 1) {
      // skip
      continue;
    }

    if (strideOuter == sizes[collapsedIndex] * strides[collapsedIndex]) {
      // collapse
      sizes[collapsedIndex] *= sizeOuter;
      continue;
    }

    // Otherwise, strides don't match; dimension `i` is not collapsible.
    --collapsedIndex;
    assert(collapsedIndex >= 0);
    sizes[collapsedIndex] = sizeOuter;
    strides[collapsedIndex] = strideOuter;
  }

  // We must have filled all the dimensions we're looking for
  assert(collapsedIndex == 0);
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
