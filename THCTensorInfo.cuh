#ifndef THC_TENSOR_INFO_INC
#define THC_TENSOR_INFO_INC

#include <cuda.h>
#include <assert.h>
#include "THCGeneral.h"
#include "THCTensor.h"

// Maximum number of dimensions allowed for cutorch
#define MAX_CUTORCH_DIMS 25

// Warning string for tensor arguments that are too large or have too
// many dimensions
#define CUTORCH_STR(X) #X
#define CUTORCH_DIM_WARNING "tensor too large or too many (>" \
  CUTORCH_STR(MAX_CUTORCH_DIMS) ") dimensions"

// CUDA kernel argument that defines tensor layout
template <typename T, typename IndexType>
struct TensorInfo {
  TensorInfo(T* p,
             int dim,
             IndexType sz[MAX_CUTORCH_DIMS],
             IndexType st[MAX_CUTORCH_DIMS]);

  // Set the size of the given dimension to 1, as if it were a
  // reduction dim (allows you to calculate offsets of the reduction
  // slice)
  void reduceDim(int dim);

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

  T* data;
  IndexType sizes[MAX_CUTORCH_DIMS];
  IndexType strides[MAX_CUTORCH_DIMS];
  int dims;
};

template <typename T, typename IndexType>
TensorInfo<T, IndexType>::TensorInfo(T* p,
                                     int dim,
                                     IndexType sz[MAX_CUTORCH_DIMS],
                                     IndexType st[MAX_CUTORCH_DIMS]) {
  data = p;
  dims = dim;
  assert(dims > 0 && dims < MAX_CUTORCH_DIMS);

  for (int i = 0; i < dim; ++i) {
    sizes[i] = sz[i];
    strides[i] = st[i];
  }
}

template <typename T, typename IndexType>
void
TensorInfo<T, IndexType>::reduceDim(int dim) {
  assert(dim < dims && dim >= 0);
  sizes[dim] = 1;
}

template <typename T, typename IndexType>
int
TensorInfo<T, IndexType>::collapseDims(int excludeDim) {
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
    IndexType sizeOuter = sizes[i];
    IndexType strideOuter = strides[i];

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

// Translate a linear index for the apply to a T* offset;
// specialized on `Dims` to reduce nvcc compilation time
template <typename T, typename IndexType, int Dims>
struct IndexToOffset {
  static __host__ __device__ IndexType get(
    IndexType linearId,
    const TensorInfo<T, IndexType>& info) {
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

// For contiguous tensors, the offset = index
template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, -2> {
  static inline __host__ __device__ IndexType
    get(IndexType linearId, const TensorInfo<T, IndexType>& info) {
    return linearId;
  }
};

template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, -1> {
  static inline __host__ __device__ IndexType get(
    IndexType linearId,
    const TensorInfo<T, IndexType>& info) {

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

#endif // THC_TENSOR_INFO_INC
