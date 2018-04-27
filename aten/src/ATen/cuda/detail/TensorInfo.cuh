#pragma once

#include "ATen/ATen.h"

namespace at {
namespace cuda {
namespace detail {

#define MAX_TENSORINFO_DIMS 25

// CUDA kernel argument that defines tensor layout
template <typename T, typename IndexType>
struct TensorInfo {
  TensorInfo(T* p,
             int dim,
             IndexType sz[MAX_TENSORINFO_DIMS],
             IndexType st[MAX_TENSORINFO_DIMS]);

  // Set the size of the given dimension to 1, as if it were a
  // reduction dim (allows you to calculate offsets of the reduction
  // slice)
  void reduceDim(int dim);

  /*
  Updates the TensorInfo's dims, sizes, and strides to reflect a "collapse" of
  the info, possibly excluding the optional excludeDim. A "collapsed" version
  of the info is the fewest dims that order the tensor's elements in the same
  way as the original info. If excludeDim is specified, the collapse is the
  fewest dims that order the tensor's elements as the original and preserve the
  excluded dimension, unless the tensor collapses to a point.

  Returns the (new) index of the preserved dimension if excludeDim is
  specified. Returns 0 if the tensor is collapsed to a point. Returns -1
  otherwise.
  */
  int collapseDims(const int excludeDim = -1);

  // Contiguous tensors of more than one dimension are collapsed down
  // to one tensor
  __host__ __device__ inline bool isContiguous() const {
    return (dims == 1 && strides[0] == 1);
  }

  T* data;
  IndexType sizes[MAX_TENSORINFO_DIMS];
  IndexType strides[MAX_TENSORINFO_DIMS];
  int dims;
};

template <typename T, typename IndexType>
TensorInfo<T, IndexType>::TensorInfo(T* p,
                                     int dim,
                                     IndexType sz[MAX_TENSORINFO_DIMS],
                                     IndexType st[MAX_TENSORINFO_DIMS]) {
  data = p;
  dims = dim;
  assert(dims < MAX_TENSORINFO_DIMS);

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
TensorInfo<T, IndexType>::collapseDims(const int excludeDim) {

  int stopDim = (excludeDim == -1) ? dims : excludeDim;
  int nidx = -1;
  int cidx = 0;
  int rVal = -1;

  while (cidx < dims) {
    // Finds a dimension to collapse into
    for (; cidx < stopDim; ++cidx) {
      if (sizes[cidx] == 1) {
        continue;
      }
      ++nidx;
      sizes[nidx] = sizes[cidx];
      strides[nidx] = strides[cidx];
      ++cidx;
      break; 
    }

    // Collapses dims
    for (; cidx < stopDim; ++cidx) {
      if (sizes[cidx] == 1) {
        continue;
      }
  
      if (strides[nidx] == sizes[cidx] * strides[cidx]) {
        sizes[nidx] *= sizes[cidx];
        strides[nidx] = strides[cidx];
      } else {
        ++nidx;
        sizes[nidx] = sizes[cidx];
        strides[nidx] = strides[cidx];
      }
    }

    // Handles excludeDim being set (cidx == excludeDim)
    if (cidx != dims) {
      
      // Preserves excluded dimension
      ++nidx;
      sizes[nidx] = sizes[cidx];
      strides[nidx] = strides[cidx];
      rVal = nidx;

      // Restarts iteration after excludeDim
      ++cidx;
      stopDim = dims;
    }
  }

  // Handles special case of all dims size 1
  if (nidx == -1 || (nidx == 0 && sizes[0] == 1)) {
    dims = 1;
    sizes[0] = 1;
    strides[0] = 1;

    return 0;
  }

  dims = nidx + 1;
  return rVal;
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

} // detail
} // cuda
} // at
