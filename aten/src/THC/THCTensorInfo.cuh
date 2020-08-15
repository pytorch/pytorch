#ifndef THC_TENSOR_INFO_INC
#define THC_TENSOR_INFO_INC

#include <cuda.h>
#include <assert.h>
#include <THC/THCGeneral.h>
#include <THC/THCIntegerDivider.cuh>
#include <THC/THCTensor.h>

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
  TORCH_INTERNAL_ASSERT(dim < dims && dim >= 0);
  sizes[dim] = 1;
}

template <typename T, typename IndexType>
int
TensorInfo<T, IndexType>::collapseDims(const int excludeDim) {

  TORCH_INTERNAL_ASSERT(excludeDim >= -1 && excludeDim < dims);

  int stopDim = (excludeDim == -1) ? dims : excludeDim;
  int newIndex = -1;
  int oldIndex = 0;
  int remappedExcludedDim = -1;

  while (oldIndex < dims) {
    // Finds a dimension to collapse into
    for (; oldIndex < stopDim; ++oldIndex) {
      if (sizes[oldIndex] == 1) {
        continue;
      }
      ++newIndex;
      sizes[newIndex] = sizes[oldIndex];
      strides[newIndex] = strides[oldIndex];
      ++oldIndex;
      break;
    }

    // Collapses dims
    for (; oldIndex < stopDim; ++oldIndex) {
      if (sizes[oldIndex] == 1) {
        continue;
      }

      if (strides[newIndex] == sizes[oldIndex] * strides[oldIndex]) {
        sizes[newIndex] *= sizes[oldIndex];
        strides[newIndex] = strides[oldIndex];
      } else {
        ++newIndex;
        sizes[newIndex] = sizes[oldIndex];
        strides[newIndex] = strides[oldIndex];
      }
    }

    // Handles excludeDim being set (oldIndex == excludeDim)
    if (oldIndex != dims) {

      // Preserves excluded dimension
      ++newIndex;
      sizes[newIndex] = sizes[oldIndex];
      strides[newIndex] = strides[oldIndex];
      remappedExcludedDim = newIndex;

      // Restarts iteration after excludeDim
      ++oldIndex;
      stopDim = dims;
    }
  }

  // Handles special case of all dims size 1
  if (newIndex == -1 || (newIndex == 0 && sizes[0] == 1)) {
    dims = 1;
    sizes[0] = 1;
    strides[0] = 1;

    return 0;
  }

  dims = newIndex + 1;
  return remappedExcludedDim;
}

// Translate a linear index for the apply to a T* offset;
// specialized on `Dims` to reduce nvcc compilation time
template <typename T, typename IndexType, int Dims>
struct IndexToOffset {
  static __host__ __device__ IndexType get(
    IndexType linearId,
    const TensorInfo<T, IndexType>& info) {

    IndexType offset = 0;

    // Uses static dims
    for (int i = Dims - 1; i > 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;
      linearId /= info.sizes[i];
    }

    return offset + linearId * info.strides[0];
  }
};

template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, -1> {
  static inline __host__ __device__ IndexType get(
    IndexType linearId,
    const TensorInfo<T, IndexType>& info) {

    IndexType offset = 0;

    // Uses dynamic dims
    for (int i = info.dims - 1; i > 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;
      linearId /= info.sizes[i];
    }

    return offset + linearId * info.strides[0];
  }
};

// OffsetInfo is a faster implementation of IndexToOffset that uses faster
// integer division: we transform each division into integer multiplication by a
// pre-computed constant.  (See IntDivider for details.)
template <typename T, typename IndexType, int Dims>
struct OffsetInfo {
  explicit OffsetInfo(const TensorInfo<T, IndexType>& tinfo) {
    assert(tinfo.dims == Dims);
    data = tinfo.data;

    for (int i = 0; i < Dims; ++i) {
      sizes[i] = IntDivider<IndexType>(tinfo.sizes[i]);
      strides[i] = tinfo.strides[i];
    }
  }

  __host__ __device__ T* get(IndexType linearIndex) const {
    IndexType offset = 0;

    for (int i = Dims - 1; i > 0; --i) {
      DivMod<IndexType> divmod = sizes[i].divmod(linearIndex);
      linearIndex = divmod.div;
      offset += divmod.mod * strides[i];
    }

    return &data[offset + linearIndex * strides[0]];
  }

  T* data;
  IntDivider<IndexType> sizes[Dims];
  IndexType strides[Dims];
};

// For 1D tensors the offset equals linear index * stride.
template <typename T, typename IndexType>
struct OffsetInfo<T, IndexType, 1> {
  explicit OffsetInfo(const TensorInfo<T, IndexType>& tinfo)
    : data{tinfo.data}, stride{tinfo.strides[0]} {}

  __host__ __device__ T* get(IndexType linearIndex) const {
    return &data[linearIndex * stride];
  }

  T* data;
  const IndexType stride;
};

// Dims=-1 is used when the dimension is unknown at compile time.
//
// Unfortunately, pre-computation does not work here, because of a bug in nvcc
// (tested on CUDA 8.0): if a kernel argument contains an array that is
// dynamically accessed, the whole array is first copied into the local memory.
// (That is, every kernel thread makes its own copy of the argument, even if it
// is never updated.)  Pre-computation makes it worse because now we have more
// data to copy.
//
// So let's fall back to vanilla division approach.

template <typename T, typename IndexType>
struct OffsetInfo<T, IndexType, -1> {
  explicit OffsetInfo(const TensorInfo<T, IndexType>& tinfo)
    : tinfo(tinfo) { }

  __host__ __device__ T* get(IndexType linearIndex) const {
    IndexType offset = IndexToOffset<T, IndexType, -1>::get(linearIndex, tinfo);
    return &tinfo.data[offset];
  }

  TensorInfo<T, IndexType> tinfo;
};

#endif // THC_TENSOR_INFO_INC
