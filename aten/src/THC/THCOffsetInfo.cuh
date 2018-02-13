#ifndef THC_OFFSET_INFO_INC
#define THC_OFFSET_INFO_INC

#include "THCIntegerDivider.cuh"
#include "THCTensorInfo.cuh"

// A faster implementation of IndexToOffset that uses faster integer division:
// we transform each division into integer multiplication by a pre-computed
// constant.  (See IntDivider for details.)

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

    offset += linearIndex * strides[0];
    return &data[offset];
  }

  T* data;
  IntDivider<IndexType> sizes[Dims];
  IndexType strides[Dims];
};

// For contiguous tensors (Dims=-2), offset equals linear index.
template <typename T, typename IndexType>
struct OffsetInfo<T, IndexType, -2> {
  explicit OffsetInfo(const TensorInfo<T, IndexType>& tinfo)
    : data(tinfo.data) {
    assert(tinfo.isContiguous());
  }

  __host__ __device__ T* get(IndexType linearIndex) const {
    return &data[linearIndex];
  }

  T* data;
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
    IndexType offset = 0;

    for (int i = tinfo.dims - 1; i > 0; --i) {
      IndexType curDimIndex = linearIndex % tinfo.sizes[i];
      linearIndex /= tinfo.sizes[i];
      offset += curDimIndex * tinfo.strides[i];
    }

    offset += linearIndex * tinfo.strides[0];

    return &tinfo.data[offset];
  }

  TensorInfo<T, IndexType> tinfo;
};

#endif // THC_OFFSET_INFO_INC
