#pragma once

#include <ATen/ATen.h>
#include <ATen/CPUApplyUtils.h>

namespace at {
namespace cuda {
namespace detail {

#define MAX_TENSORINFO_DIMS 25

// CUDA kernel argument that defines tensor layout
template <typename T, typename IndexType>
struct TensorInfo {
  TensorInfo();
  TensorInfo(T* p,
             int dim,
             IndexType sz[MAX_TENSORINFO_DIMS],
             IndexType st[MAX_TENSORINFO_DIMS]);

  // Set the size of the given dimension to 1, as if it were a
  // reduction dim (allows you to calculate offsets of the reduction
  // slice)
  void reduceDim(int dim);

  // See note on [collapse dims].
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
TensorInfo<T, IndexType>::TensorInfo() {
  data = nullptr;
  dims = 0;
}

template <typename T, typename IndexType>
TensorInfo<T, IndexType>::TensorInfo(T* p,
                                     int dim,
                                     IndexType sz[MAX_TENSORINFO_DIMS],
                                     IndexType st[MAX_TENSORINFO_DIMS]) {
  data = p;
  dims = dim;
  AT_ASSERT(dims < MAX_TENSORINFO_DIMS);

  for (int i = 0; i < dim; ++i) {
    sizes[i] = sz[i];
    strides[i] = st[i];
  }
}

template <typename T, typename IndexType>
void
TensorInfo<T, IndexType>::reduceDim(int dim) {
  TORCH_CHECK(dim < dims && dim >= 0, "expected dim between 0 and dims - 1");
  sizes[dim] = 1;
}

template <typename T, typename IndexType>
int
TensorInfo<T, IndexType>::collapseDims(const int excludeDim) {
  auto result = at::collapse_dims(sizes, strides, dims, excludeDim);
  dims = std::get<1>(result);
  return std::get<0>(result);
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

// Uses dynamic (runtime) instead of static (compiletime) dims
template <typename T, typename IndexType>
struct IndexToOffset<T, IndexType, -1> {
  static inline __host__ __device__ IndexType get(
    IndexType linearId,
    const TensorInfo<T, IndexType>& info) {

      IndexType offset = 0;

      for (int i = info.dims - 1; i > 0; --i) {
        IndexType curDimIndex = linearId % info.sizes[i];
        IndexType curDimOffset = curDimIndex * info.strides[i];
        offset += curDimOffset;
        linearId /= info.sizes[i];
      }

      return offset + linearId * info.strides[0];
  }
};

} // detail
} // cuda
} // at
