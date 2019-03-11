#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <limits>

namespace at {
namespace cuda {
namespace detail {

CAFFE2_API bool maybeOverlappingIndices(const at::Tensor& t);
CAFFE2_API bool canUse32BitIndexMath(const at::Tensor &t, int64_t max_elem=std::numeric_limits<int32_t>::max());

template <typename scalar, typename IndexType>
TensorInfo<scalar, IndexType>
getTensorInfo(const at::Tensor& t) {
  IndexType sz[MAX_TENSORINFO_DIMS];
  IndexType st[MAX_TENSORINFO_DIMS];

  int dims = t.dim();
  for (int i = 0; i < dims; ++i) {
    sz[i] = t.size(i);
    st[i] = t.stride(i);
  }

  return TensorInfo<scalar, IndexType>(
    t.data<scalar>(), dims, sz, st);
}

} // detail
} // cuda
} // at
