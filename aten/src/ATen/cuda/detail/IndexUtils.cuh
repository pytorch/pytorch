#pragma once

#include "ATen/ATen.h"
#include "TensorInfo.cuh"

namespace at {
namespace cuda {
namespace detail {

bool overlappingIndices(const at::Tensor& t);
bool canUse32BitIndexMath(const at::Tensor &t, ptrdiff_t max_elem=UINT32_MAX);

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
