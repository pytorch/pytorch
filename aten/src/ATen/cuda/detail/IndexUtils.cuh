#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/IndexingUtils.h>

namespace at {
namespace cuda {
namespace detail {

TORCH_CUDA_CU_API bool maybeOverlappingIndices(const at::Tensor& t);
using at::native::canUse32BitIndexMath;

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
    t.data_ptr<scalar>(), dims, sz, st);
}

} // detail
} // cuda
} // at
