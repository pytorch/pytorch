#pragma once

#include <ATen/core/TensorBase.h>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/CanUse32BitIndexMath.h>

namespace at::cuda::detail {

TORCH_CUDA_CU_API bool maybeOverlappingIndices(const at::TensorBase &t);
using at::native::canUse32BitIndexMath;

template <typename scalar, typename IndexType>
TensorInfo<scalar, IndexType>
getTensorInfo(const at::TensorBase &t) {
  IndexType sz[MAX_TENSORINFO_DIMS];
  IndexType st[MAX_TENSORINFO_DIMS];

  int dims = t.dim();
  for (int i = 0; i < dims; ++i) {
    sz[i] = t.size(i);
    st[i] = t.stride(i);
  }

  scalar* data_ptr = nullptr;

  if constexpr (std::is_const_v<scalar>) {
    data_ptr = t.const_data_ptr<scalar>();
  } else {
    data_ptr = t.mutable_data_ptr<scalar>();
  }

  return TensorInfo<scalar, IndexType>(
    data_ptr, dims, sz, st);
}

} // namespace at::cuda::detail
