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

// ForwardIt: only legacy random access iterator is supported.
template<class ForwardIt, class T, bool is_lower = true>
static __host__ __device__ __forceinline__
ForwardIt find_bound(ForwardIt first, ForwardIt last, const T& value) {
    ForwardIt it;
    typename std::iterator_traits<ForwardIt>::difference_type count, step;
    // NOTE: std::distance(first, last) compiles but produces wrong results here,
    // so only legacy random access iterators are safe in this code.
    count = last - first;

    while (count > 0) {
      it = first;
      step = count / 2;
      // avoiding std::advance(it, step),
      // although it does work unlike std::distance
      it += step;
      if (is_lower ? *it < value : value >= *it) {
        first = ++it;
        count -= step + 1;
      }
      else {
        count = step;
      }
    }
    return first;
}

} // namespace at::cuda::detail
