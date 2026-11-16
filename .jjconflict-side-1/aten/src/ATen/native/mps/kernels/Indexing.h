#pragma once
#include <c10/metal/common.h>

template <unsigned N = c10::metal::max_ndim>
struct IndexReduceParams {
  int32_t ndim;
  int32_t reduce_dim;
  ::c10::metal::array<uint32_t, N> self_strides;
  ::c10::metal::array<uint32_t, N> self_sizes;
  ::c10::metal::array<uint32_t, N> source_strides;
  ::c10::metal::array<uint32_t, N> source_sizes;
  uint32_t index_stride;
};

// Dense index_select: tensors viewed as [outer, dim, inner] contiguous blocks.
struct IndexSelectParams {
  uint32_t inner;
  uint32_t in_dim_size;
  uint32_t out_dim_size;
  uint32_t index_stride;
};
