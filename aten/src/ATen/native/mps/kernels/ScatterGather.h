#pragma once
#include <c10/metal/common.h>

template <unsigned N = c10::metal::max_ndim>
struct ScatterGatherParams {
  int32_t ndim;
  int32_t dim;
  ::c10::metal::array<uint32_t, N> self_strides;
  ::c10::metal::array<uint32_t, N> self_sizes;
  ::c10::metal::array<uint32_t, N> src_strides;
  ::c10::metal::array<uint32_t, N> src_sizes;
  ::c10::metal::array<uint32_t, N> index_strides;
  ::c10::metal::array<uint32_t, N> index_sizes;
};
