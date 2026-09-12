#pragma once
#include <c10/metal/common.h>

C10_METAL_CONSTEXPR unsigned kSoftmaxThreads = 256;
C10_METAL_CONSTEXPR unsigned kSoftmaxMaxThreads = 1024;

template <typename index_t = uint64_t>
struct SoftmaxParams {
  index_t dim_size;
  index_t num_rows;
  index_t inner_size;
  index_t chunk_size;
  index_t n_chunks;
  uint32_t ndim;
  uint32_t dim;
  ::c10::metal::array<index_t, ::c10::metal::max_ndim> sizes;
  ::c10::metal::array<index_t, ::c10::metal::max_ndim> strides;
};
