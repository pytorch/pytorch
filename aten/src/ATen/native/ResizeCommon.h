#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/irange.h>

namespace at { namespace native {

inline int64_t storage_size_for(IntArrayRef size, IntArrayRef stride) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(size.size() == stride.size(),
      "storage_size_for(size, stride) requires that size and stride ",
      "have the same size as a precondition.");
  int64_t storage_size = 1;
  for (const auto dim : c10::irange(size.size())) {
    if (size[dim] == 0) {
      storage_size = 0;
      break;
    }
    storage_size += (size[dim] - 1) * stride[dim];
  }
  return storage_size;
}

}} // at::native
