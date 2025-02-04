#pragma once

#include <c10/metal/utils.h>

namespace c10 {
namespace metal {

template<typename T>
opmath_t<T> threadgroup_sum(threadgroup T* data, unsigned size) {
  opmath_t<T> rc = 0;
  // TODO: Use `simd_shuffle_down`
  for(auto idx = 0; idx < size; ++idx) {
    rc += data[idx];
  }
  return rc;
}

} // namespace metal
} // namespace c10
