#pragma once

#include <c10/metal/utils.h>
#include <metal_compute>

namespace c10 {
namespace metal {

template <typename T>
opmath_t<T> threadgroup_sum(threadgroup T* data, unsigned size) {
  opmath_t<T> rc = 0;
  // TODO: This should be moved to the callee
  ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);
  // TODO: Use `simd_shuffle_down`
  for (auto idx = 0; idx < size; ++idx) {
    rc += data[idx];
  }
  return rc;
}

template <typename T>
T threadgroup_max(threadgroup T* data, unsigned size) {
  // TODO: This should be moved to the callee
  ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);
  T rc = data[0];
  // TODO: Use `simd_shuffle_down`
  for (auto idx = 1; idx < size; ++idx) {
    rc = ::c10::metal::max(rc, data[idx]);
  }
  return rc;
}

} // namespace metal
} // namespace c10
