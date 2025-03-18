#pragma once

#include <c10/metal/utils.h>
#include <metal_compute>

namespace c10 {
namespace metal {

template <typename T>
opmath_t<T> threadgroup_sum(threadgroup T* data, unsigned size) {
  // TODO: This should be moved to the callee
  ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);
  opmath_t<T> rc = data[0];
  // TODO: Use `simd_shuffle_down`
  for (unsigned idx = 1; idx < size; ++idx) {
    rc += data[idx];
  }
  return rc;
}

template <typename T>
opmath_t<T> threadgroup_prod(threadgroup T* data, unsigned size) {
  // TODO: This should be moved to the callee
  ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);
  opmath_t<T> rc = data[0];
  for (unsigned idx = 1; idx < size; ++idx) {
    rc *= data[idx];
  }
  return rc;
}

template <typename T>
float2 threadgroup_welford_reduce(threadgroup T* data, unsigned size) {
  float m = data[0];
  float m2 = 0;
  for (unsigned idx = 1; idx < size; ++idx) {
    float delta = data[idx] - m;
    m += delta / (idx + 1);
    m2 += delta * (data[idx] - m);
  }
  return float2(m, m2);
}

template <typename T>
T threadgroup_max(threadgroup T* data, unsigned size) {
  // TODO: This should be moved to the callee
  ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);
  T rc = data[0];
  for (unsigned idx = 1; idx < size; ++idx) {
    rc = ::c10::metal::max(rc, data[idx]);
  }
  return rc;
}

template <typename T>
T threadgroup_min(threadgroup T* data, unsigned size) {
  // TODO: This should be moved to the callee
  ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);
  T rc = data[0];
  for (unsigned idx = 1; idx < size; ++idx) {
    rc = ::c10::metal::min(rc, data[idx]);
  }
  return rc;
}

template <typename T>
int threadgroup_argmax(threadgroup T* data, unsigned size) {
  // TODO: This should be moved to the callee
  ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);
  int rc = 0;
  for (int idx = 1; idx < size; ++idx) {
    if (data[idx] > data[rc]) {
      rc = idx;
    }
  }
  return rc;
}

template <typename T>
int threadgroup_argmin(threadgroup T* data, unsigned size) {
  // TODO: This should be moved to the callee
  ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);
  int rc = 0;
  for (int idx = 1; idx < size; ++idx) {
    if (data[idx] < data[rc]) {
      rc = idx;
    }
  }
  return rc;
}

} // namespace metal
} // namespace c10
