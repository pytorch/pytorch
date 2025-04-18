#pragma once

#include <c10/metal/utils.h>
#include <metal_compute>

namespace c10 {
namespace metal {

constant constexpr ushort simdgroup_size = 32;

template <typename T>
inline ::metal::enable_if_t<!::metal::is_same_v<T, long>, T> simd_sum(T val) {
  return ::metal::simd_sum(val);
}

template <typename T>
inline ::metal::enable_if_t<!::metal::is_same_v<T, long>, T> simd_prod(T val) {
  return ::metal::simd_product(val);
}

// Metal does not support SIMD reductions over 64-bit types, but it could be
// implement using simd_shuffle_down, that yields result in log2(simdgroup_size)
// iterations Use fill variant, as shuffle down returns garbage if inactive
// thread is referenced (on M1/M2, works fine on M4) and broadcast result to all
// threads in the end. Implementation heavily borrows from
// https://github.com/ml-explore/mlx/blob/86389bf9707f46101af45d90510e8e97c8a90b93/mlx/backend/metal/kernels/reduction/ops.h#L16
template <typename T>
inline ::metal::enable_if_t<::metal::is_same_v<T, long>, T> simd_sum(T val) {
  for (ushort i = simdgroup_size / 2; i > 0; i /= 2) {
    val += as_type<T>(
        ::metal::simd_shuffle_and_fill_down(as_type<int2>(val), int2(0), i));
  }
  return as_type<T>(::metal::simd_broadcast(as_type<int2>(val), 0));
}

template <typename T>
inline ::metal::enable_if_t<::metal::is_same_v<T, long>, T> simd_prod(T val) {
  for (ushort i = simdgroup_size / 2; i > 0; i /= 2) {
    val *= as_type<T>(
        ::metal::simd_shuffle_and_fill_down(as_type<int2>(val), int2(0), i));
  }
  return as_type<T>(::metal::simd_broadcast(as_type<int2>(val), 0));
}

// Below algorithms are  written with hardcoded assumption that simdgroup is 32
// and threadgroup_max is 1024, i.e. reduction can be done in two stages max
template <typename T>
opmath_t<T> threadgroup_sum(
    threadgroup opmath_t<T>* data,
    T val,
    unsigned idx,
    unsigned size) {
  auto rc = simd_sum(static_cast<opmath_t<T>>(val));
  if (idx % simdgroup_size == 0) {
    data[idx / simdgroup_size] = rc;
  }
  if (size > simdgroup_size) {
    ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);
    if (idx < ((size + simdgroup_size - 1) / simdgroup_size)) {
      auto rc1 = simd_sum(data[idx]);
      if (idx == 0) {
        data[0] = rc1;
      }
    }
  }
  ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);
  return data[0];
}

template <typename T>
opmath_t<T> threadgroup_prod(
    threadgroup opmath_t<T>* data,
    T val,
    unsigned idx,
    unsigned size) {
  auto rc = simd_prod(static_cast<opmath_t<T>>(val));
  if (idx % simdgroup_size == 0) {
    data[idx / simdgroup_size] = rc;
  }
  if (size > simdgroup_size) {
    ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);
    if (idx < ((size + simdgroup_size - 1) / simdgroup_size)) {
      auto rc1 = simd_prod(data[idx]);
      if (idx == 0) {
        data[0] = rc1;
      }
    }
  }
  ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);
  return data[0];
}

template <typename T>
float3 threadgroup_welford_reduce(threadgroup T* data, unsigned size) {
  ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);
  float m = data[0];
  float m2 = 0;
  for (unsigned idx = 1; idx < size; ++idx) {
    float delta = data[idx] - m;
    m += delta / (idx + 1);
    m2 += delta * (data[idx] - m);
  }
  return float3(m, m2, size);
}

// Each vec3type is tuple of mean, m2 and weight
template <typename T>
float3 welford_combine(T a, T b) {
  float delta = b.x - a.x;
  float new_weight = a.z + b.z;
  auto w2_over_w = new_weight != 0 ? b.z / new_weight : 0.0;
  return float3(
      a.x + delta * w2_over_w,
      a.y + b.y + delta * delta * a.z * w2_over_w,
      new_weight);
}

template <typename T>
float3 threadgroup_welford_combine(threadgroup T* data, unsigned size) {
  ::metal::threadgroup_barrier(::metal::mem_flags::mem_threadgroup);
  float3 rc = data[0];
  for (unsigned idx = 1; idx < size; ++idx) {
    rc = welford_combine(rc, data[idx]);
  }
  return rc;
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
  for (unsigned idx = 1; idx < size; ++idx) {
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
  for (unsigned idx = 1; idx < size; ++idx) {
    if (data[idx] < data[rc]) {
      rc = idx;
    }
  }
  return rc;
}

} // namespace metal
} // namespace c10
