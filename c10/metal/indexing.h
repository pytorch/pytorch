#pragma once
#include <metal_stdlib>

namespace c10 {
namespace metal {
constant constexpr unsigned max_ndim = 16;

// Given coordinates and strides, calculates offset from the start of the
// tensors
template <typename T>
inline T offset_from_coord(
    thread T idx[max_ndim],
    constant long* strides,
    uint ndim) {
  T rc = 0;
  for (uint i = 0; i < ndim; ++i) {
    rc += idx[i] * T(strides[i]);
  }
  return rc;
}

// Given thread index calculates position in the ndim tensor
template <typename T>
inline void pos_from_thread_index(
    T idx,
    thread T pos[max_ndim],
    constant long* sizes,
    uint ndim) {
  for (uint i = 0; i < ndim; ++i) {
    pos[i] = idx % T(sizes[i]);
    idx /= T(sizes[i]);
  }
}

inline long offset_from_thread_index(
    long idx,
    constant long* sizes,
    constant long* strides,
    uint ndim) {
  long pos[max_ndim];
  pos_from_thread_index(idx, pos, sizes, ndim);
  return offset_from_coord(pos, strides, ndim);
}

} // namespace metal
} // namespace c10
