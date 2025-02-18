#pragma once
#include <metal_stdlib>

namespace c10 {
namespace metal {
constant constexpr unsigned max_ndim = 16;

// Given coordinates and strides, calculates offset from the start of the
// tensors
inline long offset_from_coord(
    thread long idx[max_ndim],
    constant long* strides,
    uint ndim) {
  long rc = 0;
  for (uint i = 0; i < ndim; ++i) {
    rc += idx[i] * strides[i];
  }
  return rc;
}

// Given thread index calculates position in the ndim tensor
inline void pos_from_thread_index(
    long idx,
    thread long pos[max_ndim],
    constant long* sizes,
    uint ndim) {
  for (uint i = 0; i < ndim; ++i) {
    pos[i] = idx % sizes[i];
    idx /= sizes[i];
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

template <typename T>
struct StridedTensor {
  StridedTensor(
      device T* ptr_,
      constant long* sizes_,
      constant long* strides_,
      uint ndim_)
      : ptr(ptr_), sizes(sizes_), strides(strides_), ndim(ndim_) {}
  T operator[](long idx) const {
    auto offs = offset_from_thread_index(idx, sizes, strides, ndim);
    return ptr[offs];
  }
  device T& operator[](long idx) {
    auto offs = offset_from_thread_index(idx, sizes, strides, ndim);
    return ptr[offs];
  }

 protected:
  device T* ptr;
  constant long* sizes;
  constant long* strides;
  uint ndim;
};
template <typename T>
struct ConstStridedTensor {
  ConstStridedTensor(
      constant T* ptr_,
      constant long* sizes_,
      constant long* strides_,
      uint ndim_)
      : ptr(ptr_), sizes(sizes_), strides(strides_), ndim(ndim_) {}
  T operator[](long idx) const {
    auto offs = offset_from_thread_index(idx, sizes, strides, ndim);
    return ptr[offs];
  }

 protected:
  constant T* ptr;
  constant long* sizes;
  constant long* strides;
  uint ndim;
};

} // namespace metal
} // namespace c10
