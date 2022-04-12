#pragma once

// A fixed-size array type usable from both host and
// device code.

#include <c10/macros/Macros.h>
#include <c10/util/irange.h>

namespace at { namespace detail {

template <typename T, int size_>
struct Array {
  T data[size_];

  C10_HOST_DEVICE T operator[](int i) const {
    return data[i];
  }
  C10_HOST_DEVICE T& operator[](int i) {
    return data[i];
  }
#if defined(USE_ROCM)
  C10_HOST_DEVICE Array() = default;
  C10_HOST_DEVICE Array(const Array&) = default;
  C10_HOST_DEVICE Array& operator=(const Array&) = default;
#else
  Array() = default;
  Array(const Array&) = default;
  Array& operator=(const Array&) = default;
#endif
  static constexpr int size(){return size_;}
  // Fill the array with x.
  C10_HOST_DEVICE Array(T x) {
    for (int i = 0; i < size_; i++) {
      data[i] = x;
    }
  }
};

}}
