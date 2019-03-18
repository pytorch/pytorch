#pragma once

// A fixed-size array type usable from both host and
// device code.

#include <c10/macros/Macros.h>

namespace at { namespace detail {

template <typename T, int size>
struct alignas(16) Array {
  T data[size];

  C10_HOST_DEVICE T operator[](int i) const {
    return data[i];
  }
  C10_HOST_DEVICE T& operator[](int i) {
    return data[i];
  }

  // Fill the array with x.
  C10_HOST_DEVICE Array(T x) {
    for (int i = 0; i < size; i++) {
      data[i] = x;
    }
  }
};

}}
