#pragma once

// A fixed-size array type usable from CUDA kernels.

#include <ATen/core/Macros.h>

namespace at { namespace cuda {

template <typename T, int size>
struct Array {
  T data[size];

  AT_HOST_DEVICE T operator[](int i) const { return data[i]; }
  AT_HOST_DEVICE T& operator[](int i) { return data[i]; }

  HIP_HOST_DEVICE Array() = default;
  HIP_HOST_DEVICE Array(const Array&) = default;
  HIP_HOST_DEVICE Array& operator=(const Array&) = default;

  // Fill the array with x.
  AT_HOST_DEVICE Array(T x) {
    for (int i = 0; i < size; i++) {
      data[i] = x;
    }
  }
};

}}
