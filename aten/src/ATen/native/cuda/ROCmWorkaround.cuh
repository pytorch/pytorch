#pragma once

namespace rocm { namespace workaround {

template<typename T>
struct enable_default_constructor {
  T value;
  enable_default_constructor() : T(0) {}
  __device__ inline operator T&() {
    return *value;
  }
  __device__ inline enable_default_constructor &operator=(T x) {
    value = x;
    return *this;
  }
};

}}  // namespace rocm::workaround
