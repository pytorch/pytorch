#pragma once

#include <ATen/native/cuda/MemoryAccess.cuh>

namespace rocm { namespace workaround {

template<typename T>
struct enable_default_constructor {
  struct { char bytes[sizeof(T)]; } value;
  T operator T() {
    return typeless_cast<T>(value);
  }
  enable_default_constructor &operator=(T x) {
    value = typeless_cast<decltype(value)>(x);
    return *this;
  }
};

}}  // namespace rocm::workaround
