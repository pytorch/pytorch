#pragma once

#include <type_traits>
#include <new>

namespace rocm { namespace workaround {

template<typename T>
struct enable_default_constructor {
  static_assert(std::is_trivially_copyable<T>{} && std::is_trivially_destructible<T>{}, "Not trivial type");
  struct { char bytes[sizeof(T)]; } value;
  __device__ operator T&() {
    return *reinterpret_cast<T *>(&value);
  }
  __device__ enable_default_constructor &operator=(T x) {
    value = *reinterpret_cast<decltype(&value)>(&x);
    return *this;
  }
};

}}  // namespace rocm::workaround
