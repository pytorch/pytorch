#pragma once

namespace rocm { namespace workaround {

template<typename T>
struct enable_default_constructor {
  struct { char bytes[sizeof(T)]; } value;
  T &operator T() {
    return *reinterpret_cast<T *>(&value);
  }
  enable_default_constructor &operator=(T x) {
    value = *reinterpret_cast<decltype(&value)>(&x);
    return *this;
  }
};

}}  // namespace rocm::workaround
