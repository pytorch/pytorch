#pragma once

namespace rocm { namespace workaround {

template<typename T>
struct enable_default_constructor {
  T value;
  enable_default_constructor() : T(0) {}
  operator T&() {
    return *reinterpret_cast<T *>(&value);
  }
  enable_default_constructor &operator=(T x) {
    value = *reinterpret_cast<decltype(&value)>(&x);
    return *this;
  }
};

}}  // namespace rocm::workaround
