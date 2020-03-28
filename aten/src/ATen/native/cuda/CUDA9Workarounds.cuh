#pragma once

namespace cuda9 { namespace workaround {

#if CUDA_VERSION < 10000
template<typename T>
struct alignas(T) enable_default_constructor {
  struct { char bytes[sizeof(T)]; } value;
  __device__ operator T&() {
    return *reinterpret_cast<T *>(&value);
  }
  __device__ enable_default_constructor &operator=(T x) {
    value = *reinterpret_cast<decltype(&value)>(&x);
    return *this;
  }
};
#else
template<typename T>
using enable_default_constructor = T;
#endif

}}  // namespace cuda9::workaround
