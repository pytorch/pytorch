#pragma once

#include<stdint.h>
#ifdef AT_CUDA_ENABLED
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

namespace at {

template<typename To, typename From> To convert(From f) {
  return static_cast<To>(f);
}

#if defined(__GNUC__)
#define AT_ALIGN(n) __attribute__((aligned(n)))
#elif defined(_WIN32)
#define AT_ALIGN(n) __declspec(align(n))
#else
#define AT_ALIGN(n)
#endif


typedef struct  AT_ALIGN(2) {
  unsigned short x;
#ifdef AT_CUDA_ENABLED
#if CUDA_VERSION < 9000
  operator half() { return half{ x }; }
#else
  operator half() {
    __half_raw x_raw;
    x_raw.x = x;
    return half(x_raw);
  }
#endif
#endif
  operator double();
} Half;

template<> Half convert(double f);
template<> double convert(Half f);
template<> Half convert(int64_t f);
template<> int64_t convert(Half f);

inline Half::operator double() {
  return convert<double,Half>(*this);
}
#ifdef AT_CUDA_ENABLED
template<> half convert(double d);
#endif

template<typename To, typename From>
static inline To HalfFix(From h) {
  return To { h.x };
}

#ifdef AT_CUDA_ENABLED
#if CUDA_VERSION >= 9000
template<>
  inline __half HalfFix<__half, Half>(Half h) {
  __half_raw raw;
  raw.x = h.x;
  return __half { raw };
}

template<>
  inline Half HalfFix<Half, __half>(__half h) {
  __half_raw raw(h);
  return Half { raw.x };
}
#endif
#endif
}
