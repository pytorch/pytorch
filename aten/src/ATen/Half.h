#pragma once

#include <limits>
#include <string>
#include <stdint.h>
#include <cmath>
#ifdef AT_CUDA_ENABLED
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

namespace at {

template<typename To, typename From> To convert(From f) {
  return static_cast<To>(f);
}

// skip isnan and isinf check for integral types
template<typename To, typename From>
typename std::enable_if<std::is_integral<From>::value, bool>::type overflows(From f) {
  using limit = std::numeric_limits<To>;
  return f < limit::lowest() || f > limit::max();
}

template<typename To, typename From>
typename std::enable_if<!std::is_integral<From>::value, bool>::type overflows(From f) {
  using limit = std::numeric_limits<To>;
  if (limit::has_infinity && std::isinf(f)) {
    return false;
  }
  if (!limit::has_quiet_NaN && std::isnan(f)) {
    return true;
  }
  return f < limit::lowest() || f > limit::max();
}

template<typename To, typename From> To checked_convert(From f, const char* name) {
  if (overflows<To, From>(f)) {
    std::string msg = "value cannot be converted to type ";
    msg += name;
    msg += " without overflow: ";
    msg += std::to_string(f);
    throw std::domain_error(std::move(msg));
  }
  return convert<To, From>(f);
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

template<> AT_API Half convert(double f);
template<> AT_API double convert(Half f);
template<> AT_API Half convert(int64_t f);
template<> AT_API int64_t convert(Half f);

template<> bool overflows<Half, double>(double f);
template<> bool overflows<Half, int64_t>(int64_t f);

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
