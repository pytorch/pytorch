#pragma once

#include <cstring>
#include <limits>
#include <ATen/core/Macros.h>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#endif

#if defined(__HIP_DEVICE_COMPILE__)
#include <hip/hip_fp16.h>
#endif

namespace at {

/// Constructors

inline AT_HOST_DEVICE Half::Half(float value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  x = __half_as_short(__float2half(value));
#else
  x = detail::float2halfbits(value);
#endif
}

/// Implicit conversions

inline AT_HOST_DEVICE Half::operator float() const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __half2float(*reinterpret_cast<const __half*>(&x));
#else
  return detail::halfbits2float(x);
#endif
}

#ifdef __CUDACC__
inline AT_HOST_DEVICE Half::Half(const __half& value) {
  x = *reinterpret_cast<const unsigned short*>(&value);
}
inline AT_HOST_DEVICE Half::operator __half() const {
  return *reinterpret_cast<const __half*>(&x);
}
#endif

// CUDA intrinsics

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
inline __device__ Half __ldg(const Half* ptr) {
    return __ldg(reinterpret_cast<const __half*>(ptr));
}
#endif

/// Arithmetic

inline AT_HOST_DEVICE Half operator+(const Half& a, const Half& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

inline AT_HOST_DEVICE Half operator-(const Half& a, const Half& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

inline AT_HOST_DEVICE Half operator*(const Half& a, const Half& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

inline AT_HOST_DEVICE Half operator/(const Half& a, const Half& b) {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline AT_HOST_DEVICE Half operator-(const Half& a) {
  return -static_cast<float>(a);
}

inline AT_HOST_DEVICE Half& operator+=(Half& a, const Half& b) {
  a = a + b;
  return a;
}

inline AT_HOST_DEVICE Half& operator-=(Half& a, const Half& b) {
  a = a - b;
  return a;
}

inline AT_HOST_DEVICE Half& operator*=(Half& a, const Half& b) {
  a = a * b;
  return a;
}

inline AT_HOST_DEVICE Half& operator/=(Half& a, const Half& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

inline AT_HOST_DEVICE float operator+(Half a, float b) {
  return static_cast<float>(a) + b;
}
inline AT_HOST_DEVICE float operator-(Half a, float b) {
  return static_cast<float>(a) - b;
}
inline AT_HOST_DEVICE float operator*(Half a, float b) {
  return static_cast<float>(a) * b;
}
inline AT_HOST_DEVICE float operator/(Half a, float b) {
  return static_cast<float>(a) / b;
}

inline AT_HOST_DEVICE float operator+(float a, Half b) {
  return a + static_cast<float>(b);
}
inline AT_HOST_DEVICE float operator-(float a, Half b) {
  return a - static_cast<float>(b);
}
inline AT_HOST_DEVICE float operator*(float a, Half b) {
  return a * static_cast<float>(b);
}
inline AT_HOST_DEVICE float operator/(float a, Half b) {
  return a / static_cast<float>(b);
}

inline AT_HOST_DEVICE float& operator+=(float& a, const Half& b) {
  return a += static_cast<float>(b);
}
inline AT_HOST_DEVICE float& operator-=(float& a, const Half& b) {
  return a -= static_cast<float>(b);
}
inline AT_HOST_DEVICE float& operator*=(float& a, const Half& b) {
  return a *= static_cast<float>(b);
}
inline AT_HOST_DEVICE float& operator/=(float& a, const Half& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

inline AT_HOST_DEVICE double operator+(Half a, double b) {
  return static_cast<double>(a) + b;
}
inline AT_HOST_DEVICE double operator-(Half a, double b) {
  return static_cast<double>(a) - b;
}
inline AT_HOST_DEVICE double operator*(Half a, double b) {
  return static_cast<double>(a) * b;
}
inline AT_HOST_DEVICE double operator/(Half a, double b) {
  return static_cast<double>(a) / b;
}

inline AT_HOST_DEVICE double operator+(double a, Half b) {
  return a + static_cast<double>(b);
}
inline AT_HOST_DEVICE double operator-(double a, Half b) {
  return a - static_cast<double>(b);
}
inline AT_HOST_DEVICE double operator*(double a, Half b) {
  return a * static_cast<double>(b);
}
inline AT_HOST_DEVICE double operator/(double a, Half b) {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

inline AT_HOST_DEVICE Half operator+(Half a, int b) {
  return a + static_cast<Half>(b);
}
inline AT_HOST_DEVICE Half operator-(Half a, int b) {
  return a - static_cast<Half>(b);
}
inline AT_HOST_DEVICE Half operator*(Half a, int b) {
  return a * static_cast<Half>(b);
}
inline AT_HOST_DEVICE Half operator/(Half a, int b) {
  return a / static_cast<Half>(b);
}

inline AT_HOST_DEVICE Half operator+(int a, Half b) {
  return static_cast<Half>(a) + b;
}
inline AT_HOST_DEVICE Half operator-(int a, Half b) {
  return static_cast<Half>(a) - b;
}
inline AT_HOST_DEVICE Half operator*(int a, Half b) {
  return static_cast<Half>(a) * b;
}
inline AT_HOST_DEVICE Half operator/(int a, Half b) {
  return static_cast<Half>(a) / b;
}

//// Arithmetic with int64_t

inline AT_HOST_DEVICE Half operator+(Half a, int64_t b) {
  return a + static_cast<Half>(b);
}
inline AT_HOST_DEVICE Half operator-(Half a, int64_t b) {
  return a - static_cast<Half>(b);
}
inline AT_HOST_DEVICE Half operator*(Half a, int64_t b) {
  return a * static_cast<Half>(b);
}
inline AT_HOST_DEVICE Half operator/(Half a, int64_t b) {
  return a / static_cast<Half>(b);
}

inline AT_HOST_DEVICE Half operator+(int64_t a, Half b) {
  return static_cast<Half>(a) + b;
}
inline AT_HOST_DEVICE Half operator-(int64_t a, Half b) {
  return static_cast<Half>(a) - b;
}
inline AT_HOST_DEVICE Half operator*(int64_t a, Half b) {
  return static_cast<Half>(a) * b;
}
inline AT_HOST_DEVICE Half operator/(int64_t a, Half b) {
  return static_cast<Half>(a) / b;
}

/// NOTE: we do not define comparisons directly and instead rely on the implicit
/// conversion from at::Half to float.

} // namespace at

namespace std {

template <>
class numeric_limits<at::Half> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss =
      numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 11;
  static constexpr int digits10 = 3;
  static constexpr int max_digits10 = 5;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;
  static constexpr at::Half min() {
    return at::Half(0x0400, at::Half::from_bits);
  }
  static constexpr at::Half lowest() {
    return at::Half(0xFBFF, at::Half::from_bits);
  }
  static constexpr at::Half max() {
    return at::Half(0x7BFF, at::Half::from_bits);
  }
  static constexpr at::Half epsilon() {
    return at::Half(0x1400, at::Half::from_bits);
  }
  static constexpr at::Half round_error() {
    return at::Half(0x3800, at::Half::from_bits);
  }
  static constexpr at::Half infinity() {
    return at::Half(0x7C00, at::Half::from_bits);
  }
  static constexpr at::Half quiet_NaN() {
    return at::Half(0x7E00, at::Half::from_bits);
  }
  static constexpr at::Half signaling_NaN() {
    return at::Half(0x7D00, at::Half::from_bits);
  }
  static constexpr at::Half denorm_min() {
    return at::Half(0x0001, at::Half::from_bits);
  }
};

} // namespace std
