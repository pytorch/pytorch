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

inline AT_HOSTDEVICE Half::Half(float value) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  x = __half_as_short(__float2half(value));
#else
  x = detail::float2halfbits(value);
#endif
}

/// Implicit conversions

inline AT_HOSTDEVICE Half::operator float() const {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __half2float(*reinterpret_cast<const __half*>(&x));
#else
  return detail::halfbits2float(x);
#endif
}

#ifdef __CUDACC__
inline AT_HOSTDEVICE Half::Half(const __half& value) {
  x = *reinterpret_cast<const unsigned short*>(&value);
}
inline AT_HOSTDEVICE Half::operator __half() const {
  return *reinterpret_cast<const __half*>(&x);
}
#endif

/// Arithmetic

inline AT_HOSTDEVICE Half operator+(const Half& a, const Half& b) {
  return (float)a + (float)b;
}

inline AT_HOSTDEVICE Half operator-(const Half& a, const Half& b) {
  return (float)a - (float)b;
}

inline AT_HOSTDEVICE Half operator*(const Half& a, const Half& b) {
  return (float)a * (float)b;
}

inline AT_HOSTDEVICE Half operator/(const Half& a, const Half& b) {
  return (float)a / (float)b;
}

inline AT_HOSTDEVICE Half operator-(const Half& a) {
  return -(float)a;
}

inline AT_HOSTDEVICE Half& operator+=(Half& a, const Half& b) {
  a = a + b;
  return a;
}

inline AT_HOSTDEVICE Half& operator-=(Half& a, const Half& b) {
  a = a - b;
  return a;
}

inline AT_HOSTDEVICE Half& operator*=(Half& a, const Half& b) {
  a = a * b;
  return a;
}

inline AT_HOSTDEVICE Half& operator/=(Half& a, const Half& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

inline AT_HOSTDEVICE float operator+(Half a, float b) {
  return (float)a + b;
}
inline AT_HOSTDEVICE float operator-(Half a, float b) {
  return (float)a - b;
}
inline AT_HOSTDEVICE float operator*(Half a, float b) {
  return (float)a * b;
}
inline AT_HOSTDEVICE float operator/(Half a, float b) {
  return (float)a / b;
}

inline AT_HOSTDEVICE float operator+(float a, Half b) {
  return a + (float)b;
}
inline AT_HOSTDEVICE float operator-(float a, Half b) {
  return a - (float)b;
}
inline AT_HOSTDEVICE float operator*(float a, Half b) {
  return a * (float)b;
}
inline AT_HOSTDEVICE float operator/(float a, Half b) {
  return a / (float)b;
}

inline AT_HOSTDEVICE float& operator+=(float& a, const Half& b) {
  return a += (float)b;
}
inline AT_HOSTDEVICE float& operator-=(float& a, const Half& b) {
  return a -= (float)b;
}
inline AT_HOSTDEVICE float& operator*=(float& a, const Half& b) {
  return a *= (float)b;
}
inline AT_HOSTDEVICE float& operator/=(float& a, const Half& b) {
  return a /= (float)b;
}

/// Arithmetic with doubles

inline AT_HOSTDEVICE double operator+(Half a, double b) {
  return (double)a + b;
}
inline AT_HOSTDEVICE double operator-(Half a, double b) {
  return (double)a - b;
}
inline AT_HOSTDEVICE double operator*(Half a, double b) {
  return (double)a * b;
}
inline AT_HOSTDEVICE double operator/(Half a, double b) {
  return (double)a / b;
}

inline AT_HOSTDEVICE double operator+(double a, Half b) {
  return a + (double)b;
}
inline AT_HOSTDEVICE double operator-(double a, Half b) {
  return a - (double)b;
}
inline AT_HOSTDEVICE double operator*(double a, Half b) {
  return a * (double)b;
}
inline AT_HOSTDEVICE double operator/(double a, Half b) {
  return a / (double)b;
}

/// Arithmetic with ints

inline AT_HOSTDEVICE Half operator+(Half a, int b) {
  return a + (Half)b;
}
inline AT_HOSTDEVICE Half operator-(Half a, int b) {
  return a - (Half)b;
}
inline AT_HOSTDEVICE Half operator*(Half a, int b) {
  return a * (Half)b;
}
inline AT_HOSTDEVICE Half operator/(Half a, int b) {
  return a / (Half)b;
}

inline AT_HOSTDEVICE Half operator+(int a, Half b) {
  return (Half)a + b;
}
inline AT_HOSTDEVICE Half operator-(int a, Half b) {
  return (Half)a - b;
}
inline AT_HOSTDEVICE Half operator*(int a, Half b) {
  return (Half)a * b;
}
inline AT_HOSTDEVICE Half operator/(int a, Half b) {
  return (Half)a / b;
}

/// NOTE: we do not define comparisons directly and instead rely on the implicit
/// conversion from at::Half to float.

} // namespace at

namespace std {
  
// specializing std for common mathematical functions to be used with at::Half.
// For most of this functions the cmath header/math_functions.hpp in cuda 
// would call the corresponding float specialization. 
// However, here we are calling the float functions
// directly to save one API call, like it's been done in the old THCNumerics.cuh
// Also note that some of these functions might not have been used in the library
// at all. However, keeping all the function definitions to provide backward
// compatibility

inline AT_HOSTDEVICE at::Half lgamma(at::Half a) {
  return (at::Half)lgammaf((float)a);
}

inline AT_HOSTDEVICE at::Half exp(at::Half a) {
  return (at::Half)expf((float)a);
}

inline AT_HOSTDEVICE at::Half log(at::Half a) {
  return (at::Half)logf((float)a);
}

inline AT_HOSTDEVICE at::Half log10(at::Half a) {
  return (at::Half)log10f((float)a); 
}

inline AT_HOSTDEVICE at::Half log1p(at::Half a) {
  return (at::Half)log1pf((float)a);
}

inline AT_HOSTDEVICE at::Half log2(at::Half a) {
  return (at::Half)log2f(a);
}

inline AT_HOSTDEVICE at::Half expm1(at::Half a) {
  return (at::Half)expm1f((float)a); 
}

inline AT_HOSTDEVICE at::Half cos(at::Half a) {
  return (at::Half)cosf((float)a);
}

inline AT_HOSTDEVICE at::Half sin(at::Half a) {
  return (at::Half)sinf((float)a);
}

inline AT_HOSTDEVICE at::Half sqrt(at::Half a) {
  return (at::Half)sqrtf((float)a);
}

inline AT_HOSTDEVICE at::Half ceil(at::Half a) {
  return (at::Half)ceilf((float)a);
}

inline AT_HOSTDEVICE at::Half floor(at::Half a) {
  return (at::Half)floorf((float)a);
}

inline AT_HOSTDEVICE at::Half trunc(at::Half a) {
  return (at::Half)truncf((float)a);
}

inline AT_HOSTDEVICE at::Half acos(at::Half a) {
  return (at::Half)acosf((float)a);
}

inline AT_HOSTDEVICE at::Half cosh(at::Half a) {
  return (at::Half)coshf((float)a);
}

inline AT_HOSTDEVICE at::Half acosh(at::Half a) {
  return (at::Half)acoshf((float)a);
}

inline AT_HOSTDEVICE at::Half asin(at::Half a) {
  return (at::Half)asinf((float)a);
}

inline AT_HOSTDEVICE at::Half sinh(at::Half a) {
  return (at::Half)sinhf((float)a);
}

inline AT_HOSTDEVICE at::Half asinh(at::Half a) {
  return (at::Half)asinhf((float)a);
}

inline AT_HOSTDEVICE at::Half tan(at::Half a) {
  return (at::Half)tanf((float)a);
}

inline AT_HOSTDEVICE at::Half atan(at::Half a) {
  return (at::Half)atanf((float)a);
}

inline AT_HOSTDEVICE at::Half tanh(at::Half a) {
  return (at::Half)tanhf((float)a);
}

inline AT_HOSTDEVICE at::Half erf(at::Half a) {
  return (at::Half)erff((float)a);
}

inline AT_HOSTDEVICE at::Half erfc(at::Half a) {
  return (at::Half)erfcf((float)a);
}

inline AT_HOSTDEVICE at::Half abs(at::Half a) {
  return (at::Half)fabs((float)a);
}

inline AT_HOSTDEVICE at::Half round(at::Half a) {
  return (at::Half)roundf((float)a);
}

inline AT_HOSTDEVICE at::Half pow(at::Half a, at::Half b) {
  return (at::Half)powf((float)a, (float)b);
}

inline AT_HOSTDEVICE at::Half atan2(at::Half a, at::Half b) {
  return (at::Half)atan2f((float)a, (float)b);
}

inline AT_HOSTDEVICE bool isnan(at::Half a) {
  return isnan((float)a);
}

inline AT_HOSTDEVICE bool isinf(at::Half a) {
  return isinf((float)a);
}

// std specialization for numeric limits

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
