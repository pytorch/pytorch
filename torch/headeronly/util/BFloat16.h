#pragma once

// Defines the bloat16 type (brain floating-point). This representation uses
// 1 bit for the sign, 8 bits for the exponent and 7 bits for the mantissa.

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/bit_cast.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <ostream>

#if defined(__CUDACC__) && !defined(USE_ROCM)
#include <cuda_bf16.h>
#endif

#if defined(CL_SYCL_LANGUAGE_VERSION)
#include <CL/sycl.hpp> // for SYCL 1.2.1
#elif defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp> // for SYCL 2020
#endif

namespace torch::headeronly {

struct alignas(2) BFloat16 {
  uint16_t x;

  // HIP wants __host__ __device__ tag, CUDA does not
#if defined(USE_ROCM) && defined(__HIPCC__)
  C10_HOST_DEVICE BFloat16() = default;
#else
  BFloat16() = default;
#endif

  struct from_bits_t {};
  static constexpr C10_HOST_DEVICE from_bits_t from_bits() {
    return from_bits_t();
  }

  constexpr C10_HOST_DEVICE BFloat16(unsigned short bits, from_bits_t)
      : x(bits) {}
  /* implicit */ inline C10_HOST_DEVICE BFloat16(float value);
  inline C10_HOST_DEVICE operator float() const;

#if defined(__CUDACC__) && !defined(USE_ROCM)
  inline C10_HOST_DEVICE BFloat16(const __nv_bfloat16& value);
  explicit inline C10_HOST_DEVICE operator __nv_bfloat16() const;
#endif

#if defined(SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS)
  inline C10_HOST_DEVICE BFloat16(const sycl::ext::oneapi::bfloat16& value);
  explicit inline C10_HOST_DEVICE operator sycl::ext::oneapi::bfloat16() const;
#endif
};

inline std::ostream& operator<<(std::ostream& out, const BFloat16& value) {
  out << (float)value;
  return out;
}

namespace detail {
inline C10_HOST_DEVICE float f32_from_bits(uint16_t src) {
  float res = 0;
  uint32_t tmp = src;
  tmp <<= 16;

#if defined(USE_ROCM) && defined(__HIPCC__)
  float* tempRes;

  // We should be using memcpy in order to respect the strict aliasing rule
  // but it fails in the HIP environment.
  tempRes = reinterpret_cast<float*>(&tmp);
  res = *tempRes;
#else
  std::memcpy(&res, &tmp, sizeof(tmp));
#endif

  return res;
}

inline C10_HOST_DEVICE uint16_t bits_from_f32(float src) {
  uint32_t res = 0;

#if defined(USE_ROCM) && defined(__HIPCC__)
  // We should be using memcpy in order to respect the strict aliasing rule
  // but it fails in the HIP environment.
  uint32_t* tempRes = reinterpret_cast<uint32_t*>(&src);
  res = *tempRes;
#else
  std::memcpy(&res, &src, sizeof(res));
#endif

  return res >> 16;
}

inline C10_HOST_DEVICE uint16_t round_to_nearest_even(float src) {
#if defined(USE_ROCM) && defined(__HIPCC__)
  if (src != src) {
#elif defined(_MSC_VER)
  if (isnan(src)) {
#else
  if (std::isnan(src)) {
#endif
    return UINT16_C(0x7FC0);
  } else {
    const uint32_t U32 = c10::bit_cast<uint32_t>(src);
    uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
    return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
  }
}

} // namespace detail

//-------- the following is copied from c10/util/BFloat16-inl.h ---------//
C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

/// Constructors
inline C10_HOST_DEVICE BFloat16::BFloat16(float value)
    :
#if defined(__CUDACC__) && !defined(USE_ROCM) && defined(__CUDA_ARCH__) && \
    __CUDA_ARCH__ >= 800
      x(__bfloat16_as_ushort(__float2bfloat16(value)))
#elif defined(__SYCL_DEVICE_ONLY__) && \
    defined(SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS)
      x(c10::bit_cast<uint16_t>(sycl::ext::oneapi::bfloat16(value)))
#else
      // RNE by default
      x(detail::round_to_nearest_even(value))
#endif
{
}

/// Implicit conversions
inline C10_HOST_DEVICE BFloat16::operator float() const {
#if defined(__CUDACC__) && !defined(USE_ROCM)
  return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&x));
#elif defined(__SYCL_DEVICE_ONLY__) && \
    defined(SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS)
  return float(*reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(&x));
#else
  return detail::f32_from_bits(x);
#endif
}

#if defined(__CUDACC__) && !defined(USE_ROCM)
inline C10_HOST_DEVICE BFloat16::BFloat16(const __nv_bfloat16& value) {
  x = *reinterpret_cast<const unsigned short*>(&value);
}
inline C10_HOST_DEVICE BFloat16::operator __nv_bfloat16() const {
  return *reinterpret_cast<const __nv_bfloat16*>(&x);
}
#endif

#if defined(SYCL_EXT_ONEAPI_BFLOAT16_MATH_FUNCTIONS)
inline C10_HOST_DEVICE BFloat16::BFloat16(
    const sycl::ext::oneapi::bfloat16& value) {
  x = *reinterpret_cast<const unsigned short*>(&value);
}
inline C10_HOST_DEVICE BFloat16::operator sycl::ext::oneapi::bfloat16() const {
  return *reinterpret_cast<const sycl::ext::oneapi::bfloat16*>(&x);
}
#endif

// CUDA intrinsics

#if defined(__CUDACC__) || defined(__HIPCC__)
inline C10_DEVICE BFloat16 __ldg(const BFloat16* ptr) {
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __ldg(reinterpret_cast<const __nv_bfloat16*>(ptr));
#else
  return *ptr;
#endif
}
#endif

/// Arithmetic

inline C10_HOST_DEVICE BFloat16
operator+(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

inline C10_HOST_DEVICE BFloat16
operator-(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

inline C10_HOST_DEVICE BFloat16
operator*(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

inline C10_HOST_DEVICE BFloat16 operator/(const BFloat16& a, const BFloat16& b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline C10_HOST_DEVICE BFloat16 operator-(const BFloat16& a) {
  return -static_cast<float>(a);
}

inline C10_HOST_DEVICE BFloat16& operator+=(BFloat16& a, const BFloat16& b) {
  a = a + b;
  return a;
}

inline C10_HOST_DEVICE BFloat16& operator-=(BFloat16& a, const BFloat16& b) {
  a = a - b;
  return a;
}

inline C10_HOST_DEVICE BFloat16& operator*=(BFloat16& a, const BFloat16& b) {
  a = a * b;
  return a;
}

inline C10_HOST_DEVICE BFloat16& operator/=(BFloat16& a, const BFloat16& b) {
  a = a / b;
  return a;
}

inline C10_HOST_DEVICE BFloat16& operator|(BFloat16& a, const BFloat16& b) {
  a.x = a.x | b.x;
  return a;
}

inline C10_HOST_DEVICE BFloat16& operator^(BFloat16& a, const BFloat16& b) {
  a.x = a.x ^ b.x;
  return a;
}

inline C10_HOST_DEVICE BFloat16& operator&(BFloat16& a, const BFloat16& b) {
  a.x = a.x & b.x;
  return a;
}

/// Arithmetic with floats

inline C10_HOST_DEVICE float operator+(BFloat16 a, float b) {
  return static_cast<float>(a) + b;
}
inline C10_HOST_DEVICE float operator-(BFloat16 a, float b) {
  return static_cast<float>(a) - b;
}
inline C10_HOST_DEVICE float operator*(BFloat16 a, float b) {
  return static_cast<float>(a) * b;
}
inline C10_HOST_DEVICE float operator/(BFloat16 a, float b) {
  return static_cast<float>(a) / b;
}

inline C10_HOST_DEVICE float operator+(float a, BFloat16 b) {
  return a + static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator-(float a, BFloat16 b) {
  return a - static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator*(float a, BFloat16 b) {
  return a * static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator/(float a, BFloat16 b) {
  return a / static_cast<float>(b);
}

inline C10_HOST_DEVICE float& operator+=(float& a, const BFloat16& b) {
  return a += static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator-=(float& a, const BFloat16& b) {
  return a -= static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator*=(float& a, const BFloat16& b) {
  return a *= static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator/=(float& a, const BFloat16& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

inline C10_HOST_DEVICE double operator+(BFloat16 a, double b) {
  return static_cast<double>(a) + b;
}
inline C10_HOST_DEVICE double operator-(BFloat16 a, double b) {
  return static_cast<double>(a) - b;
}
inline C10_HOST_DEVICE double operator*(BFloat16 a, double b) {
  return static_cast<double>(a) * b;
}
inline C10_HOST_DEVICE double operator/(BFloat16 a, double b) {
  return static_cast<double>(a) / b;
}

inline C10_HOST_DEVICE double operator+(double a, BFloat16 b) {
  return a + static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator-(double a, BFloat16 b) {
  return a - static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator*(double a, BFloat16 b) {
  return a * static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator/(double a, BFloat16 b) {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

inline C10_HOST_DEVICE BFloat16 operator+(BFloat16 a, int b) {
  return a + static_cast<BFloat16>(b);
}
inline C10_HOST_DEVICE BFloat16 operator-(BFloat16 a, int b) {
  return a - static_cast<BFloat16>(b);
}
inline C10_HOST_DEVICE BFloat16 operator*(BFloat16 a, int b) {
  return a * static_cast<BFloat16>(b);
}
inline C10_HOST_DEVICE BFloat16 operator/(BFloat16 a, int b) {
  return a / static_cast<BFloat16>(b);
}

inline C10_HOST_DEVICE BFloat16 operator+(int a, BFloat16 b) {
  return static_cast<BFloat16>(a) + b;
}
inline C10_HOST_DEVICE BFloat16 operator-(int a, BFloat16 b) {
  return static_cast<BFloat16>(a) - b;
}
inline C10_HOST_DEVICE BFloat16 operator*(int a, BFloat16 b) {
  return static_cast<BFloat16>(a) * b;
}
inline C10_HOST_DEVICE BFloat16 operator/(int a, BFloat16 b) {
  return static_cast<BFloat16>(a) / b;
}

//// Arithmetic with int64_t

inline C10_HOST_DEVICE BFloat16 operator+(BFloat16 a, int64_t b) {
  return a + static_cast<BFloat16>(b);
}
inline C10_HOST_DEVICE BFloat16 operator-(BFloat16 a, int64_t b) {
  return a - static_cast<BFloat16>(b);
}
inline C10_HOST_DEVICE BFloat16 operator*(BFloat16 a, int64_t b) {
  return a * static_cast<BFloat16>(b);
}
inline C10_HOST_DEVICE BFloat16 operator/(BFloat16 a, int64_t b) {
  return a / static_cast<BFloat16>(b);
}

inline C10_HOST_DEVICE BFloat16 operator+(int64_t a, BFloat16 b) {
  return static_cast<BFloat16>(a) + b;
}
inline C10_HOST_DEVICE BFloat16 operator-(int64_t a, BFloat16 b) {
  return static_cast<BFloat16>(a) - b;
}
inline C10_HOST_DEVICE BFloat16 operator*(int64_t a, BFloat16 b) {
  return static_cast<BFloat16>(a) * b;
}
inline C10_HOST_DEVICE BFloat16 operator/(int64_t a, BFloat16 b) {
  return static_cast<BFloat16>(a) / b;
}

// Overloading < and > operators, because std::max and std::min use them.

inline C10_HOST_DEVICE bool operator>(BFloat16& lhs, BFloat16& rhs) {
  return float(lhs) > float(rhs);
}

inline C10_HOST_DEVICE bool operator<(BFloat16& lhs, BFloat16& rhs) {
  return float(lhs) < float(rhs);
}

C10_CLANG_DIAGNOSTIC_POP()
} // namespace torch::headeronly

namespace c10 {

namespace detail {
  using torch::headeronly::detail::f32_from_bits;
  using torch::headeronly::detail::bits_from_f32;
  using torch::headeronly::detail::round_to_nearest_even;
} // namespace detail

  using torch::headeronly::BFloat16;
  using torch::headeronly::operator+;
  using torch::headeronly::operator-;
  using torch::headeronly::operator*;
  using torch::headeronly::operator/;
  using torch::headeronly::operator+=;
  using torch::headeronly::operator-=;
  using torch::headeronly::operator*=;
  using torch::headeronly::operator/=;
  using torch::headeronly::operator<;
  using torch::headeronly::operator>;
  using torch::headeronly::operator<<;
} // namespace c10


namespace std {

template <>
class numeric_limits<c10::BFloat16> {
 public:
  static constexpr bool is_signed = true;
  static constexpr bool is_specialized = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss =
      numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 8;
  static constexpr int digits10 = 2;
  static constexpr int max_digits10 = 4;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -125;
  static constexpr int min_exponent10 = -37;
  static constexpr int max_exponent = 128;
  static constexpr int max_exponent10 = 38;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;

  static constexpr c10::BFloat16 min() {
    return c10::BFloat16(0x0080, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 lowest() {
    return c10::BFloat16(0xFF7F, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 max() {
    return c10::BFloat16(0x7F7F, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 epsilon() {
    return c10::BFloat16(0x3C00, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 round_error() {
    return c10::BFloat16(0x3F00, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 infinity() {
    return c10::BFloat16(0x7F80, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 quiet_NaN() {
    return c10::BFloat16(0x7FC0, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 signaling_NaN() {
    return c10::BFloat16(0x7F80, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 denorm_min() {
    return c10::BFloat16(0x0001, c10::BFloat16::from_bits());
  }
};

} // namespace std
