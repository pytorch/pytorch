#pragma once

#include <c10/macros/Macros.h>
#include <cstring>
#include <limits>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

namespace c10 {

/// Constructors

inline C10_HOST_DEVICE Float8_e4m3::Float8_e4m3(float value) {
  x = detail::fp8e4m3_from_fp32_value(value);
}

/// Implicit conversions

inline C10_HOST_DEVICE Float8_e4m3::operator float() const {
  return detail::fp8e4m3_to_fp32_value(x);
}

/// Arithmetic

inline C10_HOST_DEVICE Float8_e4m3 operator+(const Float8_e4m3& a, const Float8_e4m3& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8_e4m3 operator-(const Float8_e4m3& a, const Float8_e4m3& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8_e4m3 operator*(const Float8_e4m3& a, const Float8_e4m3& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8_e4m3 operator/(const Float8_e4m3& a, const Float8_e4m3& b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8_e4m3 operator-(const Float8_e4m3& a) {
  return -static_cast<float>(a);
}

inline C10_HOST_DEVICE Float8_e4m3& operator+=(Float8_e4m3& a, const Float8_e4m3& b) {
  a = a + b;
  return a;
}

inline C10_HOST_DEVICE Float8_e4m3& operator-=(Float8_e4m3& a, const Float8_e4m3& b) {
  a = a - b;
  return a;
}

inline C10_HOST_DEVICE Float8_e4m3& operator*=(Float8_e4m3& a, const Float8_e4m3& b) {
  a = a * b;
  return a;
}

inline C10_HOST_DEVICE Float8_e4m3& operator/=(Float8_e4m3& a, const Float8_e4m3& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

inline C10_HOST_DEVICE float operator+(Float8_e4m3 a, float b) {
  return static_cast<float>(a) + b;
}
inline C10_HOST_DEVICE float operator-(Float8_e4m3 a, float b) {
  return static_cast<float>(a) - b;
}
inline C10_HOST_DEVICE float operator*(Float8_e4m3 a, float b) {
  return static_cast<float>(a) * b;
}
inline C10_HOST_DEVICE float operator/(Float8_e4m3 a, float b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / b;
}

inline C10_HOST_DEVICE float operator+(float a, Float8_e4m3 b) {
  return a + static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator-(float a, Float8_e4m3 b) {
  return a - static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator*(float a, Float8_e4m3 b) {
  return a * static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator/(float a, Float8_e4m3 b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<float>(b);
}

inline C10_HOST_DEVICE float& operator+=(float& a, const Float8_e4m3& b) {
  return a += static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator-=(float& a, const Float8_e4m3& b) {
  return a -= static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator*=(float& a, const Float8_e4m3& b) {
  return a *= static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator/=(float& a, const Float8_e4m3& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

inline C10_HOST_DEVICE double operator+(Float8_e4m3 a, double b) {
  return static_cast<double>(a) + b;
}
inline C10_HOST_DEVICE double operator-(Float8_e4m3 a, double b) {
  return static_cast<double>(a) - b;
}
inline C10_HOST_DEVICE double operator*(Float8_e4m3 a, double b) {
  return static_cast<double>(a) * b;
}
inline C10_HOST_DEVICE double operator/(Float8_e4m3 a, double b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<double>(a) / b;
}

inline C10_HOST_DEVICE double operator+(double a, Float8_e4m3 b) {
  return a + static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator-(double a, Float8_e4m3 b) {
  return a - static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator*(double a, Float8_e4m3 b) {
  return a * static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator/(double a, Float8_e4m3 b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

inline C10_HOST_DEVICE Float8_e4m3 operator+(Float8_e4m3 a, int b) {
  return a + static_cast<Float8_e4m3>(b);
}
inline C10_HOST_DEVICE Float8_e4m3 operator-(Float8_e4m3 a, int b) {
  return a - static_cast<Float8_e4m3>(b);
}
inline C10_HOST_DEVICE Float8_e4m3 operator*(Float8_e4m3 a, int b) {
  return a * static_cast<Float8_e4m3>(b);
}
inline C10_HOST_DEVICE Float8_e4m3 operator/(Float8_e4m3 a, int b) {
  return a / static_cast<Float8_e4m3>(b);
}

inline C10_HOST_DEVICE Float8_e4m3 operator+(int a, Float8_e4m3 b) {
  return static_cast<Float8_e4m3>(a) + b;
}
inline C10_HOST_DEVICE Float8_e4m3 operator-(int a, Float8_e4m3 b) {
  return static_cast<Float8_e4m3>(a) - b;
}
inline C10_HOST_DEVICE Float8_e4m3 operator*(int a, Float8_e4m3 b) {
  return static_cast<Float8_e4m3>(a) * b;
}
inline C10_HOST_DEVICE Float8_e4m3 operator/(int a, Float8_e4m3 b) {
  return static_cast<Float8_e4m3>(a) / b;
}

//// Arithmetic with int64_t

inline C10_HOST_DEVICE Float8_e4m3 operator+(Float8_e4m3 a, int64_t b) {
  return a + static_cast<Float8_e4m3>(b);
}
inline C10_HOST_DEVICE Float8_e4m3 operator-(Float8_e4m3 a, int64_t b) {
  return a - static_cast<Float8_e4m3>(b);
}
inline C10_HOST_DEVICE Float8_e4m3 operator*(Float8_e4m3 a, int64_t b) {
  return a * static_cast<Float8_e4m3>(b);
}
inline C10_HOST_DEVICE Float8_e4m3 operator/(Float8_e4m3 a, int64_t b) {
  return a / static_cast<Float8_e4m3>(b);
}

inline C10_HOST_DEVICE Float8_e4m3 operator+(int64_t a, Float8_e4m3 b) {
  return static_cast<Float8_e4m3>(a) + b;
}
inline C10_HOST_DEVICE Float8_e4m3 operator-(int64_t a, Float8_e4m3 b) {
  return static_cast<Float8_e4m3>(a) - b;
}
inline C10_HOST_DEVICE Float8_e4m3 operator*(int64_t a, Float8_e4m3 b) {
  return static_cast<Float8_e4m3>(a) * b;
}
inline C10_HOST_DEVICE Float8_e4m3 operator/(int64_t a, Float8_e4m3 b) {
  return static_cast<Float8_e4m3>(a) / b;
}

/// NOTE: we do not define comparisons directly and instead rely on the implicit
/// conversion from c10::Float8_e4m3 to float.

} // namespace c10

namespace std {

template <>
class numeric_limits<c10::Float8_e4m3> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = false;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = false;
  static constexpr auto has_denorm = true;
  static constexpr auto has_denorm_loss = true;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 4;
  static constexpr int digits10 = 0;
  static constexpr int max_digits10 = 3;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -5;
  static constexpr int min_exponent10 = -1;
  static constexpr int max_exponent = 8;
  static constexpr int max_exponent10 = 2;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before = false;

  static constexpr c10::Float8_e4m3 min() {
    return c10::Float8_e4m3(0x08, c10::Float8_e4m3::from_bits());
  }
  static constexpr c10::Float8_e4m3 lowest() {
    return c10::Float8_e4m3(0xFE, c10::Float8_e4m3::from_bits());
  }
  static constexpr c10::Float8_e4m3 max() {
    return c10::Float8_e4m3(0x7E, c10::Float8_e4m3::from_bits());
  }
  static constexpr c10::Float8_e4m3 epsilon() {
    return c10::Float8_e4m3(0x20, c10::Float8_e4m3::from_bits());
  }
  static constexpr c10::Float8_e4m3 round_error() {
    return c10::Float8_e4m3(0x30, c10::Float8_e4m3::from_bits());
  }
  static constexpr c10::Float8_e4m3 quiet_NaN() {
    return c10::Float8_e4m3(0x7F, c10::Float8_e4m3::from_bits());
  }
  static constexpr c10::Float8_e4m3 denorm_min() {
    return c10::Float8_e4m3(0x01, c10::Float8_e4m3::from_bits());
  }
};

} // namespace std

C10_CLANG_DIAGNOSTIC_POP()
