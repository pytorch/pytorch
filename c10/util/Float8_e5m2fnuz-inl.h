#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Float8_fnuz_cvt.h>
#include <cstring>
#include <limits>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

namespace c10 {

/// Constructors

inline C10_HOST_DEVICE Float8_e5m2fnuz::Float8_e5m2fnuz(float value)
    : x(detail::fp8e5m2fnuz_from_fp32_value(value)) {}

/// Implicit conversions

inline C10_HOST_DEVICE Float8_e5m2fnuz::operator float() const {
  return detail::fp8_fnuz_to_fp32_value<5, 2>(x);
}

/// Special values helpers

inline C10_HOST_DEVICE bool Float8_e5m2fnuz::isnan() const {
  return x == 0b10000000;
}

inline C10_HOST_DEVICE bool Float8_e5m2fnuz::isinf() const {
  return false;
}

/// Arithmetic

inline C10_HOST_DEVICE Float8_e5m2fnuz
operator+(const Float8_e5m2fnuz& a, const Float8_e5m2fnuz& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8_e5m2fnuz
operator-(const Float8_e5m2fnuz& a, const Float8_e5m2fnuz& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8_e5m2fnuz
operator*(const Float8_e5m2fnuz& a, const Float8_e5m2fnuz& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8_e5m2fnuz operator/(
    const Float8_e5m2fnuz& a,
    const Float8_e5m2fnuz& b) __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8_e5m2fnuz operator-(const Float8_e5m2fnuz& a) {
  return -static_cast<float>(a);
}

inline C10_HOST_DEVICE Float8_e5m2fnuz& operator+=(
    Float8_e5m2fnuz& a,
    const Float8_e5m2fnuz& b) {
  a = a + b;
  return a;
}

inline C10_HOST_DEVICE Float8_e5m2fnuz& operator-=(
    Float8_e5m2fnuz& a,
    const Float8_e5m2fnuz& b) {
  a = a - b;
  return a;
}

inline C10_HOST_DEVICE Float8_e5m2fnuz& operator*=(
    Float8_e5m2fnuz& a,
    const Float8_e5m2fnuz& b) {
  a = a * b;
  return a;
}

inline C10_HOST_DEVICE Float8_e5m2fnuz& operator/=(
    Float8_e5m2fnuz& a,
    const Float8_e5m2fnuz& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

inline C10_HOST_DEVICE float operator+(Float8_e5m2fnuz a, float b) {
  return static_cast<float>(a) + b;
}
inline C10_HOST_DEVICE float operator-(Float8_e5m2fnuz a, float b) {
  return static_cast<float>(a) - b;
}
inline C10_HOST_DEVICE float operator*(Float8_e5m2fnuz a, float b) {
  return static_cast<float>(a) * b;
}
inline C10_HOST_DEVICE float operator/(Float8_e5m2fnuz a, float b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / b;
}

inline C10_HOST_DEVICE float operator+(float a, Float8_e5m2fnuz b) {
  return a + static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator-(float a, Float8_e5m2fnuz b) {
  return a - static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator*(float a, Float8_e5m2fnuz b) {
  return a * static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator/(float a, Float8_e5m2fnuz b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<float>(b);
}

inline C10_HOST_DEVICE float& operator+=(float& a, const Float8_e5m2fnuz& b) {
  return a += static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator-=(float& a, const Float8_e5m2fnuz& b) {
  return a -= static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator*=(float& a, const Float8_e5m2fnuz& b) {
  return a *= static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator/=(float& a, const Float8_e5m2fnuz& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

inline C10_HOST_DEVICE double operator+(Float8_e5m2fnuz a, double b) {
  return static_cast<double>(a) + b;
}
inline C10_HOST_DEVICE double operator-(Float8_e5m2fnuz a, double b) {
  return static_cast<double>(a) - b;
}
inline C10_HOST_DEVICE double operator*(Float8_e5m2fnuz a, double b) {
  return static_cast<double>(a) * b;
}
inline C10_HOST_DEVICE double operator/(Float8_e5m2fnuz a, double b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<double>(a) / b;
}

inline C10_HOST_DEVICE double operator+(double a, Float8_e5m2fnuz b) {
  return a + static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator-(double a, Float8_e5m2fnuz b) {
  return a - static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator*(double a, Float8_e5m2fnuz b) {
  return a * static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator/(double a, Float8_e5m2fnuz b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

inline C10_HOST_DEVICE Float8_e5m2fnuz operator+(Float8_e5m2fnuz a, int b) {
  return a + static_cast<Float8_e5m2fnuz>(b);
}
inline C10_HOST_DEVICE Float8_e5m2fnuz operator-(Float8_e5m2fnuz a, int b) {
  return a - static_cast<Float8_e5m2fnuz>(b);
}
inline C10_HOST_DEVICE Float8_e5m2fnuz operator*(Float8_e5m2fnuz a, int b) {
  return a * static_cast<Float8_e5m2fnuz>(b);
}
inline C10_HOST_DEVICE Float8_e5m2fnuz operator/(Float8_e5m2fnuz a, int b) {
  return a / static_cast<Float8_e5m2fnuz>(b);
}

inline C10_HOST_DEVICE Float8_e5m2fnuz operator+(int a, Float8_e5m2fnuz b) {
  return static_cast<Float8_e5m2fnuz>(a) + b;
}
inline C10_HOST_DEVICE Float8_e5m2fnuz operator-(int a, Float8_e5m2fnuz b) {
  return static_cast<Float8_e5m2fnuz>(a) - b;
}
inline C10_HOST_DEVICE Float8_e5m2fnuz operator*(int a, Float8_e5m2fnuz b) {
  return static_cast<Float8_e5m2fnuz>(a) * b;
}
inline C10_HOST_DEVICE Float8_e5m2fnuz operator/(int a, Float8_e5m2fnuz b) {
  return static_cast<Float8_e5m2fnuz>(a) / b;
}

//// Arithmetic with int64_t

inline C10_HOST_DEVICE Float8_e5m2fnuz operator+(Float8_e5m2fnuz a, int64_t b) {
  return a + static_cast<Float8_e5m2fnuz>(b);
}
inline C10_HOST_DEVICE Float8_e5m2fnuz operator-(Float8_e5m2fnuz a, int64_t b) {
  return a - static_cast<Float8_e5m2fnuz>(b);
}
inline C10_HOST_DEVICE Float8_e5m2fnuz operator*(Float8_e5m2fnuz a, int64_t b) {
  return a * static_cast<Float8_e5m2fnuz>(b);
}
inline C10_HOST_DEVICE Float8_e5m2fnuz operator/(Float8_e5m2fnuz a, int64_t b) {
  return a / static_cast<Float8_e5m2fnuz>(b);
}

inline C10_HOST_DEVICE Float8_e5m2fnuz operator+(int64_t a, Float8_e5m2fnuz b) {
  return static_cast<Float8_e5m2fnuz>(a) + b;
}
inline C10_HOST_DEVICE Float8_e5m2fnuz operator-(int64_t a, Float8_e5m2fnuz b) {
  return static_cast<Float8_e5m2fnuz>(a) - b;
}
inline C10_HOST_DEVICE Float8_e5m2fnuz operator*(int64_t a, Float8_e5m2fnuz b) {
  return static_cast<Float8_e5m2fnuz>(a) * b;
}
inline C10_HOST_DEVICE Float8_e5m2fnuz operator/(int64_t a, Float8_e5m2fnuz b) {
  return static_cast<Float8_e5m2fnuz>(a) / b;
}

/// NOTE: we do not define comparisons directly and instead rely on the implicit
/// conversion from c10::Float8_e5m2fnuz to float.

} // namespace c10

namespace std {

template <>
class numeric_limits<c10::Float8_e5m2fnuz> {
 public:
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_specialized = true;
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
  static constexpr int digits = 3;
  static constexpr int digits10 = 0;
  static constexpr int max_digits10 = 2;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -14;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;

  static constexpr c10::Float8_e5m2fnuz min() {
    return c10::Float8_e5m2fnuz(0x04, c10::Float8_e5m2fnuz::from_bits());
  }
  static constexpr c10::Float8_e5m2fnuz max() {
    return c10::Float8_e5m2fnuz(0x7F, c10::Float8_e5m2fnuz::from_bits());
  }
  static constexpr c10::Float8_e5m2fnuz lowest() {
    return c10::Float8_e5m2fnuz(0xFF, c10::Float8_e5m2fnuz::from_bits());
  }
  static constexpr c10::Float8_e5m2fnuz epsilon() {
    return c10::Float8_e5m2fnuz(0x34, c10::Float8_e5m2fnuz::from_bits());
  }
  static constexpr c10::Float8_e5m2fnuz round_error() {
    return c10::Float8_e5m2fnuz(0x38, c10::Float8_e5m2fnuz::from_bits());
  }
  static constexpr c10::Float8_e5m2fnuz infinity() {
    return c10::Float8_e5m2fnuz(0x80, c10::Float8_e5m2fnuz::from_bits());
  }
  static constexpr c10::Float8_e5m2fnuz denorm_min() {
    return c10::Float8_e5m2fnuz(0x01, c10::Float8_e5m2fnuz::from_bits());
  }
};

} // namespace std

C10_CLANG_DIAGNOSTIC_POP()
