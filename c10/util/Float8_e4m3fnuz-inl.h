#pragma once

#include <c10/macros/Macros.h>
#include <limits>

C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

namespace c10 {

/// Constructors

C10_HOST_DEVICE inline Float8_e4m3fnuz::Float8_e4m3fnuz(float value)
    : x(detail::fp8e4m3fnuz_from_fp32_value(value)) {}

/// Implicit conversions

C10_HOST_DEVICE inline Float8_e4m3fnuz::operator float() const {
  return detail::fp8e4m3fnuz_to_fp32_value(x);
}

/// Special values helper

C10_HOST_DEVICE inline bool Float8_e4m3fnuz::isnan() const {
  return x == 0b10000000;
}

} // namespace c10

namespace std {

template <>
class numeric_limits<c10::Float8_e4m3fnuz> {
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
  static constexpr int min_exponent = -6;
  static constexpr int min_exponent10 = -1;
  static constexpr int max_exponent = 8;
  static constexpr int max_exponent10 = 2;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before = false;

  static constexpr c10::Float8_e4m3fnuz min() {
    return c10::Float8_e4m3fnuz(0x08, c10::Float8_e4m3fnuz::from_bits());
  }
  static constexpr c10::Float8_e4m3fnuz lowest() {
    return c10::Float8_e4m3fnuz(0xFF, c10::Float8_e4m3fnuz::from_bits());
  }
  static constexpr c10::Float8_e4m3fnuz max() {
    return c10::Float8_e4m3fnuz(0x7F, c10::Float8_e4m3fnuz::from_bits());
  }
  static constexpr c10::Float8_e4m3fnuz epsilon() {
    return c10::Float8_e4m3fnuz(0x28, c10::Float8_e4m3fnuz::from_bits());
  }
  static constexpr c10::Float8_e4m3fnuz round_error() {
    return c10::Float8_e4m3fnuz(0x38, c10::Float8_e4m3fnuz::from_bits());
  }
  static constexpr c10::Float8_e4m3fnuz infinity() {
    // NaN (no infinities)
    return c10::Float8_e4m3fnuz(0x80, c10::Float8_e4m3fnuz::from_bits());
  }
  static constexpr c10::Float8_e4m3fnuz quiet_NaN() {
    return c10::Float8_e4m3fnuz(0x80, c10::Float8_e4m3fnuz::from_bits());
  }
  static constexpr c10::Float8_e4m3fnuz denorm_min() {
    return c10::Float8_e4m3fnuz(0x01, c10::Float8_e4m3fnuz::from_bits());
  }
};

} // namespace std

C10_CLANG_DIAGNOSTIC_POP()
