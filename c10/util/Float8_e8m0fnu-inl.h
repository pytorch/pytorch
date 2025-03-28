#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/floating_point_utils.h>
#include <cstring>
#include <limits>

// TODO(#146647): Can we remove the below warning?
C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

namespace c10 {

/// Constructors

inline C10_HOST_DEVICE Float8_e8m0fnu::Float8_e8m0fnu(float value)
    : x(detail::fp8e8m0fnu_from_fp32_value(value)) {}

/// Implicit conversions

inline C10_HOST_DEVICE Float8_e8m0fnu::operator float() const {
  // TODO(#146647): maybe rewrite without control flow

  // if exponent is zero, need to special case to return 2^-127 instead of zero
  if (x == 0) {
    return c10::detail::fp32_from_bits(0x00400000);
  }

  // if exponent is NaN, need to special case to return properly encoded NaN
  if (isnan()) {
    return c10::detail::fp32_from_bits(0x7f800001);
  }

  // leave sign at 0, set the exponent bits, leave stored mantissa at 0
  uint32_t res = x << 23;

  return c10::detail::fp32_from_bits(res);
}

/// Special values helper

inline C10_HOST_DEVICE bool Float8_e8m0fnu::isnan() const {
  return x == 0b11111111;
}

/// NOTE: we do not define comparisons directly and instead rely on the implicit
/// conversion from c10::Float8_e8m0fnu to float.

} // namespace c10

namespace std {

template <>
class numeric_limits<c10::Float8_e8m0fnu> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = false;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = false;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = false;
  static constexpr auto has_denorm = false;
  static constexpr auto has_denorm_loss = false;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 1;
  static constexpr int digits10 = 0;
  static constexpr int max_digits10 = 1; // just a 2!
  static constexpr int radix = 2;
  static constexpr int min_exponent = -126;
  static constexpr int min_exponent10 = -38;
  static constexpr int max_exponent = 128;
  static constexpr int max_exponent10 = 38;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before = false;

  static constexpr c10::Float8_e8m0fnu min() {
    // 2^-127
    return c10::Float8_e8m0fnu(0b00000000, c10::Float8_e8m0fnu::from_bits());
  }
  static constexpr c10::Float8_e8m0fnu lowest() {
    // 2^-127
    return c10::Float8_e8m0fnu(0b00000000, c10::Float8_e8m0fnu::from_bits());
  }
  static constexpr c10::Float8_e8m0fnu max() {
    // 254 biased, which is 127 unbiased, so 2^127
    return c10::Float8_e8m0fnu(0b11111110, c10::Float8_e8m0fnu::from_bits());
  }
  static constexpr c10::Float8_e8m0fnu epsilon() {
    // according to https://en.cppreference.com/w/cpp/types/numeric_limits, this
    // is "the difference between 1.0 and the next representable value of the
    // given floating-point type". The next representable value is 2.0, so the
    // difference is 1.0 which is 2^0. 0 unbiased is 127 biased.
    return c10::Float8_e8m0fnu(0b01111111, c10::Float8_e8m0fnu::from_bits());
  }
  static constexpr c10::Float8_e8m0fnu round_error() {
    // 0.5 in float, which is 2^-1, and -1 + 127 = 126
    return c10::Float8_e8m0fnu(0b01111110, c10::Float8_e8m0fnu::from_bits());
  }
  static constexpr c10::Float8_e8m0fnu quiet_NaN() {
    return c10::Float8_e8m0fnu(0b11111111, c10::Float8_e8m0fnu::from_bits());
  }
};

} // namespace std

C10_CLANG_DIAGNOSTIC_POP()
