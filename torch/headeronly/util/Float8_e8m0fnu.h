#pragma once

/// Defines the Float8_e8m0fnu type (8-bit floating-point) including
/// conversions to standard C types
/// Binary configuration :
/// eeeeeeee
/// no sign bits
/// 8 exponent bits
/// no mantissa bits
///
/// This is the E8M0 dtype from the OCP MX format spec
/// (https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf,
/// Section 5.4.1)

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/floating_point_utils.h>

// TODO(#146647): do we need to special case OPENCL?
#if defined(__cplusplus)
#include <cstdint>
#elif !defined(__OPENCL_VERSION__)
#include <math.h>
#include <stdint.h>
#endif

#include <iosfwd>
#include <limits>
#include <ostream>

namespace c10 {

struct alignas(1) Float8_e8m0fnu {
  uint8_t x;

  struct from_bits_t {};
  C10_HOST_DEVICE static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  Float8_e8m0fnu() = default;

  constexpr C10_HOST_DEVICE Float8_e8m0fnu(uint8_t bits, from_bits_t)
      : x(bits) {}
  inline C10_HOST_DEVICE Float8_e8m0fnu(float value);
  inline C10_HOST_DEVICE operator float() const;
  inline C10_HOST_DEVICE bool isnan() const;
};

inline std::ostream& operator<<(
    std::ostream& out,
    const Float8_e8m0fnu& value) {
  out << (float)value;
  return out;
}

namespace detail {
/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 e8m0fnu format, in bit representation.
 */
inline C10_HOST_DEVICE uint8_t fp8e8m0fnu_from_fp32_value(float f) {
  // TODO(#146647): maybe rewrite without control flow

  uint32_t f_bits = c10::detail::fp32_to_bits(f);

  // extract the exponent
  uint32_t exponent = (f_bits >> 23) & 0b11111111;

  // special case float32 NaN and +-inf to map to e8m0 nan
  if (exponent == 0b11111111) {
    return exponent;
  }

  // next, we use guard, round, sticky bits and the LSB to implement round to
  // nearest, with ties to even

  // guard bit - bit 23, or 22 zero-indexed
  uint8_t g = (f_bits & 0x400000) > 0;
  // round bit - bit 22, or 21 zero-indexed
  uint8_t r = (f_bits & 0x200000) > 0;
  // sticky bit - bits 21 to 1, or 20 to 0 zero-indexed
  uint8_t s = (f_bits & 0x1FFFFF) > 0;
  // in casting to e8m0, LSB is the implied mantissa bit. It equals to 0 if the
  // original float32 is denormal, and to 1 if the original float32 is normal.
  uint8_t lsb = exponent > 0;

  // implement the RNE logic
  bool round_up = false;

  // if g == 0, round down (no-op)
  if (g == 1) {
    if ((r == 1) || (s == 1)) {
      // round up
      round_up = true;
    } else {
      if (lsb == 1) {
        // round up
        round_up = true;
      }
      // if lsb == 0, round down (no-op)
    }
  }

  if (round_up) {
    // adjust exponent
    // note that if exponent was 255 we would have already returned earlier, so
    // we know we can add one safely without running out of bounds
    exponent++;
  }

  return exponent;
}

} // namespace detail

//------- the below is from c10/util/Float8_e8m0fnu-inl.h  ------//
// TODO(#146647): Can we remove the below warning?
C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

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
C10_CLANG_DIAGNOSTIC_POP()

} // namespace c10

namespace torch::headeronly {
using c10::Float8_e8m0fnu;
using c10::operator<<;

namespace detail {
using c10::detail::fp8e8m0fnu_from_fp32_value;
} // namespace detail
} // namespace torch::headeronly

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
