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

#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/floating_point_utils.h>
#include <type_traits>

// TODO(#146647): do we need to special case OPENCL?
#if defined(__cplusplus)
#include <cstdint>
#elif !defined(__OPENCL_VERSION__)
#include <math.h>
#include <stdint.h>
#endif

#include <iosfwd>
#include <ostream>

namespace c10 {

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

C10_API inline std::ostream& operator<<(
    std::ostream& out,
    const Float8_e8m0fnu& value) {
  out << (float)value;
  return out;
}

} // namespace c10

#include <c10/util/Float8_e8m0fnu-inl.h> // IWYU pragma: keep
