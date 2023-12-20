#pragma once

/// Defines the Float8_e4m3fnuz type (8-bit floating-point) including conversions
/// to standard C types and basic arithmetic operations. Note that arithmetic
/// operations are implemented by converting to floating point and
/// performing the operation in float32.
/// Binary configuration remains the same as Float8_e4m3fn:
/// s eeee mmm
/// 1 sign bit
/// 4 exponent bits
/// 3 mantissa bits
/// The key differences versus Float8_e4m3fn are:
/// bias = 8
/// no infinities or negative zero
/// NaN only when sign bit is 1, rest all 0s
///
/// Implementation based on the paper https://arxiv.org/pdf/2206.02915.pdf and
/// the existing Float8_e4m3fn implementation.

#include <c10/macros/Macros.h>
#include <c10/util/C++17.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/floating_point_utils.h>
#include <type_traits>

#if defined(__cplusplus) && (__cplusplus >= 201103L)
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
 * Convert a 8-bit floating-point number in fp8 E4M3FNUZ format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
inline C10_HOST_DEVICE float fp8e4m3fnuz_to_fp32_value(uint8_t input) {
  constexpr std::array<float, 256> e4m3fnuz_lut = {
      0.0f,
      0.0009765625f,
      0.001953125f,
      0.0029296875f,
      0.00390625f,
      0.0048828125f,
      0.005859375f,
      0.0068359375f,
      0.0078125f,
      0.0087890625f,
      0.009765625f,
      0.0107421875f,
      0.01171875f,
      0.0126953125f,
      0.013671875f,
      0.0146484375f,
      0.015625f,
      0.017578125f,
      0.01953125f,
      0.021484375f,
      0.0234375f,
      0.025390625f,
      0.02734375f,
      0.029296875f,
      0.03125f,
      0.03515625f,
      0.0390625f,
      0.04296875f,
      0.046875f,
      0.05078125f,
      0.0546875f,
      0.05859375f,
      0.0625f,
      0.0703125f,
      0.078125f,
      0.0859375f,
      0.09375f,
      0.1015625f,
      0.109375f,
      0.1171875f,
      0.125f,
      0.140625f,
      0.15625f,
      0.171875f,
      0.1875f,
      0.203125f,
      0.21875f,
      0.234375f,
      0.25f,
      0.28125f,
      0.3125f,
      0.34375f,
      0.375f,
      0.40625f,
      0.4375f,
      0.46875f,
      0.5f,
      0.5625f,
      0.625f,
      0.6875f,
      0.75f,
      0.8125f,
      0.875f,
      0.9375f,
      1.0f,
      1.125f,
      1.25f,
      1.375f,
      1.5f,
      1.625f,
      1.75f,
      1.875f,
      2.0f,
      2.25f,
      2.5f,
      2.75f,
      3.0f,
      3.25f,
      3.5f,
      3.75f,
      4.0f,
      4.5f,
      5.0f,
      5.5f,
      6.0f,
      6.5f,
      7.0f,
      7.5f,
      8.0f,
      9.0f,
      10.0f,
      11.0f,
      12.0f,
      13.0f,
      14.0f,
      15.0f,
      16.0f,
      18.0f,
      20.0f,
      22.0f,
      24.0f,
      26.0f,
      28.0f,
      30.0f,
      32.0f,
      36.0f,
      40.0f,
      44.0f,
      48.0f,
      52.0f,
      56.0f,
      60.0f,
      64.0f,
      72.0f,
      80.0f,
      88.0f,
      96.0f,
      104.0f,
      112.0f,
      120.0f,
      128.0f,
      144.0f,
      160.0f,
      176.0f,
      192.0f,
      208.0f,
      224.0f,
      240.0f,
      std::numeric_limits<float>::signaling_NaN(),
      -0.0009765625f,
      -0.001953125f,
      -0.0029296875f,
      -0.00390625f,
      -0.0048828125f,
      -0.005859375f,
      -0.0068359375f,
      -0.0078125f,
      -0.0087890625f,
      -0.009765625f,
      -0.0107421875f,
      -0.01171875f,
      -0.0126953125f,
      -0.013671875f,
      -0.0146484375f,
      -0.015625f,
      -0.017578125f,
      -0.01953125f,
      -0.021484375f,
      -0.0234375f,
      -0.025390625f,
      -0.02734375f,
      -0.029296875f,
      -0.03125f,
      -0.03515625f,
      -0.0390625f,
      -0.04296875f,
      -0.046875f,
      -0.05078125f,
      -0.0546875f,
      -0.05859375f,
      -0.0625f,
      -0.0703125f,
      -0.078125f,
      -0.0859375f,
      -0.09375f,
      -0.1015625f,
      -0.109375f,
      -0.1171875f,
      -0.125f,
      -0.140625f,
      -0.15625f,
      -0.171875f,
      -0.1875f,
      -0.203125f,
      -0.21875f,
      -0.234375f,
      -0.25f,
      -0.28125f,
      -0.3125f,
      -0.34375f,
      -0.375f,
      -0.40625f,
      -0.4375f,
      -0.46875f,
      -0.5f,
      -0.5625f,
      -0.625f,
      -0.6875f,
      -0.75f,
      -0.8125f,
      -0.875f,
      -0.9375f,
      -1.0f,
      -1.125f,
      -1.25f,
      -1.375f,
      -1.5f,
      -1.625f,
      -1.75f,
      -1.875f,
      -2.0f,
      -2.25f,
      -2.5f,
      -2.75f,
      -3.0f,
      -3.25f,
      -3.5f,
      -3.75f,
      -4.0f,
      -4.5f,
      -5.0f,
      -5.5f,
      -6.0f,
      -6.5f,
      -7.0f,
      -7.5f,
      -8.0f,
      -9.0f,
      -10.0f,
      -11.0f,
      -12.0f,
      -13.0f,
      -14.0f,
      -15.0f,
      -16.0f,
      -18.0f,
      -20.0f,
      -22.0f,
      -24.0f,
      -26.0f,
      -28.0f,
      -30.0f,
      -32.0f,
      -36.0f,
      -40.0f,
      -44.0f,
      -48.0f,
      -52.0f,
      -56.0f,
      -60.0f,
      -64.0f,
      -72.0f,
      -80.0f,
      -88.0f,
      -96.0f,
      -104.0f,
      -112.0f,
      -120.0f,
      -128.0f,
      -144.0f,
      -160.0f,
      -176.0f,
      -192.0f,
      -208.0f,
      -224.0f,
      -240.0f,
  };

  return e4m3fnuz_lut[input];
}

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E4M3FNUZ format, in bit representation.
 */
inline C10_HOST_DEVICE uint8_t fp8e4m3fnuz_from_fp32_value(float f) {
  /*
   * Binary representation of 256.0f, which is the first value not representable
   * (i.e. the first value which would overflow in to the sign bit, resulting in
   * a NaN) in fp8e4m3fnuz range:
   * 1 0000 000 - fp8e4m3fnuz
   * 0 10000111 00000000000000000000000 - fp32
   */
  constexpr uint32_t fnuz_max = UINT32_C(0x87) << 23;

  /*
   * A mask for converting fp32 numbers lower than fp8e4m3fnuz normal range
   * into denorm representation
   * magic number: ((127 - 8) + (23 - 3) + 1)
   */
  constexpr uint32_t denorm_mask = UINT32_C(0x8C) << 23;

  uint32_t f_bits = fp32_to_bits(f);

  uint32_t result = 0u;

  /*
   * Extract the sign of the input number into the high bit of the 32-bit word:
   *
   *      +---+----------------------------------+
   *      | S |0000000 00000000 00000000 00000000|
   *      +---+----------------------------------+
   * Bits  31                 0-31
   */
  const uint32_t sign = f_bits & UINT32_C(0x80000000);

  /*
   * Set sign bit to 0
   */
  f_bits ^= sign;

  if (f_bits >= fnuz_max) {
    // NaN -- sign bit set to 1, rest 0s.
    return 0x80;
  }

  if (f_bits < (UINT32_C(0x78) << 23) /* 2^-7 in float32 */) {
    // Input exponent is less than -7, the smallest e4m3fnuz exponent, so the
    // number will become subnormal.
    f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
    result = static_cast<uint8_t>(f_bits - denorm_mask);
    if (result == 0) {
      // fnuz types don't have negative zero.
      return 0;
    }
  } else {
    // resulting mantissa is odd
    uint8_t mant_odd = (f_bits >> 20) & 1;

    // update exponent, rounding bias part 1
    f_bits += ((uint32_t)(8 - 127) << 23) + 0x7FFFF;

    // rounding bias part 2
    f_bits += mant_odd;

    // take the bits!
    result = static_cast<uint8_t>(f_bits >> 20);
  }

  result |= sign >> 24;
  return result;
}

} // namespace detail

struct alignas(1) Float8_e4m3fnuz {
  uint8_t x;

  struct from_bits_t {};
  C10_HOST_DEVICE static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  Float8_e4m3fnuz() = default;

  constexpr C10_HOST_DEVICE Float8_e4m3fnuz(uint8_t bits, from_bits_t)
      : x(bits){};
  inline C10_HOST_DEVICE Float8_e4m3fnuz(float value);
  inline C10_HOST_DEVICE operator float() const;
  inline C10_HOST_DEVICE bool isnan() const;
};

C10_API std::ostream& operator<<(std::ostream& out, const Float8_e4m3fnuz& value);

} // namespace c10

#include <c10/util/Float8_e4m3fnuz-inl.h> // IWYU pragma: keep
