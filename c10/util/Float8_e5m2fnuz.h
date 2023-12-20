#pragma once

/// Defines the Float8_e5m2fnuz type (8-bit floating-point) including conversions
/// to standard C types and basic arithmetic operations. Note that arithmetic
/// operations are implemented by converting to floating point and
/// performing the operation in float32.
/// Binary configuration remains the same as e5m2:
/// s eeeee mm
/// 1 sign bit
/// 5 exponent bits
/// 2 mantissa bits
/// The key differences that e5m2fnuz brings are:
/// bias = 16
/// no infinities or negative zero
/// NaN only when sign bit is 1, rest all 0s
///
/// Implementation based on the paper https://arxiv.org/pdf/2206.02915.pdf and
/// the existing Float8_e4m3fn implementation.

#include <c10/macros/Macros.h>
#include <c10/util/C++17.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/floating_point_utils.h>

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
 * Convert a 8-bit floating-point number in fp8 E5M2FNUZ format, in bit
 * representation, to a 32-bit floating-point number in IEEE single-precision
 * format, in bit representation.
 *
 * @note The implementation doesn't use any floating-point operations.
 */
inline C10_HOST_DEVICE float fp8e5m2fnuz_to_fp32_value(uint8_t input) {
  constexpr std::array<float, 256> e5m2fnuz_lut = {
      0.0f,
      7.62939453125e-06f,
      1.52587890625e-05f,
      2.288818359375e-05f,
      3.0517578125e-05f,
      3.814697265625e-05f,
      4.57763671875e-05f,
      5.340576171875e-05f,
      6.103515625e-05f,
      7.62939453125e-05f,
      9.1552734375e-05f,
      0.0001068115234375f,
      0.0001220703125f,
      0.000152587890625f,
      0.00018310546875f,
      0.000213623046875f,
      0.000244140625f,
      0.00030517578125f,
      0.0003662109375f,
      0.00042724609375f,
      0.00048828125f,
      0.0006103515625f,
      0.000732421875f,
      0.0008544921875f,
      0.0009765625f,
      0.001220703125f,
      0.00146484375f,
      0.001708984375f,
      0.001953125f,
      0.00244140625f,
      0.0029296875f,
      0.00341796875f,
      0.00390625f,
      0.0048828125f,
      0.005859375f,
      0.0068359375f,
      0.0078125f,
      0.009765625f,
      0.01171875f,
      0.013671875f,
      0.015625f,
      0.01953125f,
      0.0234375f,
      0.02734375f,
      0.03125f,
      0.0390625f,
      0.046875f,
      0.0546875f,
      0.0625f,
      0.078125f,
      0.09375f,
      0.109375f,
      0.125f,
      0.15625f,
      0.1875f,
      0.21875f,
      0.25f,
      0.3125f,
      0.375f,
      0.4375f,
      0.5f,
      0.625f,
      0.75f,
      0.875f,
      1.0f,
      1.25f,
      1.5f,
      1.75f,
      2.0f,
      2.5f,
      3.0f,
      3.5f,
      4.0f,
      5.0f,
      6.0f,
      7.0f,
      8.0f,
      10.0f,
      12.0f,
      14.0f,
      16.0f,
      20.0f,
      24.0f,
      28.0f,
      32.0f,
      40.0f,
      48.0f,
      56.0f,
      64.0f,
      80.0f,
      96.0f,
      112.0f,
      128.0f,
      160.0f,
      192.0f,
      224.0f,
      256.0f,
      320.0f,
      384.0f,
      448.0f,
      512.0f,
      640.0f,
      768.0f,
      896.0f,
      1024.0f,
      1280.0f,
      1536.0f,
      1792.0f,
      2048.0f,
      2560.0f,
      3072.0f,
      3584.0f,
      4096.0f,
      5120.0f,
      6144.0f,
      7168.0f,
      8192.0f,
      10240.0f,
      12288.0f,
      14336.0f,
      16384.0f,
      20480.0f,
      24576.0f,
      28672.0f,
      32768.0f,
      40960.0f,
      49152.0f,
      57344.0f,
      std::numeric_limits<float>::signaling_NaN(),
      -7.62939453125e-06f,
      -1.52587890625e-05f,
      -2.288818359375e-05f,
      -3.0517578125e-05f,
      -3.814697265625e-05f,
      -4.57763671875e-05f,
      -5.340576171875e-05f,
      -6.103515625e-05f,
      -7.62939453125e-05f,
      -9.1552734375e-05f,
      -0.0001068115234375f,
      -0.0001220703125f,
      -0.000152587890625f,
      -0.00018310546875f,
      -0.000213623046875f,
      -0.000244140625f,
      -0.00030517578125f,
      -0.0003662109375f,
      -0.00042724609375f,
      -0.00048828125f,
      -0.0006103515625f,
      -0.000732421875f,
      -0.0008544921875f,
      -0.0009765625f,
      -0.001220703125f,
      -0.00146484375f,
      -0.001708984375f,
      -0.001953125f,
      -0.00244140625f,
      -0.0029296875f,
      -0.00341796875f,
      -0.00390625f,
      -0.0048828125f,
      -0.005859375f,
      -0.0068359375f,
      -0.0078125f,
      -0.009765625f,
      -0.01171875f,
      -0.013671875f,
      -0.015625f,
      -0.01953125f,
      -0.0234375f,
      -0.02734375f,
      -0.03125f,
      -0.0390625f,
      -0.046875f,
      -0.0546875f,
      -0.0625f,
      -0.078125f,
      -0.09375f,
      -0.109375f,
      -0.125f,
      -0.15625f,
      -0.1875f,
      -0.21875f,
      -0.25f,
      -0.3125f,
      -0.375f,
      -0.4375f,
      -0.5f,
      -0.625f,
      -0.75f,
      -0.875f,
      -1.0f,
      -1.25f,
      -1.5f,
      -1.75f,
      -2.0f,
      -2.5f,
      -3.0f,
      -3.5f,
      -4.0f,
      -5.0f,
      -6.0f,
      -7.0f,
      -8.0f,
      -10.0f,
      -12.0f,
      -14.0f,
      -16.0f,
      -20.0f,
      -24.0f,
      -28.0f,
      -32.0f,
      -40.0f,
      -48.0f,
      -56.0f,
      -64.0f,
      -80.0f,
      -96.0f,
      -112.0f,
      -128.0f,
      -160.0f,
      -192.0f,
      -224.0f,
      -256.0f,
      -320.0f,
      -384.0f,
      -448.0f,
      -512.0f,
      -640.0f,
      -768.0f,
      -896.0f,
      -1024.0f,
      -1280.0f,
      -1536.0f,
      -1792.0f,
      -2048.0f,
      -2560.0f,
      -3072.0f,
      -3584.0f,
      -4096.0f,
      -5120.0f,
      -6144.0f,
      -7168.0f,
      -8192.0f,
      -10240.0f,
      -12288.0f,
      -14336.0f,
      -16384.0f,
      -20480.0f,
      -24576.0f,
      -28672.0f,
      -32768.0f,
      -40960.0f,
      -49152.0f,
      -57344.0f,
  };

  return e5m2fnuz_lut[input];
}

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 E5M2 format, in bit representation.
 */
inline C10_HOST_DEVICE uint8_t fp8e5m2fnuz_from_fp32_value(float f) {
  /*
   * Binary representation of 65536.0f, which is the first value not
   * representable (i.e. the first value which would overflow in to the sign
   * bit, resulting in a NaN) in fp8e4m3fnuz range:
   * 1 00000 00 - fp8e5m2fnuz
   * 0 10001111 00000000000000000000000 - fp32
   */
  constexpr uint32_t fnuz_max = UINT32_C(0x8F) << 23;

  /*
   * A mask for converting fp32 numbers lower than fp8e5m2fnuz normal range
   * into denormalized representation.
   * magic number: ((127 - 16) + (23 - 2) + 1)
   */
  constexpr uint32_t denorm_mask = UINT32_C(0x85) << 23;

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
    // NaN -- sign bit set to 1, rest 0s
    return 0x80;
  }

  if (f_bits < (UINT32_C(0x70) << 23) /* 2^-15 in float32 */) {
    // Input exponent is less than -15, the smallest e5m2fnuz exponent, so the
    // number will become subnormal.
    f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
    result = static_cast<uint8_t>(f_bits - denorm_mask);
    if (result == 0) {
      // fnuz types don't have negative zero.
      return 0;
    }
  } else {
    // resulting mantissa is odd
    uint8_t mant_odd = (f_bits >> 21) & 1;

    // update exponent, rounding bias part 1
    f_bits += ((uint32_t)(16 - 127) << 23) + 0xFFFFF;

    // rounding bias part 2
    f_bits += mant_odd;

    // take the bits!
    result = static_cast<uint8_t>(f_bits >> 21);
  }

  result |= sign >> 24;
  return result;
}

} // namespace detail

struct alignas(1) Float8_e5m2fnuz {
  uint8_t x;

  struct from_bits_t {};
  C10_HOST_DEVICE static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  Float8_e5m2fnuz() = default;

  constexpr C10_HOST_DEVICE Float8_e5m2fnuz(uint8_t bits, from_bits_t)
      : x(bits) {}
  inline C10_HOST_DEVICE Float8_e5m2fnuz(float value);
  inline C10_HOST_DEVICE operator float() const;
  inline C10_HOST_DEVICE bool isnan() const;
  inline C10_HOST_DEVICE bool isinf() const;
};

C10_API std::ostream& operator<<(std::ostream& out, const Float8_e5m2fnuz& value);

} // namespace c10

#include <c10/util/Float8_e5m2fnuz-inl.h> // IWYU pragma: keep
