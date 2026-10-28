#pragma once

/// Defines the Float8_e5m3fnu type (8-bit unsigned floating-point) including
/// conversions to standard C types
/// Binary configuration :
/// eeeeemmm
/// no sign bit
/// 5 exponent bits, bias 15
/// 3 mantissa bits
///
/// This is NVIDIA's UE5M3 scale format (cutlass::float_ue5m3_t). It has
/// subnormals, no infinity, and a single NaN encoding, 0xFF. Exponent 31 with
/// mantissa 0b000..0b110 encodes finite values, so the range is
/// 2^-17 (min subnormal) .. 114688 (1.75 * 2^16).
///
/// Conversion from float32 rounds |value| to nearest, ties to even: the sign
/// is ignored (as for Float8_e8m0fnu), finite values above the max and inf
/// saturate to the max, and NaN converts to NaN.

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/floating_point_utils.h>

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

struct alignas(1) Float8_e5m3fnu {
  uint8_t x;

  struct from_bits_t {};
  C10_HOST_DEVICE static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  Float8_e5m3fnu() = default;

  constexpr C10_HOST_DEVICE Float8_e5m3fnu(uint8_t bits, from_bits_t /*unused*/)
      : x(bits) {}
  inline C10_HOST_DEVICE Float8_e5m3fnu(float value);
  inline C10_HOST_DEVICE operator float() const;
  inline C10_HOST_DEVICE bool isnan() const;
};

inline std::ostream& operator<<(
    std::ostream& out,
    const Float8_e5m3fnu& value) {
  out << (float)value;
  return out;
}

namespace detail {

inline C10_HOST_DEVICE float fp8e5m3fnu_to_fp32_value(uint8_t input) {
  if (input == 0xFF) {
    return fp32_from_bits(UINT32_C(0x7FC00000));
  }
  const uint32_t exponent = input >> 3;
  const uint32_t mantissa = input & 0x7;
  if (exponent == 0) {
    // subnormal: mantissa * 2^-17
    return static_cast<float>(mantissa) * fp32_from_bits(UINT32_C(110) << 23);
  }
  // rebias the exponent from 15 to 127 and widen the mantissa from 3 to 23 bits
  return fp32_from_bits(((exponent + (127 - 15)) << 23) | (mantissa << 20));
}

/*
 * Convert a 32-bit floating-point number in IEEE single-precision format to a
 * 8-bit floating-point number in fp8 e5m3fnu format, in bit representation.
 */
inline C10_HOST_DEVICE uint8_t fp8e5m3fnu_from_fp32_value(float f) {
  // 114688 = 1.75 * 2^16, the largest finite fp8e5m3fnu value:
  // biased fp32 exponent 127 + 16, mantissa 0b11
  constexpr uint32_t fp8_max = (UINT32_C(143) << 23) | (UINT32_C(3) << 21);

  // A mask for converting fp32 numbers lower than fp8e5m3fnu normal range
  // into denorm representation
  // magic number: ((127 - 15) + (23 - 3) + 1)
  constexpr uint32_t denorm_mask = UINT32_C(133) << 23;

  // there is no sign bit, so convert |f|
  uint32_t f_bits = fp32_to_bits(f) & UINT32_C(0x7FFFFFFF);

  if (f_bits > UINT32_C(0x7F800000)) {
    // NaN -> NaN
    return 0xFF;
  }
  if (f_bits >= fp8_max) {
    // finite overflow or inf -> saturate to the max finite value
    return 0xFE;
  }
  if (f_bits < (UINT32_C(113) << 23)) {
    // Input number is smaller than 2^(-14), which is the smallest
    // fp8e5m3fnu normal number
    f_bits = fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
    return static_cast<uint8_t>(f_bits - denorm_mask);
  }

  // resulting mantissa is odd
  const uint32_t mant_odd = (f_bits >> 20) & 1;
  // update exponent, rounding bias part 1
  f_bits += (static_cast<uint32_t>(15 - 127) << 23) + 0x7FFFF;
  // rounding bias part 2
  f_bits += mant_odd;
  return static_cast<uint8_t>(f_bits >> 20);
}

} // namespace detail

/// Constructors
inline C10_HOST_DEVICE Float8_e5m3fnu::Float8_e5m3fnu(float value)
    : x(detail::fp8e5m3fnu_from_fp32_value(value)) {}

/// Implicit conversions
inline C10_HOST_DEVICE Float8_e5m3fnu::operator float() const {
  return detail::fp8e5m3fnu_to_fp32_value(x);
}

/// Special values helper
inline C10_HOST_DEVICE bool Float8_e5m3fnu::isnan() const {
  return x == 0xFF;
}

/// NOTE: we do not define comparisons directly and instead rely on the implicit
/// conversion from c10::Float8_e5m3fnu to float.

} // namespace c10

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)
using c10::Float8_e5m3fnu;
using c10::operator<<;

namespace detail {
using c10::detail::fp8e5m3fnu_from_fp32_value;
using c10::detail::fp8e5m3fnu_to_fp32_value;
} // namespace detail
HIDDEN_NAMESPACE_END(torch, headeronly)

namespace std {

template <>
class numeric_limits<c10::Float8_e5m3fnu> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = false;
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
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 17;
  static constexpr int max_exponent10 = 5;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before = false;

  static constexpr c10::Float8_e5m3fnu min() {
    // 2^-14, the smallest normal value
    return c10::Float8_e5m3fnu(0x08, c10::Float8_e5m3fnu::from_bits());
  }
  static constexpr c10::Float8_e5m3fnu lowest() {
    return c10::Float8_e5m3fnu(0x00, c10::Float8_e5m3fnu::from_bits());
  }
  static constexpr c10::Float8_e5m3fnu max() {
    // 1.75 * 2^16 = 114688
    return c10::Float8_e5m3fnu(0xFE, c10::Float8_e5m3fnu::from_bits());
  }
  static constexpr c10::Float8_e5m3fnu epsilon() {
    // 2^-3
    return c10::Float8_e5m3fnu(0x60, c10::Float8_e5m3fnu::from_bits());
  }
  static constexpr c10::Float8_e5m3fnu round_error() {
    // 0.5
    return c10::Float8_e5m3fnu(0x70, c10::Float8_e5m3fnu::from_bits());
  }
  static constexpr c10::Float8_e5m3fnu quiet_NaN() {
    return c10::Float8_e5m3fnu(0xFF, c10::Float8_e5m3fnu::from_bits());
  }
  static constexpr c10::Float8_e5m3fnu denorm_min() {
    // 2^-17
    return c10::Float8_e5m3fnu(0x01, c10::Float8_e5m3fnu::from_bits());
  }
};

} // namespace std
