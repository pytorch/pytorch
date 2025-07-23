#pragma once

/// Defines the Float8_e4m3fnuz type (8-bit floating-point) including
/// conversions to standard C types and basic arithmetic operations. Note that
/// arithmetic operations are implemented by converting to floating point and
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

#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/floating_point_utils.h>
#include <torch/headeronly/util/Float8_fnuz_cvt.h>

#include <limits>

#if defined(__cplusplus)
#include <cstdint>
#elif !defined(__OPENCL_VERSION__)
#include <math.h>
#include <stdint.h>
#endif

#include <iosfwd>
#include <ostream>

namespace torch::headeronly {

struct alignas(1) Float8_e4m3fnuz {
  uint8_t x;

  struct from_bits_t {};
  C10_HOST_DEVICE static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }

  Float8_e4m3fnuz() = default;

  constexpr C10_HOST_DEVICE Float8_e4m3fnuz(uint8_t bits, from_bits_t)
      : x(bits) {}
  inline C10_HOST_DEVICE Float8_e4m3fnuz(float value);
  inline C10_HOST_DEVICE operator float() const;
  inline C10_HOST_DEVICE bool isnan() const;
};

inline std::ostream& operator<<(
    std::ostream& out,
    const Float8_e4m3fnuz& value) {
  out << (float)value;
  return out;
}

namespace detail {

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

//------ below is copied from c10/util/Float8_e4m3fnuz-inl.h ------//
C10_CLANG_DIAGNOSTIC_PUSH()
#if C10_CLANG_HAS_WARNING("-Wimplicit-int-float-conversion")
C10_CLANG_DIAGNOSTIC_IGNORE("-Wimplicit-int-float-conversion")
#endif

/// Constructors

inline C10_HOST_DEVICE Float8_e4m3fnuz::Float8_e4m3fnuz(float value)
    : x(detail::fp8e4m3fnuz_from_fp32_value(value)) {}

/// Implicit conversions

inline C10_HOST_DEVICE Float8_e4m3fnuz::operator float() const {
  return detail::fp8_fnuz_to_fp32_value<4, 3>(x);
}

/// Special values helper

inline C10_HOST_DEVICE bool Float8_e4m3fnuz::isnan() const {
  return x == 0b10000000;
}

/// Arithmetic

inline C10_HOST_DEVICE Float8_e4m3fnuz
operator+(const Float8_e4m3fnuz& a, const Float8_e4m3fnuz& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8_e4m3fnuz
operator-(const Float8_e4m3fnuz& a, const Float8_e4m3fnuz& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8_e4m3fnuz
operator*(const Float8_e4m3fnuz& a, const Float8_e4m3fnuz& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8_e4m3fnuz operator/(
    const Float8_e4m3fnuz& a,
    const Float8_e4m3fnuz& b) __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline C10_HOST_DEVICE Float8_e4m3fnuz operator-(const Float8_e4m3fnuz& a) {
  return -static_cast<float>(a);
}

inline C10_HOST_DEVICE Float8_e4m3fnuz& operator+=(
    Float8_e4m3fnuz& a,
    const Float8_e4m3fnuz& b) {
  a = a + b;
  return a;
}

inline C10_HOST_DEVICE Float8_e4m3fnuz& operator-=(
    Float8_e4m3fnuz& a,
    const Float8_e4m3fnuz& b) {
  a = a - b;
  return a;
}

inline C10_HOST_DEVICE Float8_e4m3fnuz& operator*=(
    Float8_e4m3fnuz& a,
    const Float8_e4m3fnuz& b) {
  a = a * b;
  return a;
}

inline C10_HOST_DEVICE Float8_e4m3fnuz& operator/=(
    Float8_e4m3fnuz& a,
    const Float8_e4m3fnuz& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

inline C10_HOST_DEVICE float operator+(Float8_e4m3fnuz a, float b) {
  return static_cast<float>(a) + b;
}
inline C10_HOST_DEVICE float operator-(Float8_e4m3fnuz a, float b) {
  return static_cast<float>(a) - b;
}
inline C10_HOST_DEVICE float operator*(Float8_e4m3fnuz a, float b) {
  return static_cast<float>(a) * b;
}
inline C10_HOST_DEVICE float operator/(Float8_e4m3fnuz a, float b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<float>(a) / b;
}

inline C10_HOST_DEVICE float operator+(float a, Float8_e4m3fnuz b) {
  return a + static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator-(float a, Float8_e4m3fnuz b) {
  return a - static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator*(float a, Float8_e4m3fnuz b) {
  return a * static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator/(float a, Float8_e4m3fnuz b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<float>(b);
}

inline C10_HOST_DEVICE float& operator+=(float& a, const Float8_e4m3fnuz& b) {
  return a += static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator-=(float& a, const Float8_e4m3fnuz& b) {
  return a -= static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator*=(float& a, const Float8_e4m3fnuz& b) {
  return a *= static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator/=(float& a, const Float8_e4m3fnuz& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

inline C10_HOST_DEVICE double operator+(Float8_e4m3fnuz a, double b) {
  return static_cast<double>(a) + b;
}
inline C10_HOST_DEVICE double operator-(Float8_e4m3fnuz a, double b) {
  return static_cast<double>(a) - b;
}
inline C10_HOST_DEVICE double operator*(Float8_e4m3fnuz a, double b) {
  return static_cast<double>(a) * b;
}
inline C10_HOST_DEVICE double operator/(Float8_e4m3fnuz a, double b)
    __ubsan_ignore_float_divide_by_zero__ {
  return static_cast<double>(a) / b;
}

inline C10_HOST_DEVICE double operator+(double a, Float8_e4m3fnuz b) {
  return a + static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator-(double a, Float8_e4m3fnuz b) {
  return a - static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator*(double a, Float8_e4m3fnuz b) {
  return a * static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator/(double a, Float8_e4m3fnuz b)
    __ubsan_ignore_float_divide_by_zero__ {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

inline C10_HOST_DEVICE Float8_e4m3fnuz operator+(Float8_e4m3fnuz a, int b) {
  return a + static_cast<Float8_e4m3fnuz>(b);
}
inline C10_HOST_DEVICE Float8_e4m3fnuz operator-(Float8_e4m3fnuz a, int b) {
  return a - static_cast<Float8_e4m3fnuz>(b);
}
inline C10_HOST_DEVICE Float8_e4m3fnuz operator*(Float8_e4m3fnuz a, int b) {
  return a * static_cast<Float8_e4m3fnuz>(b);
}
inline C10_HOST_DEVICE Float8_e4m3fnuz operator/(Float8_e4m3fnuz a, int b) {
  return a / static_cast<Float8_e4m3fnuz>(b);
}

inline C10_HOST_DEVICE Float8_e4m3fnuz operator+(int a, Float8_e4m3fnuz b) {
  return static_cast<Float8_e4m3fnuz>(a) + b;
}
inline C10_HOST_DEVICE Float8_e4m3fnuz operator-(int a, Float8_e4m3fnuz b) {
  return static_cast<Float8_e4m3fnuz>(a) - b;
}
inline C10_HOST_DEVICE Float8_e4m3fnuz operator*(int a, Float8_e4m3fnuz b) {
  return static_cast<Float8_e4m3fnuz>(a) * b;
}
inline C10_HOST_DEVICE Float8_e4m3fnuz operator/(int a, Float8_e4m3fnuz b) {
  return static_cast<Float8_e4m3fnuz>(a) / b;
}

//// Arithmetic with int64_t

inline C10_HOST_DEVICE Float8_e4m3fnuz operator+(Float8_e4m3fnuz a, int64_t b) {
  return a + static_cast<Float8_e4m3fnuz>(b);
}
inline C10_HOST_DEVICE Float8_e4m3fnuz operator-(Float8_e4m3fnuz a, int64_t b) {
  return a - static_cast<Float8_e4m3fnuz>(b);
}
inline C10_HOST_DEVICE Float8_e4m3fnuz operator*(Float8_e4m3fnuz a, int64_t b) {
  return a * static_cast<Float8_e4m3fnuz>(b);
}
inline C10_HOST_DEVICE Float8_e4m3fnuz operator/(Float8_e4m3fnuz a, int64_t b) {
  return a / static_cast<Float8_e4m3fnuz>(b);
}

inline C10_HOST_DEVICE Float8_e4m3fnuz operator+(int64_t a, Float8_e4m3fnuz b) {
  return static_cast<Float8_e4m3fnuz>(a) + b;
}
inline C10_HOST_DEVICE Float8_e4m3fnuz operator-(int64_t a, Float8_e4m3fnuz b) {
  return static_cast<Float8_e4m3fnuz>(a) - b;
}
inline C10_HOST_DEVICE Float8_e4m3fnuz operator*(int64_t a, Float8_e4m3fnuz b) {
  return static_cast<Float8_e4m3fnuz>(a) * b;
}
inline C10_HOST_DEVICE Float8_e4m3fnuz operator/(int64_t a, Float8_e4m3fnuz b) {
  return static_cast<Float8_e4m3fnuz>(a) / b;
}

/// NOTE: we do not define comparisons directly and instead rely on the implicit
/// conversion from c10::Float8_e4m3fnuz to float.

C10_CLANG_DIAGNOSTIC_POP()

} // namespace torch::headeronly

namespace c10 {
  using torch::headeronly::Float8_e4m3fnuz;
  using torch::headeronly::operator<<;

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
