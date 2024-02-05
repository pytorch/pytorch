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

namespace detail {

// Move from .cpp to header. The implementation could be inlined in kernel to
// avoid device code relocation.
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

} // namespace detail

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
