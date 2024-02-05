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

} // namespace detail

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
