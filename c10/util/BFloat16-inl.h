#pragma once

#include <c10/macros/Macros.h>
#include <limits>
#include <c10/util/math_compat.h>

namespace c10 {

/// Constructors
inline C10_HOST_DEVICE BFloat16::BFloat16(float value) {
  // RNE by default
  x = detail::round_to_nearest_even(value);
}

/// Implicit conversions
inline C10_HOST_DEVICE BFloat16::operator float() const {
  return detail::f32_from_bits(x);
}

/// Arithmetic

inline C10_HOST_DEVICE BFloat16 operator+(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

inline C10_HOST_DEVICE BFloat16 operator-(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

inline C10_HOST_DEVICE BFloat16 operator*(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

inline C10_HOST_DEVICE BFloat16 operator/(const BFloat16& a, const BFloat16& b) {
  return static_cast<float>(a) / static_cast<float>(b);
}

inline C10_HOST_DEVICE BFloat16 operator-(const BFloat16& a) {
  return -static_cast<float>(a);
}

inline C10_HOST_DEVICE BFloat16& operator+=(BFloat16& a, const BFloat16& b) {
  a = a + b;
  return a;
}

inline C10_HOST_DEVICE BFloat16& operator-=(BFloat16& a, const BFloat16& b) {
  a = a - b;
  return a;
}

inline C10_HOST_DEVICE BFloat16& operator*=(BFloat16& a, const BFloat16& b) {
  a = a * b;
  return a;
}

inline C10_HOST_DEVICE BFloat16& operator/=(BFloat16& a, const BFloat16& b) {
  a = a / b;
  return a;
}

inline C10_HOST_DEVICE BFloat16& operator|(BFloat16& a, const BFloat16& b) {
  a.x = a.x | b.x;
  return a;
}

inline C10_HOST_DEVICE BFloat16& operator^(BFloat16& a, const BFloat16& b) {
  a.x = a.x ^ b.x;
  return a;
}

inline C10_HOST_DEVICE BFloat16& operator&(BFloat16& a, const BFloat16& b) {
  a.x = a.x & b.x;
  return a;
}

/// Arithmetic with floats

inline C10_HOST_DEVICE float operator+(BFloat16 a, float b) {
  return static_cast<float>(a) + b;
}
inline C10_HOST_DEVICE float operator-(BFloat16 a, float b) {
  return static_cast<float>(a) - b;
}
inline C10_HOST_DEVICE float operator*(BFloat16 a, float b) {
  return static_cast<float>(a) * b;
}
inline C10_HOST_DEVICE float operator/(BFloat16 a, float b) {
  return static_cast<float>(a) / b;
}

inline C10_HOST_DEVICE float operator+(float a, BFloat16 b) {
  return a + static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator-(float a, BFloat16 b) {
  return a - static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator*(float a, BFloat16 b) {
  return a * static_cast<float>(b);
}
inline C10_HOST_DEVICE float operator/(float a, BFloat16 b) {
  return a / static_cast<float>(b);
}

inline C10_HOST_DEVICE float& operator+=(float& a, const BFloat16& b) {
  return a += static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator-=(float& a, const BFloat16& b) {
  return a -= static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator*=(float& a, const BFloat16& b) {
  return a *= static_cast<float>(b);
}
inline C10_HOST_DEVICE float& operator/=(float& a, const BFloat16& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

inline C10_HOST_DEVICE double operator+(BFloat16 a, double b) {
  return static_cast<double>(a) + b;
}
inline C10_HOST_DEVICE double operator-(BFloat16 a, double b) {
  return static_cast<double>(a) - b;
}
inline C10_HOST_DEVICE double operator*(BFloat16 a, double b) {
  return static_cast<double>(a) * b;
}
inline C10_HOST_DEVICE double operator/(BFloat16 a, double b) {
  return static_cast<double>(a) / b;
}

inline C10_HOST_DEVICE double operator+(double a, BFloat16 b) {
  return a + static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator-(double a, BFloat16 b) {
  return a - static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator*(double a, BFloat16 b) {
  return a * static_cast<double>(b);
}
inline C10_HOST_DEVICE double operator/(double a, BFloat16 b) {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

inline C10_HOST_DEVICE BFloat16 operator+(BFloat16 a, int b) {
  return a + static_cast<BFloat16>(b);
}
inline C10_HOST_DEVICE BFloat16 operator-(BFloat16 a, int b) {
  return a - static_cast<BFloat16>(b);
}
inline C10_HOST_DEVICE BFloat16 operator*(BFloat16 a, int b) {
  return a * static_cast<BFloat16>(b);
}
inline C10_HOST_DEVICE BFloat16 operator/(BFloat16 a, int b) {
  return a / static_cast<BFloat16>(b);
}

inline C10_HOST_DEVICE BFloat16 operator+(int a, BFloat16 b) {
  return static_cast<BFloat16>(a) + b;
}
inline C10_HOST_DEVICE BFloat16 operator-(int a, BFloat16 b) {
  return static_cast<BFloat16>(a) - b;
}
inline C10_HOST_DEVICE BFloat16 operator*(int a, BFloat16 b) {
  return static_cast<BFloat16>(a) * b;
}
inline C10_HOST_DEVICE BFloat16 operator/(int a, BFloat16 b) {
  return static_cast<BFloat16>(a) / b;
}

//// Arithmetic with int64_t

inline C10_HOST_DEVICE BFloat16 operator+(BFloat16 a, int64_t b) {
  return a + static_cast<BFloat16>(b);
}
inline C10_HOST_DEVICE BFloat16 operator-(BFloat16 a, int64_t b) {
  return a - static_cast<BFloat16>(b);
}
inline C10_HOST_DEVICE BFloat16 operator*(BFloat16 a, int64_t b) {
  return a * static_cast<BFloat16>(b);
}
inline C10_HOST_DEVICE BFloat16 operator/(BFloat16 a, int64_t b) {
  return a / static_cast<BFloat16>(b);
}

inline C10_HOST_DEVICE BFloat16 operator+(int64_t a, BFloat16 b) {
  return static_cast<BFloat16>(a) + b;
}
inline C10_HOST_DEVICE BFloat16 operator-(int64_t a, BFloat16 b) {
  return static_cast<BFloat16>(a) - b;
}
inline C10_HOST_DEVICE BFloat16 operator*(int64_t a, BFloat16 b) {
  return static_cast<BFloat16>(a) * b;
}
inline C10_HOST_DEVICE BFloat16 operator/(int64_t a, BFloat16 b) {
  return static_cast<BFloat16>(a) / b;
}

} // namespace c10

namespace std {

template <>
class numeric_limits<c10::BFloat16> {
public:
  static constexpr bool is_signed = true;
  static constexpr bool is_specialized = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss =
      numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 8;
  static constexpr int digits10 = 2;
  static constexpr int max_digits10 = 4;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -125;
  static constexpr int min_exponent10 = -37;
  static constexpr int max_exponent = 128;
  static constexpr int max_exponent10 = 38;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;

  static constexpr c10::BFloat16 min() {
    return c10::BFloat16(0x0080, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 lowest() {
    return c10::BFloat16(0xFF7F, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 max() {
    return c10::BFloat16(0x7F7F, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 epsilon() {
    return c10::BFloat16(0x3C00, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 round_error() {
    return c10::BFloat16(0x3F00, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 infinity() {
    return c10::BFloat16(0x7F80, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 quiet_NaN() {
    return c10::BFloat16(0x7FC0, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 signaling_NaN() {
    return c10::BFloat16(0x7F80, c10::BFloat16::from_bits());
  }
  static constexpr c10::BFloat16 denorm_min() {
    return c10::BFloat16(0x0001, c10::BFloat16::from_bits());
  }
};

/// Used by vec256<c10::BFloat16>::map
inline c10::BFloat16 acos(c10::BFloat16 a) { return std::acos(float(a));}
inline c10::BFloat16 asin(c10::BFloat16 a) { return std::asin(float(a));}
inline c10::BFloat16 atan(c10::BFloat16 a) { return std::atan(float(a));}
inline c10::BFloat16 erf(c10::BFloat16 a) { return std::erf(float(a));}
inline c10::BFloat16 erfc(c10::BFloat16 a) { return std::erfc(float(a));}
inline c10::BFloat16 exp(c10::BFloat16 a) { return std::exp(float(a));}
inline c10::BFloat16 expm1(c10::BFloat16 a) { return std::expm1(float(a));}
inline c10::BFloat16 log(c10::BFloat16 a) { return std::log(float(a));}
inline c10::BFloat16 log10(c10::BFloat16 a) { return std::log10(float(a));}
inline c10::BFloat16 log1p(c10::BFloat16 a) { return std::log1p(float(a));}
inline c10::BFloat16 log2(c10::BFloat16 a) { return std::log2(float(a));}
inline c10::BFloat16 ceil(c10::BFloat16 a) { return std::ceil(float(a));}
inline c10::BFloat16 cos(c10::BFloat16 a) { return std::cos(float(a));}
inline c10::BFloat16 floor(c10::BFloat16 a) { return std::floor(float(a));}
inline c10::BFloat16 nearbyint(c10::BFloat16 a) { return std::nearbyint(float(a));}
inline c10::BFloat16 sin(c10::BFloat16 a) { return std::sin(float(a));}
inline c10::BFloat16 tan(c10::BFloat16 a) { return std::tan(float(a));}
inline c10::BFloat16 tanh(c10::BFloat16 a) { return std::tanh(float(a));}
inline c10::BFloat16 trunc(c10::BFloat16 a) { return std::trunc(float(a));}
inline c10::BFloat16 lgamma(c10::BFloat16 a) { return std::lgamma(float(a));}
inline c10::BFloat16 sqrt(c10::BFloat16 a) { return std::sqrt(float(a));}
inline c10::BFloat16 rsqrt(c10::BFloat16 a) { return 1.0 / std::sqrt(float(a));}
inline c10::BFloat16 abs(c10::BFloat16 a) { return std::abs(float(a));}
inline c10::BFloat16 min(c10::BFloat16 a, c10::BFloat16 b) { return std::min(float(a), float(b));}
inline c10::BFloat16 max(c10::BFloat16 a, c10::BFloat16 b) { return std::max(float(a), float(b));}
inline c10::BFloat16 pow(c10::BFloat16 a, double b) { return std::pow(float(a), b);}
inline c10::BFloat16 pow(c10::BFloat16 a, c10::BFloat16 b) { return std::pow(float(a), float(b));}

} // namespace std
