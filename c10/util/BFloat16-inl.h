#pragma once

#include <c10/macros/Macros.h>
#include <limits>

namespace c10 {

/// Constructors
inline C10_HOST_DEVICE BFloat16::BFloat16(float value) {
  x = detail::bits_from_f32(value);
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
  return a | b;
}

inline C10_HOST_DEVICE BFloat16& operator^(BFloat16& a, const BFloat16& b) {
  return a ^ b;
}

inline C10_HOST_DEVICE BFloat16& operator&(BFloat16& a, const BFloat16& b) {
  return a & b;
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
    static constexpr bool is_integer = false;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr c10::BFloat16 lowest() {
      return at::BFloat16(0xFF7F, at::BFloat16::from_bits());
    }
    static constexpr c10::BFloat16 max() {
      return at::BFloat16(0x7F7F, at::BFloat16::from_bits());
    }
};

} // namespace std
