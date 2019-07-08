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

} // namespace c10

namespace std {

template <>
class numeric_limits<c10::BFloat16> {
  public:
    static constexpr bool is_signed = true;
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
