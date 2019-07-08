#pragma once

#include <c10/macros/Macros.h>

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
