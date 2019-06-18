#pragma once

#include <c10/macros/Macros.h>

namespace c10 {

/// Constructors

inline BFloat16::BFloat16(float value) {
  val_ = detail::bits_from_f32(value);
}

/// Implicit conversions

inline BFloat16::operator float() const {
  return detail::f32_from_bits(val_);
}

} // namespace c10
