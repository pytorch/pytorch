#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/math_compat.h>

namespace c10 {

// Note: Explicit implementation of copysign for Half and BFloat16
// is needed to workaround g++-7/8 crash on aarch64, but also makes
// copysign faster for the half-precision types
template <typename T, typename U>
inline auto copysign(const T& a, const U& b) {
  return std::copysign(a, b);
}

// Implement copysign for half precision floats using bit ops
// Sign is the most significant bit for both half and bfloat16 types
inline c10::Half copysign(c10::Half a, c10::Half b) {
  return c10::Half((a.x & 0x7fff) | (b.x & 0x8000), c10::Half::from_bits());
}

inline c10::BFloat16 copysign(c10::BFloat16 a, c10::BFloat16 b) {
  return c10::BFloat16(
      (a.x & 0x7fff) | (b.x & 0x8000), c10::BFloat16::from_bits());
}

} // namespace c10
