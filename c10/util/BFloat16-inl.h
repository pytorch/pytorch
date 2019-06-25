#pragma once

#include <c10/macros/Macros.h>
#include <limits>

namespace std {

template <>
class numeric_limits<c10::BFloat16> {
  public:
    static constexpr bool is_signed = true;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr c10::BFloat16 lowest() {
      return at::BFloat16(0xFBFF, at::BFloat16::from_bits());
    }
    static constexpr c10::BFloat16 max() {
      return at::BFloat16(0x7BFF, at::BFloat16::from_bits());
    }
};

} // namespace std
