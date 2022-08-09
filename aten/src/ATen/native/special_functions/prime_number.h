#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>
#include <ATen/native/special_functions/detail/prime_number.h>

namespace at {
namespace native {
namespace special_functions {
inline constexpr
std::uint32_t
prime_number(std::uint16_t n) {
  if (n < 10000) {
    if (n < 6542) {
      return static_cast<std::uint32_t>(detail::PRIME_NUMBERS[n]) + 0;
    } else {
      return static_cast<std::uint32_t>(detail::PRIME_NUMBERS[n]) + std::numeric_limits<std::uint16_t>::max();
    }
  } else {
    return 0;
  }
}
} // namespace special_functions
} // namespace native
} // namespace at
