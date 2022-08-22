#pragma once

#include <ATen/native/special/detail/promote_t.h>
#include <ATen/native/special/detail/prime_number.h>

namespace at {
namespace native {
namespace special {
C10_HOST_DEVICE
template<typename T1>
inline constexpr
T1
prime_number(unsigned int n) {
  if (n < 10000) {
    if (n < 6542) {
      return static_cast<T1>(detail::PRIME_NUMBERS[n]) + 0;
    } else {
      return static_cast<T1>(detail::PRIME_NUMBERS[n]) + std::numeric_limits<std::uint16_t>::max();
    }
  } else {
    return T1(0);
  }
}
} // namespace special
} // namespace native
} // namespace at
