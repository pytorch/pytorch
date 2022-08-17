#pragma once

#include <c10/util/numbers.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
inline constexpr
T1
factorial(unsigned int n)
noexcept {
  if (n < c10::numbers::factorials_size<T1>()) {
    return c10::numbers::factorials_v[n];
  } else {
    return std::numeric_limits<T1>::infinity();
  }
}
}
}
}
}
