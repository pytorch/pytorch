#pragma once

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
inline constexpr
T1
max_abs(T1 a, T1 b)
noexcept {
  if (std::isnan(a) || std::isnan(b)) {
    return std::numeric_limits<T1>::quiet_NaN();
  } else {
    return std::max(std::abs(a), std::abs(b));
  }
}
}
}
}
}
