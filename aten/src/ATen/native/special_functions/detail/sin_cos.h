#pragma once

#include <ATen/native/special_functions/detail/sin_cos_t.h>

namespace at {
namespace native {
namespace special_functions {
namespace detail {
template<typename T1>
inline constexpr
sin_cos_t<T1>
sin_cos(T1 x)
noexcept {
  return sin_cos_t < T1 > {std::sin(x), std::cos(x)};
}

inline constexpr
sin_cos_t<double>
sin_cos(double x)
noexcept {
  return sin_cos<double>(x);
}
}
}
}
}
