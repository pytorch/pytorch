#pragma once

#include <ATen/native/special_functions/detail/sine_integral_si.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
sine_integral_si(T1 z) {
  using T2 = detail::promote_t<T1>;

  return detail::sine_integral_si<T2>(z);
}
} // namespace special_functions
} // namespace native
} // namespace at
