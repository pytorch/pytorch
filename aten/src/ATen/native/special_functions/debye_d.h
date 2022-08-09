#pragma once

#include <ATen/native/special_functions/detail/debye_d.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
debye_d(unsigned int n, T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::debye_d<T2>(n, x);
}
} // namespace special_functions
} // namespace native
} // namespace at
