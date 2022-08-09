#pragma once

#include <ATen/native/special_functions/detail/clausen_cl.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
clausen_cl(unsigned int m, T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::clausen_cl<T2>(m, x);
}
} // namespace special_functions
} // namespace native
} // namespace at
