#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1, typename T2>
inline constexpr
detail::promote_t<T1, T2>
fermi_dirac_integral_f(T1 s, T2 x) {
  using T3 = detail::promote_t<T1, T2>;

  return detail::fermi_dirac_integral_f<T3>(s, x);
}
} // namespace special_functions
} // namespace native
} // namespace at
