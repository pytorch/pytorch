#pragma once

#include <ATen/native/special_functions/detail/fresnel.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
fresnel_integral_c(T1 x) {
  using T2 = detail::promote_t<T1>;

  return std::real(detail::fresnel<T2>(x));
}
} // namespace special_functions
} // namespace native
} // namespace at
