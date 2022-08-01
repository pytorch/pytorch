#pragma once

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::sin_cos_t<detail::promote_t<T1>>
sin_cos(T1 x) {
  using T2 = detail::promote_t<T1>;

  return sin_cos<T2>(x);
} // detail::sin_cos_t<detail::promote_t<T1>> sin_cos(T1 x)
} // namespace special_functions
} // namespace native
} // namespace at
