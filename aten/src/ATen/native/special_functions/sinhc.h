#pragma once

#include <ATen/native/special_functions/detail/promote_t.h>
#include <ATen/native/special_functions/detail/sinhc.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr
detail::promote_t<T1>
sinhc(T1 x) {
  using T2 = detail::promote_t<T1>;

  return detail::sinhc<T2>(x);
}
} // namespace special_functions
} // namespace native
} // namespace at
