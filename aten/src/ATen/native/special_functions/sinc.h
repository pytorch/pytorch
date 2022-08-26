#pragma once

#include <ATen/native/special_functions/detail/sinc.h>
#include <ATen/native/special_functions/detail/promote_t.h>
#include <c10/macros/Macros.h>
#include <c10/util/complex.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1>
sinc(T1 z) {
  using T2 = detail::promote_t<T1>;

  return detail::sinc<T2>(z);
}
} // namespace special_functions
} // namespace native
} // namespace at
