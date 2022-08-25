#pragma once

#include <ATen/native/special/detail/airy.h>
#include <ATen/native/special/detail/promote_t.h>

namespace at {
namespace native {
namespace special {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1>
airy_ai(T1 z) {
  using T2 = detail::promote_t<T1>;

  return detail::airy<T2>(z).ai;
} // detail::promote_t<T1> airy_ai(T1 z)
} // namespace special
} // namespace native
} // namespace at
