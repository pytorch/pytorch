#pragma once

#include <ATen/native/special/detail/promote_t.h>
#include <ATen/native/special/detail/bernoulli_number.h>
#include <c10/macros/Macros.h>

namespace at {
namespace native {
namespace special {
template<typename T1>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1>
bernoulli_number(unsigned int n) {
  using T2 = detail::promote_t<T1>;

  return detail::bernoulli_number<T2>(n);
}

template<typename T1>
C10_HOST_DEVICE
inline constexpr
detail::promote_t<T1>
bernoulli_number(T1 n) {
  using T2 = detail::promote_t<T1>;

  if (n < 0) {
    return std::numeric_limits<T2>::quiet_NaN();
  }

  return bernoulli_number<T2>(static_cast<unsigned int>(n));
}
} // namespace special
} // namespace native
} // namespace at
