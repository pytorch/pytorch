#pragma once

#include <ATen/native/special_functions/detail/nome_q.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr T1
nome_q(T1 k) {
  using T2 = detail::promote_t<T1>;

  return detail::nome_q<T2>(k);
}
} // namespace special_functions
} // namespace native
} // namespace at
