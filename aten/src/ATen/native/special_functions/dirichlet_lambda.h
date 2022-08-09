#pragma once

#include <ATen/native/special_functions/detail/dirichlet_lambda.h>
#include <ATen/native/special_functions/detail/promote_t.h>

namespace at {
namespace native {
namespace special_functions {
template<typename T1>
inline constexpr T1
dirichlet_lambda(T1 s) {
  using T2 = detail::promote_t<T1>;

  return detail::dirichlet_lambda<T2>(s);
}
} // namespace special_functions
} // namespace native
} // namespace at
