#pragma once

#include <c10/macros/Macros.h>

#if !defined(__CUDACC__) && !defined(__HIPCC__)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif
#include <ATen/OpMathType.h>
#include <type_traits>

namespace at::native::ufunc {

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T add(T self, T other, at::opmath_type<T> alpha) __ubsan_ignore_undefined__ {
  using opmath_t = at::opmath_type<T>;
  return static_cast<opmath_t>(self) + alpha * static_cast<opmath_t>(other);
}

#if !defined(__CUDACC__) && !defined(__HIPCC__)
using vec::Vectorized;
template <typename T>
C10_ALWAYS_INLINE Vectorized<T> add(Vectorized<T> self, Vectorized<T> other, at::opmath_type<T> alpha) __ubsan_ignore_undefined__ {
  using opmath_t = at::opmath_type<T>;
  if constexpr (std::is_same_v<T, at::Half> || std::is_same_v<T, at::BFloat16>) {
    Vectorized<opmath_t> vec_alpha(alpha);
    auto [self0, self1] = convert_to_float<T>(self);
    auto [other0, other1] = convert_to_float<T>(other);
    return convert_from_float<T>(
        vec::fmadd(other0, vec_alpha, self0),
        vec::fmadd(other1, vec_alpha, self1));
  } else {
    return vec::fmadd(other, Vectorized<T>(alpha), self);
  }
}
#endif

} // namespace at::native::ufunc
