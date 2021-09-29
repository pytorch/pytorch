// TODO: this is a pessimization

#pragma once

#include <c10/macros/Macros.h>

#if !defined(__CUDACC__) && !defined(__HIPCC__)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif

namespace at {
namespace native {
namespace ufunc {

template <typename T>
C10_HOST_DEVICE T sub(T self, T other, T alpha) __ubsan_ignore_undefined__ {
  return self - alpha * other;
}

#if !defined(__CUDACC__) && !defined(__HIPCC__)
using vec::Vectorized;
template <typename T>
C10_HOST_DEVICE Vectorized<T> sub(Vectorized<T> self, Vectorized<T> other, Vectorized<T> alpha) __ubsan_ignore_undefined__ {
  // TODO: this is a pessimization
  return vec::fmadd(other, alpha.neg(), self);
}
#endif

}}}  // namespace at::native::ufunc
