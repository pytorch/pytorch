#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {

struct TensorIterator;

namespace native {

#if defined(__CUDACC__) || defined(__HIPCC__)
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

// integral power in pytorch allows for negative exponents, giving truncated integral results.
// e.g. since 2**-1==0.5, the truncated integral result is zero. 1**negative_exponent is the
// only non-zero result.
template <class T,
  typename std::enable_if<std::is_integral<T>::value, T>::type* = nullptr>
static inline HOST_DEVICE __ubsan_ignore_signed_int_overflow__ T powi_impl(T a, T b) {
  T result = 1;
  while (b) {
    if (b & 1) {
       result *= a;
    }
    b /= 2;
    a *= a;
  }
  return result;
}

template <class T,
  typename std::enable_if<std::is_integral<T>::value && !std::is_signed<T>::value, T>::type* = nullptr>
static inline HOST_DEVICE T powi(T a, T b) {
  return powi_impl(a, b);
}

template <class T,
  typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, T>::type* = nullptr>
static inline HOST_DEVICE T powi(T a, T b) {
  if ( b < 0 ) {
      if ( a == 1 ) {
          return 1;
      } else if ( a == -1 ) {
          auto negative = (-b) % static_cast<T>(2);
          return negative ? -1 : 1;
      } else {
          return 0;
      }
  }
  return powi_impl(a, b);
}

using pow_tensor_tensor_fn = void (*)(TensorIteratorBase&);
using pow_tensor_scalar_fn = void (*)(TensorIteratorBase&, const Scalar&);

DECLARE_DISPATCH(pow_tensor_tensor_fn, pow_tensor_tensor_stub);
DECLARE_DISPATCH(pow_tensor_scalar_fn, pow_tensor_scalar_stub);

} // namespace native

} // namespace at
