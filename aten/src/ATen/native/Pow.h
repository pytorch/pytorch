#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at {

struct TensorIterator;

namespace native {

template <typename T>
static inline
#ifdef __CUDACC__
 __host__ __device__
#endif
T powi(T a, T b) {
  if ( b < 0 ) {
      if ( a == 1 ) {
          return 1;
      } else if ( a == -1 ) {
          auto negative = std::abs(b) % 2;
          return negative ? -1 : 1;
      } else {
          return 0;
      }
  }
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

using pow_tensor_tensor_fn = void (*)(TensorIterator&);
using pow_tensor_scalar_fn = void (*)(TensorIterator&, Scalar);

DECLARE_DISPATCH(pow_tensor_tensor_fn, pow_tensor_tensor_stub);
DECLARE_DISPATCH(pow_tensor_scalar_fn, pow_tensor_scalar_stub);

} // namespace native

} // namespace at
