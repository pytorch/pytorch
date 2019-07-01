#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/Generator.h>
#include <stdexcept>

namespace at { struct TensorIterator; }

namespace at { namespace native {

using unary_fn = void(*)(TensorIterator&);

DECLARE_DISPATCH(void(*)(TensorIterator&, Scalar), fill_stub);

DECLARE_DISPATCH(unary_fn, abs_stub);
DECLARE_DISPATCH(unary_fn, acos_stub);
DECLARE_DISPATCH(unary_fn, asin_stub);
DECLARE_DISPATCH(unary_fn, atan_stub);
DECLARE_DISPATCH(unary_fn, bitwise_not_stub);
DECLARE_DISPATCH(unary_fn, ceil_stub);
DECLARE_DISPATCH(unary_fn, cos_stub);
DECLARE_DISPATCH(unary_fn, cosh_stub);
DECLARE_DISPATCH(unary_fn, erf_stub);
DECLARE_DISPATCH(unary_fn, erfc_stub);
DECLARE_DISPATCH(unary_fn, exp_stub);
DECLARE_DISPATCH(unary_fn, expm1_stub);
DECLARE_DISPATCH(unary_fn, floor_stub);
DECLARE_DISPATCH(unary_fn, frac_stub);
DECLARE_DISPATCH(unary_fn, log_stub);
DECLARE_DISPATCH(unary_fn, log10_stub);
DECLARE_DISPATCH(unary_fn, log1p_stub);
DECLARE_DISPATCH(unary_fn, log2_stub);
DECLARE_DISPATCH(unary_fn, neg_stub);
DECLARE_DISPATCH(unary_fn, reciprocal_stub);
DECLARE_DISPATCH(unary_fn, round_stub);
DECLARE_DISPATCH(unary_fn, rsqrt_stub);
DECLARE_DISPATCH(unary_fn, sigmoid_stub);
DECLARE_DISPATCH(unary_fn, sin_stub);
DECLARE_DISPATCH(unary_fn, sinh_stub);
DECLARE_DISPATCH(unary_fn, sqrt_stub);
DECLARE_DISPATCH(unary_fn, tan_stub);
DECLARE_DISPATCH(unary_fn, tanh_stub);
DECLARE_DISPATCH(unary_fn, trunc_stub);

DECLARE_DISPATCH(void(*)(Tensor&, const double, Generator *), bernoulli_mkl_stub);

inline void propagate_names_if_namedtensor_enabled(Tensor& result, const Tensor& src) {
#ifdef NAMEDTENSOR_ENABLED
      at::namedinference::propagate_names(result, src);
#endif
}

#ifdef __CUDACC__
  #define CUDA_HOST_DEVICE __host__ __device__
#else
  #define CUDA_HOST_DEVICE
#endif  // __CUDACC__

// Boolean type does not work with ~ (bitwise NOT) in C++. bitwise_not wraps this operation for both Boolean and
// integral types.
inline CUDA_HOST_DEVICE bool bitwise_not(bool a) {
  return !a;
}

template <typename scalar_t>
inline CUDA_HOST_DEVICE
typename std::enable_if<std::is_integral<scalar_t>::value, scalar_t>::type
bitwise_not(scalar_t a) {
  return ~a;
}

// Missing unary functions
// digamma
// lgamma
// erfinv
// clone
// contiguous
// clamp/_min/_max
// sign
// zero
}} // namespace at::native
