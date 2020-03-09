#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/Generator.h>
#include <stdexcept>

namespace at { struct TensorIterator; }

namespace at { namespace native {

using unary_fn = void(*)(TensorIterator&);
using unary_fn_with_scalar = void(*)(TensorIterator&, Scalar a);

DECLARE_DISPATCH(unary_fn, abs_stub);
DECLARE_DISPATCH(unary_fn, angle_stub);
DECLARE_DISPATCH(unary_fn, real_stub);
DECLARE_DISPATCH(unary_fn, imag_stub);
DECLARE_DISPATCH(unary_fn, conj_stub);
DECLARE_DISPATCH(unary_fn, acos_stub);
DECLARE_DISPATCH(unary_fn, asin_stub);
DECLARE_DISPATCH(unary_fn, atan_stub);
DECLARE_DISPATCH(unary_fn, bitwise_not_stub);
DECLARE_DISPATCH(unary_fn, logical_not_stub);
DECLARE_DISPATCH(unary_fn, ceil_stub);
DECLARE_DISPATCH(unary_fn_with_scalar, clamp_max_stub);
DECLARE_DISPATCH(unary_fn_with_scalar, clamp_min_stub);
DECLARE_DISPATCH(unary_fn, cos_stub);
DECLARE_DISPATCH(unary_fn, cosh_stub);
DECLARE_DISPATCH(unary_fn, digamma_stub);
DECLARE_DISPATCH(unary_fn, erf_stub);
DECLARE_DISPATCH(unary_fn, erfc_stub);
DECLARE_DISPATCH(unary_fn, erfinv_stub);
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
DECLARE_DISPATCH(unary_fn, sign_stub);
DECLARE_DISPATCH(unary_fn, sin_stub);
DECLARE_DISPATCH(unary_fn, sinh_stub);
DECLARE_DISPATCH(unary_fn, sqrt_stub);
DECLARE_DISPATCH(unary_fn, tan_stub);
DECLARE_DISPATCH(unary_fn, tanh_stub);
DECLARE_DISPATCH(unary_fn, trigamma_stub);
DECLARE_DISPATCH(unary_fn, trunc_stub);
DECLARE_DISPATCH(unary_fn, lgamma_stub);

DECLARE_DISPATCH(void(*)(Tensor&, const double, GeneratorHolder), bernoulli_mkl_stub);
DECLARE_DISPATCH(void(*)(TensorIterator&, const double, const double, GeneratorHolder), cauchy_stub);
DECLARE_DISPATCH(void(*)(TensorIterator&, const double, GeneratorHolder), exponential_stub);
DECLARE_DISPATCH(void(*)(TensorIterator&, const double, GeneratorHolder), geometric_stub);
DECLARE_DISPATCH(void(*)(TensorIterator&, const double, const double, GeneratorHolder), log_normal_stub);
DECLARE_DISPATCH(void(*)(Tensor&, const double, const double, GeneratorHolder), normal_stub);
DECLARE_DISPATCH(void(*)(TensorIterator&, const uint64_t, const int64_t, GeneratorHolder), random_from_to_stub);
DECLARE_DISPATCH(void(*)(TensorIterator&, GeneratorHolder), random_full_64_bits_range_stub);
DECLARE_DISPATCH(void(*)(TensorIterator&, GeneratorHolder), random_stub);
DECLARE_DISPATCH(void(*)(TensorIterator&, const int64_t), polygamma_stub);
DECLARE_DISPATCH(void(*)(TensorIterator&, Scalar a, Scalar b), clamp_stub);
DECLARE_DISPATCH(void(*)(Tensor&, const Tensor&, int64_t, bool, GeneratorHolder), multinomial_stub);

// Missing unary functions
// digamma
// lgamma
// erfinv
// clone
// contiguous
// clamp/_min/_max
// zero
}} // namespace at::native
