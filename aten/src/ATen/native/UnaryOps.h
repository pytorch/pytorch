#pragma once

#include <ATen/native/DispatchStub.h>
#include <ATen/Generator.h>
#include <c10/core/Scalar.h>
#include <stdexcept>

namespace at {
class Tensor;
class TensorBase;
struct TensorIteratorBase;
}

namespace at::native {

using unary_fn = void(*)(TensorIteratorBase&);
using unary_fn_with_scalar = void(*)(TensorIteratorBase&, const Scalar& a);

inline namespace CPU_CAPABILITY {
void conj_kernel(TensorIteratorBase &iter);
void neg_kernel(TensorIteratorBase &iter);
void reciprocal_kernel(TensorIteratorBase &iter);
void rsqrt_kernel(TensorIteratorBase& iter);
void sqrt_kernel(TensorIteratorBase& iter);
} // namespace CPU_CAPABILITY

DECLARE_DISPATCH(unary_fn, abs_stub);
DECLARE_DISPATCH(unary_fn, angle_stub);
DECLARE_DISPATCH(unary_fn, conj_physical_stub);
DECLARE_DISPATCH(unary_fn, acos_stub);
DECLARE_DISPATCH(unary_fn, acosh_stub);
DECLARE_DISPATCH(unary_fn, asinh_stub);
DECLARE_DISPATCH(unary_fn, atanh_stub);
DECLARE_DISPATCH(unary_fn, asin_stub);
DECLARE_DISPATCH(unary_fn, atan_stub);
DECLARE_DISPATCH(unary_fn, bitwise_not_stub);
DECLARE_DISPATCH(unary_fn, logical_not_stub);
DECLARE_DISPATCH(unary_fn, ceil_stub);
DECLARE_DISPATCH(unary_fn, cos_stub);
DECLARE_DISPATCH(unary_fn, cosh_stub);
DECLARE_DISPATCH(unary_fn, digamma_stub);
DECLARE_DISPATCH(unary_fn, special_entr_stub);
DECLARE_DISPATCH(unary_fn, special_erfcx_stub);
DECLARE_DISPATCH(unary_fn, erf_stub);
DECLARE_DISPATCH(unary_fn, erfc_stub);
DECLARE_DISPATCH(unary_fn, erfinv_stub);
DECLARE_DISPATCH(unary_fn, exp_stub);
DECLARE_DISPATCH(unary_fn, exp2_stub);
DECLARE_DISPATCH(unary_fn, expm1_stub);
DECLARE_DISPATCH(unary_fn, floor_stub);
DECLARE_DISPATCH(unary_fn, frac_stub);
DECLARE_DISPATCH(unary_fn, frexp_stub);
DECLARE_DISPATCH(unary_fn, i0_stub);
DECLARE_DISPATCH(unary_fn, special_i0e_stub);
DECLARE_DISPATCH(unary_fn, special_i1_stub);
DECLARE_DISPATCH(unary_fn, special_i1e_stub);
DECLARE_DISPATCH(unary_fn, log_stub);
DECLARE_DISPATCH(unary_fn, log10_stub);
DECLARE_DISPATCH(unary_fn, log1p_stub);
DECLARE_DISPATCH(unary_fn, log2_stub);
DECLARE_DISPATCH(unary_fn, special_ndtri_stub);
DECLARE_DISPATCH(unary_fn, special_log_ndtr_stub);
DECLARE_DISPATCH(unary_fn, neg_stub);

DECLARE_DISPATCH(unary_fn, reciprocal_stub);
DECLARE_DISPATCH(unary_fn, round_stub);
DECLARE_DISPATCH(unary_fn, rsqrt_stub);
DECLARE_DISPATCH(unary_fn, sigmoid_stub);
DECLARE_DISPATCH(unary_fn_with_scalar, logit_stub);
DECLARE_DISPATCH(unary_fn, sign_stub);
DECLARE_DISPATCH(unary_fn, signbit_stub);
DECLARE_DISPATCH(unary_fn, sgn_stub);
DECLARE_DISPATCH(unary_fn, sin_stub);
DECLARE_DISPATCH(unary_fn, sinc_stub);
DECLARE_DISPATCH(unary_fn, sinh_stub);
DECLARE_DISPATCH(unary_fn, sqrt_stub);
DECLARE_DISPATCH(unary_fn, tan_stub);
DECLARE_DISPATCH(unary_fn, tanh_stub);
DECLARE_DISPATCH(unary_fn, trigamma_stub);
DECLARE_DISPATCH(unary_fn, trunc_stub);
DECLARE_DISPATCH(unary_fn, lgamma_stub);
DECLARE_DISPATCH(unary_fn, special_airy_ai_stub);
DECLARE_DISPATCH(unary_fn, special_bessel_j0_stub);
DECLARE_DISPATCH(unary_fn, special_bessel_j1_stub);
DECLARE_DISPATCH(unary_fn, special_bessel_y0_stub);
DECLARE_DISPATCH(unary_fn, special_bessel_y1_stub);
DECLARE_DISPATCH(unary_fn, special_modified_bessel_i0_stub);
DECLARE_DISPATCH(unary_fn, special_modified_bessel_i1_stub);
DECLARE_DISPATCH(unary_fn, special_modified_bessel_k0_stub);
DECLARE_DISPATCH(unary_fn, special_modified_bessel_k1_stub);
DECLARE_DISPATCH(unary_fn, special_scaled_modified_bessel_k0_stub);
DECLARE_DISPATCH(unary_fn, special_scaled_modified_bessel_k1_stub);
DECLARE_DISPATCH(unary_fn, special_spherical_bessel_j0_stub);

// NB: these are actually defined in Distribution
DECLARE_DISPATCH(void(*)(const TensorBase&, const TensorBase&, c10::optional<Generator>), bernoulli_tensor_stub);
DECLARE_DISPATCH(void(*)(const TensorBase&, const double, c10::optional<Generator>), bernoulli_scalar_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, const double, const double, c10::optional<Generator>), cauchy_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, const double, c10::optional<Generator>), exponential_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, const double, c10::optional<Generator>), geometric_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, const double, const double, c10::optional<Generator>), log_normal_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, const double, const double, c10::optional<Generator>), uniform_stub);
DECLARE_DISPATCH(void(*)(const TensorBase&, const double, const double, c10::optional<Generator>), normal_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, const uint64_t, const int64_t, c10::optional<Generator>), random_from_to_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, c10::optional<Generator>), random_full_64_bits_range_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, c10::optional<Generator>), random_stub);

DECLARE_DISPATCH(void(*)(TensorIteratorBase&, const int64_t, const double), kaiser_window_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, const int64_t), polygamma_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&, const Scalar& a, const Scalar& b), clamp_stub);
DECLARE_DISPATCH(
    void (*)(Tensor&, const Tensor&, int64_t, c10::optional<Generator>),
    multinomial_with_replacement_stub);
DECLARE_DISPATCH(
    void (*)(
        TensorIteratorBase&,
        c10::optional<double>,
        c10::optional<double>,
        c10::optional<double>),
    nan_to_num_stub);
DECLARE_DISPATCH(void (*)(TensorIteratorBase&, int64_t), round_decimals_stub);

// Missing unary functions
// digamma
// lgamma
// erfinv
// clone
// contiguous
// zero
} // namespace at::native
