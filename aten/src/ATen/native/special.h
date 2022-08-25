#pragma once

#include <ATen/core/TensorBase.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/Scalar.h>

namespace at {
struct TensorIterator;
struct TensorIteratorBase;
namespace native {
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_airy_bi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_bernoulli_number_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_bessel_j_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_bessel_y_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_beta_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_binomial_coefficient_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_bulirsch_elliptic_integral_el1_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_carlson_elliptic_r_c_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_complete_carlson_elliptic_r_f_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_complete_carlson_elliptic_r_g_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_complete_legendre_elliptic_integral_d_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_complete_legendre_elliptic_integral_e_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_complete_legendre_elliptic_integral_k_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_complete_legendre_elliptic_integral_pi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_cos_pi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_cosh_pi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_double_factorial_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_exp_airy_ai_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_exp_airy_bi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_exp_modified_bessel_i_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_exp_modified_bessel_k_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_factorial_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_falling_factorial_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_gamma_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_hankel_h_1_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_hankel_h_2_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_harmonic_number_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_hurwitz_zeta_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_incomplete_legendre_elliptic_integral_d_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_incomplete_legendre_elliptic_integral_e_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_incomplete_legendre_elliptic_integral_f_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_ln_binomial_coefficient_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_ln_double_factorial_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_ln_factorial_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_ln_falling_factorial_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_ln_gamma_sign_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_ln_gamma_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_ln_rising_factorial_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_lower_incomplete_gamma_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_modified_bessel_i_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_modified_bessel_k_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_prime_number_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_reciprocal_gamma_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_riemann_zeta_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_rising_factorial_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_sin_pi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_sinh_pi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_spherical_bessel_j_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_spherical_bessel_y_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_spherical_hankel_h_1_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_spherical_hankel_h_2_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_spherical_modified_bessel_i_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_spherical_modified_bessel_k_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_tan_pi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_tanh_pi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_upper_incomplete_gamma_stub);
} // namespace native
} // namespace at
