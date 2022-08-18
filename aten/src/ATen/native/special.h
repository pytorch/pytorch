#pragma once

#include <ATen/core/TensorBase.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/Scalar.h>

namespace at {
struct TensorIterator;
struct TensorIteratorBase;
namespace native {
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_bernoulli_number_stub);
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
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_factorial_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_harmonic_number_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_incomplete_legendre_elliptic_integral_d_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_incomplete_legendre_elliptic_integral_e_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_incomplete_legendre_elliptic_integral_f_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_ln_gamma_sign_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_sin_pi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_sinh_pi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_tan_pi_stub);
DECLARE_DISPATCH(void(*)(TensorIteratorBase&), special_tanh_pi_stub);
} // namespace native
} // namespace at
