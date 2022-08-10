#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/special_functions.h>

#include <cmath>
#include <limits>
#include <type_traits>

#include <ATen/Config.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vml.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/cpu/zmath.h>
#include <ATen/OpMathType.h>

#include <c10/util/math_compat.h>
#include <c10/util/MathConstants.h>
#include <c10/core/Scalar.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/irange.h>

namespace at {
namespace native {
inline namespace CPU_CAPABILITY {
/*
 *  UNARY OPERATORS
 */

void airy_ai_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_ai_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void airy_ai_cpu_kernel(TensorIteratorBase &iterator)

void airy_bi_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "airy_bi_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void airy_bi_cpu_kernel(TensorIteratorBase &iterator)

void bernoulli_number_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bernoulli_number_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void bernoulli_number_cpu_kernel(TensorIteratorBase &iterator)

void bessel_j_0_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j_0_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void bessel_j_0_cpu_kernel(TensorIteratorBase &iterator)

void bessel_j_1_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j_1_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void bessel_j_1_cpu_kernel(TensorIteratorBase &iterator)

void bessel_y_0_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y_0_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void bessel_y_0_cpu_kernel(TensorIteratorBase &iterator)

void bessel_y_1_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y_1_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void bessel_y_1_cpu_kernel(TensorIteratorBase &iterator)

void complete_elliptic_integral_e_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_elliptic_integral_e_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void complete_elliptic_integral_e_cpu_kernel(TensorIteratorBase &iterator)

void complete_elliptic_integral_k_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_elliptic_integral_k_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void complete_elliptic_integral_k_cpu_kernel(TensorIteratorBase &iterator)

void complete_legendre_elliptic_integral_d_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_legendre_elliptic_integral_d_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void complete_legendre_elliptic_integral_d_cpu_kernel(TensorIteratorBase &iterator)

void cos_pi_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "cos_pi_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void cos_pi_cpu_kernel(TensorIteratorBase &iterator)

void cosh_pi_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "cosh_pi_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void cosh_pi_cpu_kernel(TensorIteratorBase &iterator)

void cosine_integral_ci_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "cosine_integral_ci_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void cosine_integral_ci_cpu_kernel(TensorIteratorBase &iterator)

void dilogarithm_li_2_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dilogarithm_li_2_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void dilogarithm_li_2_cpu_kernel(TensorIteratorBase &iterator)

void dirichlet_beta_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dirichlet_beta_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void dirichlet_beta_cpu_kernel(TensorIteratorBase &iterator)

void dirichlet_eta_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dirichlet_eta_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void dirichlet_eta_cpu_kernel(TensorIteratorBase &iterator)

void dirichlet_lambda_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "dirichlet_lambda_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void dirichlet_lambda_cpu_kernel(TensorIteratorBase &iterator)

void double_factorial_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "double_factorial_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void double_factorial_cpu_kernel(TensorIteratorBase &iterator)

void exp_airy_ai_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_airy_ai_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void exp_airy_ai_cpu_kernel(TensorIteratorBase &iterator)

void exp_airy_bi_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_airy_bi_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void exp_airy_bi_cpu_kernel(TensorIteratorBase &iterator)

void exp_modified_bessel_k_0_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_k_0_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void exp_modified_bessel_k_0_cpu_kernel(TensorIteratorBase &iterator)

void exp_modified_bessel_k_1_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_k_1_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void exp_modified_bessel_k_1_cpu_kernel(TensorIteratorBase &iterator)

void exponential_integral_ei_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exponential_integral_ei_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void exponential_integral_ei_cpu_kernel(TensorIteratorBase &iterator)

void factorial_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "factorial_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void factorial_cpu_kernel(TensorIteratorBase &iterator)

void fresnel_integral_c_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "fresnel_integral_c_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void fresnel_integral_c_cpu_kernel(TensorIteratorBase &iterator)

void fresnel_integral_s_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "fresnel_integral_s_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void fresnel_integral_s_cpu_kernel(TensorIteratorBase &iterator)

void harmonic_number_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "harmonic_number_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void harmonic_number_cpu_kernel(TensorIteratorBase &iterator)

void hyperbolic_cosine_integral_chi_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hyperbolic_cosine_integral_chi_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void hyperbolic_cosine_integral_chi_cpu_kernel(TensorIteratorBase &iterator)

void hyperbolic_sine_integral_shi_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hyperbolic_sine_integral_shi_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void hyperbolic_sine_integral_shi_cpu_kernel(TensorIteratorBase &iterator)

void ln_double_factorial_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_double_factorial_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void ln_double_factorial_cpu_kernel(TensorIteratorBase &iterator)

void ln_factorial_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_factorial_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void ln_factorial_cpu_kernel(TensorIteratorBase &iterator)

void ln_gamma_sign_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_gamma_sign_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void ln_gamma_sign_cpu_kernel(TensorIteratorBase &iterator)

void ln_gamma_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_gamma_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void ln_gamma_cpu_kernel(TensorIteratorBase &iterator)

void logarithmic_integral_li_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "logarithmic_integral_li_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void logarithmic_integral_li_cpu_kernel(TensorIteratorBase &iterator)

void modified_bessel_i_0_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_0_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void modified_bessel_i_0_cpu_kernel(TensorIteratorBase &iterator)

void modified_bessel_i_1_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_1_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void modified_bessel_i_1_cpu_kernel(TensorIteratorBase &iterator)

void modified_bessel_k_0_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_0_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void modified_bessel_k_0_cpu_kernel(TensorIteratorBase &iterator)

void modified_bessel_k_1_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_1_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void modified_bessel_k_1_cpu_kernel(TensorIteratorBase &iterator)

void nome_q_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "nome_q_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void nome_q_cpu_kernel(TensorIteratorBase &iterator)

void prime_number_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "prime_number_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void prime_number_cpu_kernel(TensorIteratorBase &iterator)

void reciprocal_gamma_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "reciprocal_gamma_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void reciprocal_gamma_cpu_kernel(TensorIteratorBase &iterator)

void riemann_zeta_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "riemann_zeta_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void riemann_zeta_cpu_kernel(TensorIteratorBase &iterator)

void sin_pi_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sin_pi_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void sin_pi_cpu_kernel(TensorIteratorBase &iterator)

void sinc_pi_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinc_pi_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void sinc_pi_cpu_kernel(TensorIteratorBase &iterator)

void sinh_pi_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinh_pi_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void sinh_pi_cpu_kernel(TensorIteratorBase &iterator)

void sinhc_pi_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinhc_pi_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void sinhc_pi_cpu_kernel(TensorIteratorBase &iterator)

void sinhc_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "sinhc_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void sinhc_cpu_kernel(TensorIteratorBase &iterator)

void spherical_bessel_j_0_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_j_0_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void spherical_bessel_j_0_cpu_kernel(TensorIteratorBase &iterator)

void tan_pi_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "tan_pi_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void tan_pi_cpu_kernel(TensorIteratorBase &iterator)

void tanh_pi_cpu_kernel(TensorIteratorBase &iterator) {
  TORCH_INTERNAL_ASSERT(iterator.ntensors() == 2);

  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "tanh_pi_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x) {
      return x;
    });
  });
} // void tanh_pi_cpu_kernel(TensorIteratorBase &iterator)

/*
 *  BINARY OPERATORS
 */

void bell_polynomial_b_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bell_polynomial_b_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void bell_polynomial_b_cpu_kernel(TensorIteratorBase &iterator)

void bernoulli_polynomial_b_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bernoulli_polynomial_b_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void bernoulli_polynomial_b_cpu_kernel(TensorIteratorBase &iterator)

void bessel_j_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void bessel_j_cpu_kernel(TensorIteratorBase &iterator)

void bessel_y_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_y_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void bessel_y_cpu_kernel(TensorIteratorBase &iterator)

void beta_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "beta_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void beta_cpu_kernel(TensorIteratorBase &iterator)

void binomial_coefficient_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "binomial_coefficient_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void binomial_coefficient_cpu_kernel(TensorIteratorBase &iterator)

void bose_einstein_integral_g_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bose_einstein_integral_g_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void bose_einstein_integral_g_cpu_kernel(TensorIteratorBase &iterator)

void bulirsch_elliptic_integral_el1_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_el1_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void bulirsch_elliptic_integral_el1_cpu_kernel(TensorIteratorBase &iterator)

void carlson_elliptic_r_c_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_c_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void carlson_elliptic_r_c_cpu_kernel(TensorIteratorBase &iterator)

void chebyshev_polynomial_t_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_t_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void chebyshev_polynomial_t_cpu_kernel(TensorIteratorBase &iterator)

void chebyshev_polynomial_u_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_u_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void chebyshev_polynomial_u_cpu_kernel(TensorIteratorBase &iterator)

void chebyshev_polynomial_v_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_v_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void chebyshev_polynomial_v_cpu_kernel(TensorIteratorBase &iterator)

void chebyshev_polynomial_w_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_w_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void chebyshev_polynomial_w_cpu_kernel(TensorIteratorBase &iterator)

void clausen_cl_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "clausen_cl_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void clausen_cl_cpu_kernel(TensorIteratorBase &iterator)

void clausen_sl_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "clausen_sl_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void clausen_sl_cpu_kernel(TensorIteratorBase &iterator)

void complete_carlson_elliptic_r_f_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_carlson_elliptic_r_f_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void complete_carlson_elliptic_r_f_cpu_kernel(TensorIteratorBase &iterator)

void complete_carlson_elliptic_r_g_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_carlson_elliptic_r_g_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void complete_carlson_elliptic_r_g_cpu_kernel(TensorIteratorBase &iterator)

void complete_elliptic_integral_pi_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "complete_elliptic_integral_pi_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void complete_elliptic_integral_pi_cpu_kernel(TensorIteratorBase &iterator)

void confluent_hypergeometric_0_f_1_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "confluent_hypergeometric_0_f_1_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void confluent_hypergeometric_0_f_1_cpu_kernel(TensorIteratorBase &iterator)

void debye_d_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "debye_d_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void debye_d_cpu_kernel(TensorIteratorBase &iterator)

void exp_modified_bessel_i_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_i_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void exp_modified_bessel_i_cpu_kernel(TensorIteratorBase &iterator)

void exp_modified_bessel_k_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exp_modified_bessel_k_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void exp_modified_bessel_k_cpu_kernel(TensorIteratorBase &iterator)

void exponential_integral_e_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "exponential_integral_e_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void exponential_integral_e_cpu_kernel(TensorIteratorBase &iterator)

void falling_factorial_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "falling_factorial_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void falling_factorial_cpu_kernel(TensorIteratorBase &iterator)

void fermi_dirac_integral_f_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "fermi_dirac_integral_f_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void fermi_dirac_integral_f_cpu_kernel(TensorIteratorBase &iterator)

void hankel_h_1_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hankel_h_1_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void hankel_h_1_cpu_kernel(TensorIteratorBase &iterator)

void hankel_h_2_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hankel_h_2_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void hankel_h_2_cpu_kernel(TensorIteratorBase &iterator)

void hermite_polynomial_h_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hermite_polynomial_h_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void hermite_polynomial_h_cpu_kernel(TensorIteratorBase &iterator)

void hermite_polynomial_he_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hermite_polynomial_he_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void hermite_polynomial_he_cpu_kernel(TensorIteratorBase &iterator)

void heuman_lambda_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "heuman_lambda_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void heuman_lambda_cpu_kernel(TensorIteratorBase &iterator)

void hurwitz_zeta_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "hurwitz_zeta_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void hurwitz_zeta_cpu_kernel(TensorIteratorBase &iterator)

void incomplete_elliptic_integral_e_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_elliptic_integral_e_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void incomplete_elliptic_integral_e_cpu_kernel(TensorIteratorBase &iterator)

void incomplete_elliptic_integral_f_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_elliptic_integral_f_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void incomplete_elliptic_integral_f_cpu_kernel(TensorIteratorBase &iterator)

void incomplete_legendre_elliptic_integral_d_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_legendre_elliptic_integral_d_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void incomplete_legendre_elliptic_integral_d_cpu_kernel(TensorIteratorBase &iterator)

void jacobi_theta_1_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_1_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void jacobi_theta_1_cpu_kernel(TensorIteratorBase &iterator)

void jacobi_theta_2_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_2_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void jacobi_theta_2_cpu_kernel(TensorIteratorBase &iterator)

void jacobi_theta_3_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_3_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void jacobi_theta_3_cpu_kernel(TensorIteratorBase &iterator)

void jacobi_theta_4_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_theta_4_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void jacobi_theta_4_cpu_kernel(TensorIteratorBase &iterator)

void jacobi_zeta_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_zeta_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void jacobi_zeta_cpu_kernel(TensorIteratorBase &iterator)

void laguerre_polynomial_l_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "laguerre_polynomial_l_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void laguerre_polynomial_l_cpu_kernel(TensorIteratorBase &iterator)

void lah_number_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "lah_number_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void lah_number_cpu_kernel(TensorIteratorBase &iterator)

void legendre_polynomial_p_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "legendre_polynomial_p_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void legendre_polynomial_p_cpu_kernel(TensorIteratorBase &iterator)

void legendre_q_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "legendre_q_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void legendre_q_cpu_kernel(TensorIteratorBase &iterator)

void ln_binomial_coefficient_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_binomial_coefficient_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void ln_binomial_coefficient_cpu_kernel(TensorIteratorBase &iterator)

void ln_falling_factorial_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_falling_factorial_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void ln_falling_factorial_cpu_kernel(TensorIteratorBase &iterator)

void ln_rising_factorial_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "ln_rising_factorial_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void ln_rising_factorial_cpu_kernel(TensorIteratorBase &iterator)

void lower_incomplete_gamma_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "lower_incomplete_gamma_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void lower_incomplete_gamma_cpu_kernel(TensorIteratorBase &iterator)

void modified_bessel_i_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_i_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void modified_bessel_i_cpu_kernel(TensorIteratorBase &iterator)

void modified_bessel_k_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "modified_bessel_k_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void modified_bessel_k_cpu_kernel(TensorIteratorBase &iterator)

void neville_theta_c_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_c_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void neville_theta_c_cpu_kernel(TensorIteratorBase &iterator)

void neville_theta_d_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_d_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void neville_theta_d_cpu_kernel(TensorIteratorBase &iterator)

void neville_theta_n_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_n_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void neville_theta_n_cpu_kernel(TensorIteratorBase &iterator)

void neville_theta_s_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "neville_theta_s_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void neville_theta_s_cpu_kernel(TensorIteratorBase &iterator)

void owens_t_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "owens_t_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void owens_t_cpu_kernel(TensorIteratorBase &iterator)

void polar_pi_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "polar_pi_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void polar_pi_cpu_kernel(TensorIteratorBase &iterator)

void polylogarithm_li_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "polylogarithm_li_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void polylogarithm_li_cpu_kernel(TensorIteratorBase &iterator)

void rising_factorial_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "rising_factorial_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void rising_factorial_cpu_kernel(TensorIteratorBase &iterator)

void shifted_chebyshev_polynomial_t_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_t_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void shifted_chebyshev_polynomial_t_cpu_kernel(TensorIteratorBase &iterator)

void shifted_chebyshev_polynomial_u_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_u_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void shifted_chebyshev_polynomial_u_cpu_kernel(TensorIteratorBase &iterator)

void shifted_chebyshev_polynomial_v_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_v_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void shifted_chebyshev_polynomial_v_cpu_kernel(TensorIteratorBase &iterator)

void shifted_chebyshev_polynomial_w_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_w_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void shifted_chebyshev_polynomial_w_cpu_kernel(TensorIteratorBase &iterator)

void spherical_bessel_j_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_j_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void spherical_bessel_j_cpu_kernel(TensorIteratorBase &iterator)

void spherical_bessel_y_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_bessel_y_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void spherical_bessel_y_cpu_kernel(TensorIteratorBase &iterator)

void spherical_hankel_h_1_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_hankel_h_1_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void spherical_hankel_h_1_cpu_kernel(TensorIteratorBase &iterator)

void spherical_hankel_h_2_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_hankel_h_2_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void spherical_hankel_h_2_cpu_kernel(TensorIteratorBase &iterator)

void spherical_modified_bessel_i_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_modified_bessel_i_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void spherical_modified_bessel_i_cpu_kernel(TensorIteratorBase &iterator)

void spherical_modified_bessel_k_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_modified_bessel_k_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void spherical_modified_bessel_k_cpu_kernel(TensorIteratorBase &iterator)

void stirling_number_1_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "stirling_number_1_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void stirling_number_1_cpu_kernel(TensorIteratorBase &iterator)

void stirling_number_2_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "stirling_number_2_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void stirling_number_2_cpu_kernel(TensorIteratorBase &iterator)

void theta_1_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_1_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void theta_1_cpu_kernel(TensorIteratorBase &iterator)

void theta_2_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_2_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void theta_2_cpu_kernel(TensorIteratorBase &iterator)

void theta_3_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_3_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void theta_3_cpu_kernel(TensorIteratorBase &iterator)

void theta_4_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "theta_4_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void theta_4_cpu_kernel(TensorIteratorBase &iterator)

void upper_incomplete_gamma_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "upper_incomplete_gamma_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y) -> scalar_t {
      return x;
    });
  });
} // void upper_incomplete_gamma_cpu_kernel(TensorIteratorBase &iterator)

/*
 *  TERNARY OPERATORS
 */

void associated_laguerre_polynomial_l_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "associated_laguerre_polynomial_l_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y, scalar_t z) -> scalar_t {
      return x;
    });
  });
} // void associated_laguerre_polynomial_l_cpu_kernel(TensorIteratorBase &iterator)

void associated_legendre_p_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "associated_legendre_p_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y, scalar_t z) -> scalar_t {
      return x;
    });
  });
} // void associated_legendre_p_cpu_kernel(TensorIteratorBase &iterator)

void associated_legendre_q_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "associated_legendre_q_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y, scalar_t z) -> scalar_t {
      return x;
    });
  });
} // void associated_legendre_q_cpu_kernel(TensorIteratorBase &iterator)

void bulirsch_elliptic_integral_el3_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_el3_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y, scalar_t z) -> scalar_t {
      return x;
    });
  });
} // void bulirsch_elliptic_integral_el3_cpu_kernel(TensorIteratorBase &iterator)

void carlson_elliptic_r_d_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_d_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y, scalar_t z) -> scalar_t {
      return x;
    });
  });
} // void carlson_elliptic_r_d_cpu_kernel(TensorIteratorBase &iterator)

void carlson_elliptic_r_f_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_f_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y, scalar_t z) -> scalar_t {
      return x;
    });
  });
} // void carlson_elliptic_r_f_cpu_kernel(TensorIteratorBase &iterator)

void carlson_elliptic_r_g_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_g_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y, scalar_t z) -> scalar_t {
      return x;
    });
  });
} // void carlson_elliptic_r_g_cpu_kernel(TensorIteratorBase &iterator)

void gegenbauer_polynomial_c_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "gegenbauer_polynomial_c_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y, scalar_t z) -> scalar_t {
      return x;
    });
  });
} // void gegenbauer_polynomial_c_cpu_kernel(TensorIteratorBase &iterator)

void incomplete_beta_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_beta_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y, scalar_t z) -> scalar_t {
      return x;
    });
  });
} // void incomplete_beta_cpu_kernel(TensorIteratorBase &iterator)

void incomplete_elliptic_integral_pi_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "incomplete_elliptic_integral_pi_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y, scalar_t z) -> scalar_t {
      return x;
    });
  });
} // void incomplete_elliptic_integral_pi_cpu_kernel(TensorIteratorBase &iterator)

void kummer_confluent_hypergeometric_1_f_1_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "kummer_confluent_hypergeometric_1_f_1_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y, scalar_t z) -> scalar_t {
      return x;
    });
  });
} // void kummer_confluent_hypergeometric_1_f_1_cpu_kernel(TensorIteratorBase &iterator)

void radial_polynomial_r_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "radial_polynomial_r_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y, scalar_t z) -> scalar_t {
      return x;
    });
  });
} // void radial_polynomial_r_cpu_kernel(TensorIteratorBase &iterator)

void spherical_legendre_y_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_legendre_y_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y, scalar_t z) -> scalar_t {
      return x;
    });
  });
} // void spherical_legendre_y_cpu_kernel(TensorIteratorBase &iterator)

void tricomi_confluent_hypergeometric_u_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "tricomi_confluent_hypergeometric_u_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t x, scalar_t y, scalar_t z) -> scalar_t {
      return x;
    });
  });
} // void tricomi_confluent_hypergeometric_u_cpu_kernel(TensorIteratorBase &iterator)

void bulirsch_elliptic_integral_cel_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_cel_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
} // void bulirsch_elliptic_integral_cel_cpu_kernel(TensorIteratorBase &iterator)

void bulirsch_elliptic_integral_el2_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bulirsch_elliptic_integral_el2_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
} // void bulirsch_elliptic_integral_el2_cpu_kernel(TensorIteratorBase &iterator)

void carlson_elliptic_r_j_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "carlson_elliptic_r_j_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
} // void carlson_elliptic_r_j_cpu_kernel(TensorIteratorBase &iterator)

void gauss_hypergeometric_2_f_1_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "gauss_hypergeometric_2_f_1_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
} // void gauss_hypergeometric_2_f_1_cpu_kernel(TensorIteratorBase &iterator)

void jacobi_polynomial_p_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "jacobi_polynomial_p_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
} // void jacobi_polynomial_p_cpu_kernel(TensorIteratorBase &iterator)

void spherical_harmonic_y_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "spherical_harmonic_y_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
} // void spherical_harmonic_y_cpu_kernel(TensorIteratorBase &iterator)

void zernike_polynomial_z_cpu_kernel(TensorIteratorBase &iterator) {
  AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "zernike_polynomial_z_cpu_kernel", [&]() {
    cpu_kernel(iterator, [](scalar_t a, scalar_t b, scalar_t c, scalar_t d) -> scalar_t {
      return a;
    });
  });
} // void zernike_polynomial_z_cpu_kernel(TensorIteratorBase &iterator)
} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(special_airy_ai_stub, &CPU_CAPABILITY::airy_ai_cpu_kernel);
REGISTER_DISPATCH(special_airy_bi_stub, &CPU_CAPABILITY::airy_bi_cpu_kernel);
REGISTER_DISPATCH(special_associated_laguerre_polynomial_l_stub, &CPU_CAPABILITY::associated_laguerre_polynomial_l_cpu_kernel);
REGISTER_DISPATCH(special_associated_legendre_p_stub, &CPU_CAPABILITY::associated_legendre_p_cpu_kernel);
REGISTER_DISPATCH(special_associated_legendre_q_stub, &CPU_CAPABILITY::associated_legendre_q_cpu_kernel);
REGISTER_DISPATCH(special_bell_polynomial_b_stub, &CPU_CAPABILITY::bell_polynomial_b_cpu_kernel);
REGISTER_DISPATCH(special_bernoulli_number_stub, &CPU_CAPABILITY::bernoulli_number_cpu_kernel);
REGISTER_DISPATCH(special_bernoulli_polynomial_b_stub, &CPU_CAPABILITY::bernoulli_polynomial_b_cpu_kernel);
REGISTER_DISPATCH(special_bessel_j_0_stub, &CPU_CAPABILITY::bessel_j_0_cpu_kernel);
REGISTER_DISPATCH(special_bessel_j_1_stub, &CPU_CAPABILITY::bessel_j_1_cpu_kernel);
REGISTER_DISPATCH(special_bessel_j_stub, &CPU_CAPABILITY::bessel_j_cpu_kernel);
REGISTER_DISPATCH(special_bessel_y_0_stub, &CPU_CAPABILITY::bessel_y_0_cpu_kernel);
REGISTER_DISPATCH(special_bessel_y_1_stub, &CPU_CAPABILITY::bessel_y_1_cpu_kernel);
REGISTER_DISPATCH(special_bessel_y_stub, &CPU_CAPABILITY::bessel_y_cpu_kernel);
REGISTER_DISPATCH(special_beta_stub, &CPU_CAPABILITY::beta_cpu_kernel);
REGISTER_DISPATCH(special_binomial_coefficient_stub, &CPU_CAPABILITY::binomial_coefficient_cpu_kernel);
REGISTER_DISPATCH(special_bose_einstein_integral_g_stub, &CPU_CAPABILITY::bose_einstein_integral_g_cpu_kernel);
REGISTER_DISPATCH(special_bulirsch_elliptic_integral_cel_stub, &CPU_CAPABILITY::bulirsch_elliptic_integral_cel_cpu_kernel);
REGISTER_DISPATCH(special_bulirsch_elliptic_integral_el1_stub, &CPU_CAPABILITY::bulirsch_elliptic_integral_el1_cpu_kernel);
REGISTER_DISPATCH(special_bulirsch_elliptic_integral_el2_stub, &CPU_CAPABILITY::bulirsch_elliptic_integral_el2_cpu_kernel);
REGISTER_DISPATCH(special_bulirsch_elliptic_integral_el3_stub, &CPU_CAPABILITY::bulirsch_elliptic_integral_el3_cpu_kernel);
REGISTER_DISPATCH(special_carlson_elliptic_r_c_stub, &CPU_CAPABILITY::carlson_elliptic_r_c_cpu_kernel);
REGISTER_DISPATCH(special_carlson_elliptic_r_d_stub, &CPU_CAPABILITY::carlson_elliptic_r_d_cpu_kernel);
REGISTER_DISPATCH(special_carlson_elliptic_r_f_stub, &CPU_CAPABILITY::carlson_elliptic_r_f_cpu_kernel);
REGISTER_DISPATCH(special_carlson_elliptic_r_g_stub, &CPU_CAPABILITY::carlson_elliptic_r_g_cpu_kernel);
REGISTER_DISPATCH(special_carlson_elliptic_r_j_stub, &CPU_CAPABILITY::carlson_elliptic_r_j_cpu_kernel);
REGISTER_DISPATCH(special_chebyshev_polynomial_t_stub, &CPU_CAPABILITY::chebyshev_polynomial_t_cpu_kernel);
REGISTER_DISPATCH(special_chebyshev_polynomial_u_stub, &CPU_CAPABILITY::chebyshev_polynomial_u_cpu_kernel);
REGISTER_DISPATCH(special_chebyshev_polynomial_v_stub, &CPU_CAPABILITY::chebyshev_polynomial_v_cpu_kernel);
REGISTER_DISPATCH(special_chebyshev_polynomial_w_stub, &CPU_CAPABILITY::chebyshev_polynomial_w_cpu_kernel);
REGISTER_DISPATCH(special_clausen_cl_stub, &CPU_CAPABILITY::clausen_cl_cpu_kernel);
REGISTER_DISPATCH(special_clausen_sl_stub, &CPU_CAPABILITY::clausen_sl_cpu_kernel);
REGISTER_DISPATCH(special_complete_carlson_elliptic_r_f_stub, &CPU_CAPABILITY::complete_carlson_elliptic_r_f_cpu_kernel);
REGISTER_DISPATCH(special_complete_carlson_elliptic_r_g_stub, &CPU_CAPABILITY::complete_carlson_elliptic_r_g_cpu_kernel);
REGISTER_DISPATCH(special_complete_elliptic_integral_e_stub, &CPU_CAPABILITY::complete_elliptic_integral_e_cpu_kernel);
REGISTER_DISPATCH(special_complete_elliptic_integral_k_stub, &CPU_CAPABILITY::complete_elliptic_integral_k_cpu_kernel);
REGISTER_DISPATCH(special_complete_elliptic_integral_pi_stub, &CPU_CAPABILITY::complete_elliptic_integral_pi_cpu_kernel);
REGISTER_DISPATCH(special_complete_legendre_elliptic_integral_d_stub, &CPU_CAPABILITY::complete_legendre_elliptic_integral_d_cpu_kernel);
REGISTER_DISPATCH(special_confluent_hypergeometric_0_f_1_stub, &CPU_CAPABILITY::confluent_hypergeometric_0_f_1_cpu_kernel);
REGISTER_DISPATCH(special_cos_pi_stub, &CPU_CAPABILITY::cos_pi_cpu_kernel);
REGISTER_DISPATCH(special_cosh_pi_stub, &CPU_CAPABILITY::cosh_pi_cpu_kernel);
REGISTER_DISPATCH(special_cosine_integral_ci_stub, &CPU_CAPABILITY::cosine_integral_ci_cpu_kernel);
REGISTER_DISPATCH(special_debye_d_stub, &CPU_CAPABILITY::debye_d_cpu_kernel);
REGISTER_DISPATCH(special_dilogarithm_li_2_stub, &CPU_CAPABILITY::dilogarithm_li_2_cpu_kernel);
REGISTER_DISPATCH(special_dirichlet_beta_stub, &CPU_CAPABILITY::dirichlet_beta_cpu_kernel);
REGISTER_DISPATCH(special_dirichlet_eta_stub, &CPU_CAPABILITY::dirichlet_eta_cpu_kernel);
REGISTER_DISPATCH(special_dirichlet_lambda_stub, &CPU_CAPABILITY::dirichlet_lambda_cpu_kernel);
REGISTER_DISPATCH(special_double_factorial_stub, &CPU_CAPABILITY::double_factorial_cpu_kernel);
REGISTER_DISPATCH(special_exp_airy_ai_stub, &CPU_CAPABILITY::exp_airy_ai_cpu_kernel);
REGISTER_DISPATCH(special_exp_airy_bi_stub, &CPU_CAPABILITY::exp_airy_bi_cpu_kernel);
REGISTER_DISPATCH(special_exp_modified_bessel_i_stub, &CPU_CAPABILITY::exp_modified_bessel_i_cpu_kernel);
REGISTER_DISPATCH(special_exp_modified_bessel_k_0_stub, &CPU_CAPABILITY::exp_modified_bessel_k_0_cpu_kernel);
REGISTER_DISPATCH(special_exp_modified_bessel_k_1_stub, &CPU_CAPABILITY::exp_modified_bessel_k_1_cpu_kernel);
REGISTER_DISPATCH(special_exp_modified_bessel_k_stub, &CPU_CAPABILITY::exp_modified_bessel_k_cpu_kernel);
REGISTER_DISPATCH(special_exponential_integral_e_stub, &CPU_CAPABILITY::exponential_integral_e_cpu_kernel);
REGISTER_DISPATCH(special_exponential_integral_ei_stub, &CPU_CAPABILITY::exponential_integral_ei_cpu_kernel);
REGISTER_DISPATCH(special_factorial_stub, &CPU_CAPABILITY::factorial_cpu_kernel);
REGISTER_DISPATCH(special_falling_factorial_stub, &CPU_CAPABILITY::falling_factorial_cpu_kernel);
REGISTER_DISPATCH(special_fermi_dirac_integral_f_stub, &CPU_CAPABILITY::fermi_dirac_integral_f_cpu_kernel);
REGISTER_DISPATCH(special_fresnel_integral_c_stub, &CPU_CAPABILITY::fresnel_integral_c_cpu_kernel);
REGISTER_DISPATCH(special_fresnel_integral_s_stub, &CPU_CAPABILITY::fresnel_integral_s_cpu_kernel);
REGISTER_DISPATCH(special_gauss_hypergeometric_2_f_1_stub, &CPU_CAPABILITY::gauss_hypergeometric_2_f_1_cpu_kernel);
REGISTER_DISPATCH(special_gegenbauer_polynomial_c_stub, &CPU_CAPABILITY::gegenbauer_polynomial_c_cpu_kernel);
REGISTER_DISPATCH(special_hankel_h_1_stub, &CPU_CAPABILITY::hankel_h_1_cpu_kernel);
REGISTER_DISPATCH(special_hankel_h_2_stub, &CPU_CAPABILITY::hankel_h_2_cpu_kernel);
REGISTER_DISPATCH(special_harmonic_number_stub, &CPU_CAPABILITY::harmonic_number_cpu_kernel);
REGISTER_DISPATCH(special_hermite_polynomial_h_stub, &CPU_CAPABILITY::hermite_polynomial_h_cpu_kernel);
REGISTER_DISPATCH(special_hermite_polynomial_he_stub, &CPU_CAPABILITY::hermite_polynomial_he_cpu_kernel);
REGISTER_DISPATCH(special_heuman_lambda_stub, &CPU_CAPABILITY::heuman_lambda_cpu_kernel);
REGISTER_DISPATCH(special_hurwitz_zeta_stub, &CPU_CAPABILITY::hurwitz_zeta_cpu_kernel);
REGISTER_DISPATCH(special_hyperbolic_cosine_integral_chi_stub, &CPU_CAPABILITY::hyperbolic_cosine_integral_chi_cpu_kernel);
REGISTER_DISPATCH(special_hyperbolic_sine_integral_shi_stub, &CPU_CAPABILITY::hyperbolic_sine_integral_shi_cpu_kernel);
REGISTER_DISPATCH(special_incomplete_beta_stub, &CPU_CAPABILITY::incomplete_beta_cpu_kernel);
REGISTER_DISPATCH(special_incomplete_elliptic_integral_e_stub, &CPU_CAPABILITY::incomplete_elliptic_integral_e_cpu_kernel);
REGISTER_DISPATCH(special_incomplete_elliptic_integral_f_stub, &CPU_CAPABILITY::incomplete_elliptic_integral_f_cpu_kernel);
REGISTER_DISPATCH(special_incomplete_elliptic_integral_pi_stub, &CPU_CAPABILITY::incomplete_elliptic_integral_pi_cpu_kernel);
REGISTER_DISPATCH(special_incomplete_legendre_elliptic_integral_d_stub, &CPU_CAPABILITY::incomplete_legendre_elliptic_integral_d_cpu_kernel);
REGISTER_DISPATCH(special_jacobi_polynomial_p_stub, &CPU_CAPABILITY::jacobi_polynomial_p_cpu_kernel);
REGISTER_DISPATCH(special_jacobi_theta_1_stub, &CPU_CAPABILITY::jacobi_theta_1_cpu_kernel);
REGISTER_DISPATCH(special_jacobi_theta_2_stub, &CPU_CAPABILITY::jacobi_theta_2_cpu_kernel);
REGISTER_DISPATCH(special_jacobi_theta_3_stub, &CPU_CAPABILITY::jacobi_theta_3_cpu_kernel);
REGISTER_DISPATCH(special_jacobi_theta_4_stub, &CPU_CAPABILITY::jacobi_theta_4_cpu_kernel);
REGISTER_DISPATCH(special_jacobi_zeta_stub, &CPU_CAPABILITY::jacobi_zeta_cpu_kernel);
REGISTER_DISPATCH(special_kummer_confluent_hypergeometric_1_f_1_stub, &CPU_CAPABILITY::kummer_confluent_hypergeometric_1_f_1_cpu_kernel);
REGISTER_DISPATCH(special_laguerre_polynomial_l_stub, &CPU_CAPABILITY::laguerre_polynomial_l_cpu_kernel);
REGISTER_DISPATCH(special_lah_number_stub, &CPU_CAPABILITY::lah_number_cpu_kernel);
REGISTER_DISPATCH(special_legendre_polynomial_p_stub, &CPU_CAPABILITY::legendre_polynomial_p_cpu_kernel);
REGISTER_DISPATCH(special_legendre_q_stub, &CPU_CAPABILITY::legendre_q_cpu_kernel);
REGISTER_DISPATCH(special_ln_binomial_coefficient_stub, &CPU_CAPABILITY::ln_binomial_coefficient_cpu_kernel);
REGISTER_DISPATCH(special_ln_double_factorial_stub, &CPU_CAPABILITY::ln_double_factorial_cpu_kernel);
REGISTER_DISPATCH(special_ln_factorial_stub, &CPU_CAPABILITY::ln_factorial_cpu_kernel);
REGISTER_DISPATCH(special_ln_falling_factorial_stub, &CPU_CAPABILITY::ln_falling_factorial_cpu_kernel);
REGISTER_DISPATCH(special_ln_gamma_sign_stub, &CPU_CAPABILITY::ln_gamma_sign_cpu_kernel);
REGISTER_DISPATCH(special_ln_gamma_stub, &CPU_CAPABILITY::ln_gamma_cpu_kernel);
REGISTER_DISPATCH(special_ln_rising_factorial_stub, &CPU_CAPABILITY::ln_rising_factorial_cpu_kernel);
REGISTER_DISPATCH(special_logarithmic_integral_li_stub, &CPU_CAPABILITY::logarithmic_integral_li_cpu_kernel);
REGISTER_DISPATCH(special_lower_incomplete_gamma_stub, &CPU_CAPABILITY::lower_incomplete_gamma_cpu_kernel);
REGISTER_DISPATCH(special_modified_bessel_i_0_stub, &CPU_CAPABILITY::modified_bessel_i_0_cpu_kernel);
REGISTER_DISPATCH(special_modified_bessel_i_1_stub, &CPU_CAPABILITY::modified_bessel_i_1_cpu_kernel);
REGISTER_DISPATCH(special_modified_bessel_i_stub, &CPU_CAPABILITY::modified_bessel_i_cpu_kernel);
REGISTER_DISPATCH(special_modified_bessel_k_0_stub, &CPU_CAPABILITY::modified_bessel_k_0_cpu_kernel);
REGISTER_DISPATCH(special_modified_bessel_k_1_stub, &CPU_CAPABILITY::modified_bessel_k_1_cpu_kernel);
REGISTER_DISPATCH(special_modified_bessel_k_stub, &CPU_CAPABILITY::modified_bessel_k_cpu_kernel);
REGISTER_DISPATCH(special_neville_theta_c_stub, &CPU_CAPABILITY::neville_theta_c_cpu_kernel);
REGISTER_DISPATCH(special_neville_theta_d_stub, &CPU_CAPABILITY::neville_theta_d_cpu_kernel);
REGISTER_DISPATCH(special_neville_theta_n_stub, &CPU_CAPABILITY::neville_theta_n_cpu_kernel);
REGISTER_DISPATCH(special_neville_theta_s_stub, &CPU_CAPABILITY::neville_theta_s_cpu_kernel);
REGISTER_DISPATCH(special_nome_q_stub, &CPU_CAPABILITY::nome_q_cpu_kernel);
REGISTER_DISPATCH(special_owens_t_stub, &CPU_CAPABILITY::owens_t_cpu_kernel);
REGISTER_DISPATCH(special_polar_pi_stub, &CPU_CAPABILITY::polar_pi_cpu_kernel);
REGISTER_DISPATCH(special_polylogarithm_li_stub, &CPU_CAPABILITY::polylogarithm_li_cpu_kernel);
REGISTER_DISPATCH(special_prime_number_stub, &CPU_CAPABILITY::prime_number_cpu_kernel);
REGISTER_DISPATCH(special_radial_polynomial_r_stub, &CPU_CAPABILITY::radial_polynomial_r_cpu_kernel);
REGISTER_DISPATCH(special_reciprocal_gamma_stub, &CPU_CAPABILITY::reciprocal_gamma_cpu_kernel);
REGISTER_DISPATCH(special_riemann_zeta_stub, &CPU_CAPABILITY::riemann_zeta_cpu_kernel);
REGISTER_DISPATCH(special_rising_factorial_stub, &CPU_CAPABILITY::rising_factorial_cpu_kernel);
REGISTER_DISPATCH(special_shifted_chebyshev_polynomial_t_stub, &CPU_CAPABILITY::shifted_chebyshev_polynomial_t_cpu_kernel);
REGISTER_DISPATCH(special_shifted_chebyshev_polynomial_u_stub, &CPU_CAPABILITY::shifted_chebyshev_polynomial_u_cpu_kernel);
REGISTER_DISPATCH(special_shifted_chebyshev_polynomial_v_stub, &CPU_CAPABILITY::shifted_chebyshev_polynomial_v_cpu_kernel);
REGISTER_DISPATCH(special_shifted_chebyshev_polynomial_w_stub, &CPU_CAPABILITY::shifted_chebyshev_polynomial_w_cpu_kernel);
REGISTER_DISPATCH(special_sin_pi_stub, &CPU_CAPABILITY::sin_pi_cpu_kernel);
REGISTER_DISPATCH(special_sinc_pi_stub, &CPU_CAPABILITY::sinc_pi_cpu_kernel);
REGISTER_DISPATCH(special_sinh_pi_stub, &CPU_CAPABILITY::sinh_pi_cpu_kernel);
REGISTER_DISPATCH(special_sinhc_pi_stub, &CPU_CAPABILITY::sinhc_pi_cpu_kernel);
REGISTER_DISPATCH(special_sinhc_stub, &CPU_CAPABILITY::sinhc_cpu_kernel);
REGISTER_DISPATCH(special_spherical_bessel_j_0_stub, &CPU_CAPABILITY::spherical_bessel_j_0_cpu_kernel);
REGISTER_DISPATCH(special_spherical_bessel_j_stub, &CPU_CAPABILITY::spherical_bessel_j_cpu_kernel);
REGISTER_DISPATCH(special_spherical_bessel_y_stub, &CPU_CAPABILITY::spherical_bessel_y_cpu_kernel);
REGISTER_DISPATCH(special_spherical_hankel_h_1_stub, &CPU_CAPABILITY::spherical_hankel_h_1_cpu_kernel);
REGISTER_DISPATCH(special_spherical_hankel_h_2_stub, &CPU_CAPABILITY::spherical_hankel_h_2_cpu_kernel);
REGISTER_DISPATCH(special_spherical_harmonic_y_stub, &CPU_CAPABILITY::spherical_harmonic_y_cpu_kernel);
REGISTER_DISPATCH(special_spherical_legendre_y_stub, &CPU_CAPABILITY::spherical_legendre_y_cpu_kernel);
REGISTER_DISPATCH(special_spherical_modified_bessel_i_stub, &CPU_CAPABILITY::spherical_modified_bessel_i_cpu_kernel);
REGISTER_DISPATCH(special_spherical_modified_bessel_k_stub, &CPU_CAPABILITY::spherical_modified_bessel_k_cpu_kernel);
REGISTER_DISPATCH(special_stirling_number_1_stub, &CPU_CAPABILITY::stirling_number_1_cpu_kernel);
REGISTER_DISPATCH(special_stirling_number_2_stub, &CPU_CAPABILITY::stirling_number_2_cpu_kernel);
REGISTER_DISPATCH(special_tan_pi_stub, &CPU_CAPABILITY::tan_pi_cpu_kernel);
REGISTER_DISPATCH(special_tanh_pi_stub, &CPU_CAPABILITY::tanh_pi_cpu_kernel);
REGISTER_DISPATCH(special_theta_1_stub, &CPU_CAPABILITY::theta_1_cpu_kernel);
REGISTER_DISPATCH(special_theta_2_stub, &CPU_CAPABILITY::theta_2_cpu_kernel);
REGISTER_DISPATCH(special_theta_3_stub, &CPU_CAPABILITY::theta_3_cpu_kernel);
REGISTER_DISPATCH(special_theta_4_stub, &CPU_CAPABILITY::theta_4_cpu_kernel);
REGISTER_DISPATCH(special_tricomi_confluent_hypergeometric_u_stub, &CPU_CAPABILITY::tricomi_confluent_hypergeometric_u_cpu_kernel);
REGISTER_DISPATCH(special_upper_incomplete_gamma_stub, &CPU_CAPABILITY::upper_incomplete_gamma_cpu_kernel);
REGISTER_DISPATCH(special_zernike_polynomial_z_stub, &CPU_CAPABILITY::zernike_polynomial_z_cpu_kernel);
} // namespace native
} // namespace at
