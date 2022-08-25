#include <ATen/native/special.h>

#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/ExpandUtils.h>
#include <ATen/RedispatchFunctions.h>
#include <torch/library.h>

namespace at {
namespace meta {
TORCH_META_FUNC (special_airy_bi)                                 (const Tensor &z)                    { build_borrowing_unary_float_op  (maybe_get_output(), z);      }
TORCH_META_FUNC (special_bernoulli_number)                        (const Tensor &n)                    { build_borrowing_unary_float_op  (maybe_get_output(), n);      }
TORCH_META_FUNC (special_bessel_j)                                (const Tensor &v, const Tensor &z)   { build_borrowing_binary_float_op (maybe_get_output(), v, z);   }
TORCH_META_FUNC (special_bessel_y)                                (const Tensor &v, const Tensor &z)   { build_borrowing_binary_float_op (maybe_get_output(), v, z);   }
TORCH_META_FUNC (special_beta)                                    (const Tensor &a, const Tensor &b)   { build_borrowing_binary_float_op (maybe_get_output(), a, b);   }
TORCH_META_FUNC (special_binomial_coefficient)                    (const Tensor &m, const Tensor &n)   { build_borrowing_binary_float_op (maybe_get_output(), m, n);   }
TORCH_META_FUNC (special_bulirsch_elliptic_integral_el1)          (const Tensor &x, const Tensor &k_c) { build_borrowing_binary_float_op (maybe_get_output(), x, k_c); }
TORCH_META_FUNC (special_carlson_elliptic_r_c)                    (const Tensor &x, const Tensor &y)   { build_borrowing_binary_float_op (maybe_get_output(), x, y);   }
TORCH_META_FUNC (special_complete_carlson_elliptic_r_f)           (const Tensor &x, const Tensor &y)   { build_borrowing_binary_float_op (maybe_get_output(), x, y);   }
TORCH_META_FUNC (special_complete_carlson_elliptic_r_g)           (const Tensor &x, const Tensor &y)   { build_borrowing_binary_float_op (maybe_get_output(), x, y);   }
TORCH_META_FUNC (special_complete_legendre_elliptic_integral_d)   (const Tensor &k)                    { build_borrowing_unary_float_op  (maybe_get_output(), k);      }
TORCH_META_FUNC (special_complete_legendre_elliptic_integral_e)   (const Tensor &k)                    { build_borrowing_unary_float_op  (maybe_get_output(), k);      }
TORCH_META_FUNC (special_complete_legendre_elliptic_integral_k)   (const Tensor &k)                    { build_borrowing_unary_float_op  (maybe_get_output(), k);      }
TORCH_META_FUNC (special_complete_legendre_elliptic_integral_pi)  (const Tensor &n, const Tensor &k)   { build_borrowing_binary_float_op (maybe_get_output(), n, k);   }
TORCH_META_FUNC (special_cos_pi)                                  (const Tensor &z)                    { build_borrowing_unary_float_op  (maybe_get_output(), z);      }
TORCH_META_FUNC (special_cosh_pi)                                 (const Tensor &z)                    { build_borrowing_unary_float_op  (maybe_get_output(), z);      }
TORCH_META_FUNC (special_double_factorial)                        (const Tensor &n)                    { build_borrowing_unary_float_op  (maybe_get_output(), n);      }
TORCH_META_FUNC (special_exp_airy_ai)                             (const Tensor &z)                    { build_borrowing_unary_float_op  (maybe_get_output(), z);      }
TORCH_META_FUNC (special_exp_airy_bi)                             (const Tensor &z)                    { build_borrowing_unary_float_op  (maybe_get_output(), z);      }
TORCH_META_FUNC (special_exp_modified_bessel_i)                   (const Tensor &v, const Tensor &z)   { build_borrowing_binary_float_op (maybe_get_output(), v, z);   }
TORCH_META_FUNC (special_exp_modified_bessel_k)                   (const Tensor &v, const Tensor &z)   { build_borrowing_binary_float_op (maybe_get_output(), v, z);   }
TORCH_META_FUNC (special_factorial)                               (const Tensor &n)                    { build_borrowing_unary_float_op  (maybe_get_output(), n);      }
TORCH_META_FUNC (special_falling_factorial)                       (const Tensor &x, const Tensor &n)   { build_borrowing_binary_float_op (maybe_get_output(), x, n);   }
TORCH_META_FUNC (special_gamma)                                   (const Tensor &z)                    { build_borrowing_unary_float_op  (maybe_get_output(), z);      }
TORCH_META_FUNC (special_hankel_h_1)                              (const Tensor &v, const Tensor &z)   { build_borrowing_binary_float_op (maybe_get_output(), v, z);   }
TORCH_META_FUNC (special_hankel_h_2)                              (const Tensor &v, const Tensor &z)   { build_borrowing_binary_float_op (maybe_get_output(), v, z);   }
TORCH_META_FUNC (special_harmonic_number)                         (const Tensor &n)                    { build_borrowing_unary_float_op  (maybe_get_output(), n);      }
TORCH_META_FUNC (special_incomplete_legendre_elliptic_integral_d) (const Tensor &k, const Tensor &phi) { build_borrowing_binary_float_op (maybe_get_output(), k, phi); }
TORCH_META_FUNC (special_incomplete_legendre_elliptic_integral_e) (const Tensor &k, const Tensor &phi) { build_borrowing_binary_float_op (maybe_get_output(), k, phi); }
TORCH_META_FUNC (special_incomplete_legendre_elliptic_integral_f) (const Tensor &k, const Tensor &phi) { build_borrowing_binary_float_op (maybe_get_output(), k, phi); }
TORCH_META_FUNC (special_ln_binomial_coefficient)                 (const Tensor &m, const Tensor &n)   { build_borrowing_binary_float_op (maybe_get_output(), m, n);   }
TORCH_META_FUNC (special_ln_double_factorial)                     (const Tensor &n)                    { build_borrowing_unary_float_op  (maybe_get_output(), n);      }
TORCH_META_FUNC (special_ln_factorial)                            (const Tensor &n)                    { build_borrowing_unary_float_op  (maybe_get_output(), n);      }
TORCH_META_FUNC (special_ln_falling_factorial)                    (const Tensor &x, const Tensor &n)   { build_borrowing_binary_float_op (maybe_get_output(), x, n);   }
TORCH_META_FUNC (special_ln_gamma)                                (const Tensor &z)                    { build_borrowing_unary_float_op  (maybe_get_output(), z);      }
TORCH_META_FUNC (special_ln_gamma_sign)                           (const Tensor &z)                    { build_borrowing_unary_float_op  (maybe_get_output(), z);      }
TORCH_META_FUNC (special_ln_rising_factorial)                     (const Tensor &x, const Tensor &n)   { build_borrowing_binary_float_op (maybe_get_output(), x, n);   }
TORCH_META_FUNC (special_lower_incomplete_gamma)                  (const Tensor &a, const Tensor &z)   { build_borrowing_binary_float_op (maybe_get_output(), a, z);   }
TORCH_META_FUNC (special_modified_bessel_i)                       (const Tensor &v, const Tensor &z)   { build_borrowing_binary_float_op (maybe_get_output(), v, z);   }
TORCH_META_FUNC (special_modified_bessel_k)                       (const Tensor &v, const Tensor &z)   { build_borrowing_binary_float_op (maybe_get_output(), v, z);   }
TORCH_META_FUNC (special_prime_number)                            (const Tensor &n)                    { build_borrowing_unary_float_op  (maybe_get_output(), n);      }
TORCH_META_FUNC (special_reciprocal_gamma)                        (const Tensor &z)                    { build_borrowing_unary_float_op  (maybe_get_output(), z);      }
TORCH_META_FUNC (special_rising_factorial)                        (const Tensor &x, const Tensor &n)   { build_borrowing_binary_float_op (maybe_get_output(), x, n);   }
TORCH_META_FUNC (special_sin_pi)                                  (const Tensor &z)                    { build_borrowing_unary_float_op  (maybe_get_output(), z);      }
TORCH_META_FUNC (special_sinh_pi)                                 (const Tensor &z)                    { build_borrowing_unary_float_op  (maybe_get_output(), z);      }
TORCH_META_FUNC (special_spherical_bessel_j)                      (const Tensor &n, const Tensor &z)   { build_borrowing_binary_float_op (maybe_get_output(), n, z);   }
TORCH_META_FUNC (special_spherical_bessel_y)                      (const Tensor &n, const Tensor &z)   { build_borrowing_binary_float_op (maybe_get_output(), n, z);   }
TORCH_META_FUNC (special_spherical_hankel_h_1)                    (const Tensor &n, const Tensor &z)   { build_borrowing_binary_float_op (maybe_get_output(), n, z);   }
TORCH_META_FUNC (special_spherical_hankel_h_2)                    (const Tensor &n, const Tensor &z)   { build_borrowing_binary_float_op (maybe_get_output(), n, z);   }
TORCH_META_FUNC (special_spherical_modified_bessel_i)             (const Tensor &n, const Tensor &z)   { build_borrowing_binary_float_op (maybe_get_output(), n, z);   }
TORCH_META_FUNC (special_spherical_modified_bessel_k)             (const Tensor &n, const Tensor &z)   { build_borrowing_binary_float_op (maybe_get_output(), n, z);   }
TORCH_META_FUNC (special_tan_pi)                                  (const Tensor &z)                    { build_borrowing_unary_float_op  (maybe_get_output(), z);      }
TORCH_META_FUNC (special_tanh_pi)                                 (const Tensor &z)                    { build_borrowing_unary_float_op  (maybe_get_output(), z);      }
TORCH_META_FUNC (special_upper_incomplete_gamma)                  (const Tensor &a, const Tensor &z)   { build_borrowing_binary_float_op (maybe_get_output(), a, z);   }
} // namespace meta

namespace native {
DEFINE_DISPATCH(special_airy_bi_stub);
DEFINE_DISPATCH(special_bernoulli_number_stub);
DEFINE_DISPATCH(special_bessel_j_stub);
DEFINE_DISPATCH(special_bessel_y_stub);
DEFINE_DISPATCH(special_beta_stub);
DEFINE_DISPATCH(special_binomial_coefficient_stub);
DEFINE_DISPATCH(special_bulirsch_elliptic_integral_el1_stub);
DEFINE_DISPATCH(special_carlson_elliptic_r_c_stub);
DEFINE_DISPATCH(special_complete_carlson_elliptic_r_f_stub);
DEFINE_DISPATCH(special_complete_carlson_elliptic_r_g_stub);
DEFINE_DISPATCH(special_complete_legendre_elliptic_integral_d_stub);
DEFINE_DISPATCH(special_complete_legendre_elliptic_integral_e_stub);
DEFINE_DISPATCH(special_complete_legendre_elliptic_integral_k_stub);
DEFINE_DISPATCH(special_complete_legendre_elliptic_integral_pi_stub);
DEFINE_DISPATCH(special_cos_pi_stub);
DEFINE_DISPATCH(special_cosh_pi_stub);
DEFINE_DISPATCH(special_double_factorial_stub);
DEFINE_DISPATCH(special_exp_airy_ai_stub);
DEFINE_DISPATCH(special_exp_airy_bi_stub);
DEFINE_DISPATCH(special_exp_modified_bessel_i_stub);
DEFINE_DISPATCH(special_exp_modified_bessel_k_stub);
DEFINE_DISPATCH(special_factorial_stub);
DEFINE_DISPATCH(special_falling_factorial_stub);
DEFINE_DISPATCH(special_gamma_stub);
DEFINE_DISPATCH(special_hankel_h_1_stub);
DEFINE_DISPATCH(special_hankel_h_2_stub);
DEFINE_DISPATCH(special_harmonic_number_stub);
DEFINE_DISPATCH(special_incomplete_legendre_elliptic_integral_d_stub);
DEFINE_DISPATCH(special_incomplete_legendre_elliptic_integral_e_stub);
DEFINE_DISPATCH(special_incomplete_legendre_elliptic_integral_f_stub);
DEFINE_DISPATCH(special_ln_binomial_coefficient_stub);
DEFINE_DISPATCH(special_ln_double_factorial_stub);
DEFINE_DISPATCH(special_ln_factorial_stub);
DEFINE_DISPATCH(special_ln_falling_factorial_stub);
DEFINE_DISPATCH(special_ln_gamma_sign_stub);
DEFINE_DISPATCH(special_ln_gamma_stub);
DEFINE_DISPATCH(special_ln_rising_factorial_stub);
DEFINE_DISPATCH(special_lower_incomplete_gamma_stub);
DEFINE_DISPATCH(special_modified_bessel_i_stub);
DEFINE_DISPATCH(special_modified_bessel_k_stub);
DEFINE_DISPATCH(special_prime_number_stub);
DEFINE_DISPATCH(special_reciprocal_gamma_stub);
DEFINE_DISPATCH(special_rising_factorial_stub);
DEFINE_DISPATCH(special_sin_pi_stub);
DEFINE_DISPATCH(special_sinh_pi_stub);
DEFINE_DISPATCH(special_spherical_bessel_j_stub);
DEFINE_DISPATCH(special_spherical_bessel_y_stub);
DEFINE_DISPATCH(special_spherical_hankel_h_1_stub);
DEFINE_DISPATCH(special_spherical_hankel_h_2_stub);
DEFINE_DISPATCH(special_spherical_modified_bessel_i_stub);
DEFINE_DISPATCH(special_spherical_modified_bessel_k_stub);
DEFINE_DISPATCH(special_tan_pi_stub);
DEFINE_DISPATCH(special_tanh_pi_stub);
DEFINE_DISPATCH(special_upper_incomplete_gamma_stub);

TORCH_IMPL_FUNC (special_airy_bi_out)                                 (const Tensor &z,                    const Tensor &out) { special_airy_bi_stub                                 (device_type(), *this); }
TORCH_IMPL_FUNC (special_bernoulli_number_out)                        (const Tensor &n,                    const Tensor &out) { special_bernoulli_number_stub                        (device_type(), *this); }
TORCH_IMPL_FUNC (special_bessel_j_out)                                (const Tensor &v, const Tensor &z,   const Tensor &out) { special_bessel_j_stub                                (device_type(), *this); }
TORCH_IMPL_FUNC (special_bessel_y_out)                                (const Tensor &v, const Tensor &z,   const Tensor &out) { special_bessel_y_stub                                (device_type(), *this); }
TORCH_IMPL_FUNC (special_beta_out)                                    (const Tensor &a, const Tensor &b,   const Tensor &out) { special_beta_stub                                    (device_type(), *this); }
TORCH_IMPL_FUNC (special_binomial_coefficient_out)                    (const Tensor &m, const Tensor &n,   const Tensor &out) { special_binomial_coefficient_stub                    (device_type(), *this); }
TORCH_IMPL_FUNC (special_bulirsch_elliptic_integral_el1_out)          (const Tensor &x, const Tensor &k_c, const Tensor &out) { special_bulirsch_elliptic_integral_el1_stub          (device_type(), *this); }
TORCH_IMPL_FUNC (special_carlson_elliptic_r_c_out)                    (const Tensor &x, const Tensor &y,   const Tensor &out) { special_carlson_elliptic_r_c_stub                    (device_type(), *this); }
TORCH_IMPL_FUNC (special_complete_carlson_elliptic_r_f_out)           (const Tensor &x, const Tensor &y,   const Tensor &out) { special_complete_carlson_elliptic_r_f_stub           (device_type(), *this); }
TORCH_IMPL_FUNC (special_complete_carlson_elliptic_r_g_out)           (const Tensor &x, const Tensor &y,   const Tensor &out) { special_complete_carlson_elliptic_r_g_stub           (device_type(), *this); }
TORCH_IMPL_FUNC (special_complete_legendre_elliptic_integral_d_out)   (const Tensor &k,                    const Tensor &out) { special_complete_legendre_elliptic_integral_d_stub   (device_type(), *this); }
TORCH_IMPL_FUNC (special_complete_legendre_elliptic_integral_e_out)   (const Tensor &k,                    const Tensor &out) { special_complete_legendre_elliptic_integral_e_stub   (device_type(), *this); }
TORCH_IMPL_FUNC (special_complete_legendre_elliptic_integral_k_out)   (const Tensor &k,                    const Tensor &out) { special_complete_legendre_elliptic_integral_k_stub   (device_type(), *this); }
TORCH_IMPL_FUNC (special_complete_legendre_elliptic_integral_pi_out)  (const Tensor &n, const Tensor &k,   const Tensor &out) { special_complete_legendre_elliptic_integral_pi_stub  (device_type(), *this); }
TORCH_IMPL_FUNC (special_cos_pi_out)                                  (const Tensor &z,                    const Tensor &out) { special_cos_pi_stub                                  (device_type(), *this); }
TORCH_IMPL_FUNC (special_cosh_pi_out)                                 (const Tensor &z,                    const Tensor &out) { special_cosh_pi_stub                                 (device_type(), *this); }
TORCH_IMPL_FUNC (special_double_factorial_out)                        (const Tensor &n,                    const Tensor &out) { special_double_factorial_stub                        (device_type(), *this); }
TORCH_IMPL_FUNC (special_exp_airy_ai_out)                             (const Tensor &z,                    const Tensor &out) { special_exp_airy_ai_stub                             (device_type(), *this); }
TORCH_IMPL_FUNC (special_exp_airy_bi_out)                             (const Tensor &z,                    const Tensor &out) { special_exp_airy_bi_stub                             (device_type(), *this); }
TORCH_IMPL_FUNC (special_exp_modified_bessel_i_out)                   (const Tensor &v, const Tensor &z,   const Tensor &out) { special_exp_modified_bessel_i_stub                   (device_type(), *this); }
TORCH_IMPL_FUNC (special_exp_modified_bessel_k_out)                   (const Tensor &v, const Tensor &z,   const Tensor &out) { special_exp_modified_bessel_k_stub                   (device_type(), *this); }
TORCH_IMPL_FUNC (special_factorial_out)                               (const Tensor &n,                    const Tensor &out) { special_factorial_stub                               (device_type(), *this); }
TORCH_IMPL_FUNC (special_falling_factorial_out)                       (const Tensor &x, const Tensor &n,   const Tensor &out) { special_falling_factorial_stub                       (device_type(), *this); }
TORCH_IMPL_FUNC (special_gamma_out)                                   (const Tensor &z,                    const Tensor &out) { special_gamma_stub                                   (device_type(), *this); }
TORCH_IMPL_FUNC (special_hankel_h_1_out)                              (const Tensor &v, const Tensor &z,   const Tensor &out) { special_hankel_h_1_stub                              (device_type(), *this); }
TORCH_IMPL_FUNC (special_hankel_h_2_out)                              (const Tensor &v, const Tensor &z,   const Tensor &out) { special_hankel_h_2_stub                              (device_type(), *this); }
TORCH_IMPL_FUNC (special_harmonic_number_out)                         (const Tensor &n,                    const Tensor &out) { special_harmonic_number_stub                         (device_type(), *this); }
TORCH_IMPL_FUNC (special_incomplete_legendre_elliptic_integral_d_out) (const Tensor &k, const Tensor &phi, const Tensor &out) { special_incomplete_legendre_elliptic_integral_d_stub (device_type(), *this); }
TORCH_IMPL_FUNC (special_incomplete_legendre_elliptic_integral_e_out) (const Tensor &k, const Tensor &phi, const Tensor &out) { special_incomplete_legendre_elliptic_integral_e_stub (device_type(), *this); }
TORCH_IMPL_FUNC (special_incomplete_legendre_elliptic_integral_f_out) (const Tensor &k, const Tensor &phi, const Tensor &out) { special_incomplete_legendre_elliptic_integral_f_stub (device_type(), *this); }
TORCH_IMPL_FUNC (special_ln_binomial_coefficient_out)                 (const Tensor &m, const Tensor &n,   const Tensor &out) { special_ln_binomial_coefficient_stub                 (device_type(), *this); }
TORCH_IMPL_FUNC (special_ln_double_factorial_out)                     (const Tensor &n,                    const Tensor &out) { special_ln_double_factorial_stub                     (device_type(), *this); }
TORCH_IMPL_FUNC (special_ln_factorial_out)                            (const Tensor &n,                    const Tensor &out) { special_ln_factorial_stub                            (device_type(), *this); }
TORCH_IMPL_FUNC (special_ln_falling_factorial_out)                    (const Tensor &x, const Tensor &n,   const Tensor &out) { special_ln_falling_factorial_stub                    (device_type(), *this); }
TORCH_IMPL_FUNC (special_ln_gamma_out)                                (const Tensor &z,                    const Tensor &out) { special_ln_gamma_stub                                (device_type(), *this); }
TORCH_IMPL_FUNC (special_ln_gamma_sign_out)                           (const Tensor &z,                    const Tensor &out) { special_ln_gamma_sign_stub                           (device_type(), *this); }
TORCH_IMPL_FUNC (special_ln_rising_factorial_out)                     (const Tensor &x, const Tensor &n,   const Tensor &out) { special_ln_rising_factorial_stub                     (device_type(), *this); }
TORCH_IMPL_FUNC (special_lower_incomplete_gamma_out)                  (const Tensor &a, const Tensor &z,   const Tensor &out) { special_lower_incomplete_gamma_stub                  (device_type(), *this); }
TORCH_IMPL_FUNC (special_modified_bessel_i_out)                       (const Tensor &v, const Tensor &z,   const Tensor &out) { special_modified_bessel_i_stub                       (device_type(), *this); }
TORCH_IMPL_FUNC (special_modified_bessel_k_out)                       (const Tensor &v, const Tensor &z,   const Tensor &out) { special_modified_bessel_k_stub                       (device_type(), *this); }
TORCH_IMPL_FUNC (special_prime_number_out)                            (const Tensor &n,                    const Tensor &out) { special_prime_number_stub                            (device_type(), *this); }
TORCH_IMPL_FUNC (special_reciprocal_gamma_out)                        (const Tensor &z,                    const Tensor &out) { special_reciprocal_gamma_stub                        (device_type(), *this); }
TORCH_IMPL_FUNC (special_rising_factorial_out)                        (const Tensor &x, const Tensor &n,   const Tensor &out) { special_rising_factorial_stub                        (device_type(), *this); }
TORCH_IMPL_FUNC (special_sin_pi_out)                                  (const Tensor &z,                    const Tensor &out) { special_sin_pi_stub                                  (device_type(), *this); }
TORCH_IMPL_FUNC (special_sinh_pi_out)                                 (const Tensor &z,                    const Tensor &out) { special_sinh_pi_stub                                 (device_type(), *this); }
TORCH_IMPL_FUNC (special_spherical_bessel_j_out)                      (const Tensor &n, const Tensor &z,   const Tensor &out) { special_spherical_bessel_j_stub                      (device_type(), *this); }
TORCH_IMPL_FUNC (special_spherical_bessel_y_out)                      (const Tensor &n, const Tensor &z,   const Tensor &out) { special_spherical_bessel_y_stub                      (device_type(), *this); }
TORCH_IMPL_FUNC (special_spherical_hankel_h_1_out)                    (const Tensor &n, const Tensor &z,   const Tensor &out) { special_spherical_hankel_h_1_stub                    (device_type(), *this); }
TORCH_IMPL_FUNC (special_spherical_hankel_h_2_out)                    (const Tensor &n, const Tensor &z,   const Tensor &out) { special_spherical_hankel_h_2_stub                    (device_type(), *this); }
TORCH_IMPL_FUNC (special_spherical_modified_bessel_i_out)             (const Tensor &n, const Tensor &z,   const Tensor &out) { special_spherical_modified_bessel_i_stub             (device_type(), *this); }
TORCH_IMPL_FUNC (special_spherical_modified_bessel_k_out)             (const Tensor &n, const Tensor &z,   const Tensor &out) { special_spherical_modified_bessel_k_stub             (device_type(), *this); }
TORCH_IMPL_FUNC (special_tan_pi_out)                                  (const Tensor &z,                    const Tensor &out) { special_tan_pi_stub                                  (device_type(), *this); }
TORCH_IMPL_FUNC (special_tanh_pi_out)                                 (const Tensor &z,                    const Tensor &out) { special_tanh_pi_stub                                 (device_type(), *this); }
TORCH_IMPL_FUNC (special_upper_incomplete_gamma_out)                  (const Tensor &a, const Tensor &z,   const Tensor &out) { special_upper_incomplete_gamma_stub                  (device_type(), *this); }

Tensor  special_bessel_j                                    (const Scalar &v, const Tensor &z)                { return at::special_bessel_j                                    (     wrapped_scalar_tensor(v), z);                           }
Tensor  special_bessel_j                                    (const Tensor &v, const Scalar &z)                { return at::special_bessel_j                                    (     v,                        wrapped_scalar_tensor(z));    }
Tensor& special_bessel_j_out                                (const Tensor &v, const Scalar &z,   Tensor &out) { return at::special_bessel_j_out                                (out, wrapped_scalar_tensor(z), v);                           }
Tensor& special_bessel_j_out                                (const Scalar &v, const Tensor &z,   Tensor &out) { return at::special_bessel_j_out                                (out, z,                        wrapped_scalar_tensor(v));    }

Tensor  special_bessel_y                                    (const Scalar &v, const Tensor &z)                { return at::special_bessel_y                                    (     wrapped_scalar_tensor(v), z);                           }
Tensor  special_bessel_y                                    (const Tensor &v, const Scalar &z)                { return at::special_bessel_y                                    (     v,                        wrapped_scalar_tensor(z));    }
Tensor& special_bessel_y_out                                (const Tensor &v, const Scalar &z,   Tensor &out) { return at::special_bessel_y_out                                (out, wrapped_scalar_tensor(z), v);                           }
Tensor& special_bessel_y_out                                (const Scalar &v, const Tensor &z,   Tensor &out) { return at::special_bessel_y_out                                (out, z,                        wrapped_scalar_tensor(v));    }

Tensor  special_beta                                        (const Scalar &a, const Tensor &b)                { return at::special_beta                                        (     wrapped_scalar_tensor(a), b);                           }
Tensor  special_beta                                        (const Tensor &a, const Scalar &b)                { return at::special_beta                                        (     a,                        wrapped_scalar_tensor(b));    }
Tensor& special_beta_out                                    (const Tensor &a, const Scalar &b,   Tensor &out) { return at::special_beta_out                                    (out, wrapped_scalar_tensor(b), a);                           }
Tensor& special_beta_out                                    (const Scalar &a, const Tensor &b,   Tensor &out) { return at::special_beta_out                                    (out, b,                        wrapped_scalar_tensor(a));    }

Tensor  special_binomial_coefficient                        (const Scalar &m, const Tensor &n)                { return at::special_binomial_coefficient                        (     wrapped_scalar_tensor(m), n);                           }
Tensor  special_binomial_coefficient                        (const Tensor &m, const Scalar &n)                { return at::special_binomial_coefficient                        (     m,                        wrapped_scalar_tensor(n));    }
Tensor& special_binomial_coefficient_out                    (const Tensor &m, const Scalar &n,   Tensor &out) { return at::special_binomial_coefficient_out                    (out, wrapped_scalar_tensor(n), m);                           }
Tensor& special_binomial_coefficient_out                    (const Scalar &m, const Tensor &n,   Tensor &out) { return at::special_binomial_coefficient_out                    (out, n,                        wrapped_scalar_tensor(m));    }

Tensor  special_bulirsch_elliptic_integral_el1              (const Scalar &x, const Tensor &k_c)              { return at::special_bulirsch_elliptic_integral_el1              (     wrapped_scalar_tensor(x), k_c);                         }
Tensor  special_bulirsch_elliptic_integral_el1              (const Tensor &x, const Scalar &k_c)              { return at::special_bulirsch_elliptic_integral_el1              (     x,                        wrapped_scalar_tensor(k_c));  }
Tensor& special_bulirsch_elliptic_integral_el1_out          (const Tensor &x, const Scalar &k_c, Tensor &out) { return at::special_bulirsch_elliptic_integral_el1_out          (out, wrapped_scalar_tensor(k_c), x);                         }
Tensor& special_bulirsch_elliptic_integral_el1_out          (const Scalar &x, const Tensor &k_c, Tensor &out) { return at::special_bulirsch_elliptic_integral_el1_out          (out, k_c,                        wrapped_scalar_tensor(x));  }

Tensor  special_carlson_elliptic_r_c                        (const Scalar &x, const Tensor &y)                { return at::special_carlson_elliptic_r_c                        (     wrapped_scalar_tensor(x), y);                           }
Tensor  special_carlson_elliptic_r_c                        (const Tensor &x, const Scalar &y)                { return at::special_carlson_elliptic_r_c                        (     x,                        wrapped_scalar_tensor(y));    }
Tensor& special_carlson_elliptic_r_c_out                    (const Tensor &x, const Scalar &y,   Tensor &out) { return at::special_carlson_elliptic_r_c_out                    (out, wrapped_scalar_tensor(y), x);                           }
Tensor& special_carlson_elliptic_r_c_out                    (const Scalar &x, const Tensor &y,   Tensor &out) { return at::special_carlson_elliptic_r_c_out                    (out, y,                        wrapped_scalar_tensor(x));    }

Tensor  special_complete_carlson_elliptic_r_f               (const Scalar &x, const Tensor &y)                { return at::special_complete_carlson_elliptic_r_f               (     wrapped_scalar_tensor(x), y);                           }
Tensor  special_complete_carlson_elliptic_r_f               (const Tensor &x, const Scalar &y)                { return at::special_complete_carlson_elliptic_r_f               (     x,                        wrapped_scalar_tensor(y));    }
Tensor& special_complete_carlson_elliptic_r_f_out           (const Tensor &x, const Scalar &y,   Tensor &out) { return at::special_complete_carlson_elliptic_r_f_out           (out, wrapped_scalar_tensor(y), x);                           }
Tensor& special_complete_carlson_elliptic_r_f_out           (const Scalar &x, const Tensor &y,   Tensor &out) { return at::special_complete_carlson_elliptic_r_f_out           (out, y,                        wrapped_scalar_tensor(x));    }

Tensor  special_complete_carlson_elliptic_r_g               (const Scalar &x, const Tensor &y)                { return at::special_complete_carlson_elliptic_r_g               (     wrapped_scalar_tensor(x), y);                           }
Tensor  special_complete_carlson_elliptic_r_g               (const Tensor &x, const Scalar &y)                { return at::special_complete_carlson_elliptic_r_g               (     x,                        wrapped_scalar_tensor(y));    }
Tensor& special_complete_carlson_elliptic_r_g_out           (const Tensor &x, const Scalar &y,   Tensor &out) { return at::special_complete_carlson_elliptic_r_g_out           (out, wrapped_scalar_tensor(y), x);                           }
Tensor& special_complete_carlson_elliptic_r_g_out           (const Scalar &x, const Tensor &y,   Tensor &out) { return at::special_complete_carlson_elliptic_r_g_out           (out, y,                        wrapped_scalar_tensor(x));    }

Tensor  special_complete_legendre_elliptic_integral_pi      (const Scalar &n, const Tensor &k)                { return at::special_complete_legendre_elliptic_integral_pi      (     wrapped_scalar_tensor(n), k);                           }
Tensor  special_complete_legendre_elliptic_integral_pi      (const Tensor &n, const Scalar &k)                { return at::special_complete_legendre_elliptic_integral_pi      (     n,                        wrapped_scalar_tensor(k));    }
Tensor& special_complete_legendre_elliptic_integral_pi_out  (const Tensor &n, const Scalar &k,   Tensor &out) { return at::special_complete_legendre_elliptic_integral_pi_out  (out, wrapped_scalar_tensor(k), n);                           }
Tensor& special_complete_legendre_elliptic_integral_pi_out  (const Scalar &n, const Tensor &k,   Tensor &out) { return at::special_complete_legendre_elliptic_integral_pi_out  (out, k,                        wrapped_scalar_tensor(n));    }

Tensor  special_exp_modified_bessel_i                       (const Scalar &v, const Tensor &z)                { return at::special_exp_modified_bessel_i                       (     wrapped_scalar_tensor(v), z);                           }
Tensor  special_exp_modified_bessel_i                       (const Tensor &v, const Scalar &z)                { return at::special_exp_modified_bessel_i                       (     v,                        wrapped_scalar_tensor(z));    }
Tensor& special_exp_modified_bessel_i_out                   (const Tensor &v, const Scalar &z,   Tensor &out) { return at::special_exp_modified_bessel_i_out                   (out, wrapped_scalar_tensor(z), v);                           }
Tensor& special_exp_modified_bessel_i_out                   (const Scalar &v, const Tensor &z,   Tensor &out) { return at::special_exp_modified_bessel_i_out                   (out, z,                        wrapped_scalar_tensor(v));    }

Tensor  special_exp_modified_bessel_k                       (const Scalar &v, const Tensor &z)                { return at::special_exp_modified_bessel_k                       (     wrapped_scalar_tensor(v), z);                           }
Tensor  special_exp_modified_bessel_k                       (const Tensor &v, const Scalar &z)                { return at::special_exp_modified_bessel_k                       (     v,                        wrapped_scalar_tensor(z));    }
Tensor& special_exp_modified_bessel_k_out                   (const Tensor &v, const Scalar &z,   Tensor &out) { return at::special_exp_modified_bessel_k_out                   (out, wrapped_scalar_tensor(z), v);                           }
Tensor& special_exp_modified_bessel_k_out                   (const Scalar &v, const Tensor &z,   Tensor &out) { return at::special_exp_modified_bessel_k_out                   (out, z,                        wrapped_scalar_tensor(v));    }

Tensor  special_falling_factorial                           (const Scalar &x, const Tensor &n)                { return at::special_falling_factorial                           (     wrapped_scalar_tensor(x), n);                           }
Tensor  special_falling_factorial                           (const Tensor &x, const Scalar &n)                { return at::special_falling_factorial                           (     x,                        wrapped_scalar_tensor(n));    }
Tensor& special_falling_factorial_out                       (const Tensor &x, const Scalar &n,   Tensor &out) { return at::special_falling_factorial_out                       (out, wrapped_scalar_tensor(n), x);                           }
Tensor& special_falling_factorial_out                       (const Scalar &x, const Tensor &n,   Tensor &out) { return at::special_falling_factorial_out                       (out, n,                        wrapped_scalar_tensor(x));    }

Tensor  special_hankel_h_1                                  (const Scalar &v, const Tensor &z)                { return at::special_hankel_h_1                                  (     wrapped_scalar_tensor(v), z);                           }
Tensor  special_hankel_h_1                                  (const Tensor &v, const Scalar &z)                { return at::special_hankel_h_1                                  (     v,                        wrapped_scalar_tensor(z));    }
Tensor& special_hankel_h_1_out                              (const Tensor &v, const Scalar &z,   Tensor &out) { return at::special_hankel_h_1_out                              (out, wrapped_scalar_tensor(z), v);                           }
Tensor& special_hankel_h_1_out                              (const Scalar &v, const Tensor &z,   Tensor &out) { return at::special_hankel_h_1_out                              (out, z,                        wrapped_scalar_tensor(v));    }

Tensor  special_hankel_h_2                                  (const Scalar &v, const Tensor &z)                { return at::special_hankel_h_2                                  (     wrapped_scalar_tensor(v), z);                           }
Tensor  special_hankel_h_2                                  (const Tensor &v, const Scalar &z)                { return at::special_hankel_h_2                                  (     v,                        wrapped_scalar_tensor(z));    }
Tensor& special_hankel_h_2_out                              (const Tensor &v, const Scalar &z,   Tensor &out) { return at::special_hankel_h_2_out                              (out, wrapped_scalar_tensor(z), v);                           }
Tensor& special_hankel_h_2_out                              (const Scalar &v, const Tensor &z,   Tensor &out) { return at::special_hankel_h_2_out                              (out, z,                        wrapped_scalar_tensor(v));    }

Tensor  special_incomplete_legendre_elliptic_integral_d     (const Scalar &k, const Tensor &phi)              { return at::special_incomplete_legendre_elliptic_integral_d     (     wrapped_scalar_tensor(k), phi);                         }
Tensor  special_incomplete_legendre_elliptic_integral_d     (const Tensor &k, const Scalar &phi)              { return at::special_incomplete_legendre_elliptic_integral_d     (     k,                        wrapped_scalar_tensor(phi));  }
Tensor& special_incomplete_legendre_elliptic_integral_d_out (const Tensor &k, const Scalar &phi, Tensor &out) { return at::special_incomplete_legendre_elliptic_integral_d_out (out, wrapped_scalar_tensor(phi), k);                         }
Tensor& special_incomplete_legendre_elliptic_integral_d_out (const Scalar &k, const Tensor &phi, Tensor &out) { return at::special_incomplete_legendre_elliptic_integral_d_out (out, phi,                        wrapped_scalar_tensor(k));  }

Tensor  special_incomplete_legendre_elliptic_integral_e     (const Scalar &k, const Tensor &phi)              { return at::special_incomplete_legendre_elliptic_integral_e     (     wrapped_scalar_tensor(k), phi);                         }
Tensor  special_incomplete_legendre_elliptic_integral_e     (const Tensor &k, const Scalar &phi)              { return at::special_incomplete_legendre_elliptic_integral_e     (     k,                        wrapped_scalar_tensor(phi));  }
Tensor& special_incomplete_legendre_elliptic_integral_e_out (const Tensor &k, const Scalar &phi, Tensor &out) { return at::special_incomplete_legendre_elliptic_integral_e_out (out, wrapped_scalar_tensor(phi), k);                         }
Tensor& special_incomplete_legendre_elliptic_integral_e_out (const Scalar &k, const Tensor &phi, Tensor &out) { return at::special_incomplete_legendre_elliptic_integral_e_out (out, phi,                        wrapped_scalar_tensor(k));  }

Tensor  special_incomplete_legendre_elliptic_integral_f     (const Scalar &k, const Tensor &phi)              { return at::special_incomplete_legendre_elliptic_integral_f     (     wrapped_scalar_tensor(k), phi);                         }
Tensor  special_incomplete_legendre_elliptic_integral_f     (const Tensor &k, const Scalar &phi)              { return at::special_incomplete_legendre_elliptic_integral_f     (     k,                        wrapped_scalar_tensor(phi));  }
Tensor& special_incomplete_legendre_elliptic_integral_f_out (const Tensor &k, const Scalar &phi, Tensor &out) { return at::special_incomplete_legendre_elliptic_integral_f_out (out, wrapped_scalar_tensor(phi), k);                         }
Tensor& special_incomplete_legendre_elliptic_integral_f_out (const Scalar &k, const Tensor &phi, Tensor &out) { return at::special_incomplete_legendre_elliptic_integral_f_out (out, phi,                        wrapped_scalar_tensor(k));  }

Tensor  special_ln_binomial_coefficient                     (const Scalar &m, const Tensor &n)                { return at::special_ln_binomial_coefficient                     (     wrapped_scalar_tensor(m), n);                           }
Tensor  special_ln_binomial_coefficient                     (const Tensor &m, const Scalar &n)                { return at::special_ln_binomial_coefficient                     (     m,                        wrapped_scalar_tensor(n));    }
Tensor& special_ln_binomial_coefficient_out                 (const Tensor &m, const Scalar &n,   Tensor &out) { return at::special_ln_binomial_coefficient_out                 (out, wrapped_scalar_tensor(n), m);                           }
Tensor& special_ln_binomial_coefficient_out                 (const Scalar &m, const Tensor &n,   Tensor &out) { return at::special_ln_binomial_coefficient_out                 (out, n,                        wrapped_scalar_tensor(m));    }

Tensor  special_ln_falling_factorial                        (const Scalar &x, const Tensor &n)                { return at::special_ln_falling_factorial                        (     wrapped_scalar_tensor(x), n);                           }
Tensor  special_ln_falling_factorial                        (const Tensor &x, const Scalar &n)                { return at::special_ln_falling_factorial                        (     x,                        wrapped_scalar_tensor(n));    }
Tensor& special_ln_falling_factorial_out                    (const Tensor &x, const Scalar &n,   Tensor &out) { return at::special_ln_falling_factorial_out                    (out, wrapped_scalar_tensor(n), x);                           }
Tensor& special_ln_falling_factorial_out                    (const Scalar &x, const Tensor &n,   Tensor &out) { return at::special_ln_falling_factorial_out                    (out, n,                        wrapped_scalar_tensor(x));    }

Tensor  special_ln_rising_factorial                         (const Scalar &x, const Tensor &n)                { return at::special_ln_rising_factorial                         (     wrapped_scalar_tensor(x), n);                           }
Tensor  special_ln_rising_factorial                         (const Tensor &x, const Scalar &n)                { return at::special_ln_rising_factorial                         (     x,                        wrapped_scalar_tensor(n));    }
Tensor& special_ln_rising_factorial_out                     (const Tensor &x, const Scalar &n,   Tensor &out) { return at::special_ln_rising_factorial_out                     (out, wrapped_scalar_tensor(n), x);                           }
Tensor& special_ln_rising_factorial_out                     (const Scalar &x, const Tensor &n,   Tensor &out) { return at::special_ln_rising_factorial_out                     (out, n,                        wrapped_scalar_tensor(x));    }

Tensor  special_lower_incomplete_gamma                      (const Scalar &a, const Tensor &z)                { return at::special_lower_incomplete_gamma                      (     wrapped_scalar_tensor(a), z);                           }
Tensor  special_lower_incomplete_gamma                      (const Tensor &a, const Scalar &z)                { return at::special_lower_incomplete_gamma                      (     a,                        wrapped_scalar_tensor(z));    }
Tensor& special_lower_incomplete_gamma_out                  (const Tensor &a, const Scalar &z,   Tensor &out) { return at::special_lower_incomplete_gamma_out                  (out, wrapped_scalar_tensor(z), a);                           }
Tensor& special_lower_incomplete_gamma_out                  (const Scalar &a, const Tensor &z,   Tensor &out) { return at::special_lower_incomplete_gamma_out                  (out, z,                        wrapped_scalar_tensor(a));    }

Tensor  special_modified_bessel_i                           (const Scalar &v, const Tensor &z)                { return at::special_modified_bessel_i                           (     wrapped_scalar_tensor(v), z);                           }
Tensor  special_modified_bessel_i                           (const Tensor &v, const Scalar &z)                { return at::special_modified_bessel_i                           (     v,                        wrapped_scalar_tensor(z));    }
Tensor& special_modified_bessel_i_out                       (const Tensor &v, const Scalar &z,   Tensor &out) { return at::special_modified_bessel_i_out                       (out, wrapped_scalar_tensor(z), v);                           }
Tensor& special_modified_bessel_i_out                       (const Scalar &v, const Tensor &z,   Tensor &out) { return at::special_modified_bessel_i_out                       (out, z,                        wrapped_scalar_tensor(v));    }

Tensor  special_modified_bessel_k                           (const Scalar &v, const Tensor &z)                { return at::special_modified_bessel_k                           (     wrapped_scalar_tensor(v), z);                           }
Tensor  special_modified_bessel_k                           (const Tensor &v, const Scalar &z)                { return at::special_modified_bessel_k                           (     v,                        wrapped_scalar_tensor(z));    }
Tensor& special_modified_bessel_k_out                       (const Tensor &v, const Scalar &z,   Tensor &out) { return at::special_modified_bessel_k_out                       (out, wrapped_scalar_tensor(z), v);                           }
Tensor& special_modified_bessel_k_out                       (const Scalar &v, const Tensor &z,   Tensor &out) { return at::special_modified_bessel_k_out                       (out, z,                        wrapped_scalar_tensor(v));    }

Tensor  special_rising_factorial                            (const Scalar &x, const Tensor &n)                { return at::special_rising_factorial                            (     wrapped_scalar_tensor(x), n);                           }
Tensor  special_rising_factorial                            (const Tensor &x, const Scalar &n)                { return at::special_rising_factorial                            (     x,                        wrapped_scalar_tensor(n));    }
Tensor& special_rising_factorial_out                        (const Tensor &x, const Scalar &n,   Tensor &out) { return at::special_rising_factorial_out                        (out, wrapped_scalar_tensor(n), x);                           }
Tensor& special_rising_factorial_out                        (const Scalar &x, const Tensor &n,   Tensor &out) { return at::special_rising_factorial_out                        (out, n,                        wrapped_scalar_tensor(x));    }

Tensor  special_spherical_bessel_j                          (const Scalar &n, const Tensor &z)                { return at::special_spherical_bessel_j                          (     wrapped_scalar_tensor(n), z);                           }
Tensor  special_spherical_bessel_j                          (const Tensor &n, const Scalar &z)                { return at::special_spherical_bessel_j                          (     n,                        wrapped_scalar_tensor(z));    }
Tensor& special_spherical_bessel_j_out                      (const Tensor &n, const Scalar &z,   Tensor &out) { return at::special_spherical_bessel_j_out                      (out, wrapped_scalar_tensor(z), n);                           }
Tensor& special_spherical_bessel_j_out                      (const Scalar &n, const Tensor &z,   Tensor &out) { return at::special_spherical_bessel_j_out                      (out, z,                        wrapped_scalar_tensor(n));    }

Tensor  special_spherical_bessel_y                          (const Scalar &n, const Tensor &z)                { return at::special_spherical_bessel_y                          (     wrapped_scalar_tensor(n), z);                           }
Tensor  special_spherical_bessel_y                          (const Tensor &n, const Scalar &z)                { return at::special_spherical_bessel_y                          (     n,                        wrapped_scalar_tensor(z));    }
Tensor& special_spherical_bessel_y_out                      (const Tensor &n, const Scalar &z,   Tensor &out) { return at::special_spherical_bessel_y_out                      (out, wrapped_scalar_tensor(z), n);                           }
Tensor& special_spherical_bessel_y_out                      (const Scalar &n, const Tensor &z,   Tensor &out) { return at::special_spherical_bessel_y_out                      (out, z,                        wrapped_scalar_tensor(n));    }

Tensor  special_spherical_hankel_h_1                        (const Scalar &n, const Tensor &z)                { return at::special_spherical_hankel_h_1                        (     wrapped_scalar_tensor(n), z);                           }
Tensor  special_spherical_hankel_h_1                        (const Tensor &n, const Scalar &z)                { return at::special_spherical_hankel_h_1                        (     n,                        wrapped_scalar_tensor(z));    }
Tensor& special_spherical_hankel_h_1_out                    (const Tensor &n, const Scalar &z,   Tensor &out) { return at::special_spherical_hankel_h_1_out                    (out, wrapped_scalar_tensor(z), n);                           }
Tensor& special_spherical_hankel_h_1_out                    (const Scalar &n, const Tensor &z,   Tensor &out) { return at::special_spherical_hankel_h_1_out                    (out, z,                        wrapped_scalar_tensor(n));    }

Tensor  special_spherical_hankel_h_2                        (const Scalar &n, const Tensor &z)                { return at::special_spherical_hankel_h_2                        (     wrapped_scalar_tensor(n), z);                           }
Tensor  special_spherical_hankel_h_2                        (const Tensor &n, const Scalar &z)                { return at::special_spherical_hankel_h_2                        (     n,                        wrapped_scalar_tensor(z));    }
Tensor& special_spherical_hankel_h_2_out                    (const Tensor &n, const Scalar &z,   Tensor &out) { return at::special_spherical_hankel_h_2_out                    (out, wrapped_scalar_tensor(z), n);                           }
Tensor& special_spherical_hankel_h_2_out                    (const Scalar &n, const Tensor &z,   Tensor &out) { return at::special_spherical_hankel_h_2_out                    (out, z,                        wrapped_scalar_tensor(n));    }

Tensor  special_spherical_modified_bessel_i                 (const Scalar &n, const Tensor &z)                { return at::special_spherical_modified_bessel_i                 (     wrapped_scalar_tensor(n), z);                           }
Tensor  special_spherical_modified_bessel_i                 (const Tensor &n, const Scalar &z)                { return at::special_spherical_modified_bessel_i                 (     n,                        wrapped_scalar_tensor(z));    }
Tensor& special_spherical_modified_bessel_i_out             (const Tensor &n, const Scalar &z,   Tensor &out) { return at::special_spherical_modified_bessel_i_out             (out, wrapped_scalar_tensor(z), n);                           }
Tensor& special_spherical_modified_bessel_i_out             (const Scalar &n, const Tensor &z,   Tensor &out) { return at::special_spherical_modified_bessel_i_out             (out, z,                        wrapped_scalar_tensor(n));    }

Tensor  special_spherical_modified_bessel_k                 (const Scalar &n, const Tensor &z)                { return at::special_spherical_modified_bessel_k                 (     wrapped_scalar_tensor(n), z);                           }
Tensor  special_spherical_modified_bessel_k                 (const Tensor &n, const Scalar &z)                { return at::special_spherical_modified_bessel_k                 (     n,                        wrapped_scalar_tensor(z));    }
Tensor& special_spherical_modified_bessel_k_out             (const Tensor &n, const Scalar &z,   Tensor &out) { return at::special_spherical_modified_bessel_k_out             (out, wrapped_scalar_tensor(z), n);                           }
Tensor& special_spherical_modified_bessel_k_out             (const Scalar &n, const Tensor &z,   Tensor &out) { return at::special_spherical_modified_bessel_k_out             (out, z,                        wrapped_scalar_tensor(n));    }

Tensor  special_upper_incomplete_gamma                      (const Scalar &a, const Tensor &z)                { return at::special_upper_incomplete_gamma                      (     wrapped_scalar_tensor(a), z);                           }
Tensor  special_upper_incomplete_gamma                      (const Tensor &a, const Scalar &z)                { return at::special_upper_incomplete_gamma                      (     a,                        wrapped_scalar_tensor(z));    }
Tensor& special_upper_incomplete_gamma_out                  (const Tensor &a, const Scalar &z,   Tensor &out) { return at::special_upper_incomplete_gamma_out                  (out, wrapped_scalar_tensor(z), a);                           }
Tensor& special_upper_incomplete_gamma_out                  (const Scalar &a, const Tensor &z,   Tensor &out) { return at::special_upper_incomplete_gamma_out                  (out, z,                        wrapped_scalar_tensor(a));    }
} // namespace native
} // namespace at
