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
TORCH_META_FUNC (special_bernoulli_number)                        (const Tensor &n)                    { build_borrowing_unary_float_op  (maybe_get_output(), n);      }
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
TORCH_META_FUNC (special_factorial)                               (const Tensor &n)                    { build_borrowing_unary_float_op  (maybe_get_output(), n);      }
TORCH_META_FUNC (special_harmonic_number)                         (const Tensor &n)                    { build_borrowing_unary_float_op  (maybe_get_output(), n);      }
TORCH_META_FUNC (special_incomplete_legendre_elliptic_integral_d) (const Tensor &k, const Tensor &phi) { build_borrowing_binary_float_op (maybe_get_output(), k, phi); }
TORCH_META_FUNC (special_incomplete_legendre_elliptic_integral_e) (const Tensor &k, const Tensor &phi) { build_borrowing_binary_float_op (maybe_get_output(), k, phi); }
TORCH_META_FUNC (special_incomplete_legendre_elliptic_integral_f) (const Tensor &k, const Tensor &phi) { build_borrowing_binary_float_op (maybe_get_output(), k, phi); }
TORCH_META_FUNC (special_ln_gamma_sign)                           (const Tensor &z)                    { build_borrowing_unary_float_op  (maybe_get_output(), z);      }
TORCH_META_FUNC (special_sin_pi)                                  (const Tensor &z)                    { build_borrowing_unary_float_op  (maybe_get_output(), z);      }
TORCH_META_FUNC (special_sinh_pi)                                 (const Tensor &z)                    { build_borrowing_unary_float_op  (maybe_get_output(), z);      }
TORCH_META_FUNC (special_tan_pi)                                  (const Tensor &z)                    { build_borrowing_unary_float_op  (maybe_get_output(), z);      }
TORCH_META_FUNC (special_tanh_pi)                                 (const Tensor &z)                    { build_borrowing_unary_float_op  (maybe_get_output(), z);      }
} // namespace meta

namespace native {
DEFINE_DISPATCH(special_bernoulli_number_stub);
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
DEFINE_DISPATCH(special_factorial_stub);
DEFINE_DISPATCH(special_harmonic_number_stub);
DEFINE_DISPATCH(special_incomplete_legendre_elliptic_integral_d_stub);
DEFINE_DISPATCH(special_incomplete_legendre_elliptic_integral_e_stub);
DEFINE_DISPATCH(special_incomplete_legendre_elliptic_integral_f_stub);
DEFINE_DISPATCH(special_ln_gamma_sign_stub);
DEFINE_DISPATCH(special_sin_pi_stub);
DEFINE_DISPATCH(special_sinh_pi_stub);
DEFINE_DISPATCH(special_tan_pi_stub);
DEFINE_DISPATCH(special_tanh_pi_stub);

TORCH_IMPL_FUNC (special_bernoulli_number_out)                        (const Tensor &n,                    const Tensor &out) { special_bernoulli_number_stub                        (device_type(), *this); }
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
TORCH_IMPL_FUNC (special_factorial_out)                               (const Tensor &n,                    const Tensor &out) { special_factorial_stub                               (device_type(), *this); }
TORCH_IMPL_FUNC (special_harmonic_number_out)                         (const Tensor &n,                    const Tensor &out) { special_harmonic_number_stub                         (device_type(), *this); }
TORCH_IMPL_FUNC (special_incomplete_legendre_elliptic_integral_d_out) (const Tensor &k, const Tensor &phi, const Tensor &out) { special_incomplete_legendre_elliptic_integral_d_stub (device_type(), *this); }
TORCH_IMPL_FUNC (special_incomplete_legendre_elliptic_integral_e_out) (const Tensor &k, const Tensor &phi, const Tensor &out) { special_incomplete_legendre_elliptic_integral_e_stub (device_type(), *this); }
TORCH_IMPL_FUNC (special_incomplete_legendre_elliptic_integral_f_out) (const Tensor &k, const Tensor &phi, const Tensor &out) { special_incomplete_legendre_elliptic_integral_f_stub (device_type(), *this); }
TORCH_IMPL_FUNC (special_ln_gamma_sign_out)                           (const Tensor &z,                    const Tensor &out) { special_ln_gamma_sign_stub                           (device_type(), *this); }
TORCH_IMPL_FUNC (special_sin_pi_out)                                  (const Tensor &z,                    const Tensor &out) { special_sin_pi_stub                                  (device_type(), *this); }
TORCH_IMPL_FUNC (special_sinh_pi_out)                                 (const Tensor &z,                    const Tensor &out) { special_sinh_pi_stub                                 (device_type(), *this); }
TORCH_IMPL_FUNC (special_tan_pi_out)                                  (const Tensor &z,                    const Tensor &out) { special_tan_pi_stub                                  (device_type(), *this); }
TORCH_IMPL_FUNC (special_tanh_pi_out)                                 (const Tensor &z,                    const Tensor &out) { special_tanh_pi_stub                                 (device_type(), *this); }

Tensor  special_bulirsch_elliptic_integral_el1              (const Scalar &x, const Tensor &k_c)              { return at::special_bulirsch_elliptic_integral_el1              (     wrapped_scalar_tensor(x),   k_c);                        }
Tensor  special_bulirsch_elliptic_integral_el1              (const Tensor &x, const Scalar &k_c)              { return at::special_bulirsch_elliptic_integral_el1              (     x,                          wrapped_scalar_tensor(k_c)); }
Tensor& special_bulirsch_elliptic_integral_el1_out          (const Tensor &x, const Scalar &k_c, Tensor &out) { return at::special_bulirsch_elliptic_integral_el1_out          (out, wrapped_scalar_tensor(k_c), x);                          }
Tensor& special_bulirsch_elliptic_integral_el1_out          (const Scalar &x, const Tensor &k_c, Tensor &out) { return at::special_bulirsch_elliptic_integral_el1_out          (out, k_c,                        wrapped_scalar_tensor(x));   }

Tensor  special_carlson_elliptic_r_c                        (const Scalar &x, const Tensor &y)                { return at::special_carlson_elliptic_r_c                        (     wrapped_scalar_tensor(x),   y);                          }
Tensor  special_carlson_elliptic_r_c                        (const Tensor &x, const Scalar &y)                { return at::special_carlson_elliptic_r_c                        (     x,                          wrapped_scalar_tensor(y));   }
Tensor& special_carlson_elliptic_r_c_out                    (const Tensor &x, const Scalar &y, Tensor &out)   { return at::special_carlson_elliptic_r_c_out                    (out, wrapped_scalar_tensor(y),   x);                          }
Tensor& special_carlson_elliptic_r_c_out                    (const Scalar &x, const Tensor &y, Tensor &out)   { return at::special_carlson_elliptic_r_c_out                    (out, y,                          wrapped_scalar_tensor(x));   }

Tensor  special_complete_carlson_elliptic_r_f               (const Scalar &x, const Tensor &y)                { return at::special_complete_carlson_elliptic_r_f               (     wrapped_scalar_tensor(x),   y);                          }
Tensor  special_complete_carlson_elliptic_r_f               (const Tensor &x, const Scalar &y)                { return at::special_complete_carlson_elliptic_r_f               (     x,                          wrapped_scalar_tensor(y));   }
Tensor& special_complete_carlson_elliptic_r_f_out           (const Tensor &x, const Scalar &y, Tensor &out)   { return at::special_complete_carlson_elliptic_r_f_out           (out, wrapped_scalar_tensor(y),   x);                          }
Tensor& special_complete_carlson_elliptic_r_f_out           (const Scalar &x, const Tensor &y, Tensor &out)   { return at::special_complete_carlson_elliptic_r_f_out           (out, y,                          wrapped_scalar_tensor(x));   }

Tensor  special_complete_carlson_elliptic_r_g               (const Scalar &x, const Tensor &y)                { return at::special_complete_carlson_elliptic_r_g               (     wrapped_scalar_tensor(x),   y);                          }
Tensor  special_complete_carlson_elliptic_r_g               (const Tensor &x, const Scalar &y)                { return at::special_complete_carlson_elliptic_r_g               (     x,                          wrapped_scalar_tensor(y));   }
Tensor& special_complete_carlson_elliptic_r_g_out           (const Tensor &x, const Scalar &y, Tensor &out)   { return at::special_complete_carlson_elliptic_r_g_out           (out, wrapped_scalar_tensor(y),   x);                          }
Tensor& special_complete_carlson_elliptic_r_g_out           (const Scalar &x, const Tensor &y, Tensor &out)   { return at::special_complete_carlson_elliptic_r_g_out           (out, y,                          wrapped_scalar_tensor(x));   }

Tensor  special_complete_legendre_elliptic_integral_pi      (const Scalar &n, const Tensor &k)                { return at::special_complete_legendre_elliptic_integral_pi      (     wrapped_scalar_tensor(n),   k);                          }
Tensor  special_complete_legendre_elliptic_integral_pi      (const Tensor &n, const Scalar &k)                { return at::special_complete_legendre_elliptic_integral_pi      (     n,                          wrapped_scalar_tensor(k));   }
Tensor& special_complete_legendre_elliptic_integral_pi_out  (const Tensor &n, const Scalar &k, Tensor &out)   { return at::special_complete_legendre_elliptic_integral_pi_out  (out, wrapped_scalar_tensor(k),   n);                          }
Tensor& special_complete_legendre_elliptic_integral_pi_out  (const Scalar &n, const Tensor &k, Tensor &out)   { return at::special_complete_legendre_elliptic_integral_pi_out  (out, k,                          wrapped_scalar_tensor(n));   }

Tensor  special_incomplete_legendre_elliptic_integral_d     (const Scalar &k, const Tensor &phi)              { return at::special_incomplete_legendre_elliptic_integral_d     (     wrapped_scalar_tensor(k),   phi);                        }
Tensor  special_incomplete_legendre_elliptic_integral_d     (const Tensor &k, const Scalar &phi)              { return at::special_incomplete_legendre_elliptic_integral_d     (     k,                          wrapped_scalar_tensor(phi)); }
Tensor& special_incomplete_legendre_elliptic_integral_d_out (const Tensor &k, const Scalar &phi, Tensor &out) { return at::special_incomplete_legendre_elliptic_integral_d_out (out, wrapped_scalar_tensor(phi), k);                          }
Tensor& special_incomplete_legendre_elliptic_integral_d_out (const Scalar &k, const Tensor &phi, Tensor &out) { return at::special_incomplete_legendre_elliptic_integral_d_out (out, phi,                        wrapped_scalar_tensor(k));   }

Tensor  special_incomplete_legendre_elliptic_integral_e     (const Scalar &k, const Tensor &phi)              { return at::special_incomplete_legendre_elliptic_integral_e     (     wrapped_scalar_tensor(k),   phi);                        }
Tensor  special_incomplete_legendre_elliptic_integral_e     (const Tensor &k, const Scalar &phi)              { return at::special_incomplete_legendre_elliptic_integral_e     (     k,                          wrapped_scalar_tensor(phi)); }
Tensor& special_incomplete_legendre_elliptic_integral_e_out (const Tensor &k, const Scalar &phi, Tensor &out) { return at::special_incomplete_legendre_elliptic_integral_e_out (out, wrapped_scalar_tensor(phi), k);                          }
Tensor& special_incomplete_legendre_elliptic_integral_e_out (const Scalar &k, const Tensor &phi, Tensor &out) { return at::special_incomplete_legendre_elliptic_integral_e_out (out, phi,                        wrapped_scalar_tensor(k));   }

Tensor  special_incomplete_legendre_elliptic_integral_f     (const Scalar &k, const Tensor &phi)              { return at::special_incomplete_legendre_elliptic_integral_f     (     wrapped_scalar_tensor(k),   phi);                        }
Tensor  special_incomplete_legendre_elliptic_integral_f     (const Tensor &k, const Scalar &phi)              { return at::special_incomplete_legendre_elliptic_integral_f     (     k,                          wrapped_scalar_tensor(phi)); }
Tensor& special_incomplete_legendre_elliptic_integral_f_out (const Tensor &k, const Scalar &phi, Tensor &out) { return at::special_incomplete_legendre_elliptic_integral_f_out (out, wrapped_scalar_tensor(phi), k);                          }
Tensor& special_incomplete_legendre_elliptic_integral_f_out (const Scalar &k, const Tensor &phi, Tensor &out) { return at::special_incomplete_legendre_elliptic_integral_f_out (out, phi,                        wrapped_scalar_tensor(k));   }
} // namespace native
} // namespace at
