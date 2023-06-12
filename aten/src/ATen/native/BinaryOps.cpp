#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/BinaryOps.h>

#include <type_traits>
#include <utility>

#include <ATen/core/Tensor.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorMeta.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_add_relu_native.h>
#include <ATen/ops/_efficientzerotensor.h>
#include <ATen/ops/_test_serialization_subcmul_native.h>
#include <ATen/ops/_to_copy.h>
#include <ATen/ops/add.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/add_ops.h>
#include <ATen/ops/and_native.h>
#include <ATen/ops/arctan2_native.h>
#include <ATen/ops/atan2.h>
#include <ATen/ops/atan2_native.h>
#include <ATen/ops/bitwise_and.h>
#include <ATen/ops/bitwise_and_native.h>
#include <ATen/ops/bitwise_left_shift.h>
#include <ATen/ops/bitwise_left_shift_native.h>
#include <ATen/ops/bitwise_or.h>
#include <ATen/ops/bitwise_or_native.h>
#include <ATen/ops/bitwise_right_shift.h>
#include <ATen/ops/bitwise_right_shift_native.h>
#include <ATen/ops/bitwise_xor.h>
#include <ATen/ops/bitwise_xor_native.h>
#include <ATen/ops/copysign.h>
#include <ATen/ops/copysign_native.h>
#include <ATen/ops/div.h>
#include <ATen/ops/div_native.h>
#include <ATen/ops/div_ops.h>
#include <ATen/ops/divide_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/eq_native.h>
#include <ATen/ops/floor_divide.h>
#include <ATen/ops/floor_divide_native.h>
#include <ATen/ops/fmax_native.h>
#include <ATen/ops/fmin_native.h>
#include <ATen/ops/fmod.h>
#include <ATen/ops/fmod_native.h>
#include <ATen/ops/full.h>
#include <ATen/ops/gcd_native.h>
#include <ATen/ops/ge.h>
#include <ATen/ops/ge_native.h>
#include <ATen/ops/greater_equal_native.h>
#include <ATen/ops/greater_native.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/gt_native.h>
#include <ATen/ops/heaviside_native.h>
#include <ATen/ops/hypot_native.h>
#include <ATen/ops/igamma.h>
#include <ATen/ops/igamma_native.h>
#include <ATen/ops/igammac.h>
#include <ATen/ops/igammac_native.h>
#include <ATen/ops/lcm_native.h>
#include <ATen/ops/ldexp.h>
#include <ATen/ops/ldexp_native.h>
#include <ATen/ops/le.h>
#include <ATen/ops/le_native.h>
#include <ATen/ops/less_equal_native.h>
#include <ATen/ops/less_native.h>
#include <ATen/ops/linalg_cross_native.h>
#include <ATen/ops/linalg_cross_ops.h>
#include <ATen/ops/logaddexp2_native.h>
#include <ATen/ops/logaddexp_native.h>
#include <ATen/ops/logical_and.h>
#include <ATen/ops/logical_and_native.h>
#include <ATen/ops/logical_or.h>
#include <ATen/ops/logical_or_native.h>
#include <ATen/ops/logical_xor.h>
#include <ATen/ops/logical_xor_native.h>
#include <ATen/ops/logit_backward_native.h>
#include <ATen/ops/lshift_native.h>
#include <ATen/ops/lt.h>
#include <ATen/ops/lt_native.h>
#include <ATen/ops/max_native.h>
#include <ATen/ops/maximum.h>
#include <ATen/ops/maximum_native.h>
#include <ATen/ops/min_native.h>
#include <ATen/ops/minimum.h>
#include <ATen/ops/minimum_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/mul_ops.h>
#include <ATen/ops/multiply_native.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/ne_native.h>
#include <ATen/ops/nextafter_native.h>
#include <ATen/ops/not_equal_native.h>
#include <ATen/ops/or_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/remainder.h>
#include <ATen/ops/remainder_native.h>
#include <ATen/ops/rshift_native.h>
#include <ATen/ops/rsub_native.h>
#include <ATen/ops/sigmoid_backward_native.h>
#include <ATen/ops/special_chebyshev_polynomial_t.h>
#include <ATen/ops/special_chebyshev_polynomial_t_native.h>
#include <ATen/ops/special_chebyshev_polynomial_u.h>
#include <ATen/ops/special_chebyshev_polynomial_u_native.h>
#include <ATen/ops/special_chebyshev_polynomial_v.h>
#include <ATen/ops/special_chebyshev_polynomial_v_native.h>
#include <ATen/ops/special_chebyshev_polynomial_w.h>
#include <ATen/ops/special_chebyshev_polynomial_w_native.h>
#include <ATen/ops/special_gammainc_native.h>
#include <ATen/ops/special_gammaincc_native.h>
#include <ATen/ops/special_hermite_polynomial_h.h>
#include <ATen/ops/special_hermite_polynomial_h_native.h>
#include <ATen/ops/special_hermite_polynomial_he.h>
#include <ATen/ops/special_hermite_polynomial_he_native.h>
#include <ATen/ops/special_laguerre_polynomial_l.h>
#include <ATen/ops/special_laguerre_polynomial_l_native.h>
#include <ATen/ops/special_legendre_polynomial_p.h>
#include <ATen/ops/special_legendre_polynomial_p_native.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_t_native.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_u.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_u_native.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_v.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_v_native.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_w.h>
#include <ATen/ops/special_shifted_chebyshev_polynomial_w_native.h>
#include <ATen/ops/special_xlog1py.h>
#include <ATen/ops/special_xlog1py_native.h>
#include <ATen/ops/special_xlogy_native.h>
#include <ATen/ops/special_zeta.h>
#include <ATen/ops/special_zeta_native.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/sub_native.h>
#include <ATen/ops/subtract_native.h>
#include <ATen/ops/tanh_backward_native.h>
#include <ATen/ops/true_divide_native.h>
#include <ATen/ops/xlogy.h>
#include <ATen/ops/xlogy_native.h>
#include <ATen/ops/xor_native.h>
#endif

namespace at {

namespace meta {

TORCH_META_FUNC2(add, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}

TORCH_META_FUNC2(sub, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  native::sub_check(self, other);
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}

TORCH_META_FUNC2(mul, Tensor) (
  const Tensor& self, const Tensor& other
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(div, Tensor) (const Tensor& self, const Tensor& other) {
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(div, Tensor_mode) (const Tensor& self, const Tensor& other, c10::optional<c10::string_view> rounding_mode) {
  if (!rounding_mode.has_value()) {
    build_borrowing_binary_float_op(maybe_get_output(), self, other);
  // NOLINTNEXTLINE(bugprone-branch-clone)
  } else if (*rounding_mode == "trunc") {
    build_borrowing_binary_op(maybe_get_output(), self, other);
  } else if (*rounding_mode == "floor") {
    build_borrowing_binary_op(maybe_get_output(), self, other);
  } else {
    TORCH_CHECK(false,
        "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
        "but found '", *rounding_mode, "'");
  }
}

TORCH_META_FUNC(special_xlog1py) (const Tensor& self, const Tensor& other) {
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(special_zeta) (const Tensor& self, const Tensor& other) {
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(special_chebyshev_polynomial_t) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_chebyshev_polynomial_u) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_chebyshev_polynomial_v) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_chebyshev_polynomial_w) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_hermite_polynomial_h) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_hermite_polynomial_he) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_laguerre_polynomial_l) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_legendre_polynomial_p) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_shifted_chebyshev_polynomial_t) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_shifted_chebyshev_polynomial_u) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_shifted_chebyshev_polynomial_v) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC(special_shifted_chebyshev_polynomial_w) (const Tensor& self, const Tensor& n) {
  build_borrowing_binary_float_op(maybe_get_output(), self, n);
}

TORCH_META_FUNC2(copysign, Tensor) (
  const Tensor& self, const Tensor& other
) {
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(heaviside) (
  const Tensor& self, const Tensor& other
) {
  TORCH_CHECK(!self.is_complex() && !other.is_complex() &&
              (maybe_get_output().defined() ? !maybe_get_output().is_complex() : true),
              "heaviside is not yet implemented for complex tensors.");
  TORCH_CHECK(self.dtype() == other.dtype() &&
              (maybe_get_output().defined() ? maybe_get_output().dtype() == self.dtype() : true),
              "heaviside is not yet implemented for tensors with different dtypes.");

  build_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(atan2) (const Tensor& self, const Tensor& other) {
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(remainder, Tensor)(const Tensor& self, const Tensor& other) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(bitwise_left_shift, Tensor) (
  const Tensor& self, const Tensor& other
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(bitwise_right_shift, Tensor) (
  const Tensor& self, const Tensor& other
) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(bitwise_and, Tensor) (const Tensor& self, const Tensor& other) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(bitwise_or, Tensor) (const Tensor& self, const Tensor& other) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(bitwise_xor, Tensor) (const Tensor& self, const Tensor& other) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(fmod, Tensor) (const Tensor& self, const Tensor& other) {
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC2(xlogy, Tensor) (const Tensor& self, const Tensor& other) {
  build_borrowing_binary_float_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(logit_backward) (const Tensor& grad_output, const Tensor& input, c10::optional<double> eps) {
  build_borrowing_binary_op(maybe_get_output(), grad_output, input);
}

TORCH_META_FUNC(sigmoid_backward) (const Tensor& grad_output, const Tensor& output) {
  build_borrowing_binary_op(maybe_get_output(), grad_output, output);
}

TORCH_META_FUNC(tanh_backward) (const Tensor& grad_output, const Tensor& output) {
  build_borrowing_binary_op(maybe_get_output(), grad_output, output);
}

// These are normal binary ops that preserve dtype
#define CREATE_BINARY_META_FUNC(func)                                 \
  TORCH_META_FUNC(func) (const Tensor& self, const Tensor& other) {   \
    build_borrowing_binary_op(maybe_get_output(), self, other);                 \
  }

CREATE_BINARY_META_FUNC(logaddexp);
CREATE_BINARY_META_FUNC(logaddexp2);
CREATE_BINARY_META_FUNC(gcd);
CREATE_BINARY_META_FUNC(lcm);
CREATE_BINARY_META_FUNC(hypot);
CREATE_BINARY_META_FUNC(igamma);
CREATE_BINARY_META_FUNC(igammac);
CREATE_BINARY_META_FUNC(nextafter);

TORCH_META_FUNC(maximum) (const Tensor& self, const Tensor& other) {
  TORCH_CHECK(!self.is_complex() && !other.is_complex(), "maximum not implemented for complex tensors.");
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(minimum) (const Tensor& self, const Tensor& other) {
  TORCH_CHECK(!self.is_complex() && !other.is_complex(), "minimum not implemented for complex tensors.");
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(fmax) (const Tensor& self, const Tensor& other) {
    TORCH_CHECK(!self.is_complex() && !other.is_complex(), "fmax not implemented for complex tensors.");
    build_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(fmin) (const Tensor& self, const Tensor& other) {
    TORCH_CHECK(!self.is_complex() && !other.is_complex(), "fmin not implemented for complex tensors.");
    build_binary_op(maybe_get_output(), self, other);
}

#define CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(func)                     \
  TORCH_META_FUNC2(func, Tensor)(const Tensor& self, const Tensor& other) { \
    const Tensor& result = maybe_get_output();                              \
    build_borrowing_comparison_op(result, self, other);                     \
  }                                                                         \
                                                                            \
  TORCH_META_FUNC2(func, Scalar)(const Tensor& self, const Scalar& other) { \
    auto other_tensor =                                                     \
        native::wrapped_scalar_tensor(other);                               \
    build_borrowing_except_last_argument_comparison_op(maybe_get_output(), self, other_tensor);  \
  }

CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(eq);
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(ne);
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(lt);
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(le);
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(gt);
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(ge);

} // namespace meta


namespace native {

DEFINE_DISPATCH(add_clamp_stub);
DEFINE_DISPATCH(mul_stub);
DEFINE_DISPATCH(sub_stub);
DEFINE_DISPATCH(div_true_stub);
DEFINE_DISPATCH(div_floor_stub);
DEFINE_DISPATCH(div_trunc_stub);
DEFINE_DISPATCH(remainder_stub);
DEFINE_DISPATCH(atan2_stub);
DEFINE_DISPATCH(bitwise_and_stub);
DEFINE_DISPATCH(bitwise_or_stub);
DEFINE_DISPATCH(bitwise_xor_stub);
DEFINE_DISPATCH(lshift_stub);
DEFINE_DISPATCH(rshift_stub);
DEFINE_DISPATCH(logical_and_stub);
DEFINE_DISPATCH(logical_or_stub);
DEFINE_DISPATCH(logical_xor_stub);
DEFINE_DISPATCH(lt_stub);
DEFINE_DISPATCH(le_stub);
DEFINE_DISPATCH(gt_stub);
DEFINE_DISPATCH(ge_stub);
DEFINE_DISPATCH(eq_stub);
DEFINE_DISPATCH(ne_stub);
DEFINE_DISPATCH(sigmoid_backward_stub);
DEFINE_DISPATCH(logit_backward_stub);
DEFINE_DISPATCH(tanh_backward_stub);
DEFINE_DISPATCH(maximum_stub);
DEFINE_DISPATCH(minimum_stub);
DEFINE_DISPATCH(fmax_stub);
DEFINE_DISPATCH(fmin_stub);
DEFINE_DISPATCH(fmod_stub);
DEFINE_DISPATCH(logaddexp_stub);
DEFINE_DISPATCH(logaddexp2_stub);
DEFINE_DISPATCH(gcd_stub);
DEFINE_DISPATCH(lcm_stub);
DEFINE_DISPATCH(hypot_stub);
DEFINE_DISPATCH(igamma_stub);
DEFINE_DISPATCH(igammac_stub);
DEFINE_DISPATCH(nextafter_stub);
DEFINE_DISPATCH(heaviside_stub);
DEFINE_DISPATCH(copysign_stub);
DEFINE_DISPATCH(xlogy_stub);
DEFINE_DISPATCH(xlog1py_stub);
DEFINE_DISPATCH(zeta_stub);
DEFINE_DISPATCH(chebyshev_polynomial_t_stub);
DEFINE_DISPATCH(chebyshev_polynomial_u_stub);
DEFINE_DISPATCH(chebyshev_polynomial_v_stub);
DEFINE_DISPATCH(chebyshev_polynomial_w_stub);
DEFINE_DISPATCH(hermite_polynomial_h_stub);
DEFINE_DISPATCH(hermite_polynomial_he_stub);
DEFINE_DISPATCH(laguerre_polynomial_l_stub);
DEFINE_DISPATCH(legendre_polynomial_p_stub);
DEFINE_DISPATCH(shifted_chebyshev_polynomial_t_stub);
DEFINE_DISPATCH(shifted_chebyshev_polynomial_u_stub);
DEFINE_DISPATCH(shifted_chebyshev_polynomial_v_stub);
DEFINE_DISPATCH(shifted_chebyshev_polynomial_w_stub);

TORCH_IMPL_FUNC(sub_out) (
  const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& result
) {
  add_stub(device_type(), *this, -alpha);
  TORCH_INTERNAL_ASSERT(result.scalar_type() == output().dtype());
}

TORCH_IMPL_FUNC(mul_out) (
  const Tensor& self, const Tensor& other, const Tensor& result
) {
  mul_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(div_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  div_true_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(div_out_mode) (
  const Tensor& self, const Tensor& other, c10::optional<c10::string_view> rounding_mode, const Tensor& result
) {
  if (!rounding_mode.has_value()) {
    div_true_stub(device_type(), *this);
  } else if (*rounding_mode == "trunc") {
    div_trunc_stub(device_type(), *this);
  } else if (*rounding_mode == "floor") {
    div_floor_stub(device_type(), *this);
  }
}

TORCH_IMPL_FUNC(logit_backward_out) (const Tensor& grad_output, const Tensor& input, c10::optional<double> eps, const Tensor& result) {
  logit_backward_stub(device_type(), *this, Scalar(eps ? eps.value() : -1.0));
}

TORCH_IMPL_FUNC(sigmoid_backward_out) (const Tensor& grad_output, const Tensor& output, const Tensor& result) {
  sigmoid_backward_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(special_xlog1py_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  xlog1py_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(special_zeta_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  zeta_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(special_chebyshev_polynomial_t_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  chebyshev_polynomial_t_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(special_chebyshev_polynomial_u_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  chebyshev_polynomial_u_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(special_chebyshev_polynomial_v_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  chebyshev_polynomial_v_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(special_chebyshev_polynomial_w_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  chebyshev_polynomial_w_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(special_hermite_polynomial_h_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  hermite_polynomial_h_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(special_hermite_polynomial_he_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  hermite_polynomial_he_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(special_laguerre_polynomial_l_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  laguerre_polynomial_l_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(special_legendre_polynomial_p_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  legendre_polynomial_p_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(special_shifted_chebyshev_polynomial_t_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  shifted_chebyshev_polynomial_t_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(special_shifted_chebyshev_polynomial_u_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  shifted_chebyshev_polynomial_u_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(special_shifted_chebyshev_polynomial_v_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  shifted_chebyshev_polynomial_v_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(special_shifted_chebyshev_polynomial_w_out) (const Tensor& self, const Tensor& n, const Tensor& result) {
  shifted_chebyshev_polynomial_w_stub(device_type(), *this);
}

TORCH_IMPL_FUNC(tanh_backward_out) (const Tensor& grad_output, const Tensor& output, const Tensor& result) {
  tanh_backward_stub(device_type(), *this);
}

#define CREATE_BINARY_TORCH_IMPL_FUNC(func_out, func_stub)                                                    \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const Tensor& other, const Tensor& result) {  \
  func_stub(device_type(), *this);                                                           \
}

CREATE_BINARY_TORCH_IMPL_FUNC(bitwise_and_out, bitwise_and_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(bitwise_or_out, bitwise_or_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(bitwise_xor_out, bitwise_xor_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(maximum_out, maximum_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(minimum_out, minimum_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(fmax_out, fmax_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(fmin_out, fmin_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(fmod_out, fmod_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(logaddexp_out, logaddexp_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(logaddexp2_out, logaddexp2_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(gcd_out, gcd_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(lcm_out, lcm_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(hypot_out, hypot_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(igamma_out, igamma_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(igammac_out, igammac_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(nextafter_out, nextafter_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(remainder_out, remainder_stub);
CREATE_BINARY_TORCH_IMPL_FUNC(xlogy_out, xlogy_stub);

Tensor special_xlog1py(const Scalar& x, const Tensor& y) {
  return at::special_xlog1py(wrapped_scalar_tensor(x), y);
}

Tensor special_xlog1py(const Tensor& x, const Scalar& y) {
  return at::special_xlog1py(x, wrapped_scalar_tensor(y));
}

Tensor& special_xlog1py_out(const Scalar& self, const Tensor& other, Tensor& result) {
  return at::special_xlog1py_out(result, wrapped_scalar_tensor(self), other);
}

Tensor& special_xlog1py_out(const Tensor& self, const Scalar& other, Tensor& result) {
  return at::special_xlog1py_out(result, self, wrapped_scalar_tensor(other));
}

Tensor special_zeta(const Scalar& x, const Tensor& y) {
  return at::special_zeta(wrapped_scalar_tensor(x), y);
}

Tensor special_zeta(const Tensor& x, const Scalar& y) {
  return at::special_zeta(x, wrapped_scalar_tensor(y));
}

Tensor& special_zeta_out(const Scalar& self, const Tensor& other, Tensor& result) {
  return at::special_zeta_out(result, wrapped_scalar_tensor(self), other);
}

Tensor& special_zeta_out(const Tensor& self, const Scalar& other, Tensor& result) {
  return at::special_zeta_out(result, self, wrapped_scalar_tensor(other));
}

Tensor special_chebyshev_polynomial_t(const Scalar& x, const Tensor& n) {
  return at::special_chebyshev_polynomial_t(wrapped_scalar_tensor(x), n);
}

Tensor special_chebyshev_polynomial_t(const Tensor& x, const Scalar& n) {
  return at::special_chebyshev_polynomial_t(x, wrapped_scalar_tensor(n));
}

Tensor& special_chebyshev_polynomial_t_out(const Scalar& self, const Tensor& n, Tensor& result) {
  return at::special_chebyshev_polynomial_t_out(result, wrapped_scalar_tensor(self), n);
}

Tensor& special_chebyshev_polynomial_t_out(const Tensor& self, const Scalar& n, Tensor& result) {
  return at::special_chebyshev_polynomial_t_out(result, self, wrapped_scalar_tensor(n));
}

Tensor special_chebyshev_polynomial_u(const Scalar& x, const Tensor& n) {
  return at::special_chebyshev_polynomial_u(wrapped_scalar_tensor(x), n);
}

Tensor special_chebyshev_polynomial_u(const Tensor& x, const Scalar& n) {
  return at::special_chebyshev_polynomial_u(x, wrapped_scalar_tensor(n));
}

Tensor& special_chebyshev_polynomial_u_out(const Scalar& self, const Tensor& n, Tensor& result) {
  return at::special_chebyshev_polynomial_u_out(result, wrapped_scalar_tensor(self), n);
}

Tensor& special_chebyshev_polynomial_u_out(const Tensor& self, const Scalar& n, Tensor& result) {
  return at::special_chebyshev_polynomial_u_out(result, self, wrapped_scalar_tensor(n));
}

Tensor special_chebyshev_polynomial_v(const Scalar& x, const Tensor& n) {
  return at::special_chebyshev_polynomial_v(wrapped_scalar_tensor(x), n);
}

Tensor special_chebyshev_polynomial_v(const Tensor& x, const Scalar& n) {
  return at::special_chebyshev_polynomial_v(x, wrapped_scalar_tensor(n));
}

Tensor& special_chebyshev_polynomial_v_out(const Scalar& self, const Tensor& n, Tensor& result) {
  return at::special_chebyshev_polynomial_v_out(result, wrapped_scalar_tensor(self), n);
}

Tensor& special_chebyshev_polynomial_v_out(const Tensor& self, const Scalar& n, Tensor& result) {
  return at::special_chebyshev_polynomial_v_out(result, self, wrapped_scalar_tensor(n));
}

Tensor special_chebyshev_polynomial_w(const Scalar& x, const Tensor& n) {
  return at::special_chebyshev_polynomial_w(wrapped_scalar_tensor(x), n);
}

Tensor special_chebyshev_polynomial_w(const Tensor& x, const Scalar& n) {
  return at::special_chebyshev_polynomial_w(x, wrapped_scalar_tensor(n));
}

Tensor& special_chebyshev_polynomial_w_out(const Scalar& self, const Tensor& n, Tensor& result) {
  return at::special_chebyshev_polynomial_w_out(result, wrapped_scalar_tensor(self), n);
}

Tensor& special_chebyshev_polynomial_w_out(const Tensor& self, const Scalar& n, Tensor& result) {
  return at::special_chebyshev_polynomial_w_out(result, self, wrapped_scalar_tensor(n));
}

Tensor special_hermite_polynomial_h(const Scalar& x, const Tensor& n) {
  return at::special_hermite_polynomial_h(wrapped_scalar_tensor(x), n);
}

Tensor special_hermite_polynomial_h(const Tensor& x, const Scalar& n) {
  return at::special_hermite_polynomial_h(x, wrapped_scalar_tensor(n));
}

Tensor& special_hermite_polynomial_h_out(const Scalar& self, const Tensor& n, Tensor& result) {
  return at::special_hermite_polynomial_h_out(result, wrapped_scalar_tensor(self), n);
}

Tensor& special_hermite_polynomial_h_out(const Tensor& self, const Scalar& n, Tensor& result) {
  return at::special_hermite_polynomial_h_out(result, self, wrapped_scalar_tensor(n));
}

Tensor special_hermite_polynomial_he(const Scalar& x, const Tensor& n) {
  return at::special_hermite_polynomial_he(wrapped_scalar_tensor(x), n);
}

Tensor special_hermite_polynomial_he(const Tensor& x, const Scalar& n) {
  return at::special_hermite_polynomial_he(x, wrapped_scalar_tensor(n));
}

Tensor& special_hermite_polynomial_he_out(const Scalar& self, const Tensor& n, Tensor& result) {
  return at::special_hermite_polynomial_he_out(result, wrapped_scalar_tensor(self), n);
}

Tensor& special_hermite_polynomial_he_out(const Tensor& self, const Scalar& n, Tensor& result) {
  return at::special_hermite_polynomial_he_out(result, self, wrapped_scalar_tensor(n));
}

Tensor special_laguerre_polynomial_l(const Scalar& x, const Tensor& n) {
  return at::special_laguerre_polynomial_l(wrapped_scalar_tensor(x), n);
}

Tensor special_laguerre_polynomial_l(const Tensor& x, const Scalar& n) {
  return at::special_laguerre_polynomial_l(x, wrapped_scalar_tensor(n));
}

Tensor& special_laguerre_polynomial_l_out(const Scalar& self, const Tensor& n, Tensor& result) {
  return at::special_laguerre_polynomial_l_out(result, wrapped_scalar_tensor(self), n);
}

Tensor& special_laguerre_polynomial_l_out(const Tensor& self, const Scalar& n, Tensor& result) {
  return at::special_laguerre_polynomial_l_out(result, self, wrapped_scalar_tensor(n));
}

Tensor special_legendre_polynomial_p(const Scalar& x, const Tensor& n) {
  return at::special_legendre_polynomial_p(wrapped_scalar_tensor(x), n);
}

Tensor special_legendre_polynomial_p(const Tensor& x, const Scalar& n) {
  return at::special_legendre_polynomial_p(x, wrapped_scalar_tensor(n));
}

Tensor& special_legendre_polynomial_p_out(const Scalar& self, const Tensor& n, Tensor& result) {
  return at::special_legendre_polynomial_p_out(result, wrapped_scalar_tensor(self), n);
}

Tensor& special_legendre_polynomial_p_out(const Tensor& self, const Scalar& n, Tensor& result) {
  return at::special_legendre_polynomial_p_out(result, self, wrapped_scalar_tensor(n));
}

Tensor special_shifted_chebyshev_polynomial_t(const Scalar& x, const Tensor& n) {
  return at::special_shifted_chebyshev_polynomial_t(wrapped_scalar_tensor(x), n);
}

Tensor special_shifted_chebyshev_polynomial_t(const Tensor& x, const Scalar& n) {
  return at::special_shifted_chebyshev_polynomial_t(x, wrapped_scalar_tensor(n));
}

Tensor& special_shifted_chebyshev_polynomial_t_out(const Scalar& self, const Tensor& n, Tensor& result) {
  return at::special_shifted_chebyshev_polynomial_t_out(result, wrapped_scalar_tensor(self), n);
}

Tensor& special_shifted_chebyshev_polynomial_t_out(const Tensor& self, const Scalar& n, Tensor& result) {
  return at::special_shifted_chebyshev_polynomial_t_out(result, self, wrapped_scalar_tensor(n));
}

Tensor special_shifted_chebyshev_polynomial_u(const Scalar& x, const Tensor& n) {
  return at::special_shifted_chebyshev_polynomial_u(wrapped_scalar_tensor(x), n);
}

Tensor special_shifted_chebyshev_polynomial_u(const Tensor& x, const Scalar& n) {
  return at::special_shifted_chebyshev_polynomial_u(x, wrapped_scalar_tensor(n));
}

Tensor& special_shifted_chebyshev_polynomial_u_out(const Scalar& self, const Tensor& n, Tensor& result) {
  return at::special_shifted_chebyshev_polynomial_u_out(result, wrapped_scalar_tensor(self), n);
}

Tensor& special_shifted_chebyshev_polynomial_u_out(const Tensor& self, const Scalar& n, Tensor& result) {
  return at::special_shifted_chebyshev_polynomial_u_out(result, self, wrapped_scalar_tensor(n));
}

Tensor special_shifted_chebyshev_polynomial_v(const Scalar& x, const Tensor& n) {
  return at::special_shifted_chebyshev_polynomial_v(wrapped_scalar_tensor(x), n);
}

Tensor special_shifted_chebyshev_polynomial_v(const Tensor& x, const Scalar& n) {
  return at::special_shifted_chebyshev_polynomial_v(x, wrapped_scalar_tensor(n));
}

Tensor& special_shifted_chebyshev_polynomial_v_out(const Scalar& self, const Tensor& n, Tensor& result) {
  return at::special_shifted_chebyshev_polynomial_v_out(result, wrapped_scalar_tensor(self), n);
}

Tensor& special_shifted_chebyshev_polynomial_v_out(const Tensor& self, const Scalar& n, Tensor& result) {
  return at::special_shifted_chebyshev_polynomial_v_out(result, self, wrapped_scalar_tensor(n));
}

Tensor special_shifted_chebyshev_polynomial_w(const Scalar& x, const Tensor& n) {
  return at::special_shifted_chebyshev_polynomial_w(wrapped_scalar_tensor(x), n);
}

Tensor special_shifted_chebyshev_polynomial_w(const Tensor& x, const Scalar& n) {
  return at::special_shifted_chebyshev_polynomial_w(x, wrapped_scalar_tensor(n));
}

Tensor& special_shifted_chebyshev_polynomial_w_out(const Scalar& self, const Tensor& n, Tensor& result) {
  return at::special_shifted_chebyshev_polynomial_w_out(result, wrapped_scalar_tensor(self), n);
}

Tensor& special_shifted_chebyshev_polynomial_w_out(const Tensor& self, const Scalar& n, Tensor& result) {
  return at::special_shifted_chebyshev_polynomial_w_out(result, self, wrapped_scalar_tensor(n));
}

Tensor& special_gammainc_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return at::igamma_out(result, self, other);
}

Tensor special_gammainc(const Tensor& self, const Tensor& other) {
  return at::igamma(self, other);
}

Tensor& special_gammaincc_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return at::igammac_out(result, self, other);
}

Tensor special_gammaincc(const Tensor& self, const Tensor& other) {
  return at::igammac(self, other);
}

TORCH_IMPL_FUNC(atan2_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  atan2_stub(device_type(), *this);
}

Tensor arctan2(const Tensor& self, const Tensor& other) {
  return at::atan2(self, other);
}

Tensor& arctan2_(Tensor& self, const Tensor& other) {
  return self.atan2_(other);
}

Tensor& arctan2_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return at::atan2_out(result, self, other);
}

static Tensor& add_relu_impl(
    Tensor& result, const Tensor& self, const Tensor& other, const Scalar& alpha) {
  auto iter = TensorIterator::binary_op(result, self, other);
  Scalar min_val;
  Scalar max_val;
  if (self.dtype() == at::kInt) {
    min_val = 0;
    max_val = std::numeric_limits<int32_t>::max();
  } else if (self.dtype() == at::kLong) {
    min_val = 0;
    max_val = std::numeric_limits<int64_t>::max();
  } else if (self.dtype() == at::kShort) {
    min_val = 0;
    max_val = std::numeric_limits<int16_t>::max();
  } else if (self.dtype() == at::kChar) {
    min_val = 0;
    max_val = std::numeric_limits<int8_t>::max();
  } else if (self.dtype() == at::kFloat) {
    min_val = 0.0;
    max_val = std::numeric_limits<float>::max();
  } else if (self.dtype() == at::kDouble) {
    min_val = 0.0;
    max_val = std::numeric_limits<double>::max();
  } else {
    TORCH_INTERNAL_ASSERT(
        false, "Unsupported datatype for add_relu:", self.dtype().name());
  }

  result = iter.output();
  add_clamp_stub(iter.device_type(), iter, alpha, min_val, max_val);
  return result;
}

Tensor& add_relu_out(const Tensor& self, const Tensor& other, const Scalar& alpha, Tensor& result) {
  return add_relu_impl(result, self, other, alpha);
}

Tensor add_relu(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  Tensor result;
  return add_relu_impl(result, self, other, alpha);
}

Tensor add_relu(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return add_relu(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& add_relu_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return add_relu_impl(self, self, other, alpha);
}

Tensor& add_relu_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return add_relu_(self, wrapped_scalar_tensor(other), alpha);
}

TORCH_IMPL_FUNC(copysign_out) (
  const Tensor& self, const Tensor& other, const Tensor& result
) {
  copysign_stub(device_type(), *this);
}

Tensor copysign(const Tensor& self, const Scalar& other) {
  // redispatch!
  return at::copysign(self, wrapped_scalar_tensor(other));
}

Tensor& copysign_(Tensor& self, const Scalar& other) {
  // redispatch!
  return self.copysign_(wrapped_scalar_tensor(other));
}

Tensor& copysign_out(const Tensor& self, const Scalar& other, Tensor& result) {
  // redispatch!
  return at::copysign_out(result, self, wrapped_scalar_tensor(other));
}

// WARNING: There doesn't appear to be any testing for this function
// with sparse self input.
Tensor div(const Tensor& self, const Scalar& other) {
  return self.div(wrapped_scalar_tensor(other)); // redispatch!
}

// WARNING: This function, with a sparse self, is currently only
// exercised by DistributedDataParallelTest.test_sparse_gradients
// (you need to exercise it from C++, because this overload is never
// used for Python)
Tensor& div_(Tensor& self, const Scalar& other) {
  return self.div_(wrapped_scalar_tensor(other)); // redispatch!
}

Tensor div(const Tensor& self, const Scalar& other, c10::optional<c10::string_view> rounding_mode) {
  return self.div(wrapped_scalar_tensor(other), std::move(rounding_mode)); // redispatch!
}

Tensor& div_(Tensor& self, const Scalar& other, c10::optional<c10::string_view> rounding_mode) {
  return self.div_(wrapped_scalar_tensor(other), std::move(rounding_mode)); // redispatch!
}

// divide, alias for div
Tensor& divide_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return at::div_out(result, self, other);
}

Tensor divide(const Tensor& self, const Tensor& other) {
  return self.div(other);
}

Tensor& divide_(Tensor& self, const Tensor& other) {
  return self.div_(other);
}

Tensor divide(const Tensor& self, const Scalar& other) {
  return self.div(other);
}

Tensor& divide_(Tensor& self, const Scalar& other) {
  return self.div_(other);
}

Tensor& divide_out(const Tensor& self, const Tensor& other, c10::optional<c10::string_view> rounding_mode, Tensor& result) {
  return at::div_out(result, self, other, std::move(rounding_mode));
}

Tensor divide(const Tensor& self, const Tensor& other, c10::optional<c10::string_view> rounding_mode) {
  return self.div(other, std::move(rounding_mode));
}

Tensor& divide_(Tensor& self, const Tensor& other, c10::optional<c10::string_view> rounding_mode) {
  return self.div_(other, std::move(rounding_mode));
}

Tensor divide(const Tensor& self, const Scalar& other, c10::optional<c10::string_view> rounding_mode) {
  return self.div(other, std::move(rounding_mode));
}

Tensor& divide_(Tensor& self, const Scalar& other, c10::optional<c10::string_view> rounding_mode) {
  return self.div_(other, std::move(rounding_mode));
}

// true_divide, an alias for div
Tensor& true_divide_out(const Tensor& self, const Tensor& divisor, Tensor& result) {
  return at::div_out(result, self, divisor);
}

Tensor true_divide(const Tensor& self, const Tensor& divisor) {
  return self.div(divisor);
}

Tensor& true_divide_(Tensor& self, const Tensor& divisor) {
  return self.div_(divisor);
}

Tensor true_divide(const Tensor& self, const Scalar& divisor) {
  return self.div(divisor);
}

Tensor& true_divide_(Tensor& self, const Scalar& divisor) {
  return self.div_(divisor);
}

Tensor& floor_divide_out(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIterator::binary_op(result, self, other);
  div_floor_stub(iter.device_type(), iter);
  if (!result.defined()) {
    result = iter.output();
  }
  return result;
}

Tensor floor_divide(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  div_floor_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& floor_divide_(Tensor& self, const Tensor& other) {
  return native::floor_divide_out(self, other, self);
}

// TODO: Make this structured to undo the perf regression from native:: removal
// in call here
Tensor mul(const Tensor& self, const Scalar& other) {
  return at::mul(self, wrapped_scalar_tensor(other)); // redispatch!
}

Tensor& mul_(Tensor& self, const Scalar& other) {
  return at::mul_out(self, wrapped_scalar_tensor(other), self); // redispatch!
}

Tensor& mul__scalar_sparse_csr(Tensor& self, const Scalar& other) {
  self.values().mul_(other);
  return self;
}

static Device correct_out_device(const Tensor& self, const Tensor& other) {
  if (self.device() == at::kCPU){
      return other.device();
  } else {
    return self.device();
  }
}

Tensor mul_zerotensor(const Tensor& self, const Tensor& other) {
  auto out_device = correct_out_device(self, other);
  // hack to use the TensorIterator to get the correct broadcasting and type promotion logic
  auto device_ = Device(DeviceType::Meta);
  constexpr c10::DispatchKeySet meta_dks(at::DispatchKey::Meta);
  auto meta_out = at::_ops::mul_Tensor::redispatch(meta_dks, self.to(device_), other.to(device_));
  return at::_efficientzerotensor(meta_out.sizes(), meta_out.options().device(out_device));
}

Tensor div_zerotensor(const Tensor& self, const Tensor& other) {
  auto out_device = correct_out_device(self, other);
  // hack to use the TensorIterator to get the correct broadcasting and type promotion logic
  auto device_ = Device(DeviceType::Meta);
  constexpr c10::DispatchKeySet meta_dks(at::DispatchKey::Meta);
  auto meta_out = at::_ops::div_Tensor::redispatch(meta_dks, self.to(device_), other.to(device_));

  if (self._is_zerotensor()) {
    if (other._is_zerotensor()) {
      // 0/0, return full NAN
      return at::full(meta_out.sizes(), std::numeric_limits<float>::quiet_NaN(), meta_out.options().device(out_device));
    }
    else {
      // 0/x, return zero tensor
      return at::_efficientzerotensor(meta_out.sizes(), meta_out.options().device(out_device));
    }
  }
  else {
    if (other._is_zerotensor()) {
      // x/0, return full INF
      return at::full(meta_out.sizes(), std::numeric_limits<float>::infinity(), meta_out.options().device(out_device));
    }
    else {
      // x/y -- unreachable, see TORCH_INTERNAL_ASSERT above
      return at::_efficientzerotensor(meta_out.sizes(), meta_out.options().device(out_device));
    }
  }
}

static Tensor maybe_add_maybe_sub(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  auto out_device = correct_out_device(self, other);
  // hack to use the TensorIterator to get the correct broadcasting and type promotion logic
  auto device_ = Device(DeviceType::Meta);
  constexpr c10::DispatchKeySet meta_dks(at::DispatchKey::Meta);
  auto meta_out = at::_ops::add_Tensor::redispatch(
      meta_dks, self.to(device_), other.to(device_), alpha);

  auto get_out_like = [&] (const Tensor& tensor)
  {
      auto sizes = meta_out.sizes();
      return at::_to_copy(tensor.expand(sizes), meta_out.options().device(out_device));
  };

  if (self._is_zerotensor()) {
    if (other._is_zerotensor()) {
      return at::_efficientzerotensor(meta_out.sizes(), meta_out.options().device(out_device));
    }
    auto res = get_out_like(other);
    return alpha.equal(1) ? std::move(res) : res.mul(alpha);
  } else {
    return get_out_like(self);
  }
}
Tensor add_zerotensor(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return maybe_add_maybe_sub(self, other, alpha);
}

Tensor sub_zerotensor(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return maybe_add_maybe_sub(self, other, -alpha);
}

Tensor linalg_cross_zerotensor(
  const Tensor& input,
  const Tensor& other,
  const int64_t dim)
{
  auto out_device = correct_out_device(input, other);
  // hack to use the TensorIterator to get the correct broadcasting and type
  // promotion logic (see add_zerotensor)
  auto device = Device(DeviceType::Meta);
  auto meta_out = at::_ops::linalg_cross::redispatch(
    c10::DispatchKeySet(at::DispatchKey::Meta),
    input.to(device),
    other.to(device),
    dim);

  return at::_efficientzerotensor(
    meta_out.sizes(),
    meta_out.options().device(out_device));
}

// multiply, alias for mul
Tensor& multiply_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return at::mul_out(result, self, other);
}

Tensor multiply(const Tensor& self, const Tensor& other) {
  return self.mul(other);
}

Tensor& multiply_(Tensor& self, const Tensor& other) {
  return self.mul_(other);
}

Tensor multiply(const Tensor& self, const Scalar& other) {
  return self.mul(other);
}

Tensor& multiply_(Tensor& self, const Scalar& other) {
  return self.mul_(other);
}

Tensor sub(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return at::sub(self, wrapped_scalar_tensor(other), alpha); // redispatch!
}

Tensor& sub_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.sub_(wrapped_scalar_tensor(other), alpha); // redispatch!
}

// subtract, alias for sub
Tensor& subtract_out(const Tensor& self, const Tensor& other, const Scalar& alpha, Tensor& result) {
  return at::sub_out(result, self, other, alpha);
}

Tensor subtract(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return self.sub(other, alpha);
}

Tensor& subtract_(Tensor& self, const Tensor& other, const Scalar& alpha) {
  return self.sub_(other, alpha);
}

Tensor subtract(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.sub(other, alpha);
}

Tensor& subtract_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.sub_(other, alpha);
}

Tensor rsub(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return at::sub(other, self, alpha); // redispatch!
}

// TODO: Make this structured to undo the perf regression from native:: removal
// in call here

Tensor add(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return at::add(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& add_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.add_(wrapped_scalar_tensor(other), alpha);
}

Tensor remainder(const Tensor& self, const Scalar& other) {
  // redispatch
  return at::remainder(self, wrapped_scalar_tensor(other));
}

Tensor& remainder_(Tensor& self, const Scalar& other) {
  // redispatch
  return self.remainder_(wrapped_scalar_tensor(other));
}

Tensor& remainder_out(const Tensor& self, const Scalar& other, Tensor& result) {
  // redispatch
  return at::remainder_out(result, self, wrapped_scalar_tensor(other));
}

Tensor remainder(const Scalar& self, const Tensor& other) {
  return at::remainder(wrapped_scalar_tensor(self), other);
}

Tensor rsub(const Tensor& self, const Scalar& other, const Scalar& alpha) {
  return native::rsub(self, wrapped_scalar_tensor(other), alpha);
}

Tensor& bitwise_and_out(const Tensor& self, const Scalar& other, Tensor& result) {
  return at::bitwise_and_out(result, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_and(const Tensor& self, const Scalar& other) {
  return at::bitwise_and(self, wrapped_scalar_tensor(other));
}

Tensor bitwise_and(const Scalar& self, const Tensor& other) {
  return at::bitwise_and(wrapped_scalar_tensor(self), other);
}

Tensor& bitwise_and_(Tensor& self, const Scalar& other) {
  return self.bitwise_and_(wrapped_scalar_tensor(other));
}

// Legacy and interfaces. They are aliased to bitwise_and* functions
Tensor __and__(const Tensor& self, const Tensor& other) {
  return at::bitwise_and(self, other);
}

Tensor __and__(const Tensor& self, const Scalar& other) {
  return at::bitwise_and(self, other);
}

Tensor& __iand__(Tensor& self, const Tensor& other) {
  return self.bitwise_and_(other);
}

Tensor& __iand__(Tensor& self, const Scalar& other) {
  return self.bitwise_and_(other);
}

Tensor& bitwise_or_out(const Tensor& self, const Scalar& other, Tensor& result) {
  return at::bitwise_or_out(result, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_or(const Tensor& self, const Scalar& other) {
  return at::bitwise_or(self, wrapped_scalar_tensor(other));
}

Tensor bitwise_or(const Scalar& self, const Tensor& other) {
  return at::bitwise_or(wrapped_scalar_tensor(self), other);
}

Tensor& bitwise_or_(Tensor& self, const Scalar& other) {
  return self.bitwise_or_(wrapped_scalar_tensor(other));
}

// Legacy or interfaces. They are aliased to bitwise_or* functions
Tensor __or__(const Tensor& self, const Tensor& other) {
  return at::bitwise_or(self, other);
}

Tensor __or__(const Tensor& self, const Scalar& other) {
  return at::bitwise_or(self, other);
}

Tensor& __ior__(Tensor& self, const Tensor& other) {
  return self.bitwise_or_(other);
}

Tensor& __ior__(Tensor& self, const Scalar& other) {
  return self.bitwise_or_(other);
}

Tensor& bitwise_xor_out(const Tensor& self, const Scalar& other, Tensor& result) {
  return at::bitwise_xor_out(result, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_xor(const Tensor& self, const Scalar& other) {
  return at::bitwise_xor(self, wrapped_scalar_tensor(other));
}

Tensor bitwise_xor(const Scalar& self, const Tensor& other) {
  return at::bitwise_xor(wrapped_scalar_tensor(self), other);
}

Tensor& bitwise_xor_(Tensor& self, const Scalar& other) {
  return self.bitwise_xor_(wrapped_scalar_tensor(other));
}

// Legacy xor interfaces. They are aliased to bitwise_xor* functions
Tensor __xor__(const Tensor& self, const Tensor& other) {
  return at::bitwise_xor(self, other);
}

Tensor __xor__(const Tensor& self, const Scalar& other) {
  return at::bitwise_xor(self, other);
}

Tensor& __ixor__(Tensor& self, const Tensor& other) {
  return self.bitwise_xor_(other);
}

Tensor& __ixor__(Tensor& self, const Scalar& other) {
  return self.bitwise_xor_(other);
}

Tensor __lshift__(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  lshift_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor __lshift__(const Tensor& self, const Scalar& other) {
  Tensor result;
  auto wrapper = wrapped_scalar_tensor(other);
  auto iter = TensorIterator::binary_op(result, self, wrapper);
  lshift_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& __ilshift__(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(self, self, other);
  lshift_stub(iter.device_type(), iter);
  return self;
}

Tensor& __ilshift__(Tensor& self, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other);
  auto iter = TensorIterator::binary_op(self, self, wrapper);
  lshift_stub(iter.device_type(), iter);
  return self;
}

TORCH_IMPL_FUNC(bitwise_left_shift_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  lshift_stub(device_type(), *this);
}

Tensor& bitwise_left_shift_out(const Tensor& self, const Scalar& other, Tensor& result) {
  return at::bitwise_left_shift_out(result, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_left_shift(const Tensor& self, const Scalar& other) {
  return at::bitwise_left_shift(self, wrapped_scalar_tensor(other));
}

Tensor& bitwise_left_shift_(Tensor& self, const Scalar& other) {
  return at::bitwise_left_shift_out(self, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_left_shift(const Scalar& self, const Tensor& other) {
  return at::bitwise_left_shift(wrapped_scalar_tensor(self), other);
}

Tensor __rshift__(const Tensor& self, const Tensor& other) {
  Tensor result;
  auto iter = TensorIterator::binary_op(result, self, other);
  rshift_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor __rshift__(const Tensor& self, const Scalar& other) {
  Tensor result;
  auto wrapper = wrapped_scalar_tensor(other);
  auto iter = TensorIterator::binary_op(result, self, wrapper);
  rshift_stub(iter.device_type(), iter);
  return iter.output();
}

Tensor& __irshift__(Tensor& self, const Tensor& other) {
  auto iter = TensorIterator::binary_op(self, self, other);
  rshift_stub(iter.device_type(), iter);
  return self;
}

Tensor& __irshift__(Tensor& self, const Scalar& other) {
  auto wrapper = wrapped_scalar_tensor(other);
  auto iter = TensorIterator::binary_op(self, self, wrapper);
  rshift_stub(iter.device_type(), iter);
  return self;
}

TORCH_IMPL_FUNC(bitwise_right_shift_out) (const Tensor& self, const Tensor& other, const Tensor& result) {
  rshift_stub(device_type(), *this);
}

Tensor& bitwise_right_shift_out(const Tensor& self, const Scalar& other, Tensor& result) {
  return at::bitwise_right_shift_out(result, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_right_shift(const Tensor& self, const Scalar& other) {
  return at::bitwise_right_shift(self, wrapped_scalar_tensor(other));
}

Tensor& bitwise_right_shift_(Tensor& self, const Scalar& other) {
  return at::bitwise_right_shift_out(self, self, wrapped_scalar_tensor(other));
}

Tensor bitwise_right_shift(const Scalar& self, const Tensor& other) {
  return at::bitwise_right_shift(wrapped_scalar_tensor(self), other);
}

template <typename Stub>
Tensor& comparison_op_out(Tensor& result, const Tensor& self, const Tensor& other, Stub& stub) {
  auto iter = TensorIterator::comparison_op(result, self, other);
  stub(iter.device_type(), iter);
  return result;
}

template <typename OutImpl>
Tensor comparison_op(const Tensor& self, const Tensor& other, OutImpl& out_impl) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return out_impl(result, self, other);
}

template <typename OutImpl>
Tensor& comparison_op_(Tensor& self, const Tensor& other, OutImpl& out_impl) {
  return out_impl(self, self, other);
}

template <typename OutImpl>
Tensor& comparison_op_out(Tensor& result, const Tensor& self, const Scalar& other, OutImpl& out_impl) {
  return out_impl(result, self, wrapped_scalar_tensor(other));
}

template <typename OutImpl>
Tensor comparison_op(const Tensor& self, const Scalar& other, OutImpl& out_impl) {
  return comparison_op(self, wrapped_scalar_tensor(other), out_impl);
}

template <typename OutImpl>
Tensor& comparison_op_(Tensor& self, const Scalar& other, OutImpl& out_impl) {
  return out_impl(self, self, wrapped_scalar_tensor(other));
}

// We need explicit cast to OutFunc because each *_out func is overloaded twice. Without An explicit cast, merely
// referring to *_out function is ambiguious.
using OutFunc = std::add_const<Tensor&(&)(Tensor&, const Tensor&, const Tensor&)>::type;

// less, alias for torch.lt
Tensor& less_out(const Tensor& self, const Tensor& other, Tensor& result) { return at::lt_out(result, self, other); }
Tensor less(const Tensor& self, const Tensor& other) { return self.lt(other); }
Tensor& less_(Tensor& self, const Tensor& other) { return self.lt_(other); }
Tensor& less_out(const Tensor& self, const Scalar& other, Tensor& result) { return at::lt_out(result, self, other); }
Tensor less(const Tensor& self, const Scalar& other) { return self.lt(other); }
Tensor& less_(Tensor& self, const Scalar& other) { return self.lt_(other); }

// less_equal, alias for torch.le
Tensor& less_equal_out(const Tensor& self, const Tensor& other, Tensor& result) { return at::le_out(result, self, other); }
Tensor less_equal(const Tensor& self, const Tensor& other) { return self.le(other); }
Tensor& less_equal_(Tensor& self, const Tensor& other) { return self.le_(other); }
Tensor& less_equal_out(const Tensor& self, const Scalar& other, Tensor& result) { return at::le_out(result, self, other); }
Tensor less_equal(const Tensor& self, const Scalar& other) { return self.le(other); }
Tensor& less_equal_(Tensor& self, const Scalar& other) { return self.le_(other); }

// greater, alias for torch.gt
Tensor& greater_out(const Tensor& self, const Tensor& other, Tensor& result) { return at::gt_out(result, self, other); }
Tensor greater(const Tensor& self, const Tensor& other) { return self.gt(other); }
Tensor& greater_(Tensor& self, const Tensor& other) { return self.gt_(other); }
Tensor& greater_out(const Tensor& self, const Scalar& other, Tensor& result) { return at::gt_out(result, self, other); }
Tensor greater(const Tensor& self, const Scalar& other) { return self.gt(other); }
Tensor& greater_(Tensor& self, const Scalar& other) { return self.gt_(other); }

// greater_equal, alias for torch.ge
Tensor& greater_equal_out(const Tensor& self, const Tensor& other, Tensor& result) { return at::ge_out(result, self, other); }
Tensor greater_equal(const Tensor& self, const Tensor& other) { return self.ge(other); }
Tensor& greater_equal_(Tensor& self, const Tensor& other) { return self.ge_(other); }
Tensor& greater_equal_out(const Tensor& self, const Scalar& other, Tensor& result) { return at::ge_out(result, self, other); }
Tensor greater_equal(const Tensor& self, const Scalar& other) { return self.ge(other); }
Tensor& greater_equal_(Tensor& self, const Scalar& other) { return self.ge_(other); }

#define CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(func)             \
  TORCH_IMPL_FUNC(func##_Tensor_out)                                \
  (const Tensor& self, const Tensor& other, const Tensor& result) { \
    func##_stub(device_type(), *this);                              \
  }                                                                 \
                                                                    \
  TORCH_IMPL_FUNC(func##_Scalar_out)                                \
  (const Tensor& self, const Scalar& other, const Tensor& result) { \
    func##_stub(device_type(), *this);                              \
  }

CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(eq);
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(ne);
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(gt);
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(ge);
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(lt);
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(le);

// not_equal, alias for torch.ne
Tensor& not_equal_out(const Tensor& self, const Tensor& other, Tensor& result) { return at::ne_out(result, self, other); }
Tensor not_equal(const Tensor& self, const Tensor& other) { return self.ne(other); }
Tensor& not_equal_(Tensor& self, const Tensor& other) { return self.ne_(other); }
Tensor& not_equal_out(const Tensor& self, const Scalar& other, Tensor& result) { return at::ne_out(result, self, other); }
Tensor not_equal(const Tensor& self, const Scalar& other) { return self.ne(other); }
Tensor& not_equal_(Tensor& self, const Scalar& other) { return self.ne_(other); }

Tensor& logical_and_out(const Tensor& self, const Tensor& other, Tensor& result) { return comparison_op_out(result, self, other, logical_and_stub); }
Tensor logical_and(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_and_out)); }
Tensor& logical_and_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_and_out)); }
Tensor& logical_and_out(Tensor& result, const Tensor& self, const Scalar& other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::logical_and_out)); }
Tensor logical_and(const Tensor& self, const Scalar& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_and_out)); }
Tensor& logical_and_(Tensor& self, const Scalar& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_and_out)); }

Tensor& logical_or_out(const Tensor& self, const Tensor& other, Tensor& result) { return comparison_op_out(result, self, other, logical_or_stub); }
Tensor logical_or(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_or_out)); }
Tensor& logical_or_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_or_out)); }
Tensor& logical_or_out(Tensor& result, const Tensor& self, const Scalar& other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::logical_or_out)); }
Tensor logical_or(const Tensor& self, const Scalar& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_or_out)); }
Tensor& logical_or_(Tensor& self, const Scalar& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_or_out)); }

Tensor& logical_xor_out(const Tensor& self, const Tensor& other, Tensor& result) { return comparison_op_out(result, self, other, logical_xor_stub); }
Tensor logical_xor(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_xor_out)); }
Tensor& logical_xor_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_xor_out)); }
Tensor& logical_xor_out(Tensor& result, const Tensor& self, const Scalar& other) { return comparison_op_out(result, self, other, static_cast<OutFunc>(at::logical_xor_out)); }
Tensor logical_xor(const Tensor& self, const Scalar& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_xor_out)); }
Tensor& logical_xor_(Tensor& self, const Scalar& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_xor_out)); }

// binary max, alias for maximum
Tensor& max_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return at::maximum_out(result, self, other);
}

Tensor max(const Tensor& self, const Tensor& other) {
  return at::maximum(self, other);
}

// binary min, alias for minimum
Tensor& min_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return at::minimum_out(result, self, other);
}

Tensor min(const Tensor& self, const Tensor& other) {
  return at::minimum(self, other);
}

Tensor floor_divide(const Tensor& self, const Scalar& other) {
  return at::floor_divide(self, wrapped_scalar_tensor(other));
}

Tensor& floor_divide_(Tensor& self, const Scalar& other) {
  return at::floor_divide_out(self, self, wrapped_scalar_tensor(other));
}

Tensor& fmod_out(const Tensor& self, const Scalar& other, Tensor & result) {
  // redispatch
  return at::fmod_out(result, self, wrapped_scalar_tensor(other));
}

Tensor fmod(const Tensor& self, const Scalar& other) {
  // redispatch
  return at::fmod(self, wrapped_scalar_tensor(other));
}

Tensor& fmod_(Tensor& self, const Scalar& other) {
  // redispatch
  return self.fmod_(wrapped_scalar_tensor(other));
}

// Note: this function is only for testing.
// It is undocumented and should not be used outside of tests.
Tensor _test_serialization_subcmul(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  return self - (other * alpha);
}

TORCH_IMPL_FUNC(heaviside_out) (
  const Tensor& self, const Tensor& other, const Tensor& result
) {
  heaviside_stub(device_type(), *this);
}

Tensor& ldexp_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return at::mul_out(result, self, at::pow(2.0, other));
}

Tensor ldexp(const Tensor& self, const Tensor& other) {
  return at::mul(self, at::pow(2.0, other));
}

Tensor& ldexp_(Tensor& self, const Tensor& other) {
  return at::ldexp_out(self, self, other);
}

Tensor& xlogy_out(const Scalar& self, const Tensor& other, Tensor& result) {
  return at::xlogy_out(result, wrapped_scalar_tensor(self), other);
}

Tensor& xlogy_out(const Tensor& self, const Scalar& other, Tensor& result) {
  return at::xlogy_out(result, self, wrapped_scalar_tensor(other));
}

Tensor xlogy(const Scalar& x, const Tensor& y) {
  return at::xlogy(wrapped_scalar_tensor(x), y);
}

Tensor xlogy(const Tensor& x, const Scalar& y) {
  return at::xlogy(x, wrapped_scalar_tensor(y));
}

Tensor& xlogy_(Tensor& x, const Scalar& y) {
  return at::xlogy_(x, wrapped_scalar_tensor(y));
}

Tensor& special_xlogy_out(const Tensor& self, const Tensor& other, Tensor& result) {
  return at::xlogy_out(result, self, other);
}

Tensor& special_xlogy_out(const Scalar& self, const Tensor& other, Tensor& result) {
  return at::xlogy_out(result, self, other);
}

Tensor& special_xlogy_out(const Tensor& self, const Scalar& other, Tensor& result) {
  return at::xlogy_out(result, self, other);
}

Tensor special_xlogy(const Tensor& x, const Tensor& y) {
  return at::xlogy(x, y);
}

Tensor special_xlogy(const Scalar& x, const Tensor& y) {
  return at::xlogy(x, y);
}

Tensor special_xlogy(const Tensor& x, const Scalar& y) {
  return at::xlogy(x, y);
}

} // namespace native
} // namespace at
