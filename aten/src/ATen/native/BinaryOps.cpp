#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/BinaryOps.h>

#include <algorithm>
#include <array>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include <ATen/core/Tensor.h>
#include <ATen/EmptyTensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorMeta.h>
#include <ATen/native/TypeProperties.h>
#include <c10/core/DefaultDtype.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/util/irange.h>

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
#include <ATen/ops/empty_native.h>
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

namespace at::meta {

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

TORCH_META_FUNC2(div, Tensor_mode) (const Tensor& self, const Tensor& other, std::optional<std::string_view> rounding_mode) {
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
  TORCH_CHECK_NOT_IMPLEMENTED(!self.is_complex() && !other.is_complex() &&
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

TORCH_META_FUNC(logit_backward) (const Tensor& grad_output, const Tensor& input, std::optional<double> eps) {
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

CREATE_BINARY_META_FUNC(logaddexp)
CREATE_BINARY_META_FUNC(logaddexp2)
CREATE_BINARY_META_FUNC(gcd)
CREATE_BINARY_META_FUNC(lcm)
CREATE_BINARY_META_FUNC(hypot)
CREATE_BINARY_META_FUNC(igamma)
CREATE_BINARY_META_FUNC(igammac)
CREATE_BINARY_META_FUNC(nextafter)

TORCH_META_FUNC(maximum) (const Tensor& self, const Tensor& other) {
  TORCH_CHECK_TYPE(!self.is_complex() && !other.is_complex(), "maximum not implemented for complex tensors.");
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(minimum) (const Tensor& self, const Tensor& other) {
  TORCH_CHECK_TYPE(!self.is_complex() && !other.is_complex(), "minimum not implemented for complex tensors.");
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(fmax) (const Tensor& self, const Tensor& other) {
    TORCH_CHECK_TYPE(!self.is_complex() && !other.is_complex(), "fmax not implemented for complex tensors.");
    build_binary_op(maybe_get_output(), self, other);
}

TORCH_META_FUNC(fmin) (const Tensor& self, const Tensor& other) {
    TORCH_CHECK_TYPE(!self.is_complex() && !other.is_complex(), "fmin not implemented for complex tensors.");
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

CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(eq)
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(ne)
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(lt)
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(le)
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(gt)
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC(ge)

} // namespace at::meta


namespace at::native {

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
DEFINE_DISPATCH(ldexp_stub);

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
  const Tensor& self, const Tensor& other, std::optional<std::string_view> rounding_mode, const Tensor& result
) {
  if (!rounding_mode.has_value()) {
    div_true_stub(device_type(), *this);
  } else if (*rounding_mode == "trunc") {
    div_trunc_stub(device_type(), *this);
  } else if (*rounding_mode == "floor") {
    div_floor_stub(device_type(), *this);
  }
}

TORCH_IMPL_FUNC(logit_backward_out) (const Tensor& grad_output, const Tensor& input, std::optional<double> eps, const Tensor& result) {
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

CREATE_BINARY_TORCH_IMPL_FUNC(bitwise_and_out, bitwise_and_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(bitwise_or_out, bitwise_or_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(bitwise_xor_out, bitwise_xor_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(maximum_out, maximum_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(minimum_out, minimum_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(fmax_out, fmax_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(fmin_out, fmin_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(fmod_out, fmod_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(logaddexp_out, logaddexp_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(logaddexp2_out, logaddexp2_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(gcd_out, gcd_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(lcm_out, lcm_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(hypot_out, hypot_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(igamma_out, igamma_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(igammac_out, igammac_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(nextafter_out, nextafter_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(remainder_out, remainder_stub)
CREATE_BINARY_TORCH_IMPL_FUNC(xlogy_out, xlogy_stub)

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

Tensor div(const Tensor& self, const Scalar& other, std::optional<std::string_view> rounding_mode) {
  return self.div(wrapped_scalar_tensor(other), std::move(rounding_mode)); // redispatch!
}

Tensor& div_(Tensor& self, const Scalar& other, std::optional<std::string_view> rounding_mode) {
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

Tensor& divide_out(const Tensor& self, const Tensor& other, std::optional<std::string_view> rounding_mode, Tensor& result) {
  return at::div_out(result, self, other, std::move(rounding_mode));
}

Tensor divide(const Tensor& self, const Tensor& other, std::optional<std::string_view> rounding_mode) {
  return self.div(other, std::move(rounding_mode));
}

Tensor& divide_(Tensor& self, const Tensor& other, std::optional<std::string_view> rounding_mode) {
  return self.div_(other, std::move(rounding_mode));
}

Tensor divide(const Tensor& self, const Scalar& other, std::optional<std::string_view> rounding_mode) {
  return self.div(other, std::move(rounding_mode));
}

Tensor& divide_(Tensor& self, const Scalar& other, std::optional<std::string_view> rounding_mode) {
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

static Tensor send_to_meta(const Tensor& self, const Device& device) {
  Tensor out_meta;
  if (self._is_zerotensor() && self.unsafeGetTensorImpl()->is_wrapped_number()) {
    out_meta = at::_efficientzerotensor(self.sizes(), self.options().device(device));
    out_meta.unsafeGetTensorImpl()->set_wrapped_number(true);
  } else {
    out_meta = self.to(device);
  }
  return out_meta;
}

Tensor mul_zerotensor(const Tensor& self, const Tensor& other) {
  auto out_device = correct_out_device(self, other);
  // hack to use the TensorIterator to get the correct broadcasting and type promotion logic
  auto device_ = Device(DeviceType::Meta);
  constexpr c10::DispatchKeySet meta_dks(at::DispatchKey::Meta);
  auto self_meta = send_to_meta(self, device_);
  auto other_meta = send_to_meta(other, device_);
  auto meta_out = at::_ops::mul_Tensor::redispatch(meta_dks, self_meta, other_meta);
  return at::_efficientzerotensor(meta_out.sizes(), meta_out.options().device(out_device));
}

Tensor div_zerotensor(const Tensor& self, const Tensor& other) {
  auto out_device = correct_out_device(self, other);
  // hack to use the TensorIterator to get the correct broadcasting and type promotion logic
  auto device_ = Device(DeviceType::Meta);
  constexpr c10::DispatchKeySet meta_dks(at::DispatchKey::Meta);
  auto self_meta = send_to_meta(self, device_);
  auto other_meta = send_to_meta(other, device_);
  auto meta_out = at::_ops::div_Tensor::redispatch(meta_dks, self_meta, other_meta);

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
  auto self_meta = send_to_meta(self, device_);
  auto other_meta = send_to_meta(other, device_);
  auto meta_out = at::_ops::add_Tensor::redispatch(meta_dks, self_meta, other_meta, alpha);

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
static Tensor& comparison_op_out(Tensor& result, const Tensor& self, const Tensor& other, Stub& stub) {
  auto iter = TensorIterator::comparison_op(result, self, other);
  stub(iter.device_type(), iter);
  return result;
}

template <typename OutImpl>
static Tensor comparison_op(const Tensor& self, const Tensor& other, OutImpl& out_impl) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return out_impl(result, self, other);
}

template <typename OutImpl>
static Tensor& comparison_op_(Tensor& self, const Tensor& other, OutImpl& out_impl) {
  return out_impl(self, self, other);
}

template <typename OutImpl>
static Tensor& comparison_op_out(Tensor& result, const Tensor& self, const Scalar& other, OutImpl& out_impl) {
  return out_impl(result, self, wrapped_scalar_tensor(other));
}

template <typename OutImpl>
static Tensor comparison_op(const Tensor& self, const Scalar& other, OutImpl& out_impl) {
  return comparison_op(self, wrapped_scalar_tensor(other), out_impl);
}

template <typename OutImpl>
static Tensor& comparison_op_(Tensor& self, const Scalar& other, OutImpl& out_impl) {
  return out_impl(self, self, wrapped_scalar_tensor(other));
}

// We need explicit cast to OutFunc because each *_out func is overloaded twice. Without An explicit cast, merely
// referring to *_out function is ambiguous.
using OutFunc = std::add_const_t<Tensor&(&)(Tensor&, const Tensor&, const Tensor&)>;

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

CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(eq)
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(ne)
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(gt)
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(ge)
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(lt)
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC(le)

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

Tensor& logical_or_out(const Tensor& self, const Tensor& other, Tensor& result) { return comparison_op_out(result, self, other, logical_or_stub); }
Tensor logical_or(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_or_out)); }
Tensor& logical_or_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_or_out)); }

Tensor& logical_xor_out(const Tensor& self, const Tensor& other, Tensor& result) { return comparison_op_out(result, self, other, logical_xor_stub); }
Tensor logical_xor(const Tensor& self, const Tensor& other) { return comparison_op(self, other, static_cast<OutFunc>(at::logical_xor_out)); }
Tensor& logical_xor_(Tensor& self, const Tensor& other) { return comparison_op_(self, other, static_cast<OutFunc>(at::logical_xor_out)); }

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

static inline Tensor _pow2(const Tensor& self, const Tensor& other) {
  const auto self_dtype = self.scalar_type();
  // All integral types are promoted to float32
  if (isIntegralType(self_dtype, true) || self_dtype == kFloat) {
      return at::pow(2.0, other);
  }
  // For double and reduced floating types do regular type promotion
  return at::full({}, 2.0, self.options()).pow(other);
}

// This function is used to dispatch to kernels that use std::ldexp on CPU and the global namespaces ::ldexp on CUDA
// Both of these require floating types for 'self' and integer types for 'other'.
static inline Tensor& _ldexp_int_exponent(const Tensor& self, const Tensor& other, Tensor& result) {
  auto iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(result)
    .add_const_input(self)
    .add_const_input(other)
    .build();

  ldexp_stub(iter.device_type(), iter);
  return result;
}

Tensor& ldexp_out(const Tensor& self, const Tensor& other, Tensor& result) {
  TORCH_CHECK(!isIntegralType(result.scalar_type(), /*includeBool=*/true),
              "ldexp can't be cast to the desired output type ", result.scalar_type());

  if (isIntegralType(other.scalar_type(), /*includeBool=*/true) &&
      isFloatingType(self.scalar_type()) &&
      result.scalar_type() == self.scalar_type() &&
      ldexp_stub.is_device_supported(self.device().type())) {
    return _ldexp_int_exponent(self, other, result);
  }

  return at::mul_out(result, self, _pow2(self, other));
}

Tensor ldexp(const Tensor& self, const Tensor& other) {
  if (isIntegralType(other.scalar_type(), /*includeBool=*/true) &&
      isFloatingType(self.scalar_type()) &&
      ldexp_stub.is_device_supported(self.device().type())) {
    Tensor result = at::empty_like(self);
    return _ldexp_int_exponent(self, other, result);
  }

  return at::mul(self, _pow2(self, other));
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

namespace {

// ELEMENTWISE_TYPE_PROMOTION_KIND
enum class TypePromotionKind { DEFAULT, INT_TO_FLOAT };

// Tensor metadata as the Python refs see it. Wrapped numbers are Python
// numbers there and never take part in shape or stride logic.
struct MetaDesc {
  c10::SymDimVector sizes;
  c10::SymDimVector strides;
  ScalarType dtype = ScalarType::Undefined;
  Device device = kMeta;
  // utils.is_cpu_scalar_tensor
  bool cpu_scalar = false;
  bool is_number = false;
  // has_symbolic_sizes_strides; picks refs.expand over ATen's expand
  bool symbolic = false;

  int64_t dim() const {
    return static_cast<int64_t>(sizes.size());
  }
};

bool is_nested_int(const c10::SymInt& s) {
  return s.is_heap_allocated() && s.toSymNodeImplUnowned()->is_nested_int();
}

bool any_heap(c10::SymIntArrayRef xs) {
  return std::any_of(xs.begin(), xs.end(), [](const c10::SymInt& s) { return s.is_heap_allocated(); });
}

bool desc_is_symbolic(const MetaDesc& d) {
  return any_heap(d.sizes) || any_heap(d.strides);
}

c10::SymInt sym_numel(c10::SymIntArrayRef sizes) {
  c10::SymInt n = 1;
  for (const auto& s : sizes) {
    n = n * s;
  }
  return n;
}

// Python fake runs the ref under FakeTensorMode when the inputs are symbolic
// (fake devices visible, so cpu scalars are detected) and as the Meta kernel
// otherwise (every tensor reports device meta).
MetaDesc meta_desc(const Tensor& t, bool fake_devices) {
  MetaDesc d;
  d.sizes = c10::SymDimVector(t.sym_sizes().begin(), t.sym_sizes().end());
  d.strides = c10::SymDimVector(t.sym_strides().begin(), t.sym_strides().end());
  d.dtype = t.scalar_type();
  d.is_number = t.unsafeGetTensorImpl()->is_wrapped_number();
  const auto fake_device = t.unsafeGetTensorImpl()->fake_device();
  d.device = fake_device.has_value() ? (fake_devices ? *fake_device : Device(kMeta)) : t.device();
  d.cpu_scalar = !d.is_number && t.dim() == 0 && d.device.is_cpu();
  d.symbolic = t.unsafeGetTensorImpl()->has_symbolic_sizes_strides();
  return d;
}

// Python evaluates `a <op> b` with an int a and a SymInt b as b's reflected
// op, so the recorded guard is Eq(s0, 8) rather than Eq(8, s0).
bool reflects(const c10::SymInt& a, const c10::SymInt& b) {
  return !a.is_symbolic() && b.is_symbolic();
}

c10::SymBool py_eq(const c10::SymInt& a, const c10::SymInt& b) {
  return reflects(a, b) ? b.sym_eq(a) : a.sym_eq(b);
}

c10::SymBool py_ne(const c10::SymInt& a, const c10::SymInt& b) {
  return reflects(a, b) ? b.sym_ne(a) : a.sym_ne(b);
}

c10::SymBool py_lt(const c10::SymInt& a, const c10::SymInt& b) {
  return reflects(a, b) ? b.sym_gt(a) : a.sym_lt(b);
}

c10::SymBool py_ge(const c10::SymInt& a, const c10::SymInt& b) {
  return reflects(a, b) ? b.sym_le(a) : a.sym_ge(b);
}

// Identical nodes fold to true, like sympy's Eq(x, x).
c10::SymBool sym_eq_folded(const c10::SymInt& a, const c10::SymInt& b) {
  if (a.is_heap_allocated() && b.is_heap_allocated() && a.toSymNodeImplUnowned() == b.toSymNodeImplUnowned()) {
    return c10::SymBool(true);
  }
  return py_eq(a, b);
}

// bool(utils.is_same_shape(a, b))
bool is_same_shape(c10::SymIntArrayRef a, c10::SymIntArrayRef b) {
  if (a.size() != b.size()) {
    return false;
  }
  c10::SymBool result(true);
  for (const auto i : c10::irange(a.size())) {
    result = result.sym_and(sym_eq_folded(a[i], b[i]));
  }
  return result.guard_bool(__FILE__, __LINE__);
}

// utils.check_same_device(*args, allow_cpu_scalar_tensors=True)
void check_same_device(ArrayRef<MetaDesc> args) {
  const MetaDesc* first = nullptr;
  for (const auto& arg : args) {
    if (arg.is_number || arg.cpu_scalar) {
      continue;
    }
    if (first == nullptr) {
      first = &arg;
    }
    TORCH_CHECK(
        arg.device == first->device,
        "Tensor on device ", arg.device, " is not on the expected device ", first->device, "!");
  }
}

// utils.check_same_shape(*args, allow_cpu_scalar_tensors=True)
void check_same_shape(ArrayRef<MetaDesc> args) {
  const MetaDesc* first = nullptr;
  for (const auto& arg : args) {
    if (arg.is_number || arg.cpu_scalar) {
      continue;
    }
    if (first == nullptr) {
      first = &arg;
    }
    TORCH_CHECK(
        is_same_shape(first->sizes, arg.sizes),
        "Shape ", c10::SymIntArrayRef(arg.sizes), " is not the expected shape ", c10::SymIntArrayRef(first->sizes), "!");
  }
}

std::vector<const MetaDesc*> filter_tensors(ArrayRef<MetaDesc> args) {
  std::vector<const MetaDesc*> tensors;
  for (const auto& arg : args) {
    if (!arg.is_number && !arg.cpu_scalar) {
      tensors.push_back(&arg);
    }
  }
  return tensors;
}

// check_contiguous_sizes_strides(sizes, strides, false_if_dde=True)
bool check_contiguous_sizes_strides_or_false(c10::SymIntArrayRef sizes, c10::SymIntArrayRef strides) {
  c10::SymInt expected_stride = 1;
  c10::SymInt expected_stride_max = 1;
  for (int64_t i = static_cast<int64_t>(std::min(sizes.size(), strides.size())) - 1; i >= 0; --i) {
    const auto& x = sizes[i];
    const auto& y = strides[i];
    if (TORCH_GUARD_OR_FALSE(x.sym_eq(1))) {
      continue;
    }
    if (TORCH_GUARD_OR_TRUE(py_ne(y, expected_stride)) && TORCH_GUARD_OR_TRUE(py_ne(y, expected_stride_max))) {
      return false;
    }
    expected_stride_max = expected_stride_max * (is_nested_int(x) ? x : x.max(1));
    expected_stride = expected_stride * x;
  }
  return true;
}

// utils.is_contiguous_or_false
bool is_contiguous_or_false(const MetaDesc& a) {
  if (TORCH_GUARD_OR_FALSE(sym_numel(a.sizes).sym_lt(2))) {
    return true;
  }
  return check_contiguous_sizes_strides_or_false(a.sizes, a.strides);
}

// utils.is_channels_last_contiguous_or_false_2d
bool is_channels_last_contiguous_or_false(const MetaDesc& a) {
  if (a.dim() != 4) {
    return false;
  }
  c10::SymInt expected_stride = 1;
  for (const int64_t idx : {1, 3, 2, 0}) {
    const auto& length = a.sizes[idx];
    if (TORCH_GUARD_OR_FALSE(length.sym_eq(1))) {
      continue;
    }
    if (TORCH_GUARD_OR_TRUE(py_ne(a.strides[idx], expected_stride))) {
      return false;
    }
    expected_stride = expected_stride * length;
  }
  return true;
}

// K.__lt__ in _prims_common, on strides only.
bool stride_lt(const c10::SymInt& s, const c10::SymInt& o) {
  return TORCH_GUARD_OR_FALSE(py_lt(s, o)) ||
      ((TORCH_GUARD_OR_FALSE(s.sym_eq(0)) || TORCH_GUARD_OR_FALSE((o % s).sym_eq(0))) && TORCH_GUARD_OR_TRUE(py_ne(s, o)));
}

// CPython <= 3.12's list.sort for n < 64 (count_run + binarysort), so the
// non-transitive K.__lt__ sees the same calls in the same order as sorted().
template <typename Lt>
void python_sort(std::vector<int64_t>& a, Lt lt) {
  const size_t n = a.size();
  if (n < 2) {
    return;
  }
  if (n >= 64) {
    std::stable_sort(a.begin(), a.end(), lt);
    return;
  }
  size_t run = 2;
  if (lt(a[1], a[0])) {
    while (run < n && lt(a[run], a[run - 1])) {
      ++run;
    }
    std::reverse(a.begin(), a.begin() + run);
  } else {
    while (run < n && !lt(a[run], a[run - 1])) {
      ++run;
    }
  }
  for (size_t start = run; start < n; ++start) {
    const int64_t pivot = a[start];
    size_t l = 0;
    size_t r = start;
    while (l < r) {
      const size_t p = l + ((r - l) >> 1);
      if (lt(pivot, a[p])) {
        r = p;
      } else {
        l = p + 1;
      }
    }
    std::move_backward(a.begin() + l, a.begin() + start, a.begin() + start + 1);
    a[l] = pivot;
  }
}

// utils.is_non_overlapping_and_dense_or_false
bool is_non_overlapping_and_dense_or_false(const MetaDesc& a) {
  if (TORCH_GUARD_OR_FALSE(sym_numel(a.sizes).sym_lt(2))) {
    return true;
  }
  if (a.dim() == 1) {
    return TORCH_GUARD_OR_FALSE(a.strides[0].sym_eq(1));
  }
  std::vector<int64_t> order(a.dim());
  std::iota(order.begin(), order.end(), 0);
  python_sort(order, [&](int64_t i, int64_t j) { return stride_lt(a.strides[i], a.strides[j]); });
  c10::SymDimVector sorted_sizes;
  c10::SymDimVector sorted_strides;
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    sorted_sizes.push_back(a.sizes[*it]);
    sorted_strides.push_back(a.strides[*it]);
  }
  return check_contiguous_sizes_strides_or_false(sorted_sizes, sorted_strides);
}

// ge() inside should_swap: a >= b assuming a >= 0, b >= 0.
bool stride_ge(const c10::SymInt& a, const c10::SymInt& b) {
  if (TORCH_GUARD_OR_FALSE(b.sym_eq(0))) {
    return true;
  } else if (TORCH_GUARD_OR_FALSE(a.sym_eq(0))) {
    return false;
  }
  return TORCH_GUARD_OR_FALSE(py_ge(a, b)) || TORCH_GUARD_OR_FALSE((a % b).sym_eq(0));
}

// utils.compute_elementwise_output_logical_to_physical_perm after the shape
// check and the cpu scalar filtering.
DimVector l2p_perm(const std::vector<const MetaDesc*>& tensors) {
  if (tensors.empty()) {
    return {};
  }
  const int64_t ndim = tensors[0]->dim();
  if (ndim == 0) {
    return {};
  }
  if (ndim == 1) {
    return DimVector{0};
  }

  bool is_contiguous = true;
  bool is_channels_last = true;
  for (const auto* t : tensors) {
    is_contiguous = is_contiguous && is_contiguous_or_false(*t);
    is_channels_last = is_channels_last && is_channels_last_contiguous_or_false(*t);
  }

  DimVector perm(ndim);
  if (is_contiguous && !is_channels_last) {
    std::iota(perm.begin(), perm.end(), 0);
    return perm;
  }
  if (is_channels_last && !is_contiguous) {
    perm[0] = 0;
    std::iota(perm.begin() + 1, perm.end() - 1, 2);
    perm[ndim - 1] = 1;
    return perm;
  }

  const auto& shape = tensors[0]->sizes;
  auto should_swap = [&](int64_t idx_a, int64_t idx_b) -> int {
    for (const auto* t : tensors) {
      const auto& stride_a = t->strides[idx_a];
      const auto& stride_b = t->strides[idx_b];
      if (TORCH_GUARD_OR_FALSE(stride_a.sym_eq(0)) || TORCH_GUARD_OR_FALSE(stride_b.sym_eq(0))) {
        continue;
      }
      if (TORCH_GUARD_OR_FALSE(py_eq(stride_a, stride_b))) {
        if (stride_ge(shape[idx_b], shape[idx_a])) {
          continue;
        }
        return 1;
      }
      if (stride_ge(stride_b, stride_a)) {
        return -1;
      }
      if (stride_ge(stride_a, stride_b)) {
        return 1;
      }
    }
    return 0;
  };

  for (const auto i : c10::irange(ndim)) {
    perm[i] = ndim - 1 - i;
  }
  for (const auto i : c10::irange(1, ndim)) {
    int64_t dim1 = i;
    for (int64_t dim0 = i - 1; dim0 >= 0; --dim0) {
      const int comparison = should_swap(perm[dim0], perm[dim1]);
      if (comparison > 0) {
        std::swap(perm[dim0], perm[dim1]);
        dim1 = dim0;
      } else if (comparison < 0) {
        break;
      }
    }
  }
  std::reverse(perm.begin(), perm.end());
  return perm;
}

// torch.empty_permuted(shape, l2p_perm), i.e. empty_permuted_symint
MetaDesc empty_permuted_desc(c10::SymIntArrayRef shape, IntArrayRef l2p_perm, ScalarType dtype) {
  const int64_t dim = static_cast<int64_t>(shape.size());
  c10::SymDimVector phys_size(dim);
  for (const auto i : c10::irange(dim)) {
    phys_size[i] = shape[l2p_perm[i]];
  }
  // Contiguous strides as computed by empty_tensor_restride_symint.
  c10::SymDimVector phys_strides(dim);
  if (dim > 0) {
    phys_strides[dim - 1] = c10::SymInt(1);
    for (int64_t i = dim - 2; i >= 0; --i) {
      phys_strides[i] = phys_strides[i + 1] * phys_size[i + 1].max(1);
    }
  }
  MetaDesc out;
  out.sizes = c10::SymDimVector(shape.begin(), shape.end());
  out.strides = c10::SymDimVector(dim);
  for (const auto i : c10::irange(dim)) {
    out.strides[l2p_perm[i]] = phys_strides[i];
  }
  out.dtype = dtype;
  out.symbolic = desc_is_symbolic(out);
  return out;
}

// refs._broadcast_shapes
c10::SymDimVector broadcast_shapes(ArrayRef<c10::SymIntArrayRef> shapes) {
  size_t maxlen = 0;
  for (const auto& shape : shapes) {
    maxlen = std::max(maxlen, shape.size());
  }
  const int64_t common_len = static_cast<int64_t>(maxlen);
  c10::SymDimVector common_shape(maxlen, c10::SymInt(1));
  for (const auto arg_idx : c10::irange(shapes.size())) {
    const auto& shape = shapes[arg_idx];
    const int64_t len = static_cast<int64_t>(shape.size());
    for (int64_t idx = -1; idx >= -len; --idx) {
      const auto& s = shape[len + idx];
      auto& common = common_shape[common_len + idx];
      if (is_nested_int(s)) {
        if (is_nested_int(common) && TORCH_GUARD_OR_FALSE(py_eq(s, common))) {
          continue;
        }
      } else if (TORCH_GUARD_OR_FALSE(py_eq(s, common))) {
        continue;
      }

      if (TORCH_GUARD_OR_FALSE(common.sym_eq(1))) {
        TORCH_CHECK_VALUE(
            !s.sym_lt(0).guard_bool(__FILE__, __LINE__), "Attempting to broadcast a dimension with negative length!");
        common = s;
      }

      if (!is_nested_int(s) && TORCH_GUARD_OR_FALSE(s.sym_eq(1))) {
        continue;
      }
      TORCH_SYM_CHECK(
          sym_eq_folded(common, s),
          "Attempting to broadcast a dimension of length ", s, " at ", idx, "! Mismatching argument at index ", arg_idx,
          " had ", shape, "; but expected shape should be broadcastable to ", c10::SymIntArrayRef(common_shape));
    }
  }
  return common_shape;
}

// should_expand inside refs._maybe_broadcast
bool should_expand(c10::SymIntArrayRef a, c10::SymIntArrayRef b) {
  if (a.size() != b.size()) {
    return true;
  }
  for (const auto i : c10::irange(a.size())) {
    const auto& x = a[i];
    const auto& y = b[i];
    if (TORCH_GUARD_OR_FALSE(py_ne(x, y))) {
      return true;
    }
    if (!TORCH_GUARD_OR_FALSE(x.sym_eq(1).sym_and(y.sym_eq(1))) && TORCH_GUARD_OR_FALSE(x.sym_eq(1).sym_or(y.sym_eq(1)))) {
      return true;
    }
    TORCH_SYM_CHECK(py_eq(x, y), "sizes assumed to be the same due to unbacked broadcasting semantics");
  }
  return false;
}

// prims.broadcast_in_dim meta
MetaDesc broadcast_in_dim_desc(const MetaDesc& a, c10::SymIntArrayRef shape, IntArrayRef broadcast_dimensions) {
  const int64_t ndim = a.dim();
  const int64_t out_ndim = static_cast<int64_t>(shape.size());
  for (const auto idx : c10::irange(ndim)) {
    const auto new_idx = broadcast_dimensions[idx];
    TORCH_SYM_CHECK(
        a.sizes[idx].sym_eq(1).sym_or(py_eq(shape[new_idx], a.sizes[idx])),
        a.sizes[idx], " must be broadcastable to ", shape[new_idx]);
  }

  c10::SymDimVector new_strides;
  new_strides.reserve(out_ndim);
  int64_t original_idx = 0;
  for (const auto idx : c10::irange(out_ndim)) {
    if (std::find(broadcast_dimensions.begin(), broadcast_dimensions.end(), idx) != broadcast_dimensions.end()) {
      const auto& size = a.sizes[original_idx];
      if (TORCH_GUARD_OR_FALSE(size.sym_eq(1))) {
        new_strides.push_back(TORCH_GUARD_OR_FALSE(py_eq(size, shape[idx])) ? a.strides[original_idx] : c10::SymInt(0));
      } else {
        TORCH_SYM_CHECK(py_eq(size, shape[idx]), "non-broadcasting semantics require ", size, " == ", shape[idx]);
        new_strides.push_back(a.strides[original_idx]);
      }
      original_idx++;
    } else if (TORCH_GUARD_OR_TRUE(shape[idx].sym_ne(1))) {
      new_strides.push_back(c10::SymInt(0));
    } else if (original_idx == ndim) {
      new_strides.push_back(c10::SymInt(1));
    } else {
      new_strides.push_back(a.strides[original_idx] * a.sizes[original_idx]);
    }
  }

  MetaDesc out;
  out.sizes = c10::SymDimVector(shape.begin(), shape.end());
  out.strides = std::move(new_strides);
  out.dtype = a.dtype;
  out.device = a.device;
  out.cpu_scalar = a.cpu_scalar && shape.empty();
  out.symbolic = desc_is_symbolic(out);
  return out;
}

// refs.expand(a, shape) lowering to prims.broadcast_in_dim
MetaDesc expand_desc(const MetaDesc& a, c10::SymIntArrayRef shape) {
  const int64_t ndim = a.dim();
  TORCH_CHECK(static_cast<int64_t>(shape.size()) >= ndim, "expand: the requested shape has too few dimensions!");
  const int64_t offset = static_cast<int64_t>(shape.size()) - ndim;
  c10::SymDimVector shape_(shape.begin(), shape.end());
  for (const auto idx : c10::irange(ndim)) {
    const auto& x = a.sizes[idx];
    const int64_t offset_idx = idx + offset;
    const auto& requested_length = shape[offset_idx];
    if (TORCH_GUARD_OR_FALSE(requested_length.sym_eq(-1))) {
      shape_[offset_idx] = x;
    } else {
      TORCH_SYM_CHECK(
          x.sym_eq(1).sym_or(py_eq(requested_length, x)),
          "expand: attempting to expand a dimension of length ", x, " -> ", requested_length, "!");
      TORCH_SYM_CHECK(requested_length.sym_ge(0), "expand: expected a non-negative length, got ", requested_length);
      shape_[offset_idx] = requested_length;
    }
  }
  for (const auto& l : shape_) {
    TORCH_SYM_CHECK(l.sym_ge(0), "Expected a non-negative length, got ", l);
  }
  DimVector broadcast_dimensions(ndim);
  std::iota(broadcast_dimensions.begin(), broadcast_dimensions.end(), offset);
  return broadcast_in_dim_desc(a, shape_, broadcast_dimensions);
}

// Tensor.expand's CompositeExplicitAutograd kernel, which Python fake runs
// when neither the tensor nor the size is symbolic.
MetaDesc aten_expand_desc(const MetaDesc& a, c10::SymIntArrayRef size) {
  TORCH_CHECK(
      size.size() >= static_cast<size_t>(a.dim()),
      "expand(size=", size, "): the number of sizes provided (", size.size(),
      ") must be greater or equal to the number of dimensions in the tensor (", a.dim(), ")");
  auto geometry = at::inferExpandGeometry_dimvector(
      c10::asIntArrayRefUnchecked(a.sizes), c10::asIntArrayRefUnchecked(a.strides), c10::asIntArrayRefUnchecked(size));
  MetaDesc out;
  out.sizes = c10::SymDimVector(geometry.sizes.begin(), geometry.sizes.end());
  out.strides = c10::SymDimVector(geometry.strides.begin(), geometry.strides.end());
  out.dtype = a.dtype;
  out.device = a.device;
  out.cpu_scalar = a.cpu_scalar && out.sizes.empty();
  return out;
}

// refs._maybe_broadcast(*args, preserve_cpu_scalar_tensors=True)
std::vector<MetaDesc> maybe_broadcast(ArrayRef<MetaDesc> args) {
  std::vector<MetaDesc> out(args.begin(), args.end());
  std::vector<c10::SymIntArrayRef> shapes;
  for (const auto& arg : args) {
    if (!arg.is_number) {
      shapes.emplace_back(arg.sizes);
    }
  }
  if (shapes.empty()) {
    return out;
  }
  const auto common_shape = broadcast_shapes(shapes);
  // x.expand(common_shape) runs refs.expand only when x or the shape is
  // symbolic; otherwise it runs ATen's expand.
  const bool common_symbolic = any_heap(common_shape);
  for (auto& x : out) {
    if (x.is_number || x.cpu_scalar) {
      continue;
    }
    if (should_expand(x.sizes, common_shape)) {
      x = (x.symbolic || common_symbolic) ? expand_desc(x, common_shape) : aten_expand_desc(x, common_shape);
    }
  }
  return out;
}

// utils.get_computation_dtype
ScalarType get_computation_dtype(ScalarType dtype) {
  switch (dtype) {
    case ScalarType::BFloat16:
    case ScalarType::Half:
      return ScalarType::Float;
    case ScalarType::ComplexHalf:
      return ScalarType::ComplexFloat;
    default:
      return dtype;
  }
}

// utils.elementwise_dtypes -> (computation dtype, result dtype); wrapped
// numbers promote as Python numbers.
std::pair<ScalarType, ScalarType> elementwise_dtypes(const Tensor& a, const Tensor& b, TypePromotionKind kind) {
  auto result_dtype = result_type(update_result_type_state(b, update_result_type_state(a, ResultTypeState{})));
  if (kind == TypePromotionKind::INT_TO_FLOAT && isIntegralType(result_dtype, /*includeBool=*/true)) {
    result_dtype = c10::get_default_dtype_as_scalartype();
  }
  return {get_computation_dtype(result_dtype), result_dtype};
}

// prims.convert_element_type meta. A non-dense tensor gets
// compute_elementwise_output_strides(a), which for one tensor of rank >= 2 is
// torch.empty_like(a) (refs.empty_like: empty_permuted with the l2p perm).
MetaDesc convert_element_type_desc(const MetaDesc& a, ScalarType dtype) {
  MetaDesc out = a;
  out.dtype = dtype;
  if (!is_non_overlapping_and_dense_or_false(a)) {
    out.strides = a.dim() == 1 ? c10::SymDimVector{c10::SymInt(1)} : empty_permuted_desc(a.sizes, l2p_perm({&a}), dtype).strides;
  }
  out.symbolic = desc_is_symbolic(out);
  return out;
}

// _maybe_convert_to_dtype: tensors go through Tensor.to (the _to_copy
// decomposition), numbers through utils.dtype_to_type_ctor.
MetaDesc maybe_convert_desc(const MetaDesc& a, ScalarType dtype) {
  if (a.is_number) {
    MetaDesc out = a;
    if (dtype == kBool) {
      out.dtype = kBool;
    } else if (isIntegralType(dtype, /*includeBool=*/false)) {
      out.dtype = kLong;
    } else if (isComplexType(dtype)) {
      out.dtype = toComplexType(c10::get_default_dtype_as_scalartype());
    } else {
      out.dtype = c10::get_default_dtype_as_scalartype();
    }
    return out;
  }
  return a.dtype == dtype ? a : convert_element_type_desc(a, dtype);
}

// _prim_elementwise_meta over already broadcast args
MetaDesc prim_elementwise_desc(ArrayRef<MetaDesc> args, ScalarType dtype) {
  check_same_device(args);
  check_same_shape(args);
  const auto perm = l2p_perm(filter_tensors(args));

  // utils.extract_shape
  const MetaDesc* shape = nullptr;
  const MetaDesc* scalar_shape = nullptr;
  bool shape_mismatch = false;
  for (const auto& arg : args) {
    if (arg.is_number) {
      continue;
    }
    if (arg.cpu_scalar) {
      scalar_shape = &arg;
      continue;
    }
    if (shape == nullptr) {
      shape = &arg;
    }
    if (!is_same_shape(shape->sizes, arg.sizes)) {
      shape_mismatch = true;
      break;
    }
  }

  if (shape == nullptr && scalar_shape == nullptr) {
    MetaDesc out;
    out.dtype = dtype;
    out.is_number = true;
    return out;
  }
  TORCH_CHECK(!shape_mismatch, "shape must not be None when device is not None");
  const auto& like = shape != nullptr ? *shape : *scalar_shape;
  auto out = empty_permuted_desc(like.sizes, perm, dtype);
  out.device = like.device;
  out.cpu_scalar = shape == nullptr;
  return out;
}

// alpha != 1 in Python
bool python_ne_one(const Scalar& s) {
  if (s.isSymInt()) {
    return s.toSymInt().sym_ne(1).guard_bool(__FILE__, __LINE__);
  }
  if (s.isSymFloat()) {
    return s.toSymFloat().sym_ne(1.0).guard_bool(__FILE__, __LINE__);
  }
  if (s.isSymBool()) {
    // Python evaluates SymBool != 1 to True without guarding.
    return true;
  }
  return s.isComplex() ? s.toComplexDouble() != c10::complex<double>(1, 0) : s.toDouble() != 1;
}

// _make_elementwise_binary_reference / refs.add (alpha is None when unset) /
// refs.sub: elementwise_type_promotion_wrapper -> _maybe_broadcast ->
// [prims.mul(b, alpha)] -> prim -> conversion to the result dtype.
Tensor binary_ref_meta(
    const Tensor& self,
    const Tensor& other,
    TypePromotionKind kind,
    bool fake_devices,
    const std::optional<Scalar>& alpha = std::nullopt,
    bool is_sub = false) {
  const auto [compute_dtype, result_dtype] = elementwise_dtypes(self, other, kind);
  auto args = maybe_broadcast(
      {maybe_convert_desc(meta_desc(self, fake_devices), compute_dtype),
       maybe_convert_desc(meta_desc(other, fake_devices), compute_dtype)});
  if (is_sub) {
    TORCH_CHECK_NOT_IMPLEMENTED(
        args[0].is_number || args[1].is_number || (args[0].dtype != kBool && args[1].dtype != kBool),
        "Subtraction, the `-` operator, with two bool tensors is not supported. "
        "Use the `^` or `logical_xor()` operator instead.");
  }
  // refs.sub applies alpha when alpha != 1, after broadcasting (the check may
  // guard), and has no bool exemption in the type check below.
  if (alpha.has_value() && (!is_sub || python_ne_one(*alpha))) {
    // utils.is_weakly_lesser_type over bool < int < float < complex
    auto python_type_rank = [](ScalarType t) {
      return t == kBool ? 0 : isIntegralType(t, /*includeBool=*/false) ? 1 : isFloatingType(t) ? 2 : 3;
    };
    static constexpr std::array<const char*, 4> python_type_names = {
        "<class 'bool'>", "<class 'int'>", "<class 'float'>", "<class 'complex'>"};
    const auto rank = python_type_rank(compute_dtype);
    const auto alpha_rank = python_type_rank(alpha->type());
    TORCH_CHECK_VALUE(
        (rank == 0 && !is_sub) || alpha_rank <= rank,
        "alpha argument of type ", python_type_names[alpha_rank], " cannot be safely cast to type ",
        python_type_names[rank], "!");
    auto& b = args[1];
    if (!b.is_number) {
      MetaDesc alpha_desc;
      alpha_desc.dtype = alpha->type();
      alpha_desc.is_number = true;
      b = prim_elementwise_desc({b, alpha_desc}, b.dtype);
    }
  }
  const auto out = prim_elementwise_desc(args, compute_dtype);
  if (out.dtype == result_dtype) {
    // torch.empty_permuted: contiguous physical allocation, then restrided
    auto result = at::detail::empty_symint_meta(out.sizes, out.dtype, std::nullopt, kMeta, std::nullopt, std::nullopt);
    result.unsafeGetTensorImpl()->set_sizes_and_strides(out.sizes, out.strides);
    return Tensor(std::move(result));
  }
  const auto converted = convert_element_type_desc(out, result_dtype);
  return Tensor(at::detail::empty_strided_symint_meta(converted.sizes, converted.strides, result_dtype));
}

// fake_impls.infer_size. Unlike at::infer_size_symdimvector, it compares
// sizeA == sizeB in Python's operand order.
c10::SymDimVector fake_infer_size(c10::SymIntArrayRef a, c10::SymIntArrayRef b) {
  const auto dims_a = static_cast<int64_t>(a.size());
  const auto dims_b = static_cast<int64_t>(b.size());
  const auto ndim = std::max(dims_a, dims_b);
  c10::SymDimVector expanded_sizes(ndim);
  for (int64_t i = ndim - 1; i >= 0; --i) {
    const int64_t offset = ndim - 1 - i;
    const int64_t dim_a = dims_a - 1 - offset;
    const int64_t dim_b = dims_b - 1 - offset;
    const c10::SymInt size_a = dim_a >= 0 ? a[dim_a] : c10::SymInt(1);
    const c10::SymInt size_b = dim_b >= 0 ? b[dim_b] : c10::SymInt(1);
    if (!TORCH_GUARD_OR_FALSE(size_a.sym_eq(1)) && !TORCH_GUARD_OR_FALSE(size_b.sym_eq(1))) {
      TORCH_SYM_CHECK(
          py_eq(size_a, size_b),
          "The size of tensor a (", size_a, ") must match the size of tensor b (", size_b,
          ") at non-singleton dimension ", i);
    }
    expanded_sizes[i] = TORCH_GUARD_OR_FALSE(size_a.sym_eq(1)) ? size_b : size_a;
  }
  return expanded_sizes;
}

// fake_impls.make_fast_binary_impl, which Python fake tries first when the
// inputs are symbolic. Returns an undefined tensor where it falls back to the
// ref (the ref then raises for mismatched devices). The output device is left
// to the caller.
Tensor fast_binary_meta(const Tensor& self, const Tensor& other, TypePromotionKind kind) {
  const std::array<const Tensor*, 2> operands = {&self, &other};
  c10::SymDimVector final_shape(self.sym_sizes().begin(), self.sym_sizes().end());
  for (const auto* op : operands) {
    final_shape = fake_infer_size(final_shape, op->sym_sizes());
  }

  bool obvious = false;
  for (const auto* op : operands) {
    if (op->unsafeGetTensorImpl()->is_wrapped_number() || op->dim() != static_cast<int64_t>(final_shape.size())) {
      continue;
    }
    const auto sizes = op->sym_sizes();
    c10::SymBool eq(true);
    for (const auto i : c10::irange(final_shape.size())) {
      eq = eq.sym_and(py_eq(sizes[i], final_shape[i]));
    }
    if (TORCH_GUARD_OR_FALSE(eq)) {
      obvious = true;
      break;
    }
  }
  if (!obvious) {
    return {};
  }

  const bool self_number = self.unsafeGetTensorImpl()->is_wrapped_number();
  const bool other_number = other.unsafeGetTensorImpl()->is_wrapped_number();
  auto dtype = self.scalar_type();
  if (kind != TypePromotionKind::DEFAULT || self_number || other_number || other.scalar_type() != dtype) {
    dtype = elementwise_dtypes(self, other, kind).second;
  }

  c10::SmallVector<MetaDesc, 2> descs;
  for (const auto* op : operands) {
    if (!op->unsafeGetTensorImpl()->is_wrapped_number()) {
      descs.push_back(meta_desc(*op, /*fake_devices=*/true));
    }
  }
  // With two operands at most one can be the allowed CPU scalar.
  Device common_device = kCPU;
  for (const auto& desc : descs) {
    if (common_device.is_cpu() && !desc.device.is_cpu()) {
      common_device = desc.device;
    }
  }
  for (const auto& desc : descs) {
    const bool cpu_scalar_on_non_cpu = !common_device.is_cpu() && desc.dim() == 0 && desc.device == Device(kCPU);
    if (!cpu_scalar_on_non_cpu && desc.device != common_device) {
      return {};
    }
  }

  bool contiguous = true;
  bool channels_last = true;
  for (const auto& desc : descs) {
    contiguous = contiguous && is_contiguous_or_false(desc);
    channels_last = channels_last && is_channels_last_contiguous_or_false(desc);
  }
  if (!contiguous && !channels_last) {
    return {};
  }
  return at::native::empty_meta_symint(
      final_shape, dtype, std::nullopt, kMeta, std::nullopt,
      contiguous ? MemoryFormat::Contiguous : MemoryFormat::ChannelsLast);
}

} // namespace

// Mirrors what Python fake tensor runs for add.Tensor: the fast path for
// symbolic inputs, then refs.add (registered as the Meta kernel by
// activate_meta).
Tensor add_Tensor_meta(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  // FakeTensorMode's has_symbolic_sizes counts SymInt arguments only.
  // Symbolic wrapped numbers are not detected here: they carry a dummy value
  // and keep their SymInt only on the Python side.
  const bool symbolic = self.unsafeGetTensorImpl()->has_symbolic_sizes_strides() ||
      other.unsafeGetTensorImpl()->has_symbolic_sizes_strides() || alpha.isSymInt();
  if (symbolic) {
    if (auto out = fast_binary_meta(self, other, TypePromotionKind::DEFAULT); out.defined()) {
      return out;
    }
  }
  // A default alpha is dropped before reaching Python, so the ref sees None.
  const bool default_alpha = !alpha.isSymbolic() && alpha.type() == kLong && alpha.toLong() == 1;
  return binary_ref_meta(
      self, other, TypePromotionKind::DEFAULT, symbolic, default_alpha ? std::nullopt : std::optional<Scalar>(alpha));
}

// Mirrors what Python fake tensor runs for sub.Tensor: the fast path for
// symbolic inputs, then refs.sub.
Tensor sub_Tensor_meta(const Tensor& self, const Tensor& other, const Scalar& alpha) {
  // FakeTensorMode's has_symbolic_sizes counts SymInt arguments only.
  const bool symbolic = self.unsafeGetTensorImpl()->has_symbolic_sizes_strides() ||
      other.unsafeGetTensorImpl()->has_symbolic_sizes_strides() || alpha.isSymInt();
  if (symbolic) {
    if (auto out = fast_binary_meta(self, other, TypePromotionKind::DEFAULT); out.defined()) {
      return out;
    }
  }
  return binary_ref_meta(self, other, TypePromotionKind::DEFAULT, symbolic, alpha, /*is_sub=*/true);
}

// Mirrors what Python fake tensor runs for mul.Tensor: the fast path for
// symbolic inputs, then refs.mul.
Tensor mul_Tensor_meta(const Tensor& self, const Tensor& other) {
  const bool symbolic = self.unsafeGetTensorImpl()->has_symbolic_sizes_strides() ||
      other.unsafeGetTensorImpl()->has_symbolic_sizes_strides();
  if (symbolic) {
    if (auto out = fast_binary_meta(self, other, TypePromotionKind::DEFAULT); out.defined()) {
      return out;
    }
  }
  return binary_ref_meta(self, other, TypePromotionKind::DEFAULT, symbolic);
}

} // namespace at::native
