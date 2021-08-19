#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { struct TensorIterator; }

namespace at { namespace native {

inline void alpha_check(const ScalarType dtype, const Scalar& alpha) {
  TORCH_CHECK(! alpha.isBoolean() || dtype == ScalarType::Bool,
              "Boolean alpha only supported for Boolean results.");
  TORCH_CHECK(isFloatingType(dtype) || isComplexType(dtype)
              || alpha.isIntegral(true),
              "For integral input tensors, argument alpha must not be a floating point number.");
  TORCH_CHECK(isComplexType(dtype) || !alpha.isComplex(),
              "For non-complex input tensors, argument alpha must not be a complex number.")
}

// Basic checking for all sub functions.
inline void sub_check(const Tensor& self, const Tensor& other) {
  TORCH_CHECK(self.scalar_type() != kBool || other.scalar_type() != kBool,
              "Subtraction, the `-` operator, with two bool tensors is not supported. "
              "Use the `^` or `logical_xor()` operator instead.")
  TORCH_CHECK(self.scalar_type() != kBool && other.scalar_type() != kBool,
              "Subtraction, the `-` operator, with a bool tensor is not supported. "
              "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
}

inline void sub_check(const Tensor& self, const Scalar& scalar) {
  TORCH_CHECK(self.scalar_type() != kBool || !scalar.isBoolean(),
              "Subtraction, the `-` operator, with two bool tensors is not supported. "
              "Use the `^` or `logical_xor()` operator instead.")
  TORCH_CHECK(self.scalar_type() != kBool && !scalar.isBoolean(),
              "Subtraction, the `-` operator, with a bool tensor is not supported. "
              "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
}

using structured_binary_fn_alpha = void(*)(TensorIteratorBase&, const Scalar& alpha);
using structured_binary_fn = void(*)(TensorIteratorBase&);

using binary_fn_alpha = void(*)(TensorIteratorBase&, const Scalar& alpha);
using binary_fn_double = void(*)(TensorIterator&, double);
using binary_fn = void(*)(TensorIterator&);
using binary_clamp_fn_alpha =
    void(*)(TensorIterator&, const Scalar& alpha, const Scalar& min_val, const Scalar& max_val);

DECLARE_DISPATCH(structured_binary_fn_alpha, add_stub);
DECLARE_DISPATCH(binary_clamp_fn_alpha, add_clamp_stub);
DECLARE_DISPATCH(structured_binary_fn_alpha, sub_stub);
DECLARE_DISPATCH(structured_binary_fn, mul_stub);
DECLARE_DISPATCH(structured_binary_fn, div_true_stub);
DECLARE_DISPATCH(structured_binary_fn, div_floor_stub);
DECLARE_DISPATCH(structured_binary_fn, div_trunc_stub);
DECLARE_DISPATCH(structured_binary_fn, atan2_stub);
DECLARE_DISPATCH(structured_binary_fn, remainder_stub);
DECLARE_DISPATCH(structured_binary_fn, bitwise_and_stub);
DECLARE_DISPATCH(structured_binary_fn, bitwise_or_stub);
DECLARE_DISPATCH(structured_binary_fn, bitwise_xor_stub);
DECLARE_DISPATCH(structured_binary_fn, lshift_stub);
DECLARE_DISPATCH(structured_binary_fn, rshift_stub);
DECLARE_DISPATCH(binary_fn, logical_xor_stub);
DECLARE_DISPATCH(binary_fn, logical_and_stub);
DECLARE_DISPATCH(binary_fn, logical_or_stub);
DECLARE_DISPATCH(structured_binary_fn, lt_stub);
DECLARE_DISPATCH(structured_binary_fn, le_stub);
DECLARE_DISPATCH(structured_binary_fn, gt_stub);
DECLARE_DISPATCH(structured_binary_fn, ge_stub);
DECLARE_DISPATCH(structured_binary_fn, eq_stub);
DECLARE_DISPATCH(structured_binary_fn, ne_stub);
DECLARE_DISPATCH(binary_fn, max_elementwise_stub);
DECLARE_DISPATCH(binary_fn, min_elementwise_stub);
DECLARE_DISPATCH(structured_binary_fn, maximum_stub);
DECLARE_DISPATCH(structured_binary_fn, minimum_stub);
DECLARE_DISPATCH(structured_binary_fn, fmax_stub);
DECLARE_DISPATCH(structured_binary_fn, fmin_stub);
DECLARE_DISPATCH(binary_fn_double, smooth_l1_stub);
DECLARE_DISPATCH(binary_fn_double, huber_stub);
DECLARE_DISPATCH(structured_binary_fn, sigmoid_backward_stub);
DECLARE_DISPATCH(binary_fn_alpha, logit_backward_stub);
DECLARE_DISPATCH(structured_binary_fn, tanh_backward_stub);
DECLARE_DISPATCH(binary_fn, mse_stub);
DECLARE_DISPATCH(structured_binary_fn, fmod_stub);
DECLARE_DISPATCH(structured_binary_fn, logaddexp_stub);
DECLARE_DISPATCH(structured_binary_fn, logaddexp2_stub);
DECLARE_DISPATCH(structured_binary_fn, gcd_stub);
DECLARE_DISPATCH(structured_binary_fn, lcm_stub);
DECLARE_DISPATCH(structured_binary_fn, hypot_stub);
DECLARE_DISPATCH(structured_binary_fn, igamma_stub);
DECLARE_DISPATCH(structured_binary_fn, igammac_stub);
DECLARE_DISPATCH(structured_binary_fn, nextafter_stub);
DECLARE_DISPATCH(structured_binary_fn, heaviside_stub);
DECLARE_DISPATCH(structured_binary_fn, copysign_stub);
DECLARE_DISPATCH(structured_binary_fn, xlogy_stub);
DECLARE_DISPATCH(structured_binary_fn, xlog1py_stub);
DECLARE_DISPATCH(structured_binary_fn, zeta_stub);

}} // namespace at::native
