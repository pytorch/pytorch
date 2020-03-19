#pragma once

#include <ATen/ATen.h>
#include <ATen/native/DispatchStub.h>

namespace at { struct TensorIterator; }

namespace at { namespace native {

inline void alpha_check(const ScalarType dtype, Scalar alpha) {
  TORCH_CHECK(! alpha.isBoolean() || dtype == ScalarType::Bool,
              "Boolean alpha only supported for Boolean results.");
  TORCH_CHECK(isFloatingType(dtype) || alpha.isIntegral(true),
              "For integral input tensors, argument alpha must not be a floating point number.");
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

using binary_fn_alpha = void(*)(TensorIterator&, Scalar alpha);
using binary_fn = void(*)(TensorIterator&);

DECLARE_DISPATCH(binary_fn_alpha, add_stub);
DECLARE_DISPATCH(binary_fn_alpha, sub_stub);
DECLARE_DISPATCH(binary_fn, mul_stub);
DECLARE_DISPATCH(binary_fn, div_stub);
DECLARE_DISPATCH(binary_fn, remainder_stub);
DECLARE_DISPATCH(binary_fn, atan2_stub);
DECLARE_DISPATCH(binary_fn, bitwise_and_stub);
DECLARE_DISPATCH(binary_fn, bitwise_or_stub);
DECLARE_DISPATCH(binary_fn, bitwise_xor_stub);
DECLARE_DISPATCH(binary_fn, lshift_stub);
DECLARE_DISPATCH(binary_fn, rshift_stub);
DECLARE_DISPATCH(binary_fn, logical_xor_stub);
DECLARE_DISPATCH(binary_fn, logical_and_stub);
DECLARE_DISPATCH(binary_fn, logical_or_stub);
DECLARE_DISPATCH(binary_fn, lt_stub);
DECLARE_DISPATCH(binary_fn, le_stub);
DECLARE_DISPATCH(binary_fn, gt_stub);
DECLARE_DISPATCH(binary_fn, ge_stub);
DECLARE_DISPATCH(binary_fn, eq_stub);
DECLARE_DISPATCH(binary_fn, ne_stub);
DECLARE_DISPATCH(binary_fn, max_elementwise_stub);
DECLARE_DISPATCH(binary_fn, min_elementwise_stub);
DECLARE_DISPATCH(binary_fn, smooth_l1_stub);
DECLARE_DISPATCH(binary_fn, sigmoid_backward_stub);
DECLARE_DISPATCH(binary_fn, tanh_backward_stub);
DECLARE_DISPATCH(binary_fn, mse_stub);
DECLARE_DISPATCH(binary_fn, fmod_stub);
DECLARE_DISPATCH(binary_fn_alpha, fmod_scalar_stub);

}} // namespace at::native
