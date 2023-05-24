#pragma once

#include <ATen/core/TensorBase.h>
#include <ATen/native/DispatchStub.h>
#include <c10/core/Scalar.h>
#include <c10/util/TypeSafeSignMath.h>
#include <c10/util/copysign.h>


namespace at {
struct TensorIterator;
struct TensorIteratorBase;
}

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
inline void sub_check(const TensorBase& self, const TensorBase& other) {
  TORCH_CHECK(self.scalar_type() != kBool || other.scalar_type() != kBool,
              "Subtraction, the `-` operator, with two bool tensors is not supported. "
              "Use the `^` or `logical_xor()` operator instead.")
  TORCH_CHECK(self.scalar_type() != kBool && other.scalar_type() != kBool,
              "Subtraction, the `-` operator, with a bool tensor is not supported. "
              "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
}

inline void sub_check(const TensorBase& self, const Scalar& scalar) {
  TORCH_CHECK(self.scalar_type() != kBool || !scalar.isBoolean(),
              "Subtraction, the `-` operator, with two bool tensors is not supported. "
              "Use the `^` or `logical_xor()` operator instead.")
  TORCH_CHECK(self.scalar_type() != kBool && !scalar.isBoolean(),
              "Subtraction, the `-` operator, with a bool tensor is not supported. "
              "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
}

// NOTE: [Floor Division in Python]
// Python's __floordiv__ operator is more complicated than just floor(a / b).
// It aims to maintain the property: a == (a // b) * b + remainder(a, b)
// which can otherwise fail due to rounding errors in the remainder.
// So, instead it is calculated as: a // b = (a - remainder(a, b)) / b
// With some additional fix-ups added to the result.
//
// For reference, see CPython's implementation:
// https://github.com/python/cpython/blob/ace008c531dd685a30c1dd68f9b5ba35f20171cf/Objects/floatobject.c#L636

template <typename scalar_t>
inline scalar_t div_floor_floating(scalar_t a, scalar_t b) __ubsan_ignore_float_divide_by_zero__ {
  if (C10_UNLIKELY(b == 0)) {
    // Divide by zero: return standard IEEE result
    return a / b;
  }

  auto mod = std::fmod(a, b);
  auto div = (a - mod) / b;
  if ((mod != 0) && (b < 0) != (mod < 0)) {
    div -= scalar_t(1);
  }

  scalar_t floordiv;
  if (div != 0) {
    floordiv = std::floor(div);
    if (div - floordiv > scalar_t(0.5)) {
      floordiv += scalar_t(1.0);
    }
  } else {
    floordiv = c10::copysign(scalar_t(0), a / b);
  }
  return floordiv;
}

template <typename scalar_t>
inline scalar_t div_floor_integer(scalar_t a, scalar_t b) {
  if (c10::signs_differ(a, b)) {
    // Subtracts one from the results of truncation division if the
    // divisor and dividend have different sign(bit)s and the remainder of
    // the division is nonzero
    const auto quot = a / b;
    const auto rem = a % b;
    return rem ? quot - 1 : quot;
  }
  return a / b;
}

using structured_binary_fn_alpha = void(*)(TensorIteratorBase&, const Scalar& alpha);
using structured_binary_fn_double = void(*)(TensorIteratorBase&, double);
using structured_binary_fn = void(*)(TensorIteratorBase&);

using binary_fn_alpha = void(*)(TensorIteratorBase&, const Scalar& alpha);
using binary_fn_double = void(*)(TensorIterator&, double);
using binary_fn = void(*)(TensorIterator&);
using binary_clamp_fn_alpha =
    void(*)(TensorIterator&, const Scalar& alpha, const Scalar& min_val, const Scalar& max_val);

// NB: codegenned
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
DECLARE_DISPATCH(structured_binary_fn_double, smooth_l1_stub);
DECLARE_DISPATCH(binary_fn_double, huber_stub);
DECLARE_DISPATCH(structured_binary_fn, sigmoid_backward_stub);
DECLARE_DISPATCH(binary_fn_alpha, logit_backward_stub);
DECLARE_DISPATCH(structured_binary_fn, tanh_backward_stub);
DECLARE_DISPATCH(structured_binary_fn, mse_stub);
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
DECLARE_DISPATCH(structured_binary_fn, chebyshev_polynomial_t_stub);
DECLARE_DISPATCH(structured_binary_fn, chebyshev_polynomial_u_stub);
DECLARE_DISPATCH(structured_binary_fn, chebyshev_polynomial_v_stub);
DECLARE_DISPATCH(structured_binary_fn, chebyshev_polynomial_w_stub);
DECLARE_DISPATCH(structured_binary_fn, hermite_polynomial_h_stub);
DECLARE_DISPATCH(structured_binary_fn, hermite_polynomial_he_stub);
DECLARE_DISPATCH(structured_binary_fn, laguerre_polynomial_l_stub);
DECLARE_DISPATCH(structured_binary_fn, legendre_polynomial_p_stub);
DECLARE_DISPATCH(structured_binary_fn, shifted_chebyshev_polynomial_t_stub);
DECLARE_DISPATCH(structured_binary_fn, shifted_chebyshev_polynomial_u_stub);
DECLARE_DISPATCH(structured_binary_fn, shifted_chebyshev_polynomial_v_stub);
DECLARE_DISPATCH(structured_binary_fn, shifted_chebyshev_polynomial_w_stub);

}} // namespace at::native
