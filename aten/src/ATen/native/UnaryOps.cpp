#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/WrapDimUtils.h>

#include <ATen/native/Resize.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/ComplexHelper.h>

#include <c10/util/MathConstants.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_conj_native.h>
#include <ATen/ops/_conj_physical.h>
#include <ATen/ops/_conj_physical_native.h>
#include <ATen/ops/_neg_view_native.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/absolute_native.h>
#include <ATen/ops/acos.h>
#include <ATen/ops/acos_native.h>
#include <ATen/ops/acosh.h>
#include <ATen/ops/acosh_native.h>
#include <ATen/ops/angle.h>
#include <ATen/ops/angle_native.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/arccos_native.h>
#include <ATen/ops/arccosh_native.h>
#include <ATen/ops/arcsin_native.h>
#include <ATen/ops/arcsinh_native.h>
#include <ATen/ops/arctan_native.h>
#include <ATen/ops/arctanh_native.h>
#include <ATen/ops/asin.h>
#include <ATen/ops/asin_native.h>
#include <ATen/ops/asinh.h>
#include <ATen/ops/asinh_native.h>
#include <ATen/ops/atan.h>
#include <ATen/ops/atan_native.h>
#include <ATen/ops/atanh.h>
#include <ATen/ops/atanh_native.h>
#include <ATen/ops/bitwise_not_native.h>
#include <ATen/ops/can_cast.h>
#include <ATen/ops/ceil_native.h>
#include <ATen/ops/conj_native.h>
#include <ATen/ops/conj_physical.h>
#include <ATen/ops/conj_physical_native.h>
#include <ATen/ops/cos_native.h>
#include <ATen/ops/cosh_native.h>
#include <ATen/ops/deg2rad.h>
#include <ATen/ops/deg2rad_native.h>
#include <ATen/ops/digamma.h>
#include <ATen/ops/digamma_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/erf.h>
#include <ATen/ops/erf_native.h>
#include <ATen/ops/erfc.h>
#include <ATen/ops/erfc_native.h>
#include <ATen/ops/erfinv.h>
#include <ATen/ops/erfinv_native.h>
#include <ATen/ops/exp2.h>
#include <ATen/ops/exp2_native.h>
#include <ATen/ops/exp_native.h>
#include <ATen/ops/expm1.h>
#include <ATen/ops/expm1_native.h>
#include <ATen/ops/fix_native.h>
#include <ATen/ops/floor_native.h>
#include <ATen/ops/frac_native.h>
#include <ATen/ops/frexp.h>
#include <ATen/ops/frexp_native.h>
#include <ATen/ops/i0.h>
#include <ATen/ops/i0_native.h>
#include <ATen/ops/imag_native.h>
#include <ATen/ops/lgamma.h>
#include <ATen/ops/lgamma_native.h>
#include <ATen/ops/log10_native.h>
#include <ATen/ops/log1p.h>
#include <ATen/ops/log1p_native.h>
#include <ATen/ops/log2_native.h>
#include <ATen/ops/log_native.h>
#include <ATen/ops/logical_not.h>
#include <ATen/ops/logical_not_native.h>
#include <ATen/ops/logit.h>
#include <ATen/ops/logit_native.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/mvlgamma.h>
#include <ATen/ops/mvlgamma_native.h>
#include <ATen/ops/nan_to_num.h>
#include <ATen/ops/nan_to_num_native.h>
#include <ATen/ops/neg.h>
#include <ATen/ops/neg_native.h>
#include <ATen/ops/negative_native.h>
#include <ATen/ops/polygamma.h>
#include <ATen/ops/polygamma_native.h>
#include <ATen/ops/positive_native.h>
#include <ATen/ops/pow.h>
#include <ATen/ops/rad2deg.h>
#include <ATen/ops/rad2deg_native.h>
#include <ATen/ops/real.h>
#include <ATen/ops/real_native.h>
#include <ATen/ops/reciprocal_native.h>
#include <ATen/ops/resolve_conj_native.h>
#include <ATen/ops/resolve_neg_native.h>
#include <ATen/ops/round.h>
#include <ATen/ops/round_native.h>
#include <ATen/ops/rsqrt_native.h>
#include <ATen/ops/select.h>
#include <ATen/ops/sgn_native.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/sigmoid_native.h>
#include <ATen/ops/sign_native.h>
#include <ATen/ops/signbit_native.h>
#include <ATen/ops/sin_native.h>
#include <ATen/ops/sinc.h>
#include <ATen/ops/sinc_native.h>
#include <ATen/ops/sinh_native.h>
#include <ATen/ops/special_airy_ai_native.h>
#include <ATen/ops/special_bessel_j0_native.h>
#include <ATen/ops/special_bessel_j1_native.h>
#include <ATen/ops/special_bessel_y0_native.h>
#include <ATen/ops/special_bessel_y1_native.h>
#include <ATen/ops/special_digamma_native.h>
#include <ATen/ops/special_entr_native.h>
#include <ATen/ops/special_erf_native.h>
#include <ATen/ops/special_erfc_native.h>
#include <ATen/ops/special_erfcx_native.h>
#include <ATen/ops/special_erfinv_native.h>
#include <ATen/ops/special_exp2_native.h>
#include <ATen/ops/special_expit_native.h>
#include <ATen/ops/special_expm1_native.h>
#include <ATen/ops/special_gammaln_native.h>
#include <ATen/ops/special_i0_native.h>
#include <ATen/ops/special_i0e_native.h>
#include <ATen/ops/special_i1_native.h>
#include <ATen/ops/special_i1e_native.h>
#include <ATen/ops/special_log1p_native.h>
#include <ATen/ops/special_log_ndtr_native.h>
#include <ATen/ops/special_logit_native.h>
#include <ATen/ops/special_modified_bessel_i0_native.h>
#include <ATen/ops/special_modified_bessel_i1_native.h>
#include <ATen/ops/special_modified_bessel_k0_native.h>
#include <ATen/ops/special_modified_bessel_k1_native.h>
#include <ATen/ops/special_multigammaln_native.h>
#include <ATen/ops/special_ndtr_native.h>
#include <ATen/ops/special_ndtri_native.h>
#include <ATen/ops/special_polygamma_native.h>
#include <ATen/ops/special_psi_native.h>
#include <ATen/ops/special_round_native.h>
#include <ATen/ops/special_scaled_modified_bessel_k0_native.h>
#include <ATen/ops/special_scaled_modified_bessel_k1_native.h>
#include <ATen/ops/special_sinc_native.h>
#include <ATen/ops/special_spherical_bessel_j0_native.h>
#include <ATen/ops/sqrt_native.h>
#include <ATen/ops/square_native.h>
#include <ATen/ops/tan_native.h>
#include <ATen/ops/tanh_native.h>
#include <ATen/ops/trunc.h>
#include <ATen/ops/trunc_native.h>
#include <ATen/ops/view_as_real.h>
#endif

#include <cmath>

namespace at::meta {

// Unary float operations always produce floating point
// outputs for floating point and integral types
// For complex inputs, the output type should be the same as input type.
#define CREATE_UNARY_FLOAT_META_FUNC(func)                  \
  TORCH_META_FUNC(func) (const Tensor& self) {        \
    build_borrowing_unary_float_op(maybe_get_output(), self);   \
  }

CREATE_UNARY_FLOAT_META_FUNC(acos)
CREATE_UNARY_FLOAT_META_FUNC(acosh)
CREATE_UNARY_FLOAT_META_FUNC(asin)
CREATE_UNARY_FLOAT_META_FUNC(asinh)
CREATE_UNARY_FLOAT_META_FUNC(atan)
CREATE_UNARY_FLOAT_META_FUNC(atanh)
CREATE_UNARY_FLOAT_META_FUNC(cos)
CREATE_UNARY_FLOAT_META_FUNC(cosh)
CREATE_UNARY_FLOAT_META_FUNC(digamma)
CREATE_UNARY_FLOAT_META_FUNC(erf)
CREATE_UNARY_FLOAT_META_FUNC(erfc)
CREATE_UNARY_FLOAT_META_FUNC(erfinv)
CREATE_UNARY_FLOAT_META_FUNC(exp)
CREATE_UNARY_FLOAT_META_FUNC(exp2)
CREATE_UNARY_FLOAT_META_FUNC(expm1)
CREATE_UNARY_FLOAT_META_FUNC(i0)
CREATE_UNARY_FLOAT_META_FUNC(lgamma)
CREATE_UNARY_FLOAT_META_FUNC(log)
CREATE_UNARY_FLOAT_META_FUNC(log10)
CREATE_UNARY_FLOAT_META_FUNC(log1p)
CREATE_UNARY_FLOAT_META_FUNC(log2)
CREATE_UNARY_FLOAT_META_FUNC(reciprocal)
CREATE_UNARY_FLOAT_META_FUNC(rsqrt)
CREATE_UNARY_FLOAT_META_FUNC(sigmoid)
CREATE_UNARY_FLOAT_META_FUNC(sin)
CREATE_UNARY_FLOAT_META_FUNC(sinc)
CREATE_UNARY_FLOAT_META_FUNC(sinh)
CREATE_UNARY_FLOAT_META_FUNC(special_entr)
CREATE_UNARY_FLOAT_META_FUNC(special_erfcx)
CREATE_UNARY_FLOAT_META_FUNC(special_i0e)
CREATE_UNARY_FLOAT_META_FUNC(special_i1)
CREATE_UNARY_FLOAT_META_FUNC(special_i1e)
CREATE_UNARY_FLOAT_META_FUNC(special_ndtri)
CREATE_UNARY_FLOAT_META_FUNC(special_log_ndtr)
CREATE_UNARY_FLOAT_META_FUNC(sqrt)
CREATE_UNARY_FLOAT_META_FUNC(tan)
CREATE_UNARY_FLOAT_META_FUNC(tanh)
CREATE_UNARY_FLOAT_META_FUNC(special_airy_ai)
CREATE_UNARY_FLOAT_META_FUNC(special_bessel_j0)
CREATE_UNARY_FLOAT_META_FUNC(special_bessel_j1)
CREATE_UNARY_FLOAT_META_FUNC(special_bessel_y0)
CREATE_UNARY_FLOAT_META_FUNC(special_bessel_y1)
CREATE_UNARY_FLOAT_META_FUNC(special_modified_bessel_i0)
CREATE_UNARY_FLOAT_META_FUNC(special_modified_bessel_i1)
CREATE_UNARY_FLOAT_META_FUNC(special_modified_bessel_k0)
CREATE_UNARY_FLOAT_META_FUNC(special_modified_bessel_k1)
CREATE_UNARY_FLOAT_META_FUNC(special_scaled_modified_bessel_k0)
CREATE_UNARY_FLOAT_META_FUNC(special_scaled_modified_bessel_k1)
CREATE_UNARY_FLOAT_META_FUNC(special_spherical_bessel_j0)

TORCH_META_FUNC(polygamma)(int64_t n, const Tensor& self) {
  TORCH_CHECK(n >= 0, "polygamma(n, x) does not support negative n.");
  build_borrowing_unary_float_op(maybe_get_output(), self);
}

// These are normal unary ops that preserve dtype
#define CREATE_UNARY_META_FUNC(func)                  \
  TORCH_META_FUNC(func) (const Tensor& self) {        \
    build_borrowing_unary_op(maybe_get_output(), self);   \
  }
CREATE_UNARY_META_FUNC(bitwise_not)
CREATE_UNARY_META_FUNC(frac)
CREATE_UNARY_META_FUNC(round)
CREATE_UNARY_META_FUNC(sgn)

TORCH_META_FUNC2(round, decimals)(const Tensor& self, int64_t decimals){
  build_unary_op(maybe_get_output(), self);
}

TORCH_META_FUNC(neg)(const Tensor& self) {
  TORCH_CHECK(self.scalar_type() != kBool,
              "Negation, the `-` operator, on a bool tensor is not supported. "
              "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
  build_borrowing_unary_op(maybe_get_output(), self);
}

TORCH_META_FUNC(trunc) (const Tensor& self) {
  // Note: this is consistent with NumPy
  TORCH_CHECK(!self.is_complex(),
    "trunc is not supported for complex inputs");
  build_borrowing_unary_op(maybe_get_output(), self);
}

TORCH_META_FUNC(floor) (const Tensor& self) {
  // Note: this is consistent with NumPy
  TORCH_CHECK(!self.is_complex(),
    "floor is not supported for complex inputs");
  build_borrowing_unary_op(maybe_get_output(), self);
}

TORCH_META_FUNC(sign) (const Tensor& self) {
  TORCH_CHECK(!self.is_complex(),
              "Unlike NumPy, torch.sign is not intended to support complex numbers. Please use torch.sgn instead.");
  build_borrowing_unary_op(maybe_get_output(), self);
}

TORCH_META_FUNC(signbit) (const Tensor& self) {
  TORCH_CHECK(!self.is_complex(), "signbit is not implemented for complex tensors.");
  TORCH_CHECK(maybe_get_output().defined() ? maybe_get_output().dtype() == at::kBool : true,
              "signbit does not support non-boolean outputs.");
  build_borrowing_unary_force_boolean_op(maybe_get_output(), self);
}

TORCH_META_FUNC(ceil) (const Tensor& self) {
  // Note: this is consistent with NumPy
  TORCH_CHECK(!self.is_complex(),
    "ceil is not supported for complex inputs");
  build_borrowing_unary_op(maybe_get_output(), self);
}

} // namespace at::meta

namespace at::native {
// NOTE: These are helper functions that reduce redundant code in implementing the most typical kind of unary operators.
// YOU ARE NOT OBLIGED TO USE THESE HELPERS---if you're writing something more specialized, please don't try to make
// them work for your case, but just write something new instead. Here we use helper functions instead of a flat fat
// macro that implements everything, because the former allows some simple preprocessing that are unique to some
// operators (more is foreseeable) and is more flexible and elegant than the latter.
#define CREATE_UNARY_TORCH_IMPL_FUNC(func_out, func_stub)                                \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const Tensor& result) {  \
  func_stub(device_type(), *this);                                      \
}

// This macro is as optional as the one above. torch.(ceil|floor|round|trunc) are no-ops for integers
// See gh-70918
#define CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(func_out, func_stub)                                \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const Tensor& result) {  \
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/false)) {                                      \
    result.copy_(self);                                                 \
  } else {                                                              \
    func_stub(device_type(), *this);                                    \
  }                                                                     \
}
CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(ceil_out, ceil_stub)
CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(floor_out, floor_stub)
CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(round_out, round_stub)
CREATE_UNARY_TORCH_IMPL_INTEGER_NO_OP_FUNC(trunc_out, trunc_stub)

CREATE_UNARY_TORCH_IMPL_FUNC(acos_out, acos_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(acosh_out, acosh_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(asin_out, asin_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(asinh_out, asinh_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(atan_out, atan_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(atanh_out, atanh_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(bitwise_not_out, bitwise_not_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(cos_out, cos_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(cosh_out, cosh_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(digamma_out, digamma_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(erf_out, erf_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(erfc_out, erfc_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(erfinv_out, erfinv_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(exp_out, exp_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(exp2_out, exp2_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(expm1_out, expm1_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(frac_out, frac_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(i0_out, i0_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(lgamma_out, lgamma_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(log_out, log_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(log10_out, log10_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(log1p_out, log1p_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(log2_out, log2_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(neg_out, neg_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(reciprocal_out, reciprocal_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(rsqrt_out, rsqrt_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(sigmoid_out, sigmoid_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(sign_out, sign_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(sin_out, sin_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(sinc_out, sinc_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(sinh_out, sinh_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_entr_out, special_entr_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_erfcx_out, special_erfcx_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_i0e_out, special_i0e_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_i1e_out, special_i1e_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_i1_out, special_i1_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_ndtri_out, special_ndtri_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_log_ndtr_out, special_log_ndtr_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(sqrt_out, sqrt_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(tan_out, tan_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(tanh_out, tanh_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_airy_ai_out, special_airy_ai_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_bessel_j0_out, special_bessel_j0_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_bessel_j1_out, special_bessel_j1_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_bessel_y0_out, special_bessel_y0_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_bessel_y1_out, special_bessel_y1_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_modified_bessel_i0_out, special_modified_bessel_i0_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_modified_bessel_i1_out, special_modified_bessel_i1_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_modified_bessel_k0_out, special_modified_bessel_k0_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_modified_bessel_k1_out, special_modified_bessel_k1_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_scaled_modified_bessel_k0_out, special_scaled_modified_bessel_k0_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_scaled_modified_bessel_k1_out, special_scaled_modified_bessel_k1_stub)
CREATE_UNARY_TORCH_IMPL_FUNC(special_spherical_bessel_j0_out, special_spherical_bessel_j0_stub)

TORCH_IMPL_FUNC(round_decimals_out)
(const Tensor& self, int64_t decimals, const Tensor& result) {
  if (decimals != 0) {
    round_decimals_stub(device_type(), *this, decimals);
  } else {
    round_stub(device_type(), *this);
  }
}

TORCH_IMPL_FUNC(polygamma_out)
(int64_t n, const Tensor& self, const Tensor& result) {
  polygamma_stub(device_type(), *this, n);
}

TORCH_IMPL_FUNC(signbit_out) (const Tensor& self, const Tensor& result) {
  if (self.dtype() == at::kBool) {
    result.fill_(false);
  } else {
    signbit_stub(device_type(), *this);
  }
}

// since polygamma_ has different signature from its
// out and functional variant, we explicitly
// define it (instead of using structured kernel).
Tensor& polygamma_(Tensor& self, int64_t n) {
  return at::polygamma_out(self, n, self);
}

template <typename Stub>
static inline Tensor& unary_op_impl_out(Tensor& result, const Tensor& self, Stub& stub) {
  auto iter = TensorIterator::unary_op(result, self);
  stub(iter.device_type(), iter);
  return result;
}

template <typename Stub, typename ...Args>
static inline Tensor& unary_op_impl_float_out(Tensor& result, const Tensor& self, Stub& stub, Args... args) {
  auto iter = TensorIterator::unary_float_op(result, self);
  stub(iter.device_type(), iter, args...);
  return result;
}

template <typename Stub, typename ...Args>
static inline Tensor unary_op_impl_float(const Tensor& self, Stub& stub, Args... args) {
  Tensor result;
  auto iter = TensorIterator::unary_float_op(result, self);
  stub(iter.device_type(), iter, args...);
  return iter.output();
}

// An alternate version of unary_op_impl_out that follows the same pattern
// for non-complex inputs, but returns a floating point tensor
// for complex inputs by default.
// Note: This is done by running the operation as usual and then copying the
// operation's result to the expected result type.
template <typename Stub>
static inline Tensor& unary_op_impl_with_complex_to_float_out(Tensor& result, const Tensor& self, Stub& stub, bool promotes_integer_to_float) {
    if (self.is_complex() && !result.is_complex()) {
      // Checks if the corresponding float type can be cast to the desired dtype
      const auto float_type = c10::toRealValueType(self.scalar_type());
      TORCH_CHECK(canCast(float_type, result.scalar_type()),
            "result type ", float_type, " can't be cast to the desired output type ",
            result.scalar_type());

      // Runs the function complex->complex, as TensorIterator expects
      Tensor complex_result = at::empty({0}, self.options());
      auto iter = TensorIterator::unary_op(complex_result, self);
      stub(iter.device_type(), iter);

      // Copies the complex result to the actual result and returns it
      at::native::resize_output(result, complex_result.sizes());
      result.copy_(at::real(complex_result));
      return result;
    }

    if (promotes_integer_to_float) {
      return unary_op_impl_float_out(result, self, stub);
    }

    return unary_op_impl_out(result, self, stub);
}

// out_impl passed into unary_op_impl and unary_op_impl_  must go through at:: device dispatch
// otherwise it won't dispatch to out-of-source devices like XLA.
// For example it must be at::bitwise_not_out instead of bitwise_not_out(which is at::native!).
template <typename OutImpl>
static inline Tensor unary_op_impl(const Tensor& self, OutImpl& out_impl) {
  Tensor result = at::empty({0}, self.options());
  return out_impl(result, self);
}

// An alternate version of unary_op_impl that follows the same pattern
// for non-complex inputs, but returns a floating point tensor
// for complex inputs by default.
template <typename OutImpl>
static inline Tensor unary_op_impl_with_complex_to_float(const Tensor& self, OutImpl& out_impl) {
  if (self.is_complex()) {
    const auto float_type = c10::toRealValueType(self.scalar_type());
    Tensor result = at::empty_like(self, self.options().dtype(float_type));
    return out_impl(result, self);
  }

  Tensor result = at::empty({0}, self.options());
  return out_impl(result, self);
}

template <typename OutImpl>
static inline Tensor& unary_op_impl_(Tensor& self, OutImpl& out_impl) {
  return out_impl(self, self);
}

// arccos, alias for acos
Tensor& arccos_out(const Tensor& self, Tensor& result) { return at::acos_out(result, self); }
Tensor arccos(const Tensor& self) { return self.acos(); }
Tensor& arccos_(Tensor& self) { return self.acos_(); }

Tensor& rad2deg_out(const Tensor& self, Tensor& result) {
  TORCH_CHECK(!self.is_complex(), "rad2deg is not supported for complex tensors.");
  constexpr double M_180_PI = 57.295779513082320876798154814105170332405472466564;
  return at::mul_out(result, self, wrapped_scalar_tensor(Scalar(M_180_PI)));
}
Tensor rad2deg(const Tensor& self) {
  // Note: int-> float promotion handled differently from other Unary ops,
  // as it does not use the usual TensorIterator + Kernel Dispatch pattern.
  auto options = self.options();
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    options = options.dtype(c10::get_default_dtype());
  }
  auto result = at::empty_like(self, options);
  at::rad2deg_out(result, self);
  return result;
}
Tensor& rad2deg_(Tensor& self) { return unary_op_impl_(self, at::rad2deg_out); }

Tensor& deg2rad_out(const Tensor& self, Tensor& result) {
  TORCH_CHECK(!self.is_complex(), "deg2rad is not supported for complex tensors.");
  constexpr double M_PI_180 = 0.017453292519943295769236907684886127134428718885417;
  return at::mul_out(result, self, wrapped_scalar_tensor(Scalar(M_PI_180)));
}
Tensor deg2rad(const Tensor& self) {
  // Note: int-> float promotion handled differently from other Unary ops,
  // as it does not use the usual TensorIterator + Kernel Dispatch pattern.
  auto options = self.options();
  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    options = options.dtype(c10::get_default_dtype());
  }
  auto result = at::empty_like(self, options);
  at::deg2rad_out(result, self);
  return result;
}
Tensor& deg2rad_(Tensor& self) { return unary_op_impl_(self, at::deg2rad_out); }

// arcsin, alias of asin
Tensor& arcsin_out(const Tensor& self, Tensor& result) { return at::asin_out(result, self); }
Tensor arcsin(const Tensor& self) { return self.asin(); }
Tensor& arcsin_(Tensor& self) { return self.asin_(); }

// arctan, alias of atan
Tensor& arctan_out(const Tensor& self, Tensor& result) { return at::atan_out(result, self); }
Tensor arctan(const Tensor& self) { return self.atan(); }
Tensor& arctan_(Tensor& self) { return self.atan_(); }

// Note [Complex abs and angle]
// Complex inputs to abs and angle return float results by default.
// abs and angle, in both NumPy and C++, returns a float result when given a
// complex input. This makes sense mathematically since the absolute value
// and angle of a complex number has no imaginary part.
Tensor& abs_out(const Tensor& self, Tensor& result) {
  return unary_op_impl_with_complex_to_float_out(result, self, abs_stub, /*promotes_integer_to_float=*/false);
}
Tensor abs(const Tensor& self) {
  return unary_op_impl_with_complex_to_float(self, at::abs_out);
}
Tensor& abs_(Tensor& self) {
  TORCH_CHECK(!self.is_complex(), "In-place abs is not supported for complex tensors.");
  return unary_op_impl_(self, at::abs_out);
}

// Absolute, alias for abs
Tensor& absolute_out(const Tensor& self, Tensor& result) {
  return at::abs_out(result, self);
}
Tensor absolute(const Tensor& self) {
  return self.abs();
}
Tensor& absolute_(Tensor& self) {
  return self.abs_();
}

Tensor& angle_out(const Tensor& self, Tensor& result) {
  return unary_op_impl_with_complex_to_float_out(result, self, angle_stub, /*promotes_integer_to_float=*/true);
}
Tensor angle(const Tensor& self) {
  if (self.is_complex()) {
    const auto float_type = c10::toRealValueType(self.scalar_type());
    Tensor result = at::empty({0}, self.options().dtype(float_type));
    return at::angle_out(result, self);
  }

  return unary_op_impl_float(self, angle_stub);
}

Tensor real(const Tensor& self) {
  if (self.is_complex()) {
    Tensor real_tensor;
    if (self.is_conj()) {
      real_tensor = at::view_as_real(self._conj());
    } else {
      real_tensor = at::view_as_real(self);
    }
    return at::select(real_tensor, real_tensor.dim() - 1, 0);
  } else {
    return self;
  }
}

Tensor _neg_view(const Tensor& self) {
  Tensor self_ = self.alias();
  self_._set_neg(!self.is_neg());
  namedinference::propagate_names(self_, self);
  return self_;
}

Tensor imag(const Tensor& self) {
  if (self.is_complex()) {
    Tensor real_tensor;
    if (self.is_conj()) {
      real_tensor = at::view_as_real(self._conj());
      // preemptively set the negative flag for the final imag tensor
      real_tensor = real_tensor._neg_view();
    } else {
      real_tensor = at::view_as_real(self);
    }
    return at::select(real_tensor, real_tensor.dim() - 1, 1);
  } else {
    TORCH_CHECK(false, "imag is not implemented for tensors with non-complex dtypes.");
  }
}

Tensor& conj_physical_out(const Tensor& self, Tensor& result) {
  return unary_op_impl_out(result, self, conj_physical_stub);
}

Tensor _conj_physical(const Tensor& self) {
  if (self.is_conj()) {
    return self.conj().clone();
  }
  return unary_op_impl(self, at::conj_physical_out);
}

Tensor conj_physical(const Tensor& self) {
  if (!self.is_complex()) return self;
  return at::_conj_physical(self);
}

Tensor& conj_physical_(Tensor& self) {
  if (!self.is_complex()) return self;
  return unary_op_impl_out(self, self, conj_physical_stub);
}

// No op if the neg bit is not set
// else returns a new negated tensor with neg bit set to 0
Tensor resolve_neg(const Tensor& self) {
  if (!self.is_neg()) { return self; }
  // negation is materialized in `copy_()` that clone ultimately calls into
  return self.clone();
}

// No op if the conj bit is not set
// else returns a new negated tensor with neg bit set to 0
Tensor resolve_conj(const Tensor& self) {
  if (!self.is_conj()) { return self; }
  // conjugation is materialized in `copy_()` that clone ultimately calls into
  return self.clone();
}

Tensor _conj(const Tensor& self) {
  Tensor self_ = self.alias();
  self_._set_conj(!self.is_conj());
  namedinference::propagate_names(self_, self);
  return self_;
}

Tensor conj(const Tensor& self) {
  // This might look like an infinite recursion but it's not.
  // This actually calls into `conj()` defined in the Tensor class.
  return self.conj();
}

// special_exp2, alias for exp2
Tensor& special_exp2_out(const Tensor& self, Tensor& result) { return at::exp2_out(result, self); }
Tensor special_exp2(const Tensor& self) { return self.exp2(); }

// special_expm1, alias for expm1
Tensor& special_expm1_out(const Tensor& self, Tensor& result) { return at::expm1_out(result, self); }
Tensor special_expm1(const Tensor& self) { return self.expm1(); }

// special_erf, alias for erf
Tensor& special_erf_out(const Tensor& self, Tensor& result) { return at::erf_out(result, self); }
Tensor special_erf(const Tensor& self) { return self.erf(); }

// special_erfc, alias for erfc
Tensor& special_erfc_out(const Tensor& self, Tensor& result) { return at::erfc_out(result, self); }
Tensor special_erfc(const Tensor& self) { return self.erfc(); }

// special_erfinv, alias for erfinv
Tensor& special_erfinv_out(const Tensor& self, Tensor& result) { return at::erfinv_out(result, self); }
Tensor special_erfinv(const Tensor& self) { return self.erfinv(); }

// special_polygamma, alias for polygamma
Tensor& special_polygamma_out(int64_t n, const Tensor& self, Tensor& result) { return at::polygamma_out(result, n, self); }
Tensor special_polygamma(int64_t n, const Tensor& self) { return self.polygamma(n); }

// special_psi, alias for digamma
Tensor& special_psi_out(const Tensor& self, Tensor& result) { return at::digamma_out(result, self); }
Tensor special_psi(const Tensor& self) { return self.digamma(); }
// special_digamma, alias for digamma
Tensor& special_digamma_out(const Tensor& self, Tensor& result) { return at::digamma_out(result, self); }
Tensor special_digamma(const Tensor& self) { return self.digamma(); }

// special_i0, alias for i0
Tensor& special_i0_out(const Tensor& self, Tensor& result) { return at::i0_out(result, self); }
Tensor special_i0(const Tensor& self) { return self.i0(); }

// special_log1p, alias for log1p
Tensor& special_log1p_out(const Tensor& self, Tensor& result) { return at::log1p_out(result, self); }
Tensor special_log1p(const Tensor& self) { return self.log1p(); }

// special_round, alias for round
Tensor& special_round_out(const Tensor& self, int64_t decimals, Tensor& result) { return at::round_out(result, self, decimals); }
Tensor special_round(const Tensor& self, int64_t decimals) { return self.round(decimals); }

// special_sinc, alias for sinc
Tensor& special_sinc_out(const Tensor& self, Tensor& result) { return at::sinc_out(result, self); }
Tensor special_sinc(const Tensor& self) { return self.sinc(); }

namespace {

inline Tensor calc_ndtr(const Tensor& self) {
  auto x_sqrt_2 = self * M_SQRT1_2;
  return (1 + at::erf(x_sqrt_2)) * 0.5;
}

} // namespace

// special_ndtr
Tensor& special_ndtr_out(const Tensor& self, Tensor& result) {
  TORCH_CHECK(
      self.device() == result.device(),
      "Expected all tensors to be on the same device, but found at least two devices, ",
      self.device(),
      " and ",
      result.device(),
      "!");

  auto ndtr = calc_ndtr(self);
  TORCH_CHECK(
      at::can_cast(ndtr.scalar_type(), result.scalar_type()),
      "result type ",
      ndtr.scalar_type(),
      " can't be cast to the desired output type ",
      result.scalar_type());

  at::native::resize_output(result, ndtr.sizes());
  return result.copy_(ndtr);
}
Tensor special_ndtr(const Tensor& self) {
  return calc_ndtr(self);
}

// FIXME: remove const_cast once unary_op_impl_out is updated
TORCH_IMPL_FUNC(sgn_out) (const Tensor& self, const Tensor& result) {
  if (self.is_complex()) {
    sgn_stub(device_type(), *this);
  } else {
    sign_stub(device_type(), *this);
  }
}

// arccosh, alias for acosh
Tensor& arccosh_out(const Tensor& self, Tensor& result) { return at::acosh_out(result, self); }
Tensor arccosh(const Tensor& self) { return at::acosh(self); }
Tensor& arccosh_(Tensor& self) { return at::acosh_(self); }

// arcsinh, alias for asinh
Tensor& arcsinh_out(const Tensor& self, Tensor& result) { return at::asinh_out(result, self); }
Tensor arcsinh(const Tensor& self) { return self.asinh(); }
Tensor& arcsinh_(Tensor& self) { return self.asinh_(); }

// arctanh, alias for atanh
Tensor& arctanh_out(const Tensor& self, Tensor& result) { return at::atanh_out(result, self); }
Tensor arctanh(const Tensor& self) { return self.atanh(); }
Tensor& arctanh_(Tensor& self) { return self.atanh_(); }

Tensor& square_out(const Tensor& self, Tensor& result) { return at::pow_out(result, self, 2); }
Tensor square(const Tensor& self) { return at::pow(self, 2); }
Tensor& square_(Tensor& self) { return self.pow_(2); }

Tensor& logit_out(const Tensor& self,
    std::optional<double> eps,
    Tensor& result) {
  return unary_op_impl_float_out(
      result, self, logit_stub, Scalar(eps ? eps.value() : -1.0));
}
Tensor logit(const Tensor& self, std::optional<double> eps) {
  return unary_op_impl_float(
      self, logit_stub, Scalar(eps ? eps.value() : -1.0));
}
Tensor& logit_(Tensor& self, std::optional<double> eps) {
  return at::logit_out(self, self, eps);
}

Tensor& special_logit_out(const Tensor& self, std::optional<double> eps, Tensor& result) {
  return at::logit_out(result, self, eps);
}
Tensor special_logit(const Tensor& self, std::optional<double> eps) {
  return self.logit(eps);
}

// special_expit, alias for sigmoid
Tensor& special_expit_out(const Tensor& self, Tensor& result) {
  return at::sigmoid_out(result, self);
}
Tensor special_expit(const Tensor& self) {
  return self.sigmoid();
}

Tensor& nan_to_num_out(const Tensor& self,
    std::optional<double> nan,
    std::optional<double> pos_inf,
    std::optional<double> neg_inf,
    Tensor& result) {
  TORCH_CHECK(
      self.scalar_type() == result.scalar_type(),
      "nan_to_num: dtype of out: ",
      result.scalar_type(),
      " should be same as input: ",
      self.scalar_type());

  if (c10::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    at::native::resize_output(result, self.sizes());
    result.copy_(self);
    return result;
  }

  auto iter = TensorIterator::unary_op(result, self);
  nan_to_num_stub(iter.device_type(), iter, nan, pos_inf, neg_inf);
  return result;
}

Tensor nan_to_num(
    const Tensor& self,
    std::optional<double> nan,
    std::optional<double> pos_inf,
    std::optional<double> neg_inf) {
  auto result = at::empty_like(self);
  return at::nan_to_num_out(result, self, nan, pos_inf, neg_inf);
}

Tensor& nan_to_num_(
    Tensor& self,
    std::optional<double> nan,
    std::optional<double> pos_inf,
    std::optional<double> neg_inf) {
  return at::nan_to_num_out(self, self, nan, pos_inf, neg_inf);
}

// Alias for trunc
Tensor& fix_out(const Tensor& self, Tensor& result) { return at::trunc_out(result, self); }
Tensor fix(const Tensor& self) { return self.trunc(); }
Tensor& fix_(Tensor& self) { return self.trunc_(); }

Tensor positive(const Tensor& self) {
  TORCH_CHECK(self.scalar_type() != kBool, "The `+` operator, on a bool tensor is not supported.");
  return self;
}

Tensor& negative_out(const Tensor& self, Tensor& result) { return at::neg_out(result, self); }
Tensor negative(const Tensor& self) { return self.neg(); }
Tensor& negative_(Tensor& self) { return self.neg_(); }

Tensor logical_not(const Tensor& self) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::logical_not_out(result, self);
}

Tensor& logical_not_(Tensor& self) {
  return at::logical_not_out(self, self);
}

Tensor& logical_not_out(const Tensor& self, Tensor& result) {
  TensorIterator iter = TensorIteratorConfig()
    .check_all_same_dtype(false)
    .add_output(result)
    .add_const_input(self)
    .build();
  logical_not_stub(iter.device_type(), iter);
  return result;
}

namespace {
constexpr double HALF = 0.5;
constexpr double QUARTER = 0.25;
}

static inline void mvlgamma_check(const Tensor& self, int64_t p) {
  TORCH_CHECK(self.scalar_type() != kBool, "The input tensor may not be a boolean tensor.");
  TORCH_CHECK(p >= 1, "p has to be greater than or equal to 1");
}

Tensor mvlgamma(const Tensor& self, int64_t p) {
  mvlgamma_check(self, p);
  auto dtype = c10::scalarTypeToTypeMeta(self.scalar_type());
  if (at::isIntegralType(self.scalar_type(), /*include_bool=*/true)) {
    // int -> float promotion
    dtype = c10::get_default_dtype();
  }
  Tensor args = native::arange(
      -p * HALF + HALF,
      HALF,
      HALF,
      optTypeMetaToScalarType(dtype),
      self.options().layout_opt(),
      self.options().device_opt(),
      self.options().pinned_memory_opt());
  args = args.add(self.unsqueeze(-1));
  const auto p2_sub_p = static_cast<double>(p * (p - 1));
  return args.lgamma_().sum(-1).add_(p2_sub_p * std::log(c10::pi<double>) * QUARTER);
}

Tensor& mvlgamma_(Tensor& self, int64_t p) {
  mvlgamma_check(self, p);
  Tensor args = native::arange(
      -p *HALF  + HALF,
      HALF,
      HALF,
      optTypeMetaToScalarType(self.options().dtype_opt()),
      self.options().layout_opt(),
      self.options().device_opt(),
      self.options().pinned_memory_opt());
  args = args.add(self.unsqueeze(-1));
  const auto p2_sub_p = static_cast<double>(p * (p - 1));
  return self.copy_(args.lgamma_().sum(-1).add_(p2_sub_p * std::log(c10::pi<double>) * QUARTER));
}

Tensor& mvlgamma_out(const Tensor& self, int64_t p, Tensor& result) {
  auto out = self.mvlgamma(p);
  TORCH_CHECK(
      at::can_cast(out.scalar_type(), result.scalar_type()),
      "mvlgamma: result type ",
      self.scalar_type(),
      " can't be cast to the desired output type ",
      out.scalar_type());
  at::native::resize_output(result, out.sizes());
  return result.copy_(out);
}

Tensor special_multigammaln(const Tensor& self, int64_t p) {
  return self.mvlgamma(p);
};

Tensor& special_multigammaln_out(const Tensor& self, int64_t p, Tensor& result) {
  return at::mvlgamma_out(result, self, p);
};

std::tuple<Tensor, Tensor> frexp(const Tensor& self) {
  Tensor mantissa = at::empty_like(self);
  Tensor exponent = at::empty_like(self, self.options().dtype(at::kInt));

  at::frexp_out(mantissa, exponent, self);
  return std::tuple<Tensor, Tensor>(mantissa, exponent);
}

std::tuple<Tensor&, Tensor&> frexp_out(const Tensor& self,
                                       Tensor& mantissa, Tensor& exponent) {
  // torch.frexp is implemented for floating-point dtypes for now,
  // should add support for integral dtypes in the future.
  TORCH_CHECK(at::isFloatingType(self.scalar_type()),
              "torch.frexp() only supports floating-point dtypes");

  TORCH_CHECK(mantissa.dtype() == self.dtype(),
              "torch.frexp() expects mantissa to have dtype ", self.dtype(),
              " but got ", mantissa.dtype());
  TORCH_CHECK(exponent.dtype() == at::kInt,
              "torch.frexp() expects exponent to have int dtype "
              "but got ", exponent.dtype());

  auto iter = TensorIteratorConfig()
    .add_output(mantissa)
    .add_output(exponent)
    .add_const_input(self)
    .check_all_same_dtype(false)
    .set_check_mem_overlap(true)
    .build();
  frexp_stub(iter.device_type(), iter);

  return std::tuple<Tensor&, Tensor&>(mantissa, exponent);
}

// alias for lgamma, implements special.gammaln equivalent to
// scipy.special.gammaln
Tensor special_gammaln(const Tensor& self) { return self.lgamma(); }
Tensor& special_gammaln_out(const Tensor& self, Tensor& result) { return at::lgamma_out(result, self); }

DEFINE_DISPATCH(abs_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(angle_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(conj_physical_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(acos_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(acosh_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(asinh_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(atanh_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(asin_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(atan_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(bitwise_not_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(ceil_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(cos_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(cosh_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(digamma_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_entr_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_erfcx_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(erf_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(erfc_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(erfinv_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(exp_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(exp2_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(expm1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(floor_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(frac_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(frexp_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(i0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_i0e_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_i1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_i1e_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(log_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(log10_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(log1p_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(log2_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(logical_not_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_ndtri_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_log_ndtr_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(neg_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(nan_to_num_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(polygamma_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(reciprocal_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(round_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(round_decimals_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(rsqrt_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sigmoid_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(logit_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sign_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(signbit_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sgn_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sin_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sinc_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sinh_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(sqrt_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(tan_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(tanh_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(trigamma_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(trunc_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(lgamma_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_airy_ai_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_bessel_j0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_bessel_j1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_bessel_y0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_bessel_y1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_modified_bessel_i0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_modified_bessel_i1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_modified_bessel_k0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_modified_bessel_k1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_scaled_modified_bessel_k0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_scaled_modified_bessel_k1_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(special_spherical_bessel_j0_stub); // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

} // namespace at::native
