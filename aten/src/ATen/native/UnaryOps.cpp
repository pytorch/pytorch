#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/LegacyTHFunctionsCPU.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>

#include <ATen/CPUApplyUtils.h>
#include <ATen/Parallel.h>
#include <ATen/native/Math.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/ComplexHelper.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include <map>

namespace at {

namespace meta {

TORCH_META_FUNC(sin) (
  const Tensor& self
) {
  build_unary_float_op(maybe_get_output(), self);
}

} // namespace meta

namespace native {
// NOTE: These are helper functions that reduce redundant code in implementing the most typical kind of unary operators.
// YOU ARE NOT OBLIGED TO USE THESE HELPERS---if you're writing something more specialized, please don't try to make
// them work for your case, but just write something new instead. Here we use helper functions instead of a flat fat
// macro that implements everything, because the former allows some simple preprocessing that are unique to some
// operators (more is foreseeable) and is more flexible and elegant than the latter.
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
  iter.cast_outputs();
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
      const auto float_type = c10::toValueType(self.scalar_type());
      TORCH_CHECK(canCast(float_type, result.scalar_type()),
            "result type ", float_type, " can't be cast to the desired output type ",
            result.scalar_type());

      // Runs the function complex->complex, as TensorIterator expects
      Tensor complex_result = at::empty({0}, self.options());
      auto iter = TensorIterator::unary_op(complex_result, self);
      stub(iter.device_type(), iter);

      // Copies the complex result to the actual result and returns it
      result.resize_(complex_result.sizes());
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
    const auto float_type = c10::toValueType(self.scalar_type());
    Tensor result = at::empty({0}, self.options().dtype(float_type));
    return out_impl(result, self);
  }

  Tensor result = at::empty({0}, self.options());
  return out_impl(result, self);
}

template <typename OutImpl>
static inline Tensor& unary_op_impl_(Tensor& self, OutImpl& out_impl) {
  return out_impl(self, self);
}

Tensor& acos_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, acos_stub); }
Tensor acos(const Tensor& self) { return unary_op_impl_float(self, acos_stub); }
Tensor& acos_(Tensor& self) { return unary_op_impl_(self, at::acos_out); }

// arccos, alias for acos
Tensor& arccos_out(const Tensor& self, Tensor& result) { return at::acos_out(result, self); }
Tensor arccos(const Tensor& self) { return self.acos(); }
Tensor& arccos_(Tensor& self) { return self.acos_(); }

static Tensor wrapped_scalar_tensor(const Scalar& scalar) {
  auto tensor = scalar_to_tensor(scalar);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  return tensor;
}

Tensor& rad2deg_out(Tensor& result, const Tensor& self) {
  TORCH_CHECK(!self.is_complex(), "rad2deg is not supported for complex tensors.");
  constexpr double M_180_PI = 57.295779513082320876798154814105170332405472466564;
  return at::mul_out(result, self, wrapped_scalar_tensor(Scalar(M_180_PI)));
}
Tensor rad2deg(const Tensor& self) {
  // Note: int-> float promotion handled differently from other Unary ops,
  // as it does not use the usual TensorIterator + Kernel Dispatch pattern.
  auto options = self.options();
  if (c10::isIntegralType(self.scalar_type(), /*include_bool=*/true)) {
    options = options.dtype(c10::get_default_dtype());
  }
  auto result = at::empty_like(self, options);
  at::rad2deg_out(result, self);
  return result;
}
Tensor& rad2deg_(Tensor& self) { return unary_op_impl_(self, at::rad2deg_out); }

Tensor& deg2rad_out(Tensor& result, const Tensor& self) {
  TORCH_CHECK(!self.is_complex(), "deg2rad is not supported for complex tensors.");
  constexpr double M_PI_180 = 0.017453292519943295769236907684886127134428718885417;
  return at::mul_out(result, self, wrapped_scalar_tensor(Scalar(M_PI_180)));
}
Tensor deg2rad(const Tensor& self) {
  // Note: int-> float promotion handled differently from other Unary ops,
  // as it does not use the usual TensorIterator + Kernel Dispatch pattern.
  auto options = self.options();
  if (c10::isIntegralType(self.scalar_type(), /*include_bool=*/true)) {
    options = options.dtype(c10::get_default_dtype());
  }
  auto result = at::empty_like(self, options);
  at::deg2rad_out(result, self);
  return result;
}
Tensor& deg2rad_(Tensor& self) { return unary_op_impl_(self, at::deg2rad_out); }

Tensor& asin_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, asin_stub); }
Tensor asin(const Tensor& self) { return unary_op_impl_float(self, asin_stub); }
Tensor& asin_(Tensor& self) { return unary_op_impl_(self, at::asin_out); }

// arcsin, alias of asin
Tensor& arcsin_out(const Tensor& self, Tensor& result) { return at::asin_out(result, self); }
Tensor arcsin(const Tensor& self) { return self.asin(); }
Tensor& arcsin_(Tensor& self) { return self.asin_(); }

Tensor& atan_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, atan_stub); }
Tensor atan(const Tensor& self) { return unary_op_impl_float(self, atan_stub); }
Tensor& atan_(Tensor& self) { return unary_op_impl_(self, at::atan_out); }

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
    const auto float_type = c10::toValueType(self.scalar_type());
    Tensor result = at::empty({0}, self.options().dtype(float_type));
    return at::angle_out(result, self);
  }

  return unary_op_impl_float(self, angle_stub);
}

Tensor real(const Tensor& self) {
  if (self.is_complex()) {
    auto real_tensor = at::view_as_real(self);
    return at::select(real_tensor, real_tensor.dim() - 1, 0);
  } else {
    TORCH_CHECK(false, "real is not implemented for tensors with non-complex dtypes.");
  }
}

Tensor imag(const Tensor& self) {
  if (self.is_complex()) {
    auto real_tensor = at::view_as_real(self);
    return at::select(real_tensor, real_tensor.dim() - 1, 1);
  } else {
    TORCH_CHECK(false, "imag is not implemented for tensors with non-complex dtypes.");
  }
}

Tensor& conj_out(const Tensor& self, Tensor& result) {
  return unary_op_impl_out(result, self, conj_stub);
}

Tensor _conj(const Tensor& self) { return unary_op_impl(self, at::conj_out); }

Tensor conj(const Tensor& self) {
  if (!self.is_complex()) {
    return self;
  }
  return at::_conj(self);
}

Tensor& bitwise_not_out(const Tensor& self, Tensor& result) { return unary_op_impl_out(result, self, bitwise_not_stub); }
Tensor bitwise_not(const Tensor& self) { return unary_op_impl(self, at::bitwise_not_out); }
Tensor& bitwise_not_(Tensor& self) { return unary_op_impl_(self, at::bitwise_not_out); }

Tensor& ceil_out(const Tensor& self, Tensor& result) {
  // Note: this is consistent with NumPy
  TORCH_CHECK(!self.is_complex(),
    "ceil is not supported for complex inputs");

  return unary_op_impl_out(result, self, ceil_stub);
}
Tensor ceil(const Tensor& self) { return unary_op_impl(self, at::ceil_out); }
Tensor& ceil_(Tensor& self) { return unary_op_impl_(self, at::ceil_out); }

Tensor& exp_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, exp_stub); }
Tensor exp(const Tensor& self) { return unary_op_impl_float(self, exp_stub); }
Tensor& exp_(Tensor& self) { return unary_op_impl_(self, at::exp_out); }

Tensor& exp2_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, exp2_stub); }
Tensor exp2(const Tensor& self) { return unary_op_impl_float(self, exp2_stub); }
Tensor& exp2_(Tensor& self) { return unary_op_impl_(self, at::exp2_out); }

Tensor& expm1_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, expm1_stub); }
Tensor expm1(const Tensor& self) { return unary_op_impl_float(self, expm1_stub); }
Tensor& expm1_(Tensor& self) { return unary_op_impl_(self, at::expm1_out); }

Tensor& erf_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, erf_stub); }
Tensor erf(const Tensor& self) { return unary_op_impl_float(self, erf_stub); }
Tensor& erf_(Tensor& self) { return unary_op_impl_(self, at::erf_out); }

Tensor& erfc_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, erfc_stub); }
Tensor erfc(const Tensor& self) { return unary_op_impl_float(self, erfc_stub); }
Tensor& erfc_(Tensor& self) { return unary_op_impl_(self, at::erfc_out); }

Tensor& erfinv_out(Tensor& result, const Tensor& self) { return unary_op_impl_float_out(result, self, erfinv_stub); }
Tensor erfinv(const Tensor& self) { return unary_op_impl_float(self, erfinv_stub); }
Tensor& erfinv_(Tensor& self) { return unary_op_impl_(self, at::erfinv_out); }

Tensor& special_erf_out(const Tensor& self, Tensor& result) { return at::erf_out(result, self); }
Tensor special_erf(const Tensor& self) { return self.erf(); }

Tensor& special_erfc_out(const Tensor& self, Tensor& result) { return at::erfc_out(result, self); }
Tensor special_erfc(const Tensor& self) { return self.erfc(); }

Tensor& special_erfinv_out(const Tensor& self, Tensor& result) { return at::erfinv_out(result, self); }
Tensor special_erfinv(const Tensor& self) { return self.erfinv(); }

Tensor& frac_out(const Tensor& self, Tensor& result) { return unary_op_impl_out(result, self, frac_stub); }
Tensor frac(const Tensor& self) { return unary_op_impl(self, at::frac_out); }
Tensor& frac_(Tensor& self) { return unary_op_impl_(self, at::frac_out); }

Tensor& floor_out(const Tensor& self, Tensor& result) {
  // Note: this is consistent with NumPy
  TORCH_CHECK(!self.is_complex(),
    "floor is not supported for complex inputs");

  return unary_op_impl_out(result, self, floor_stub);
}
Tensor floor(const Tensor& self) { return unary_op_impl(self, at::floor_out); }
Tensor& floor_(Tensor& self) { return unary_op_impl_(self, at::floor_out); }

Tensor& i0_out(Tensor& result, const Tensor& self) { return unary_op_impl_out(result, self, i0_stub); }
Tensor i0(const Tensor& self) { return unary_op_impl(self, at::i0_out); }
Tensor& i0_(Tensor& self) { return unary_op_impl_(self, at::i0_out); }

Tensor& log_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, log_stub); }
Tensor log(const Tensor& self) { return unary_op_impl_float(self, log_stub); }
Tensor& log_(Tensor& self) { return unary_op_impl_(self, at::log_out); }

Tensor& log10_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, log10_stub); }
Tensor log10(const Tensor& self) { return unary_op_impl_float(self, log10_stub); }
Tensor& log10_(Tensor& self) { return unary_op_impl_(self, at::log10_out); }

Tensor& log1p_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, log1p_stub); }
Tensor log1p(const Tensor& self) { return unary_op_impl_float(self, log1p_stub); }
Tensor& log1p_(Tensor& self) { return unary_op_impl_(self, at::log1p_out); }

Tensor& log2_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, log2_stub); }
Tensor log2(const Tensor& self) { return unary_op_impl_float(self, log2_stub); }
Tensor& log2_(Tensor& self) { return unary_op_impl_(self, at::log2_out); }

Tensor& round_out(Tensor& result, const Tensor& self) { return unary_op_impl_out(result, self, round_stub); }
Tensor round(const Tensor& self) { return unary_op_impl(self, at::round_out); }
Tensor& round_(Tensor& self) { return unary_op_impl_(self, at::round_out); }

Tensor& digamma_out(Tensor& result, const Tensor& self) { return unary_op_impl_float_out(result, self, digamma_stub); }
Tensor digamma(const Tensor& self) { return unary_op_impl_float(self, digamma_stub); }
Tensor& digamma_(Tensor& self) { return unary_op_impl_(self, digamma_out); }

Tensor& reciprocal_out(Tensor& result, const Tensor& self) { return unary_op_impl_float_out(result, self, reciprocal_stub); }
Tensor reciprocal(const Tensor& self) { return unary_op_impl_float(self, reciprocal_stub); }
Tensor& reciprocal_(Tensor& self) { return unary_op_impl_(self, at::reciprocal_out); }

Tensor& rsqrt_out(Tensor& result, const Tensor& self) {
  return unary_op_impl_float_out(result, self, rsqrt_stub);
}
Tensor rsqrt(const Tensor& self) {
  return unary_op_impl_float(self, rsqrt_stub);
}
Tensor& rsqrt_(Tensor& self) { return unary_op_impl_(self, at::rsqrt_out); }

Tensor& sign_out(const Tensor& self, Tensor& result) {
  TORCH_CHECK(!self.is_complex(),
              "Unlike NumPy, torch.sign is not intended to support complex numbers. Please use torch.sgn instead.");
  return unary_op_impl_out(result, self, sign_stub);
}
Tensor sign(const Tensor& self) { return unary_op_impl(self, at::sign_out); }
Tensor& sign_(Tensor& self) { return unary_op_impl_(self, at::sign_out); }

Tensor& sgn_out(const Tensor& self, Tensor& result) {
  if (self.is_complex()) {
    return unary_op_impl_out(result, self, sgn_stub);
  } else {
    return unary_op_impl_out(result, self, sign_stub);
  }
}

Tensor sgn(const Tensor& self) { return unary_op_impl(self, at::sgn_out); }
Tensor& sgn_(Tensor& self) { return unary_op_impl_(self, at::sgn_out); }

TORCH_IMPL_FUNC(sin_out) (const Tensor& self, const Tensor& result) {
  sin_stub(device_type(), *this);
}

Tensor& cos_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, cos_stub); }
Tensor cos(const Tensor& self) { return unary_op_impl_float(self, cos_stub); }
Tensor& cos_(Tensor& self) { return unary_op_impl_(self, at::cos_out); }

Tensor& sinc_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, sinc_stub); }
Tensor sinc(const Tensor& self) { return unary_op_impl_float(self, sinc_stub); }
Tensor& sinc_(Tensor& self) { return unary_op_impl_(self, at::sinc_out); }

Tensor& sinh_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, sinh_stub); }
Tensor sinh(const Tensor& self) { return unary_op_impl_float(self, sinh_stub); }
Tensor& sinh_(Tensor& self) { return unary_op_impl_(self, at::sinh_out); }

Tensor& cosh_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, cosh_stub); }
Tensor cosh(const Tensor& self) { return unary_op_impl_float(self, cosh_stub); }
Tensor& cosh_(Tensor& self) { return unary_op_impl_(self, at::cosh_out); }

Tensor& acosh_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, acosh_stub); }
Tensor acosh(const Tensor& self) { return unary_op_impl_float(self, acosh_stub); }
Tensor& acosh_(Tensor& self) { return unary_op_impl_(self, at::acosh_out); }

// arccosh, alias for acosh
Tensor& arccosh_out(const Tensor& self, Tensor& result) { return at::acosh_out(result, self); }
Tensor arccosh(const Tensor& self) { return at::acosh(self); }
Tensor& arccosh_(Tensor& self) { return at::acosh_(self); }

Tensor& asinh_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, asinh_stub); }
Tensor asinh(const Tensor& self) { return unary_op_impl_float(self, asinh_stub); }
Tensor& asinh_(Tensor& self) { return unary_op_impl_(self, at::asinh_out); }

// arcsinh, alias for asinh
Tensor& arcsinh_out(const Tensor& self, Tensor& result) { return at::asinh_out(result, self); }
Tensor arcsinh(const Tensor& self) { return self.asinh(); }
Tensor& arcsinh_(Tensor& self) { return self.asinh_(); }

Tensor& atanh_out(const Tensor& self, Tensor& result) { return unary_op_impl_float_out(result, self, atanh_stub); }
Tensor atanh(const Tensor& self) { return unary_op_impl_float(self, atanh_stub); }
Tensor& atanh_(Tensor& self) { return unary_op_impl_(self, at::atanh_out); }

// arctanh, alias for atanh
Tensor& arctanh_out(const Tensor& self, Tensor& result) { return at::atanh_out(result, self); }
Tensor arctanh(const Tensor& self) { return self.atanh(); }
Tensor& arctanh_(Tensor& self) { return self.atanh_(); }

Tensor& sqrt_out(Tensor& result, const Tensor& self) { return unary_op_impl_float_out(result, self, sqrt_stub); }
Tensor sqrt(const Tensor& self) { return unary_op_impl_float(self, sqrt_stub); }
Tensor& sqrt_(Tensor& self) { return unary_op_impl_(self, at::sqrt_out); }

Tensor square(const Tensor& self) { return at::pow(self, 2); }
Tensor& square_(Tensor& self) { return at::pow_out(self, self, 2); }

Tensor& sigmoid_out(Tensor& result, const Tensor& self) { return unary_op_impl_float_out(result, self, sigmoid_stub);  }
Tensor sigmoid(const Tensor& self) { return unary_op_impl_float(self, sigmoid_stub);  }
Tensor& sigmoid_(Tensor& self) { return unary_op_impl_(self, at::sigmoid_out);  }

Tensor& logit_out(
    Tensor& result,
    const Tensor& self,
    c10::optional<double> eps) {
  return unary_op_impl_float_out(
      result, self, logit_stub, Scalar(eps ? eps.value() : -1.0));
}
Tensor logit(const Tensor& self, c10::optional<double> eps) {
  return unary_op_impl_float(
      self, logit_stub, Scalar(eps ? eps.value() : -1.0));
}
Tensor& logit_(Tensor& self, c10::optional<double> eps) {
  return at::logit_out(self, self, eps);
}

Tensor& nan_to_num_out(const Tensor& self,
    c10::optional<double> nan,
    c10::optional<double> pos_inf,
    c10::optional<double> neg_inf,
    Tensor& result) {
  TORCH_CHECK(
      self.scalar_type() == result.scalar_type(),
      "nan_to_num: dtype of out: ",
      result.scalar_type(),
      " should be same as input: ",
      self.scalar_type());

  if (c10::isIntegralType(self.scalar_type(), /*include_bool=*/true)) {
    result.resize_as_(self);
    result.copy_(self);
    return result;
  }

  auto iter = TensorIterator::unary_op(result, self);
  nan_to_num_stub(iter.device_type(), iter, nan, pos_inf, neg_inf);
  return result;
}

Tensor nan_to_num(
    const Tensor& self,
    c10::optional<double> nan,
    c10::optional<double> pos_inf,
    c10::optional<double> neg_inf) {
  auto result = at::empty_like(self);
  return at::nan_to_num_out(result, self, nan, pos_inf, neg_inf);
}

Tensor& nan_to_num_(
    Tensor& self,
    c10::optional<double> nan,
    c10::optional<double> pos_inf,
    c10::optional<double> neg_inf) {
  return at::nan_to_num_out(self, self, nan, pos_inf, neg_inf);
}

Tensor& tanh_out(Tensor& result, const Tensor& self) { return unary_op_impl_float_out(result, self, tanh_stub); }
Tensor tanh(const Tensor& self) { return unary_op_impl_float(self, tanh_stub); }
Tensor& tanh_(Tensor& self) { return unary_op_impl_(self, at::tanh_out); }

Tensor& tan_out(Tensor& result, const Tensor& self) { return unary_op_impl_float_out(result, self, tan_stub);  }
Tensor tan(const Tensor& self) { return unary_op_impl_float(self, tan_stub);  }
Tensor& tan_(Tensor& self) { return unary_op_impl_(self, at::tan_out);  }

Tensor& trunc_out(Tensor& result, const Tensor& self) {
  // Note: this is consistent with NumPy
  TORCH_CHECK(!self.is_complex(),
    "trunc is not supported for complex inputs");

  return unary_op_impl_out(result, self, trunc_stub);
}
Tensor trunc(const Tensor& self) { return unary_op_impl(self, at::trunc_out); }
Tensor& trunc_(Tensor& self) { return unary_op_impl_(self, at::trunc_out); }

// Alias for trunc
Tensor& fix_out(Tensor& result, const Tensor& self) { return at::trunc_out(result, self); }
Tensor fix(const Tensor& self) { return self.trunc(); }
Tensor& fix_(Tensor& self) { return self.trunc_(); }

Tensor& neg_out(Tensor& result, const Tensor& self) {
  TORCH_CHECK(self.scalar_type() != kBool,
              "Negation, the `-` operator, on a bool tensor is not supported. "
              "If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");
  return unary_op_impl_out(result, self, neg_stub);
}
Tensor neg(const Tensor& self) { return unary_op_impl(self, at::neg_out); }
Tensor& neg_(Tensor& self) { return unary_op_impl_(self, at::neg_out); }

Tensor& negative_out(Tensor& result, const Tensor& self) { return at::neg_out(result, self); }
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
    .add_input(self)
    .build();
  logical_not_stub(iter.device_type(), iter);
  return result;
}

Tensor& signbit_out(const Tensor& self, Tensor& result) {
  TORCH_CHECK(!self.is_complex(), "signbit is not implemented for complex tensors.");
  TORCH_CHECK(result.scalar_type() == at::kBool, "signbit does not support non-boolean outputs.");
  result.resize_(self.sizes());

  if (self.dtype() == at::kBool) {
    return result.fill_(false);
  } else {
    TensorIterator iter = TensorIteratorConfig()
      .check_all_same_dtype(false)
      .add_output(result)
      .add_input(self)
      .build();
    signbit_stub(iter.device_type(), iter);
  }
  return result;
}

Tensor signbit(const Tensor& self) {
  Tensor result = at::empty({0}, self.options().dtype(kBool));
  return at::signbit_out(result, self);
}

Tensor& clamp_out(const Tensor& self, const optional<Scalar>& min, const optional<Scalar>& max, Tensor& result) {
  if (min && max) {
    TORCH_CHECK(self.layout() == Layout::Strided,
                "clamp only supports strided layout, got: ", self.layout());
    auto iter = TensorIterator::unary_op(result, self);
    clamp_stub(iter.device_type(), iter, *min, *max);
  } else if (max) {
    at::clamp_max_out(result, self, *max);
  } else if (min) {
    at::clamp_min_out(result, self, *min);
  } else {
    TORCH_CHECK(false, "At least one of 'min' or 'max' must not be None");
  }
  return result;
}

Tensor clamp(const Tensor& self, const optional<Scalar>& min, const optional<Scalar>& max) {
  Tensor result = at::empty({0}, self.options());
  return at::clamp_out(result, self, min, max);
}

Tensor& clamp_(Tensor& self, const optional<Scalar>& min, const optional<Scalar>& max) {
  return at::clamp_out(self, self, min, max);
}

Tensor& clamp_max_out(const Tensor& self, const Scalar& max, Tensor& result) {
  TORCH_CHECK(self.layout() == Layout::Strided,
              "clamp_max only supports strided layout, got: ", self.layout());
  auto iter = TensorIterator::unary_op(result, self);
  clamp_max_stub(iter.device_type(), iter, max);
  return result;
}

Tensor clamp_max(const Tensor& self, const Scalar& max) {
  Tensor result = at::empty({0}, self.options());
  return at::clamp_max_out(result, self, max);
}

Tensor& clamp_max_(Tensor& self, const Scalar& max) {
  return at::clamp_max_out(self, self, max);
}

Tensor& clamp_min_out(const Tensor& self, const Scalar& min, Tensor& result) {
  TORCH_CHECK(self.layout() == Layout::Strided,
              "clamp_min only supports strided layout, got: ", self.layout());
  auto iter = TensorIterator::unary_op(result, self);
  clamp_min_stub(iter.device_type(), iter, min);
  return result;
}

Tensor clamp_min(const Tensor& self, const Scalar& min) {
  Tensor result = at::empty({0}, self.options());
  return at::clamp_min_out(result, self, min);
}

Tensor& clamp_min_(Tensor& self, const Scalar& min) {
  return at::clamp_min_out(self, self, min);
}

// Implements the "clip" alias for clamp
Tensor& clip_out(const Tensor& self, const optional<Scalar>& min, const optional<Scalar>& max, Tensor& result) {
  return at::clamp_out(result, self, min, max);
}

Tensor clip(const Tensor& self, const optional<Scalar>& min, const optional<Scalar>& max) {
  return at::clamp(self, min, max);
}

Tensor& clip_(Tensor& self, const optional<Scalar>& min, const optional<Scalar>& max) {
  return at::clamp_(self, min, max);
}

Tensor polygamma(int64_t n, const Tensor& self) {
  Tensor result = at::empty({0}, self.options());
  at::polygamma_out(result, n, self);
  return result;
}
Tensor& polygamma_(Tensor& self, int64_t n) {
  return at::polygamma_out(self, n, self);
}
Tensor& polygamma_out(Tensor& result, int64_t n, const Tensor& self) {
  TORCH_CHECK(n >= 0, "polygamma(n, x) does not support negative n.");
  auto iter = TensorIterator::unary_op(result, self);
  polygamma_stub(iter.device_type(), iter, n);
  return result;
}

static inline void mvlgamma_check(const Tensor& self, int64_t p) {
  TORCH_CHECK(at::isFloatingType(self.scalar_type()),
              "mvlgamma is not implemented for ", self.scalar_type());
  TORCH_CHECK((self > 0.5f * (p - 1)).all().item<bool>(),
              "All elements must be greater than (p-1)/2");
  TORCH_CHECK(p >= 1, "p has to be greater than or equal to 1");
}

Tensor mvlgamma(const Tensor& self, int64_t p) {
  mvlgamma_check(self, p);
  Tensor args = native::arange(-p / 2. + 0.5, 0.5, 0.5, self.options());
  args = args.add(self.unsqueeze(-1));
  return args.lgamma_().sum(-1).add_(p * (p - 1) * std::log(c10::pi<double>) / 4.);
}

Tensor& mvlgamma_(Tensor& self, int64_t p) {
  mvlgamma_check(self, p);
  Tensor args = native::arange(-p / 2. + 0.5, 0.5, 0.5, self.options());
  args = args.add(self.unsqueeze(-1));
  return self.copy_(args.lgamma_().sum(-1).add_(p * (p - 1) * std::log(c10::pi<double>) / 4.));
}

Tensor& lgamma_out(Tensor& result, const Tensor& self) { return unary_op_impl_float_out(result, self, lgamma_stub); }
Tensor lgamma(const Tensor& self) { return unary_op_impl_float(self, lgamma_stub); }
Tensor& lgamma_(Tensor& self) { return unary_op_impl_(self, at::lgamma_out); }

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
    .add_input(self)
    .check_all_same_dtype(false)
    .set_check_mem_overlap(true)
    .build();
  frexp_stub(iter.device_type(), iter);

  return std::tuple<Tensor&, Tensor&>(mantissa, exponent);
}

// alias for lgamma, implements special.gammanln equivalent to
// scipy.special.gammaln
Tensor special_gammaln(const Tensor& self) { return self.lgamma(); }
Tensor& special_gammaln_out(const Tensor& self, Tensor& result) { return at::lgamma_out(result, self); }

DEFINE_DISPATCH(abs_stub);
DEFINE_DISPATCH(angle_stub);
DEFINE_DISPATCH(real_stub);
DEFINE_DISPATCH(imag_stub);
DEFINE_DISPATCH(conj_stub);
DEFINE_DISPATCH(acos_stub);
DEFINE_DISPATCH(acosh_stub);
DEFINE_DISPATCH(asinh_stub);
DEFINE_DISPATCH(atanh_stub);
DEFINE_DISPATCH(asin_stub);
DEFINE_DISPATCH(atan_stub);
DEFINE_DISPATCH(bitwise_not_stub);
DEFINE_DISPATCH(ceil_stub);
DEFINE_DISPATCH(clamp_stub);
DEFINE_DISPATCH(clamp_max_stub);
DEFINE_DISPATCH(clamp_min_stub);
DEFINE_DISPATCH(cos_stub);
DEFINE_DISPATCH(cosh_stub);
DEFINE_DISPATCH(digamma_stub);
DEFINE_DISPATCH(erf_stub);
DEFINE_DISPATCH(erfc_stub);
DEFINE_DISPATCH(erfinv_stub);
DEFINE_DISPATCH(exp_stub);
DEFINE_DISPATCH(exp2_stub);
DEFINE_DISPATCH(expm1_stub);
DEFINE_DISPATCH(floor_stub);
DEFINE_DISPATCH(frac_stub);
DEFINE_DISPATCH(frexp_stub);
DEFINE_DISPATCH(i0_stub);
DEFINE_DISPATCH(log_stub);
DEFINE_DISPATCH(log10_stub);
DEFINE_DISPATCH(log1p_stub);
DEFINE_DISPATCH(log2_stub);
DEFINE_DISPATCH(logical_not_stub);
DEFINE_DISPATCH(neg_stub);
DEFINE_DISPATCH(nan_to_num_stub);
DEFINE_DISPATCH(polygamma_stub);
DEFINE_DISPATCH(reciprocal_stub);
DEFINE_DISPATCH(round_stub);
DEFINE_DISPATCH(rsqrt_stub);
DEFINE_DISPATCH(sigmoid_stub);
DEFINE_DISPATCH(logit_stub);
DEFINE_DISPATCH(sign_stub);
DEFINE_DISPATCH(signbit_stub);
DEFINE_DISPATCH(sgn_stub);
DEFINE_DISPATCH(sin_stub);
DEFINE_DISPATCH(sinc_stub);
DEFINE_DISPATCH(sinh_stub);
DEFINE_DISPATCH(sqrt_stub);
DEFINE_DISPATCH(tan_stub);
DEFINE_DISPATCH(tanh_stub);
DEFINE_DISPATCH(trigamma_stub);
DEFINE_DISPATCH(trunc_stub);
DEFINE_DISPATCH(lgamma_stub);
} // namespace native
} // namespace at
