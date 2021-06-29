#include <ATen/native/Pow.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/ScalarOps.h>
#include <ATen/native/Resize.h>

namespace at {
namespace meta {

TORCH_META_FUNC2(pow, Tensor_Tensor) (const Tensor& base, const Tensor& exp) {
  build_borrowing_binary_op(maybe_get_output(), base, exp);
}

TORCH_META_FUNC2(pow, Tensor_Scalar) (const Tensor& base, const Scalar& exp) {
  // Numpy compatibility check:
  TORCH_CHECK(!(isIntegralType(base.scalar_type(), true) &&
              exp.isIntegral(true) && exp.toLong() < 0),
              "Integers to negative integer powers are not allowed.");

  auto common_dtype = at::result_type(base, exp);
  build_unary_op(maybe_get_output(), base.to(common_dtype));
}

TORCH_META_FUNC2(pow, Scalar) (const Scalar& base, const Tensor& exp) {
  // This overload doesn't directly use TensorIterator. It attempts to short-circuit,
  // but otherwise redispatches to the Tensor_Tensor overload.
  auto dtype = maybe_get_output().defined() ? maybe_get_output().scalar_type() : at::result_type(base, exp);
  set_output(0, exp.sizes(), {}, exp.options().dtype(dtype), exp.has_names() ? exp.names() : ArrayRef<Dimname>());
}

#define SET_OUTPUT_FOR_FLOAT_POWER(result, t, dtype)                                                                              \
  if (result.defined()) {                                                                                                        \
    TORCH_CHECK(result.scalar_type() == dtype,                                                                            \
                "the output given to float_power has dtype ", result.scalar_type(),                                              \
                " but the operation's result requires dtype ", dtype);                                                           \
    set_output(0, result.sizes(), {}, result.options().dtype(dtype), result.has_names() ? result.names() : ArrayRef<Dimname>()); \
  } else {                                                                                                                       \
    set_output(0, t.sizes(), {}, t.options().dtype(dtype), t.has_names() ? t.names() : ArrayRef<Dimname>());                     \
  }

TORCH_META_FUNC2(float_power, Tensor_Tensor) (const Tensor& base, const Tensor& exp) {
  auto dtype = (at::isComplexType(base.scalar_type()) || at::isComplexType(exp.scalar_type())) ?
                at::kComplexDouble : at::kDouble;
  SET_OUTPUT_FOR_FLOAT_POWER(maybe_get_output(), base, dtype);
}

TORCH_META_FUNC2(float_power, Tensor_Scalar) (const Tensor& base, const Scalar& exp) {
  auto dtype = (at::isComplexType(base.scalar_type()) || exp.isComplex()) ? at::kComplexDouble : at::kDouble;
  SET_OUTPUT_FOR_FLOAT_POWER(maybe_get_output(), base, dtype);
}

TORCH_META_FUNC2(float_power, Scalar) (const Scalar& base, const Tensor& exp) {
  auto dtype = (at::isComplexType(exp.scalar_type()) || base.isComplex()) ? at::kComplexDouble : at::kDouble;
  SET_OUTPUT_FOR_FLOAT_POWER(maybe_get_output(), exp, dtype);
}

} // namespace meta

namespace native {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(pow_tensor_tensor_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(pow_tensor_scalar_stub);

TORCH_IMPL_FUNC(pow_Tensor_Tensor_out) (const Tensor& base, const Tensor& exp, const Tensor& out) {
  if (exp.dim() == 0 && exp.device().is_cpu() && base.is_cuda()) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    at::pow_out(const_cast<Tensor&>(out), base, exp.item()); // redispatch!
  } else {
    pow_tensor_tensor_stub(device_type(), *this);
  }
}

TORCH_IMPL_FUNC(pow_Tensor_Scalar_out) (const Tensor& base, const Scalar& exp, const Tensor& out) {
  if (exp.equal(0.0)) {
    out.fill_(1);
  } else if (exp.equal(1.0)) {
    out.copy_(base);
  } else {
    pow_tensor_scalar_stub(device_type(), *this, exp);
  }
}

TORCH_IMPL_FUNC(pow_Scalar_out) (const Scalar& base, const Tensor& exp, const Tensor& out) {
  // NOLINTNEXTLINE(bugprone-branch-clone)
  if (base.isComplex() && base.toComplexDouble() == 1.0) {
    out.fill_(1);
  } else if (!base.isComplex() && base.toDouble() == 1.0) {
    out.fill_(1);
  } else {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    at::pow_out(const_cast<Tensor&>(out), wrapped_scalar_tensor(base, exp.device()), exp); // redispatch!
  }
}

TORCH_IMPL_FUNC(float_power_Tensor_Tensor_out) (const Tensor& base, const Tensor& exp, const Tensor& out) {
  auto dtype = out.scalar_type();
  at::pow_out(const_cast<Tensor&>(out), base.to(dtype), exp.to(dtype));
}

TORCH_IMPL_FUNC(float_power_Tensor_Scalar_out) (const Tensor& base, const Scalar& exp, const Tensor& out) {
  auto dtype = out.scalar_type();
  // Note: need the casts inside the ternary because conversion functions return e.g. c10::complex,
  // which causes a complex scalar to always be returned.
  auto casted_exp = (dtype == at::kComplexDouble) ? Scalar(exp.toComplexDouble()) : Scalar(exp.toDouble());
  at::pow_out(const_cast<Tensor&>(out), base.to(dtype), casted_exp);
}

TORCH_IMPL_FUNC(float_power_Scalar_out) (const Scalar& base, const Tensor& exp, const Tensor& out) {
  auto dtype = out.scalar_type();
  auto casted_base = (dtype == at::kComplexDouble) ? Scalar(base.toComplexDouble()) : Scalar(base.toDouble());
  at::pow_out(const_cast<Tensor&>(out), casted_base, exp.to(dtype));
}

} // namespace native

} // namespace at
