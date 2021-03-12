#include <ATen/native/Pow.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/ScalarOps.h>
#include <ATen/native/Resize.h>

namespace at {
namespace meta {

TORCH_META_FUNC2(pow, Tensor_Tensor) (const Tensor& base, const Tensor& exp) {
  build_binary_op(maybe_get_output(), base, exp);
}

TORCH_META_FUNC2(pow, Tensor_Scalar) (const Tensor& base, const Scalar exp) {
  // Numpy compatibility check:
  TORCH_CHECK(!(isIntegralType(base.scalar_type(), true) &&
              exp.isIntegral(true) && exp.toLong() < 0),
              "Integers to negative integer powers are not allowed.");

  auto common_dtype = at::result_type(base, exp);
  build_unary_op(maybe_get_output(), base.to(common_dtype));
}

TORCH_META_FUNC2(pow, Scalar) (const Scalar base, const Tensor& exp) {
    // This overload doesn't directly use TensorIterator. It attempts to short-circuit,
    // but otherwise redispatches to the Tensor_Tensor overload.
    auto dtype = at::result_type(base, exp);
    // Need to call the set_output() overload of set_output to forward named tensor information.
    set_output(0, exp.sizes(), {}, exp.options().dtype(dtype), exp.names());
}

} // namespace meta

namespace native {

DEFINE_DISPATCH(pow_tensor_tensor_stub);
DEFINE_DISPATCH(pow_tensor_scalar_stub);

TORCH_IMPL_FUNC(pow_Tensor_Tensor_out) (const Tensor& base, const Tensor& exp, const Tensor& out) {
  if (exp.dim() == 0 && exp.device().is_cpu() && base.is_cuda()) {
    at::pow_out(const_cast<Tensor&>(out), base, exp.item()); // redispatch!
  } else {
    pow_tensor_tensor_stub(device_type(), *this);
  }
}

TORCH_IMPL_FUNC(pow_Tensor_Scalar_out) (const Tensor& base, const Scalar exp, const Tensor& out) {
  auto common_dtype = at::result_type(base, exp);
  if (exp.equal(0.0)) {
    out.fill_(1);
  } else if (exp.equal(1.0)) {
    out.copy_(base);
  } else {
    pow_tensor_scalar_stub(device_type(), *this, exp);
  }
}

TORCH_IMPL_FUNC(pow_Scalar_out) (const Scalar base, const Tensor& exp, const Tensor& out) {
  if (base.isComplex() && base.toComplexDouble() == 1.0) {
    out.fill_(1);
  } else if (!base.isComplex() && base.toDouble() == 1.0) {
    out.fill_(1);
  } else {
    at::pow_out(const_cast<Tensor&>(out), c10::scalar_to_tensor(base, exp.device()), exp); // redispatch!
  }
}

Tensor& float_power_out(Tensor& result, const Tensor& base, const Tensor& exp) {
  auto dtype = (at::isComplexType(base.scalar_type()) || at::isComplexType(exp.scalar_type())) ?
                at::kComplexDouble : at::kDouble;
  TORCH_CHECK(result.scalar_type() == dtype,
              "the output given to float_power has dtype ", result.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  return at::pow_out(result, base.to(dtype), exp.to(dtype));
}

Tensor& float_power_out(Tensor& result, const Tensor& base, Scalar exp) {
  auto dtype = (at::isComplexType(base.scalar_type()) || exp.isComplex()) ? at::kComplexDouble : at::kDouble;
  TORCH_CHECK(result.scalar_type() == dtype,
              "the output given to float_power has dtype ", result.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  // Note: need the casts inside the ternary because conversion functions return e.g. c10::complex,
  // which causes a complex scalar to always be returned.
  exp = (dtype == at::kComplexDouble) ? Scalar(exp.toComplexDouble()) : Scalar(exp.toDouble());
  return at::pow_out(result, base.to(dtype), exp);
}

Tensor& float_power_out(Tensor& result, Scalar base, const Tensor& exp) {
  auto dtype = (at::isComplexType(exp.scalar_type()) || base.isComplex()) ? at::kComplexDouble : at::kDouble;
  TORCH_CHECK(result.scalar_type() == dtype,
              "the output given to float_power has dtype ", result.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  base = (dtype == at::kComplexDouble) ? Scalar(base.toComplexDouble()) : Scalar(base.toDouble());
  return at::pow_out(result, base, exp.to(dtype));
}

Tensor float_power(const Tensor& base, Scalar exp) {
  auto dtype = (at::isComplexType(base.scalar_type()) || exp.isComplex()) ? at::kComplexDouble : at::kDouble;
  exp = (dtype == at::kComplexDouble) ? Scalar(exp.toComplexDouble()) : Scalar(exp.toDouble());
  return at::pow(base.to(dtype), exp);
}

Tensor float_power(Scalar base, const Tensor& exp) {
  auto dtype = (at::isComplexType(exp.scalar_type()) || base.isComplex()) ? at::kComplexDouble : at::kDouble;
  base = (dtype == at::kComplexDouble) ? Scalar(base.toComplexDouble()) : Scalar(base.toDouble());
  return at::pow(base, exp.to(dtype));
}

Tensor float_power(const Tensor& base, const Tensor& exp) {
  auto dtype = (at::isComplexType(base.scalar_type()) || at::isComplexType(exp.scalar_type())) ? at::kComplexDouble : at::kDouble;
  return at::pow(base.to(dtype), exp.to(dtype));
}

Tensor& float_power_(Tensor& base, const Tensor& exp) {
  auto dtype = (at::isComplexType(base.scalar_type()) || at::isComplexType(exp.scalar_type())) ? at::kComplexDouble : at::kDouble;
  TORCH_CHECK(base.scalar_type() == dtype,
              "the base given to float_power_ has dtype ", base.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  return base.pow_(exp.to(dtype));
}

Tensor& float_power_(Tensor& base, Scalar exp) {
  auto dtype = (at::isComplexType(base.scalar_type()) || exp.isComplex()) ? at::kComplexDouble : at::kDouble;
  TORCH_CHECK(base.scalar_type() == dtype,
              "the base given to float_power_ has dtype ", base.scalar_type(),
              " but the operation's result requires dtype ", dtype);

  exp = (dtype == at::kComplexDouble) ? Scalar(exp.toComplexDouble()) : Scalar(exp.toDouble());
  return base.pow_(exp);
}

} // namespace native

} // namespace at
